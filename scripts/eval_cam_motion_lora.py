#!/usr/bin/env python
"""Evaluate Dirichlet-trained LoRA checkpoint on real-world camera-motion VQA.

This is a video-input spatial-reasoning task. For each scene, the model
sees the multi-frame video and must answer a 6-MC question about the
dominant camera motion (forward / back / turn-left / turn-right /
tilt-up / tilt-down). Ground-truth labels are computed from extrinsic
camera poses.

This is a different real-world video dataset than VSI-Bench — we use
Microsoft 7-Scenes (indoor RGB-D) and KITTI (outdoor driving) videos
from data/tier_d_7scenes/ and data/tier_d_kitti/.

Scoring: log-prob of letter A-F as the assistant's response, pick
highest-scoring letter.

Usage
-----
    CUDA_VISIBLE_DEVICES=4 python scripts/eval_cam_motion_lora.py \\
        --model-id Qwen/Qwen2.5-VL-7B-Instruct \\
        --checkpoint checkpoints/qwen_lam1_seed0/lora \\
        --scenes-root data/tier_d_7scenes \\
        --out reports/cam_motion_eval/qwen_lam1_seed0_7scenes.json \\
        --limit 40
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from train_qwen_dirichlet import make_labels  # noqa: E402

logger = logging.getLogger("cam_motion")


OPTIONS = ["A", "B", "C", "D", "E", "F"]
OPTION_LABELS = {
    "A": "moves forward (toward the scene)",
    "B": "moves backward (away from the scene)",
    "C": "turns left (pans left)",
    "D": "turns right (pans right)",
    "E": "tilts up",
    "F": "tilts down",
}


def ground_truth_label(extrs: list[np.ndarray]) -> str:
    """Return the correct letter from the camera extrinsics list.

    Uses the OpenCV camera convention.
    """
    E_prev = np.asarray(extrs[0], dtype=np.float64)
    E_curr = np.asarray(extrs[-1], dtype=np.float64)
    # Camera positions in world.
    cam_prev = -E_prev[:3, :3].T @ E_prev[:3, 3]
    cam_curr = -E_curr[:3, :3].T @ E_curr[:3, 3]
    disp_world = cam_curr - cam_prev
    R_wc_prev = E_prev[:3, :3]
    disp_local = R_wc_prev @ disp_world
    right_m, down_m, forward_m = disp_local

    R_rel = E_curr[:3, :3] @ R_wc_prev.T
    yaw, pitch, _roll = Rotation.from_matrix(R_rel).as_euler("YXZ", degrees=True)
    yaw_deg = -yaw            # +yaw_deg = right turn
    pitch_up_deg = -pitch     # +pitch_up_deg = look up

    # Pick dominant component.
    candidates = {
        "A": forward_m,        # forward
        "B": -forward_m,       # backward
        "C": -yaw_deg,         # left turn
        "D": yaw_deg,          # right turn
        "E": pitch_up_deg,     # tilt up
        "F": -pitch_up_deg,    # tilt down
    }
    # Convert rotations to "equivalent meters" for comparison: 1 deg ≈ 0.05 m.
    weighted = {
        "A": candidates["A"],
        "B": candidates["B"],
        "C": 0.05 * candidates["C"],
        "D": 0.05 * candidates["D"],
        "E": 0.05 * candidates["E"],
        "F": 0.05 * candidates["F"],
    }
    return max(weighted, key=weighted.get)


PROMPT_TEXT = (
    "Looking at the whole video above, which of the following best describes "
    "the DOMINANT camera motion during the video?\n"
    "A. The camera moves forward (toward the scene).\n"
    "B. The camera moves backward (away from the scene).\n"
    "C. The camera turns left (pans left).\n"
    "D. The camera turns right (pans right).\n"
    "E. The camera tilts up.\n"
    "F. The camera tilts down.\n\n"
    "Answer with only one letter: A, B, C, D, E, or F."
)


def load_scene(scene_dir: Path, max_frames: int = 16):
    """Load up to max_frames frames + extrinsics for a scene."""
    scene = json.loads((scene_dir / "scene.json").read_text())
    frames = scene.get("frames", [])
    if len(frames) > max_frames:
        # Uniform subsample
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]
    images = []
    extrinsics = []
    for f in frames:
        path = scene_dir / f["image_path"]
        if not path.exists():
            continue
        images.append(Image.open(path).convert("RGB"))
        extrinsics.append(np.asarray(f["camera"]["extrinsics"], dtype=np.float64))
    return images, extrinsics, scene["scene_id"]


@torch.no_grad()
def score_letter(model, processor, images, prompt, candidate, device) -> float:
    """Mean log-prob of `candidate` letter as assistant turn."""
    # Construct chat with image+text per the processor's template
    user_content = []
    for _img in images:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": prompt})
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": candidate}]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = make_labels(inputs["input_ids"][0], candidate, processor).unsqueeze(0)
    n = int((labels != -100).sum())
    if n == 0:
        return float("-inf")
    out = model(**inputs, labels=labels)
    return float(-out.loss)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--scenes-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--max-frames", type=int, default=8,
                   help="Frames per video (subsampled uniformly)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda:0")

    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_id)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16).to(device)
    if args.checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, str(args.checkpoint))
        logger.info("Loaded LoRA: %s", args.checkpoint)
    else:
        model = base
        logger.info("Evaluating un-tuned base model")
    model.eval()

    scene_dirs = sorted([d for d in args.scenes_root.iterdir() if d.is_dir()])
    if args.limit > 0:
        scene_dirs = scene_dirs[: args.limit]
    logger.info("Eval %d scenes from %s", len(scene_dirs), args.scenes_root)

    rows = []
    correct = 0
    by_motion = {}
    t0 = time.time()
    for i, sd in enumerate(scene_dirs):
        try:
            images, extrs, scene_id = load_scene(sd, max_frames=args.max_frames)
            if len(extrs) < 2:
                continue
            gt = ground_truth_label(extrs)
            scores = {}
            for letter in OPTIONS:
                scores[letter] = score_letter(
                    model, processor, images, PROMPT_TEXT, letter, device)
            pred = max(scores, key=scores.get)
            is_correct = pred == gt
        except Exception as e:  # noqa: BLE001
            logger.warning("scene %d failed: %s", i, e)
            scene_id, gt, pred, is_correct = sd.name, None, None, False
        rows.append({"scene_id": scene_id, "gt": gt, "pred": pred,
                     "correct": is_correct})
        if is_correct:
            correct += 1
        by_motion.setdefault(gt, {"n": 0, "c": 0})
        by_motion[gt]["n"] += 1
        by_motion[gt]["c"] += int(is_correct)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            logger.info("  progress %d/%d  acc=%.3f  (%.1fs/it)",
                        i + 1, len(scene_dirs), correct / (i + 1),
                        elapsed / (i + 1))

    summary = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else "base",
        "model_id": args.model_id,
        "scenes_root": str(args.scenes_root),
        "n_total": len(rows),
        "n_correct": correct,
        "accuracy": correct / max(len(rows), 1),
        "by_motion": {k: {"n": v["n"], "acc": v["c"] / v["n"]}
                      for k, v in by_motion.items() if k},
        "wall_time_s": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    logger.info("Done: acc=%.4f (%d/%d) in %.1fs",
                summary["accuracy"], summary["n_correct"],
                summary["n_total"], summary["wall_time_s"])


if __name__ == "__main__":
    main()
