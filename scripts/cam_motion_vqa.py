#!/usr/bin/env python
"""Multiple-choice VQA evaluation of camera motion across a video.

For each Scene, we compute the total relative pose from the first frame to
the last frame, decompose it into camera-local motions (forward/back/left/
right/up/down translation, yaw/pitch/roll rotation), and label the video
with whichever motion component is dominant. We then ask each VLM the same
multiple-choice question and score by matching the model's letter answer.

Options (one letter per option):
    A. moves forward
    B. moves backward
    C. turns left    (yaw left, i.e. pans left)
    D. turns right   (yaw right)
    E. tilts up      (pitch up)
    F. tilts down    (pitch down)

We call this the "6-MC cam motion" protocol.

Prompt (no frame-index references):
    <video>
    Looking at the whole video above, which of the following best
    describes the DOMINANT camera motion?
      A. The camera moves forward.
      ...
      F. The camera tilts down.
    Answer with only the letter (A / B / C / D / E / F).

Usage:
    python scripts/cam_motion_vqa.py \
        --scenes data/tier_d_kitti \
        --model-config configs/models/qwen25vl.yaml \
        --out data/vqa/cam_motion/kitti_qwen25vl_7b.json \
        --limit 100
"""
import _bootstrap  # noqa: F401

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from spatial_subspace.models import (
    InternVL3Wrapper,
    LlavaOnevisionWrapper,
    Qwen25VLWrapper,
)
from spatial_subspace.scene import Scene
from spatial_subspace.utils import ensure_dir, load_yaml


_WRAPPERS = {
    "qwen25vl": Qwen25VLWrapper,
    "llava_onevision": LlavaOnevisionWrapper,
    "internvl3": InternVL3Wrapper,
}


OPTIONS = {
    "A": "The camera moves forward (toward the scene).",
    "B": "The camera moves backward (away from the scene).",
    "C": "The camera turns left (pans left).",
    "D": "The camera turns right (pans right).",
    "E": "The camera tilts up.",
    "F": "The camera tilts down.",
}

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


def ground_truth_label(extrs: list[np.ndarray]) -> tuple[str, dict]:
    """Return the correct letter plus the decomposed motion metrics.

    Uses the OpenCV camera convention (+x right, +y down, +z forward).
      - yaw around the body's y-axis: positive yaw = turn LEFT (via scipy).
      - pitch around the body's x-axis: positive pitch = tilt DOWN.
      - roll around the body's z-axis: positive roll = bank right.
    """
    E_prev = np.asarray(extrs[0], dtype=np.float64)
    E_curr = np.asarray(extrs[-1], dtype=np.float64)
    # Camera positions in world.
    cam_prev = -E_prev[:3, :3].T @ E_prev[:3, 3]
    cam_curr = -E_curr[:3, :3].T @ E_curr[:3, 3]
    disp_world = cam_curr - cam_prev
    # Displacement in the *first* camera's local frame.
    R_wc_prev = E_prev[:3, :3]
    disp_local = R_wc_prev @ disp_world
    right_m, down_m, forward_m = disp_local

    # Relative rotation first→last in first-camera frame.
    R_rel = E_curr[:3, :3] @ R_wc_prev.T
    yaw, pitch, _roll = Rotation.from_matrix(R_rel).as_euler("YXZ", degrees=True)
    # Convention (verified elsewhere): +yaw = left, +pitch = down.
    yaw_deg = -yaw       # we want +yaw_deg = right turn (natural-language right)
    pitch_up_deg = -pitch  # +pitch_up_deg = look up

    # Score each option by its "signed magnitude" along its axis.
    scores = {
        "A": forward_m,       # forward
        "B": -forward_m,      # backward
        "C": -yaw_deg,        # left turn = negative of right-turn axis
        "D": yaw_deg,         # right turn
        "E": pitch_up_deg,    # tilt up
        "F": -pitch_up_deg,   # tilt down
    }
    # Normalise translation (m) and rotation (deg) to comparable scales:
    # 1 m of translation ↔ 30° of rotation ↔ comparable importance.
    rot_scale = 30.0  # 30° of rotation equivalent to 1 m translation
    normalised = {
        "A": scores["A"],
        "B": scores["B"],
        "C": scores["C"] / rot_scale,
        "D": scores["D"] / rot_scale,
        "E": scores["E"] / rot_scale,
        "F": scores["F"] / rot_scale,
    }
    best = max(normalised.items(), key=lambda kv: kv[1])
    return best[0], {
        "forward_m": float(forward_m),
        "right_m": float(right_m),
        "down_m": float(down_m),
        "yaw_right_deg": float(yaw_deg),
        "pitch_up_deg": float(pitch_up_deg),
        "best_letter": best[0],
        "best_score_norm": float(best[1]),
    }


def parse_answer(text: str) -> str | None:
    """Extract a single letter A–F from the model's free-form answer."""
    # Try the first isolated A–F character.
    m = re.search(r"\b([A-Fa-f])\b", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\(?([A-Fa-f])[\).]", text)
    if m:
        return m.group(1).upper()
    # Last resort: any A-F letter.
    m = re.search(r"[A-Fa-f]", text)
    if m:
        return m.group(0).upper()
    return None


def run_vlm_on_video(wrapper, frame_paths: list[str], prompt: str, max_new_tokens: int = 10) -> str:
    """Call the model's `.generate` with a video + prompt and return decoded text.

    We go through the same processor path used in `forward`, then call
    `model.generate` directly for a short completion. All three wrappers
    share their processor + chat_template patterns, so this works
    uniformly.
    """
    import torch
    is_qwen = isinstance(wrapper, Qwen25VLWrapper)
    is_llava = isinstance(wrapper, LlavaOnevisionWrapper)
    is_internvl = isinstance(wrapper, InternVL3Wrapper)
    assert is_qwen or is_llava or is_internvl

    # Build chat messages in each model's expected format.
    if is_qwen:
        from qwen_vl_utils import process_vision_info
        content = [
            {"type": "video", "video": list(frame_paths), "fps": 1.0},
            {"type": "text", "text": prompt},
        ]
        messages = [{"role": "user", "content": content}]
        text = wrapper.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        proc_inputs = wrapper.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(wrapper.device)
    else:
        # LLaVA-OV + InternVL3: their processors accept `videos=[frames]`.
        from PIL import Image
        def _load(x):
            if isinstance(x, str) and x.startswith("file://"):
                return Image.open(x[len("file://"):]).convert("RGB")
            if isinstance(x, str):
                return Image.open(x).convert("RGB")
            return x
        frames = [_load(f) for f in frame_paths]
        content = [{"type": "video"}, {"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        text = wrapper.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        proc_kwargs: dict = {"text": [text], "videos": [frames], "return_tensors": "pt"}
        proc_inputs = wrapper.processor(**proc_kwargs).to(wrapper.device)

    with torch.no_grad():
        out = wrapper.model.generate(
            **proc_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_len = proc_inputs["input_ids"].shape[1]
    gen_tokens = out[0, input_len:]
    decoded = wrapper.processor.tokenizer.decode(
        gen_tokens, skip_special_tokens=True
    ).strip()
    return decoded


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", required=True,
                   help="Directory of Scene-schema dirs (e.g. data/tier_d_kitti)")
    p.add_argument("--model-config", required=True)
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    mcfg = load_yaml(args.model_config)
    family = mcfg.get("family", "qwen25vl")
    wrapper_cls = _WRAPPERS[family]
    print(f"[setup] loading {mcfg['hf_id']} ({family})")
    wrapper = wrapper_cls(
        hf_id=mcfg["hf_id"],
        torch_dtype=mcfg.get("torch_dtype", "bfloat16"),
        device=mcfg.get("device", "cuda"),
        device_map=mcfg.get("device_map"),
    )

    scene_dirs = sorted(Path(args.scenes).iterdir())
    scene_dirs = [d for d in scene_dirs if d.is_dir() and (d / "scene.json").exists()]
    if args.limit is not None:
        scene_dirs = scene_dirs[: args.limit]
    print(f"[setup] {len(scene_dirs)} scenes in {args.scenes}")

    out_records: list[dict] = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm
        it = tqdm(scene_dirs, desc=f"VQA {mcfg.get('name', family)}")
    except ImportError:
        it = scene_dirs

    import torch
    for sd in it:
        try:
            s = Scene.load(sd)
            extrs = [np.asarray(f.camera.extrinsics, dtype=np.float64) for f in s.frames]
            gt_letter, gt_detail = ground_truth_label(extrs)
            frame_paths = [
                f"file://{(sd / f.image_path).resolve()}" for f in s.frames
            ]
            t0 = time.time()
            answer = run_vlm_on_video(wrapper, frame_paths, PROMPT_TEXT)
            elapsed = time.time() - t0
            pred_letter = parse_answer(answer)
            correct = (pred_letter == gt_letter)
            out_records.append({
                "scene_id": s.scene_id,
                "gt_letter": gt_letter,
                "pred_letter": pred_letter,
                "answer_raw": answer,
                "correct": bool(correct),
                "elapsed_sec": float(elapsed),
                **gt_detail,
            })
        except Exception as e:
            out_records.append({
                "scene_id": sd.name,
                "error": f"{type(e).__name__}: {e}",
            })
        # Clear KV cache between samples to avoid fragmentation OOM.
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Save + summary
    out_path.write_text(json.dumps(out_records, indent=2))
    n_valid = sum(1 for r in out_records if "correct" in r)
    n_correct = sum(1 for r in out_records if r.get("correct"))
    acc = n_correct / max(n_valid, 1)
    # Per-letter distribution
    from collections import Counter
    gt_dist = Counter(r["gt_letter"] for r in out_records if "gt_letter" in r)
    pred_dist = Counter(r.get("pred_letter") for r in out_records if "pred_letter" in r)
    print(f"\n[result] saved {out_path}")
    print(f"[result] accuracy: {n_correct}/{n_valid} = {acc:.1%}")
    print(f"[result] GT distribution: {dict(gt_dist)}")
    print(f"[result] pred distribution: {dict(pred_dist)}")
    wrapper.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
