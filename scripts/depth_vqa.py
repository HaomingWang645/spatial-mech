#!/usr/bin/env python
"""Multiple-choice VQA for depth / closeness reasoning.

Question: "Which of the following objects is closest to the camera across
the video?" Options are 4 scene objects (the closest one + 3 distractors
picked from the remaining objects by depth rank so the choices aren't
all nearly-equal).

For each (scene, object) we compute the *average camera-frame depth over
frames where the object is visibly in-frame* (same ≥ 50% silhouette
criterion as the coverage check). An object never in-frame is dropped.

We prefer scenes where at least 4 objects are substantially visible, so the
VQA has 4 legitimate answer choices.

Usage:
    python scripts/depth_vqa.py \
        --scenes data/tier_c_pw_n16 \
        --model-config configs/models/qwen25vl.yaml \
        --out data/vqa/depth/pw_n16_qwen25vl_7b.json --limit 100
"""
import _bootstrap  # noqa: F401

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from spatial_subspace.models import (
    InternVL3Wrapper,
    LlavaOnevisionWrapper,
    Qwen25VLWrapper,
)
from spatial_subspace.scene import Scene
from spatial_subspace.utils import ensure_dir, load_yaml
from spatial_subspace.labels import object_depth_in_camera


_WRAPPERS = {
    "qwen25vl": Qwen25VLWrapper,
    "llava_onevision": LlavaOnevisionWrapper,
    "internvl3": InternVL3Wrapper,
}


def _describe(obj) -> str:
    color = obj.color if obj.color != "unknown" else ""
    shape = obj.shape
    if shape == "obbox":
        shape = str(obj.extras.get("label", "object"))
    return f"the {color} {shape}".strip().replace("  ", " ")


def _avg_depth_per_object(scene: Scene) -> dict[int, tuple[float, int]]:
    """Returns {object_id: (avg_depth_across_all_frames, n_frames)}."""
    out: dict[int, tuple[float, int]] = {}
    extrs = [np.asarray(f.camera.extrinsics, dtype=np.float64) for f in scene.frames]
    for o in scene.objects:
        depths = [object_depth_in_camera(np.asarray(o.centroid), E) for E in extrs]
        # Keep only positive depths (in front of camera).
        pos = [d for d in depths if d > 0.1]
        if not pos:
            continue
        out[o.object_id] = (float(np.mean(pos)), len(pos))
    return out


def build_question(scene: Scene, rng: np.random.Generator) -> tuple[str, list, str] | None:
    """Return (prompt_text, option_object_ids, correct_letter) or None."""
    depths = _avg_depth_per_object(scene)
    if len(depths) < 4:
        return None
    # Rank objects by mean depth (ascending → closest first).
    ranked = sorted(depths.items(), key=lambda kv: kv[1][0])
    closest = ranked[0]
    # Pick 3 distractors from the remaining, spread across the depth range.
    rest = ranked[1:]
    if len(rest) >= 3:
        # Take 3 evenly spaced along depth rank for nice contrast.
        idxs = np.linspace(0, len(rest) - 1, 3).astype(int)
        distractors = [rest[i] for i in idxs]
    else:
        distractors = rest
    options = [closest, *distractors]
    rng.shuffle(options)
    # Build option text.
    letters = ["A", "B", "C", "D"]
    id_to_obj = {o.object_id: o for o in scene.objects}
    prompt = (
        "Looking at the whole video above, which of the following objects is, on "
        "average, CLOSEST to the camera during the video?\n"
    )
    correct = None
    opt_object_ids = []
    for L, (oid, (d, _)) in zip(letters, options):
        desc = _describe(id_to_obj[oid])
        prompt += f"{L}. {desc}\n"
        opt_object_ids.append(oid)
        if oid == closest[0]:
            correct = L
    prompt += "\nAnswer with only one letter: A, B, C, or D."
    return prompt, opt_object_ids, correct


def parse_answer(text: str) -> str | None:
    m = re.search(r"\b([A-Da-d])\b", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"([A-Da-d])[\).]", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"[A-Da-d]", text)
    return m.group(0).upper() if m else None


def run_vlm(wrapper, frame_paths: list[str], prompt: str, max_new_tokens: int = 10) -> str:
    """Shares logic with scripts/cam_motion_vqa.py run_vlm_on_video."""
    import torch
    is_qwen = isinstance(wrapper, Qwen25VLWrapper)

    if is_qwen:
        from qwen_vl_utils import process_vision_info
        content = [
            {"type": "video", "video": list(frame_paths), "fps": 1.0},
            {"type": "text", "text": prompt},
        ]
        messages = [{"role": "user", "content": content}]
        text = wrapper.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        proc_inputs = wrapper.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(wrapper.device)
    else:
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
        text = wrapper.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        proc_inputs = wrapper.processor(
            text=[text], videos=[frames], return_tensors="pt"
        ).to(wrapper.device)
    with torch.no_grad():
        out = wrapper.model.generate(**proc_inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = proc_inputs["input_ids"].shape[1]
    gen_tokens = out[0, input_len:]
    decoded = wrapper.processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return decoded


def main() -> int:
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--out", required=True)
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

    rng = np.random.default_rng(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_records: list[dict] = []

    try:
        from tqdm import tqdm
        it = tqdm(scene_dirs, desc=f"Depth VQA {mcfg.get('name', family)}")
    except ImportError:
        it = scene_dirs

    for sd in it:
        try:
            s = Scene.load(sd)
            q = build_question(s, rng)
            if q is None:
                out_records.append({"scene_id": s.scene_id, "error": "< 4 visible objects"})
                continue
            prompt, opt_oids, gt_letter = q
            frame_paths = [f"file://{(sd / f.image_path).resolve()}" for f in s.frames]
            t0 = time.time()
            answer = run_vlm(wrapper, frame_paths, prompt)
            elapsed = time.time() - t0
            pred_letter = parse_answer(answer)
            out_records.append({
                "scene_id": s.scene_id,
                "gt_letter": gt_letter,
                "pred_letter": pred_letter,
                "answer_raw": answer,
                "correct": bool(pred_letter == gt_letter),
                "elapsed_sec": float(elapsed),
                "option_object_ids": opt_oids,
                "prompt": prompt,
            })
        except Exception as e:
            out_records.append({"scene_id": sd.name, "error": f"{type(e).__name__}: {e}"})
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    out_path.write_text(json.dumps(out_records, indent=2))
    n_valid = sum(1 for r in out_records if "correct" in r)
    n_ok = sum(1 for r in out_records if r.get("correct"))
    acc = n_ok / max(n_valid, 1)
    from collections import Counter
    gt_dist = Counter(r["gt_letter"] for r in out_records if "gt_letter" in r)
    pred_dist = Counter(r.get("pred_letter") for r in out_records if "pred_letter" in r)
    print(f"\n[result] {out_path} acc={n_ok}/{n_valid}={acc:.1%}  GT={dict(gt_dist)}  pred={dict(pred_dist)}")
    wrapper.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
