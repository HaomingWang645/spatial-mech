#!/usr/bin/env python
"""Extract object-token residual-stream features from a LoRA-finetuned VLM.

For each scene's object inventory, we run the model on (image, prompt
listing all objects) and capture the layer-L hidden state at each
object-token position. Output: (H_obj, color, shape, 3D_coords) tuples
for downstream probe analysis.

Usage
-----
    CUDA_VISIBLE_DEVICES=4 python scripts/extract_lora_features.py \\
        --model-id Qwen/Qwen2.5-VL-7B-Instruct \\
        --checkpoint checkpoints/qwen_lam0_seed0/lora \\
        --val-jsonl data/dirichlet_train_v2/val_iid.jsonl \\
        --layer 17 \\
        --out reports/probe_features/qwen_lam0_seed0.npz
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train_qwen_dirichlet import (  # noqa: E402
    JsonlDataset, build_chat_prompt, find_obj_positions, LayerHook,
)

logger = logging.getLogger("extract")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--val-jsonl", type=Path, required=True)
    p.add_argument("--layer", type=int, default=17)
    p.add_argument("--out", type=Path, required=True)
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
        base_qwen = model.base_model.model
        layer_module = base_qwen.model.language_model.layers[args.layer]
    else:
        model = base
        layer_module = model.model.language_model.layers[args.layer]
    model.eval()
    hook = LayerHook(layer_module)

    examples = JsonlDataset(args.val_jsonl).examples
    # Each example has object_names + object_coords; we extract H per object.
    # Multiple QA pairs share the same scene, so dedupe by (scene_id, image_path).
    seen_scenes = {}
    for ex in examples:
        key = ex["scene_id"]
        if key not in seen_scenes:
            seen_scenes[key] = ex
    scenes = list(seen_scenes.values())
    logger.info("Extracting from %d unique scenes", len(scenes))

    H_list = []  # (n_total_objects, d)
    coords = []  # (n_total_objects, 3)
    color_ids = []
    shape_ids = []
    scene_ids = []

    for ei, ex in enumerate(scenes):
        try:
            user_text, _ = build_chat_prompt(ex)
            image = Image.open(ex["image_path"]).convert("RGB")
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ]},
                # No assistant turn — we just need the residual stream
                {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            inputs = processor(text=[text], images=[image], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            _ = model(**inputs)

            H_all = hook.captured  # (1, T, d)
            positions, mask = find_obj_positions(
                inputs["input_ids"][0], processor.tokenizer, ex["object_names"])
            for i, (pos, valid) in enumerate(zip(positions, mask)):
                if not valid:
                    continue
                h = H_all[0, pos].float().cpu().numpy()
                H_list.append(h)
                coords.append(ex["object_coords"][i])
                # Object name format: "color shape" (e.g., "blue cube")
                parts = ex["object_names"][i].split(maxsplit=1)
                if len(parts) == 2:
                    color_ids.append(parts[0])
                    shape_ids.append(parts[1])
                else:
                    color_ids.append("?")
                    shape_ids.append("?")
                scene_ids.append(ex["scene_id"])
        except Exception as e:  # noqa: BLE001
            logger.warning("scene %d failed: %s", ei, e)

        if (ei + 1) % 10 == 0:
            logger.info("  %d / %d scenes done, %d objects captured",
                        ei + 1, len(scenes), len(H_list))

    H_arr = np.stack(H_list).astype(np.float32)  # (N, d)
    coords_arr = np.array(coords, dtype=np.float32)
    color_arr = np.array(color_ids)
    shape_arr = np.array(shape_ids)
    scene_arr = np.array(scene_ids)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, H=H_arr, coords=coords_arr, colors=color_arr,
             shapes=shape_arr, scene_ids=scene_arr,
             checkpoint=str(args.checkpoint) if args.checkpoint else "base",
             model_id=args.model_id, layer=args.layer)
    logger.info("Saved %d feature vectors (d=%d) to %s",
                H_arr.shape[0], H_arr.shape[1], args.out)
    hook.close()


if __name__ == "__main__":
    main()
