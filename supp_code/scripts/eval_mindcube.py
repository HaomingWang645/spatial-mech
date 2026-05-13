#!/usr/bin/env python
"""Evaluate a Dirichlet-trained VLM checkpoint on MindCube tinybench.

MindCube is a multi-view spatial-reasoning benchmark with multiple-choice
questions. Each question's answer is a letter A/B/C/D embedded in the
question text. Questions reference 1-4 view images of a scene.

We use the *first image* in each example for input (to match our
single-frame Free6DoF training). For each question, we score each
of the 4 candidate completions ("A.", "B.", "C.", "D.") via mean
log-prob and pick the highest.

Output: JSON with overall accuracy, per-category accuracy, and
per-example predictions.

Usage
-----
    CUDA_VISIBLE_DEVICES=4 python scripts/eval_mindcube.py \\
        --model-id Qwen/Qwen2.5-VL-7B-Instruct \\
        --checkpoint checkpoints/qwen_lam1_seed0/lora \\
        --mindcube-jsonl /home/haoming/mindcube_data/raw/MindCube_tinybench.jsonl \\
        --image-root /home/haoming/mindcube_data \\
        --out reports/mindcube_eval/qwen_lam1_seed0.json
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train_qwen_dirichlet import make_labels  # noqa: E402

logger = logging.getLogger("mindcube")


@torch.no_grad()
def score_letter(model, processor, ex, letter, device, image_root: Path) -> float:
    """Mean log-prob of `letter` (e.g., 'A') as the assistant's answer."""
    img_paths = ex.get("images", [])
    if not img_paths:
        return float("-inf")
    img_path = image_root / img_paths[0]
    if not img_path.exists():
        return float("-inf")
    image = Image.open(img_path).convert("RGB")

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": ex["question"]},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": letter}]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = make_labels(inputs["input_ids"][0], letter, processor).unsqueeze(0)
    n = int((labels != -100).sum())
    if n == 0:
        return float("-inf")
    out = model(**inputs, labels=labels)
    return float(-out.loss)  # mean log-prob per token


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--mindcube-jsonl", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-eval", type=int, default=-1)
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

    examples = [json.loads(l) for l in args.mindcube_jsonl.read_text().splitlines() if l.strip()]
    if args.n_eval > 0:
        examples = examples[: args.n_eval]
    logger.info("Eval %d examples on MindCube", len(examples))

    rows = []
    correct = 0
    by_category = {}
    by_type = {}
    t0 = time.time()
    for i, ex in enumerate(examples):
        try:
            scores = {}
            for letter in ['A', 'B', 'C', 'D']:
                scores[letter] = score_letter(
                    model, processor, ex, letter, device, args.image_root)
            pred = max(scores, key=scores.get) if scores else None
            gt = (ex.get("gt_answer") or "").strip().upper()
            is_correct = pred == gt
        except Exception as e:  # noqa: BLE001
            logger.warning("ex %d failed: %s", i, e)
            pred, is_correct = None, False
        rows.append({"id": ex.get("id"), "gt": ex.get("gt_answer"),
                     "pred": pred, "correct": is_correct,
                     "type": ex.get("type"),
                     "category": ex.get("category", [])})
        if is_correct:
            correct += 1
        cat_keys = ex.get("category", [])
        # Use top category as primary
        primary_cat = cat_keys[0] if cat_keys else "unknown"
        by_category.setdefault(primary_cat, {"n": 0, "c": 0})
        by_category[primary_cat]["n"] += 1
        by_category[primary_cat]["c"] += int(is_correct)
        type_k = ex.get("type", "unknown")
        by_type.setdefault(type_k, {"n": 0, "c": 0})
        by_type[type_k]["n"] += 1
        by_type[type_k]["c"] += int(is_correct)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            logger.info("  progress %d/%d  acc=%.3f  (%.2fs/it)",
                        i + 1, len(examples), correct / (i + 1),
                        elapsed / (i + 1))

    summary = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else "base",
        "model_id": args.model_id,
        "n_total": len(examples),
        "n_correct": correct,
        "accuracy": correct / max(len(examples), 1),
        "by_primary_category": {k: {"n": v["n"], "acc": v["c"] / v["n"]}
                                 for k, v in by_category.items()},
        "by_type": {k: {"n": v["n"], "acc": v["c"] / v["n"]}
                    for k, v in by_type.items()},
        "wall_time_s": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    logger.info("Done: acc=%.4f (%d/%d) in %.1fs",
                summary["accuracy"], summary["n_correct"],
                summary["n_total"], summary["wall_time_s"])


if __name__ == "__main__":
    main()
