#!/usr/bin/env python
"""Evaluate a Dirichlet-trained VLM checkpoint on ViewSpatial-Bench.

Single image per question, 4-way letter MC (A/B/C/D). For each
question we score each option string ("A. right", "B. front-up", ...)
as the assistant turn and pick the highest mean log-prob, then
extract its leading letter.
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

logger = logging.getLogger("viewspatial")


def _option_letter(option_str: str) -> str | None:
    s = option_str.strip()
    if len(s) >= 2 and s[0].isalpha() and s[0].isupper() and s[1] in (".", ")", " "):
        return s[0]
    return None


@torch.no_grad()
def score_candidate(model, processor, image, question, choices, candidate, device) -> float:
    user_text = f"{question}\n{choices}"
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": candidate}]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = make_labels(inputs["input_ids"][0], candidate, processor).unsqueeze(0)
    n = int((labels != -100).sum())
    if n == 0:
        return float("-inf")
    out = model(**inputs, labels=labels)
    return float(-out.loss)


def parse_choices(choices_str: str) -> list[str]:
    return [s.strip() for s in choices_str.split("\n") if s.strip()]


def resolve_image_path(json_path: str, image_root: Path) -> Path:
    p = json_path
    if p.startswith("ViewSpatial-Bench/"):
        p = p[len("ViewSpatial-Bench/"):]
    return image_root / p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--bench-json", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-eval", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for stratified subsample by question_type")
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

    data = json.load(args.bench_json.open())
    if args.n_eval > 0 and args.n_eval < len(data):
        # Stratified subsample: take first k_per_type per question_type
        import random
        rng = random.Random(args.seed)
        by_type = {}
        for ex in data:
            by_type.setdefault(ex["question_type"], []).append(ex)
        per_type = max(1, args.n_eval // len(by_type))
        subset = []
        for qt, exs in by_type.items():
            rng.shuffle(exs)
            subset.extend(exs[:per_type])
        rng.shuffle(subset)
        data = subset[:args.n_eval]
    logger.info("Eval %d examples on ViewSpatial-Bench", len(data))

    rows = []
    correct = 0
    by_type = {}
    t0 = time.time()
    for i, ex in enumerate(data):
        try:
            choices = parse_choices(ex["choices"])
            img_path = resolve_image_path(ex["image_path"][0], args.image_root)
            if not img_path.exists():
                raise FileNotFoundError(str(img_path))
            image = Image.open(img_path).convert("RGB")
            scores = {}
            for cand in choices:
                scores[cand] = score_candidate(
                    model, processor, image, ex["question"], ex["choices"], cand, device)
            best_str = max(scores, key=scores.get) if scores else None
            pred_letter = _option_letter(best_str) if best_str else None
            gt_letter = _option_letter(ex["answer"])
            is_correct = pred_letter is not None and pred_letter == gt_letter
        except Exception as e:  # noqa: BLE001
            logger.warning("ex %d failed: %s", i, e)
            best_str, pred_letter, gt_letter, is_correct = None, None, None, False
        rows.append({
            "id": i,
            "question_type": ex["question_type"],
            "gt": ex["answer"],
            "pred_str": best_str,
            "pred_letter": pred_letter,
            "gt_letter": gt_letter,
            "correct": is_correct,
        })
        if is_correct:
            correct += 1
        qt = ex["question_type"]
        by_type.setdefault(qt, {"n": 0, "c": 0})
        by_type[qt]["n"] += 1
        by_type[qt]["c"] += int(is_correct)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            logger.info("  progress %d/%d  acc=%.3f  (%.1fs/it)",
                        i + 1, len(data), correct / (i + 1),
                        elapsed / (i + 1))

    summary = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else "base",
        "model_id": args.model_id,
        "n_total": len(data),
        "n_correct": correct,
        "accuracy": correct / max(len(data), 1),
        "by_question_type": {k: {"n": v["n"], "acc": v["c"] / v["n"]}
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
