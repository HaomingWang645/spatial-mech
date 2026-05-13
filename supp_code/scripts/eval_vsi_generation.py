#!/usr/bin/env python
"""Free-form generation evaluator for VSI-Bench numeric questions.

Unlike eval_vsi.py / eval_vsi_batched.py, this script does NOT score
candidates by log-prob ranking. Instead, it asks the model to generate
an answer, parses the first number from the output, and computes the
official VSI-Bench MRA metric:

    MRA(pred, gt) = (1/|Θ|) Σ_θ 1[|pred-gt| / max(|pred|, |gt|) ≤ θ]

with Θ = {0.5, 0.4, 0.3, 0.2, 0.1, 0.05}.

Used to verify whether the high accuracies on numeric tasks in v8
(e.g., InternVL abs_distance @ λ=3 = 97%) are real spatial reasoning
or artifacts of distractor-ranking scoring.

For MC questions, falls back to letter-output parsing (also free-form).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train_qwen_dirichlet import JsonlDataset  # noqa: E402

logger = logging.getLogger("vsi_gen")

THRESHOLDS = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]


def parse_first_number(text: str) -> float | None:
    """Extract the first signed decimal number from a free-form text."""
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_letter(text: str) -> str | None:
    """Extract a leading uppercase letter (A/B/C/D)."""
    s = text.strip()
    for ch in s[:5]:
        if ch.isalpha() and ch.isupper() and ch in "ABCDEF":
            return ch
    return None


def mra(pred: float, gt: float) -> float:
    """Mean relative accuracy across the standard VSI-Bench thresholds."""
    if pred is None or not (gt > 0 or gt < 0):
        return 0.0
    rel = abs(pred - gt) / max(abs(pred), abs(gt))
    return sum(int(rel <= t) for t in THRESHOLDS) / len(THRESHOLDS)


@torch.no_grad()
def generate_answer(model, processor, image, user_text, device,
                     max_new_tokens: int = 32) -> str:
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--vsi-jsonl", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-eval", type=int, default=-1,
                   help="Cap; -1 = all")
    p.add_argument("--numeric-only", action="store_true",
                   help="Only score numeric questions (the suspect cells)")
    p.add_argument("--per-task-cap", type=int, default=-1,
                   help="Per task type, cap items (stratified subsample)")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
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

    val = JsonlDataset(args.vsi_jsonl).examples
    if args.numeric_only:
        val = [ex for ex in val if not ex.get("options")]
    if args.per_task_cap > 0:
        import random
        rng = random.Random(args.seed)
        by_kind = {}
        for ex in val:
            by_kind.setdefault(ex["kind"], []).append(ex)
        sub = []
        for k, exs in by_kind.items():
            rng.shuffle(exs)
            sub.extend(exs[:args.per_task_cap])
        rng.shuffle(sub)
        val = sub
    if args.n_eval > 0:
        val = val[:args.n_eval]
    logger.info("Eval %d examples (numeric_only=%s, per_task_cap=%d)",
                len(val), args.numeric_only, args.per_task_cap)

    rows = []
    by_kind = {}
    t0 = time.time()
    for i, ex in enumerate(val):
        try:
            image = Image.open(ex["image_path"]).convert("RGB")
            user_text = ex["question"]
            options = ex.get("options")
            if options:
                opt_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(options))
                user_text = f"{user_text}\n{opt_str}\nAnswer with the letter only."
            else:
                user_text = f"{user_text}\nAnswer with a single number, no units, no explanation."
            gen = generate_answer(model, processor, image, user_text, device,
                                   args.max_new_tokens)
            if options:
                pred = parse_letter(gen)
                gt = ex["answer"].strip()
                is_correct = (pred is not None and pred == gt[0].upper())
                metric_value = float(is_correct)
            else:
                pred_num = parse_first_number(gen)
                try:
                    gt_num = float(str(ex["answer"]).strip())
                except (ValueError, TypeError):
                    gt_num = None
                if pred_num is None or gt_num is None:
                    metric_value = 0.0
                    is_correct = False
                else:
                    metric_value = mra(pred_num, gt_num)
                    is_correct = metric_value >= 0.5  # tight threshold for "exact"
                pred = pred_num
        except Exception as e:  # noqa: BLE001
            logger.warning("ex %d failed: %s", i, e)
            gen = ""
            pred = None
            metric_value = 0.0
            is_correct = False

        rows.append({
            "vsi_id": ex.get("vsi_id"),
            "scene_id": ex.get("scene_id"),
            "kind": ex["kind"],
            "is_mc": ex.get("options") is not None,
            "gt": ex["answer"],
            "raw_gen": gen[:80],
            "pred": pred,
            "score": metric_value,
            "correct": is_correct,
        })
        k = ex["kind"]
        by_kind.setdefault(k, {"n": 0, "score_sum": 0.0, "exact": 0})
        by_kind[k]["n"] += 1
        by_kind[k]["score_sum"] += metric_value
        by_kind[k]["exact"] += int(is_correct)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            mean_score = sum(r["score"] for r in rows) / len(rows)
            logger.info("  progress %d/%d  mean_score=%.3f  (%.2fs/it)",
                        i + 1, len(val), mean_score, elapsed / (i + 1))

    summary = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else "base",
        "model_id": args.model_id,
        "n_total": len(rows),
        "mean_score": sum(r["score"] for r in rows) / max(len(rows), 1),
        "exact_match_acc": sum(1 for r in rows if r["correct"]) / max(len(rows), 1),
        "by_kind": {k: {"n": v["n"],
                         "mra": v["score_sum"] / v["n"],
                         "tight_acc": v["exact"] / v["n"]}
                    for k, v in by_kind.items()},
        "wall_time_s": time.time() - t0,
        "evaluator": "generation",
        "thresholds": THRESHOLDS,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    logger.info("Done: mean_score=%.4f tight_acc=%.4f n=%d in %.1fs",
                summary["mean_score"], summary["exact_match_acc"],
                summary["n_total"], summary["wall_time_s"])


if __name__ == "__main__":
    main()
