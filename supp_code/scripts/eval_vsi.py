#!/usr/bin/env python
"""Evaluate a Dirichlet-trained VLM checkpoint on VSI-Bench (real-world).

For multiple-choice questions, picks the option with highest mean
log-probability (length-normalized). For numeric questions, falls back
to log-prob comparison against GT vs perturbed-GT distractors.

Output: a JSON with overall accuracy, per-question-type accuracy, and
per-example predictions.
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
from train_qwen_dirichlet import (  # noqa: E402
    JsonlDataset, build_chat_prompt, make_labels,
)

logger = logging.getLogger("vsi")


def make_distractors_for_numeric(gt: str) -> list[str]:
    """Build numeric distractors from a GT scalar."""
    try:
        v = float(gt)
    except (ValueError, TypeError):
        return []
    perturbations = [v * 0.5, v * 1.5, v * 2.0, v * 0.25]
    out = []
    for p in perturbations:
        if abs(p - v) < 1e-6:
            continue
        # Format like GT
        if "." in gt:
            out.append(f"{p:.2f}")
        else:
            out.append(str(int(round(p))))
    return list(dict.fromkeys(out))[:3]  # dedupe, max 3


@torch.no_grad()
def score_candidate(model, processor, ex, candidate, device) -> float:
    """Mean log-prob per supervised token of `candidate` as the assistant turn."""
    image = Image.open(ex["image_path"]).convert("RGB")
    user_text = ex["question"]
    if ex.get("options"):
        # Append the options to the question
        opt_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(ex["options"]))
        user_text = f"{user_text}\n{opt_str}"

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
    return float(-out.loss)  # mean log-prob per token


def _option_letter(option_str: str) -> str | None:
    """Extract the leading letter from an option like 'A. back-right' -> 'A'."""
    s = option_str.strip()
    if len(s) >= 2 and s[0].isalpha() and s[0].isupper() and s[1] in (".", ")", " "):
        return s[0]
    return None


@torch.no_grad()
def predict(model, processor, ex, device) -> tuple[str | None, dict]:
    """Return (predicted_answer, score_dict).

    For MC questions: returns the letter ('A', 'B', etc.) of the option
    with highest mean log-prob — matching VSI-Bench's letter-only GT format.
    For numeric questions: returns the predicted scalar string.
    """
    options = ex.get("options")
    gt = ex["answer"]
    if options:
        candidates = list(options)
    else:
        distractors = make_distractors_for_numeric(gt)
        if not distractors:
            return None, {}
        candidates = [gt] + distractors

    scores = {}
    for cand in candidates:
        scores[cand] = score_candidate(model, processor, ex, cand, device)
    if not scores:
        return None, {}
    best_str = max(scores, key=scores.get)
    if options:
        pred = _option_letter(best_str) or best_str
    else:
        pred = best_str
    return pred, scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--checkpoint", type=Path, required=False, default=None,
                   help="LoRA adapter directory (omit to evaluate the un-tuned base model)")
    p.add_argument("--vsi-jsonl", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-eval", type=int, default=-1)
    p.add_argument("--mc-only", action="store_true",
                   help="Skip numeric questions (evaluate only multiple-choice)")
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
    if args.mc_only:
        val = [ex for ex in val if ex.get("options")]
    if args.n_eval > 0:
        val = val[: args.n_eval]
    logger.info("Eval %d examples (mc_only=%s)", len(val), args.mc_only)

    rows = []
    correct_total = 0
    by_kind = {}
    t0 = time.time()
    for i, ex in enumerate(val):
        try:
            pred, scores = predict(model, processor, ex, device)
        except Exception as e:  # noqa: BLE001
            logger.warning("ex %d failed: %s", i, e)
            pred, scores = None, {}
        is_correct = pred is not None and str(pred).strip() == str(ex["answer"]).strip()
        rows.append({
            "vsi_id": ex.get("vsi_id"),
            "scene_id": ex.get("scene_id"),
            "kind": ex["kind"],
            "is_mc": ex.get("options") is not None,
            "gt": ex["answer"],
            "pred": pred,
            "correct": is_correct,
        })
        if is_correct:
            correct_total += 1
        by_kind.setdefault(ex["kind"], {"n": 0, "correct": 0})
        by_kind[ex["kind"]]["n"] += 1
        by_kind[ex["kind"]]["correct"] += int(is_correct)
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            logger.info("  progress %d/%d  acc=%.3f  (%.1fs/it)",
                        i + 1, len(val), correct_total / (i + 1),
                        elapsed / (i + 1))

    summary = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else "base",
        "model_id": args.model_id,
        "n_total": len(val),
        "n_correct": correct_total,
        "accuracy": correct_total / max(len(val), 1),
        "by_kind": {k: {"n": v["n"], "acc": v["correct"] / v["n"]}
                    for k, v in by_kind.items()},
        "wall_time_s": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    logger.info("Done: acc=%.4f (%d/%d) in %.1fs",
                summary["accuracy"], summary["n_correct"],
                summary["n_total"], summary["wall_time_s"])


if __name__ == "__main__":
    main()
