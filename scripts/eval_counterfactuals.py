#!/usr/bin/env python
"""Method D — evaluate counterfactual color/position swap on synthetic scenes.

Reads data/counterfactuals/qa.jsonl with 3 variants per (scene, question):
  - "original"
  - "color_swap"
  - "position_swap"

For each question variant, score candidate answers and report top-1 prediction
+ logit margin. Output rows can be analyzed by:
  - For yes/no questions: how often does the answer flip vs the original?
  - For which-is-X questions: how often does the model still pick the originally-correct option?

A position-driven head should be robust to color-swap (same positions ⇒ same answer).
A color-driven head should answer-flip when colors swap.
"""
from __future__ import annotations
import _bootstrap  # noqa
import argparse
import json
import logging
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train_qwen_dirichlet import JsonlDataset  # noqa
from eval_vsi_batched import score_candidates_batched  # noqa

logger = logging.getLogger("eval_cf")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def candidates_for(qa):
    """Return candidate answers for this QA item."""
    q = qa["question"].lower()
    a = str(qa["answer"]).lower().strip()
    if a in ("yes", "no"):
        return ["yes", "no"]
    if "which is closer" in q:
        # Find the two options in the question
        # Format: "Which is closer to X: A or B?"
        # We can parse them with a simple regex.
        import re
        m = re.search(r"Which is closer to .*?: (.+?) or (.+?)\?", qa["question"])
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
    if "which is farther" in q or "which is farthest" in q:
        import re
        m = re.search(r"Which is .*?: (.+?) or (.+?)\?", qa["question"])
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
    # Fallback: include the GT and a generic distractor
    return [a, "no"]


@torch.no_grad()
def predict_one(model, processor, qa, device):
    image = Image.open(qa["image_path"]).convert("RGB")
    user_text = qa["question"]
    cands = candidates_for(qa)
    if not cands or len(set(cands)) < 2:
        return None, {}
    scores = score_candidates_batched(model, processor, image, user_text, cands, device)
    if not scores:
        return None, {}
    pred = max(scores, key=scores.get)
    return pred, scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--qa-jsonl", type=Path, default=Path("data/counterfactuals/qa.jsonl"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-eval", type=int, default=-1)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device=%s", device)

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
    model.eval()

    examples = JsonlDataset(args.qa_jsonl).examples
    if args.n_eval > 0:
        examples = examples[: args.n_eval]
    logger.info("Eval %d counterfactual examples", len(examples))

    rows = []
    t0 = time.time()
    for i, ex in enumerate(examples):
        try:
            pred, scores = predict_one(model, processor, ex, device)
        except Exception as e:
            logger.warning("ex %d failed: %s", i, e)
            pred, scores = None, {}
        gt = str(ex["answer"]).strip()
        is_correct = pred is not None and pred.strip().lower() == gt.lower()
        # logit margin
        if scores and len(scores) >= 2:
            sorted_scores = sorted(scores.values(), reverse=True)
            margin = float(sorted_scores[0] - sorted_scores[1])
        else:
            margin = float("nan")
        rows.append({
            "scene_id": ex["scene_id"],
            "base_scene_id": ex.get("base_scene_id", ""),
            "variant": ex["variant"],
            "question": ex["question"],
            "kind": ex.get("kind", "unknown"),
            "gt": gt,
            "pred": pred,
            "correct": is_correct,
            "margin": margin,
        })
        if (i+1) % 100 == 0:
            elapsed = time.time() - t0
            logger.info("  progress %d/%d  acc=%.3f  (%.2fs/it)",
                        i+1, len(examples), sum(r["correct"] for r in rows)/(i+1), elapsed/(i+1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows}, indent=1))
    by_variant = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)
    print("\nResults by variant:")
    for v, vrows in by_variant.items():
        n = len(vrows); n_correct = sum(r["correct"] for r in vrows)
        print(f"  {v:18}  n={n}  acc={n_correct/n:.3f}")
    print(f"\nTotal wall: {time.time()-t0:.1f}s")
    logger.info("Wrote %d rows to %s", len(rows), args.out)


if __name__ == "__main__":
    main()
