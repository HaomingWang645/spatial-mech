#!/usr/bin/env python
"""Evaluate a Dirichlet-trained VLM checkpoint on OST-Bench.

OST-Bench is a multi-turn online spatio-temporal benchmark; each item
has 5 images for the *current* turn (chronological) plus a question
that may reference past turns. Options come as a list of free-form
strings (not letter-prefixed); some items have 0 options (open-ended
estimation/counting) — we skip those.

For our single-frame Dirichlet adapter, we use **only the last image
of the current turn** (matching the system prompt's "based on your
state at the end (last image) of each turn"). No turn history is
provided, since standalone single-image eval matches the training
distribution.

We score each option string as the assistant turn (mean log-prob of
JSON wrapping `{"answer": "<option>"}` is too sensitive to formatting,
so we score the raw option string) and pick the highest.
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

logger = logging.getLogger("ost")


@torch.no_grad()
def score_candidate(model, processor, image, prompt, candidate, device) -> float:
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
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


def build_prompt(ex: dict) -> str:
    """Compact single-turn prompt: question + options inline."""
    q = ex["origin_question"].strip()
    opts = ex.get("option") or []
    opts_str = " / ".join(opts)
    return (f"Look at this image (the most recent observation). "
            f"{q}\nChoose one: {opts_str}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--bench-json", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-eval", type=int, default=-1)
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

    data = json.load(args.bench_json.open())
    # Keep only multi-choice items (option list non-empty)
    data = [ex for ex in data if ex.get("option")]
    logger.info("Loaded %d MC items (out of original ~10k including open-ended)", len(data))

    if args.n_eval > 0 and args.n_eval < len(data):
        import random
        rng = random.Random(args.seed)
        by_type = {}
        for ex in data:
            by_type.setdefault(ex["type"], []).append(ex)
        per_type = max(1, args.n_eval // len(by_type))
        subset = []
        for t, exs in by_type.items():
            rng.shuffle(exs)
            subset.extend(exs[:per_type])
        rng.shuffle(subset)
        data = subset[:args.n_eval]
    logger.info("Eval %d examples on OST-Bench", len(data))

    rows = []
    correct = 0
    by_type = {}
    t0 = time.time()
    for i, ex in enumerate(data):
        try:
            obs = ex.get("new_observations") or []
            if not obs:
                raise ValueError("no observations")
            img_path = args.image_root / obs[-1]  # last image of this turn
            if not img_path.exists():
                raise FileNotFoundError(str(img_path))
            image = Image.open(img_path).convert("RGB")
            opts = ex["option"]
            prompt = build_prompt(ex)
            scores = {}
            for cand in opts:
                scores[cand] = score_candidate(
                    model, processor, image, prompt, cand, device)
            pred = max(scores, key=scores.get) if scores else None
            gt = str(ex["answer"]).strip()
            is_correct = pred is not None and pred.strip() == gt
        except Exception as e:  # noqa: BLE001
            logger.warning("ex %d failed: %s", i, e)
            pred, is_correct = None, False
        rows.append({
            "id": i,
            "type": ex.get("type"),
            "turn_id": ex.get("turn_id"),
            "scan_id": ex.get("scan_id"),
            "gt": ex.get("answer"),
            "pred": pred,
            "correct": is_correct,
        })
        if is_correct:
            correct += 1
        t = ex.get("type", "unknown")
        by_type.setdefault(t, {"n": 0, "c": 0})
        by_type[t]["n"] += 1
        by_type[t]["c"] += int(is_correct)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            logger.info("  progress %d/%d  acc=%.3f  (%.1fs/it)",
                        i + 1, len(data), correct / (i + 1),
                        elapsed / (i + 1))

    # Roll up to coarse families (Agent_visible_info, Agent_object_spatial, Agent_state)
    by_family = {}
    for k, v in by_type.items():
        family = k.split("-")[0] if "-" in k else k
        by_family.setdefault(family, {"n": 0, "c": 0})
        by_family[family]["n"] += v["n"]
        by_family[family]["c"] += v["c"]

    summary = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else "base",
        "model_id": args.model_id,
        "n_total": len(data),
        "n_correct": correct,
        "accuracy": correct / max(len(data), 1),
        "by_type": {k: {"n": v["n"], "acc": v["c"] / v["n"]}
                    for k, v in by_type.items()},
        "by_family": {k: {"n": v["n"], "acc": v["c"] / v["n"]}
                      for k, v in by_family.items()},
        "wall_time_s": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    logger.info("Done: acc=%.4f (%d/%d) in %.1fs",
                summary["accuracy"], summary["n_correct"],
                summary["n_total"], summary["wall_time_s"])


if __name__ == "__main__":
    main()
