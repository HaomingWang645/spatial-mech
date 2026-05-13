#!/usr/bin/env python
"""Per-example rescue analysis: for each pair (baseline, Dirichlet)
checkpoint pair, compute the per-example correct/incorrect outcomes
and report:
  - both_correct
  - baseline_only (baseline correct, Dirichlet wrong)
  - dirichlet_only (Dirichlet correct, baseline wrong) ← rescue
  - both_wrong

A non-trivial dirichlet_only count indicates the Dirichlet model
captures information the baseline lacks even if overall accuracy is
similar.

This script writes per-example predictions to a JSONL and prints
a confusion summary.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train_qwen_dirichlet import (  # noqa: E402
    JsonlDataset, build_chat_prompt, make_labels, _build_distractor,
)


def predict_choice(model, processor, ex, device):
    """Return 'true' if the model picks the true answer over distractor,
    else 'false'.  Returns None if no distractor available."""
    distractor = _build_distractor(ex)
    if distractor is None:
        return None
    cands = [ex["answer"], distractor]
    avg_lps = []
    for cand in cands:
        image = Image.open(ex["image_path"]).convert("RGB")
        user_text, _ = build_chat_prompt(ex)
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": cand}]},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = make_labels(inputs["input_ids"][0], cand, processor).unsqueeze(0)
        n = int((labels != -100).sum())
        if n == 0:
            return None
        with torch.no_grad():
            out = model(**inputs, labels=labels)
        avg_lps.append(float(-out.loss))  # mean log-prob per token
    return "true" if avg_lps[0] > avg_lps[1] else "false"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-ckpt", type=Path, required=True)
    p.add_argument("--dirichlet-ckpt", type=Path, required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--val-jsonl", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-eval", type=int, default=500)
    args = p.parse_args()
    device = torch.device("cuda:0")

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(args.model_id)
    val = JsonlDataset(args.val_jsonl).examples[: args.n_eval]

    rows = []
    for tag, ckpt in [("baseline", args.baseline_ckpt), ("dirichlet", args.dirichlet_ckpt)]:
        print(f"Loading {tag}: {ckpt}")
        base = AutoModelForImageTextToText.from_pretrained(
            args.model_id, torch_dtype=torch.bfloat16).to(device)
        m = PeftModel.from_pretrained(base, str(ckpt))
        m.eval()
        for i, ex in enumerate(val):
            r = predict_choice(m, processor, ex, device)
            rows.append({"idx": i, "tag": tag, "ex_id": ex.get("scene_id"),
                         "kind": ex.get("kind"), "result": r})
        del m, base
        torch.cuda.empty_cache()

    # Pivot
    by_idx = {}
    for r in rows:
        by_idx.setdefault(r["idx"], {})[r["tag"]] = r["result"]

    both_correct = baseline_only = dirichlet_only = both_wrong = skipped = 0
    for idx, d in by_idx.items():
        b, dr = d.get("baseline"), d.get("dirichlet")
        if b is None or dr is None:
            skipped += 1
            continue
        if b == "true" and dr == "true":
            both_correct += 1
        elif b == "true" and dr == "false":
            baseline_only += 1
        elif b == "false" and dr == "true":
            dirichlet_only += 1
        else:
            both_wrong += 1

    summary = {
        "n_total": len(by_idx) - skipped,
        "both_correct": both_correct,
        "baseline_only_correct": baseline_only,
        "dirichlet_only_correct": dirichlet_only,
        "both_wrong": both_wrong,
        "rescue_rate": (
            dirichlet_only / max(baseline_only + dirichlet_only + both_wrong, 1)
        ),
        "baseline_acc": (both_correct + baseline_only) / max(len(by_idx) - skipped, 1),
        "dirichlet_acc": (both_correct + dirichlet_only) / max(len(by_idx) - skipped, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
