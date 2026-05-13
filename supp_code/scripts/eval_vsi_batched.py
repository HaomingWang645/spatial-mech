#!/usr/bin/env python
"""Batched VSI-Bench evaluator: scores all N candidates in a single forward pass.

vs. eval_vsi.py: that script ran 4 forwards sequentially per question, with
image preprocessing repeated each time. Here we batch all candidates,
amortizing image preprocessing/tokenization across the batch and running a
single bs=N forward. Net speedup ~3–4× wall-clock at the same accuracy.

Output schema is identical to eval_vsi.py so existing aggregation works.
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
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train_qwen_dirichlet import JsonlDataset  # noqa: E402

logger = logging.getLogger("vsi_batched")


def make_distractors_for_numeric(gt: str) -> list[str]:
    try:
        v = float(gt)
    except (ValueError, TypeError):
        return []
    perturbations = [v * 0.5, v * 1.5, v * 2.0, v * 0.25]
    out = []
    for p in perturbations:
        if abs(p - v) < 1e-6:
            continue
        if "." in gt:
            out.append(f"{p:.2f}")
        else:
            out.append(str(int(round(p))))
    return list(dict.fromkeys(out))[:3]


def _option_letter(option_str: str) -> str | None:
    s = option_str.strip()
    if len(s) >= 2 and s[0].isalpha() and s[0].isupper() and s[1] in (".", ")", " "):
        return s[0]
    return None


def _find_last_subseq(seq: list[int], sub: list[int]) -> int:
    """Return last start index where sub matches seq, or -1."""
    n, m = len(seq), len(sub)
    for i in range(n - m, -1, -1):
        if seq[i:i+m] == sub:
            return i
    return -1


@torch.no_grad()
def score_candidates_batched(
    model, processor, image: Image.Image, user_text: str,
    candidates: list[str], device,
) -> dict[str, float]:
    """Score N candidates in a single batched forward.

    Returns {candidate: mean_log_prob}.
    """
    N = len(candidates)
    # Build N chat prompts that share the same user message but end with each candidate
    texts = []
    for cand in candidates:
        msgs = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": cand}]},
        ]
        texts.append(processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False))

    # Pass [image]*N — processor will preprocess image N times (CPU work, parallelizable
    # across cores) but the GPU forward runs once at batch=N.
    inputs = processor(
        text=texts, images=[image]*N,
        return_tensors="pt", padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]  # (N, T)

    out = model(**inputs)  # logits: (N, T, V)
    logits = out.logits

    scores = {}
    for n in range(N):
        cand = candidates[n]
        cand_ids = processor.tokenizer.encode(cand, add_special_tokens=False)
        if not cand_ids:
            scores[cand] = float("-inf")
            continue
        seq_n = input_ids[n].tolist()
        # Strip trailing pad tokens for this row
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is not None:
            # find last non-pad
            non_pad = [i for i, t in enumerate(seq_n) if t != pad_id]
            if not non_pad:
                scores[cand] = float("-inf"); continue
            last_real = non_pad[-1] + 1
            seq_n = seq_n[:last_real]
        last = _find_last_subseq(seq_n, cand_ids)
        if last < 0:
            scores[cand] = float("-inf"); continue
        L = len(cand_ids)
        # Per-token log-prob: log p(seq[t] | seq[<t]) = log_softmax(logits[n, t-1])[seq[t]]
        token_logps = []
        for t in range(last, last + L):
            if t == 0:
                continue
            log_probs_t = F.log_softmax(logits[n, t-1].float(), dim=-1)
            token_logps.append(log_probs_t[seq_n[t]].item())
        if not token_logps:
            scores[cand] = float("-inf"); continue
        scores[cand] = sum(token_logps) / len(token_logps)
    return scores


@torch.no_grad()
def predict(model, processor, ex, device) -> tuple[str | None, dict]:
    """Returns (predicted_letter_or_string, scores)."""
    image = Image.open(ex["image_path"]).convert("RGB")
    user_text = ex["question"]
    options = ex.get("options")
    if options:
        opt_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(options))
        user_text = f"{user_text}\n{opt_str}"
        candidates = list(options)
    else:
        gt = ex["answer"]
        distractors = make_distractors_for_numeric(gt)
        if not distractors:
            return None, {}
        candidates = [gt] + distractors

    scores = score_candidates_batched(model, processor, image, user_text, candidates, device)
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
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--vsi-jsonl", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-eval", type=int, default=-1)
    p.add_argument("--mc-only", action="store_true")
    p.add_argument("--ablate-Q", type=Path, default=None,
                   help="NPZ file with key 'Q' of shape (d, k). Project residual"
                        " stream away from this subspace at --ablate-layer.")
    p.add_argument("--ablate-layer", type=int, default=17,
                   help="Layer index where the ablation hook is installed.")
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

    # Install optional ablation hook
    ablation_handle = None
    if args.ablate_Q is not None:
        import numpy as _np
        Q_np = _np.load(args.ablate_Q, allow_pickle=True)["Q"].astype(_np.float32)
        Q = torch.from_numpy(Q_np).to(device).to(torch.bfloat16)
        # Find layer module
        base_model = model.base_model.model if hasattr(model, "base_model") else model
        layer_module = base_model.model.language_model.layers[args.ablate_layer]
        def _ablate_fn(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            proj = h @ Q             # (B, T, k)
            new_h = h - proj @ Q.T   # project away
            return (new_h,) + out[1:] if isinstance(out, tuple) else new_h
        ablation_handle = layer_module.register_forward_hook(_ablate_fn)
        logger.info("Ablation hook installed at L%d, Q shape=%s",
                    args.ablate_layer, tuple(Q_np.shape))

    val = JsonlDataset(args.vsi_jsonl).examples
    if args.mc_only:
        val = [ex for ex in val if ex.get("options")]
    if args.n_eval > 0:
        val = val[: args.n_eval]
    logger.info("Eval %d examples (mc_only=%s) — batched", len(val), args.mc_only)

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
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            logger.info("  progress %d/%d  acc=%.3f  (%.2fs/it)",
                        i + 1, len(val), correct_total / (i + 1),
                        elapsed / (i + 1))

    summary = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else "base",
        "model_id": args.model_id,
        "ablate_Q": str(args.ablate_Q) if args.ablate_Q else None,
        "ablate_layer": args.ablate_layer if args.ablate_Q else None,
        "n_total": len(val),
        "n_correct": correct_total,
        "accuracy": correct_total / max(len(val), 1),
        "by_kind": {k: {"n": v["n"], "acc": v["correct"] / v["n"]}
                    for k, v in by_kind.items()},
        "wall_time_s": time.time() - t0,
        "evaluator": "batched",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    logger.info("Done: acc=%.4f (%d/%d) in %.1fs",
                summary["accuracy"], summary["n_correct"],
                summary["n_total"], summary["wall_time_s"])


if __name__ == "__main__":
    main()
