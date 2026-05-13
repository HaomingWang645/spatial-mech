#!/usr/bin/env python
"""Method B — VSI-Bench evaluation with mid-forward subspace ablation.

At inference time, hook a forward function at layer L = `--ablate-layer` that
projects the residual stream away from the col-span of `--ablate-Q` (an
orthonormal basis of shape (d, k)). Run VSI-Bench eval and write results.

Usage:
    python scripts/eval_vsi_with_ablation.py \\
        --model-id Qwen/Qwen2.5-VL-7B-Instruct \\
        --checkpoint checkpoints/qwen_lam0_seed0/lora \\
        --ablate-Q reports/probe_features/qwen_color_subspace.npz \\
        --ablate-layer 17 \\
        --ablate-tokens object  # only object tokens, or 'all' for full residual
        --vsi-jsonl data/vsi_bench_full/eval_full.jsonl \\
        --out reports/vsi_full_eval/abl_color_qwen_seed0.json
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
import numpy as np
from PIL import Image

logger = logging.getLogger("eval_abl")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


class AblationHook:
    """Forward-hook that projects activations away from col-span(Q)."""
    def __init__(self, layer_module, Q: torch.Tensor, mode: str = "all"):
        """Q: (d, k) orthonormal columns. mode: 'all' | 'object' (object tokens only).
        For 'object', the caller must set self.object_positions before each forward pass.
        """
        self.Q = Q
        self.mode = mode
        self.object_positions = None  # list of token indices to ablate (only if mode=='object')
        self._h = layer_module.register_forward_hook(self._fn)

    def _fn(self, _mod, _inp, out):
        # out is the residual stream after this block: (1, T, d) or (T, d)
        h = out[0] if isinstance(out, tuple) else out
        # Project away
        proj = h @ self.Q          # (B, T, k)
        sub = proj @ self.Q.T      # (B, T, d)
        if self.mode == "all":
            new = h - sub
        elif self.mode == "object":
            if self.object_positions is None:
                new = h
            else:
                new = h.clone()
                # ablate at object positions only
                idx = torch.tensor(self.object_positions, device=h.device, dtype=torch.long)
                if idx.numel() > 0:
                    new[..., idx, :] = h[..., idx, :] - sub[..., idx, :]
        else:
            raise ValueError(self.mode)
        if isinstance(out, tuple):
            return (new,) + out[1:]
        return new

    def close(self):
        self._h.remove()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="LoRA checkpoint dir. If omitted, eval base model.")
    p.add_argument("--ablate-Q", type=Path, required=True,
                   help="NPZ file with key 'Q' of shape (d, k).")
    p.add_argument("--ablate-layer", type=int, default=17)
    p.add_argument("--ablate-tokens", default="all", choices=["all", "object"])
    p.add_argument("--vsi-jsonl", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-max", type=int, default=None,
                   help="Max #items to evaluate (default: all).")
    p.add_argument("--max-new-tokens", type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device=%s", device)

    # Load model + processor
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

    # Find the layer module — works for both Qwen-VL and InternVL-VL
    base_model = model.base_model.model if hasattr(model, "base_model") else model
    layer_module = base_model.model.language_model.layers[args.ablate_layer]

    # Load ablation Q
    Q = np.load(args.ablate_Q, allow_pickle=True)["Q"].astype(np.float32)
    Q_t = torch.from_numpy(Q).to(device).to(torch.bfloat16)
    logger.info("Loaded Q: shape %s, will project-away at L%d (%s tokens)",
                tuple(Q.shape), args.ablate_layer, args.ablate_tokens)

    # Install hook
    hook = AblationHook(layer_module, Q_t, mode=args.ablate_tokens)

    # Load examples
    examples = [json.loads(l) for l in args.vsi_jsonl.read_text().splitlines() if l.strip()]
    if args.n_max:
        examples = examples[:args.n_max]
    logger.info("Eval %d examples (ablate=%s @ L%d, tokens=%s)",
                len(examples), args.ablate_Q.stem, args.ablate_layer, args.ablate_tokens)

    n_correct = 0; rows = []
    by_kind = {}
    t0 = time.time()
    for i, ex in enumerate(examples):
        try:
            image = Image.open(ex["image_path"]).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": ex["question"]},
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            new_tokens = out[0, inputs["input_ids"].shape[1]:]
            pred = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            gt = str(ex["answer"]).strip()
            # Simple "correct if first letter matches" for MC; for open-ended use exact-match
            correct = pred.lower().startswith(gt.lower()[:max(1, min(len(gt), 4))]) or pred == gt
            n_correct += int(correct)
            rows.append({"vsi_id": ex.get("vsi_id", i), "scene_id": ex.get("scene_id", ""),
                          "kind": ex.get("kind", "unknown"), "gt": gt, "pred": pred,
                          "correct": correct})
            kind = ex.get("kind", "unknown")
            by_kind.setdefault(kind, [0, 0])
            by_kind[kind][1] += 1
            if correct: by_kind[kind][0] += 1
        except Exception as e:
            logger.warning("ex %d failed: %s", i, e)
            rows.append({"vsi_id": ex.get("vsi_id", i), "error": str(e)[:100]})
        if (i+1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            logger.info("  progress %d/%d  acc=%.3f  (%.2fs/it)",
                        i+1, len(examples), n_correct/(i+1), 1/rate)

    elapsed = time.time() - t0
    summary = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "model_id": args.model_id, "ablate_Q": str(args.ablate_Q),
        "ablate_layer": args.ablate_layer, "ablate_tokens": args.ablate_tokens,
        "n_total": len(rows), "n_correct": n_correct,
        "accuracy": n_correct / max(1, len(rows)),
        "by_kind": {k: {"n": v[1], "acc": v[0]/max(1,v[1])} for k, v in by_kind.items()},
        "wall_time_s": elapsed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    logger.info("Done: acc=%.4f (%d/%d) in %.1fs", summary["accuracy"], n_correct, len(rows), elapsed)
    hook.close()


if __name__ == "__main__":
    main()
