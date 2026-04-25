#!/usr/bin/env python
"""Evaluate a trained Dirichlet-loss LoRA checkpoint on a given val set.

Loads the base VLM, applies the LoRA adapter from the checkpoint
directory, and runs the full eval (LM loss, Dirichlet ratio @ L,
3D-alignment R² @ L, VQA accuracy via log-prob comparison).

Usage
-----
    CUDA_VISIBLE_DEVICES=4 python scripts/eval_dirichlet_checkpoint.py \\
        --model-id Qwen/Qwen2.5-VL-7B-Instruct \\
        --checkpoint checkpoints/qwen7b_lam1_seed0/lora \\
        --val-jsonl data/dirichlet_train_v2/val_ood.jsonl \\
        --layer 17 --tau 2.0 \\
        --out reports/dirichlet_train_v2/qwen_lam1_seed0_ood.json
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

sys.path.insert(0, str(Path(__file__).parent))
from train_qwen_dirichlet import JsonlDataset, LayerHook, evaluate  # noqa: E402

logger = logging.getLogger("eval")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Directory containing adapter_model.safetensors (LoRA weights)")
    p.add_argument("--val-jsonl", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--layer", type=int, default=17)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--n-eval", type=int, default=-1,
                   help="-1 = full set")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda:0")

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    logger.info("Loading base model: %s", args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16).to(device)

    logger.info("Applying LoRA from: %s", args.checkpoint)
    model = PeftModel.from_pretrained(base, str(args.checkpoint))
    model.eval()

    base_qwen = model.base_model.model
    layer_module = base_qwen.model.language_model.layers[args.layer]
    hook = LayerHook(layer_module)

    val = JsonlDataset(args.val_jsonl).examples
    n_eval = len(val) if args.n_eval < 0 else min(args.n_eval, len(val))
    logger.info("Evaluating %d examples", n_eval)

    t0 = time.time()
    results = evaluate(
        model, processor, hook, val, args.layer, args.tau, device,
        n_eval=n_eval, vqa_accuracy=True,
    )
    results["wall_time_s"] = time.time() - t0
    results["checkpoint"] = str(args.checkpoint)
    results["val_jsonl"] = str(args.val_jsonl)
    results["model_id"] = args.model_id
    results["layer"] = args.layer
    results["tau"] = args.tau

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    logger.info(
        "Result: lm=%.4f  dir=%.4f  R²=%.4f  vqa_acc=%.4f (n=%d)  in %.1fs",
        results["val_lm_loss"], results["val_dirichlet_ratio"],
        results["val_alignment_R2"], results["vqa_accuracy"],
        results["vqa_n"], results["wall_time_s"],
    )
    hook.close()


if __name__ == "__main__":
    main()
