#!/usr/bin/env python
"""Run a VLM over rendered scenes and dump per-layer pooled object vectors.

Example:
    python scripts/extract_activations.py \
        --data-root data/tier_a --out-dir data/activations/tier_a \
        --model-config configs/models/qwen25vl.yaml --tier A --limit 200
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from spatial_subspace.extract import ExtractConfig, run_extraction
from spatial_subspace.models import Qwen25VLWrapper
from spatial_subspace.utils import load_yaml, set_seed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--tier", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overlap-threshold", type=float, default=0.3)
    p.add_argument("--layers", type=str, default=None,
                   help="comma-separated layer indices; default = all")
    p.add_argument("--mode", choices=["image", "video"], default="image",
                   help="image: each frame is a separate forward; "
                        "video: all frames bundled into one video forward")
    args = p.parse_args()

    set_seed(args.seed)
    mcfg = load_yaml(args.model_config)

    wrapper = Qwen25VLWrapper(
        hf_id=mcfg["hf_id"],
        torch_dtype=mcfg.get("torch_dtype", "bfloat16"),
        device=mcfg.get("device", "cuda"),
    )
    cfg = ExtractConfig(
        overlap_threshold=args.overlap_threshold,
        layers=[int(x) for x in args.layers.split(",")] if args.layers else None,
    )
    try:
        run_extraction(
            wrapper=wrapper,
            data_root=Path(args.data_root),
            out_dir=Path(args.out_dir),
            prompt=mcfg["prompt"],
            cfg=cfg,
            tier=args.tier,
            limit=args.limit,
            mode=args.mode,
        )
    finally:
        wrapper.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
