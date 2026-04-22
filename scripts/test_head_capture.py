#!/usr/bin/env python
"""Smoke test the per-head o_proj-input hook.

Runs one Tier C free6dof scene through Qwen2.5-VL-7B, captures per-head inputs
at two layers, reconstructs ``o_proj(x) = sum_h W_O^(h) @ x_h`` and compares to
a direct ``o_proj.forward(x)`` call. Exits non-zero on any mismatch.
"""
import _bootstrap  # noqa: F401

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from spatial_subspace.models import Qwen25VLWrapper
from spatial_subspace.scene import Scene
from spatial_subspace.utils import load_yaml


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="data/tier_c_free6dof")
    p.add_argument("--model-config", default="configs/models/qwen25vl.yaml")
    p.add_argument("--layers", default="5,12,20")
    p.add_argument("--rtol", type=float, default=1e-2,
                   help="relative tolerance for bf16 matmul reconstruction")
    args = p.parse_args()

    mcfg = load_yaml(args.model_config)
    layer_ids = [int(x) for x in args.layers.split(",")]

    print(f"[load] {mcfg['hf_id']}")
    w = Qwen25VLWrapper(
        hf_id=mcfg["hf_id"],
        torch_dtype=mcfg.get("torch_dtype", "bfloat16"),
        device=mcfg.get("device", "cuda"),
        device_map=mcfg.get("device_map"),
    )
    n_heads, head_dim = w.attn_head_dims()
    print(f"[cfg] n_heads={n_heads}  head_dim={head_dim}  n_layers={len(w._locate_layers())}")

    try:
        w.enable_head_capture(layer_ids)
        print(f"[hook] capturing o_proj inputs at layers {layer_ids}")

        scenes_dir = Path(args.scenes)
        first = sorted(d for d in scenes_dir.iterdir() if d.is_dir() and (d / "scene.json").exists())[0]
        scene = Scene.load(first)
        print(f"[scene] {first.name}  n_frames={len(scene.frames)}  n_objects={len(scene.objects)}")

        frame_paths = [f"file://{(first / f.image_path).resolve()}" for f in scene.frames]
        out = w.forward(frame_paths, mcfg["prompt"])
        print(f"[fwd] grid={out.grid}  visual_range={out.visual_token_range}")

        head_inputs = out.extras["head_inputs"]
        assert set(head_inputs.keys()) == set(layer_ids), head_inputs.keys()

        ok = True
        for L in layer_ids:
            x = head_inputs[L]  # (B, T, H*d_head), bf16/fp16 on GPU
            W = w.o_proj_weight(L)  # (D, H*d_head)

            # Expected: o_proj(x) under the layer's own precision.
            expected = torch.nn.functional.linear(x, W)
            # Per-head reconstruction:
            B, T, Hd = x.shape
            xh = x.view(B, T, n_heads, head_dim)
            Wh = W.view(W.shape[0], n_heads, head_dim)
            # (B, T, H, D) per-head contributions
            per_head = torch.einsum("bthd,ohd->btho", xh.to(Wh.dtype), Wh)
            recon = per_head.sum(dim=2)

            diff = (recon.to(torch.float32) - expected.to(torch.float32)).abs()
            rel = diff.mean() / expected.to(torch.float32).abs().mean().clamp_min(1e-6)
            print(f"[L{L:02d}] expected.shape={tuple(expected.shape)}  "
                  f"mean|diff|={float(diff.mean()):.3e}  "
                  f"rel={float(rel):.3e}  max|diff|={float(diff.max()):.3e}")
            if float(rel) > args.rtol:
                print(f"  FAIL: rel={float(rel):.3e} > rtol={args.rtol}")
                ok = False

        # Also: sanity-check residual identity at a visual token.
        # h_{L+1} = h_L + attn_out + mlp_out; we don't capture pre-attn h_L
        # or mlp_out here, so just confirm the per-head sum matches o_proj's
        # own output exactly, which we did above.

        if ok:
            print("\nOK: per-head decomposition matches o_proj(x) within tolerance.")
            return 0
        else:
            print("\nFAIL: per-head decomposition diverged from o_proj(x).")
            return 1
    finally:
        w.close()


if __name__ == "__main__":
    sys.exit(main())
