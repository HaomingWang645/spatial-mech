#!/usr/bin/env python
"""Per-head activation extraction.

Mirrors ``scripts/extract_activations.py`` in video mode, but for each
requested layer captures every attention head's **residual-stream
contribution** separately — the additive term

    c_h = x_h @ W_O^(h).T          (shape (D,))

where ``x_h`` is the h-th chunk (size ``head_dim``) of the concatenated
multi-head output feeding ``self_attn.o_proj`` and
``W_O^(h) = o_proj.weight[:, h*d:(h+1)*d]``. Summed over h (plus bias, if any)
this equals ``o_proj(x)``.

Pooling follows the existing ``extract_scene_video`` logic: per temporal
token, the two source-frame coverage maps are averaged, and per (object, t)
we mean-pool the per-head contributions over patches whose coverage ≥ threshold.

Output:
    <out>/head_layer_LL.npy         shape (N_rows, n_heads, D) in fp16
    <out>/head_layer_LL.parquet     scene_id / object_id / frame_id / layer / vec_row / centroid_{x,y,z}
"""
import _bootstrap  # noqa: F401

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from spatial_subspace.extract import mask_to_patch_coverage, pool_object_vector
from spatial_subspace.models import Qwen25VLWrapper
from spatial_subspace.scene import Scene
from spatial_subspace.utils import ensure_dir, load_yaml, set_seed


def parse_layers(spec: str, n_layers: int) -> list[int]:
    if spec == "all":
        return list(range(n_layers))
    return sorted(set(int(x) for x in spec.split(",")))


def pool_per_head_on_gpu(
    x: torch.Tensor,          # (B=1, T_seq, n_heads*head_dim)
    W: torch.Tensor,          # (D, n_heads*head_dim)
    n_heads: int,
    head_dim: int,
    visual_slice: tuple[int, int],
    grid: tuple[int, int, int],
    coverage_per_t: list[dict[int, np.ndarray]],
    object_ids: list[int],
    threshold: float,
) -> dict[tuple[int, int], np.ndarray]:
    """Compute per-head residual-stream contributions at the visual tokens,
    pool each (object, temporal_token) with the coverage mask, and return a
    dict {(object_id, t): (n_heads, D) float32 array}.

    Done entirely in the layer's native (bf16) dtype on the GPU to minimise
    memory + transfer cost. Per-scene budget: one (T_vis, n_heads, D) tensor
    in bf16 per layer = 2080 * 28 * 3584 * 2 bytes ≈ 400 MB. Free after pool.
    """
    start, end = visual_slice
    t_post, gh, gw = grid
    x_vis = x[0, start:end, :].detach()                  # (T_vis, n_heads*head_dim)
    xh = x_vis.view(x_vis.shape[0], n_heads, head_dim)   # (T_vis, n_heads, head_dim)
    Wh = W.detach().view(W.shape[0], n_heads, head_dim)  # (D, n_heads, head_dim)
    # Per-token per-head residual contribution: (T_vis, n_heads, D)
    with torch.no_grad():
        per = torch.einsum("thd,ohd->tho", xh.to(Wh.dtype), Wh)
    # Reshape to the video grid: (t_post, gh, gw, n_heads, D)
    per = per.view(t_post, gh, gw, n_heads, W.shape[0])
    # Pool on GPU per (object, t).
    out: dict[tuple[int, int], np.ndarray] = {}
    for t in range(t_post):
        cov_t = coverage_per_t[t]
        per_t = per[t]  # (gh, gw, n_heads, D)
        for oid in object_ids:
            cov = cov_t[oid]
            mask = cov >= threshold
            if not mask.any():
                continue
            # Gather patches above threshold.
            ii, jj = np.where(mask)
            w_np = cov[mask].astype(np.float32)
            sel = per_t[ii, jj, :, :]  # (k, n_heads, D) bf16
            w_t = torch.as_tensor(w_np, device=sel.device, dtype=torch.float32)
            sel_fp = sel.to(torch.float32)
            # Weighted mean along the patch axis.
            with torch.no_grad():
                num = (sel_fp * w_t[:, None, None]).sum(dim=0)
                pooled_t = num / w_t.sum()
            pooled = pooled_t.detach().cpu().numpy().astype(np.float16)
            out[(oid, t)] = pooled
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--layers", default="all",
                   help="comma-separated layer indices, or 'all' (default)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.3)
    args = p.parse_args()

    set_seed(args.seed)
    mcfg = load_yaml(args.model_config)
    out = ensure_dir(Path(args.out))

    print(f"[load] {mcfg['hf_id']}  device_map={mcfg.get('device_map')}")
    w = Qwen25VLWrapper(
        hf_id=mcfg["hf_id"],
        torch_dtype=mcfg.get("torch_dtype", "bfloat16"),
        device=mcfg.get("device", "cuda"),
        device_map=mcfg.get("device_map"),
    )
    n_heads, head_dim = w.attn_head_dims()
    tps = w.temporal_patch_size()
    layers_module = w._locate_layers()
    n_layers = len(layers_module)
    layers = parse_layers(args.layers, n_layers)
    w.enable_head_capture(layers)
    print(f"[cfg] n_heads={n_heads} head_dim={head_dim} "
          f"n_layers={n_layers} layers={layers}")

    # Cache W_O weights once (GPU-resident, same dtype as captures).
    W_o: dict[int, torch.Tensor] = {L: w.o_proj_weight(L) for L in layers}

    scenes_dir = Path(args.scenes)
    scene_dirs = sorted(
        d for d in scenes_dir.iterdir() if d.is_dir() and (d / "scene.json").exists()
    )
    if args.limit is not None:
        scene_dirs = scene_dirs[: args.limit]
    print(f"[data] {len(scene_dirs)} scenes under {scenes_dir}")

    rows_by_layer: dict[int, list[dict]] = defaultdict(list)
    vecs_by_layer: dict[int, list[np.ndarray]] = defaultdict(list)

    try:
        for d in tqdm(scene_dirs, desc="per-head extract"):
            scene = Scene.load(d)
            if not scene.frames:
                continue
            frame_paths = [
                f"file://{(d / f.image_path).resolve()}" for f in scene.frames
            ]
            masks_raw = [np.array(Image.open(d / f.mask_path)) for f in scene.frames]

            fwd = w.forward(frame_paths, mcfg["prompt"])
            img_h, img_w = w.image_input_hw(fwd)
            masks_resized = [
                np.array(Image.fromarray(m).resize((img_w, img_h), Image.NEAREST))
                for m in masks_raw
            ]
            t_post, gh, gw = fwd.grid
            expected_in = t_post * tps
            if len(masks_resized) != expected_in:
                raise RuntimeError(
                    f"frame count {len(masks_resized)} != T*tps={expected_in}"
                )

            object_ids = [o.object_id for o in scene.objects]
            coverage_per_t: list[dict[int, np.ndarray]] = []
            for t in range(t_post):
                slot = masks_resized[t * tps : (t + 1) * tps]
                slot_covs = [mask_to_patch_coverage(m, (gh, gw), object_ids) for m in slot]
                merged: dict[int, np.ndarray] = {}
                for oid in object_ids:
                    merged[oid] = np.mean(np.stack([sc[oid] for sc in slot_covs]), axis=0)
                coverage_per_t.append(merged)

            head_inputs = fwd.extras["head_inputs"]
            for L in layers:
                pooled = pool_per_head_on_gpu(
                    head_inputs[L], W_o[L], n_heads, head_dim,
                    fwd.visual_token_range, fwd.grid,
                    coverage_per_t, object_ids, args.threshold,
                )
                for (oid, t), vec in pooled.items():
                    obj = scene.objects[oid]
                    rows_by_layer[L].append({
                        "scene_id": scene.scene_id,
                        "object_id": int(oid),
                        "frame_id": int(t),
                        "layer": int(L),
                        "centroid_x": float(obj.centroid[0]),
                        "centroid_y": float(obj.centroid[1]),
                        "centroid_z": float(obj.centroid[2]),
                    })
                    vecs_by_layer[L].append(vec)
            # Drop GPU tensors for this scene before the next forward.
            for k in list(head_inputs):
                del head_inputs[k]
            del fwd, head_inputs
            torch.cuda.empty_cache()
    finally:
        w.close()

    # Write one parquet + one npy per layer.
    for L, rows in rows_by_layer.items():
        df = pd.DataFrame(rows)
        df["vec_row"] = np.arange(len(rows), dtype=np.int64)
        df.to_parquet(out / f"head_layer_{L:02d}.parquet")
        arr = np.stack(vecs_by_layer[L], axis=0).astype(np.float16)  # (N, H, D)
        np.save(out / f"head_layer_{L:02d}.npy", arr)
        print(f"[L{L:02d}] saved {arr.shape} fp16  →  "
              f"{(arr.nbytes / 2**20):.1f} MiB")

    print(f"\ndone. out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
