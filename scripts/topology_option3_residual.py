#!/usr/bin/env python
"""Residualized topology (linear-representation-hypothesis decomposition).

Under the LRH, the representation of an object in a VLM is
    h_obj ≈ emb(shape) + emb(color) + emb(pos) + ε
We estimate the first two terms empirically as conditional means across
scenes, then subtract to leave a position-only residual, and rerun the same
topology metrics as `topology_option3.py` on those residuals.

Conditional means (computed per layer, aggregated over ALL (scene, object,
frame) rows in the extraction):
    emb(color=c) = mean over {h_obj | color(obj)=c}
    emb(shape=s) = mean over {h_obj | shape(obj)=s}
The residual for each object-frame is
    h_pos[obj, frame] = h_obj - emb(color=color(obj)) - emb(shape=shape(obj))
which then goes through the identical per-scene pooling and topology battery
as in `topology_option3.py`.

The shape/color metadata lives in `scene.json` (not the extraction parquet),
so this script reads from both.

Outputs match `topology_option3.py`:
    <out>/layer_metrics.parquet    # per (scene, layer) row
    <out>/summary.parquet          # per layer aggregate
    <out>/pca_examples.npz         # PCA-top-3 per (scene, layer) subset
    <out>/conditional_means.npz    # dumped emb(color), emb(shape) per layer

Invocation:
    python scripts/topology_option3_residual.py \
        --activations data/activations/tier_c_free6dof_qwen25vl_7b \
        --scenes-root data/tier_c_free6dof \
        --out data/probes/topology_option3_residual/qwen25vl_7b_f16 \
        --knn-k 2 --n-permutations 100 --pca-example-scenes 12
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from topology_option3 import (  # reuse
    SceneMetrics,
    compute_scene_metrics,
    list_layers,
    load_layer,
    pool_scene_reps,
)


# ---------------------------------------------------------------------------
# Scene-metadata loader (shape/color per object)
# ---------------------------------------------------------------------------


def load_scene_labels(scenes_root: Path) -> dict[str, dict[int, dict]]:
    """Return {scene_id: {object_id: {"shape": ..., "color": ...}}} by reading scene.json."""
    out: dict[str, dict[int, dict]] = {}
    for d in sorted(Path(scenes_root).iterdir()):
        if not d.is_dir() or not (d / "scene.json").exists():
            continue
        scene = json.loads((d / "scene.json").read_text())
        sid = scene["scene_id"]
        m: dict[int, dict] = {}
        for o in scene["objects"]:
            m[int(o["object_id"])] = {"shape": o["shape"], "color": o["color"]}
        out[sid] = m
    return out


def attach_labels(df: pd.DataFrame, labels: dict[str, dict[int, dict]]) -> pd.DataFrame:
    df = df.copy()
    shapes = []
    colors = []
    keep_mask = []
    for sid, oid in zip(df["scene_id"].tolist(), df["object_id"].tolist()):
        info = labels.get(sid, {}).get(int(oid))
        if info is None:
            shapes.append(None)
            colors.append(None)
            keep_mask.append(False)
        else:
            shapes.append(info["shape"])
            colors.append(info["color"])
            keep_mask.append(True)
    df["shape"] = shapes
    df["color"] = colors
    df = df[keep_mask].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Residualization
# ---------------------------------------------------------------------------


def compute_conditional_means(
    df: pd.DataFrame, vecs: np.ndarray, key: str
) -> dict[str, np.ndarray]:
    """For each category value of ``key`` (e.g. "color" or "shape") compute
    the mean vector across all rows with that value, using df.vec_row to
    index into ``vecs``.
    """
    out: dict[str, np.ndarray] = {}
    for val, idx in df.groupby(key).groups.items():
        rows = df.loc[idx, "vec_row"].to_numpy()
        out[val] = vecs[rows].astype(np.float64).mean(axis=0).astype(np.float32)
    return out


def residualize(
    df: pd.DataFrame,
    vecs: np.ndarray,
    color_mean: dict[str, np.ndarray],
    shape_mean: dict[str, np.ndarray],
) -> np.ndarray:
    """Return a new array same shape as `vecs` with each row replaced by
    ``vecs[r] - emb(color[r]) - emb(shape[r])``. Rows must already be
    label-augmented (df must have 'color' and 'shape' columns and
    df['vec_row'] must be an identity map 0..N-1 after reset_index).
    """
    # Build the subtraction once per row:
    n_rows, d = vecs.shape
    out = np.empty_like(vecs, dtype=np.float32)
    color_arr = df["color"].to_numpy()
    shape_arr = df["shape"].to_numpy()
    row_arr = df["vec_row"].to_numpy()
    for i in range(n_rows):
        src_row = int(row_arr[i]) if i < len(row_arr) else i
        out[i] = vecs[src_row] - color_mean[color_arr[i]] - shape_mean[shape_arr[i]]
    return out


def _collect_vec_rows(df_full: pd.DataFrame, vecs: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a (df, vecs_sub) pair where vecs_sub is indexed 0..N-1 in the
    same order as df, and df['vec_row'] is reset accordingly. This is so
    downstream pooling sees a clean mapping."""
    rows = df_full["vec_row"].to_numpy()
    vecs_sub = vecs[rows].copy()
    df_reset = df_full.reset_index(drop=True)
    df_reset["vec_row"] = np.arange(len(df_reset), dtype=np.int64)
    return df_reset, vecs_sub


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True)
    ap.add_argument("--scenes-root", required=True,
                    help="Directory holding <scene_dir>/scene.json for each scene")
    ap.add_argument("--out", required=True)
    ap.add_argument("--knn-k", type=int, default=2)
    ap.add_argument("--n-permutations", type=int, default=100)
    ap.add_argument("--layers", type=str, default=None)
    ap.add_argument("--limit-scenes", type=int, default=None)
    ap.add_argument("--pca-example-scenes", type=int, default=12)
    ap.add_argument("--t-min", type=int, default=0,
                    help="Only use frames with frame_id >= t-min (latter-frames filter). "
                    "Applied before residualization, so conditional means use the same "
                    "frame subset as the pooled topology metrics.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    act_dir = Path(args.activations)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = list_layers(act_dir)
    if args.layers is not None:
        keep = {int(x) for x in args.layers.split(",")}
        layers = [l for l in layers if l in keep]
    if not layers:
        sys.exit(f"no layers in {act_dir}")
    print(f"[residual] act_dir={act_dir}")
    print(f"[residual] layers: {layers}")

    print(f"[residual] loading scene labels from {args.scenes_root}")
    labels = load_scene_labels(Path(args.scenes_root))
    print(f"[residual] got labels for {len(labels)} scenes")

    # Use layer 0 to pick the scene set
    df0, _ = load_layer(act_dir, layers[0])
    scenes = list(dict.fromkeys(df0["scene_id"].tolist()))
    if args.limit_scenes is not None:
        scenes = scenes[: args.limit_scenes]
    print(f"[residual] n_scenes: {len(scenes)}")

    # Prefer scenes with more objects for PCA examples (mirrors topology_option3)
    n_obj_by_scene = df0.groupby("scene_id")["object_id"].nunique().to_dict()
    chosen_examples = set(sorted(scenes, key=lambda s: (-n_obj_by_scene.get(s, 0), s))[: args.pca_example_scenes])

    rows_out: list[dict] = []
    pca_examples: dict[tuple[str, int], dict] = {}
    cond_dump: dict[str, dict] = {"color": {}, "shape": {}}

    for layer in layers:
        df, vecs = load_layer(act_dir, layer)
        df = df[df["scene_id"].isin(scenes)]
        if args.t_min > 0:
            df = df[df["frame_id"] >= args.t_min]

        df_lab = attach_labels(df, labels)
        if len(df_lab) == 0:
            print(f"[layer {layer:02d}] skipped: no labeled rows")
            continue

        # Build conditional means (using vecs indexed by the pre-reset df's vec_row)
        color_mean = compute_conditional_means(df_lab, vecs, "color")
        shape_mean = compute_conditional_means(df_lab, vecs, "shape")
        cond_dump["color"][str(layer)] = {k: v for k, v in color_mean.items()}
        cond_dump["shape"][str(layer)] = {k: v for k, v in shape_mean.items()}

        # Residualize: replace each row with h - emb(color) - emb(shape)
        df_reset, vecs_sub = _collect_vec_rows(df_lab, vecs)
        resid = residualize(df_reset, vecs_sub, color_mean, shape_mean)

        # Reuse topology_option3 machinery: pool per (scene, object) over frames,
        # build scene k-NN on true 3D, compute metrics vs. permutation null.
        pooled = pool_scene_reps(df_reset, resid)
        rng = np.random.default_rng(args.seed + layer)

        for sid in scenes:
            if sid not in pooled:
                continue
            H = pooled[sid]["H"]
            P = pooled[sid]["P"]
            sm: SceneMetrics = compute_scene_metrics(
                H, P, k=args.knn_k,
                n_permutations=args.n_permutations, rng=rng,
            )
            rows_out.append({
                "scene_id": sid,
                "layer": layer,
                "n_objects": H.shape[0],
                "rsa": sm.rsa,
                "energy": sm.energy,
                "energy_null_mean": sm.energy_null_mean,
                "energy_null_std": sm.energy_null_std,
                "energy_ratio": sm.energy / max(sm.energy_null_mean, 1e-12),
                "energy_z": (sm.energy - sm.energy_null_mean) / sm.energy_null_std,
                "knn_overlap": sm.knn_overlap,
                "knn_null_mean": sm.knn_null_mean,
                "knn_null_std": sm.knn_null_std,
                "knn_z": (sm.knn_overlap - sm.knn_null_mean) / sm.knn_null_std,
                "spectral_cos_1": sm.spectral_cos_1,
                "spectral_cos_2": sm.spectral_cos_2,
            })
            if sid in chosen_examples:
                Hc = H - H.mean(axis=0, keepdims=True)
                U, S, _ = np.linalg.svd(Hc, full_matrices=False)
                PCs = (U[:, :3] * S[:3]).astype(np.float32)
                pca_examples[(sid, layer)] = {
                    "pc": PCs,
                    "P": P,
                    "object_ids": np.array(pooled[sid]["object_ids"]),
                }

        layer_rows = [r for r in rows_out if r["layer"] == layer]
        print(f"[layer {layer:02d}] "
              f"RSA={np.mean([r['rsa'] for r in layer_rows]):.3f}  "
              f"E/Enull={np.mean([r['energy_ratio'] for r in layer_rows]):.3f}  "
              f"kNN={np.mean([r['knn_overlap'] for r in layer_rows]):.3f}  "
              f"cos1={np.nanmean([r['spectral_cos_1'] for r in layer_rows]):.3f}  "
              f"(n_scenes={len(layer_rows)}, n_colors={len(color_mean)}, n_shapes={len(shape_mean)})")

    # Write outputs
    metrics_df = pd.DataFrame(rows_out)
    metrics_df.to_parquet(out_dir / "layer_metrics.parquet")

    summary_rows = []
    for layer in layers:
        sub = metrics_df[metrics_df["layer"] == layer]
        if len(sub) == 0:
            continue
        summary_rows.append({
            "layer": layer,
            "n_scenes": len(sub),
            "rsa_mean": float(sub["rsa"].mean()),
            "rsa_stderr": float(sub["rsa"].std() / math.sqrt(len(sub))),
            "energy_ratio_mean": float(sub["energy_ratio"].mean()),
            "energy_ratio_stderr": float(sub["energy_ratio"].std() / math.sqrt(len(sub))),
            "energy_z_mean": float(sub["energy_z"].mean()),
            "knn_overlap_mean": float(sub["knn_overlap"].mean()),
            "knn_overlap_stderr": float(sub["knn_overlap"].std() / math.sqrt(len(sub))),
            "knn_z_mean": float(sub["knn_z"].mean()),
            "spectral_cos_1_mean": float(sub["spectral_cos_1"].mean()),
            "spectral_cos_2_mean": float(sub["spectral_cos_2"].mean()),
        })
    pd.DataFrame(summary_rows).to_parquet(out_dir / "summary.parquet")

    # PCA examples dump (matches topology_option3 format)
    if pca_examples:
        keys = list(pca_examples.keys())
        np.savez_compressed(
            out_dir / "pca_examples.npz",
            scene_layers=np.array([f"{k[0]}||{k[1]}" for k in keys]),
            **{f"pc_{i}": pca_examples[k]["pc"] for i, k in enumerate(keys)},
            **{f"P_{i}": pca_examples[k]["P"] for i, k in enumerate(keys)},
            **{f"oid_{i}": pca_examples[k]["object_ids"] for i, k in enumerate(keys)},
        )

    # Conditional means (for later inspection / visualization)
    flat = {}
    for kind in ["color", "shape"]:
        for layer_str, d in cond_dump[kind].items():
            for val, v in d.items():
                flat[f"{kind}_{val}_L{layer_str}"] = v
    if flat:
        np.savez_compressed(out_dir / "conditional_means.npz", **flat)

    (out_dir / "run_info.json").write_text(json.dumps({
        "activations": str(act_dir),
        "scenes_root": str(args.scenes_root),
        "knn_k": args.knn_k,
        "n_permutations": args.n_permutations,
        "n_scenes": len(scenes),
        "layers": layers,
        "method": "residualize by emb(color) + emb(shape) per layer (global conditional means)",
    }, indent=2))
    print(f"[residual] done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
