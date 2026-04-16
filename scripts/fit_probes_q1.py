#!/usr/bin/env python
"""Fit Q1 probes on extracted activations.

Sweeps all available layers (or the subset passed via --layers) and reports
linear / PCA-linear / optional MLP and pairwise probes. Object vectors are
averaged across frames per (scene, object) to form the "object summary"
representation described in plan §4.

Output:
    <out>/q1_probes.parquet
    <out>/q1_probes.json
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spatial_subspace.labels import per_scene_normalized_coords
from spatial_subspace.probes import (
    fit_linear_probe,
    fit_mlp_probe,
    fit_pairwise_distance_probe,
    fit_pca_linear,
    scene_split,
)
from spatial_subspace.utils import ensure_dir, set_seed


def load_layer(activ_dir: Path, layer: int) -> tuple[pd.DataFrame, np.ndarray]:
    meta = pd.read_parquet(activ_dir / f"layer_{layer:02d}.parquet")
    vecs = np.load(activ_dir / f"layer_{layer:02d}.npy")
    return meta, vecs


def object_summary(
    meta: pd.DataFrame, vecs: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    summaries = []
    summary_vecs = []
    for (sid, oid), group in meta.groupby(["scene_id", "object_id"], sort=False):
        rows = group["vec_row"].to_numpy()
        summary_vecs.append(vecs[rows].mean(axis=0))
        first = group.iloc[0]
        summaries.append(
            {
                "scene_id": sid,
                "object_id": int(oid),
                "centroid_x": float(first["centroid_x"]),
                "centroid_y": float(first["centroid_y"]),
                "centroid_z": float(first["centroid_z"]),
            }
        )
    return pd.DataFrame(summaries), np.stack(summary_vecs)


def build_labels(meta: pd.DataFrame) -> np.ndarray:
    out = np.zeros((len(meta), 3), dtype=np.float32)
    coords = meta[["centroid_x", "centroid_y", "centroid_z"]].to_numpy()
    for _sid, idx in meta.groupby("scene_id").groups.items():
        i = np.asarray(list(idx))
        out[i] = per_scene_normalized_coords(coords[i])
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--activations", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--mlp", action="store_true")
    p.add_argument("--pairwise", action="store_true")
    p.add_argument("--pca-ks", type=str, default="2,4,8,16,32,64")
    args = p.parse_args()

    set_seed(args.seed)
    activ = Path(args.activations)
    out = ensure_dir(args.out)

    available = sorted(
        int(f.stem.split("_")[1]) for f in activ.glob("layer_*.parquet")
    )
    if not available:
        raise SystemExit(f"no layer_*.parquet files found under {activ}")
    layers = [int(x) for x in args.layers.split(",")] if args.layers else available

    pca_ks = [int(x) for x in args.pca_ks.split(",")]
    results: list[dict] = []

    for layer in layers:
        meta_raw, vecs_raw = load_layer(activ, layer)
        meta, pooled = object_summary(meta_raw, vecs_raw)
        labels = build_labels(meta)

        scenes_arr = meta["scene_id"].to_numpy()
        train_s, test_s = scene_split(scenes_arr, args.train_frac, args.seed)
        train_mask = np.isin(scenes_arr, train_s)
        test_mask = np.isin(scenes_arr, test_s)

        X, Xv = pooled[train_mask], pooled[test_mask]
        y, yv = labels[train_mask], labels[test_mask]

        row: dict = {
            "layer": int(layer),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        }

        linear = fit_linear_probe(X, y, Xv, yv)
        row["linear_r2"] = linear.r2
        row["linear_procrustes"] = linear.extras["procrustes"]

        for k in pca_ks:
            pres = fit_pca_linear(X, y, Xv, yv, k=k)
            row[f"pca{k}_r2"] = pres.r2

        if args.mlp:
            mlp = fit_mlp_probe(X, y, Xv, yv)
            row["mlp_r2"] = mlp.r2

        if args.pairwise:
            _unique, scene_of = np.unique(scenes_arr, return_inverse=True)
            train_ids = np.where(np.isin(_unique, train_s))[0]
            test_ids = np.where(np.isin(_unique, test_s))[0]
            pw = fit_pairwise_distance_probe(
                pooled, labels, scene_of, train_ids, test_ids
            )
            row["pairwise_r2"] = pw.r2
            row["pairwise_spearman"] = pw.extras["spearman"]

        results.append(row)
        print(json.dumps(row))

    pd.DataFrame(results).to_parquet(out / "q1_probes.parquet")
    (out / "q1_probes.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
