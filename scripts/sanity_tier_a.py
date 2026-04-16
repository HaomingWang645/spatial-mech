#!/usr/bin/env python
"""End-to-end framework self-test — no VLM required.

Renders a handful of Tier A scenes, fabricates hidden states whose first
three dimensions are the object centroids plus isotropic noise, fits a
linear probe with per-scene normalized coordinates as the target, and
asserts that the probe explains most of the variance. This validates
every wiring step — rendering, scene I/O, label normalization, splitting,
probe fitting, metrics — except the VLM forward pass itself.
"""
import _bootstrap  # noqa: F401

import argparse
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from spatial_subspace.labels import per_scene_normalized_coords
from spatial_subspace.probes import (
    fit_linear_probe,
    fit_pairwise_distance_probe,
    scene_split,
)
from spatial_subspace.render.tier_a import render_scene
from spatial_subspace.scene import Scene
from spatial_subspace.utils import ensure_dir, load_yaml, set_seed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/tier_a.yaml")
    p.add_argument("--n-scenes", type=int, default=50)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--noise-std", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-r2", type=float, default=0.9)
    args = p.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    cfg = load_yaml(args.config)

    with tempfile.TemporaryDirectory() as tmp:
        out = ensure_dir(Path(tmp) / "scenes")
        scenes: list[Scene] = [render_scene(cfg, rng, out) for _ in range(args.n_scenes)]

        # Fabricate hidden states: first three dims = per-scene normalized
        # coords (the same target the probe is trained against), padded with
        # isotropic noise in the remaining dims. This mirrors the cleanest
        # version of H1 — the model's "spatial subspace" is exactly the first
        # three dims of the feature, the orthogonal complement is noise.
        rng_np = np.random.default_rng(args.seed)
        rows = []
        vecs = []
        for scene in scenes:
            coords = np.array([o.centroid for o in scene.objects], dtype=np.float32)
            normed = per_scene_normalized_coords(coords)
            for i, obj in enumerate(scene.objects):
                vec = np.concatenate(
                    [normed[i], rng_np.normal(scale=args.noise_std, size=args.hidden_dim - 3)]
                ).astype(np.float32)
                rows.append(
                    {
                        "scene_id": scene.scene_id,
                        "object_id": obj.object_id,
                        "centroid_x": float(obj.centroid[0]),
                        "centroid_y": float(obj.centroid[1]),
                        "centroid_z": float(obj.centroid[2]),
                    }
                )
                vecs.append(vec)

        meta = pd.DataFrame(rows)
        pooled = np.stack(vecs)
        labels = np.zeros((len(meta), 3), dtype=np.float32)
        for _sid, idx in meta.groupby("scene_id").groups.items():
            i = np.asarray(list(idx))
            coords = meta.loc[i, ["centroid_x", "centroid_y", "centroid_z"]].to_numpy()
            labels[i] = per_scene_normalized_coords(coords)

        scenes_arr = meta["scene_id"].to_numpy()
        train_s, test_s = scene_split(scenes_arr, 0.8, args.seed)
        train_mask = np.isin(scenes_arr, train_s)
        test_mask = np.isin(scenes_arr, test_s)

        lin = fit_linear_probe(
            pooled[train_mask], labels[train_mask],
            pooled[test_mask], labels[test_mask],
        )
        print(f"linear probe R² = {lin.r2:.4f} (target ≥ {args.min_r2})")
        print(f"Procrustes error = {lin.extras['procrustes']:.4f}")

        _unique, scene_of = np.unique(scenes_arr, return_inverse=True)
        tr_ids = np.where(np.isin(_unique, train_s))[0]
        te_ids = np.where(np.isin(_unique, test_s))[0]
        pw = fit_pairwise_distance_probe(pooled, labels, scene_of, tr_ids, te_ids)
        print(f"pairwise probe R² = {pw.r2:.4f}, Spearman ρ = {pw.extras['spearman']:.4f}")

        assert lin.r2 >= args.min_r2, f"framework self-test failed: R² {lin.r2:.4f}"
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
