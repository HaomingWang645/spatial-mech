#!/usr/bin/env python
"""Train linear probes on object-token residual-stream features to test
whether the model encodes color, shape, or 3D position.

Compares baseline (λ=0) vs Dirichlet (λ=1) features on:
  - Color classification accuracy (categorical, n=8 colors)
  - Shape classification accuracy (categorical, n=3-5 shapes)
  - Position regression R² (continuous, 3D)

The hypothesis: the Dirichlet loss pushes the residual stream away from
encoding nuisance visual features (color, shape) and toward encoding
spatial structure (position). If true, we should see:
  baseline:  color/shape probe accuracy HIGH, position R² LOWER
  Dirichlet: color/shape probe accuracy LOW,  position R² HIGHER

Usage
-----
    python scripts/probe_color_shape_position.py \\
        --features-dir reports/probe_features \\
        --out reports/probe_features/_summary.json
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold


def probe_categorical(H: np.ndarray, y: np.ndarray, groups: np.ndarray, *, max_iter=5000):
    """5-fold scene-grouped cross-validation accuracy."""
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    accs = []
    for tr, te in gkf.split(H, y, groups):
        clf = LogisticRegression(max_iter=max_iter, multi_class="auto",
                                 C=1.0, solver="liblinear",
                                 random_state=0)
        # Need at least 2 classes in train
        if len(np.unique(y[tr])) < 2:
            continue
        clf.fit(H[tr], y[tr])
        accs.append(clf.score(H[te], y[te]))
    return float(np.mean(accs)) if accs else float("nan"), float(np.std(accs)) if len(accs) > 1 else 0.0


def probe_continuous(H: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """5-fold scene-grouped cross-validation R² for ridge regression."""
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    r2s = []
    for tr, te in gkf.split(H, y, groups):
        reg = Ridge(alpha=1.0)
        reg.fit(H[tr], y[tr])
        pred = reg.predict(H[te])
        ss_res = ((pred - y[te]) ** 2).sum()
        ss_tot = ((y[te] - y[te].mean(axis=0)) ** 2).sum()
        r2s.append(1.0 - ss_res / max(ss_tot, 1e-8))
    return float(np.mean(r2s)) if r2s else float("nan"), float(np.std(r2s)) if len(r2s) > 1 else 0.0


def scene_mean_centered_pos(coords: np.ndarray, scenes: np.ndarray) -> np.ndarray:
    """Subtract per-scene mean from each object's coords."""
    out = coords.copy()
    for s in np.unique(scenes):
        mask = scenes == s
        out[mask] = out[mask] - out[mask].mean(axis=0)
    return out


def rsa_position(H: np.ndarray, coords: np.ndarray, scenes: np.ndarray) -> float:
    """Mean within-scene RSA(D_H, D_X)."""
    rsas = []
    for s in np.unique(scenes):
        mask = scenes == s
        if mask.sum() < 4:
            continue
        H_s = H[mask]
        X_s = coords[mask]
        # Pairwise sq distances
        from scipy.spatial.distance import pdist
        D_H = pdist(H_s) ** 2
        D_X = pdist(X_s) ** 2
        if D_H.std() < 1e-8 or D_X.std() < 1e-8:
            continue
        r = np.corrcoef(D_H, D_X)[0, 1]
        rsas.append(r)
    return float(np.mean(rsas)) if rsas else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=Path, default=Path("reports/probe_features"))
    p.add_argument("--out", type=Path, default=Path("reports/probe_features/_summary.json"))
    args = p.parse_args()

    results = {}
    for f in sorted(args.features_dir.glob("*.npz")):
        if f.stem.startswith("_"): continue
        d = np.load(f, allow_pickle=True)
        H = d["H"]
        coords = d["coords"]
        colors = d["colors"]
        shapes = d["shapes"]
        scenes = d["scene_ids"]

        # Encode categorical labels
        from sklearn.preprocessing import LabelEncoder
        col_le = LabelEncoder().fit(colors)
        sh_le = LabelEncoder().fit(shapes)

        col_acc, col_std = probe_categorical(H, col_le.transform(colors), scenes)
        sh_acc, sh_std = probe_categorical(H, sh_le.transform(shapes), scenes)
        # Scene-relative position (subtract scene mean) -- absolute coords don't generalize across scenes
        coords_rel = scene_mean_centered_pos(coords, scenes)
        pos_r2_rel, _ = probe_continuous(H, coords_rel, scenes)
        # Direct RSA (within-scene) -- same metric used for Dirichlet ratio
        rsa_pos = rsa_position(H, coords, scenes)

        results[f.stem] = {
            "n_samples": len(H),
            "color_probe_acc": col_acc, "color_std": col_std,
            "shape_probe_acc": sh_acc, "shape_std": sh_std,
            "pos_r2_scene_rel": pos_r2_rel,
            "rsa_within_scene": rsa_pos,
            "n_colors": len(col_le.classes_),
            "n_shapes": len(sh_le.classes_),
            "n_scenes": len(np.unique(scenes)),
        }
        print(f"{f.stem}:")
        print(f"  n={len(H)} ({len(np.unique(scenes))} scenes, {len(col_le.classes_)} colors, {len(sh_le.classes_)} shapes)")
        print(f"  color probe acc: {col_acc:.4f}  ({(1.0-col_acc)*100:.1f}pp gap from perfect)")
        print(f"  shape probe acc: {sh_acc:.4f}")
        print(f"  scene-relative position joint R²: {pos_r2_rel:.4f}")
        print(f"  within-scene RSA: {rsa_pos:.4f}")

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
