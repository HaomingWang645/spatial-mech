#!/usr/bin/env python
"""Tier C cross-trajectory probe — the H2 test from plan §5.2.

The same 3D scene is rendered under multiple independent camera trajectories.
For H2 we want to test whether the linear probe trained on trajectory A still
recovers per-scene normalized 3D coordinates when evaluated on trajectory B
of the *same* scene. If yes, the spatial subspace is camera-trajectory
invariant — the probe is reading scene geometry, not view-specific features.

Three protocols are reported per layer:

  same_traj       — train on traj0 of train_scenes, test on traj0 of test_scenes
                     (standard Tier-B-style cross-scene probe; baseline for
                     comparison against the cross-trajectory variants)

  cross_traj      — train on traj0 of train_scenes, test on traj1 of train_scenes
                     (every base scene is in train; the test only changes the
                     trajectory for the same scene. Tests pure trajectory
                     invariance, holding scene identity constant.)

  cross_both      — train on traj0 of train_scenes, test on traj1 of test_scenes
                     (the strict generalization test: held-out scenes AND
                     held-out trajectory)
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from spatial_subspace.labels import per_scene_normalized_coords
from spatial_subspace.metrics import procrustes_error


def _split_scene_id(scene_id: str) -> tuple[str, int]:
    """'s_xxx_t1' -> ('s_xxx', 1).  Falls back to (whole_id, 0) if no '_t' suffix."""
    if "_t" in scene_id:
        base, _, tag = scene_id.rpartition("_t")
        try:
            return base, int(tag)
        except ValueError:
            return scene_id, 0
    return scene_id, 0


def _object_summary(meta: pd.DataFrame, vecs: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """One vector per (scene_id, object_id), averaging across temporal tokens."""
    summaries: list[dict] = []
    summary_vecs: list[np.ndarray] = []
    for (sid, oid), g in meta.groupby(["scene_id", "object_id"], sort=False):
        rows = g["vec_row"].to_numpy()
        summary_vecs.append(vecs[rows].mean(axis=0))
        first = g.iloc[0]
        summaries.append(
            {
                "scene_id": sid,
                "object_id": int(oid),
                "centroid_x": float(first["centroid_x"]),
                "centroid_y": float(first["centroid_y"]),
                "centroid_z": float(first["centroid_z"]),
            }
        )
    df = pd.DataFrame(summaries)
    base_traj = df["scene_id"].apply(_split_scene_id)
    df["base_scene"] = [bt[0] for bt in base_traj]
    df["traj"] = [bt[1] for bt in base_traj]
    return df, np.stack(summary_vecs)


def _build_labels(df: pd.DataFrame) -> np.ndarray:
    """Per-scene-trajectory normalized labels.

    Note: we normalize per (scene_id including trajectory tag), so the label
    geometry is identical across trajectories of the same base scene because
    the underlying object positions are the same. This is what we want.
    """
    out = np.zeros((len(df), 3), dtype=np.float32)
    coords = df[["centroid_x", "centroid_y", "centroid_z"]].to_numpy()
    for _sid, idx in df.groupby("scene_id").groups.items():
        i = np.asarray(list(idx))
        out[i] = per_scene_normalized_coords(coords[i])
    return out


def _fit_eval(X_tr, y_tr, X_te, y_te, alpha: float) -> dict:
    if len(X_tr) == 0 or len(X_te) == 0:
        return {"r2": float("nan"), "procrustes": float("nan"),
                "n_train": int(len(X_tr)), "n_test": int(len(X_te))}
    model = Ridge(alpha=alpha).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return {
        "r2": float(r2_score(y_te, pred, multioutput="uniform_average")),
        "procrustes": float(procrustes_error(pred, y_te)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
    }


def split_base_scenes(base_scenes: np.ndarray, train_frac: float, seed: int) -> tuple[set, set]:
    unique = np.array(sorted(set(base_scenes)))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    n_tr = int(train_frac * len(unique))
    return set(unique[order[:n_tr]]), set(unique[order[n_tr:]])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--activations", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--layers", type=str, default=None)
    args = p.parse_args()

    activ = Path(args.activations)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    available = sorted(int(f.stem.split("_")[1]) for f in activ.glob("layer_*.parquet"))
    layers = [int(x) for x in args.layers.split(",")] if args.layers else available

    base_scenes_master = None
    train_base, test_base = None, None
    results: list[dict] = []

    for layer in layers:
        meta = pd.read_parquet(activ / f"layer_{layer:02d}.parquet")
        vecs = np.load(activ / f"layer_{layer:02d}.npy")
        smeta, pooled = _object_summary(meta, vecs)
        labels = _build_labels(smeta)

        base = smeta["base_scene"].to_numpy()
        traj = smeta["traj"].to_numpy()

        if base_scenes_master is None:
            base_scenes_master = base
            train_base, test_base = split_base_scenes(base, args.train_frac, args.seed)

        is_train_scene = np.isin(base, list(train_base))
        is_test_scene = np.isin(base, list(test_base))
        is_t0 = traj == 0
        is_t1 = traj == 1

        for protocol, train_mask, test_mask in [
            ("same_traj",  is_train_scene & is_t0, is_test_scene  & is_t0),
            ("cross_traj", is_train_scene & is_t0, is_train_scene & is_t1),
            ("cross_both", is_train_scene & is_t0, is_test_scene  & is_t1),
        ]:
            res = _fit_eval(
                pooled[train_mask], labels[train_mask],
                pooled[test_mask],  labels[test_mask],
                alpha=args.alpha,
            )
            results.append({"layer": int(layer), "protocol": protocol, **res})

    df = pd.DataFrame(results)
    df.to_parquet(out / "cross_trajectory.parquet")
    df.to_json(out / "cross_trajectory.json", orient="records", indent=2)

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = {"same_traj": "C0", "cross_traj": "C2", "cross_both": "C3"}
    label_map = {
        "same_traj":  "same scene, same traj  (baseline)",
        "cross_traj": "same scenes, different traj  (H2)",
        "cross_both": "held-out scene + traj  (strict)",
    }
    for proto in ["same_traj", "cross_traj", "cross_both"]:
        sub = df[df.protocol == proto].sort_values("layer")
        axes[0].plot(sub.layer, sub.r2, "-o", color=palette[proto],
                     lw=2, ms=4, label=label_map[proto])
        axes[1].plot(sub.layer, sub.procrustes, "-o", color=palette[proto],
                     lw=2, ms=4, label=label_map[proto])

    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("R²")
    axes[0].set_title("Linear probe R² across layers"); axes[0].set_ylim(-0.1, 1.0)
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("normalized scene units")
    axes[1].set_title("Procrustes error across layers")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    fig.suptitle("Tier C — cross-trajectory probing (H2 test)", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "cross_trajectory.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Console summary at the best layer per protocol
    print(f"saved to {out}/")
    for proto in ["same_traj", "cross_traj", "cross_both"]:
        sub = df[df.protocol == proto]
        if sub.empty:
            continue
        best = sub.loc[sub.r2.idxmax()]
        print(f"  best {proto:>10}: layer {int(best.layer):>2}  "
              f"R²={best.r2:.4f}  proc={best.procrustes:.4f}  "
              f"n_train={int(best.n_train)}  n_test={int(best.n_test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
