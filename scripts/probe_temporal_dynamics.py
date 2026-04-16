#!/usr/bin/env python
"""Tier B — does later in the video give a better spatial read?

Each temporal token t in Qwen2.5-VL's decoder-only LM can attend (causal mask)
to all earlier visual tokens. So a hidden state at temporal slot t has been
exposed to input frames 0..(2t+1) by the time it's produced. If the model
integrates spatial information across frames over time, the linear probe R²
should rise with t.

Two probes per (layer, t):

  per_t        — fit a fresh ridge on per-temporal-token rows in that t bin
                  (cleanest: how readable is the spatial code at exactly t)

  summary_to_t — fit one ridge on the per-(scene, object) summary
                  representation, evaluate on per-t test rows
                  (tests whether per-t reps converge to the summary as t grows)
"""
import _bootstrap  # noqa: F401

import argparse
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


def build_labels(meta: pd.DataFrame) -> np.ndarray:
    out = np.zeros((len(meta), 3), dtype=np.float32)
    coords = meta[["centroid_x", "centroid_y", "centroid_z"]].to_numpy()
    for _sid, idx in meta.groupby("scene_id").groups.items():
        i = np.asarray(list(idx))
        out[i] = per_scene_normalized_coords(coords[i])
    return out


def split_scenes(scene_ids: np.ndarray, train_frac: float, seed: int) -> tuple[set, set]:
    unique = np.array(sorted(set(scene_ids)))
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
    p.add_argument("--alpha", type=float, default=10.0,
                   help="Ridge alpha; bumped >1 because per-t bins are small")
    p.add_argument("--min-train", type=int, default=30)
    p.add_argument("--min-test", type=int, default=10)
    args = p.parse_args()

    activ = Path(args.activations)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    layers = sorted(int(f.stem.split("_")[1]) for f in activ.glob("layer_*.parquet"))
    if not layers:
        raise SystemExit(f"no layers under {activ}")

    meta0 = pd.read_parquet(activ / f"layer_{layers[0]:02d}.parquet")
    train_s, test_s = split_scenes(meta0.scene_id.to_numpy(), args.train_frac, args.seed)

    results: list[dict] = []
    for layer in layers:
        meta = pd.read_parquet(activ / f"layer_{layer:02d}.parquet")
        vecs = np.load(activ / f"layer_{layer:02d}.npy")
        labels = build_labels(meta)

        scene_arr = meta.scene_id.to_numpy()
        t_arr = meta.frame_id.to_numpy()
        is_tr = np.isin(scene_arr, list(train_s))
        is_te = np.isin(scene_arr, list(test_s))

        # --- Mode 1: per-t probe (fit fresh per bin) ---
        for t in sorted(np.unique(t_arr)):
            in_t = t_arr == t
            tr = is_tr & in_t
            te = is_te & in_t
            if tr.sum() < args.min_train or te.sum() < args.min_test:
                continue
            model = Ridge(alpha=args.alpha).fit(vecs[tr], labels[tr])
            pred = model.predict(vecs[te])
            results.append({
                "layer": int(layer),
                "t": int(t),
                "mode": "per_t",
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                "r2": float(r2_score(labels[te], pred, multioutput="uniform_average")),
                "procrustes": float(procrustes_error(pred, labels[te])),
            })

        # --- Mode 2: probe trained on summary, evaluated per-t ---
        summary_groups = meta.groupby(["scene_id", "object_id"], sort=False).agg(
            vec_row=("vec_row", list),
            centroid_x=("centroid_x", "first"),
            centroid_y=("centroid_y", "first"),
            centroid_z=("centroid_z", "first"),
        ).reset_index()
        summary_vecs = np.stack([vecs[r].mean(axis=0) for r in summary_groups["vec_row"]])
        summary_labels = build_labels(summary_groups)
        s_scenes = summary_groups.scene_id.to_numpy()
        s_tr = np.isin(s_scenes, list(train_s))
        if s_tr.sum() < args.min_train:
            continue
        shared = Ridge(alpha=args.alpha).fit(summary_vecs[s_tr], summary_labels[s_tr])

        for t in sorted(np.unique(t_arr)):
            in_t = t_arr == t
            te = is_te & in_t
            if te.sum() < args.min_test:
                continue
            pred = shared.predict(vecs[te])
            results.append({
                "layer": int(layer),
                "t": int(t),
                "mode": "summary_to_t",
                "n_train": int(s_tr.sum()),
                "n_test": int(te.sum()),
                "r2": float(r2_score(labels[te], pred, multioutput="uniform_average")),
                "procrustes": float(procrustes_error(pred, labels[te])),
            })

    df = pd.DataFrame(results)
    df.to_parquet(out / "temporal_dynamics.parquet")
    df.to_json(out / "temporal_dynamics.json", orient="records", indent=2)

    # ---- heatmaps ----
    for mode in df["mode"].unique():
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        pivot_r2 = sub.pivot(index="layer", columns="t", values="r2")
        pivot_pc = sub.pivot(index="layer", columns="t", values="procrustes")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ext = [
            pivot_r2.columns.min() - 0.5, pivot_r2.columns.max() + 0.5,
            pivot_r2.index.min() - 0.5, pivot_r2.index.max() + 0.5,
        ]
        im0 = axes[0].imshow(
            pivot_r2.values, aspect="auto", origin="lower", cmap="viridis",
            extent=ext, vmin=0.0,
        )
        axes[0].set_xlabel("Temporal token index t")
        axes[0].set_ylabel("LM decoder layer")
        axes[0].set_title(f"R² of linear probe   ({mode})")
        plt.colorbar(im0, ax=axes[0], label="R²")

        im1 = axes[1].imshow(
            pivot_pc.values, aspect="auto", origin="lower", cmap="viridis_r",
            extent=ext,
        )
        axes[1].set_xlabel("Temporal token index t")
        axes[1].set_title(f"Procrustes error  ({mode})")
        plt.colorbar(im1, ax=axes[1], label="normalized scene units")

        fig.suptitle(
            "Tier B — temporal dynamics within a single video forward "
            "(t=k integrates input frames 0..2k+1 via causal attention)",
            y=1.01,
        )
        fig.tight_layout()
        fig.savefig(out / f"temporal_dynamics_{mode}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ---- line plot of R²(t) at selected layers (per_t) ----
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sub = df[df["mode"] == "per_t"]
    for layer in [0, 4, 8, 14, 18, 22, 27]:
        s = sub[sub.layer == layer].sort_values("t")
        if len(s):
            ax.plot(s.t, s.r2, "-o", label=f"L{layer}")
    ax.set_xlabel("Temporal token index t  (causal: t has 'seen' input frames 0..2t+1)")
    ax.set_ylabel("R²  (per-t linear probe)")
    ax.set_title("Tier B — does later in the video give a better spatial read?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "temporal_dynamics_lines.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"saved to {out}/")
    print("  - temporal_dynamics.parquet / .json")
    print("  - temporal_dynamics_per_t.png")
    print("  - temporal_dynamics_summary_to_t.png")
    print("  - temporal_dynamics_lines.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
