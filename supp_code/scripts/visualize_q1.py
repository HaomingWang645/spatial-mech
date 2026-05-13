#!/usr/bin/env python
"""Visualize Q1 probe results.

Produces two figures:

  q1_layer_dynamics.png       — linear/PCA-k R², Procrustes error, pairwise
                                Spearman, and effective rank, all vs layer.

  q1_reconstruction_examples.png — for the chosen layer, a grid of test
                                    scenes showing ground-truth BEV layout
                                    overlaid with Procrustes-aligned probe
                                    predictions. Visual sanity check that
                                    the linear probe is recovering geometry,
                                    not just average position.
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

from spatial_subspace.labels import per_scene_normalized_coords
from spatial_subspace.metrics import procrustes_align
from spatial_subspace.probes import fit_linear_probe, scene_split


def _effective_rank_per_layer(df: pd.DataFrame, target_fraction: float = 0.95) -> list[int]:
    pca_cols = sorted(
        (int(c.replace("pca", "").replace("_r2", "")), c)
        for c in df.columns
        if c.startswith("pca") and c.endswith("_r2")
    )
    out = []
    for _, row in df.iterrows():
        target = target_fraction * row["linear_r2"]
        chosen = pca_cols[-1][0]
        for k, col in pca_cols:
            if row[col] >= target:
                chosen = k
                break
        out.append(chosen)
    return out


def plot_layer_dynamics(df: pd.DataFrame, out_path: Path, title: str) -> None:
    df = df.sort_values("layer")
    layers = df["layer"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Panel 1: probe R² (full + PCA-k)
    ax = axes[0, 0]
    ax.plot(layers, df["linear_r2"], "k-o", lw=2, ms=4, label="full linear")
    for k in [2, 8, 16, 32]:
        col = f"pca{k}_r2"
        if col in df.columns:
            ax.plot(layers, df[col], "--", lw=1.4, alpha=0.85, label=f"PCA-{k}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("R²")
    ax.set_title("Linear probe R² across layers")
    ax.set_ylim(-0.05, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    # Panel 2: Procrustes error
    ax = axes[0, 1]
    ax.plot(layers, df["linear_procrustes"], "C3-o", lw=2, ms=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("error (normalized scene units)")
    ax.set_title("Procrustes-aligned reconstruction error")
    ax.grid(alpha=0.3)

    # Panel 3: Pairwise distance Spearman
    ax = axes[1, 0]
    if "pairwise_spearman" in df.columns:
        ax.plot(layers, df["pairwise_spearman"], "C2-o", lw=2, ms=4)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Spearman ρ")
        ax.set_title("Pairwise distance probe (concat features)")
        ax.set_ylim(0, max(0.5, df["pairwise_spearman"].max() * 1.1))
        ax.grid(alpha=0.3)

    # Panel 4: Effective rank
    ax = axes[1, 1]
    eff = _effective_rank_per_layer(df)
    ax.plot(layers, eff, "C0-o", lw=2, ms=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("k* (smallest PCA dim ≥ 95% of full R²)")
    ax.set_title("Effective spatial-subspace dimension")
    ax.set_yscale("log", base=2)
    ax.set_yticks([2, 4, 8, 16, 32, 64])
    ax.set_yticklabels(["2", "4", "8", "16", "32", "64"])
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(title, y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _object_summary(meta: pd.DataFrame, vecs: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    summaries: list[dict] = []
    summary_vecs: list[np.ndarray] = []
    for (sid, oid), g in meta.groupby(["scene_id", "object_id"], sort=False):
        rows_idx = g["vec_row"].to_numpy()
        summary_vecs.append(vecs[rows_idx].mean(axis=0))
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
    return pd.DataFrame(summaries), np.stack(summary_vecs)


def plot_reconstruction_examples(
    activations_dir: Path,
    out_path: Path,
    layer: int,
    n_scenes: int,
    train_frac: float,
    seed: int,
) -> None:
    meta = pd.read_parquet(activations_dir / f"layer_{layer:02d}.parquet")
    vecs = np.load(activations_dir / f"layer_{layer:02d}.npy")
    smeta, pooled = _object_summary(meta, vecs)

    coords_raw = smeta[["centroid_x", "centroid_y", "centroid_z"]].to_numpy()
    labels = np.zeros((len(smeta), 3), dtype=np.float32)
    for _sid, idx in smeta.groupby("scene_id").groups.items():
        i = np.asarray(list(idx))
        labels[i] = per_scene_normalized_coords(coords_raw[i])

    scenes_arr = smeta["scene_id"].to_numpy()
    train_s, test_s = scene_split(scenes_arr, train_frac, seed)
    train_mask = np.isin(scenes_arr, train_s)
    test_mask = np.isin(scenes_arr, test_s)

    res = fit_linear_probe(
        pooled[train_mask], labels[train_mask],
        pooled[test_mask], labels[test_mask],
    )
    pred_test = res.extras["pred"]
    test_scene_ids = scenes_arr[test_mask]
    test_labels = labels[test_mask]

    rng = np.random.default_rng(seed)
    unique_test = np.unique(test_scene_ids)
    rich_scenes = [s for s in unique_test if (test_scene_ids == s).sum() >= 4]
    pool = rich_scenes if len(rich_scenes) >= n_scenes else list(unique_test)
    chosen = rng.choice(pool, size=min(n_scenes, len(pool)), replace=False)

    cols = 4
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_2d(axes).flatten()

    palette = plt.cm.tab10(np.arange(10))

    for ax_idx, sid in enumerate(chosen):
        ax = axes[ax_idx]
        idx = np.where(test_scene_ids == sid)[0]
        true = test_labels[idx]
        pred = pred_test[idx]
        aligned, *_ = procrustes_align(pred, true)

        for j in range(len(idx)):
            color = palette[j % 10]
            ax.scatter(true[j, 0], true[j, 1], c=[color], s=110, marker="o",
                       edgecolors="black", linewidths=0.6, zorder=3)
            ax.scatter(aligned[j, 0], aligned[j, 1], c=[color], s=90, marker="x",
                       linewidths=2.0, zorder=4)
            ax.plot([true[j, 0], aligned[j, 0]], [true[j, 1], aligned[j, 1]],
                    color=color, alpha=0.5, lw=0.9, zorder=2)

        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_aspect("equal")
        ax.set_title(sid, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    for j in range(len(chosen), len(axes)):
        axes[j].axis("off")

    # legend in the first axis
    axes[0].scatter([], [], c="gray", s=110, marker="o",
                    edgecolors="black", linewidths=0.6, label="true")
    axes[0].scatter([], [], c="gray", s=90, marker="x", linewidths=2.0, label="pred")
    axes[0].legend(fontsize=7, loc="upper right", framealpha=0.9)

    fig.suptitle(
        f"Tier A — predicted vs ground-truth BEV layout (layer {layer}, "
        f"R²={res.r2:.3f}, Procrustes={res.extras['procrustes']:.3f})",
        y=1.0, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--probes", required=True, help="probe results dir with q1_probes.json")
    p.add_argument("--activations", required=True, help="extraction dir with layer_NN.npy")
    p.add_argument("--out", required=True, help="figures output dir")
    p.add_argument("--example-layer", type=int, default=0)
    p.add_argument("--n-example-scenes", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--title", default="Tier A — Q1 layer dynamics (Qwen2.5-VL-7B, 1000 scenes)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = json.load(open(Path(args.probes) / "q1_probes.json"))
    df = pd.DataFrame(rows)

    plot_layer_dynamics(df, out / "q1_layer_dynamics.png", args.title)
    plot_reconstruction_examples(
        Path(args.activations),
        out / "q1_reconstruction_examples.png",
        layer=args.example_layer,
        n_scenes=args.n_example_scenes,
        train_frac=args.train_frac,
        seed=args.seed,
    )

    print(f"saved figures to {out}/")
    print("  - q1_layer_dynamics.png")
    print("  - q1_reconstruction_examples.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
