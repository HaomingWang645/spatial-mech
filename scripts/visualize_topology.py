#!/usr/bin/env python
"""Visualize topology_option3 results (paper-figure analogs).

Produces:
    per_scene_pca_grid_<scene>.png  — Fig. 9 analog: PCA-top-2 at several layers
                                       for one scene, overlay = true 3D kNN edges
    layer_curves.png                 — Fig. 4 analog: metric vs. layer
    scene_examples.png               — Fig. 1 analog: true BEV vs. rep PCA, 6 scenes
    model_compare.png                — bar chart of best metric across models
    spectral_heatmap.png             — Table 2 analog

Typical invocation (single model/extraction):
    python scripts/visualize_topology.py \
        --metrics data/probes/topology_option3/qwen25vl_7b_f16 \
        --scenes-root data/tier_c_free6dof \
        --out figures/topology_option3/qwen25vl_7b_f16

For cross-model comparison:
    python scripts/visualize_topology.py --compare \
        data/probes/topology_option3/qwen25vl_7b_f16:Qwen2.5-VL-7B \
        data/probes/topology_option3/llava_ov_7b_f16:LLaVA-OV-7B \
        data/probes/topology_option3/internvl3_8b_f16:InternVL3-8B \
        --out figures/topology_option3/compare_f16
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-scene PCA grid — Fig. 9 analog
# ---------------------------------------------------------------------------


def knn_edges(P: np.ndarray, k: int = 2) -> list[tuple[int, int]]:
    n = P.shape[0]
    k = min(k, n - 1)
    d = np.linalg.norm(P[:, None] - P[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :k]
    edges = set()
    for i in range(n):
        for j in nn[i]:
            edges.add((min(i, int(j)), max(i, int(j))))
    return sorted(edges)


def _scene_color(oid: int) -> tuple[float, float, float]:
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    return palette[oid % 10][:3]


def plot_per_scene_pca_grid(
    npz_path: Path,
    out_dir: Path,
    layers_to_show: list[int] | None = None,
    max_scenes: int = 6,
) -> None:
    data = np.load(npz_path, allow_pickle=True)
    scene_layers = [s.split("||") for s in data["scene_layers"].tolist()]
    idx_by_scene: dict[str, list[int]] = {}
    for i, (sid, layer) in enumerate(scene_layers):
        idx_by_scene.setdefault(sid, []).append(i)

    scenes = list(idx_by_scene.keys())[:max_scenes]
    for sid in scenes:
        idxs = idx_by_scene[sid]
        layer_nums = [int(scene_layers[i][1]) for i in idxs]
        order = np.argsort(layer_nums)
        idxs = [idxs[o] for o in order]
        layer_nums = [layer_nums[o] for o in order]

        if layers_to_show is not None:
            keep = [ii for ii, lyr in zip(idxs, layer_nums) if lyr in layers_to_show]
            keep_layers = [lyr for lyr in layer_nums if lyr in layers_to_show]
        else:
            # Pick 6 evenly spaced
            if len(idxs) > 6:
                sel = np.linspace(0, len(idxs) - 1, 6).astype(int)
                keep = [idxs[i] for i in sel]
                keep_layers = [layer_nums[i] for i in sel]
            else:
                keep = idxs
                keep_layers = layer_nums

        n_layers = len(keep)
        cols = n_layers + 1  # +1 for BEV panel
        fig, axes = plt.subplots(1, cols, figsize=(2.3 * cols, 2.5))
        if cols == 1:
            axes = [axes]

        # BEV ground truth
        P = data[f"P_{keep[0]}"]
        oids = data[f"oid_{keep[0]}"]
        edges = knn_edges(P, k=2)
        ax = axes[0]
        for (i, j) in edges:
            ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color="0.7", lw=0.8, zorder=0)
        for k, oid in enumerate(oids):
            ax.scatter(P[k, 0], P[k, 1], s=70, color=_scene_color(int(oid)), edgecolor="k", lw=0.5)
            ax.text(P[k, 0], P[k, 1], str(int(oid)), fontsize=6, ha="center", va="center")
        ax.set_title(f"{sid}\nGround truth BEV", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        for i, (ii, lyr) in enumerate(zip(keep, keep_layers)):
            pc = data[f"pc_{ii}"]
            ax = axes[i + 1]
            for (a, b) in edges:
                ax.plot([pc[a, 0], pc[b, 0]], [pc[a, 1], pc[b, 1]], color="0.8", lw=0.8, zorder=0)
            for k, oid in enumerate(oids):
                ax.scatter(pc[k, 0], pc[k, 1], s=70, color=_scene_color(int(oid)), edgecolor="k", lw=0.5)
                ax.text(pc[k, 0], pc[k, 1], str(int(oid)), fontsize=6, ha="center", va="center")
            ax.set_title(f"Layer {lyr}", fontsize=8)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"Scene {sid}: true BEV vs. rep PCA across layers", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"per_scene_pca_{sid}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Layer curves — Fig. 4 analog
# ---------------------------------------------------------------------------


def plot_layer_curves(summary: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharex=True)

    L = summary["layer"].to_numpy()
    # RSA
    axes[0].errorbar(L, summary["rsa_mean"], yerr=summary["rsa_stderr"], marker="o", ms=3, lw=1)
    axes[0].axhline(0.0, color="k", ls=":", lw=0.5)
    axes[0].set_title("RSA (Spearman dist)")
    axes[0].set_ylabel("metric")

    # Dirichlet energy ratio
    axes[1].errorbar(L, summary["energy_ratio_mean"], yerr=summary["energy_ratio_stderr"],
                     marker="o", ms=3, lw=1, color="tab:orange")
    axes[1].axhline(1.0, color="k", ls=":", lw=0.5)
    axes[1].set_title("Dirichlet energy ratio\n(<1 better)")

    # kNN overlap
    axes[2].errorbar(L, summary["knn_overlap_mean"], yerr=summary["knn_overlap_stderr"],
                     marker="o", ms=3, lw=1, color="tab:green")
    axes[2].set_title("k-NN overlap (k=2)")

    # Spectral cos
    axes[3].plot(L, summary["spectral_cos_1_mean"], marker="o", ms=3, lw=1, label="|cos(PC1, z2)|")
    axes[3].plot(L, summary["spectral_cos_2_mean"], marker="o", ms=3, lw=1, label="|cos(PC2, z3)|")
    axes[3].set_title("Spectral cosine")
    axes[3].legend(fontsize=7)

    for ax in axes:
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------


def plot_model_compare(specs: list[tuple[str, Path]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharex=True)
    for label, mdir in specs:
        summ = pd.read_parquet(mdir / "summary.parquet")
        L = summ["layer"].to_numpy()
        axes[0].plot(L, summ["rsa_mean"], marker="o", ms=2.5, lw=1, label=label)
        axes[1].plot(L, summ["energy_ratio_mean"], marker="o", ms=2.5, lw=1, label=label)
        axes[2].plot(L, summ["knn_overlap_mean"], marker="o", ms=2.5, lw=1, label=label)
        axes[3].plot(L, summ["spectral_cos_1_mean"], marker="o", ms=2.5, lw=1, label=label)
    axes[0].set_title("RSA")
    axes[0].axhline(0, color="k", ls=":", lw=0.5)
    axes[1].set_title("Dirichlet ratio")
    axes[1].axhline(1.0, color="k", ls=":", lw=0.5)
    axes[2].set_title("k-NN overlap")
    axes[3].set_title("|cos(PC1, z2)|")
    for ax in axes:
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Cross-model comparison: topology metrics vs. layer", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Frame-count emergence
# ---------------------------------------------------------------------------


def plot_frame_sweep(specs: list[tuple[str, Path]], out_path: Path, title: str) -> None:
    """specs: list of (frame_count_label, metrics_dir). Plots best-layer metric vs. N_frames."""
    rows = []
    for label, mdir in specs:
        summ = pd.read_parquet(mdir / "summary.parquet")
        best_rsa = float(summ["rsa_mean"].max())
        best_er = float(summ["energy_ratio_mean"].min())
        best_knn = float(summ["knn_overlap_mean"].max())
        best_cos = float(summ["spectral_cos_1_mean"].max())
        rows.append({
            "N_frames": label,
            "rsa_best": best_rsa,
            "energy_ratio_best": best_er,
            "knn_best": best_knn,
            "cos_best": best_cos,
        })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    axes[0].plot(df["N_frames"], df["rsa_best"], marker="o", lw=1)
    axes[0].set_title("Best-layer RSA")
    axes[0].set_ylabel("metric")
    axes[1].plot(df["N_frames"], df["energy_ratio_best"], marker="o", lw=1, color="tab:orange")
    axes[1].set_title("Best-layer Dirichlet ratio\n(lower = better)")
    axes[2].plot(df["N_frames"], df["knn_best"], marker="o", lw=1, color="tab:green")
    axes[2].set_title("Best-layer k-NN overlap")
    axes[3].plot(df["N_frames"], df["cos_best"], marker="o", lw=1, color="tab:red")
    axes[3].set_title("Best-layer spectral cos")
    for ax in axes:
        ax.set_xlabel("N_frames (context length)")
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default=None, help="Single-metrics-dir mode")
    p.add_argument("--out", required=True)
    p.add_argument("--compare", nargs="+", default=None,
                   help="Compare mode: list of metrics_dir:label")
    p.add_argument("--frame-sweep", nargs="+", default=None,
                   help="Frame sweep mode: list of N_frames:metrics_dir")
    p.add_argument("--frame-sweep-title", default="Frame count emergence")
    p.add_argument("--title", default="Topology metrics (Option 3)")
    p.add_argument("--pca-layers", type=str, default=None,
                   help="comma-separated layers to show in per-scene PCA grid")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pca_layers = [int(x) for x in args.pca_layers.split(",")] if args.pca_layers else None

    if args.metrics is not None:
        mdir = Path(args.metrics)
        summ = pd.read_parquet(mdir / "summary.parquet")
        plot_layer_curves(summ, out_dir / "layer_curves.png", title=args.title)
        npz_path = mdir / "pca_examples.npz"
        if npz_path.exists():
            plot_per_scene_pca_grid(npz_path, out_dir, layers_to_show=pca_layers)

    if args.compare is not None:
        specs = []
        for spec in args.compare:
            if ":" in spec:
                mdir, label = spec.rsplit(":", 1)
            else:
                mdir, label = spec, Path(spec).name
            specs.append((label, Path(mdir)))
        plot_model_compare(specs, out_dir / "model_compare.png")

    if args.frame_sweep is not None:
        specs = []
        for spec in args.frame_sweep:
            label, mdir = spec.split(":", 1)
            specs.append((label, Path(mdir)))
        plot_frame_sweep(specs, out_dir / "frame_sweep.png", title=args.frame_sweep_title)

    print(f"[visualize_topology] wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
