#!/usr/bin/env python
"""Detailed per-scene PCA grid with all layers + per-layer metrics annotated.

For each (scene, model, frame-count) combination this renders a grid of
subplots:
    - top-left: ground-truth BEV layout
    - remaining: PCA-top-2 at every layer, annotated with the scene-specific
      RSA and k-NN overlap values at that layer

Input:  a directory produced by topology_option3.py or topology_option3_residual.py
        (containing layer_metrics.parquet and pca_examples.npz)
Output: one PNG per chosen scene

Example:
    python scripts/visualize_topology_detailed.py \
        --metrics data/probes/topology_option3_residual/internvl3_8b_f16 \
        --out figures/topology_option3_residual/internvl3_8b_f16_detailed \
        --max-scenes 4 --cols 8
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def procrustes_align(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Translate + rotate/reflect + uniformly scale ``X`` to best match ``Y``.

    PCA is rotation/sign-ambiguous; without this step each panel is plotted
    in an arbitrary frame and the eye cannot judge whether the layout
    matches GT. With it, the residual visual mismatch after alignment
    *is* the geometric error.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    M = Yc.T @ Xc
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    denom = float((Xc ** 2).sum())
    s = float(S.sum() / denom) if denom > 1e-12 else 1.0
    return (Xc @ R.T) * s + Y.mean(axis=0, keepdims=True)


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


def _obj_color(oid: int) -> tuple[float, float, float]:
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    return palette[oid % 10][:3]


def plot_scene(
    sid: str,
    layer_to_pc: dict[int, dict],
    metrics_df: pd.DataFrame,
    out_path: Path,
    cols: int = 8,
    show_edges: bool = True,
) -> None:
    """Plot GT BEV + one panel per layer for a single scene."""
    layers = sorted(layer_to_pc.keys())
    P = layer_to_pc[layers[0]]["P"]
    oids = layer_to_pc[layers[0]]["object_ids"]
    edges = knn_edges(P, k=2) if show_edges else []

    n_panels = 1 + len(layers)
    rows = (n_panels + cols - 1) // cols
    # Layout: PCA grid (rows) on top, per-scene metric curve strip (1 row) underneath
    fig = plt.figure(figsize=(2.5 * cols, 2.7 * rows + 2.2))
    gs = fig.add_gridspec(
        rows + 1, cols,
        height_ratios=[1.0] * rows + [0.8],
        hspace=0.35, wspace=0.12,
    )

    # Ground truth BEV
    ax = fig.add_subplot(gs[0, 0])
    for (i, j) in edges:
        ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color="0.7", lw=0.8, zorder=0)
    for k, oid in enumerate(oids):
        ax.scatter(P[k, 0], P[k, 1], s=110, color=_obj_color(int(oid)), edgecolor="k", lw=0.6)
        ax.text(P[k, 0], P[k, 1], str(int(oid)), fontsize=7, ha="center", va="center")
    ax.set_title(f"{sid}\nGround truth BEV", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # One panel per layer with metric annotation
    scene_metrics = metrics_df[metrics_df["scene_id"] == sid].set_index("layer")

    for li, lyr in enumerate(layers):
        slot = li + 1
        r, c = divmod(slot, cols)
        ax = fig.add_subplot(gs[r, c])
        pc = layer_to_pc[lyr]["pc"]
        # Align each layer's PCA panel to the GT 2D positions via Procrustes
        # so the eye can directly compare layout shapes; alignment is a
        # similarity transform (rotation+reflection+scale+translate), so it
        # preserves all topology metrics.
        pc_aligned = procrustes_align(pc[:, :2], P[:, :2])
        for (a, b) in edges:
            ax.plot([pc_aligned[a, 0], pc_aligned[b, 0]],
                    [pc_aligned[a, 1], pc_aligned[b, 1]],
                    color="0.82", lw=0.8, zorder=0)
        for k, oid in enumerate(oids):
            ax.scatter(pc_aligned[k, 0], pc_aligned[k, 1], s=95,
                       color=_obj_color(int(oid)), edgecolor="k", lw=0.6)
            ax.text(pc_aligned[k, 0], pc_aligned[k, 1], str(int(oid)),
                    fontsize=6.5, ha="center", va="center")
        if lyr in scene_metrics.index:
            m = scene_metrics.loc[lyr]
            title = (
                f"L{int(lyr)}\n"
                f"RSA={m['rsa']:.2f}  kNN={m['knn_overlap']:.2f}\n"
                f"E/E_null={m['energy_ratio']:.2f}"
            )
        else:
            title = f"L{int(lyr)}"
        ax.set_title(title, fontsize=8.2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    # Per-scene metric curves across all layers (bottom strip)
    layer_arr = np.asarray(sorted(scene_metrics.index))
    rsa_arr = scene_metrics.loc[layer_arr, "rsa"].to_numpy()
    kn_arr = scene_metrics.loc[layer_arr, "knn_overlap"].to_numpy()
    er_arr = scene_metrics.loc[layer_arr, "energy_ratio"].to_numpy()
    sc_arr = scene_metrics.loc[layer_arr, "spectral_cos_1"].to_numpy()

    ax_rsa = fig.add_subplot(gs[rows, :cols // 4 + 1])
    ax_rsa.plot(layer_arr, rsa_arr, marker="o", ms=3, lw=1, color="tab:blue")
    ax_rsa.set_title("per-scene RSA", fontsize=9)
    ax_rsa.set_xlabel("Layer", fontsize=8)
    ax_rsa.grid(alpha=0.3)
    ax_rsa.tick_params(axis="both", labelsize=8)

    ax_er = fig.add_subplot(gs[rows, cols // 4 + 1 : cols // 2 + 1])
    ax_er.plot(layer_arr, er_arr, marker="o", ms=3, lw=1, color="tab:orange")
    ax_er.axhline(1.0, color="k", ls=":", lw=0.6)
    ax_er.set_title("Dirichlet E/E_null (<1 better)", fontsize=9)
    ax_er.set_xlabel("Layer", fontsize=8)
    ax_er.grid(alpha=0.3)
    ax_er.tick_params(axis="both", labelsize=8)

    ax_kn = fig.add_subplot(gs[rows, cols // 2 + 1 : 3 * cols // 4 + 1])
    ax_kn.plot(layer_arr, kn_arr, marker="o", ms=3, lw=1, color="tab:green")
    ax_kn.set_title("k-NN overlap", fontsize=9)
    ax_kn.set_xlabel("Layer", fontsize=8)
    ax_kn.grid(alpha=0.3)
    ax_kn.tick_params(axis="both", labelsize=8)

    ax_sc = fig.add_subplot(gs[rows, 3 * cols // 4 + 1:])
    ax_sc.plot(layer_arr, sc_arr, marker="o", ms=3, lw=1, color="tab:red")
    ax_sc.set_title("spectral cos(PC1, z2)", fontsize=9)
    ax_sc.set_xlabel("Layer", fontsize=8)
    ax_sc.grid(alpha=0.3)
    ax_sc.tick_params(axis="both", labelsize=8)

    fig.suptitle(
        f"Detailed per-scene topology: {sid} (all {len(layers)} layers + metric curves)",
        fontsize=11, y=1.005,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True,
                    help="Dir with layer_metrics.parquet + pca_examples.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-scenes", type=int, default=4,
                    help="How many scenes to plot (picks those with most objects first)")
    ap.add_argument("--cols", type=int, default=8, help="Columns in the layer grid")
    ap.add_argument("--scenes", type=str, default=None,
                    help="Comma-separated scene_ids to plot (overrides --max-scenes ranking)")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_parquet(metrics_dir / "layer_metrics.parquet")
    npz = np.load(metrics_dir / "pca_examples.npz", allow_pickle=True)
    scene_layer_keys = [s.split("||") for s in npz["scene_layers"].tolist()]

    # Group PCA dumps by scene
    by_scene: dict[str, dict[int, dict]] = {}
    for i, (sid, lyr) in enumerate(scene_layer_keys):
        by_scene.setdefault(sid, {})[int(lyr)] = {
            "pc": npz[f"pc_{i}"],
            "P": npz[f"P_{i}"],
            "object_ids": npz[f"oid_{i}"],
        }

    # Decide which scenes to plot
    if args.scenes is not None:
        scene_list = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        # Rank by object count, break ties by scene_id
        ranked = sorted(
            by_scene.keys(),
            key=lambda s: (-int(by_scene[s][list(by_scene[s].keys())[0]]["P"].shape[0]), s),
        )
        scene_list = ranked[: args.max_scenes]

    print(f"Plotting {len(scene_list)} scenes with {len(metrics_df['layer'].unique())} layers each")
    for sid in scene_list:
        if sid not in by_scene:
            print(f"[skip] {sid} not in PCA examples")
            continue
        out_path = out_dir / f"detailed_pca_{sid}.png"
        plot_scene(sid, by_scene[sid], metrics_df, out_path, cols=args.cols)
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
