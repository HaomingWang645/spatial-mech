#!/usr/bin/env python
"""Hero figure: GT BEV vs peak-layer PCA side-by-side, several scenes per pattern.

Purpose: with a patterned ground-truth layout (3x3 grid or circle), the PCA-top-2
of object representations at the peak layer should visually echo that pattern.
This figure makes that comparison obvious at a glance.
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _obj_color(oid: int):
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    return palette[oid % 10][:3]


def procrustes_align(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Return X aligned to Y by best translation + rotation/reflection + uniform
    scale (orthogonal Procrustes with scaling).

    Both ``X`` and ``Y`` are (n, 2). The returned array is a transformed copy
    of ``X`` whose mean and orientation match ``Y`` as closely as possible.
    PCA is rotation/sign-ambiguous, so without this step ``X`` is plotted in
    an arbitrary frame and the eye cannot tell whether the layout matches GT.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    # Best rotation+reflection: U Vᵀ from SVD of Yᵀ X
    M = Yc.T @ Xc
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt  # (2, 2) — orthogonal, det may be ±1 (allows reflection)
    # Optimal uniform scale that minimises ||sR Xc - Yc||²
    denom = float((Xc ** 2).sum())
    s = float(S.sum() / denom) if denom > 1e-12 else 1.0
    aligned = (Xc @ R.T) * s + Y.mean(axis=0, keepdims=True)
    return aligned


def knn_edges(P, k=2):
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


def _draw(ax, X, oids, edges, *, title=None):
    for (i, j) in edges:
        ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color="0.78", lw=0.9, zorder=0)
    for k, oid in enumerate(oids):
        ax.scatter(X[k, 0], X[k, 1], s=130, color=_obj_color(int(oid)), edgecolor="k", lw=0.6)
        ax.text(X[k, 0], X[k, 1], str(int(oid)), fontsize=7.5, ha="center", va="center")
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, nargs="+",
                    help="One or more probe directories (each used as a row of the figure)")
    ap.add_argument("--labels", required=True, nargs="+",
                    help="Row labels (one per --metrics dir)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-scenes", type=int, default=4,
                    help="Scenes per row")
    args = ap.parse_args()
    assert len(args.metrics) == len(args.labels), "labels and metrics must match"

    rows_data = []
    for mdir, label in zip(args.metrics, args.labels):
        mdir = Path(mdir)
        df = pd.read_parquet(mdir / "summary.parquet")
        peak_layer = int(df.iloc[df["rsa_mean"].idxmax()].layer)

        per_scene_df = pd.read_parquet(mdir / "layer_metrics.parquet")
        per_scene_df = per_scene_df[per_scene_df["layer"] == peak_layer]

        npz = np.load(mdir / "pca_examples.npz", allow_pickle=True)
        keys = [s.split("||") for s in npz["scene_layers"].tolist()]
        # Find the (sid, peak_layer) entries
        candidates = []
        for i, (sid, lyr) in enumerate(keys):
            if int(lyr) != peak_layer:
                continue
            row = per_scene_df[per_scene_df["scene_id"] == sid]
            if len(row) == 0:
                continue
            candidates.append((sid, i, float(row.iloc[0]["rsa"])))
        # Pick top-N by per-scene RSA so the hero shows the best matches
        candidates.sort(key=lambda x: -x[2])
        chosen = candidates[: args.n_scenes]
        scenes = [{
            "sid": sid,
            "rsa": rsa,
            "pc": npz[f"pc_{i}"],
            "P": npz[f"P_{i}"],
            "oid": npz[f"oid_{i}"],
        } for sid, i, rsa in chosen]
        rows_data.append({
            "label": label,
            "peak_layer": peak_layer,
            "rsa_mean": float(df.iloc[df["rsa_mean"].idxmax()]["rsa_mean"]),
            "knn_mean": float(df.iloc[df["rsa_mean"].idxmax()]["knn_overlap_mean"]),
            "scenes": scenes,
        })

    n_rows = len(rows_data)
    cols = 1 + 2 * args.n_scenes  # 1 label col + (GT, PCA) per scene
    # Build figure: each row has n_scenes pairs of (GT, PCA)
    fig, axes = plt.subplots(n_rows, cols, figsize=(2.0 * cols, 2.4 * n_rows),
                             gridspec_kw={"width_ratios": [0.22] + [1.0] * (cols - 1)})
    if n_rows == 1:
        axes = axes[None, :]

    for r, row in enumerate(rows_data):
        # Label cell
        lab_ax = axes[r, 0]
        lab_ax.text(0.5, 0.55,
                    f"{row['label']}\n" + r"$\mathit{peak\ L}$=" + f"{row['peak_layer']}\n" +
                    f"RSA={row['rsa_mean']:.2f}\nkNN={row['knn_mean']:.2f}",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    transform=lab_ax.transAxes)
        lab_ax.set_xticks([]); lab_ax.set_yticks([])
        for spine in lab_ax.spines.values(): spine.set_visible(False)

        for s_idx, sc in enumerate(row["scenes"]):
            P = sc["P"]
            pc = sc["pc"]
            oids = sc["oid"]
            edges = knn_edges(P, k=2)

            ax_gt = axes[r, 1 + 2 * s_idx]
            ax_pc = axes[r, 1 + 2 * s_idx + 1]
            t_gt = "GT BEV" if r == 0 else None
            t_pc = f"PCA L{row['peak_layer']} (Procrustes-aligned)\n(scene RSA={sc['rsa']:.2f})" if r == 0 else f"scene RSA={sc['rsa']:.2f}"
            _draw(ax_gt, P, oids, edges, title=t_gt)
            # Align PCA-top-2 to the GT 2D layout via translation + rotation/reflection + scale
            pc_aligned = procrustes_align(pc[:, :2], P[:, :2])
            _draw(ax_pc, pc_aligned, oids, edges, title=t_pc)

    fig.suptitle(
        "Patterned ground-truth layouts (top: 3x3 grid, bottom: circle of 8) — "
        "PCA of object reps at peak layer recovers the pattern",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
