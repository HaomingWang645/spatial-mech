#!/usr/bin/env python
"""Head-specialization analysis.

Consumes ``<per_head>/per_head.parquet`` and computes:

  - Best-target-per-head: for each (layer, head), which target (depth /
    cam translation / cam rotation / individual components) does this head
    explain best? Answers "do heads specialize?".
  - Cosine similarity matrix of head R² profiles across targets — heads that
    rank similarly across depth/trans/rot are generalists; heads that rank
    well on one but low on others are specialists.
  - Layer-level ranking: how many distinct "good heads" per layer (count at
    R² ≥ threshold).

Outputs:
  <out>/specialization.parquet
  <out>/specialization.png          (specialization score + target-wise bars)
  <out>/head_r2_matrix.png          (targets × (layer, head) stack)
  <out>/top_heads.csv               top heads per target with layer info
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGETS = [
    ("depth_r2", "depth"),
    ("cam_r2_translation", "cam trans"),
    ("cam_r2_rotation", "cam rot"),
    ("cam_r2_tx", "tx"),
    ("cam_r2_ty", "ty"),
    ("cam_r2_tz", "tz"),
    ("cam_r2_rx", "rx"),
    ("cam_r2_ry", "ry"),
    ("cam_r2_rz", "rz"),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--per-head", required=True, help="dir with per_head.parquet")
    p.add_argument("--out", required=True)
    p.add_argument("--min-r2", type=float, default=0.1,
                   help="threshold for 'this head carries signal'")
    p.add_argument("--top-k", type=int, default=20,
                   help="top heads per target to report")
    args = p.parse_args()

    df = pd.read_parquet(Path(args.per_head) / "per_head.parquet")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cols = [c for c, _ in TARGETS if c in df.columns]
    labels = [lbl for c, lbl in TARGETS if c in df.columns]

    # Per-(layer, head) specialization: max R² across targets, and which.
    mat = df[cols].to_numpy()  # (N_rows, n_targets)
    best_idx = np.nanargmax(mat, axis=1)
    best_r2 = np.nanmax(mat, axis=1)
    df_sp = df[["layer", "head"]].copy()
    df_sp["best_target"] = [labels[i] for i in best_idx]
    df_sp["best_r2"] = best_r2
    # "Specialization score": gap between best and 2nd-best R² across targets.
    sorted_r2 = np.sort(mat, axis=1)
    df_sp["spec_gap"] = sorted_r2[:, -1] - sorted_r2[:, -2]
    df_sp.to_parquet(out / "specialization.parquet")

    # Top heads per target (CSV).
    rows = []
    for c, lbl in zip(cols, labels):
        sub = df.sort_values(c, ascending=False).head(args.top_k)
        for rank, (_, r) in enumerate(sub.iterrows(), start=1):
            rows.append({
                "target": lbl, "rank": int(rank),
                "layer": int(r["layer"]), "head": int(r["head"]),
                "r2": float(r[c]),
            })
    pd.DataFrame(rows).to_csv(out / "top_heads.csv", index=False)

    # ---- Figure 1: specialization summary ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    # Count heads above min_r2 per target per layer.
    layers = sorted(df.layer.unique())
    counts = {lbl: [] for lbl in labels}
    for L in layers:
        sub = df[df.layer == L]
        for c, lbl in zip(cols, labels):
            counts[lbl].append(int((sub[c] >= args.min_r2).sum()))
    for lbl in labels:
        ax.plot(layers, counts[lbl], "-o", lw=1.5, ms=4, label=lbl)
    ax.set_xlabel("LM decoder layer")
    ax.set_ylabel(f"# heads with R² ≥ {args.min_r2}")
    ax.set_title("Count of informative heads per (layer, target)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    # Distribution of specialization gap.
    ax.hist(df_sp["spec_gap"].dropna(), bins=40, color="C0", alpha=0.8)
    ax.set_xlabel("R²(best target) − R²(2nd-best target)")
    ax.set_ylabel("# (layer, head) cells")
    ax.set_title("Specialization gap per (layer, head)")
    ax.axvline(0.05, color="C3", lw=0.8, ls="--", label="0.05")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("Per-head specialization across targets", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "specialization.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: head × target R² heatmap (best-layer per head) ----
    # For each head (0..n_heads-1), show its best R² across layers, for each target.
    n_heads = int(df["head"].max()) + 1
    heat = np.full((len(cols), n_heads), np.nan)
    for ti, c in enumerate(cols):
        for h in range(n_heads):
            sub = df[df.head == h][c]
            if len(sub):
                heat[ti, h] = float(sub.max())
    fig, ax = plt.subplots(figsize=(max(10, n_heads * 0.3), 5))
    im = ax.imshow(heat, aspect="auto", origin="lower", cmap="viridis",
                   vmin=0.0, vmax=float(np.nanquantile(heat, 0.98)))
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels(range(n_heads), fontsize=7)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("head index")
    ax.set_title("Each head's best-layer R² on each target "
                 "(brighter = this head carries the signal at its best layer)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out / "head_r2_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"saved to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
