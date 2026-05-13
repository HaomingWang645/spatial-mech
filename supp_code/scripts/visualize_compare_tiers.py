#!/usr/bin/env python
"""Overlay multiple tiers' Q1 probe results on the same axes.

Useful for the headline cross-tier comparison: same model, same probes,
different stimulus tier (Tier A single BEV vs Tier B fragmented BEV video).
The comparison is what tells us whether the spatial subspace shifts location
in the model when the model is forced to integrate across views.
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _load(path: Path) -> pd.DataFrame:
    return pd.DataFrame(json.load(open(path / "q1_probes.json"))).sort_values("layer")


def _effective_rank(df: pd.DataFrame, target: float = 0.95) -> list[int]:
    pca_cols = sorted(
        (int(c.replace("pca", "").replace("_r2", "")), c)
        for c in df.columns
        if c.startswith("pca") and c.endswith("_r2")
    )
    out = []
    for _, row in df.iterrows():
        thr = target * row["linear_r2"]
        chosen = pca_cols[-1][0]
        for k, col in pca_cols:
            if row[col] >= thr:
                chosen = k
                break
        out.append(chosen)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", action="append", required=True,
                   help="LABEL=path/to/probes_dir, can repeat")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="Tier comparison — Qwen2.5-VL-7B Q1 probes")
    args = p.parse_args()

    items = []
    for spec in args.tier:
        label, path = spec.split("=", 1)
        items.append((label, _load(Path(path))))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    colors = ["C0", "C3", "C2", "C1", "C4"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Panel 1: full linear R²
    ax = axes[0, 0]
    for (label, df), c in zip(items, colors):
        ax.plot(df["layer"], df["linear_r2"], "-o", color=c, lw=2, ms=4, label=label)
    ax.set_xlabel("Layer")
    ax.set_ylabel("R²")
    ax.set_title("Full linear probe R² across layers")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 2: PCA-16 R² (regularized read-out)
    ax = axes[0, 1]
    for (label, df), c in zip(items, colors):
        if "pca16_r2" in df.columns:
            ax.plot(df["layer"], df["pca16_r2"], "-o", color=c, lw=2, ms=4, label=label)
    ax.set_xlabel("Layer")
    ax.set_ylabel("R²")
    ax.set_title("PCA-16 → linear R²  (regularized; smaller-sample-friendly)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 3: Procrustes error
    ax = axes[1, 0]
    for (label, df), c in zip(items, colors):
        ax.plot(df["layer"], df["linear_procrustes"], "-o", color=c, lw=2, ms=4, label=label)
    ax.set_xlabel("Layer")
    ax.set_ylabel("normalized scene units")
    ax.set_title("Procrustes-aligned reconstruction error")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 4: Pairwise distance Spearman
    ax = axes[1, 1]
    for (label, df), c in zip(items, colors):
        if "pairwise_spearman" in df.columns:
            ax.plot(df["layer"], df["pairwise_spearman"], "-o", color=c, lw=2, ms=4, label=label)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Pairwise distance probe Spearman")
    ax.set_ylim(0.0, 0.6)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(args.title, y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
