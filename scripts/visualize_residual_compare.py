#!/usr/bin/env python
"""Side-by-side visualization: raw vs. residualized (shape+color subtracted)
topology metrics, per model, same layer axis.
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


def plot_pair(
    raw_dir: Path, res_dir: Path, out_path: Path, title: str
) -> None:
    raw = pd.read_parquet(raw_dir / "summary.parquet")
    res = pd.read_parquet(res_dir / "summary.parquet")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharex=True)
    specs = [
        ("rsa_mean", "rsa_stderr", "RSA (Spearman dist)", "max"),
        ("energy_ratio_mean", "energy_ratio_stderr", "Dirichlet ratio\n(<1 better)", "min"),
        ("knn_overlap_mean", "knn_overlap_stderr", "k-NN overlap (k=2)", "max"),
        ("spectral_cos_1_mean", None, "|cos(PC1, z2)|", "max"),
    ]
    for ax, (col, err, label, _) in zip(axes, specs):
        Lr = raw["layer"].to_numpy()
        ax.errorbar(Lr, raw[col], yerr=raw[err] if err else None,
                    marker="o", ms=3, lw=1, label="raw", color="tab:gray", alpha=0.8)
        Ls = res["layer"].to_numpy()
        ax.errorbar(Ls, res[col], yerr=res[err] if err else None,
                    marker="o", ms=3, lw=1.5, label="residualized\n(shape+color subtracted)",
                    color="tab:red")
        ax.set_title(label)
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.3)
        if col == "energy_ratio_mean":
            ax.axhline(1.0, color="k", ls=":", lw=0.5)
    axes[0].set_ylabel("metric")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_multi_model(
    specs: list[tuple[str, Path, Path]], out_path: Path
) -> None:
    """specs: list of (label, raw_dir, res_dir). Plot one row per model, 4
    metric columns, each showing raw (dotted gray) and residualized (solid
    colored)."""
    fig, axes = plt.subplots(len(specs), 4, figsize=(14, 3 * len(specs)), sharex=True)
    if len(specs) == 1:
        axes = axes.reshape(1, 4)
    metric_specs = [
        ("rsa_mean", "RSA"),
        ("energy_ratio_mean", "Dirichlet ratio"),
        ("knn_overlap_mean", "k-NN overlap"),
        ("spectral_cos_1_mean", "|cos(PC1, z2)|"),
    ]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for r, (label, rdir, sdir) in enumerate(specs):
        raw = pd.read_parquet(rdir / "summary.parquet")
        res = pd.read_parquet(sdir / "summary.parquet")
        for c, (col, mname) in enumerate(metric_specs):
            ax = axes[r, c]
            ax.plot(raw["layer"], raw[col], marker="o", ms=2.5, lw=1,
                    label="raw", color="tab:gray", alpha=0.8, linestyle=":")
            ax.plot(res["layer"], res[col], marker="o", ms=2.5, lw=1.5,
                    label="residualized", color=colors[r % len(colors)])
            if r == 0:
                ax.set_title(mname)
            if c == 0:
                ax.set_ylabel(f"{label}\nmetric")
            ax.grid(alpha=0.3)
            if col == "energy_ratio_mean":
                ax.axhline(1.0, color="k", ls=":", lw=0.5)
            if r == len(specs) - 1:
                ax.set_xlabel("Layer")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Raw vs. residualized topology (shape + color subtracted per layer)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="data/probes/topology_option3")
    ap.add_argument("--res-root", default="data/probes/topology_option3_residual")
    ap.add_argument("--out-dir", default="figures/topology_option3_residual")
    ap.add_argument("--models", nargs="+",
                    default=["qwen25vl_7b_f16:Qwen2.5-VL-7B",
                             "llava_ov_7b_f16:LLaVA-OV-7B",
                             "internvl3_8b_f16:InternVL3-8B"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    raw_root = Path(args.raw_root)
    res_root = Path(args.res_root)

    # Per-model pair plots
    for spec in args.models:
        slug, label = spec.split(":", 1)
        rd = raw_root / slug
        sd = res_root / slug
        if not (rd / "summary.parquet").exists() or not (sd / "summary.parquet").exists():
            print(f"[skip] {slug} — missing summary")
            continue
        plot_pair(rd, sd, out_dir / f"pair_{slug}.png",
                  title=f"{label} — raw vs. residualized")

    # Multi-row stacked plot
    specs = []
    for spec in args.models:
        slug, label = spec.split(":", 1)
        rd = raw_root / slug
        sd = res_root / slug
        if (rd / "summary.parquet").exists() and (sd / "summary.parquet").exists():
            specs.append((label, rd, sd))
    if specs:
        plot_multi_model(specs, out_dir / "raw_vs_residualized_multi.png")

    print(f"[visualize_residual_compare] wrote to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
