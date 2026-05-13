"""Dirichlet-energy ratio over decoder layers, averaged over scenes.

Two figures (circle8, grid3x3), each with three model lines.
Legend is placed BELOW the axes to avoid overlapping the curves.

Output: figures/fig_dirichlet_energy_layers_{circle8,grid3x3}.{pdf,png}
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/home/haoming/x-spatial-manual/data/probes/topology_option3_residual")
OUT  = Path("/home/haoming/x-spatial-manual/figures")

MODELS = [
    ("Qwen2.5-VL-7B", "qwen25vl_7b_f16_latter",  "#d62728"),
    ("InternVL3-8B",  "internvl3_8b_f16_latter", "#1f77b4"),
    ("LLaVA-OV-7B",   "llava_ov_7b_f16_latter",  "#2ca02c"),
]

TOPOLOGIES = ["circle8", "grid3x3"]


def load_layer_means(tag: str):
    df = pd.read_parquet(ROOT / tag / "layer_metrics.parquet")
    g = df.groupby("layer")["energy_ratio"].agg(["mean", "sem", "count"]).reset_index()
    return g


def make_plot(topology: str, out_path: Path):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.linewidth": 1.6,
    })

    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    for label, suffix, color in MODELS:
        tag = f"{topology}_{suffix}"
        if not (ROOT / tag).exists():
            print(f"[skip] {tag}")
            continue
        g = load_layer_means(tag)
        layers = g["layer"].values
        mean   = g["mean"].values
        sem    = g["sem"].values

        ax.fill_between(layers, mean - sem, mean + sem, color=color, alpha=0.18, linewidth=0)
        ax.plot(layers, mean, color=color, linewidth=3.0, label=label,
                marker="o", markersize=5, markeredgewidth=0)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(0.985, 0.965, "permutation null", transform=ax.transAxes,
            ha="right", va="top", fontsize=12, color="black", alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="black", linewidth=0.6, alpha=0.9))

    ax.set_xlabel("Decoder layer", fontsize=20, fontweight="bold", labelpad=10)
    ax.set_ylabel(r"Dirichlet energy / null", fontsize=20, fontweight="bold", labelpad=10)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(15)
        tick.set_fontweight("bold")
    ax.tick_params(axis="both", which="major", length=6, width=1.6, pad=6)
    ax.tick_params(axis="both", which="minor", length=3, width=1.0)

    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="-", linewidth=0.7, alpha=0.30)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend anchored OUTSIDE the axes, below the plot, in one row
    leg = ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=len(MODELS), fontsize=14, frameon=True, framealpha=0.95,
        edgecolor="black", borderpad=0.6, handlelength=2.4,
        columnspacing=1.6, handletextpad=0.6,
    )
    for text in leg.get_texts():
        text.set_fontweight("bold")

    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.30)
    plt.savefig(str(out_path).replace(".pdf", ".png"), dpi=240,
                bbox_inches="tight", pad_inches=0.30)
    plt.close(fig)
    print(f"saved {out_path}")


for topology in TOPOLOGIES:
    make_plot(topology, OUT / f"fig_dirichlet_energy_layers_{topology}.pdf")
