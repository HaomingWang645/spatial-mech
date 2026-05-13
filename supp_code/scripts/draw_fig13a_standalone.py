#!/usr/bin/env python
"""Standalone, conference-ready redraw of fig13 panel (a).

Pretrained Qwen2.5-VL-7B L17 object-token activations are probed for color
(4-way), shape (8-way), x/z position (R^2), and 3D pairwise RSA. The figure
shows that color/shape are encoded far above chance while 3D position is
barely above chance — the motivating diagnosis behind the Dirichlet method.

Reads:  reports/dirichlet_train/probe_parity_quick.json  (n=4 seeds)
Writes: figures/fig13a_color_vs_position_standalone.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path("reports/dirichlet_train/probe_parity_quick.json")
OUT_PNG = Path("figures/fig13a_color_vs_position_standalone.png")
OUT_PDF = Path("figures/fig13a_color_vs_position_standalone.pdf")

# (display label, json key, chance level, bar color)
BARS = [
    ("color\n(4-way)",     "color_acc",  0.25,  "#D24E4E"),  # red
    ("shape\n(8-way)",     "shape_acc",  0.125, "#F08F2A"),  # orange
    ("x position\n(R$^2$)", "x_r2",       0.0,   "#3C7DC2"),  # blue
    ("z position\n(R$^2$)", "z_r2",       0.0,   "#3C7DC2"),
    ("3D pairwise\nRSA",    "pos3D_rsa",  0.0,   "#3C7DC2"),
]


def main():
    runs = json.load(open(DATA))["results"]
    means, stds = [], []
    for _, key, _, _ in BARS:
        vals = np.array([r[key] for r in runs], dtype=float)
        means.append(vals.mean())
        stds.append(vals.std())
    means = np.array(means)
    stds = np.array(stds)
    print("Bar values (mean ± std, n=4 seeds):")
    for (lab, key, _, _), m, s in zip(BARS, means, stds):
        lab1 = lab.replace("\n", " ")
        print(f"  {lab1:25s} {m:+.3f} ± {s:.3f}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 14,
    })

    fig, ax = plt.subplots(figsize=(9.5, 6.4), dpi=140)

    x = np.arange(len(BARS))
    colors = [b[3] for b in BARS]
    chances = np.array([b[2] for b in BARS])
    labels = [b[0] for b in BARS]

    # Bars with subtle edge
    bars = ax.bar(
        x, means, width=0.68,
        color=colors, edgecolor="#222222", linewidth=1.1,
        yerr=stds, error_kw=dict(ecolor="#181818", capsize=6, lw=1.4),
        zorder=3,
    )

    # Per-bar dotted chance line drawn ONLY across that bar's width
    half_w = 0.34
    for xi, c in zip(x, chances):
        ax.hlines(c, xi - half_w - 0.02, xi + half_w + 0.02,
                  colors="#222222", linestyles=":", linewidth=1.6, zorder=4)

    # Numeric value labels above each bar (or below if negative)
    for xi, m, s in zip(x, means, stds):
        if m >= 0:
            y = m + s + 0.035
            va = "bottom"
        else:
            y = m - s - 0.045
            va = "top"
        ax.text(xi, y, f"{m:+.2f}", ha="center", va=va,
                fontsize=16, fontweight="bold", color="#101010", zorder=5)

    # Reference baseline (zero) line
    ax.axhline(0.0, color="#888888", linewidth=0.8, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15, fontweight="bold")
    ax.set_ylabel("Probe score   (accuracy  or  $R^2$)", fontsize=17)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([ 0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"])

    # ax.set_title(
    #     "Color and shape are strongly encoded; 3D position is near chance",
    #     fontsize=18, pad=18, weight="bold",
    # )
    # ax.text(
    #     0.5, 1.012,
    #     "Qwen2.5-VL-7B,  layer 17 object-token residual stream  (n=4 seeds; dotted = chance / 0)",
    #     transform=ax.transAxes, ha="center", va="bottom",
    #     fontsize=13, color="#444444",
    # )

    # Light grid behind bars
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.6, zorder=1)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(axis="both", colors="#333333", length=4)

    # Legend-like annotation: the chance line meaning
    # ax.text(
    #     0.985, 0.71,
    #     "$\\cdots\\cdots$  chance level\n(1/$k$ for accuracy, 0 for $R^2$/RSA)",
    #     transform=ax.transAxes, ha="right", va="top",
    #     fontsize=12, color="#333333",
    #     bbox=dict(boxstyle="round,pad=0.35",
    #               facecolor="white", edgecolor="#BBBBBB", linewidth=0.8),
    # )

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
