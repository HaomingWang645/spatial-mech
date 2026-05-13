"""Dirichlet-energy ratio over decoder layers: raw vs. residualized.

Mirrors `plot_dirichlet_energy_layers.py` but compares raw and
residualized (shape+color subtracted) curves for each of the three VLMs
at N=16 frames.  Setting matches `tier_c_topology_option3.md` §5.1
("Raw vs. residualized, all three VLMs at N=16"): same 400 free6dof
scenes, both curves drawn from `topology_option3` (raw) and
`topology_option3_residual` (residualized).

Output: figures/fig_dirichlet_energy_layers_raw_vs_residualized.{pdf,png}
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

DATA_ROOT = Path("/home/haoming/x-spatial-manual/data/probes")
OUT       = Path("/home/haoming/x-spatial-manual/figures")

MODELS = [
    ("InternVL3-8B",  "internvl3_8b_f16", "#1f77b4"),
]


def load_layer_means(parquet: Path):
    df = pd.read_parquet(parquet)
    g = df.groupby("layer")["energy_ratio"].agg(
        ["mean", "sem", "count"]).reset_index()
    return g


def main():
    plt.rcParams.update({
        "font.family":   "DejaVu Sans",
        "axes.linewidth": 1.6,
    })

    fig, axes = plt.subplots(
        1, len(MODELS), figsize=(7.4 * len(MODELS), 5.4),
        sharey=True,
    )
    if len(MODELS) == 1:
        axes = [axes]

    for ax, (label, slug, color) in zip(axes, MODELS):
        raw_pq = DATA_ROOT / "topology_option3"          / slug / "layer_metrics.parquet"
        res_pq = DATA_ROOT / "topology_option3_residual" / slug / "layer_metrics.parquet"
        if not raw_pq.exists() or not res_pq.exists():
            print(f"[skip] {slug}: missing parquet")
            continue

        raw = load_layer_means(raw_pq)
        res = load_layer_means(res_pq)

        # raw: lighter/dashed, in the model's color
        ax.fill_between(raw["layer"], raw["mean"] - raw["sem"], raw["mean"] + raw["sem"],
                        color=color, alpha=0.10, linewidth=0)
        ax.plot(raw["layer"], raw["mean"], color=color, linewidth=2.4,
                linestyle="--", marker="o", markersize=5,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.4, label="before disentanglement", alpha=0.95)

        # residualized: solid, same color
        ax.fill_between(res["layer"], res["mean"] - res["sem"], res["mean"] + res["sem"],
                        color=color, alpha=0.22, linewidth=0)
        ax.plot(res["layer"], res["mean"], color=color, linewidth=3.0,
                marker="o", markersize=5, markeredgewidth=0,
                label="after disentanglement")

        # permutation null
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(0.985, 0.965, "permutation null", transform=ax.transAxes,
                ha="right", va="top", fontsize=11, color="black", alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="black", linewidth=0.6, alpha=0.9))

        ax.set_xlabel("Decoder layer", fontsize=18, fontweight="bold", labelpad=8)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(14)
            tick.set_fontweight("bold")
        ax.tick_params(axis="both", which="major", length=6, width=1.6, pad=6)
        ax.tick_params(axis="both", which="minor", length=3, width=1.0)
        ax.minorticks_on()
        ax.grid(True, which="major", linestyle="-", linewidth=0.7, alpha=0.30)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        leg = ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.16),
            ncol=2, fontsize=12, frameon=True, framealpha=0.95,
            edgecolor="black", borderpad=0.5, handlelength=2.4,
            columnspacing=1.4, handletextpad=0.6,
        )
        for text in leg.get_texts():
            text.set_fontweight("bold")

    axes[0].set_ylabel(r"Dirichlet energy / null",
                       fontsize=18, fontweight="bold", labelpad=10)

    out_pdf = OUT / "fig_dirichlet_energy_layers_raw_vs_residualized.pdf"
    out_png = OUT / "fig_dirichlet_energy_layers_raw_vs_residualized.png"
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.30)
    plt.savefig(out_png, dpi=240, bbox_inches="tight", pad_inches=0.30)
    plt.close(fig)
    print(f"saved {out_pdf}")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
