"""Circle-8 Dirichlet-energy figure with scene selection.

For each of {Qwen2.5-VL-7B, InternVL3-8B} we score per-scene energy_ratio
curves over layers by how "U-shaped" (indented) they are, then keep the
top-K scenes per model and plot the mean ± SEM of those.

Score = (baseline - nadir) + (rebound - nadir), where:
  baseline = mean of energy_ratio at layers 0..2
  nadir    = min of energy_ratio at layers 5..23 (mid-decoder)
  rebound  = mean of energy_ratio at the last 3 layers

Output: figures/fig_dirichlet_energy_layers_circle8_selected.{pdf,png}
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/haoming/x-spatial-manual/data/probes/topology_option3_residual")
OUT  = Path("/home/haoming/x-spatial-manual/figures")

MODELS = [
    ("Qwen2.5-VL-7B", "circle8_qwen25vl_7b_f16_latter",  "#d62728"),
    ("InternVL3-8B",  "circle8_internvl3_8b_f16_latter", "#1f77b4"),
]
TOP_K = 20  # scenes to keep per model (out of 100)


def select_indented(tag: str, top_k: int = TOP_K):
    df = pd.read_parquet(ROOT / tag / "layer_metrics.parquet")
    pivot = df.pivot(index="scene_id", columns="layer", values="energy_ratio")
    pivot = pivot.sort_index(axis=1)
    layers = pivot.columns.values

    baseline = pivot.iloc[:, :3].mean(axis=1)
    nadir    = pivot.iloc[:, 5:24].min(axis=1)
    rebound  = pivot.iloc[:, -3:].mean(axis=1)
    score    = (baseline - nadir) + (rebound - nadir)

    selected = score.sort_values(ascending=False).head(top_k).index.tolist()
    sub = pivot.loc[selected]
    return layers, sub  # sub: (top_k, n_layers)


fig, ax = plt.subplots(figsize=(7.4, 5.0))
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.6})

for label, tag, color in MODELS:
    layers, sub = select_indented(tag)
    mean = sub.mean(axis=0).values
    sem  = sub.std(axis=0, ddof=1).values / np.sqrt(len(sub))

    # Faint individual curves for selected scenes
    for _, row in sub.iterrows():
        ax.plot(layers, row.values, color=color, linewidth=0.7,
                alpha=0.18, zorder=1)

    # Mean line + SEM band on top
    ax.fill_between(layers, mean - sem, mean + sem,
                    color=color, alpha=0.28, linewidth=0, zorder=2)
    ax.plot(layers, mean, color=color, linewidth=3.4, zorder=3,
            marker="o", markersize=6, markeredgewidth=0,
            label=f"{label}  (top-{TOP_K} of 100 scenes)")

# Reference: ratio = 1 (permutation null)
ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7, zorder=0)
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

leg = ax.legend(loc="lower center", fontsize=13, frameon=True, framealpha=0.95,
                edgecolor="black", borderpad=0.6, handlelength=2.4)
for text in leg.get_texts():
    text.set_fontweight("bold")

ax.set_title("Circle-8: scenes with the most pronounced cognitive-map dip", fontsize=14, pad=10)

plt.tight_layout()
out_pdf = OUT / "fig_dirichlet_energy_layers_circle8_selected.pdf"
out_png = OUT / "fig_dirichlet_energy_layers_circle8_selected.png"
plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.25)
plt.savefig(out_png, dpi=240, bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
print(f"saved {out_pdf}")
print(f"saved {out_png}")
