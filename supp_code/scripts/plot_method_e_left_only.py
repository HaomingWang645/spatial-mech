"""Standalone version of the left panel of fig_method_e_steering_L12.

Single-panel figure: Δ probe-readout(x) vs steering α, with v_axis (the
probe direction) and v_perp (length-matched null-space control). Bars are
SEM (n=15 per cell). Source data: data/steering/tier_c_free6dof_7b_L12_x.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/haoming/x-spatial-manual/data/steering/tier_c_free6dof_7b_L12_x")
OUT  = Path("/home/haoming/x-spatial-manual/figures")
OUT.mkdir(parents=True, exist_ok=True)

shifts = pd.read_parquet(ROOT / "steering_shifts.parquet")

agg_x = (shifts.groupby(["direction", "alpha"])
               .agg(mean=("dread_x", "mean"),
                    sem=("dread_x", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
                    n=("dread_x", "count"))
               .reset_index())

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.5})
fig, ax = plt.subplots(figsize=(7.0, 5.0))

for d, color, label, marker in [
    ("v_axis", "#d62728", r"$v_{\mathrm{axis}}$  (probe direction)", "o"),
    ("v_perp", "#7f7f7f", r"$v_{\perp}$  (null-space ctrl)",         "s"),
]:
    sub = agg_x[agg_x.direction == d].sort_values("alpha")
    ax.errorbar(sub.alpha, sub["mean"], yerr=sub["sem"],
                fmt=f"-{marker}", color=color, capsize=4, linewidth=2.6,
                markersize=9, markeredgewidth=0, label=label)

ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)

ax.set_xlabel(r"steering $\alpha$  (norm. $x$ units)", fontsize=18, fontweight="bold", labelpad=10)
ax.set_ylabel(r"$\Delta$ probe-readout($x$)", fontsize=18, fontweight="bold", labelpad=10)

for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontsize(14); t.set_fontweight("bold")
ax.tick_params(axis="both", which="major", length=6, width=1.6, pad=6)
ax.tick_params(axis="both", which="minor", length=3, width=1.0)
ax.minorticks_on()
ax.grid(True, which="major", linestyle="-", linewidth=0.7, alpha=0.30)
ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.20)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

leg = ax.legend(fontsize=13, loc="lower right", frameon=True, framealpha=0.95,
                edgecolor="black", borderpad=0.6, handlelength=2.4)
for t in leg.get_texts():
    t.set_fontweight("bold")

fig.tight_layout()
out_pdf = OUT / "fig_method_e_steering_L12_left.pdf"
out_png = OUT / "fig_method_e_steering_L12_left.png"
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.25)
fig.savefig(out_png, dpi=240, bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
print(f"saved {out_pdf}")
print(f"saved {out_png}")
