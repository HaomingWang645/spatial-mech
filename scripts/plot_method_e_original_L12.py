"""Causal verification figure from the original L12 probe-readout steering.

Two-panel figure:
  (a) Δ probe readout for the targeted x-axis vs steering α — v_axis is
      monotonic and signed, v_perp is null.
  (b) Per-axis decomposition at the largest α: bars showing that v_axis
      moves only the targeted x-coordinate, leaving y and z untouched, and
      that v_perp moves nothing.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/haoming/x-spatial-manual/data/steering/tier_c_free6dof_7b_L12_x")
OUT  = Path("/home/haoming/x-spatial-manual/figures")
OUT.mkdir(parents=True, exist_ok=True)

shifts = pd.read_parquet(ROOT / "steering_shifts.parquet")
# dread_x = post − pre on the probe-predicted x; dread_l2 = norm of full Δreadout

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.5})
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8),
                         gridspec_kw={"width_ratios": [1.05, 1.0]})

# ---- (a) Δ readout (x) vs α ----
ax = axes[0]
agg_x = (shifts.groupby(["direction", "alpha"])
               .agg(mean=("dread_x", "mean"),
                    sem=("dread_x", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
                    n=("dread_x", "count"))
               .reset_index())

for d, color, label, marker in [
    ("v_axis", "#d62728", r"$v_{\mathrm{axis}}$  (probe direction)", "o"),
    ("v_perp", "#7f7f7f", r"$v_{\perp}$  (null-space ctrl)", "s"),
]:
    sub = agg_x[agg_x.direction == d].sort_values("alpha")
    ax.errorbar(sub.alpha, sub["mean"], yerr=sub["sem"],
                fmt=f"-{marker}", color=color, capsize=4, linewidth=2.6,
                markersize=8, markeredgewidth=0, label=label)

ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
ax.grid(alpha=0.30)
ax.set_xlabel(r"steering $\alpha$  (norm. $x$ units)", fontsize=15, fontweight="bold")
ax.set_ylabel(r"$\Delta$ probe-readout($x$)", fontsize=15, fontweight="bold")
ax.set_title("(a)  Targeted axis responds linearly to $\\alpha$ along $v_{\\mathrm{axis}}$",
             fontsize=13)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontsize(12); t.set_fontweight("bold")
ax.tick_params(axis="both", which="major", length=5, width=1.5)
leg = ax.legend(fontsize=12, loc="lower right", frameon=True, framealpha=0.95,
                edgecolor="black")
for t in leg.get_texts(): t.set_fontweight("bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ---- (b) per-axis (x, y, z) decomposition at α = +0.30 ----
ax = axes[1]
amax = shifts.alpha.abs().max()  # 0.30
positive = shifts[(shifts.alpha == amax) & (shifts.direction.isin(["v_axis", "v_perp"]))]
agg_xyz = (positive.groupby("direction")
                   .agg({"dread_x": "mean", "dread_y": "mean", "dread_z": "mean"})
                   .reindex(["v_axis", "v_perp"]))

axes_labels = ["x  (target)", "y", "z"]
xs = np.arange(3)
width = 0.36
bars1 = ax.bar(xs - width/2, agg_xyz.loc["v_axis", ["dread_x", "dread_y", "dread_z"]].values,
               width, color="#d62728", edgecolor="black", linewidth=0.8,
               label=r"$v_{\mathrm{axis}}$")
bars2 = ax.bar(xs + width/2, agg_xyz.loc["v_perp", ["dread_x", "dread_y", "dread_z"]].values,
               width, color="#7f7f7f", edgecolor="black", linewidth=0.8,
               label=r"$v_{\perp}$")

for b in list(bars1) + list(bars2):
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2, h + (0.0008 if h >= 0 else -0.0014),
            f"{h:+.3f}", ha="center", va="bottom" if h >= 0 else "top",
            fontsize=10.5, fontweight="bold")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(xs); ax.set_xticklabels(axes_labels, fontsize=13, fontweight="bold")
ax.set_ylabel(r"mean $\Delta$ probe-readout at $\alpha=+0.30$",
              fontsize=14, fontweight="bold")
ax.set_title("(b)  Edit is selective: $v_{\\mathrm{axis}}$ moves only $x$",
             fontsize=13)
for t in ax.get_yticklabels(): t.set_fontsize(12); t.set_fontweight("bold")
ax.tick_params(axis="y", which="major", length=5, width=1.5)
ax.grid(axis="y", alpha=0.30)
leg = ax.legend(fontsize=12, loc="upper right", frameon=True, framealpha=0.95,
                edgecolor="black")
for t in leg.get_texts(): t.set_fontweight("bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.tight_layout()
out_pdf = OUT / "fig_method_e_steering_L12.pdf"
out_png = OUT / "fig_method_e_steering_L12.png"
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.25)
fig.savefig(out_png, dpi=240, bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
print(f"saved {out_pdf}")
print(f"saved {out_png}")
