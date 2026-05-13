"""Cross-method summary figure for the real-world counterfactual pilots.

Two panels:
  (a) Per-method baseline accuracy vs. modified accuracy (paired bars).
  (b) Per-method answer-flip rate vs. baseline.

Pulls from the three pilot output dirs under data/realworld_counterfactual/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/haoming/x-spatial-manual/data/realworld_counterfactual")
OUT  = Path("/home/haoming/x-spatial-manual/figures")
OUT.mkdir(parents=True, exist_ok=True)


def load_method1():
    df = pd.read_parquet(ROOT / "pilot_qwen" / "pilot_results.parquet")
    rows = []
    for manip in ("baseline", "grayscale", "hue_shift"):
        sub = df[df.manipulation == manip]
        rows.append({
            "method": f"M1·{manip}",
            "n": len(sub),
            "acc": sub.correct.mean(),
            "flip": sub.flip_vs_baseline.mean() if manip != "baseline" else 0.0,
        })
    return rows


def load_method2():
    df = pd.read_parquet(ROOT / "pilot_qwen_method2" / "pilot_results.parquet")
    return [
        {"method": "M2·baseline (per-obj hue)", "n": len(df),
         "acc": df.base_correct.mean(), "flip": 0.0},
        {"method": "M2·hue-shift target obj",   "n": len(df),
         "acc": df.mod_correct.mean(),  "flip": df.flip.mean()},
    ]


def load_method3():
    df = pd.read_parquet(ROOT / "pilot_qwen_method3" / "pilot_results.parquet")
    return [
        {"method": "M3·baseline (pos-swap)",     "n": len(df),
         "acc": df.base_correct.mean(), "flip": 0.0},
        {"method": "M3·2D position swap",        "n": len(df),
         "acc": df.mod_correct.mean(),  "flip": df.flip.mean()},
    ]


# Build the long table
rows = load_method1() + load_method2() + load_method3()
df = pd.DataFrame(rows)
print(df)

# ---- Plot ----
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.4})
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

# Panel A: accuracy
ax = axes[0]
xs = np.arange(len(df))
colors = []
for m in df.method:
    if "baseline" in m:
        colors.append("#7f7f7f")
    elif m.startswith("M1"):
        colors.append("#2ca02c")
    elif m.startswith("M2"):
        colors.append("#1f77b4")
    elif m.startswith("M3"):
        colors.append("#d62728")
    else:
        colors.append("#888888")
bars = ax.bar(xs, df.acc * 100, color=colors, edgecolor="black", linewidth=0.7)
for b, v, n in zip(bars, df.acc, df.n):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
            f"{v*100:.1f}%\n(n={n})", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels(df.method, rotation=24, ha="right", fontsize=10)
ax.set_ylabel("VSI-Bench accuracy (%)", fontsize=12, fontweight="bold")
ax.set_ylim(0, max(df.acc * 100) * 1.30)
ax.set_title("(a)  Accuracy under each manipulation", fontsize=12, pad=8)
ax.grid(axis="y", alpha=0.30)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel B: flip rate (excluding baselines which are 0 by construction)
ax = axes[1]
mask = ~df.method.str.contains("baseline")
sub = df[mask].reset_index(drop=True)
xs = np.arange(len(sub))
colors2 = []
for m in sub.method:
    if m.startswith("M1"):
        colors2.append("#2ca02c")
    elif m.startswith("M2"):
        colors2.append("#1f77b4")
    elif m.startswith("M3"):
        colors2.append("#d62728")
    else:
        colors2.append("#888888")
bars = ax.bar(xs, sub.flip * 100, color=colors2, edgecolor="black", linewidth=0.7)
for b, v in zip(bars, sub.flip):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.7,
            f"{v*100:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels(sub.method, rotation=24, ha="right", fontsize=10)
ax.set_ylabel("answer-flip rate vs. baseline (%)", fontsize=12, fontweight="bold")
ax.set_ylim(0, max(sub.flip * 100) * 1.40)
ax.set_title("(b)  Answer flip rate", fontsize=12, pad=8)
ax.grid(axis="y", alpha=0.30)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("Real-world counterfactual pilot — Qwen2.5-VL-7B on VSI-Bench (Tier-C arkitscenes/scannet)",
             fontsize=13, y=1.00)
fig.tight_layout()
fig.savefig(OUT / "fig_realworld_counterfactual_pilot.pdf",
            bbox_inches="tight", pad_inches=0.25)
fig.savefig(OUT / "fig_realworld_counterfactual_pilot.png",
            dpi=200, bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
print("saved fig_realworld_counterfactual_pilot.{pdf,png}")
