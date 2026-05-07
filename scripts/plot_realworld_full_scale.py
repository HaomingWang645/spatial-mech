"""Cross-VLM summary for the full-scale Method-1 real-world counterfactual.

Reads pilot_results.parquet from each model run dir under
data/realworld_counterfactual/full_*/ and produces:
  fig_realworld_full_scale.{pdf,png}    cross-model summary
  data/realworld_counterfactual/full_summary.csv  paired numbers

Two panels:
  (a) Accuracy by (model, manipulation), grouped bars.
  (b) Answer-flip rate by (model, manipulation).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/haoming/x-spatial-manual/data/realworld_counterfactual")
OUT  = Path("/home/haoming/x-spatial-manual/figures")

RUNS = [
    ("Qwen2.5-VL-7B",  ROOT / "full_qwen"),
    ("InternVL3-8B",   ROOT / "full_internvl"),
    ("LLaVA-OV-7B",    ROOT / "full_llava"),
]
MANIPS = ["baseline", "grayscale", "hue_shift"]
MANIP_COLORS = {"baseline": "#7f7f7f", "grayscale": "#2ca02c", "hue_shift": "#ff7f0e"}


def summarize_run(run_dir: Path):
    fp = run_dir / "pilot_results.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    rows = []
    for m in MANIPS:
        sub = df[df.manipulation == m]
        rows.append({
            "manipulation": m,
            "n":   len(sub),
            "acc": float(sub.correct.mean()) if len(sub) else float("nan"),
            "flip": float(sub.flip_vs_baseline.mean()) if len(sub) and m != "baseline" else 0.0,
        })
    return pd.DataFrame(rows)


# Build long tidy frame
all_rows = []
for label, run in RUNS:
    s = summarize_run(run)
    if s is None:
        for m in MANIPS:
            all_rows.append({"model": label, "manipulation": m,
                             "n": 0, "acc": float("nan"), "flip": float("nan")})
        continue
    s["model"] = label
    all_rows.append(s)
big = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
big.to_csv(ROOT / "full_summary.csv", index=False)
print(big.to_string(index=False))


# ---- Plot ----
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.4})
fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
models = [r[0] for r in RUNS]
xs = np.arange(len(models))
width = 0.27

for ax_idx, (ax, metric, ylabel, title) in enumerate([
    (axes[0], "acc",  "Accuracy (%)",                   "(a)  Accuracy under each manipulation"),
    (axes[1], "flip", "Answer-flip rate vs. baseline (%)", "(b)  Answer-flip rate"),
]):
    for i, m in enumerate(MANIPS):
        if ax_idx == 1 and m == "baseline":
            continue
        sub = big[big.manipulation == m].set_index("model").reindex(models)
        vals = (sub[metric] * 100).values
        b = ax.bar(xs + (i - 1) * width if ax_idx == 0 else
                   xs + (i - 1.5) * width,  # shift when 2 bars
                   vals, width, color=MANIP_COLORS[m], edgecolor="black",
                   linewidth=0.7, label=m)
        for bb, v in zip(b, vals):
            if np.isfinite(v):
                ax.text(bb.get_x() + bb.get_width()/2, bb.get_height() + 0.5,
                        f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(xs); ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(axis="y", alpha=0.30)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if ax_idx == 0:
        ax.legend(fontsize=10, loc="upper right", framealpha=0.95, edgecolor="black")

fig.suptitle("Method 1 (image-wide colour ablation) — full-scale on VSI-Bench spatial-rel subset (n≈1,678)",
             fontsize=13, y=1.00)
fig.tight_layout()
fig.savefig(OUT / "fig_realworld_full_scale.pdf", bbox_inches="tight", pad_inches=0.25)
fig.savefig(OUT / "fig_realworld_full_scale.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
print(f"\nsaved {OUT/'fig_realworld_full_scale.pdf'}")
