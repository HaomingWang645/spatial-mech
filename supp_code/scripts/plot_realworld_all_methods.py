"""Final consolidated cross-method figure — M1 + M2 v2 + M3 v2.

Three groups of bars per VLM: M1 (image-wide colour), M2 (per-object hue),
M3 (depth-aware position swap). Two panels: accuracy + flip rate.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/haoming/x-spatial-manual/data/realworld_counterfactual")
OUT  = Path("/home/haoming/x-spatial-manual/figures")
OUT.mkdir(parents=True, exist_ok=True)


def m1_summary(model_dir: Path):
    df = pd.read_parquet(model_dir / "pilot_results.parquet")
    base = df[df.manipulation == "baseline"]
    gray = df[df.manipulation == "grayscale"]
    return {"n": len(gray),
            "base_acc": base.correct.mean(),
            "mod_acc":  gray.correct.mean(),
            "flip":     gray.flip_vs_baseline.mean()}


def m2v2_summary(model_dir: Path):
    df = pd.read_parquet(model_dir / "results.parquet")
    return {"n": len(df),
            "base_acc": df.base_correct.mean(),
            "mod_acc":  df.mod_correct.mean(),
            "flip":     df.flip.mean()}


def m3v2_summary(model_dir: Path):
    df = pd.read_parquet(model_dir / "results.parquet")
    return {"n": len(df),
            "base_acc": df.base_correct.mean(),
            "mod_acc":  df.mod_correct.mean(),
            "flip":     df.flip.mean()}


MODELS = ["Qwen2.5-VL-7B", "InternVL3-8B", "LLaVA-OV-7B"]
M1_DIRS = ["full_qwen", "full_internvl", "full_llava"]
M2_DIRS = ["m2v2_qwen", "m2v2_internvl", "m2v2_llava"]
M3_DIRS = ["m3v2_qwen", "m3v2_internvl", "m3v2_llava"]

rows = []
for model, m1d, m2d, m3d in zip(MODELS, M1_DIRS, M2_DIRS, M3_DIRS):
    for label, fn, sub in [("M1 grayscale", m1_summary, m1d),
                           ("M2 per-obj hue", m2v2_summary, m2d),
                           ("M3 depth swap", m3v2_summary, m3d)]:
        path = ROOT / sub
        try:
            s = fn(path)
            rows.append({"model": model, "method": label, **s})
        except Exception as e:
            print(f"[skip] {model} / {label}: {e}")
            rows.append({"model": model, "method": label, "n": 0,
                         "base_acc": float("nan"), "mod_acc": float("nan"),
                         "flip": float("nan")})

df = pd.DataFrame(rows)
df.to_csv(ROOT / "all_methods_summary.csv", index=False)
print(df.to_string(index=False))


# ---- Plot ----
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.4})
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))

method_colors = {"M1 grayscale": "#2ca02c",
                 "M2 per-obj hue": "#1f77b4",
                 "M3 depth swap": "#d62728"}

xs = np.arange(len(MODELS))
width = 0.27

# Panel A: accuracy with paired (base vs mod) per method
ax = axes[0]
for i, method in enumerate(["M1 grayscale", "M2 per-obj hue", "M3 depth swap"]):
    sub = df[df.method == method].set_index("model").reindex(MODELS)
    base = sub.base_acc.values * 100
    mod  = sub.mod_acc.values  * 100
    pos = xs + (i - 1) * width
    ax.bar(pos - 0.06, base, 0.12, color=method_colors[method], alpha=0.45,
           edgecolor="black", linewidth=0.5,
           label=f"{method} baseline" if i == 0 else None)
    ax.bar(pos + 0.06, mod,  0.12, color=method_colors[method], alpha=1.0,
           edgecolor="black", linewidth=0.5,
           label=method)
    for x, mv in zip(pos + 0.06, mod):
        if np.isfinite(mv):
            ax.text(x, mv + 0.6, f"{mv:.1f}", ha="center", fontsize=8.5, fontweight="bold")

ax.set_xticks(xs); ax.set_xticklabels(MODELS, fontsize=11, fontweight="bold")
ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
ax.set_title("(a)  Accuracy: faded = baseline, solid = manipulated", fontsize=12, pad=8)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(fontsize=9, loc="upper right", ncol=1, framealpha=0.95, edgecolor="black")

# Panel B: flip rate
ax = axes[1]
for i, method in enumerate(["M1 grayscale", "M2 per-obj hue", "M3 depth swap"]):
    sub = df[df.method == method].set_index("model").reindex(MODELS)
    flip = sub.flip.values * 100
    pos = xs + (i - 1) * width
    bars = ax.bar(pos, flip, width * 0.85, color=method_colors[method],
                  edgecolor="black", linewidth=0.7, label=method)
    for b, v in zip(bars, flip):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.7,
                    f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels(MODELS, fontsize=11, fontweight="bold")
ax.set_ylabel("Answer-flip rate (%)", fontsize=12, fontweight="bold")
ax.set_title("(b)  Flip rate vs. unmanipulated baseline", fontsize=12, pad=8)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(fontsize=10, loc="upper right", framealpha=0.95, edgecolor="black")

fig.suptitle("Real-world counterfactual: three perturbations on VSI-Bench spatial-rel subset",
             fontsize=13, y=1.00)
fig.tight_layout()
fig.savefig(OUT / "fig_realworld_all_methods.pdf", bbox_inches="tight", pad_inches=0.25)
fig.savefig(OUT / "fig_realworld_all_methods.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
print(f"\nsaved {OUT/'fig_realworld_all_methods.pdf'}")
