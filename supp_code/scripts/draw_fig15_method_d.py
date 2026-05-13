#!/usr/bin/env python
"""Redraw fig15: Method D pre-registered predictions vs observed accuracy.

Replaces the placeholder fig15 (predicted-only bars under H_position / H_color)
with a figure that overlays the same predictions with the actual observed
accuracies on the discriminating question kind (`distance_order`, n=131
unique pairs * 4 seeds, Qwen2.5-VL-7B).

Outputs:
  figures/fig15_method_d_predictions.{png,pdf}
"""
from __future__ import annotations

import json
from collections import defaultdict
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = Path("reports/method_d")
OUT_PNG = Path("figures/fig15_method_d_predictions.png")
OUT_PDF = Path("figures/fig15_method_d_predictions.pdf")

VARIANTS = ("original", "color_swap", "position_swap")
VARIANT_LABEL = {
    "original":      "original",
    "color_swap":    "color-swap\n(positions same)",
    "position_swap": "position-swap\n(positions changed)",
}

# Pre-registered predictions (the same numbers shown in the original fig15).
PRED_HPOS   = {"original": 1.00, "color_swap": 1.00, "position_swap": 0.50}
PRED_HCOLOR = {"original": 1.00, "color_swap": 0.40, "position_swap": 0.50}


def load_observed():
    """Return mean+std accuracy per (ckpt, variant) restricted to distance_order."""
    by_seed = defaultdict(lambda: defaultdict(list))  # ckpt -> variant -> [acc per seed]
    for fn in sorted(REPORT_DIR.glob("cf_*.json")):
        name = fn.name
        ckpt = "residMulti" if "residMulti" in name else "lam0"
        seed = name.split("seed")[1].split(".")[0]
        rows = json.load(open(fn))["rows"]
        per_var = defaultdict(list)
        for r in rows:
            if r["kind"] != "distance_order":
                continue
            per_var[r["variant"]].append(r["correct"])
        for v in VARIANTS:
            arr = per_var[v]
            if arr:
                by_seed[ckpt][v].append(sum(arr) / len(arr))

    out = {}
    for ckpt, varmap in by_seed.items():
        out[ckpt] = {}
        for v in VARIANTS:
            arr = np.asarray(varmap[v], dtype=float)
            out[ckpt][v] = (float(arr.mean()), float(arr.std()), int(len(arr)))
    return out


def main():
    obs = load_observed()
    print("Observed (distance_order):")
    for ckpt, vmap in obs.items():
        for v in VARIANTS:
            m, s, n = vmap[v]
            print(f"  {ckpt:11s} {v:14s} {m*100:5.1f} ± {s*100:4.1f}  (n_seeds={n})")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=110, gridspec_kw={"width_ratios": [1.0, 1.0]})

    n_var = len(VARIANTS)
    x = np.arange(n_var)

    # ----- Panel (a): predictions vs observed (distance_order) -----
    ax = axes[0]
    bw = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bw

    color_pred_pos   = "#A8C8E8"
    color_pred_color = "#E8B8B8"
    color_obs_lam0   = "#3A77C2"
    color_obs_resid  = "#E07060"

    pred_pos   = np.array([PRED_HPOS[v]   for v in VARIANTS])
    pred_color = np.array([PRED_HCOLOR[v] for v in VARIANTS])
    obs_l_mean = np.array([obs["lam0"][v][0]       for v in VARIANTS])
    obs_l_std  = np.array([obs["lam0"][v][1]       for v in VARIANTS])
    obs_r_mean = np.array([obs["residMulti"][v][0] for v in VARIANTS])
    obs_r_std  = np.array([obs["residMulti"][v][1] for v in VARIANTS])

    ax.bar(x + offsets[0], pred_pos,   bw, color=color_pred_pos,
           edgecolor="#406090", linewidth=0.8, hatch="//", label="H$_{\\mathrm{position}}$ prediction")
    ax.bar(x + offsets[1], pred_color, bw, color=color_pred_color,
           edgecolor="#A04040", linewidth=0.8, hatch="\\\\", label="H$_{\\mathrm{color}}$ prediction")
    ax.bar(x + offsets[2], obs_l_mean, bw, yerr=obs_l_std, color=color_obs_lam0,
           edgecolor="#1F4570", linewidth=0.8,
           error_kw=dict(ecolor="#1F2F40", capsize=3, lw=0.9),
           label="observed: LM-only LoRA ($\\lambda{=}0$)")
    ax.bar(x + offsets[3], obs_r_mean, bw, yerr=obs_r_std, color=color_obs_resid,
           edgecolor="#A03040", linewidth=0.8,
           error_kw=dict(ecolor="#401F2F", capsize=3, lw=0.9),
           label="observed: residMulti ($\\lambda{=}0.3$)")

    # numeric labels
    for xi, v in zip(x, VARIANTS):
        ax.text(xi + offsets[0], pred_pos[VARIANTS.index(v)]   + 0.02,
                f"{int(round(PRED_HPOS[v]*100))}", ha="center", va="bottom",
                fontsize=7, color="#406090")
        ax.text(xi + offsets[1], pred_color[VARIANTS.index(v)] + 0.02,
                f"{int(round(PRED_HCOLOR[v]*100))}", ha="center", va="bottom",
                fontsize=7, color="#A04040")
    for xi, m, s in zip(x, obs_l_mean, obs_l_std):
        ax.text(xi + offsets[2], m + s + 0.02, f"{m*100:.0f}",
                ha="center", va="bottom", fontsize=7, color="#1F4570",
                fontweight="bold")
    for xi, m, s in zip(x, obs_r_mean, obs_r_std):
        ax.text(xi + offsets[3], m + s + 0.02, f"{m*100:.0f}",
                ha="center", va="bottom", fontsize=7, color="#A03040",
                fontweight="bold")

    ax.axhline(0.5, ls=":", lw=0.8, color="#888888")
    ax.text(-0.42, 0.505, "chance (2-way)", fontsize=7, color="#666666",
            ha="left", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABEL[v] for v in VARIANTS], fontsize=8.5)
    ax.set_ylabel("Accuracy w.r.t. original ground truth", fontsize=9)
    ax.set_ylim(0.0, 1.20)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_title("(a)  distance_order  (n=131 unique pairs $\\times$ 4 seeds)",
                 fontsize=9.5)
    ax.legend(loc="upper right", fontsize=7.2, ncol=1, frameon=True,
              handlelength=1.6, borderpad=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ----- Panel (b): relative_position (camera-fixed; uninformative) -----
    ax = axes[1]
    obs_l_rp = []
    obs_r_rp = []
    # Recompute for relative_position
    by_seed = defaultdict(lambda: defaultdict(list))
    for fn in sorted(REPORT_DIR.glob("cf_*.json")):
        ckpt = "residMulti" if "residMulti" in fn.name else "lam0"
        rows = json.load(open(fn))["rows"]
        per_var = defaultdict(list)
        for r in rows:
            if r["kind"] != "relative_position":
                continue
            per_var[r["variant"]].append(r["correct"])
        for v in VARIANTS:
            arr = per_var[v]
            if arr:
                by_seed[ckpt][v].append(sum(arr) / len(arr))
    rp_l = np.array([np.mean(by_seed["lam0"][v])       for v in VARIANTS])
    rp_r = np.array([np.mean(by_seed["residMulti"][v]) for v in VARIANTS])

    ax.bar(x + offsets[0], pred_pos,   bw, color=color_pred_pos,
           edgecolor="#406090", linewidth=0.8, hatch="//")
    ax.bar(x + offsets[1], pred_color, bw, color=color_pred_color,
           edgecolor="#A04040", linewidth=0.8, hatch="\\\\")
    ax.bar(x + offsets[2], rp_l, bw, color=color_obs_lam0, edgecolor="#1F4570",
           linewidth=0.8)
    ax.bar(x + offsets[3], rp_r, bw, color=color_obs_resid, edgecolor="#A03040",
           linewidth=0.8)

    for xi, m in zip(x, rp_l):
        ax.text(xi + offsets[2], m + 0.02, f"{m*100:.0f}", ha="center",
                va="bottom", fontsize=7, color="#1F4570", fontweight="bold")
    for xi, m in zip(x, rp_r):
        ax.text(xi + offsets[3], m + 0.02, f"{m*100:.0f}", ha="center",
                va="bottom", fontsize=7, color="#A03040", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABEL[v] for v in VARIANTS], fontsize=8.5)
    ax.set_ylabel("Accuracy w.r.t. original ground truth", fontsize=9)
    ax.set_ylim(0.0, 1.20)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_title("(b)  relative_position  (n=299 questions $\\times$ 4 seeds)",
                 fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Method D — counterfactual swap on Qwen2.5-VL-7B: pre-registered "
        "predictions vs observed accuracy",
        fontsize=11, y=1.005,
    )

    # Verdict text below the panels
    verdict = (
        "Verdict: neither hypothesis matches exactly on distance_order — color-swap "
        "drops accuracy by ~30 pp (rejecting H$_{\\mathrm{position}}$, which predicted "
        "no drop) and position-swap drops by a near-identical amount, consistent "
        "with color-keyed object lookup OR an image-region heuristic. residMulti "
        "($\\lambda{=}0.3$) does NOT shift behavior toward H$_{\\mathrm{position}}$ "
        "despite reorganizing the latent space (color decoding 0.49$\\to$0.24, 3D "
        "RSA 0.02$\\to$0.60). On relative_position, all variants are at 100% "
        "because the answer is camera-fixed and unaffected by either swap."
    )
    fig.text(0.5, -0.10, verdict, ha="center", va="top", fontsize=8.2, wrap=True)

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    fig.savefig(OUT_PDF,           bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
