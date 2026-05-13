"""Cross-model summary figure for Method E (text-output activation steering).

Reads `steering_text.parquet` from three runs (Qwen / InternVL / LLaVA) and
produces:
  fig_method_e_steering.{pdf,png}    — 2x3 grid: rows = (Δ logit gap, flip rate);
                                         cols = three models. Per panel:
                                         v_axis vs v_perp curves over alpha.

The diagnostic claim is: v_axis edits move the model's verbal answer in the
predicted direction; v_perp edits do not.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/haoming/x-spatial-manual/data/steering")
OUT  = Path("/home/haoming/x-spatial-manual/figures")
OUT.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("Qwen2.5-VL-7B (L17)", ROOT / "free6dof_qwen7b_text_L17_x"),
    ("InternVL3-8B (L18)",  ROOT / "free6dof_internvl8b_text_L18_x"),
    ("LLaVA-OV-7B (L21)",   ROOT / "free6dof_llavaov7b_text_L21_x"),
]


def per_run_curves(run_dir: Path):
    df = pd.read_parquet(run_dir / "steering_text.parquet")
    # Δ logit gap aggregated by (direction, alpha)
    gap = (df[df.direction != "baseline"]
             .groupby(["direction", "alpha"])
             .agg(mean=("d_logit_gap", "mean"),
                  sem=("d_logit_gap", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
                  n=("d_logit_gap", "count"))
             .reset_index())
    # Flip rate vs baseline argmax
    flip_rows = []
    for (sid, oa, ob), g in df.groupby(["scene_id", "obj_a", "obj_b"]):
        b = g[g.direction == "baseline"]
        if b.empty: continue
        base_pred_a = bool(b.iloc[0].logit_gap > 0)
        for _, row in g[g.direction != "baseline"].iterrows():
            flip_rows.append({"direction": row.direction, "alpha": row.alpha,
                              "flip": int((row.logit_gap > 0) != base_pred_a)})
    fdf = pd.DataFrame(flip_rows)
    if fdf.empty:
        flip_agg = pd.DataFrame(columns=["direction", "alpha", "rate", "n"])
    else:
        flip_agg = (fdf.groupby(["direction", "alpha"])
                       .agg(rate=("flip", "mean"), n=("flip", "count"))
                       .reset_index())
    return gap, flip_agg


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.4})
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))

    color_axis = "#d62728"   # principal subspace direction (causal)
    color_perp = "#7f7f7f"   # null-space control

    headline = {}
    for col, (label, run_dir) in enumerate(RUNS):
        if not (run_dir / "steering_text.parquet").exists():
            for r in range(2):
                axes[r, col].set_axis_off()
                axes[r, col].text(0.5, 0.5, f"{label}\n(no data yet)",
                                   ha="center", va="center", fontsize=12)
            continue
        gap, flip = per_run_curves(run_dir)

        # Row 0: Δ logit gap
        ax = axes[0, col]
        for d, color, lbl in [("v_axis", color_axis, r"$v_{\mathrm{axis}}$ (subspace)"),
                              ("v_perp", color_perp, r"$v_{\perp}$ (null-space ctrl)")]:
            sub = gap[gap.direction == d].sort_values("alpha")
            ax.errorbar(sub.alpha, sub["mean"], yerr=sub["sem"], fmt="-o",
                        color=color, capsize=4, linewidth=2.4, markersize=7,
                        label=lbl)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"steering $\alpha$  (norm. $x$ units)", fontsize=12)
        if col == 0:
            ax.set_ylabel(r"$\Delta$ logit gap (color$_A$ − color$_B$)", fontsize=12, fontweight="bold")
        ax.set_title(label, fontsize=13, fontweight="bold")
        if col == 0:
            ax.legend(fontsize=10, loc="best")

        # Row 1: flip rate
        ax = axes[1, col]
        for d, color in [("v_axis", color_axis), ("v_perp", color_perp)]:
            sub = flip[flip.direction == d].sort_values("alpha")
            ax.plot(sub.alpha, sub.rate, "-o", color=color, linewidth=2.4, markersize=7)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.grid(alpha=0.25)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(r"steering $\alpha$  (norm. $x$ units)", fontsize=12)
        if col == 0:
            ax.set_ylabel("answer flip rate vs baseline", fontsize=12, fontweight="bold")

        # Headline numbers: max-α flip rates and gap shift
        amax = gap.alpha.abs().max()
        v_ax_pos = gap[(gap.direction == "v_axis") & (gap.alpha == amax)]["mean"]
        v_pp_pos = gap[(gap.direction == "v_perp") & (gap.alpha == amax)]["mean"]
        v_ax_neg = gap[(gap.direction == "v_axis") & (gap.alpha == -amax)]["mean"]
        v_pp_neg = gap[(gap.direction == "v_perp") & (gap.alpha == -amax)]["mean"]
        f_ax_pos = flip[(flip.direction == "v_axis") & (flip.alpha == amax)]["rate"]
        f_pp_pos = flip[(flip.direction == "v_perp") & (flip.alpha == amax)]["rate"]
        f_ax_neg = flip[(flip.direction == "v_axis") & (flip.alpha == -amax)]["rate"]
        f_pp_neg = flip[(flip.direction == "v_perp") & (flip.alpha == -amax)]["rate"]
        headline[label] = {
            "alpha_max": float(amax),
            "Δgap_v_axis@+α": float(v_ax_pos.iloc[0]) if len(v_ax_pos) else None,
            "Δgap_v_perp@+α": float(v_pp_pos.iloc[0]) if len(v_pp_pos) else None,
            "Δgap_v_axis@-α": float(v_ax_neg.iloc[0]) if len(v_ax_neg) else None,
            "Δgap_v_perp@-α": float(v_pp_neg.iloc[0]) if len(v_pp_neg) else None,
            "flip_v_axis@+α": float(f_ax_pos.iloc[0]) if len(f_ax_pos) else None,
            "flip_v_perp@+α": float(f_pp_pos.iloc[0]) if len(f_pp_pos) else None,
            "flip_v_axis@-α": float(f_ax_neg.iloc[0]) if len(f_ax_neg) else None,
            "flip_v_perp@-α": float(f_pp_neg.iloc[0]) if len(f_pp_neg) else None,
        }

    fig.suptitle(
        "Method E — text-output activation steering on the spatial subspace at the cognitive-map layer\n"
        r"top: $\Delta$logit gap (color$_A$ − color$_B$);   bottom: answer-flip rate vs.\ no-intervention baseline",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_method_e_steering.pdf", bbox_inches="tight", pad_inches=0.25)
    fig.savefig(OUT / "fig_method_e_steering.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"saved {OUT/'fig_method_e_steering.pdf'}")

    # Also dump headline JSON
    out_json = ROOT / "method_e_headline.json"
    out_json.write_text(json.dumps(headline, indent=2))
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
