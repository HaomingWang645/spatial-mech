#!/usr/bin/env python
"""Aggregate v2 experiment results: combine training-final IID metrics
with OOD eval JSONs, write a tidy CSV, JSON, and comparison plot.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Match log lines like "FINAL  lm=0.1192  dir=0.1144  R²=0.9242  vqa_acc=0.9424 (n=330)"
FINAL_RE = re.compile(
    r"FINAL\s+lm=([\d.]+)\s+dir=([\d.]+)\s+R²=([\d.]+)\s+vqa_acc=([\d.]+)\s+\(n=(\d+)\)"
)


def parse_train_log(path: Path) -> dict | None:
    if not path.exists():
        return None
    for line in reversed(path.read_text().splitlines()):
        m = FINAL_RE.search(line)
        if m:
            return {
                "lm": float(m.group(1)), "dir": float(m.group(2)),
                "r2": float(m.group(3)), "vqa": float(m.group(4)),
                "n": int(m.group(5)),
            }
    return None


def parse_run_name(name: str) -> tuple[str, float, int] | None:
    """Extract (model, lambda, seed) from a run name like
    'qwen_lam1_seed0', 'intern_lam0.3_seed0'."""
    m = re.match(r"(qwen|intern)_lam([\d.]+)_seed(\d+)", name)
    if not m:
        return None
    return m.group(1), float(m.group(2)), int(m.group(3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-prefix", type=Path, required=True)
    args = p.parse_args()

    rows = []
    log_dir = Path("logs/v2")
    eval_dir = Path("reports/dirichlet_train_v2")
    seen = set()

    # Sweep through all training logs
    for log in sorted(log_dir.glob("*.log")):
        name = log.stem
        if name.startswith("eval"):
            continue
        if not parse_run_name(name):
            continue
        seen.add(name)

        train_final = parse_train_log(log)
        ood_path = eval_dir / f"{name}_ood.json"
        ood = json.loads(ood_path.read_text()) if ood_path.exists() else None

        family, lam, seed = parse_run_name(name)
        row = {
            "name": name,
            "model": family,
            "lambda": lam,
            "seed": seed,
            "iid_lm": train_final["lm"] if train_final else None,
            "iid_dir": train_final["dir"] if train_final else None,
            "iid_r2": train_final["r2"] if train_final else None,
            "iid_vqa": train_final["vqa"] if train_final else None,
            "iid_n": train_final["n"] if train_final else None,
            "ood_lm": ood["val_lm_loss"] if ood else None,
            "ood_dir": ood["val_dirichlet_ratio"] if ood else None,
            "ood_r2": ood["val_alignment_R2"] if ood else None,
            "ood_vqa": ood["vqa_accuracy"] if ood else None,
            "ood_n": ood["vqa_n"] if ood else None,
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["model", "lambda", "seed"])
    csv_path = args.out_prefix.with_suffix(".csv")
    json_path = args.out_prefix.with_suffix(".json")
    df.to_csv(csv_path, index=False)

    # Group statistics
    summary = {}
    for (model, lam), grp in df.groupby(["model", "lambda"]):
        key = f"{model}_lam{lam}"
        s = {}
        for col in ("iid_dir", "iid_r2", "iid_vqa", "ood_dir", "ood_r2", "ood_vqa", "ood_lm"):
            vals = [v for v in grp[col] if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if vals:
                s[col + "_mean"] = float(np.mean(vals))
                s[col + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                s[col + "_n"] = len(vals)
        s["n_runs"] = len(grp)
        summary[key] = s

    json_path.write_text(json.dumps({
        "rows": df.to_dict(orient="records"),
        "summary": summary,
    }, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x))

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print()
    # Pretty print key comparisons
    print("=" * 90)
    print(f"{'Model':<8} {'λ':<6} {'n':<3} | {'IID Dir':<14} {'IID R²':<14} {'IID VQA':<14} | {'OOD Dir':<14} {'OOD R²':<14} {'OOD VQA':<14}")
    print("=" * 90)
    for key, s in sorted(summary.items()):
        model, lam = key.split("_lam")
        n = s.get("n_runs", 0)

        def fmt(col, fmt_str=".4f"):
            mean = s.get(f"{col}_mean")
            std = s.get(f"{col}_std")
            if mean is None:
                return "—"
            return f"{mean:.4f}±{std:.4f}"

        print(f"{model:<8} {lam:<6} {n:<3} | "
              f"{fmt('iid_dir'):<14} {fmt('iid_r2'):<14} {fmt('iid_vqa'):<14} | "
              f"{fmt('ood_dir'):<14} {fmt('ood_r2'):<14} {fmt('ood_vqa'):<14}")
    print()

    # Plot: 2x4 panels (rows=IID/OOD, cols=metrics) showing bars by lambda per model
    if len(summary) >= 2:
        models = sorted({m.split("_lam")[0] for m in summary.keys()})
        metrics = [
            ("dir", "Dirichlet ratio"),
            ("r2", "Alignment R²"),
            ("vqa", "VQA accuracy"),
        ]
        splits = [("iid", "IID held-out"), ("ood", "OOD (tier_b)")]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for r, (split_key, split_name) in enumerate(splits):
            for c, (mkey, mname) in enumerate(metrics):
                ax = axes[r, c]
                col = f"{split_key}_{mkey}"
                xs = []
                for model in models:
                    for lam in sorted({k.split("_lam")[1] for k in summary if k.startswith(f"{model}_lam")}):
                        s = summary[f"{model}_lam{lam}"]
                        m = s.get(f"{col}_mean")
                        sd = s.get(f"{col}_std", 0)
                        if m is None:
                            continue
                        xs.append((f"{model}\nλ={lam}", m, sd))
                if not xs:
                    continue
                labels, means, stds = zip(*xs)
                ax.bar(range(len(labels)), means, yerr=stds, capsize=4,
                       color=["tab:gray" if "lam0.0" in lbl or "lam0\n" in lbl else "tab:purple" for lbl in labels])
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=0, fontsize=8)
                ax.set_title(f"{split_name}: {mname}")
                ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        png = args.out_prefix.with_suffix(".png")
        fig.savefig(png, dpi=130, bbox_inches="tight")
        print(f"Wrote {png}")


if __name__ == "__main__":
    main()
