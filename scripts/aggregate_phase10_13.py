#!/usr/bin/env python
"""Aggregate VSI-Bench eval JSONs from phases 10-13 into table-ready cells.

For each (stream, model, lambda) tuple, glob the per-seed JSONs, average
the overall accuracy and per-task accuracies. Print as Markdown table
ready to drop into the paper.

Usage:
    python scripts/aggregate_phase10_13.py
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path

EVAL_DIR = Path("reports/vsi_full_eval")

# Map regex -> stream name; first capture group is model short, second is lam
STREAMS = [
    # Phase 10: spatID baseline (Kang et al.)
    ("spatID", r"^spatID_(qwen|intern|llavaov)_lam([0-9.]+)_seed(\d+)\.json$"),
    # Phase 11: n=4 validation cells
    ("residMulti5L", r"^residMulti5L_(qwen|intern)_lam([0-9.]+)_seed(\d+)\.json$"),
    ("residMultiSched", r"^residMultiSched_(qwen|intern)_lam([0-9.]+)_seed(\d+)\.json$"),
    # Phase 12: LLaVA-OV
    ("llavaov_baseline", r"^llavaov_lam(0)_seed(\d+)\.json$"),
    ("residMulti_llavaov", r"^residMulti_llavaov_lam([0-9.]+)_seed(\d+)\.json$"),
    # Reference baselines (already exist in v12 report)
    ("baseline", r"^(qwen|intern)_lam0_seed(\d+)\.json$"),
    ("no-resi", r"^(qwen|intern)_lam([0-9.]+)_seed(\d+)\.json$"),  # lam != 0 → no-resi
    ("residMulti", r"^residMulti_(qwen|intern)_lam([0-9.]+)_seed(\d+)\.json$"),
    ("shufflectrl", r"^shufflectrl_(qwen|intern)_lam([0-9.]+)_seed(\d+)\.json$"),
]


def aggregate():
    cells = defaultdict(lambda: {"overall": [], "per_task": defaultdict(list)})
    for fname in sorted(os.listdir(EVAL_DIR)):
        if not fname.endswith(".json"):
            continue
        for stream_name, pat in STREAMS:
            m = re.match(pat, fname)
            if not m:
                continue
            groups = m.groups()
            if stream_name == "baseline":
                model, seed = groups
                lam = "0"
            elif stream_name == "no-resi":
                model, lam, seed = groups
                if lam in ("0", "0.0"):
                    break  # baseline rule already matched; skip
            elif stream_name == "llavaov_baseline":
                model = "llavaov"
                lam, seed = groups
            elif stream_name == "residMulti_llavaov":
                model = "llavaov"
                lam, seed = groups
            else:
                model, lam, seed = groups
            try:
                with open(EVAL_DIR / fname) as f:
                    d = json.load(f)
                summary = d["summary"]
                overall = summary["accuracy"]
                cells[(stream_name, model, lam)]["overall"].append(overall)
                for tname, tinfo in summary.get("by_kind", {}).items():
                    cells[(stream_name, model, lam)]["per_task"][tname].append(tinfo["acc"])
            except Exception as exc:
                print(f"WARN: failed to read {fname}: {exc}")
            break  # only match first stream regex
    return cells


def fmt_cell(stream, model, lam, cell):
    n = len(cell["overall"])
    if n == 0:
        return None
    mean = sum(cell["overall"]) / n
    return {
        "stream": stream, "model": model, "lambda": lam, "n": n,
        "overall": mean,
        "per_task": {t: sum(v) / len(v) for t, v in cell["per_task"].items()},
    }


def main():
    cells = aggregate()
    rows = []
    for k, c in cells.items():
        r = fmt_cell(*k, c)
        if r:
            rows.append(r)
    rows.sort(key=lambda r: (r["model"], r["stream"], r["lambda"]))

    # Print summary
    print(f"# Aggregated cells (raw VSI-Bench overall accuracy)\n")
    print(f"| stream | model | lambda | n | overall |")
    print(f"|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['stream']} | {r['model']} | {r['lambda']} | {r['n']} | {r['overall']:.4f} |")

    # Save JSON
    with open("reports/dirichlet_train/phase10_13_aggregated.json", "w") as f:
        json.dump(rows, f, indent=2, default=float)
    print(f"\nSaved JSON: reports/dirichlet_train/phase10_13_aggregated.json", file=__import__('sys').stderr)


if __name__ == "__main__":
    main()
