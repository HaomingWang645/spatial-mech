#!/usr/bin/env python
"""Phase-12: coord-shuffle control on the headline residMulti cells.

The strongest negative control we can run: same training data, same
loss formula, same seeds, same lambda --- but we randomly permute the
3D coordinates inside each example before computing the kernel graph,
via the existing --coord-shuffle flag in train_qwen_dirichlet_v2.py.
This destroys the 3D structure of the kernel without changing the
magnitude or distribution of the Dirichlet penalty.

If our gains depend on real 3D geometry, the coord-shuffle runs
should show no improvement (or a regression) vs the LoRA lambda=0
baseline. If the gains were just from "any kernel-shaped penalty",
coord-shuffle would still help. This experiment directly addresses
the most natural reviewer concern.

Sweep:
  models  : Qwen2.5-VL-7B, InternVL3-8B
  stream  : residMulti  (3-layer, residualized; matches the Dirichlet
                          headline cells in v12)
  lambda  : {0.3, 1, 3}
  seeds   : {0, 1, 2, 3}
  shuffle : ON

Total jobs: 2 * 1 * 3 * 4 * 2 = 48 (24 train + 24 eval).
"""
from __future__ import annotations
import json
from pathlib import Path

PY = "/home/haoming/miniconda3/envs/vlm-ex/bin/python3"
QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"
INTERN = "OpenGVLab/InternVL3-8B-hf"
TRAIN_PY = "scripts/train_qwen_dirichlet_v2.py"
EVAL_PY = "scripts/eval_vsi_batched.py"
TRAIN_JSONL = "data/dirichlet_train_v2/train.jsonl"
VAL_JSONL = "data/dirichlet_train_v2/val_iid.jsonl"
VSI_FULL = "data/vsi_bench_full/eval_full.jsonl"
QWEN_BASIS = "reports/probe_features/qwen_residual_basis.npz"
INTERN_BASIS = "reports/probe_features/intern_residual_basis.npz"

# residMulti layer config (3-layer, matches v12 headline cells)
RECIPE = {
    "qwen":   dict(model=QWEN,   layers="13,17,21", basis=QWEN_BASIS),
    "intern": dict(model=INTERN, layers="15,19,23", basis=INTERN_BASIS),
}

LAMBDAS = ["0.3", "1", "3.0"]
SEEDS = [0, 1, 2, 3]

jobs = []
def add(name, cmd):
    jobs.append({"name": name, "cmd": cmd})


def train_cmd(short, lam, seed) -> str:
    r = RECIPE[short]
    ckdir = f"checkpoints/shuffleMulti_{short}_lam{lam}_seed{seed}"
    parts = [
        PY, TRAIN_PY,
        f"--model-id {r['model']}",
        f"--train-jsonl {TRAIN_JSONL}",
        f"--val-jsonl {VAL_JSONL}",
        f"--output-dir {ckdir}",
        f"--layers {r['layers']}",
        f"--lambda-dir {lam}",
        "--tau 2.0",
        "--steps 500",
        "--batch-size 2",
        "--lora-rank 16",
        "--log-every 100",
        f"--seed {seed}",
        f"--residualize-basis {r['basis']}",
        "--lambda-schedule constant",
        "--coord-shuffle",  # << the control
    ]
    return " ".join(parts)


for short in ["qwen", "intern"]:
    for lam in LAMBDAS:
        for seed in SEEDS:
            ckdir = f"checkpoints/shuffleMulti_{short}_lam{lam}_seed{seed}"
            if Path(f"{ckdir}/lora/adapter_model.safetensors").exists():
                continue
            add(f"train_shuffleMulti_{short}_lam{lam}_seed{seed}",
                train_cmd(short, lam, seed))

for short in ["qwen", "intern"]:
    mid = RECIPE[short]["model"]
    for lam in LAMBDAS:
        for seed in SEEDS:
            ck = f"checkpoints/shuffleMulti_{short}_lam{lam}_seed{seed}/lora"
            out = f"reports/vsi_full_eval/shuffleMulti_{short}_lam{lam}_seed{seed}.json"
            if Path(out).exists():
                continue
            cmd = (
                f"{PY} {EVAL_PY} --model-id {mid} --checkpoint {ck} "
                f"--vsi-jsonl {VSI_FULL} --out {out}"
            )
            add(f"vsi_full_shuffleMulti_{short}_lam{lam}_seed{seed}", cmd)


def main():
    qpath = Path("queue_phase12.txt")
    with qpath.open("w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    n_train = sum(1 for j in jobs if j['name'].startswith('train_'))
    n_eval = sum(1 for j in jobs if j['name'].startswith('vsi_'))
    print(f"Wrote {len(jobs)} jobs to {qpath}")
    print(f"  trainings: {n_train}")
    print(f"  evals    : {n_eval}")


if __name__ == "__main__":
    main()
