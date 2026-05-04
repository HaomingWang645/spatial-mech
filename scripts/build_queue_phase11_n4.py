#!/usr/bin/env python
"""Phase-11: complete n=4 validation of v12 cells still at n=2.

After v12, six cells were flagged as preliminary (n=2 only). The
project's track record (3 of 5 large-effect n=2 cells failed at n=4)
makes lifting these to n=4 the highest-priority compute item.

Cells (all at full VSI-Bench, n=4 once seeds 2 & 3 land):
  - residMulti5L  Qwen   lambda=1
  - residMulti5L  Qwen   lambda=3
  - residMulti5L  Intern lambda=3
  - residMultiSched Qwen lambda=1
  - residMultiSched Intern lambda=1
  - residMultiSched Intern lambda=3

Each: seeds 2 & 3 only (0 & 1 already trained + evaluated).

Total jobs: 6 cells * 2 seeds * 2 (train + eval) = 24 jobs.
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

# Cell recipes from v12 history.json files.
# residMulti5L (5-layer Dirichlet, residualized, constant schedule):
#   Qwen   layers = 9,13,17,21,25 ; basis = qwen_residual_basis
#   Intern layers = 11,15,19,23,27 ; basis = intern_residual_basis
# residMultiSched (3-layer Dirichlet, residualized, warmup_anneal):
#   Qwen   layers = 13,17,21
#   Intern layers = 15,19,23
RECIPES = {
    ("residMulti5L", "qwen"):   dict(model=QWEN,   layers="9,13,17,21,25",  basis=QWEN_BASIS,   schedule="constant"),
    ("residMulti5L", "intern"): dict(model=INTERN, layers="11,15,19,23,27", basis=INTERN_BASIS, schedule="constant"),
    ("residMultiSched", "qwen"):   dict(model=QWEN,   layers="13,17,21", basis=QWEN_BASIS,   schedule="warmup_anneal"),
    ("residMultiSched", "intern"): dict(model=INTERN, layers="15,19,23", basis=INTERN_BASIS, schedule="warmup_anneal"),
}

# (stream, model, lambda) cells at n=2 to lift to n=4
CELLS = [
    ("residMulti5L",   "qwen",   "1"),
    ("residMulti5L",   "qwen",   "3.0"),
    ("residMulti5L",   "intern", "3.0"),
    ("residMultiSched","qwen",   "1"),
    ("residMultiSched","intern", "1"),
    ("residMultiSched","intern", "3.0"),
]
SEEDS_NEEDED = [2, 3]

jobs = []
def add(name, cmd):
    jobs.append({"name": name, "cmd": cmd})


def train_cmd(stream, short, lam, seed) -> str:
    r = RECIPES[(stream, short)]
    ckdir = f"checkpoints/{stream}_{short}_lam{lam}_seed{seed}"
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
        f"--lambda-schedule {r['schedule']}",
    ]
    if r["schedule"] == "warmup_anneal":
        parts += ["--warmup-steps 50", "--anneal-steps 100"]
    return " ".join(parts)


# Trainings
for stream, short, lam in CELLS:
    for seed in SEEDS_NEEDED:
        ckdir = f"checkpoints/{stream}_{short}_lam{lam}_seed{seed}"
        if Path(f"{ckdir}/lora/adapter_model.safetensors").exists():
            continue
        add(f"train_{stream}_{short}_lam{lam}_seed{seed}", train_cmd(stream, short, lam, seed))

# Evals
for stream, short, lam in CELLS:
    mid = RECIPES[(stream, short)]["model"]
    for seed in SEEDS_NEEDED:
        ck = f"checkpoints/{stream}_{short}_lam{lam}_seed{seed}/lora"
        out = f"reports/vsi_full_eval/{stream}_{short}_lam{lam}_seed{seed}.json"
        if Path(out).exists():
            continue
        cmd = (
            f"{PY} {EVAL_PY} --model-id {mid} --checkpoint {ck} "
            f"--vsi-jsonl {VSI_FULL} --out {out}"
        )
        add(f"vsi_full_{stream}_{short}_lam{lam}_seed{seed}", cmd)


def main():
    qpath = Path("queue_phase11.txt")
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
