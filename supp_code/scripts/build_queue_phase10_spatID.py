#!/usr/bin/env python
"""Phase-10: spatial-ID-loss baseline (Kang et al., ICLR 2026, 3D adaptation).

We add Kang et al.'s cosine-similarity-to-spatial-ID auxiliary loss as a
training-side baseline against our Dirichlet penalty. The basis V_ell is
fitted once per (model, layer) on held-out training scenes.

Sweep:
  models    : Qwen2.5-VL-7B, InternVL3-8B
  layer     : L17 (cognitive-map layer for both)
  lambda_sid: {0.3, 1, 3}        (matches our Dirichlet sweep)
  seeds     : {0, 1, 2, 3}       (n=4)

Total jobs:
  2 (models) * 3 (lambdas) * 4 (seeds) = 24 trainings
  + 24 full-VSI evals
  = 48 jobs, runs across 8 GPUs.
"""
from __future__ import annotations
import json
from pathlib import Path

PY = "/home/haoming/miniconda3/envs/vlm-ex/bin/python3"
QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"
INTERN = "OpenGVLab/InternVL3-8B-hf"
TRAIN_PY = "scripts/train_qwen_spatID.py"
EVAL_PY = "scripts/eval_vsi_batched.py"
TRAIN_JSONL = "data/dirichlet_train_v2/train.jsonl"
VAL_JSONL = "data/dirichlet_train_v2/val_iid.jsonl"
VSI_FULL = "data/vsi_bench_full/eval_full.jsonl"

jobs = []
def add(name, cmd):
    jobs.append({"name": name, "cmd": cmd})


# --------------------------------------------------------------------
# Training
# --------------------------------------------------------------------
for short, mid in [("qwen", QWEN), ("intern", INTERN)]:
    for lam in ["0.3", "1", "3.0"]:
        for seed in [0, 1, 2, 3]:
            ckdir = f"checkpoints/spatID_{short}_lam{lam}_seed{seed}"
            if Path(f"{ckdir}/lora/adapter_model.safetensors").exists():
                continue  # already trained
            cmd = (
                f"{PY} {TRAIN_PY} --model-id {mid} "
                f"--train-jsonl {TRAIN_JSONL} --val-jsonl {VAL_JSONL} "
                f"--output-dir {ckdir} "
                f"--layers 17 --steps 500 --batch-size 2 --lora-rank 16 "
                f"--n-basis 80 --log-every 100 "
                f"--lambda-sid {lam} --seed {seed}"
            )
            add(f"train_spatID_{short}_lam{lam}_seed{seed}", cmd)

# --------------------------------------------------------------------
# Full-VSI evaluation
# --------------------------------------------------------------------
for short, mid in [("qwen", QWEN), ("intern", INTERN)]:
    for lam in ["0.3", "1", "3.0"]:
        for seed in [0, 1, 2, 3]:
            ck = f"checkpoints/spatID_{short}_lam{lam}_seed{seed}/lora"
            out = f"reports/vsi_full_eval/spatID_{short}_lam{lam}_seed{seed}.json"
            if Path(out).exists():
                continue
            cmd = (
                f"{PY} {EVAL_PY} --model-id {mid} --checkpoint {ck} "
                f"--vsi-jsonl {VSI_FULL} --out {out}"
            )
            add(f"vsi_full_spatID_{short}_lam{lam}_seed{seed}", cmd)


def main():
    qpath = Path("queue_phase10.txt")
    with qpath.open("w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"Wrote {len(jobs)} jobs to {qpath}")
    print(f"  trainings: {sum(1 for j in jobs if j['name'].startswith('train_'))}")
    print(f"  evals    : {sum(1 for j in jobs if j['name'].startswith('vsi_'))}")


if __name__ == "__main__":
    main()
