#!/usr/bin/env python
"""Phase-12: extend Dirichlet finetuning to LLaVA-OneVision-7B.

Closes the architecture-symmetry between Tier C probing (3 models) and
Dirichlet finetuning (currently 2 models). LLaVA-OneVision uses a Qwen2-7B
LM backbone and the v2 training script's hardcoded layer path
(model.language_model.layers) works unchanged for it (verified via
/tmp/test_llava_path.py).

Two phases:

  12a. Basis prep (single job, sequential dependency):
       - Extract LLaVA-OV L17 activations on val_iid scenes
         -> reports/probe_features/llavaov_base.npz
       - Fit color+shape probe basis -> orthogonalize -> save
         reports/probe_features/llavaov_residual_basis.npz

  12b. Training + eval sweep (24 jobs):
       baseline (LoRA lam=0)        x 4 seeds = 4 trainings
       residMulti lam in {0.3, 1, 3} x 4 seeds = 12 trainings
       (eval on full VSI-Bench)               = 16 evals
       Phase 12b's queue is written to queue_phase12b.txt and is
       appended to queue_phase10.txt only AFTER 12a completes.
"""
from __future__ import annotations
import json
from pathlib import Path

PY = "/home/haoming/miniconda3/envs/vlm-ex/bin/python3"
LLAVAOV = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
TRAIN_PY = "scripts/train_qwen_dirichlet_v2.py"
EVAL_PY = "scripts/eval_vsi_batched.py"
EXTRACT_PY = "scripts/extract_lora_features.py"
BASIS_PY = "scripts/build_residualization_basis.py"
TRAIN_JSONL = "data/dirichlet_train_v2/train.jsonl"
VAL_JSONL = "data/dirichlet_train_v2/val_iid.jsonl"
VSI_FULL = "data/vsi_bench_full/eval_full.jsonl"
BASE_NPZ = "reports/probe_features/llavaov_base.npz"
BASIS_NPZ = "reports/probe_features/llavaov_residual_basis.npz"


# ----- 12a. prep -----
prep_cmd = (
    f"{PY} {EXTRACT_PY} --model-id {LLAVAOV} "
    f"--val-jsonl {VAL_JSONL} --layer 17 --out {BASE_NPZ} "
    f"&& {PY} {BASIS_PY} --probe-npz {BASE_NPZ} --out {BASIS_NPZ}"
)
prep_job = {"name": "prep_llavaov_basis", "cmd": prep_cmd}


# ----- 12b. training & eval sweep -----
b_jobs = []


def add_b(name, cmd):
    b_jobs.append({"name": name, "cmd": cmd})


# baseline (LoRA lam=0): no Dirichlet, no residualization basis required at runtime
for seed in [0, 1, 2, 3]:
    ckdir = f"checkpoints/llavaov_lam0_seed{seed}"
    if Path(f"{ckdir}/lora/adapter_model.safetensors").exists():
        continue
    cmd = (
        f"{PY} {TRAIN_PY} --model-id {LLAVAOV} "
        f"--train-jsonl {TRAIN_JSONL} --val-jsonl {VAL_JSONL} "
        f"--output-dir {ckdir} "
        f"--layers 17 --lambda-dir 0 --tau 2.0 "
        f"--steps 500 --batch-size 2 --lora-rank 16 "
        f"--log-every 100 --seed {seed} "
        f"--lambda-schedule constant"
    )
    add_b(f"train_llavaov_lam0_seed{seed}", cmd)

# residMulti at 3 lambdas (3-layer 13/17/21, residualized)
for lam in ["0.3", "1", "3.0"]:
    for seed in [0, 1, 2, 3]:
        ckdir = f"checkpoints/residMulti_llavaov_lam{lam}_seed{seed}"
        if Path(f"{ckdir}/lora/adapter_model.safetensors").exists():
            continue
        cmd = (
            f"{PY} {TRAIN_PY} --model-id {LLAVAOV} "
            f"--train-jsonl {TRAIN_JSONL} --val-jsonl {VAL_JSONL} "
            f"--output-dir {ckdir} "
            f"--layers 13,17,21 --lambda-dir {lam} --tau 2.0 "
            f"--steps 500 --batch-size 2 --lora-rank 16 "
            f"--log-every 100 --seed {seed} "
            f"--residualize-basis {BASIS_NPZ} "
            f"--lambda-schedule constant"
        )
        add_b(f"train_residMulti_llavaov_lam{lam}_seed{seed}", cmd)

# Evals
for seed in [0, 1, 2, 3]:
    ck = f"checkpoints/llavaov_lam0_seed{seed}/lora"
    out = f"reports/vsi_full_eval/llavaov_lam0_seed{seed}.json"
    if Path(out).exists():
        continue
    cmd = (
        f"{PY} {EVAL_PY} --model-id {LLAVAOV} --checkpoint {ck} "
        f"--vsi-jsonl {VSI_FULL} --out {out}"
    )
    add_b(f"vsi_full_llavaov_lam0_seed{seed}", cmd)

for lam in ["0.3", "1", "3.0"]:
    for seed in [0, 1, 2, 3]:
        ck = f"checkpoints/residMulti_llavaov_lam{lam}_seed{seed}/lora"
        out = f"reports/vsi_full_eval/residMulti_llavaov_lam{lam}_seed{seed}.json"
        if Path(out).exists():
            continue
        cmd = (
            f"{PY} {EVAL_PY} --model-id {LLAVAOV} --checkpoint {ck} "
            f"--vsi-jsonl {VSI_FULL} --out {out}"
        )
        add_b(f"vsi_full_residMulti_llavaov_lam{lam}_seed{seed}", cmd)


def main():
    # 12a: write prep job to its own file (one job)
    with open("queue_phase12a_prep.txt", "w") as f:
        f.write(json.dumps(prep_job) + "\n")
    # 12b: write training+eval queue to its own file (NOT yet chained)
    with open("queue_phase12b.txt", "w") as f:
        for j in b_jobs:
            f.write(json.dumps(j) + "\n")
    print(f"Phase 12a (prep): 1 job -> queue_phase12a_prep.txt")
    print(f"Phase 12b (train+eval): {len(b_jobs)} jobs -> queue_phase12b.txt")
    print(f"  trainings: {sum(1 for j in b_jobs if j['name'].startswith('train_'))}")
    print(f"  evals    : {sum(1 for j in b_jobs if j['name'].startswith('vsi_'))}")


if __name__ == "__main__":
    main()
