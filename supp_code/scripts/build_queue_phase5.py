#!/usr/bin/env python
"""Phase-5 queue: aggressive variation sweep for additional improvements.

Four experimental streams, each at 2 models × 2 seeds:
  A. Online residualization (refit W every 100 steps)
  B. Multi-layer Dirichlet (L13+L17+L21 simultaneously)
  C. Combined: multi-layer + online residualization (the union)
  D. λ schedule (warmup then anneal)
  E. Higher-λ regime (λ ∈ {5, 10})
  F. MLP LoRA targets (q,k,v,o + gate,up,down)

For each stream: train + eval on VSI MC + MindCube + ViewSpatial + OST.
"""
from __future__ import annotations
import json
from pathlib import Path

PY = "/home/haoming/miniconda3/envs/vlm-ex/bin/python3"
QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"
INTERN = "OpenGVLab/InternVL3-8B-hf"
TRAIN_PY = "scripts/train_qwen_dirichlet_v2.py"
SEEDS = [0, 1]
LAMBDAS = ["1", "3.0"]  # focus on the regimes that v8/v9 showed work best
MODELS = [("qwen", QWEN), ("intern", INTERN)]

jobs = []


def add(name, cmd):
    jobs.append({"name": name, "cmd": cmd})


def basis_arg(short):
    return f"--residualize-basis reports/probe_features/{short}_residual_basis.npz"


# ============================================================================
# Stream A: Online residualization (refit every 100 steps, default everything else)
# ============================================================================
for short, mid in MODELS:
    for lam in LAMBDAS:
        for seed in SEEDS:
            tag = f"online_{short}_lam{lam}_seed{seed}"
            ckdir = f"checkpoints/{tag}"
            ckpath = f"{ckdir}/lora/adapter_model.safetensors"
            if Path(ckpath).exists(): continue
            cmd = (
                f"{PY} {TRAIN_PY} --model-id {mid} "
                f"--train-jsonl data/dirichlet_train_v2/train.jsonl "
                f"--val-jsonl data/dirichlet_train_v2/val_iid.jsonl "
                f"--output-dir {ckdir} "
                f"--layers 17 --tau 2.0 --steps 500 --batch-size 2 --lora-rank 16 "
                f"--eval-every 250 --log-every 100 --n-eval 100 "
                f"--lambda-dir {lam} --seed {seed} "
                f"{basis_arg(short)} --online-residualize-every 100"
            )
            add(f"train_{tag}", cmd)

# ============================================================================
# Stream B: Multi-layer Dirichlet (no residualization)
# ============================================================================
for short, mid in MODELS:
    for lam in LAMBDAS:
        for seed in SEEDS:
            tag = f"multilayer_{short}_lam{lam}_seed{seed}"
            ckdir = f"checkpoints/{tag}"
            ckpath = f"{ckdir}/lora/adapter_model.safetensors"
            if Path(ckpath).exists(): continue
            cmd = (
                f"{PY} {TRAIN_PY} --model-id {mid} "
                f"--train-jsonl data/dirichlet_train_v2/train.jsonl "
                f"--val-jsonl data/dirichlet_train_v2/val_iid.jsonl "
                f"--output-dir {ckdir} "
                f"--layers 13,17,21 --tau 2.0 --steps 500 --batch-size 2 --lora-rank 16 "
                f"--eval-every 250 --log-every 100 --n-eval 100 "
                f"--lambda-dir {lam} --seed {seed}"
            )
            add(f"train_{tag}", cmd)

# ============================================================================
# Stream C: Combined: multi-layer + online residualization
# ============================================================================
for short, mid in MODELS:
    for lam in LAMBDAS:
        for seed in SEEDS:
            tag = f"combined_{short}_lam{lam}_seed{seed}"
            ckdir = f"checkpoints/{tag}"
            ckpath = f"{ckdir}/lora/adapter_model.safetensors"
            if Path(ckpath).exists(): continue
            cmd = (
                f"{PY} {TRAIN_PY} --model-id {mid} "
                f"--train-jsonl data/dirichlet_train_v2/train.jsonl "
                f"--val-jsonl data/dirichlet_train_v2/val_iid.jsonl "
                f"--output-dir {ckdir} "
                f"--layers 13,17,21 --tau 2.0 --steps 500 --batch-size 2 --lora-rank 16 "
                f"--eval-every 250 --log-every 100 --n-eval 100 "
                f"--lambda-dir {lam} --seed {seed} "
                f"{basis_arg(short)} --online-residualize-every 100"
            )
            add(f"train_{tag}", cmd)

# ============================================================================
# Stream D: λ-schedule (warmup_anneal)
# ============================================================================
for short, mid in MODELS:
    for lam in LAMBDAS:
        for seed in SEEDS:
            tag = f"sched_{short}_lam{lam}_seed{seed}"
            ckdir = f"checkpoints/{tag}"
            ckpath = f"{ckdir}/lora/adapter_model.safetensors"
            if Path(ckpath).exists(): continue
            cmd = (
                f"{PY} {TRAIN_PY} --model-id {mid} "
                f"--train-jsonl data/dirichlet_train_v2/train.jsonl "
                f"--val-jsonl data/dirichlet_train_v2/val_iid.jsonl "
                f"--output-dir {ckdir} "
                f"--layers 17 --tau 2.0 --steps 500 --batch-size 2 --lora-rank 16 "
                f"--eval-every 250 --log-every 100 --n-eval 100 "
                f"--lambda-dir {lam} --seed {seed} "
                f"--lambda-schedule warmup_anneal --warmup-steps 50 --anneal-steps 100"
            )
            add(f"train_{tag}", cmd)

# ============================================================================
# Stream E: Higher-λ regime (λ ∈ {5, 10}) - test where linear regime breaks
# ============================================================================
for short, mid in MODELS:
    for lam in ["5.0", "10.0"]:
        for seed in SEEDS:
            tag = f"highlam_{short}_lam{lam}_seed{seed}"
            ckdir = f"checkpoints/{tag}"
            ckpath = f"{ckdir}/lora/adapter_model.safetensors"
            if Path(ckpath).exists(): continue
            cmd = (
                f"{PY} {TRAIN_PY} --model-id {mid} "
                f"--train-jsonl data/dirichlet_train_v2/train.jsonl "
                f"--val-jsonl data/dirichlet_train_v2/val_iid.jsonl "
                f"--output-dir {ckdir} "
                f"--layers 17 --tau 2.0 --steps 500 --batch-size 2 --lora-rank 16 "
                f"--eval-every 250 --log-every 100 --n-eval 100 "
                f"--lambda-dir {lam} --seed {seed}"
            )
            add(f"train_{tag}", cmd)

# ============================================================================
# Stream F: MLP LoRA targets (Qwen only — InternVL has different MLP names)
# ============================================================================
for lam in LAMBDAS:
    for seed in SEEDS:
        tag = f"mlp_qwen_lam{lam}_seed{seed}"
        ckdir = f"checkpoints/{tag}"
        ckpath = f"{ckdir}/lora/adapter_model.safetensors"
        if Path(ckpath).exists(): continue
        cmd = (
            f"{PY} {TRAIN_PY} --model-id {QWEN} "
            f"--train-jsonl data/dirichlet_train_v2/train.jsonl "
            f"--val-jsonl data/dirichlet_train_v2/val_iid.jsonl "
            f"--output-dir {ckdir} "
            f"--layers 17 --tau 2.0 --steps 500 --batch-size 2 --lora-rank 16 "
            f"--eval-every 250 --log-every 100 --n-eval 100 "
            f"--lambda-dir {lam} --seed {seed} "
            f"--lora-targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        )
        add(f"train_{tag}", cmd)

# ============================================================================
# Eval suite for each new checkpoint: VSI MC + MindCube + ViewSpatial + OST
# ============================================================================
EVAL_SETS = [
    ("vsi", "scripts/eval_vsi_batched.py", "vsi_eval", [
        "--vsi-jsonl data/dirichlet_train_v2/val_vsi_arkit.jsonl --mc-only"
    ]),
    ("mc", "scripts/eval_mindcube.py", "mindcube_eval", [
        "--mindcube-jsonl /home/haoming/mindcube_data/raw/MindCube_tinybench.jsonl",
        "--image-root /home/haoming/mindcube_data",
    ]),
    ("vs", "scripts/eval_viewspatial.py", "viewspatial_eval", [
        "--bench-json data/viewspatial_bench/ViewSpatial-Bench.json",
        "--image-root data/viewspatial_bench",
        "--n-eval 500 --seed 0",
    ]),
    ("ost", "scripts/eval_ost_bench.py", "ost_eval", [
        "--bench-json data/ost_bench/OST_bench.json",
        "--image-root data/ost_bench/image_upload",
        "--n-eval 500 --seed 0",
    ]),
]

# We'll use checkpoint dirs that match our trained ones
def model_id_for(short):
    return QWEN if short == "qwen" else INTERN

new_train_jobs = list(jobs)  # copy: only training jobs so far
for tj in new_train_jobs:
    # Extract checkpoint dir from training cmd
    cmd_parts = tj["cmd"].split()
    out_dir = None
    for i, p in enumerate(cmd_parts):
        if p == "--output-dir":
            out_dir = cmd_parts[i+1]
            break
    if not out_dir: continue
    ck = f"{out_dir}/lora"
    # name structure: train_{tag}; we strip "train_"
    base_tag = tj["name"][len("train_"):]
    short = "qwen" if "_qwen_" in base_tag else "intern"
    mid = model_id_for(short)
    for eval_short, script, out_dir_name, extra_args in EVAL_SETS:
        out = f"reports/{out_dir_name}/{base_tag}.json"
        if Path(out).exists(): continue
        extra = " ".join(extra_args)
        cmd = (
            f"{PY} {script} --model-id {mid} --checkpoint {ck} "
            f"--out {out} {extra}"
        )
        add(f"{eval_short}_{base_tag}", cmd)

# ============================================================================
# Append to BOTH queue files for parallel processing
# ============================================================================
with open("queue.txt", "a") as f:
    for j in jobs[::2]: f.write(json.dumps(j) + "\n")
with open("queue_b.txt", "a") as f:
    for j in jobs[1::2]: f.write(json.dumps(j) + "\n")

from collections import Counter
prefixes = Counter(j["name"].split("_")[0] for j in jobs)
print(f"Phase-5 queue: {len(jobs)} jobs total")
print(f"  Splits: queue.txt {len(jobs[::2])}, queue_b.txt {len(jobs[1::2])}")
print("By prefix:", dict(prefixes))
