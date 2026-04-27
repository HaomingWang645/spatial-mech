#!/usr/bin/env python
"""Phase-4 queue: train + eval residualized-Dirichlet LoRAs.

For each (model, λ, seed): train a NEW LoRA where the Dirichlet loss
projects onto the orthogonal complement of the color+shape probe span
before computing energy. Then evaluate on the same suite as v5/v6/v7:

  - VSI-Bench MC subset (132 items, fast)
  - MindCube tinybench (1050)
  - ViewSpatial-Bench (500)
  - OST-Bench (500)

Compute budget: 16 LoRAs (2 models × 4 λ × 2 seeds — limited seeds for speed),
plus 16 × 4 = 64 evals. Total ≈ 16 × 7 min train + 64 × 5 min eval ≈ 7 GPU-h.
"""
from __future__ import annotations

import json
from pathlib import Path

PY = "/home/haoming/miniconda3/envs/vlm-ex/bin/python3"
QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"
INTERN = "OpenGVLab/InternVL3-8B-hf"

LAMBDAS = ["0", "0.3", "1", "3.0"]
SEEDS = [0, 1]
MODELS = [("qwen", QWEN), ("intern", INTERN)]

jobs = []


def add(name, cmd):
    jobs.append({"name": name, "cmd": cmd})


# ----- Training -----
for short, mid in MODELS:
    basis = f"reports/probe_features/{short}_residual_basis.npz"
    for lam in LAMBDAS:
        for seed in SEEDS:
            out_dir = f"checkpoints/{short}_lam{lam}_seed{seed}_resid"
            out_lora = f"{out_dir}/lora/adapter_model.safetensors"
            if Path(out_lora).exists():
                continue
            cmd = (
                f"{PY} scripts/train_qwen_dirichlet.py "
                f"--model-id {mid} "
                f"--train-jsonl data/dirichlet_train_v2/train.jsonl "
                f"--val-jsonl data/dirichlet_train_v2/val_iid.jsonl "
                f"--output-dir {out_dir} "
                f"--layer 17 --tau 2.0 --steps 500 --batch-size 2 "
                f"--lora-rank 16 --eval-every 250 --log-every 100 --n-eval 100 "
                f"--lambda-dir {lam} --seed {seed} "
                f"--residualize-basis {basis}"
            )
            add(f"train_resid_{short}_lam{lam}_seed{seed}", cmd)

# ----- VSI-Bench MC eval (132 items, the v5/v6 testbed) -----
for short, mid in MODELS:
    for lam in LAMBDAS:
        for seed in SEEDS:
            ck = f"checkpoints/{short}_lam{lam}_seed{seed}_resid/lora"
            out = f"reports/vsi_eval/{short}_lam{lam}_seed{seed}_resid.json"
            if Path(out).exists():
                continue
            cmd = (
                f"{PY} scripts/eval_vsi_batched.py "
                f"--model-id {mid} --checkpoint {ck} "
                f"--vsi-jsonl data/dirichlet_train_v2/val_vsi_arkit.jsonl "
                f"--out {out} --mc-only"
            )
            add(f"vsi_resid_{short}_lam{lam}_seed{seed}", cmd)

# ----- MindCube tinybench -----
for short, mid in MODELS:
    for lam in LAMBDAS:
        for seed in SEEDS:
            ck = f"checkpoints/{short}_lam{lam}_seed{seed}_resid/lora"
            out = f"reports/mindcube_eval/{short}_lam{lam}_seed{seed}_resid.json"
            if Path(out).exists():
                continue
            cmd = (
                f"{PY} scripts/eval_mindcube.py "
                f"--model-id {mid} --checkpoint {ck} "
                f"--mindcube-jsonl /home/haoming/mindcube_data/raw/MindCube_tinybench.jsonl "
                f"--image-root /home/haoming/mindcube_data --out {out}"
            )
            add(f"mc_resid_{short}_lam{lam}_seed{seed}", cmd)

# ----- ViewSpatial-Bench -----
for short, mid in MODELS:
    for lam in LAMBDAS:
        for seed in SEEDS:
            ck = f"checkpoints/{short}_lam{lam}_seed{seed}_resid/lora"
            out = f"reports/viewspatial_eval/{short}_lam{lam}_seed{seed}_resid.json"
            if Path(out).exists():
                continue
            cmd = (
                f"{PY} scripts/eval_viewspatial.py "
                f"--model-id {mid} --checkpoint {ck} "
                f"--bench-json data/viewspatial_bench/ViewSpatial-Bench.json "
                f"--image-root data/viewspatial_bench "
                f"--out {out} --n-eval 500 --seed 0"
            )
            add(f"vs_resid_{short}_lam{lam}_seed{seed}", cmd)

# ----- OST-Bench -----
for short, mid in MODELS:
    for lam in LAMBDAS:
        for seed in SEEDS:
            ck = f"checkpoints/{short}_lam{lam}_seed{seed}_resid/lora"
            out = f"reports/ost_eval/{short}_lam{lam}_seed{seed}_resid.json"
            if Path(out).exists():
                continue
            cmd = (
                f"{PY} scripts/eval_ost_bench.py "
                f"--model-id {mid} --checkpoint {ck} "
                f"--bench-json data/ost_bench/OST_bench.json "
                f"--image-root data/ost_bench/image_upload "
                f"--out {out} --n-eval 500 --seed 0"
            )
            add(f"ost_resid_{short}_lam{lam}_seed{seed}", cmd)

# Append to queue.txt
queue_path = Path("queue.txt")
with queue_path.open("a") as f:
    for job in jobs:
        f.write(json.dumps(job) + "\n")
print(f"Appended {len(jobs)} residualized-training/eval jobs")
from collections import Counter
prefixes = Counter(job["name"].split("_")[0] + "_" + job["name"].split("_")[1]
                    for job in jobs)
print("By prefix:")
for k, v in prefixes.most_common():
    print(f"  {k}: {v}")
