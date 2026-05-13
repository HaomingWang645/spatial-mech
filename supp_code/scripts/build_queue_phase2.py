#!/usr/bin/env python
"""Append phase-2 jobs to queue.txt: training-data + bandwidth variations."""
from __future__ import annotations

import json
from pathlib import Path

PY = "/home/haoming/miniconda3/envs/vlm-ex/bin/python3"
QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"
INTERN = "OpenGVLab/InternVL3-8B-hf"

jobs = []


def add(name, cmd):
    jobs.append({"name": name, "cmd": cmd})


# =====================================================================
# Phase 5: Bandwidth ablation (Qwen × λ=1 × seed 0..3 with τ ∈ {1.0, 4.0})
# =====================================================================
for tau in ["1.0", "4.0"]:
    for seed in [0, 1, 2, 3]:
        out_dir = f"checkpoints/qwen_lam1_seed{seed}_tau{tau}"
        out_lora = f"{out_dir}/lora/adapter_model.safetensors"
        if Path(out_lora).exists():
            continue
        cmd = (f"{PY} scripts/train_qwen_dirichlet.py "
               f"--model-id {QWEN} "
               f"--train-jsonl data/dirichlet_train_v2/train.jsonl "
               f"--val-jsonl data/dirichlet_train_v2/val_iid.jsonl "
               f"--output-dir {out_dir} "
               f"--layer 17 --tau {tau} --steps 500 --batch-size 2 "
               f"--lora-rank 16 --eval-every 250 --log-every 100 --n-eval 100 "
               f"--lambda-dir 1.0 --seed {seed}")
        add(f"train_qwen_lam1_seed{seed}_tau{tau}", cmd)

# VSI eval bandwidth
for tau in ["1.0", "4.0"]:
    for seed in [0, 1, 2, 3]:
        ck = f"checkpoints/qwen_lam1_seed{seed}_tau{tau}/lora"
        out = f"reports/vsi_eval/qwen_lam1_tau{tau}_seed{seed}.json"
        if Path(out).exists():
            continue
        cmd = (f"{PY} scripts/eval_vsi.py "
               f"--model-id {QWEN} --checkpoint {ck} "
               f"--vsi-jsonl data/dirichlet_train_v2/val_vsi_arkit.jsonl "
               f"--out {out} --mc-only")
        add(f"vsi_qwen_lam1_tau{tau}_seed{seed}", cmd)

# =====================================================================
# Phase 6: Long training (1000 steps, Qwen × λ=1 × seed 0,1)
# =====================================================================
for steps in [1000]:
    for seed in [0, 1]:
        out_dir = f"checkpoints/qwen_lam1_seed{seed}_steps{steps}"
        out_lora = f"{out_dir}/lora/adapter_model.safetensors"
        if Path(out_lora).exists():
            continue
        cmd = (f"{PY} scripts/train_qwen_dirichlet.py "
               f"--model-id {QWEN} "
               f"--train-jsonl data/dirichlet_train_v2/train.jsonl "
               f"--val-jsonl data/dirichlet_train_v2/val_iid.jsonl "
               f"--output-dir {out_dir} "
               f"--layer 17 --tau 2.0 --steps {steps} --batch-size 2 "
               f"--lora-rank 16 --eval-every 500 --log-every 100 --n-eval 100 "
               f"--lambda-dir 1.0 --seed {seed}")
        add(f"train_qwen_lam1_seed{seed}_steps{steps}", cmd)

# VSI eval
for steps in [1000]:
    for seed in [0, 1]:
        ck = f"checkpoints/qwen_lam1_seed{seed}_steps{steps}/lora"
        out = f"reports/vsi_eval/qwen_lam1_steps{steps}_seed{seed}.json"
        if Path(out).exists():
            continue
        cmd = (f"{PY} scripts/eval_vsi.py "
               f"--model-id {QWEN} --checkpoint {ck} "
               f"--vsi-jsonl data/dirichlet_train_v2/val_vsi_arkit.jsonl "
               f"--out {out} --mc-only")
        add(f"vsi_qwen_lam1_steps{steps}_seed{seed}", cmd)

# =====================================================================
# Phase 7: 7Scenes cam-motion for InternVL × λ ∈ {1.0, 3.0} only seeds 0,1 (slow)
# Already have 4 InternVL lam0 + lam1 evals on 7Scenes
# =====================================================================
# Skip — InternVL cam-motion is too slow; already shows null on the 4 done

# =====================================================================
# Phase 8: KITTI cam-motion + InternVL × λ ∈ {0, 1} × seeds 0,1 (slow)
# =====================================================================
# Skip for now

# =====================================================================
# Phase 9: tier_d cam-motion (same as ARKitScenes already in phase 1)
# =====================================================================
# Already in phase 1

# =====================================================================
# Phase 10: Eval all bandwidth/layer/rank ckpts on 7Scenes cam-motion
# =====================================================================
for var, var_dir in [
    ("L13", "L13"), ("L21", "L21"),
    ("r8", "r8"), ("r32", "r32"),
    ("tau1.0", "tau1.0"), ("tau4.0", "tau4.0"),
]:
    for seed in [0, 1, 2, 3]:
        ck = f"checkpoints/qwen_lam1_seed{seed}_{var_dir}/lora"
        out = f"reports/cam_motion_eval/qwen_lam1_{var}_seed{seed}_7scenes.json"
        if Path(out).exists():
            continue
        cmd = (f"{PY} scripts/eval_cam_motion_lora.py "
               f"--model-id {QWEN} --checkpoint {ck} "
               f"--scenes-root data/tier_d_7scenes "
               f"--out {out} --limit 40 --max-frames 8")
        add(f"cam7_qwen_lam1_{var}_seed{seed}", cmd)

# Append to queue
queue_path = Path("queue.txt")
with queue_path.open("a") as f:
    for job in jobs:
        f.write(json.dumps(job) + "\n")
print(f"Appended {len(jobs)} new jobs")
print(f"Queue total now: {sum(1 for _ in queue_path.open())}")
