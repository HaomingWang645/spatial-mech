#!/usr/bin/env python
"""Build a long experiment queue from a structured spec.

Writes JSONL to queue.txt that the runner consumes.
"""
from __future__ import annotations

import json
from pathlib import Path

PY = "/home/haoming/miniconda3/envs/vlm-ex/bin/python3"
QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"
INTERN = "OpenGVLab/InternVL3-8B-hf"

jobs = []


def add_job(name: str, cmd: str):
    """Skip if already done (output file exists)."""
    jobs.append({"name": name, "cmd": cmd})


# =====================================================================
# Phase 1: Cam-motion VQA on different real-world video datasets
# =====================================================================

# 7Scenes already done for Qwen + zero-shot. Now add KITTI for Qwen.
for lam in ["0", "0.3", "1", "3.0"]:
    for seed in [0, 1, 2, 3]:
        out = f"reports/cam_motion_eval/qwen_lam{lam}_seed{seed}_kitti.json"
        if Path(out).exists():
            continue
        ck = f"checkpoints/qwen_lam{lam}_seed{seed}/lora"
        cmd = (f"{PY} scripts/eval_cam_motion_lora.py "
               f"--model-id {QWEN} --checkpoint {ck} "
               f"--scenes-root data/tier_d_kitti "
               f"--out {out} --limit 40 --max-frames 8")
        add_job(f"kitti_qwen_lam{lam}_seed{seed}", cmd)

# Zero-shot KITTI
out = "reports/cam_motion_eval/qwen_zeroshot_kitti.json"
if not Path(out).exists():
    add_job("kitti_qwen_zeroshot",
            f"{PY} scripts/eval_cam_motion_lora.py --model-id {QWEN} "
            f"--scenes-root data/tier_d_kitti --out {out} --limit 40 --max-frames 8")

# Cam-motion on tier_d (ARKitScenes) — different scenes from VSI-Bench/tier_d we use for VSI eval
# Actually, our VSI-Bench eval uses single-frame from tier_d; this would be 8-frame from tier_d.
for lam in ["0", "0.3", "1", "3.0"]:
    for seed in [0, 1, 2, 3]:
        out = f"reports/cam_motion_eval/qwen_lam{lam}_seed{seed}_arkit.json"
        if Path(out).exists():
            continue
        ck = f"checkpoints/qwen_lam{lam}_seed{seed}/lora"
        cmd = (f"{PY} scripts/eval_cam_motion_lora.py "
               f"--model-id {QWEN} --checkpoint {ck} "
               f"--scenes-root data/tier_d "
               f"--out {out} --limit 40 --max-frames 8")
        add_job(f"arkit_qwen_lam{lam}_seed{seed}", cmd)

out = "reports/cam_motion_eval/qwen_zeroshot_arkit.json"
if not Path(out).exists():
    add_job("arkit_qwen_zeroshot",
            f"{PY} scripts/eval_cam_motion_lora.py --model-id {QWEN} "
            f"--scenes-root data/tier_d --out {out} --limit 40 --max-frames 8")

# =====================================================================
# Phase 2: Probe feature extraction for all seeds × all lambdas × both models
# Currently only seed 0 done for 4 conditions (qwen lam0/1, intern lam0/1).
# Need: 4 seeds × 4 lambdas × 2 models = 32, minus 4 done = 28
# =====================================================================
for model_short, model_id in [("qwen", QWEN), ("intern", INTERN)]:
    for lam in ["0", "0.3", "1", "3.0"]:
        for seed in [0, 1, 2, 3]:
            out = f"reports/probe_features/{model_short}_lam{lam}_seed{seed}.npz"
            if Path(out).exists():
                continue
            ck = f"checkpoints/{model_short}_lam{lam}_seed{seed}/lora"
            cmd = (f"{PY} scripts/extract_lora_features.py "
                   f"--model-id {model_id} --checkpoint {ck} "
                   f"--val-jsonl data/dirichlet_train_v2/val_iid.jsonl "
                   f"--layer 17 --out {out}")
            add_job(f"probe_{model_short}_lam{lam}_seed{seed}", cmd)

# =====================================================================
# Phase 3: Layer-ablation training (Qwen × λ=1 × seed 0..3 at different layers)
# Compare L13 (mid-early), L17 (peak baseline), L21 (late)
# =====================================================================
for layer in [13, 21]:
    for seed in [0, 1, 2, 3]:
        out_dir = f"checkpoints/qwen_lam1_seed{seed}_L{layer}"
        out_lora = f"{out_dir}/lora/adapter_model.safetensors"
        if Path(out_lora).exists():
            continue
        cmd = (f"{PY} scripts/train_qwen_dirichlet.py "
               f"--model-id {QWEN} "
               f"--train-jsonl data/dirichlet_train_v2/train.jsonl "
               f"--val-jsonl data/dirichlet_train_v2/val_iid.jsonl "
               f"--output-dir {out_dir} "
               f"--layer {layer} --tau 2.0 --steps 500 --batch-size 2 "
               f"--lora-rank 16 --eval-every 250 --log-every 100 --n-eval 100 "
               f"--lambda-dir 1.0 --seed {seed}")
        add_job(f"train_qwen_lam1_seed{seed}_L{layer}", cmd)

# Then VSI eval those new layer-ablation checkpoints
for layer in [13, 21]:
    for seed in [0, 1, 2, 3]:
        ck = f"checkpoints/qwen_lam1_seed{seed}_L{layer}/lora"
        out = f"reports/vsi_eval/qwen_lam1_L{layer}_seed{seed}.json"
        if Path(out).exists():
            continue
        cmd = (f"{PY} scripts/eval_vsi.py "
               f"--model-id {QWEN} --checkpoint {ck} "
               f"--vsi-jsonl data/dirichlet_train_v2/val_vsi_arkit.jsonl "
               f"--out {out} --mc-only")
        add_job(f"vsi_qwen_lam1_L{layer}_seed{seed}", cmd)

# =====================================================================
# Phase 4: LoRA-rank ablation
# =====================================================================
for rank in [8, 32]:
    for seed in [0, 1, 2, 3]:
        out_dir = f"checkpoints/qwen_lam1_seed{seed}_r{rank}"
        out_lora = f"{out_dir}/lora/adapter_model.safetensors"
        if Path(out_lora).exists():
            continue
        cmd = (f"{PY} scripts/train_qwen_dirichlet.py "
               f"--model-id {QWEN} "
               f"--train-jsonl data/dirichlet_train_v2/train.jsonl "
               f"--val-jsonl data/dirichlet_train_v2/val_iid.jsonl "
               f"--output-dir {out_dir} "
               f"--layer 17 --tau 2.0 --steps 500 --batch-size 2 "
               f"--lora-rank {rank} --eval-every 250 --log-every 100 --n-eval 100 "
               f"--lambda-dir 1.0 --seed {seed}")
        add_job(f"train_qwen_lam1_seed{seed}_r{rank}", cmd)

# VSI eval rank-ablation
for rank in [8, 32]:
    for seed in [0, 1, 2, 3]:
        ck = f"checkpoints/qwen_lam1_seed{seed}_r{rank}/lora"
        out = f"reports/vsi_eval/qwen_lam1_r{rank}_seed{seed}.json"
        if Path(out).exists():
            continue
        cmd = (f"{PY} scripts/eval_vsi.py "
               f"--model-id {QWEN} --checkpoint {ck} "
               f"--vsi-jsonl data/dirichlet_train_v2/val_vsi_arkit.jsonl "
               f"--out {out} --mc-only")
        add_job(f"vsi_qwen_lam1_r{rank}_seed{seed}", cmd)

# Write queue file
queue_path = Path("queue.txt")
with queue_path.open("w") as f:
    for job in jobs:
        f.write(json.dumps(job) + "\n")

print(f"Wrote {len(jobs)} jobs to {queue_path}")
# Summary by type
from collections import Counter
prefixes = Counter(job["name"].split("_")[0] for job in jobs)
print("By prefix:")
for k, v in prefixes.most_common():
    print(f"  {k}: {v}")
