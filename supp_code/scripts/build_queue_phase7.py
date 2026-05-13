#!/usr/bin/env python
"""Phase-7: 24-hour aggressive sweep.

Strategy after phase-6 disconfirmed multilayer:
  A. Multi-seed full-VSI validation of single-layer L17 baseline (we only had seed=0)
  B. Multi-seed full-VSI validation of residualized (we only had seeds 0,1)
  C. New single-layer ablations: L11, L21 (full bench, 4 seeds)
  D. Larger LoRA rank: r=32, r=64 (single layer L17, 2 seeds)
  E. λ-schedule full-bench at 4 seeds
  F. Longer training: 1000-step runs at single layer L17, 4 seeds, full bench
"""
from __future__ import annotations
import json
from pathlib import Path

PY = "/home/haoming/miniconda3/envs/vlm-ex/bin/python3"
QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"
INTERN = "OpenGVLab/InternVL3-8B-hf"
TRAIN_PY = "scripts/train_qwen_dirichlet_v2.py"

jobs = []
def add(name, cmd): jobs.append({"name": name, "cmd": cmd})

# --------------------------------------------------------------------
# A. Multi-seed full-VSI validation of single-layer L17 baseline
# --------------------------------------------------------------------
# We have seed 0 from v8 (lam0/0.3/1/3.0 for both models). Add seeds 1, 2, 3.
for short, mid in [("qwen", QWEN), ("intern", INTERN)]:
    for lam in ["0", "0.3", "1", "3.0"]:
        for seed in [1, 2, 3]:
            ck = f"checkpoints/{short}_lam{lam}_seed{seed}/lora"
            if not Path(ck, "adapter_config.json").exists(): continue
            out = f"reports/vsi_full_eval/{short}_lam{lam}_seed{seed}.json"
            if Path(out).exists(): continue
            cmd = (f"{PY} scripts/eval_vsi_batched.py --model-id {mid} --checkpoint {ck} "
                   f"--vsi-jsonl data/vsi_bench_full/eval_full.jsonl --out {out}")
            add(f"vsi_full_baseline_{short}_lam{lam}_seed{seed}", cmd)

# --------------------------------------------------------------------
# B. Multi-seed full-VSI validation of residualized (seeds 2, 3)
# --------------------------------------------------------------------
# We have seeds 0, 1 from v9 §5b/c. Need 2, 3 — but we don't have residualized
# checkpoints for those. So train them first.
for short, mid in [("qwen", QWEN), ("intern", INTERN)]:
    basis = f"reports/probe_features/{short}_residual_basis.npz"
    for lam in ["0", "0.3", "1", "3.0"]:
        for seed in [2, 3]:
            ckdir = f"checkpoints/{short}_lam{lam}_seed{seed}_resid"
            if Path(f"{ckdir}/lora/adapter_model.safetensors").exists(): continue
            cmd = (f"{PY} scripts/train_qwen_dirichlet.py --model-id {mid} "
                   f"--train-jsonl data/dirichlet_train_v2/train.jsonl "
                   f"--val-jsonl data/dirichlet_train_v2/val_iid.jsonl "
                   f"--output-dir {ckdir} "
                   f"--layer 17 --tau 2.0 --steps 500 --batch-size 2 --lora-rank 16 "
                   f"--eval-every 250 --log-every 100 --n-eval 100 "
                   f"--lambda-dir {lam} --seed {seed} --residualize-basis {basis}")
            add(f"train_resid_{short}_lam{lam}_seed{seed}", cmd)

# Then eval them on full bench (existing 0,1 + new 2,3)
for short, mid in [("qwen", QWEN), ("intern", INTERN)]:
    for lam in ["0", "0.3", "1", "3.0"]:
        for seed in [2, 3]:
            ck = f"checkpoints/{short}_lam{lam}_seed{seed}_resid/lora"
            out = f"reports/vsi_full_eval/{short}_lam{lam}_seed{seed}_resid.json"
            if Path(out).exists(): continue
            cmd = (f"{PY} scripts/eval_vsi_batched.py --model-id {mid} --checkpoint {ck} "
                   f"--vsi-jsonl data/vsi_bench_full/eval_full.jsonl --out {out}")
            add(f"vsi_full_resid_{short}_lam{lam}_seed{seed}", cmd)

# --------------------------------------------------------------------
# C. Single-layer ablations at L11 and L21 (4 seeds, full bench, λ=1 only)
# --------------------------------------------------------------------
# We already have L21 trained (qwen_lam1_seed{0..3}_L21) from v6. Add L11 fresh + full eval.
for short, mid in [("qwen", QWEN)]:  # Qwen only — InternVL too slow
    for layer in [11, 21]:
        for seed in [0, 1, 2, 3]:
            ckdir = f"checkpoints/{short}_lam1_seed{seed}_L{layer}"
            if Path(f"{ckdir}/lora/adapter_model.safetensors").exists():
                # already trained, just eval
                pass
            else:
                cmd = (f"{PY} scripts/train_qwen_dirichlet.py --model-id {mid} "
                       f"--train-jsonl data/dirichlet_train_v2/train.jsonl "
                       f"--val-jsonl data/dirichlet_train_v2/val_iid.jsonl "
                       f"--output-dir {ckdir} "
                       f"--layer {layer} --tau 2.0 --steps 500 --batch-size 2 --lora-rank 16 "
                       f"--eval-every 250 --log-every 100 --n-eval 100 "
                       f"--lambda-dir 1.0 --seed {seed}")
                add(f"train_L{layer}_{short}_lam1_seed{seed}", cmd)
            # full bench eval
            ck = f"{ckdir}/lora"
            out = f"reports/vsi_full_eval/{short}_lam1_L{layer}_seed{seed}.json"
            if Path(out).exists(): continue
            cmd = (f"{PY} scripts/eval_vsi_batched.py --model-id {mid} --checkpoint {ck} "
                   f"--vsi-jsonl data/vsi_bench_full/eval_full.jsonl --out {out}")
            add(f"vsi_full_L{layer}_{short}_lam1_seed{seed}", cmd)

# --------------------------------------------------------------------
# D. Larger LoRA rank (r=32, r=64) at L17, λ=1, 2 seeds
# --------------------------------------------------------------------
for short, mid in [("qwen", QWEN), ("intern", INTERN)]:
    for r in [32, 64]:
        for seed in [0, 1]:
            ckdir = f"checkpoints/{short}_lam1_seed{seed}_r{r}"
            if not Path(f"{ckdir}/lora/adapter_model.safetensors").exists():
                cmd = (f"{PY} scripts/train_qwen_dirichlet.py --model-id {mid} "
                       f"--train-jsonl data/dirichlet_train_v2/train.jsonl "
                       f"--val-jsonl data/dirichlet_train_v2/val_iid.jsonl "
                       f"--output-dir {ckdir} "
                       f"--layer 17 --tau 2.0 --steps 500 --batch-size 2 --lora-rank {r} "
                       f"--eval-every 250 --log-every 100 --n-eval 100 "
                       f"--lambda-dir 1.0 --seed {seed}")
                add(f"train_r{r}_{short}_lam1_seed{seed}", cmd)
            ck = f"{ckdir}/lora"
            out = f"reports/vsi_full_eval/{short}_lam1_r{r}_seed{seed}.json"
            if Path(out).exists(): continue
            cmd = (f"{PY} scripts/eval_vsi_batched.py --model-id {mid} --checkpoint {ck} "
                   f"--vsi-jsonl data/vsi_bench_full/eval_full.jsonl --out {out}")
            add(f"vsi_full_r{r}_{short}_lam1_seed{seed}", cmd)

# --------------------------------------------------------------------
# E. λ-schedule full-bench eval (we have phase-5 sched checkpoints at 2 seeds)
# --------------------------------------------------------------------
for short, mid in [("qwen", QWEN), ("intern", INTERN)]:
    for lam in ["1", "3.0"]:
        for seed in [0, 1]:
            ck = f"checkpoints/sched_{short}_lam{lam}_seed{seed}/lora"
            if not Path(ck, "adapter_config.json").exists(): continue
            out = f"reports/vsi_full_eval/sched_{short}_lam{lam}_seed{seed}.json"
            if Path(out).exists(): continue
            cmd = (f"{PY} scripts/eval_vsi_batched.py --model-id {mid} --checkpoint {ck} "
                   f"--vsi-jsonl data/vsi_bench_full/eval_full.jsonl --out {out}")
            add(f"vsi_full_sched_{short}_lam{lam}_seed{seed}", cmd)

# --------------------------------------------------------------------
# F. Longer training (1000 steps) at L17, λ=1, 2 seeds. Eval on full bench.
# We have qwen 1000-step from v6 phase 6. Need InternVL 1000-step + full bench.
# --------------------------------------------------------------------
for short, mid in [("qwen", QWEN), ("intern", INTERN)]:
    for seed in [0, 1, 2, 3]:
        ckdir = f"checkpoints/{short}_lam1_seed{seed}_steps1000"
        if not Path(f"{ckdir}/lora/adapter_model.safetensors").exists():
            cmd = (f"{PY} scripts/train_qwen_dirichlet.py --model-id {mid} "
                   f"--train-jsonl data/dirichlet_train_v2/train.jsonl "
                   f"--val-jsonl data/dirichlet_train_v2/val_iid.jsonl "
                   f"--output-dir {ckdir} "
                   f"--layer 17 --tau 2.0 --steps 1000 --batch-size 2 --lora-rank 16 "
                   f"--eval-every 500 --log-every 100 --n-eval 100 "
                   f"--lambda-dir 1.0 --seed {seed}")
            add(f"train_steps1000_{short}_lam1_seed{seed}", cmd)
        # full-bench eval
        ck = f"{ckdir}/lora"
        out = f"reports/vsi_full_eval/{short}_lam1_steps1000_seed{seed}_full.json"
        if Path(out).exists(): continue
        cmd = (f"{PY} scripts/eval_vsi_batched.py --model-id {mid} --checkpoint {ck} "
               f"--vsi-jsonl data/vsi_bench_full/eval_full.jsonl --out {out}")
        add(f"vsi_full_steps1000_{short}_lam1_seed{seed}", cmd)

print(f"Phase-7: {len(jobs)} jobs total")
from collections import Counter
prefixes = Counter(j["name"].split("_")[0] + "_" + j["name"].split("_")[1] for j in jobs)
for k, v in prefixes.most_common(): print(f"  {k}: {v}")

# Write to single queue (one runner with 4 GPUs handles it)
with open("queue_phase7.txt", "w") as f:
    for j in jobs:
        f.write(json.dumps(j) + "\n")
print(f"Written to queue_phase7.txt")
