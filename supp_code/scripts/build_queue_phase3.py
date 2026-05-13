#!/usr/bin/env python
"""Phase-3 queue: evaluate Dirichlet-trained checkpoints on three new
external benchmarks (MindCube, ViewSpatial-Bench, OST-Bench) for
Qwen + InternVL × four lambdas × four seeds.

Skips evals whose output JSON already exists.
"""
from __future__ import annotations

import json
from pathlib import Path

PY = "/home/haoming/miniconda3/envs/vlm-ex/bin/python3"
QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"
INTERN = "OpenGVLab/InternVL3-8B-hf"

LAMBDAS = ["0", "0.3", "1", "3.0"]
SEEDS = [0, 1, 2, 3]
MODELS = [("qwen", QWEN), ("intern", INTERN)]

# Per-eval caps (full set is too big for a sweep)
N_VS = 500   # ViewSpatial: 500 from 5712
N_OST = 500  # OST: 500 from 5557 MC items

jobs = []


def add(name: str, cmd: str):
    jobs.append({"name": name, "cmd": cmd})


# ---------------- MindCube ----------------
for short, mid in MODELS:
    for lam in LAMBDAS:
        for seed in SEEDS:
            ck = f"checkpoints/{short}_lam{lam}_seed{seed}/lora"
            if not Path(ck, "adapter_config.json").exists():
                continue
            out = f"reports/mindcube_eval/{short}_lam{lam}_seed{seed}.json"
            if Path(out).exists():
                continue
            cmd = (f"{PY} scripts/eval_mindcube.py "
                   f"--model-id {mid} --checkpoint {ck} "
                   f"--mindcube-jsonl /home/haoming/mindcube_data/raw/MindCube_tinybench.jsonl "
                   f"--image-root /home/haoming/mindcube_data "
                   f"--out {out}")
            add(f"mc_{short}_lam{lam}_seed{seed}", cmd)

# ---------------- ViewSpatial-Bench ----------------
for short, mid in MODELS:
    for lam in LAMBDAS:
        for seed in SEEDS:
            ck = f"checkpoints/{short}_lam{lam}_seed{seed}/lora"
            if not Path(ck, "adapter_config.json").exists():
                continue
            out = f"reports/viewspatial_eval/{short}_lam{lam}_seed{seed}.json"
            if Path(out).exists():
                continue
            cmd = (f"{PY} scripts/eval_viewspatial.py "
                   f"--model-id {mid} --checkpoint {ck} "
                   f"--bench-json data/viewspatial_bench/ViewSpatial-Bench.json "
                   f"--image-root data/viewspatial_bench "
                   f"--out {out} --n-eval {N_VS} --seed 0")
            add(f"vs_{short}_lam{lam}_seed{seed}", cmd)

# ---------------- OST-Bench ----------------
for short, mid in MODELS:
    for lam in LAMBDAS:
        for seed in SEEDS:
            ck = f"checkpoints/{short}_lam{lam}_seed{seed}/lora"
            if not Path(ck, "adapter_config.json").exists():
                continue
            out = f"reports/ost_eval/{short}_lam{lam}_seed{seed}.json"
            if Path(out).exists():
                continue
            cmd = (f"{PY} scripts/eval_ost_bench.py "
                   f"--model-id {mid} --checkpoint {ck} "
                   f"--bench-json data/ost_bench/OST_bench.json "
                   f"--image-root data/ost_bench/image_upload "
                   f"--out {out} --n-eval {N_OST} --seed 0")
            add(f"ost_{short}_lam{lam}_seed{seed}", cmd)

queue_path = Path("queue.txt")
with queue_path.open("a") as f:
    for j in jobs:
        f.write(json.dumps(j) + "\n")
print(f"Appended {len(jobs)} new phase-3 jobs to {queue_path}")
from collections import Counter
prefixes = Counter(j["name"].split("_")[0] for j in jobs)
print("By prefix:", dict(prefixes))
