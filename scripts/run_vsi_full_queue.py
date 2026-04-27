#!/usr/bin/env python
"""Dedicated queue runner for the full VSI-Bench eval (separate from main runner).

Launches 10 jobs (Qwen + InternVL × {base, lam0, lam0.3, lam1, lam3.0}) across
GPUs 0/1/3/4/6/7 (skipping 2 which has our other runner, and 5 which was just
freed).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

PY = "/home/haoming/miniconda3/envs/vlm-ex/bin/python3"
QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"
INTERN = "OpenGVLab/InternVL3-8B-hf"
SEED = 0
EVAL_JSONL = "data/vsi_bench_full/eval_full.jsonl"

logger = logging.getLogger("vsi_full_runner")


def gpu_busy(gpu: int, threshold_mb: int = 1500) -> bool:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             f"--id={gpu}"], text=True
        ).strip()
        return int(out.splitlines()[0]) > threshold_mb
    except Exception:
        return True


def make_jobs() -> list[dict]:
    jobs = []
    for short, mid in [("qwen", QWEN), ("intern", INTERN)]:
        # Base
        out = f"reports/vsi_full_eval/{short}_base.json"
        if not Path(out).exists():
            cmd = (f"{PY} scripts/eval_vsi_batched.py --model-id {mid} "
                   f"--vsi-jsonl {EVAL_JSONL} --out {out}")
            jobs.append({"name": f"vsi_full_{short}_base", "cmd": cmd})
        # Each lambda × seed=0
        for lam in ["0", "0.3", "1", "3.0"]:
            ck = f"checkpoints/{short}_lam{lam}_seed{SEED}/lora"
            if not Path(ck, "adapter_config.json").exists():
                continue
            out = f"reports/vsi_full_eval/{short}_lam{lam}_seed{SEED}.json"
            if Path(out).exists():
                continue
            cmd = (f"{PY} scripts/eval_vsi_batched.py --model-id {mid} --checkpoint {ck} "
                   f"--vsi-jsonl {EVAL_JSONL} --out {out}")
            jobs.append({"name": f"vsi_full_{short}_lam{lam}", "cmd": cmd})
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 3, 4, 6, 7])
    p.add_argument("--log-dir", type=Path, default=Path("logs/v2/vsi_full"))
    p.add_argument("--poll-interval", type=int, default=20)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args.log_dir.mkdir(parents=True, exist_ok=True)

    jobs = make_jobs()
    logger.info("Built %d jobs:", len(jobs))
    for j in jobs:
        logger.info("  %s", j["name"])
    if not jobs:
        logger.info("Nothing to do.")
        return

    pending = list(jobs)
    running: dict[int, dict] = {}

    while pending or running:
        # Reap finished
        for gpu, info in list(running.items()):
            rc = info["popen"].poll()
            if rc is not None:
                elapsed = time.time() - info["started"]
                if rc == 0:
                    logger.info("[GPU %d] %s OK (%.0fs)", gpu, info["name"], elapsed)
                else:
                    logger.warning("[GPU %d] %s FAILED rc=%d (%.0fs)", gpu, info["name"], rc, elapsed)
                del running[gpu]

        # Launch new jobs
        for gpu in args.gpus:
            if gpu in running: continue
            if not pending: break
            if gpu_busy(gpu): continue
            job = pending.pop(0)
            log_path = args.log_dir / f"{job['name']}.log"
            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            with log_path.open("w") as logf:
                p_obj = subprocess.Popen(
                    job["cmd"], shell=True, env=env, stdout=logf,
                    stderr=subprocess.STDOUT, cwd=str(Path.cwd())
                )
            running[gpu] = {"name": job["name"], "popen": p_obj,
                            "started": time.time()}
            logger.info("[GPU %d] launched: %s", gpu, job["name"])

        time.sleep(args.poll_interval)

    logger.info("All done.")


if __name__ == "__main__":
    main()
