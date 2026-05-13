#!/usr/bin/env python
"""Continuously drain an experiment queue across 4 GPUs (2/3/4/5).

Each line of the queue file is a JSON object:
    {"name": "kitti_qwen_lam0_seed0", "cmd": "python ..."}

Lines starting with `#` are skipped. The script polls GPUs 2/3/4/5;
when one is free, it pops the next job from the queue and launches it
on that GPU with CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=N.
Output goes to logs/v2/queue/{name}.log.

The script runs until the queue is empty AND no jobs are running,
which ends an "epoch". We then sleep 30s and rescan the queue file,
allowing the user to add more jobs by appending to it.

Usage
-----
    python scripts/run_experiment_queue.py \\
        --queue queue.txt \\
        --gpus 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("queue")


def gpu_busy(gpu_idx: int, threshold_mb: int = 1500) -> bool:
    """Return True if the GPU has more than threshold_mb of memory used,
    or if a process from this user is running on it."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             f"--id={gpu_idx}"],
            text=True
        ).strip()
        used = int(out.splitlines()[0])
        return used > threshold_mb
    except Exception:
        return True


def load_pending_jobs(queue_path: Path, completed: set) -> list[dict]:
    """Read queue file, return jobs whose name not in completed."""
    jobs = []
    if not queue_path.exists():
        return jobs
    for line in queue_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            job = json.loads(line)
        except Exception:
            continue
        if job.get("name") and job.get("cmd") and job["name"] not in completed:
            jobs.append(job)
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--gpus", type=int, nargs="+", default=[2, 3, 4, 5])
    p.add_argument("--log-dir", type=Path, default=Path("logs/v2/queue"))
    p.add_argument("--state-dir", type=Path, default=Path("logs/v2/queue_state"))
    p.add_argument("--poll-interval", type=int, default=15)
    p.add_argument("--epoch-sleep", type=int, default=30)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    completed_path = args.state_dir / "completed.txt"
    failed_path = args.state_dir / "failed.txt"
    completed = set()
    failed = set()
    fail_count: dict[str, int] = {}
    if completed_path.exists():
        completed = set(completed_path.read_text().splitlines())
    if failed_path.exists():
        failed = set(failed_path.read_text().splitlines())

    # Track running jobs: gpu -> {name, popen, log_path, started}
    running: dict[int, dict] = {}
    # Cooldown for GPU after failure
    gpu_cooldown_until: dict[int, float] = {}
    MAX_RETRIES = 2
    COOLDOWN_S = 60

    while True:
        now = time.time()
        # Reap finished jobs
        for gpu, info in list(running.items()):
            rc = info["popen"].poll()
            if rc is not None:
                elapsed = now - info["started"]
                if rc == 0:
                    completed.add(info["name"])
                    completed_path.write_text("\n".join(sorted(completed)))
                    logger.info("[GPU %d] %s OK (%.0fs)", gpu, info["name"], elapsed)
                    fail_count.pop(info["name"], None)
                else:
                    fail_count[info["name"]] = fail_count.get(info["name"], 0) + 1
                    logger.warning(
                        "[GPU %d] %s FAILED (rc=%d, elapsed=%.0fs, retries=%d)",
                        gpu, info["name"], rc, elapsed, fail_count[info["name"]]
                    )
                    if fail_count[info["name"]] >= MAX_RETRIES:
                        failed.add(info["name"])
                        failed_path.write_text("\n".join(sorted(failed)))
                        logger.warning("[GPU %d] %s blacklisted after %d failures",
                                       gpu, info["name"], MAX_RETRIES)
                    # Put GPU on cooldown after failure
                    gpu_cooldown_until[gpu] = now + COOLDOWN_S
                del running[gpu]

        # Skip already-failed jobs
        skip_set = completed | failed | {info["name"] for info in running.values()}
        pending = load_pending_jobs(args.queue, skip_set)

        # Launch new jobs on free GPUs
        for gpu in args.gpus:
            if gpu in running:
                continue
            if not pending:
                break
            if gpu_cooldown_until.get(gpu, 0) > now:
                continue
            if gpu_busy(gpu):
                continue
            job = pending.pop(0)
            log_path = args.log_dir / f"{job['name']}.log"
            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = job["cmd"]
            with log_path.open("w") as logf:
                p_obj = subprocess.Popen(
                    cmd, shell=True, env=env, stdout=logf, stderr=subprocess.STDOUT,
                    cwd=str(Path.cwd()),
                )
            running[gpu] = {"name": job["name"], "popen": p_obj, "log_path": str(log_path),
                            "started": now}
            logger.info("[GPU %d] launched: %s", gpu, job["name"])

        # If nothing running and nothing pending, sleep and rescan
        if not running and not pending:
            logger.info("Queue drained. Sleeping %ds and rescanning %s...",
                        args.epoch_sleep, args.queue)
            time.sleep(args.epoch_sleep)
        else:
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
