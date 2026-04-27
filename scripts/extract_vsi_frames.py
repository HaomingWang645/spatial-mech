#!/usr/bin/env python
"""Extract frame 0 from each VSI-Bench .mp4 video.

Output: data/vsi_bench_full/{dataset}/{scene}/frames/000.png
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def extract_one(mp4_path: Path, out_dir: Path) -> tuple[str, bool, str]:
    out_path = out_dir / "000.png"
    if out_path.exists() and out_path.stat().st_size > 0:
        return (str(mp4_path), True, "skip")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4_path),
           "-vf", "select=eq(n\\,0)", "-vframes", "1", str(out_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        return (str(mp4_path), True, "ok")
    except Exception as e:  # noqa: BLE001
        return (str(mp4_path), False, str(e)[:100])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/vsi_bench_full"))
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    tasks = []
    for ds in ["arkitscenes", "scannet", "scannetpp"]:
        ds_dir = args.root / ds
        for mp4 in sorted(ds_dir.glob("*.mp4")):
            scene = mp4.stem
            out_dir = args.root / ds / scene / "frames"
            tasks.append((mp4, out_dir))
    print(f"Found {len(tasks)} videos to process", flush=True)

    n_ok = n_skip = n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(extract_one, m, o) for m, o in tasks]
        for i, fut in enumerate(as_completed(futs)):
            path, ok, msg = fut.result()
            if ok and msg == "skip": n_skip += 1
            elif ok: n_ok += 1
            else:
                n_err += 1
                print(f"  ERR {path}: {msg}", flush=True)
            if (i+1) % 50 == 0:
                print(f"  progress {i+1}/{len(tasks)} (ok={n_ok}, skip={n_skip}, err={n_err})", flush=True)
    print(f"Done: ok={n_ok}, skip={n_skip}, err={n_err}")


if __name__ == "__main__":
    main()
