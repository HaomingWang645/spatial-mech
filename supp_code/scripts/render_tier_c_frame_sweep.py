#!/usr/bin/env python
"""Render tier_c_free6dof with configurable n_frames, on a fixed subset of base scenes.

For Option 3 frame-count sweep: re-use the same 100 base scenes as the existing
16-frame extraction, so the same (base_scene, traj_idx) pair can be compared across
frame counts.

Usage:
    python scripts/render_tier_c_frame_sweep.py \
        --base-scene-list /tmp/base_scenes.txt \
        --n-frames 32 --out data/tier_c_free6dof_f32 \
        --trajectories-per-scene 2 --workers 8
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from spatial_subspace.render.tier_c import render_tier_c
from spatial_subspace.scene import Scene
from spatial_subspace.utils import ensure_dir, load_yaml


def _render_one(args):
    scene_dir, cfg_path, out_dir, traj_idx, seed = args
    cfg = load_yaml(cfg_path)
    if "n_frames_override" in cfg:
        cfg["n_frames"] = cfg["n_frames_override"]
    scene = Scene.load(scene_dir)
    rng = random.Random(seed ^ (hash((scene.scene_id, traj_idx)) & 0x7FFFFFFF))
    render_tier_c(scene, cfg, Path(out_dir), rng, traj_idx=traj_idx)
    return scene.scene_id, traj_idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tier_c_free6dof.yaml")
    ap.add_argument("--base-scene-list", required=True,
                    help="File with one base scene_id per line")
    ap.add_argument("--scenes-root", default="data/scenes_3d")
    ap.add_argument("--n-frames", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trajectories-per-scene", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ids = [l.strip() for l in Path(args.base_scene_list).read_text().splitlines() if l.strip()]
    cfg = load_yaml(args.config)
    cfg["n_frames"] = int(args.n_frames)
    out = ensure_dir(Path(args.out))

    # Persist the overridden config beside the run so the extraction step picks up n_frames
    cfg_write = Path(args.out) / "render_config.yaml"
    import yaml
    cfg_write.write_text(yaml.safe_dump(cfg))
    # Also write original yaml + override into a scratch file the worker loads
    scratch_cfg = out / "_render_cfg_override.yaml"
    cfg["n_frames_override"] = int(args.n_frames)
    scratch_cfg.write_text(yaml.safe_dump(cfg))

    tasks = []
    for sid in ids:
        scene_dir = Path(args.scenes_root) / sid
        if not scene_dir.exists():
            print(f"[warn] missing {scene_dir}")
            continue
        for t in range(args.trajectories_per_scene):
            tasks.append((str(scene_dir), str(scratch_cfg), str(out), t, args.seed))

    t0 = time.time()
    done = 0
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_render_one, t) for t in tasks]
        for f in as_completed(futs):
            sid, ti = f.result()
            done += 1
            if done % 50 == 0 or done == total:
                print(f"[{done}/{total}] rendered {sid}_t{ti} ({time.time()-t0:.1f}s)")
    print(f"done in {time.time()-t0:.1f}s  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
