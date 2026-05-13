#!/usr/bin/env python
"""Generate BEV (bird's-eye-view) visualizations for every scene in
data/scenes_3d/, with color+shape annotations.

Produces one PNG per scene at figures/scenes_3d_bev/{scene_id}.png. Uses
multiprocessing to fan out across CPU cores; matplotlib in 'Agg' backend
so no display server is needed.

For each scene, the BEV plot shows:
  - working volume (x in [-4, 4], y in [-4, 4]) with a thin grid
  - one footprint per object (square for cube, circle for cylinder/sphere),
    drawn at the actual bbox extent and filled with the object's color
  - a text label "<color> <shape>" near each footprint
  - a small black '+' at the origin (camera position) for orientation
  - a title: "{scene_id} ({n_objects} objects)"

Usage:
    python scripts/visualize_scenes_3d_bev.py             # all scenes
    python scripts/visualize_scenes_3d_bev.py --limit 50  # first 50 only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Mapping the project's 8 color names to display hex codes.
COLOR_HEX = {
    "yellow":  "#F4C430",
    "magenta": "#E040A0",
    "cyan":    "#3FC0D0",
    "green":   "#3FA040",
    "orange":  "#F08030",
    "blue":    "#3060D0",
    "red":     "#D04040",
    "purple":  "#8040C0",
    "white":   "#E8E8E8",
    "black":   "#303030",
}

# Use lighter text outline on dark colors, dark on light. We pick black
# label text everywhere because we draw a thin white halo around it.

VOLUME = (-4.2, 4.2, -4.2, 4.2)


def _draw_object(ax, obj):
    """Render one object's footprint at its bbox extent with color+shape annotation."""
    x = obj["centroid"][0]
    y = obj["centroid"][1]
    bx0, by0 = obj["bbox_min"][0], obj["bbox_min"][1]
    bx1, by1 = obj["bbox_max"][0], obj["bbox_max"][1]
    color = obj["color"]
    shape = obj["shape"]
    fc = COLOR_HEX.get(color, "#808080")

    if shape == "cube":
        w, h = bx1 - bx0, by1 - by0
        patch = mpatches.Rectangle(
            (bx0, by0), w, h,
            facecolor=fc, edgecolor="#202020",
            linewidth=0.8, alpha=0.85,
        )
    else:  # cylinder, sphere -> circle footprint
        radius = max((bx1 - bx0), (by1 - by0)) / 2.0
        patch = mpatches.Circle(
            (x, y), radius,
            facecolor=fc, edgecolor="#202020",
            linewidth=0.8, alpha=0.85,
        )
    ax.add_patch(patch)

    # Label below the footprint to reduce overlap
    label = f"{color} {shape}"
    label_y = by0 - 0.12  # a bit below the bbox
    txt = ax.text(
        x, label_y, label,
        fontsize=6.5, ha="center", va="top",
        color="#101010",
    )
    # white halo for readability on dark fills
    from matplotlib import patheffects as pe
    txt.set_path_effects([
        pe.Stroke(linewidth=1.6, foreground="white"),
        pe.Normal(),
    ])


def render_scene(scene_path: Path, out_path: Path) -> str:
    """Render one scene to PNG. Returns scene_id (or 'ERR:scene_id:msg' on failure)."""
    try:
        with open(scene_path) as f:
            scene = json.load(f)
        sid = scene.get("scene_id", scene_path.parent.name)
        objects = scene.get("objects", [])
        n = len(objects)

        fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=110)
        ax.set_xlim(VOLUME[0], VOLUME[1])
        ax.set_ylim(VOLUME[2], VOLUME[3])
        ax.set_aspect("equal")
        ax.grid(True, color="#D0D0D0", linewidth=0.4, linestyle="-")
        ax.set_axisbelow(True)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-4, -2, 0, 2, 4])
        ax.tick_params(labelsize=7)

        # axis labels
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("y (m)", fontsize=8)

        # camera origin marker
        ax.plot(0, 0, "+", markersize=10, color="#303030", markeredgewidth=1.4)
        ax.text(
            0, -0.18, "origin",
            fontsize=6, ha="center", va="top", color="#404040",
        )

        # objects
        for o in objects:
            _draw_object(ax, o)

        ax.set_title(f"{sid}  ({n} object{'s' if n != 1 else ''})", fontsize=9)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        return sid
    except Exception as exc:
        return f"ERR:{scene_path.parent.name}:{exc}"


def _worker(args):
    return render_scene(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-root", default="data/scenes_3d", type=Path)
    ap.add_argument("--out-root", default="figures/scenes_3d_bev", type=Path)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, render only the first N scenes (for testing).")
    ap.add_argument("--n-workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    scene_dirs = sorted(p for p in args.scenes_root.iterdir() if p.is_dir())
    if args.limit > 0:
        scene_dirs = scene_dirs[:args.limit]
    print(f"Found {len(scene_dirs)} scene directories.")

    tasks = []
    for d in scene_dirs:
        sid = d.name
        scene_json = d / "scene.json"
        if not scene_json.exists():
            continue
        out_path = args.out_root / f"{sid}.png"
        if out_path.exists() and not args.overwrite:
            continue
        tasks.append((scene_json, out_path))

    print(f"Rendering {len(tasks)} scenes with {args.n_workers} workers...")
    if not tasks:
        print("Nothing to render.")
        return

    n_done = 0
    n_err = 0
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = [ex.submit(_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if isinstance(r, str) and r.startswith("ERR:"):
                n_err += 1
                if n_err <= 5:
                    print(" ", r)
            else:
                n_done += 1
            if i % 200 == 0:
                print(f"  {i}/{len(tasks)} done ({n_err} errors so far)")

    print(f"Rendered {n_done} successfully ({n_err} errors).")


if __name__ == "__main__":
    main()
