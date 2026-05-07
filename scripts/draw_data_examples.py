"""Render example figures from the Tier-C synthetic datasets.

Two datasets are visualized: (i) the orbital Free6DoF corpus used for
probing/training, and (ii) the random-walk corpus that drives the
person-walk control. For each dataset we render 5 figures, one scene
per figure, each showing:
  (a) simplified scene JSON
  (b) bird's-eye-view (BEV) layout
  (c) four sample rendered frames

For the Free6DoF dataset the four frames are evenly spaced; for the
person-walk dataset the four frames are chosen from those whose
segmentation mask contains at least one object pixel, so each shown
frame is guaranteed to capture >=1 object.

Outputs:
  figures/data_example_{1..5}.{pdf,png}        (Free6DoF)
  figures/data_example_walk_{1..5}.{pdf,png}   (person-walk)
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import patheffects as pe

ROOT = Path("/home/haoming/x-spatial-manual")
FREE6DOF_ROOT = ROOT / "data" / "tier_c_free6dof"
WALK_ROOT     = ROOT / "data" / "tier_c_person_walk"
QA_PATH = ROOT / "data" / "dirichlet_train" / "train.jsonl"
OUT_DIR = ROOT / "figures"

# Free6DoF picks: n_obj 4..8 with varied shape/color mixes.
PICKS_FREE6DOF = [
    {"sid": "s_01564a7c99_t2", "frames": [0, 5, 10, 15]},
    {"sid": "s_040f968706_t1", "frames": [0, 5, 10, 15]},
    {"sid": "s_00656814a4_t1", "frames": [0, 5, 10, 15]},
    {"sid": "s_04ce13ae46_t0", "frames": [0, 5, 10, 15]},
    {"sid": "s_002abd79ae_t2", "frames": [0, 5, 10, 15]},
]

# Person-walk picks: same diversity criterion (n_obj 3..8).
# Frames will be auto-selected from those with >=1 object visible
# in the segmentation mask (chosen to be evenly spread along the trajectory).
PICKS_WALK = [
    {"sid": "s_0cf19227c3_t2"},
    {"sid": "s_050a182a00_t0"},
    {"sid": "s_02e74c8b33_t1"},
    {"sid": "s_196a9b080e_t1"},
    {"sid": "s_2c5fd79ede_t1"},
]

COLOR_HEX = {
    "yellow":  "#F4C430", "magenta": "#E040A0", "cyan":    "#3FC0D0",
    "green":   "#3FA040", "orange":  "#F08030", "blue":    "#3060D0",
    "red":     "#D04040", "purple":  "#8040C0",
}
VOLUME = (-4.2, 4.2, -4.2, 4.2)


def load_qa_for_scene(sid: str) -> list[dict]:
    rows = []
    with open(QA_PATH) as f:
        for line in f:
            d = json.loads(line)
            if d["scene_id"] == sid:
                rows.append(d)
    return rows


def simplified_scene_json(scene: dict, max_objs: int = 8) -> str:
    """Build a compact single-string preview of scene.json."""
    sid = scene["scene_id"]
    objs = scene["objects"]
    lines = [
        "{",
        f'  "scene_id": "{sid}",',
        f'  "tier": "C", "n_objects": {len(objs)},',
        '  "objects": [',
    ]
    for o in objs[:max_objs]:
        cx, cy, cz = o["centroid"]
        lines.append(
            f'    {{"id":{o["object_id"]}, "shape":"{o["shape"]:<8}",'
            f' "color":"{o["color"]:<7}", "size":"{o["size"]:<6}",'
        )
        lines.append(
            f'     "centroid":[{cx:+.2f}, {cy:+.2f}, {cz:+.2f}]}},'
        )
    if len(objs) > max_objs:
        lines.append(f'    ... ({len(objs)-max_objs} more) ...')
    n_frames = len(scene.get("frames", []))
    lines.append(f'  ], "frames": [<{n_frames} entries>]')
    lines.append('}')
    return "\n".join(lines)


def draw_bev(ax, scene: dict):
    objs = scene["objects"]
    ax.set_xlim(VOLUME[0], VOLUME[1])
    ax.set_ylim(VOLUME[2], VOLUME[3])
    ax.set_aspect("equal")
    ax.grid(True, color="#D8D8D8", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)

    ax.plot(0, 0, "+", markersize=10, color="#303030", markeredgewidth=1.4)
    ax.text(0, -0.18, "origin", fontsize=6, ha="center", va="top",
            color="#404040")

    for o in objs:
        cx, cy = o["centroid"][0], o["centroid"][1]
        bx0, by0 = o["bbox_min"][0], o["bbox_min"][1]
        bx1, by1 = o["bbox_max"][0], o["bbox_max"][1]
        fc = COLOR_HEX.get(o["color"], "#888")
        if o["shape"] == "cube":
            patch = mpatches.Rectangle(
                (bx0, by0), bx1 - bx0, by1 - by0,
                facecolor=fc, edgecolor="#202020",
                linewidth=0.8, alpha=0.85,
            )
        else:
            r = max(bx1 - bx0, by1 - by0) / 2.0
            patch = mpatches.Circle(
                (cx, cy), r, facecolor=fc, edgecolor="#202020",
                linewidth=0.8, alpha=0.85,
            )
        ax.add_patch(patch)
        label = f"{o['color']} {o['shape']}"
        txt = ax.text(cx, by0 - 0.12, label, fontsize=6.2, ha="center",
                      va="top", color="#101010")
        txt.set_path_effects([pe.Stroke(linewidth=1.5, foreground="white"),
                              pe.Normal()])
    ax.set_title(f"BEV layout ({len(objs)} objects)", fontsize=9,
                 fontweight="bold")


def draw_frames(axes, scene_dir: Path, frame_ids: list[int]):
    for ax, t in zip(axes, frame_ids):
        img_path = scene_dir / "frames" / f"{t:03d}.png"
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#404040")
        ax.set_title(f"frame t={t}", fontsize=8)


def draw_text_panel(ax, body: str, title: str, mono: bool = False):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#404040")
    ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
    fontfamily = "monospace" if mono else "DejaVu Sans"
    fontsize = 6.4 if mono else 7.2
    ax.text(0.025, 0.97, body, fontsize=fontsize, ha="left", va="top",
            family=fontfamily, transform=ax.transAxes)


def pick_object_visible_frames(scene_dir: Path, n_pick: int = 4) -> list[int]:
    """Return ``n_pick`` evenly-spread frame indices whose segmentation mask
    contains at least one non-background pixel (i.e. >=1 object visible).

    Falls back to the first ``n_pick`` frame indices if too few are visible.
    """
    masks_dir = scene_dir / "masks"
    if not masks_dir.exists():
        return list(range(n_pick))
    masks = sorted(masks_dir.iterdir())
    good = []
    for i, mp in enumerate(masks):
        m = np.array(Image.open(mp))
        if (m != 0).any():
            good.append(i)
    if len(good) < n_pick:
        return good or list(range(min(n_pick, len(masks))))
    sel = np.linspace(0, len(good) - 1, n_pick).round().astype(int).tolist()
    return [good[i] for i in sel]


def render_one(pick: dict, idx: int, scene_root: Path, out_prefix: str,
               title_prefix: str, auto_pick_frames: bool = False):
    sid = pick["sid"]
    scene_dir = scene_root / sid
    with open(scene_dir / "scene.json") as f:
        scene = json.load(f)
    if auto_pick_frames or "frames" not in pick:
        frame_ids = pick_object_visible_frames(scene_dir, n_pick=4)
    else:
        frame_ids = pick["frames"]

    fig = plt.figure(figsize=(13.5, 3.2))
    gs = gridspec.GridSpec(
        1, 6, figure=fig,
        width_ratios=[2.4, 1.55, 1.0, 1.0, 1.0, 1.0],
        wspace=0.18,
        left=0.012, right=0.992, top=0.92, bottom=0.08,
    )
    # (a) JSON panel
    ax_json = fig.add_subplot(gs[0, 0])
    draw_text_panel(ax_json, simplified_scene_json(scene),
                    "(a) scene.json (simplified)", mono=True)

    # (b) BEV
    ax_bev = fig.add_subplot(gs[0, 1])
    draw_bev(ax_bev, scene)

    # (c) frames
    frame_axes = [fig.add_subplot(gs[0, 2 + i]) for i in range(4)]
    draw_frames(frame_axes, scene_dir, frame_ids)
    frame_left = frame_axes[0].get_position().x0
    fig.text(frame_left, 0.945, "(c) sample rendered frames",
             fontsize=10, fontweight="bold", ha="left", va="bottom")

    out_pdf = OUT_DIR / f"{out_prefix}_{idx}.pdf"
    out_png = OUT_DIR / f"{out_prefix}_{idx}.png"
    fig.savefig(out_pdf, pad_inches=0.10)
    fig.savefig(out_png, dpi=200, pad_inches=0.10)
    plt.close(fig)
    print(f"saved {out_pdf.name} & {out_png.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, pick in enumerate(PICKS_FREE6DOF, start=1):
        render_one(pick, i, FREE6DOF_ROOT,
                   out_prefix="data_example",
                   title_prefix="Tier-C Free6DoF",
                   auto_pick_frames=False)
    for i, pick in enumerate(PICKS_WALK, start=1):
        render_one(pick, i, WALK_ROOT,
                   out_prefix="data_example_walk",
                   title_prefix="Tier-C Person-Walk",
                   auto_pick_frames=True)


if __name__ == "__main__":
    main()
