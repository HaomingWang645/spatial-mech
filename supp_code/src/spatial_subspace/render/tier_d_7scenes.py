"""Tier D (7-Scenes) — real-world handheld Kinect sequences (Microsoft 7-Scenes).

Each 7-Scenes sequence is a continuous Kinect recording of someone walking
through a room with a handheld Kinect v1, 30 fps, ~500-1000 frames, with
ground-truth per-frame camera-to-world poses (4×4) derived from KinectFusion.

Input layout:
    <root>/<scene>/seq-NN/
        frame-<NNNNNN>.color.png   # 640×480 RGB
        frame-<NNNNNN>.depth.png   # 16-bit depth (unused)
        frame-<NNNNNN>.pose.txt    # 4 rows × 4 floats, camera-to-world

Shared Kinect calibration (approximate):
    fx = fy = 585, cx = 320, cy = 240 (image is 640×480)

For 100 samples at 16 frames each, we pick ≥2 non-overlapping 500-frame
windows per sequence and sample 16 evenly-spaced frames from each. The
adapter writes the same Scene schema used by Tier D ARKitScenes, including
a full-image dummy mask so downstream extraction pools over all visual
tokens.
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image

from ..scene import Camera, Frame, Object3D, Scene
from ..utils import ensure_dir


# Shared Kinect intrinsics for 7-Scenes (RGB stream 640×480)
_KINECT_FX = 585.0
_KINECT_FY = 585.0
_KINECT_CX = 320.0
_KINECT_CY = 240.0
_KINECT_W = 640
_KINECT_H = 480


def _read_pose(pose_path: Path) -> np.ndarray:
    """Read a 4×4 camera-to-world pose."""
    text = pose_path.read_text().strip().split()
    m = np.array(text, dtype=np.float64).reshape(4, 4)
    return m


def _list_frames(seq_dir: Path) -> list[tuple[int, Path, Path]]:
    """Return sorted list of (frame_idx, color_png, pose_txt)."""
    colors = sorted(seq_dir.glob("frame-*.color.png"))
    out: list[tuple[int, Path, Path]] = []
    for c in colors:
        m = re.match(r"frame-(\d+)\.color\.png$", c.name)
        if not m:
            continue
        idx = int(m.group(1))
        pose = seq_dir / f"frame-{idx:06d}.pose.txt"
        if pose.exists():
            out.append((idx, c, pose))
    return out


def convert_7scenes_clip(
    seq_dir: Path,
    out_dir: Path,
    clip_id: str,
    frame_indices: list[int],
    all_frames_by_idx: dict[int, tuple[Path, Path]],
    n_frames: int = 16,
) -> Scene:
    """Convert a chosen sequence of 16 frames into Scene schema."""
    sel = sorted(frame_indices)[:n_frames]
    if len(sel) != n_frames:
        raise RuntimeError(f"only {len(sel)} frames available for {clip_id}")

    scene_out_dir = ensure_dir(out_dir / clip_id)
    ensure_dir(scene_out_dir / "frames")
    ensure_dir(scene_out_dir / "masks")

    # Dummy full-image mask (all pixels = object 0, id=1).
    full_mask = np.ones((_KINECT_H, _KINECT_W), dtype=np.uint8)

    K_list = [
        [_KINECT_FX, 0.0, _KINECT_CX],
        [0.0, _KINECT_FY, _KINECT_CY],
        [0.0, 0.0, 1.0],
    ]

    new_frames: list[Frame] = []
    for i, fi in enumerate(sel):
        color_src, pose_src = all_frames_by_idx[fi]
        pose_cw = _read_pose(pose_src)
        R_cw = pose_cw[:3, :3]
        t_cw = pose_cw[:3, 3]
        R_wc = R_cw.T
        t_wc = -R_cw.T @ t_cw

        dst_png = scene_out_dir / "frames" / f"{i:03d}.png"
        if not dst_png.exists():
            try:
                dst_png.symlink_to(color_src.resolve())
            except OSError:
                Image.open(color_src).convert("RGB").save(dst_png)
        Image.fromarray(full_mask, "L").save(scene_out_dir / "masks" / f"{i:03d}.png")

        E_4x4 = [
            [float(R_wc[0, 0]), float(R_wc[0, 1]), float(R_wc[0, 2]), float(t_wc[0])],
            [float(R_wc[1, 0]), float(R_wc[1, 1]), float(R_wc[1, 2]), float(t_wc[1])],
            [float(R_wc[2, 0]), float(R_wc[2, 1]), float(R_wc[2, 2]), float(t_wc[2])],
            [0.0, 0.0, 0.0, 1.0],
        ]
        new_frames.append(
            Frame(
                frame_id=i,
                image_path=f"frames/{i:03d}.png",
                mask_path=f"masks/{i:03d}.png",
                camera=Camera(intrinsics=K_list, extrinsics=E_4x4, kind="perspective"),
            )
        )

    dummy = Object3D(
        object_id=0,
        shape="scene",
        color="unknown",
        size="scene",
        centroid=(0.0, 0.0, 0.0),
        bbox_min=(-1.0, -1.0, -1.0),
        bbox_max=(1.0, 1.0, 1.0),
        extras={"dummy": True},
    )

    scene = Scene(
        scene_id=clip_id,
        tier="D",
        objects=[dummy],
        frames=new_frames,
        qa=[],
        extras={
            "source": "7scenes",
            "seq_dir": str(seq_dir),
            "frame_indices": sel,
            "image_size": [_KINECT_W, _KINECT_H],
            "probe_mode": "cam_motion_only",
        },
    )
    scene.save(scene_out_dir)
    return scene


def _pick_clips_for_sequence(
    seq_dir: Path,
    n_clips: int,
    n_frames: int,
    frames_per_window: int,
    rng: random.Random,
) -> list[tuple[str, list[int], dict[int, tuple[Path, Path]]]]:
    """Pick up to ``n_clips`` non-overlapping 16-frame clips from one sequence."""
    frames = _list_frames(seq_dir)
    if len(frames) < n_frames:
        return []
    all_by_idx = {fi: (c, p) for (fi, c, p) in frames}
    sorted_indices = [fi for fi, _, _ in frames]

    # One "clip window" is frames_per_window frames long; we pick n_frames evenly
    # within it. Clips are non-overlapping.
    window = min(frames_per_window, len(frames))
    max_start = len(frames) - window
    if max_start < 0:
        return []
    stride = max(window, 1)
    candidates: list[int] = []
    s = 0
    while s + window <= len(frames):
        candidates.append(s)
        s += stride
    rng.shuffle(candidates)
    out: list[tuple[str, list[int], dict[int, tuple[Path, Path]]]] = []
    for start_ix in candidates[:n_clips]:
        idxs_in_window = sorted_indices[start_ix : start_ix + window]
        target = np.linspace(0, window - 1, n_frames).astype(int)
        chosen = [idxs_in_window[t] for t in target]
        clip_id = f"{seq_dir.parent.name}_{seq_dir.name}_s{start_ix:04d}"
        out.append((clip_id, chosen, all_by_idx))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert 7-Scenes to Scene schema (Tier D)")
    p.add_argument("--root", required=True,
                   help="Directory containing scene subdirs (chess, fire, heads, ...)")
    p.add_argument("--out", required=True)
    p.add_argument("--n-samples", type=int, default=100,
                   help="Total number of 16-frame clips to write across all sequences.")
    p.add_argument("--n-frames", type=int, default=16)
    p.add_argument("--window-frames", type=int, default=450,
                   help="Source window size from which we sample n_frames evenly.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    root = Path(args.root)
    out = ensure_dir(args.out)
    rng = random.Random(args.seed)

    # Enumerate all sequences across all scenes.
    seq_dirs: list[Path] = []
    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        for seq_dir in sorted(scene_dir.iterdir()):
            if seq_dir.is_dir() and seq_dir.name.startswith("seq-"):
                seq_dirs.append(seq_dir)
    print(f"found {len(seq_dirs)} sequences under {root}")

    # Round-robin: sample clips from each sequence until we've got n_samples.
    needed = args.n_samples
    all_clips: list[tuple[str, list[int], dict[int, tuple[Path, Path]], Path]] = []
    clips_per_seq = max(1, (needed + len(seq_dirs) - 1) // len(seq_dirs))
    for seq in seq_dirs:
        picks = _pick_clips_for_sequence(
            seq, clips_per_seq, args.n_frames, args.window_frames, rng
        )
        for cid, idxs, idxmap in picks:
            all_clips.append((cid, idxs, idxmap, seq))
    rng.shuffle(all_clips)
    all_clips = all_clips[:args.n_samples]

    try:
        from tqdm import tqdm
        it = tqdm(all_clips, desc="Tier D (7-Scenes)")
    except ImportError:
        it = all_clips

    n_ok = 0
    for clip_id, idxs, idxmap, seq_dir in it:
        try:
            convert_7scenes_clip(
                seq_dir, out, clip_id, idxs, idxmap, n_frames=args.n_frames
            )
            n_ok += 1
        except Exception as e:
            print(f"[skip] {clip_id}: {type(e).__name__}: {e}")
    print(f"converted {n_ok}/{len(all_clips)} clips → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
