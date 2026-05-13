"""Tier D (KITTI odometry) — real-world outdoor driving sequences.

KITTI odometry layout:
    <root>/sequences/<NN>/
        image_0/<NNNNNN>.png   # left grayscale rectified
        image_2/<NNNNNN>.png   # left color rectified (only in color zip)
        calib.txt              # per-sequence projection matrices
        times.txt              # per-frame timestamps (10 fps)
    <root>/poses/<NN>.txt      # 3×4 camera-to-world (world = frame-0 cam coords)

Only sequences 00-10 have ground-truth poses. Each sequence is an outdoor
drive of ~100-5000 frames, 10 fps. To match Tier D ARKitScenes's "16 frames
over a ~15-second window" protocol, we sample 16 frames evenly over a
150-frame window (15 s at 10 fps).

Grayscale frames are converted to RGB by channel replication — the model
just sees gray images as 3-channel. Color frames are used directly if the
``image_2`` directory is available.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

from ..scene import Camera, Frame, Object3D, Scene
from ..utils import ensure_dir


def _read_poses(poses_path: Path) -> list[np.ndarray]:
    """Return list of 4×4 camera-to-world matrices (one per frame)."""
    out: list[np.ndarray] = []
    for line in poses_path.read_text().strip().split("\n"):
        vals = np.array(line.split(), dtype=np.float64).reshape(3, 4)
        pose = np.eye(4)
        pose[:3, :4] = vals
        out.append(pose)
    return out


def _read_calib(calib_path: Path) -> np.ndarray:
    """Read KITTI calib.txt and return 3×3 K for P0 (grayscale left camera)."""
    for line in calib_path.read_text().strip().split("\n"):
        if line.startswith("P0:"):
            vals = line.split()[1:]
            P = np.array(vals, dtype=np.float64).reshape(3, 4)
            K = P[:3, :3].copy()
            return K
    raise RuntimeError(f"no P0 in {calib_path}")


def convert_kitti_clip(
    seq_dir: Path,
    poses: list[np.ndarray],
    K: np.ndarray,
    out_dir: Path,
    clip_id: str,
    frame_indices: list[int],
    n_frames: int = 16,
    use_color: bool = False,
) -> Scene:
    sel = sorted(frame_indices)[:n_frames]
    img_dir_name = "image_2" if use_color else "image_0"
    img_dir = seq_dir / img_dir_name
    if not img_dir.exists():
        raise FileNotFoundError(f"{img_dir} not found")

    scene_out_dir = ensure_dir(out_dir / clip_id)
    ensure_dir(scene_out_dir / "frames")
    ensure_dir(scene_out_dir / "masks")

    # Read image size from the first frame.
    first_png = img_dir / f"{sel[0]:06d}.png"
    first_img = Image.open(first_png)
    w, h = first_img.size

    full_mask = np.ones((h, w), dtype=np.uint8)
    K_list = [[float(K[r, c]) for c in range(3)] for r in range(3)]

    new_frames: list[Frame] = []
    for i, fi in enumerate(sel):
        src_png = img_dir / f"{fi:06d}.png"
        if not src_png.exists():
            raise FileNotFoundError(f"missing frame {src_png}")
        # Grayscale → RGB via channel replication; color kept as-is.
        src_pil = Image.open(src_png)
        dst_pil = src_pil.convert("RGB") if src_pil.mode != "RGB" else src_pil
        dst_png = scene_out_dir / "frames" / f"{i:03d}.png"
        dst_pil.save(dst_png)

        Image.fromarray(full_mask, "L").save(scene_out_dir / "masks" / f"{i:03d}.png")

        pose_cw = poses[fi]
        R_cw = pose_cw[:3, :3]
        t_cw = pose_cw[:3, 3]
        R_wc = R_cw.T
        t_wc = -R_cw.T @ t_cw

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
            "source": "kitti_odometry",
            "seq_dir": str(seq_dir),
            "frame_indices": sel,
            "image_size": [int(w), int(h)],
            "image_channel": "color" if use_color else "gray",
            "probe_mode": "cam_motion_only",
        },
    )
    scene.save(scene_out_dir)
    return scene


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert KITTI odometry to Scene schema (Tier D)")
    p.add_argument("--root", required=True,
                   help="Dataset root containing sequences/ and poses/")
    p.add_argument("--out", required=True)
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--n-frames", type=int, default=16)
    p.add_argument("--window-frames", type=int, default=150,
                   help="Window from which we sample n_frames evenly (10 fps × 15 s = 150)")
    p.add_argument("--color", action="store_true",
                   help="Use image_2 (color) if available; default is image_0 (grayscale)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    root = Path(args.root)
    out = ensure_dir(args.out)
    rng = random.Random(args.seed)

    # Enumerate training sequences (00..10).
    seq_ids = [f"{i:02d}" for i in range(11)]
    seqs: list[tuple[Path, list[np.ndarray], np.ndarray]] = []
    for sid in seq_ids:
        seq_dir = root / "sequences" / sid
        poses_path = root / "poses" / f"{sid}.txt"
        calib_path = seq_dir / "calib.txt"
        if not (seq_dir.exists() and poses_path.exists() and calib_path.exists()):
            print(f"[skip] sequence {sid}: missing data")
            continue
        poses = _read_poses(poses_path)
        K = _read_calib(calib_path)
        seqs.append((seq_dir, poses, K))
    print(f"loaded {len(seqs)} sequences")

    # Build candidate clips (one (seq_dir, start_idx) per non-overlapping window).
    candidates = []
    for seq_dir, poses, K in seqs:
        n_total = len(poses)
        w = min(args.window_frames, n_total)
        s = 0
        while s + w <= n_total:
            candidates.append((seq_dir, poses, K, s, w))
            s += w
    print(f"{len(candidates)} non-overlapping windows across all sequences")
    rng.shuffle(candidates)
    candidates = candidates[:args.n_samples]

    try:
        from tqdm import tqdm
        it = tqdm(candidates, desc="Tier D (KITTI)")
    except ImportError:
        it = candidates

    n_ok = 0
    for seq_dir, poses, K, start, window in it:
        idxs = np.linspace(start, start + window - 1, args.n_frames).astype(int).tolist()
        clip_id = f"kitti_{seq_dir.name}_s{start:06d}"
        try:
            convert_kitti_clip(
                seq_dir, poses, K, out, clip_id,
                idxs, n_frames=args.n_frames, use_color=args.color,
            )
            n_ok += 1
        except Exception as e:
            print(f"[skip] {clip_id}: {type(e).__name__}: {e}")
    print(f"converted {n_ok}/{len(candidates)} clips → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
