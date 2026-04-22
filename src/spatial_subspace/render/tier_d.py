"""Tier D — real-world 3D scenes from ARKitScenes (plan §3.4).

This pared-down Tier D adapter is built for the **camera-motion probe only**
— it does not try to project ARKitScenes 3D object bounding boxes onto each
frame. Per-object depth + per-scene normalized 3D coordinate probes would
need accurate per-object masks, and getting those right from ARKitScenes
annotations requires a frustum-clipping projector that we leave out of
scope for now. (A first cut tried it and mis-projected bboxes onto the
floor due to coordinate convention ambiguity.)

What this adapter does produce, per ARKitScenes 3dod scene:

  - 16 RGB frames sampled evenly over a centred ``window_sec`` window
  - one full-image dummy mask per frame (every pixel labelled object_id = 1)
  - camera intrinsics + extrinsics for each frame, stored in the common
    Scene schema so downstream ``extract_activations.py`` / probes can reuse
    all their Tier A/B/C machinery

The dummy mask causes ``extract_scene_video`` to pool every visual token
into a single "object" vector per temporal token, which is exactly the
input the camera-motion probe expects (it aggregates per (scene, t) anyway).

Extrinsic convention. ARKitScenes stores ``(rx ry rz tx ty tz)`` per frame
with a ``camera-to-world`` semantic: ``p_world = R_cw·p_cam + t_cw``. Our
Scene schema expects ``world-to-camera`` (so that ``p_cam = E·p_world``).
We store ``E = (R_cw.T, -R_cw.T·t_cw)`` exactly as that. The axis convention
of ARKit's camera frame is left untouched — it doesn't matter for the
camera-motion probe, whose target ``Δ = E_curr · E_prev⁻¹`` is invariant to
any rigid choice of world frame.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from ..scene import Camera, Frame, Object3D, Scene
from ..utils import ensure_dir


def _rodrigues(r: np.ndarray) -> np.ndarray:
    """Axis-angle (magnitude = angle) → 3x3 rotation matrix."""
    theta = float(np.linalg.norm(r))
    if theta < 1e-8:
        return np.eye(3)
    k = r / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return (
        np.cos(theta) * np.eye(3)
        + np.sin(theta) * K
        + (1.0 - np.cos(theta)) * np.outer(k, k)
    )


def _load_trajectory(traj_path: Path) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Return [(timestamp, R_cw, t_cw), ...] sorted by timestamp."""
    entries: list[tuple[float, np.ndarray, np.ndarray]] = []
    for line in Path(traj_path).read_text().strip().split("\n"):
        parts = line.split()
        if len(parts) < 7:
            continue
        ts = float(parts[0])
        rot = np.array(parts[1:4], dtype=np.float64)
        trans = np.array(parts[4:7], dtype=np.float64)
        entries.append((ts, _rodrigues(rot), trans))
    entries.sort(key=lambda x: x[0])
    return entries


def _load_intrinsics(k_dir: Path, frame_ts: float) -> tuple[np.ndarray, int, int]:
    candidates = sorted(k_dir.glob("*.pincam"))
    if not candidates:
        raise FileNotFoundError(f"no .pincam under {k_dir}")
    ts_list = np.array([float(p.stem.split("_")[-1]) for p in candidates])
    idx = int(np.argmin(np.abs(ts_list - frame_ts)))
    parts = candidates[idx].read_text().split()
    w, h, fx, fy, cx, cy = (float(x) for x in parts[:6])
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return K, int(w), int(h)


def _find_nearest_pose(
    traj: list[tuple[float, np.ndarray, np.ndarray]], ts: float
) -> tuple[np.ndarray, np.ndarray]:
    ts_list = np.array([e[0] for e in traj])
    idx = int(np.argmin(np.abs(ts_list - ts)))
    return traj[idx][1], traj[idx][2]


def _sample_timestamps(
    all_ts: list[float], n_frames: int, window_sec: float
) -> list[float]:
    if not all_ts:
        return []
    t_min, t_max = all_ts[0], all_ts[-1]
    total = t_max - t_min
    if total < window_sec or total < 1e-3:
        window_start, window_end = t_min, t_max
    else:
        window_start = t_min + (total - window_sec) / 2.0
        window_end = window_start + window_sec
    target = np.linspace(window_start, window_end, n_frames)
    ts_arr = np.array(all_ts)
    selected: list[float] = []
    last_idx = -1
    for tt in target:
        idx = int(np.argmin(np.abs(ts_arr - tt)))
        if idx <= last_idx and idx + 1 < len(ts_arr):
            idx = last_idx + 1
        selected.append(float(ts_arr[idx]))
        last_idx = idx
    return selected


def convert_arkit_scene(
    src_dir: Path,
    out_dir: Path,
    n_frames: int = 16,
    window_sec: float = 15.0,
) -> Scene:
    """Convert one ARKitScenes scene to our Scene schema (camera-motion-only).

    Writes 16 RGB frames (symlinked from the source), 16 full-image dummy
    masks, and a scene.json with one dummy Object3D and one Frame per frame
    carrying intrinsics + world-to-camera extrinsics.
    """
    src_dir = Path(src_dir)
    scene_id = src_dir.name
    frames_root = src_dir / f"{scene_id}_frames"
    wide_dir = frames_root / "lowres_wide"
    intrinsics_dir = frames_root / "lowres_wide_intrinsics"
    traj_path = frames_root / "lowres_wide.traj"
    if not wide_dir.exists() or not traj_path.exists() or not intrinsics_dir.exists():
        raise FileNotFoundError(
            f"expected ARKitScenes layout not found at {src_dir}; "
            f"missing one of lowres_wide/, lowres_wide.traj, lowres_wide_intrinsics/"
        )

    traj = _load_trajectory(traj_path)
    frame_files = sorted(wide_dir.glob(f"{scene_id}_*.png"))
    frame_ts = [float(p.stem.split("_")[-1]) for p in frame_files]
    if not frame_ts:
        raise RuntimeError(f"no frames under {wide_dir}")
    selected_ts = _sample_timestamps(frame_ts, n_frames, window_sec)
    ts_to_file = dict(zip(frame_ts, frame_files))

    scene_out_dir = ensure_dir(out_dir / scene_id)
    ensure_dir(scene_out_dir / "frames")
    ensure_dir(scene_out_dir / "masks")

    # One dummy object covering the whole image. Centroid/bbox are placeholders
    # — not used by the camera-motion probe, which aggregates per (scene, t).
    objects = [
        Object3D(
            object_id=0,
            shape="scene",
            color="unknown",
            size="scene",
            centroid=(0.0, 0.0, 0.0),
            bbox_min=(-1.0, -1.0, -1.0),
            bbox_max=(1.0, 1.0, 1.0),
            extras={"dummy": True, "note": "full-image mask placeholder"},
        )
    ]

    new_frames: list[Frame] = []
    full_mask_cache: np.ndarray | None = None
    for i, ts in enumerate(selected_ts):
        src_png = ts_to_file[ts]
        K, w, h = _load_intrinsics(intrinsics_dir, ts)
        R_cw, t_cw = _find_nearest_pose(traj, ts)

        # world-to-camera: E = inv(pose_cw)
        R_wc = R_cw.T
        t_wc = -R_cw.T @ t_cw

        dst_png = scene_out_dir / "frames" / f"{i:03d}.png"
        if not dst_png.exists():
            try:
                dst_png.symlink_to(src_png.resolve())
            except OSError:
                Image.open(src_png).save(dst_png)

        # Dummy full-image mask (every pixel labelled object 0 → stored as 1).
        if full_mask_cache is None or full_mask_cache.shape != (h, w):
            full_mask_cache = np.ones((h, w), dtype=np.uint8)  # object_id 0 + 1 = 1
        Image.fromarray(full_mask_cache, "L").save(scene_out_dir / "masks" / f"{i:03d}.png")

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
                camera=Camera(
                    intrinsics=[[float(K[r, c]) for c in range(3)] for r in range(3)],
                    extrinsics=E_4x4,
                    kind="perspective",
                ),
            )
        )

    scene = Scene(
        scene_id=scene_id,
        tier="D",
        objects=objects,
        frames=new_frames,
        qa=[],
        extras={
            "source": "arkitscenes",
            "split": src_dir.parent.name,
            "n_frames": n_frames,
            "window_sec": window_sec,
            "image_size": [int(w), int(h)],
            "probe_mode": "cam_motion_only",
        },
    )
    scene.save(scene_out_dir)
    return scene


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert ARKitScenes 3dod scenes to Scene schema (Tier D, cam-motion only)")
    p.add_argument("--arkit-root", required=True)
    p.add_argument("--split", default="Validation", choices=["Training", "Validation"])
    p.add_argument("--out", required=True)
    p.add_argument("--n-frames", type=int, default=16)
    p.add_argument("--window-sec", type=float, default=15.0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    root = Path(args.arkit_root) / "3dod" / args.split
    if not root.exists():
        raise SystemExit(f"ARKit root not found: {root}")
    scene_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.limit is not None:
        scene_dirs = scene_dirs[: args.limit]
    out = ensure_dir(args.out)

    try:
        from tqdm import tqdm
        it = tqdm(scene_dirs, desc="Tier D (ARKitScenes)")
    except ImportError:
        it = scene_dirs

    n_ok = 0
    for d in it:
        try:
            convert_arkit_scene(d, out, n_frames=args.n_frames, window_sec=args.window_sec)
            n_ok += 1
        except Exception as e:
            print(f"[skip] {d.name}: {type(e).__name__}: {e}")
    print(f"converted {n_ok}/{len(scene_dirs)} scenes → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
