"""Tier D — real-world 3D scenes from ARKitScenes (plan §3.4).

Adapter, not a renderer: each raw ARKitScenes 3dod scene ships with

    <scene>/
        <scene>_3dod_annotation.json   # 3D oriented bboxes for objects
        <scene>_3dod_mesh.ply          # scene mesh (unused here)
        <scene>_frames/
            lowres_wide/*.png          # RGB frames (256x192, iPhone LiDAR cam)
            lowres_wide_intrinsics/*.pincam
            lowres_wide.traj           # camera-to-world poses
            lowres_depth/*.png         # LiDAR depth (unused for now)

and we reshape it into the common Scene schema used by Tiers A/B/C:

    <scene>/
        scene.json
        frames/<i>.png                 # 16 frames sampled over a window
        masks/<i>.png                  # per-object mask from projecting 3D bboxes

Per-object masks are produced by transforming each 3D oriented bbox into the
camera frame via the inverse of the ARKit camera-to-world pose, projecting all
8 corners with the frame's intrinsics, and filling the 2D convex hull of the
visible corners. The mask's object_id is the obb's ``objectId`` (+1 to reserve
0 for background, matching the other tiers' convention).

Camera convention:
  ARKitScenes `.traj` stores a camera-to-world pose ``(R_cw, t_cw)`` using
  a right-handed frame with +x right, +y down, +z forward (OpenCV-style) —
  verified empirically: inverting the pose without an axis flip puts the 3D
  object bboxes in front of the camera and the projected silhouettes land
  on the right pixels. The world-to-camera extrinsic stored in Scene.json
  is therefore simply ``(R_cw.T, -R_cw.T @ t_cw)``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    """Return list of ``(timestamp, R_cw, t_cw)`` sorted by timestamp."""
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
    """Return (K, width, height) for the .pincam file closest to ``frame_ts``."""
    candidates = sorted(k_dir.glob("*.pincam"))
    if not candidates:
        raise FileNotFoundError(f"no .pincam under {k_dir}")
    ts_list = [float(p.stem.split("_")[-1]) for p in candidates]
    idx = int(np.argmin(np.abs(np.array(ts_list) - frame_ts)))
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


def _load_bboxes(annot_path: Path) -> list[dict]:
    """Return per-object {objectId, label, centroid_world (3,), corners_world (8,3)}."""
    raw = json.loads(Path(annot_path).read_text())
    out: list[dict] = []
    for item in raw.get("data", []):
        if not item.get("segments", {}).get("obbAligned"):
            continue
        obb = item["segments"]["obbAligned"]
        c = np.array(obb["centroid"], dtype=np.float64)
        axes = np.array(obb["normalizedAxes"], dtype=np.float64).reshape(3, 3)
        half = np.array(obb["axesLengths"], dtype=np.float64) / 2.0
        # 8 corners in world frame.
        corners = np.array(
            [
                c + sx * half[0] * axes[0] + sy * half[1] * axes[1] + sz * half[2] * axes[2]
                for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
            ]
        )
        out.append(
            {
                "uid": str(item["uid"]),
                "class_id": int(item["objectId"]),   # ARKit "objectId" is a class/category id
                "label": str(item.get("label", "object")),
                "centroid_world": c,
                "corners_world": corners,
                "axes_half": half,
            }
        )
    return out


def _convex_hull_2d(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set((round(x, 6), round(y, 6)) for x, y in points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _project_corners(
    corners_world: np.ndarray,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    K: np.ndarray,
    w: int,
    h: int,
    near: float = 0.30,
) -> tuple[list[tuple[float, float]], float] | None:
    """Project a 3D bbox's 8 corners to 2D. Returns (polygon, mean depth) or None.

    We require **all** 8 corners to be at depth ≥ ``near`` (0.30 m by default):
    partially-clipped bboxes have one or two corners with depth ≈ 0, which
    then project to (u, v) ≈ ∞ and blow up the 2D convex hull to cover the
    whole frame. Rather than implement proper near-plane polygon clipping, we
    require the whole bbox to be comfortably in front of the camera. Objects
    close enough to have any corner behind the near plane are simply dropped
    from that frame's mask.

    Additionally, we require at least one corner to land inside the image —
    otherwise the object is outside the FoV and the mask contribution is nil.
    """
    cam_z = []
    projected: list[tuple[float, float]] = []
    for p in corners_world:
        p_cam = R_wc @ p + t_wc
        cam_z.append(float(p_cam[2]))
        if p_cam[2] <= near:
            return None  # any corner behind near plane → drop whole bbox
        u = float(K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2])
        v = float(K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2])
        projected.append((u, v))

    # Require ≥1 corner strictly inside the image (not just a hull that grazes).
    inside = any(0 <= u <= w and 0 <= v <= h for u, v in projected)
    if not inside:
        return None

    return projected, float(np.mean(cam_z))


def _sample_timestamps(
    all_ts: list[float], n_frames: int, window_sec: float
) -> list[float]:
    """n_frames evenly spaced timestamps over a centred ``window_sec`` clip."""
    if not all_ts:
        return []
    t_min, t_max = all_ts[0], all_ts[-1]
    total = t_max - t_min
    if total < window_sec or total < 1e-3:
        window_start = t_min
        window_end = t_max
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
            idx = last_idx + 1  # avoid picking the same frame twice
        selected.append(ts_arr[idx])
        last_idx = idx
    return selected


def convert_arkit_scene(
    src_dir: Path,
    out_dir: Path,
    n_frames: int = 16,
    window_sec: float = 15.0,
) -> Scene:
    """Convert one ARKitScenes scene to our Scene schema; write frames + masks + JSON."""
    src_dir = Path(src_dir)
    scene_id = src_dir.name
    frames_root = src_dir / f"{scene_id}_frames"
    wide_dir = frames_root / "lowres_wide"
    intrinsics_dir = frames_root / "lowres_wide_intrinsics"
    traj_path = frames_root / "lowres_wide.traj"
    annot_path = src_dir / f"{scene_id}_3dod_annotation.json"

    if not wide_dir.exists() or not traj_path.exists() or not annot_path.exists():
        raise FileNotFoundError(
            f"expected ARKitScenes layout not found at {src_dir}; "
            f"missing one of lowres_wide/, lowres_wide.traj, *_3dod_annotation.json"
        )

    bboxes = _load_bboxes(annot_path)
    traj = _load_trajectory(traj_path)

    frame_files = sorted(wide_dir.glob(f"{scene_id}_*.png"))
    frame_ts = [float(p.stem.split("_")[-1]) for p in frame_files]
    if not frame_ts:
        raise RuntimeError(f"no frames found under {wide_dir}")
    selected_ts = _sample_timestamps(frame_ts, n_frames, window_sec)
    ts_to_file = dict(zip(frame_ts, frame_files))

    scene_out_dir = ensure_dir(out_dir / scene_id)
    ensure_dir(scene_out_dir / "frames")
    ensure_dir(scene_out_dir / "masks")

    # Build per-object canonical order (0..N-1) and mapping objectId → local id.
    # Object3D world coords stay in the ARKit world frame; the extrinsic
    # (computed below per-frame) encodes the ARKit→OpenCV-camera flip, so the
    # downstream depth probe `p_cam = E @ p_world_arkit` lands correctly.
    # Local object_id is the enumeration index; ARKit's ``objectId`` is a
    # category class id that repeats across objects in the scene, so we can't
    # use it as a per-instance key.
    objects: list[Object3D] = []
    for i, b in enumerate(bboxes):
        centroid = b["centroid_world"]
        corners = b["corners_world"]
        bbox_min = corners.min(axis=0)
        bbox_max = corners.max(axis=0)
        objects.append(
            Object3D(
                object_id=i,
                shape="obbox",
                color="unknown",
                size=b["label"],
                centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                bbox_min=(float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])),
                bbox_max=(float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])),
                extras={
                    "arkit_uid": b["uid"],
                    "arkit_class_id": int(b["class_id"]),
                    "label": b["label"],
                    "axes_half": [float(x) for x in b["axes_half"]],
                },
            )
        )

    # Iterate selected frames.
    new_frames: list[Frame] = []
    for i, ts in enumerate(selected_ts):
        src_png = ts_to_file[ts]
        K, w, h = _load_intrinsics(intrinsics_dir, ts)
        R_cw, t_cw = _find_nearest_pose(traj, ts)

        # camera-to-world (ARKit) → world-to-camera (OpenCV). The empirical
        # check (see module docstring) confirms the ARKit pose is already in
        # an OpenCV-compatible frame, so no axis flip is needed.
        R_wc_opencv = R_cw.T
        t_wc_opencv = -R_cw.T @ t_cw

        # Copy the RGB frame to our schema path. Link instead of copy for speed.
        dst_png = scene_out_dir / "frames" / f"{i:03d}.png"
        if not dst_png.exists():
            try:
                dst_png.symlink_to(src_png.resolve())
            except OSError:
                Image.open(src_png).save(dst_png)

        # Paint the mask by projecting each object's 8 corners.
        mask = Image.new("L", (w, h), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        # Sort far→near for painter's order within the mask.
        drawables = []
        for local_id, b in enumerate(bboxes):
            proj = _project_corners(
                b["corners_world"], R_wc_opencv, t_wc_opencv, K, w, h
            )
            if proj is None:
                continue
            poly, depth = proj
            hull = _convex_hull_2d(poly)
            if len(hull) < 3:
                continue
            drawables.append((depth, hull, local_id + 1))

        drawables.sort(key=lambda x: -x[0])
        for depth, hull, mid in drawables:
            draw.polygon(hull, fill=mid)

        mask.save(scene_out_dir / "masks" / f"{i:03d}.png")

        # Save Camera. Image is copied as-is; K is already in pixels of the
        # original lowres_wide image. Extrinsics are world-to-camera.
        E_4x4 = [
            [float(R_wc_opencv[0, 0]), float(R_wc_opencv[0, 1]), float(R_wc_opencv[0, 2]), float(t_wc_opencv[0])],
            [float(R_wc_opencv[1, 0]), float(R_wc_opencv[1, 1]), float(R_wc_opencv[1, 2]), float(t_wc_opencv[1])],
            [float(R_wc_opencv[2, 0]), float(R_wc_opencv[2, 1]), float(R_wc_opencv[2, 2]), float(t_wc_opencv[2])],
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
            "split": src_dir.parent.name,  # Training or Validation
            "n_frames": n_frames,
            "window_sec": window_sec,
            "image_size": [int(w), int(h)],
        },
    )
    scene.save(scene_out_dir)
    return scene


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert ARKitScenes 3dod scenes to our Scene schema (Tier D)")
    p.add_argument("--arkit-root", required=True,
                   help="Directory containing 3dod/<split>/<scene_id>/...")
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
