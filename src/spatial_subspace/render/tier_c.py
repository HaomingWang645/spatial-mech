"""Tier C — perspective ego-video over a 3D scene (plan §3.3).

Same canonical 3D scene as Tier A/B, but rendered with a pinhole perspective
camera that orbits the scene at a fixed altitude. Each base scene gets *N*
independent trajectories so we can run the cross-trajectory H2 test.

Why a hand-rolled rasterizer instead of Blender / Kubric:
  - Tier C's purpose is to test whether the model's spatial subspace exists
    when the input has real depth and parallax, not whether the model can
    parse photoreal textures or lighting. CLEVR's whole point is "if reasoning
    fails, it fails on reasoning, not perception" — so we keep the rendering
    minimal on purpose.
  - Hard-dependency on Blender / bpy adds 2 GB of binary install plus a
    GPU-bound render pass per frame, blowing up the per-scene cost from
    milliseconds to seconds for no information gain on the probing question.
  - PIL gives pixel-exact segmentation masks for free via the painters
    algorithm, which is what the extraction pipeline needs.

Geometry simplification:
  - All shapes (cube/sphere/cylinder) are rendered as filled circles whose
    pixel radius is computed from the perspective projection of the object's
    3D centroid and its world-space size. The model can still perceive depth
    via apparent size and parallax — what it cannot do is distinguish cube
    from sphere from cylinder visually. The probe target is *position*, not
    *shape*, so this does not affect the Q1 / H2 measurement; it only means
    the auto-generated QA strings about "the red cube" no longer match the
    visual rendering. We do not use those QA strings in extraction.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..scene import Camera, Frame, Scene
from ..utils import ensure_dir, load_yaml, set_seed
from .common import generate_3d_scene


# ---------------------------------------------------------------------------
# Camera math (OpenCV convention: x right, y down, z forward into the scene)
# ---------------------------------------------------------------------------


def look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> tuple[np.ndarray, np.ndarray]:
    """Build (R, t) such that p_cam = R @ p_world + t."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    norm = np.linalg.norm(x)
    if norm < 1e-6:
        # forward parallel to up — pick an alternative up axis
        alt = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(z, alt)
        norm = np.linalg.norm(x)
    x /= norm
    y = np.cross(z, x)

    R = np.stack([x, y, z], axis=0)
    t = -R @ eye
    return R, t


def project(
    p_world: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    near: float = 0.05,
) -> tuple[float, float, float] | None:
    """World point → (u, v, depth) or None if behind/at the camera."""
    p_cam = R @ np.asarray(p_world, dtype=np.float64) + t
    if p_cam[2] <= near:
        return None
    inv_z = 1.0 / p_cam[2]
    u = K[0, 0] * p_cam[0] * inv_z + K[0, 2]
    v = K[1, 1] * p_cam[1] * inv_z + K[1, 2]
    return float(u), float(v), float(p_cam[2])


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


def sample_trajectory(
    scene: Scene,
    cfg: dict[str, Any],
    rng: random.Random,
    traj_idx: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One smooth orbit pose per frame.

    ``traj_idx`` selects from the (radius, altitude) pairs in ``cfg.trajectory``
    and also flips orbit direction and start angle, so different ``traj_idx``
    values for the same scene give meaningfully different camera paths.
    """
    n = int(cfg["n_frames"])
    tcfg = cfg["trajectory"]
    radii = list(tcfg["radii"])
    altitudes = list(tcfg["altitudes"])
    arc_deg = float(tcfg.get("arc_degrees", 180.0))
    look_z = float(tcfg.get("look_at_z", 0.5))

    coords = np.array([o.centroid for o in scene.objects])
    cx, cy = float(coords[:, 0].mean()), float(coords[:, 1].mean())
    target = np.array([cx, cy, look_z])

    radius = radii[traj_idx % len(radii)]
    altitude = altitudes[traj_idx % len(altitudes)]
    n_variants = max(len(radii), 1)
    start_angle = (traj_idx / n_variants) * math.pi
    direction = 1.0 if traj_idx % 2 == 0 else -1.0

    arc = math.radians(arc_deg)
    poses: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n):
        frac = i / max(1, n - 1)
        angle = start_angle + direction * arc * frac
        eye = np.array(
            [
                cx + radius * math.cos(angle),
                cy + radius * math.sin(angle),
                altitude,
            ]
        )
        poses.append((eye, target))
    return poses


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------


def _draw_frame(
    scene: Scene,
    cfg: dict[str, Any],
    eye: np.ndarray,
    target: np.ndarray,
) -> tuple[Image.Image, Image.Image, list[list[float]], list[list[float]]]:
    image_size = int(cfg["image_size"])
    bg = int(cfg["background_gray"])
    fov_deg = float(cfg.get("fov_degrees", 60.0))

    f = image_size / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cx_px = cy_px = image_size / 2.0
    K = np.array([[f, 0.0, cx_px], [0.0, f, cy_px], [0.0, 0.0, 1.0]])

    R, t = look_at(eye, target, up=np.array([0.0, 0.0, 1.0]))

    # Project all objects, drop ones behind the camera or vanishingly small
    rendered: list[tuple[float, Any, float, float, float]] = []
    for obj in scene.objects:
        proj = project(np.array(obj.centroid), R, t, K)
        if proj is None:
            continue
        u, v, depth = proj
        size_world = float(cfg["sizes"][obj.size])
        r_px = f * size_world / depth
        if r_px < 0.5:
            continue
        rendered.append((depth, obj, u, v, r_px))

    # Painters algorithm: far → near
    rendered.sort(key=lambda x: -x[0])

    img = Image.new("RGB", (image_size, image_size), (bg, bg, bg))
    mask = Image.new("L", (image_size, image_size), 0)
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)

    for _depth, obj, u, v, r_px in rendered:
        rgb = tuple(cfg["colors"][obj.color])
        bbox = (u - r_px, v - r_px, u + r_px, v + r_px)
        mid = obj.object_id + 1
        draw_img.ellipse(bbox, fill=rgb)
        draw_mask.ellipse(bbox, fill=mid)

    K_list = K.tolist()
    E_list = [
        [float(R[0, 0]), float(R[0, 1]), float(R[0, 2]), float(t[0])],
        [float(R[1, 0]), float(R[1, 1]), float(R[1, 2]), float(t[1])],
        [float(R[2, 0]), float(R[2, 1]), float(R[2, 2]), float(t[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return img, mask, K_list, E_list


# ---------------------------------------------------------------------------
# Top-level render
# ---------------------------------------------------------------------------


def render_tier_c(
    scene: Scene,
    cfg: dict[str, Any],
    out_dir: Path,
    rng: random.Random,
    traj_idx: int = 0,
) -> Scene:
    """Render one Tier C trajectory of an existing 3D Scene.

    Output scene_id is ``<base_scene_id>_t<traj_idx>``. The base scene id is
    preserved in ``extras["base_scene_id"]`` so cross-trajectory analysis can
    group on it.
    """
    poses = sample_trajectory(scene, cfg, rng, traj_idx)
    out_id = f"{scene.scene_id}_t{traj_idx}"
    scene_dir = ensure_dir(out_dir / out_id)
    ensure_dir(scene_dir / "frames")
    ensure_dir(scene_dir / "masks")

    new_frames: list[Frame] = []
    for i, (eye, target) in enumerate(poses):
        img, mask, K, E = _draw_frame(scene, cfg, eye, target)
        img_path = f"frames/{i:03d}.png"
        mask_path = f"masks/{i:03d}.png"
        img.save(scene_dir / img_path)
        mask.save(scene_dir / mask_path)
        new_frames.append(
            Frame(
                frame_id=i,
                image_path=img_path,
                mask_path=mask_path,
                camera=Camera(intrinsics=K, extrinsics=E, kind="perspective"),
            )
        )

    scene_out = Scene(
        scene_id=out_id,
        tier="C",
        objects=scene.objects,
        frames=new_frames,
        qa=list(scene.qa),
        extras={
            **scene.extras,
            "image_size": int(cfg["image_size"]),
            "n_frames": int(cfg["n_frames"]),
            "fov_degrees": float(cfg.get("fov_degrees", 60.0)),
            "trajectory_idx": int(traj_idx),
            "base_scene_id": scene.scene_id,
        },
    )
    scene_out.save(scene_dir)
    return scene_out


def render_scene(cfg: dict[str, Any], rng: random.Random, out_dir: Path) -> Scene:
    """Single-step convenience: sample a 3D scene and render its trajectory 0."""
    scene = generate_3d_scene(cfg, rng, tier="3D")
    return render_tier_c(scene, cfg, out_dir, rng, traj_idx=0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render Tier C perspective ego-videos")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--scenes-in",
        default=None,
        help="Directory of pre-generated 3D scenes; if set, --n-scenes is ignored",
    )
    p.add_argument("--n-scenes", type=int, default=None)
    p.add_argument("--n-frames", type=int, default=None)
    p.add_argument(
        "--trajectories-per-scene",
        type=int,
        default=2,
        help="Independent orbit trajectories per base scene (H2 test needs ≥2)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    cfg = load_yaml(args.config)
    if args.n_frames is not None:
        cfg["n_frames"] = int(args.n_frames)
    out = ensure_dir(Path(args.out))
    set_seed(args.seed)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **_):
            return x

    if args.scenes_in is not None:
        scenes_dir = Path(args.scenes_in)
        scene_dirs = sorted(
            d for d in scenes_dir.iterdir() if d.is_dir() and (d / "scene.json").exists()
        )
        if args.limit is not None:
            scene_dirs = scene_dirs[: args.limit]
        for d in tqdm(scene_dirs, desc="Tier C (render existing)"):
            scene = Scene.load(d)
            for traj_idx in range(args.trajectories_per_scene):
                rng = random.Random(
                    args.seed ^ (hash((scene.scene_id, traj_idx)) & 0x7FFFFFFF)
                )
                render_tier_c(scene, cfg, out, rng, traj_idx=traj_idx)
    else:
        n = int(args.n_scenes if args.n_scenes is not None else cfg.get("n_scenes", 5000))
        if args.limit is not None:
            n = min(n, args.limit)
        master_rng = random.Random(args.seed)
        for _ in tqdm(range(n), desc="Tier C (sample + render)"):
            sub_rng = random.Random(master_rng.randint(0, 2**31 - 1))
            scene = generate_3d_scene(cfg, sub_rng, tier="3D")
            for traj_idx in range(args.trajectories_per_scene):
                tr_rng = random.Random(master_rng.randint(0, 2**31 - 1))
                render_tier_c(scene, cfg, out, tr_rng, traj_idx=traj_idx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
