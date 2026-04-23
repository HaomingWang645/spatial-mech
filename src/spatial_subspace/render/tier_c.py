"""Tier C — perspective ego-video over a 3D scene (plan §3.3).

Same canonical 3D scene as Tier A/B, but rendered with a pinhole perspective
camera that orbits the scene. Each base scene gets *N* independent
trajectories so we can run the cross-trajectory H2 test.

Rendering is a hand-rolled PIL rasterizer — good enough for the probing
question while keeping per-scene cost in the millisecond range. We do two
things differently from the earlier circle-only version:

  1. **Proper silhouettes per shape.** Cubes are rendered as the 2D convex
     hull of their 8 projected corners (a hexagonal silhouette under typical
     oblique views); cylinders as the convex hull of sampled rim points on
     top and bottom circles; spheres as a circle at the projected centroid
     with apparent radius ``f · size / depth`` (the small-angle approximation
     of a sphere's silhouette is sufficient at the depths used here). The
     segmentation mask fills exactly the same polygon.
  2. **Ground shadows.** Each object's silhouette on the floor plane is
     computed under a fixed oblique sun (NW 135°, elevation 55° — same as
     Tier A/B); the shadow polygon is the 2D convex hull of the object's
     ground-plane footprint plus its shadow-shifted copy, projected into the
     image and filled in a dark grey before any objects are drawn. Two-pass
     painter order means objects always occlude shadows.

Geometry simplification that remains: shapes are still rigid ideal primitives,
no per-material lighting, no textures, and we do not render an explicit floor
grid — just the solid background colour. Adding proper lighting would change
colour histograms and make colour-name readout harder without affecting the
spatial-probing question.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..scene import Camera, Frame, Object3D, Scene
from ..utils import ensure_dir, load_yaml, set_seed
from .common import generate_3d_scene
from .tier_a import SHADOW_RGB, _shadow_offset_world


# ---------------------------------------------------------------------------
# Camera math (OpenCV convention: x right, y down, z forward into the scene)
# ---------------------------------------------------------------------------


def look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = np.array([0.0, 0.0, 1.0]),
    roll: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (R, t) such that p_cam = R @ p_world + t.

    ``roll`` rotates the (x, y) image basis around the camera forward axis,
    giving the 6th DoF on top of the (eye, target) pose.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    norm = np.linalg.norm(x)
    if norm < 1e-6:
        alt = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(z, alt)
        norm = np.linalg.norm(x)
    x /= norm
    y = np.cross(z, x)

    if roll != 0.0:
        cr, sr = math.cos(roll), math.sin(roll)
        x_new = cr * x + sr * y
        y_new = -sr * x + cr * y
        x, y = x_new, y_new

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
# Geometry → polygon helpers
# ---------------------------------------------------------------------------


def _convex_hull_2d(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone chain. Returns vertices in counter-clockwise order."""
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


def _object_silhouette_samples(
    obj: Object3D,
    size_world: float,
    n_samples: int = 16,
) -> list[tuple[float, float, float]]:
    """World-space sample points whose 2D convex hull is the object's silhouette."""
    cx, cy, cz = obj.centroid
    floor_z = obj.bbox_min[2]
    top_z = obj.bbox_max[2]
    s = size_world

    if obj.shape == "cube":
        xs = [cx - s, cx + s]
        ys = [cy - s, cy + s]
        zs = [floor_z, top_z]
        return [(x, y, z) for x in xs for y in ys for z in zs]

    if obj.shape == "cylinder":
        pts: list[tuple[float, float, float]] = []
        for k in range(n_samples):
            th = 2.0 * math.pi * k / n_samples
            px = cx + s * math.cos(th)
            py = cy + s * math.sin(th)
            pts.append((px, py, floor_z))
            pts.append((px, py, top_z))
        return pts

    if obj.shape == "sphere":
        # Handled via direct circle projection in _object_drawable, so the
        # silhouette sampler is only called for non-sphere shapes. Keeping a
        # fallback here (a sphere's hull is well approximated by rim samples
        # on a few latitudes) in case the caller uses it anyway.
        pts = []
        for i in range(1, 4):
            phi = math.pi * i / 4.0
            r_ring = s * math.sin(phi)
            z = cz + s * math.cos(phi)
            for k in range(n_samples):
                th = 2.0 * math.pi * k / n_samples
                pts.append((cx + r_ring * math.cos(th), cy + r_ring * math.sin(th), z))
        return pts

    raise ValueError(f"unknown shape: {obj.shape}")


def _object_footprint_world(
    obj: Object3D,
    size_world: float,
    n_samples: int = 16,
) -> list[tuple[float, float, float]]:
    """Points on the floor plane tracing the object's ground-plane footprint."""
    cx, cy, _ = obj.centroid
    floor_z = obj.bbox_min[2]
    s = size_world

    if obj.shape == "cube":
        return [
            (cx - s, cy - s, floor_z),
            (cx + s, cy - s, floor_z),
            (cx + s, cy + s, floor_z),
            (cx - s, cy + s, floor_z),
        ]
    # sphere / cylinder: disc of radius s on the floor
    pts: list[tuple[float, float, float]] = []
    for k in range(n_samples):
        th = 2.0 * math.pi * k / n_samples
        pts.append((cx + s * math.cos(th), cy + s * math.sin(th), floor_z))
    return pts


def _shadow_polygon_world(
    obj: Object3D,
    size_world: float,
    n_samples: int = 16,
) -> list[tuple[float, float, float]]:
    """Shadow footprint on the floor: object's ground footprint + its shadow-shifted copy."""
    sdx, sdy = _shadow_offset_world(2.0 * size_world)
    base = _object_footprint_world(obj, size_world, n_samples=n_samples)
    shifted = [(x + sdx, y + sdy, z) for (x, y, z) in base]
    return base + shifted


def _project_points(
    pts_world: list[tuple[float, float, float]],
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
) -> tuple[list[tuple[float, float]], float] | None:
    """Project world points; return (2D list, mean depth) or None if fewer
    than three are in front of the camera.
    """
    projected: list[tuple[float, float]] = []
    depths: list[float] = []
    for p in pts_world:
        proj = project(np.asarray(p, dtype=np.float64), R, t, K)
        if proj is None:
            continue
        projected.append((proj[0], proj[1]))
        depths.append(proj[2])
    if len(projected) < 3:
        return None
    return projected, float(np.mean(depths))


def _polygon_touches_image(
    poly: list[tuple[float, float]], image_size: int, margin: float = 2.0
) -> bool:
    min_u = min(p[0] for p in poly)
    max_u = max(p[0] for p in poly)
    min_v = min(p[1] for p in poly)
    max_v = max(p[1] for p in poly)
    return not (max_u < -margin or min_u > image_size + margin
                or max_v < -margin or min_v > image_size + margin)


# ---------------------------------------------------------------------------
# Visibility (used by the free-6DoF trajectory to guarantee ≥1 object in-frame)
# ---------------------------------------------------------------------------


def _has_visible_object(
    eye: np.ndarray,
    target: np.ndarray,
    roll: float,
    scene: Scene,
    K: np.ndarray,
    image_size: int,
    cfg: dict[str, Any],
    min_r_px: float,
    margin_px: float,
) -> bool:
    R, t = look_at(eye, target, roll=roll)
    f = float(K[0, 0])
    for obj in scene.objects:
        proj = project(np.array(obj.centroid), R, t, K)
        if proj is None:
            continue
        u, v, depth = proj
        size_world = float(cfg["sizes"][obj.size])
        r_px = f * size_world / depth
        if r_px < min_r_px:
            continue
        if u < -r_px - margin_px or u > image_size + r_px + margin_px:
            continue
        if v < -r_px - margin_px or v > image_size + r_px + margin_px:
            continue
        return True
    return False


def _repair_visibility(
    eye: np.ndarray,
    target: np.ndarray,
    roll: float,
    scene: Scene,
    K: np.ndarray,
    image_size: int,
    cfg: dict[str, Any],
    max_iters: int,
    min_r_px: float,
    margin_px: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if _has_visible_object(eye, target, roll, scene, K, image_size, cfg, min_r_px, margin_px):
        return eye, target, roll
    coords = np.array([o.centroid for o in scene.objects], dtype=np.float64)
    d2 = np.sum((coords[:, :2] - target[:2]) ** 2, axis=1)
    anchor = coords[int(np.argmin(d2))].copy()
    for k in range(1, max_iters + 1):
        alpha = k / max_iters
        candidate = (1.0 - alpha) * target + alpha * anchor
        if _has_visible_object(eye, candidate, roll, scene, K, image_size, cfg, min_r_px, margin_px):
            return eye, candidate, roll
    return eye, anchor, roll


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------


Pose = tuple[np.ndarray, np.ndarray, float]  # (eye, target, roll)


def _smooth_noise(
    n_frames: int,
    n_modes: int,
    amp: float,
    rng: random.Random,
    dim: int = 1,
) -> np.ndarray:
    if n_modes <= 0 or amp == 0.0:
        return np.zeros((n_frames, dim))
    t = np.linspace(0.0, 1.0, n_frames)
    out = np.zeros((n_frames, dim))
    for d in range(dim):
        for k in range(1, n_modes + 1):
            phi = rng.uniform(0.0, 2.0 * math.pi)
            a = rng.uniform(-1.0, 1.0) / k
            out[:, d] += a * np.sin(2.0 * math.pi * k * t + phi)
        peak = float(np.max(np.abs(out[:, d]))) + 1e-9
        out[:, d] *= amp / peak
    return out


def _sample_orbit_trajectory(
    scene: Scene,
    cfg: dict[str, Any],
    rng: random.Random,
    traj_idx: int,
) -> list[Pose]:
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
    poses: list[Pose] = []
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
        poses.append((eye, target.copy(), 0.0))
    return poses


def _sample_free6dof_trajectory(
    scene: Scene,
    cfg: dict[str, Any],
    rng: random.Random,
    traj_idx: int,
) -> list[Pose]:
    n = int(cfg["n_frames"])
    tcfg = cfg["trajectory"]
    fcfg = tcfg.get("free6dof", {})

    base_radii = list(fcfg.get("base_radii", tcfg.get("radii", [8.0])))
    base_altitudes = list(fcfg.get("base_altitudes", tcfg.get("altitudes", [3.5])))
    arc_deg = float(fcfg.get("arc_degrees", tcfg.get("arc_degrees", 220.0)))
    look_z = float(fcfg.get("look_at_z", tcfg.get("look_at_z", 0.5)))

    n_modes = int(fcfg.get("n_modes", 3))
    eye_jitter = float(fcfg.get("eye_jitter", 1.0))
    radius_jitter = float(fcfg.get("radius_jitter", 1.0))
    altitude_jitter = float(fcfg.get("altitude_jitter", 0.8))
    target_jitter = float(fcfg.get("target_jitter", 1.2))
    target_z_jitter = float(fcfg.get("target_z_jitter", 0.4))
    roll_max = math.radians(float(fcfg.get("roll_max_degrees", 20.0)))

    min_r_px = float(fcfg.get("visibility_min_radius_px", 2.0))
    margin_px = float(fcfg.get("visibility_margin_px", 0.0))
    repair_iters = int(fcfg.get("repair_max_iters", 8))

    image_size = int(cfg["image_size"])
    fov_deg = float(cfg.get("fov_degrees", 60.0))
    f = image_size / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    K = np.array([[f, 0.0, image_size / 2.0], [0.0, f, image_size / 2.0], [0.0, 0.0, 1.0]])

    coords = np.array([o.centroid for o in scene.objects])
    cx, cy = float(coords[:, 0].mean()), float(coords[:, 1].mean())
    center = np.array([cx, cy, look_z])

    base_radius = base_radii[traj_idx % len(base_radii)]
    base_alt = base_altitudes[traj_idx % len(base_altitudes)]
    n_variants = max(len(base_radii), 1)
    start_angle = (traj_idx / n_variants) * 2.0 * math.pi
    direction = 1.0 if traj_idx % 2 == 0 else -1.0
    arc = math.radians(arc_deg)

    radius_drift = _smooth_noise(n, n_modes, radius_jitter, rng).flatten()
    altitude_drift = _smooth_noise(n, n_modes, altitude_jitter, rng).flatten()
    eye_jit = _smooth_noise(n, n_modes, eye_jitter, rng, dim=3)
    target_xy_jit = _smooth_noise(n, n_modes, target_jitter, rng, dim=2)
    target_z_jit = _smooth_noise(n, n_modes, target_z_jitter, rng).flatten()
    roll_seq = _smooth_noise(n, n_modes, roll_max, rng).flatten()

    poses: list[Pose] = []
    for i in range(n):
        frac = i / max(1, n - 1)
        angle = start_angle + direction * arc * frac
        radius = max(2.0, base_radius + radius_drift[i])
        altitude = max(0.5, base_alt + altitude_drift[i])
        eye = (
            np.array([cx + radius * math.cos(angle), cy + radius * math.sin(angle), altitude])
            + eye_jit[i]
        )
        target = center + np.array([target_xy_jit[i, 0], target_xy_jit[i, 1], target_z_jit[i]])
        roll = float(roll_seq[i])
        eye, target, roll = _repair_visibility(
            eye, target, roll, scene, K, image_size, cfg,
            max_iters=repair_iters, min_r_px=min_r_px, margin_px=margin_px,
        )
        poses.append((eye, target, roll))
    return poses


def _sample_person_walk_trajectory(
    scene: Scene,
    cfg: dict[str, Any],
    rng: random.Random,
    traj_idx: int,
) -> list[Pose]:
    """Truly 6-DoF random walk of a person-like camera through the scene.

    Constraints:
      1. Camera position never goes inside an object (cylindrical collision
         check around each object's centroid with a safety margin).
      2. Every object is captured (centroid projects inside the image, in
         front of the camera) in at least one frame. If the initial sample
         fails, we retry up to ``max_retries`` times.
      3. Not every frame contains every object — that emerges naturally
         because the camera looks "forward" (along its walking direction)
         instead of always at scene centre, so many frames have only a
         subset of objects in view.

    Config keys under ``trajectory.person_walk``:
      eye_height         — fixed z coord of the camera (default 1.5 m)
      speed_mean         — average forward speed (world units per frame)
      speed_amp          — smooth-noise amplitude on top of speed_mean
      yaw_rate_deg       — max per-frame yaw change
      pitch_deg          — max pitch magnitude (looks up/down slightly)
      object_margin      — safety margin around objects for collision (m)
      bounds_margin      — stay this far from working_volume edges
      max_retries        — attempts before giving up on coverage
    """
    n = int(cfg["n_frames"])
    tcfg = cfg["trajectory"]
    pcfg = tcfg.get("person_walk", {})

    eye_height    = float(pcfg.get("eye_height", 1.5))
    speed_mean    = float(pcfg.get("speed_mean", 0.5))
    speed_amp     = float(pcfg.get("speed_amp", 0.25))
    yaw_rate_deg  = float(pcfg.get("yaw_rate_deg", 25.0))
    pitch_deg_amp = float(pcfg.get("pitch_deg", 12.0))
    object_margin = float(pcfg.get("object_margin", 0.6))
    bounds_margin = float(pcfg.get("bounds_margin", 0.3))
    max_retries   = int(pcfg.get("max_retries", 40))

    x_lo, x_hi = cfg["working_volume"]["x"]
    y_lo, y_hi = cfg["working_volume"]["y"]
    image_size = int(cfg["image_size"])
    fov_deg = float(cfg.get("fov_degrees", 60.0))
    f_px = image_size / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    K = np.array([[f_px, 0.0, image_size / 2.0],
                  [0.0, f_px, image_size / 2.0],
                  [0.0, 0.0, 1.0]])

    # Object collision cylinders (xy-disc of radius size + margin).
    obj_cyls = []
    for o in scene.objects:
        size = float(cfg["sizes"][o.size])
        obj_cyls.append((float(o.centroid[0]), float(o.centroid[1]), size + object_margin))

    def collides(x: float, y: float) -> bool:
        for ox, oy, r in obj_cyls:
            if (x - ox) ** 2 + (y - oy) ** 2 < r * r:
                return True
        return False

    def clamp_bounds(x: float, y: float) -> tuple[float, float]:
        return (
            max(x_lo + bounds_margin, min(x_hi - bounds_margin, x)),
            max(y_lo + bounds_margin, min(y_hi - bounds_margin, y)),
        )

    def _poly_inframe_ratio(pts: list[tuple[float, float]]) -> float:
        """Fraction of a 2D polygon's area that falls inside the image. 0 if
        the polygon is entirely off-screen or too small to rasterize."""
        if len(pts) < 3:
            return 0.0
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
        if bx1 <= 0 or by1 <= 0 or bx0 >= image_size or by0 >= image_size:
            return 0.0
        margin = 5
        cw = int(bx1 - bx0) + 2 * margin + 1
        ch = int(by1 - by0) + 2 * margin + 1
        shift_x, shift_y = margin - bx0, margin - by0
        full_im = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(full_im).polygon(
            [(p[0] + shift_x, p[1] + shift_y) for p in pts], fill=1
        )
        full_area = int(np.asarray(full_im).sum())
        if full_area < 1:
            return 0.0
        view_im = Image.new("L", (image_size, image_size), 0)
        ImageDraw.Draw(view_im).polygon(list(pts), fill=1)
        view_area = int(np.asarray(view_im).sum())
        return view_area / full_area

    def object_inframe_ratio(eye, target, roll, obj) -> float:
        """Fraction of ``obj``'s 2D silhouette that falls inside this frame.
        Matches the ≥50%-in-frame criterion for coverage."""
        R_c, t_c = look_at(eye, target, roll=roll)
        size_world = float(cfg["sizes"][obj.size])
        if obj.shape == "sphere":
            proj = project(np.asarray(obj.centroid, dtype=np.float64), R_c, t_c, K)
            if proj is None:
                return 0.0
            u, v, depth = proj
            r_px = f_px * size_world / depth
            if r_px < 1.0:
                return 0.0
            n = 32
            pts = [(u + r_px * math.cos(2 * math.pi * k / n),
                    v + r_px * math.sin(2 * math.pi * k / n)) for k in range(n)]
            return _poly_inframe_ratio(pts)
        samples = _object_silhouette_samples(obj, size_world)
        proj = _project_points(samples, R_c, t_c, K)
        if proj is None:
            return 0.0
        projected, _ = proj
        hull = _convex_hull_2d(projected)
        return _poly_inframe_ratio([(float(u), float(v)) for u, v in hull])

    def frame_has_object(eye, target, roll, obj) -> bool:
        """Back-compat name; wraps the new ≥50% inframe check."""
        return object_inframe_ratio(eye, target, roll, obj) >= 0.5

    # Scene centre (for biasing the start pose toward the object cluster).
    obj_xy = np.array([[o.centroid[0], o.centroid[1]] for o in scene.objects])
    scene_cx, scene_cy = float(obj_xy[:, 0].mean()), float(obj_xy[:, 1].mean())
    # Scene radius (farthest object from centre).
    obj_r = float(np.linalg.norm(obj_xy - np.array([scene_cx, scene_cy]), axis=1).max())

    for attempt in range(max_retries):
        # Smart start: position on the scene periphery, facing the scene centre.
        # Bias distance so we start outside the object cluster but still inside
        # the working volume. Per-traj offset keeps different traj_idx exploring
        # different initial directions.
        peripheral_angle = rng.uniform(0.0, 2.0 * math.pi)
        peripheral_angle = (peripheral_angle + 2.0 * math.pi * (traj_idx % 4) / 4.0) % (2.0 * math.pi)
        for _ in range(200):
            radius = rng.uniform(obj_r + 0.5, obj_r + 1.8)
            sx = scene_cx + radius * math.cos(peripheral_angle)
            sy = scene_cy + radius * math.sin(peripheral_angle)
            sx, sy = clamp_bounds(sx, sy)
            if not collides(sx, sy):
                break
            peripheral_angle += rng.uniform(0.3, 0.7)
        # Initial yaw: face scene centre with small jitter.
        start_yaw = math.atan2(scene_cy - sy, scene_cx - sx) + rng.uniform(-math.pi / 6, math.pi / 6)

        # Smooth noise tracks
        speed_track = _smooth_noise(n, 2, speed_amp, rng).flatten() + speed_mean
        yaw_rate_track = _smooth_noise(n, 3, yaw_rate_deg, rng).flatten()
        pitch_track = _smooth_noise(n, 2, pitch_deg_amp, rng).flatten()

        pos = np.array([sx, sy, eye_height], dtype=np.float64)
        yaw = start_yaw
        poses: list[Pose] = []
        for i in range(n):
            yaw += math.radians(yaw_rate_track[i])
            pitch_rad = math.radians(pitch_track[i])
            forward = np.array([
                math.cos(yaw) * math.cos(pitch_rad),
                math.sin(yaw) * math.cos(pitch_rad),
                -math.sin(pitch_rad),
            ])
            step = speed_track[i] * forward
            new_x = pos[0] + step[0]
            new_y = pos[1] + step[1]
            # Clamp + reject collisions: if we would hit an object, rotate 90°
            new_x, new_y = clamp_bounds(new_x, new_y)
            hits = collides(new_x, new_y)
            if hits:
                # Emergency-turn: sidestep at 90° from current heading
                side_sign = 1.0 if rng.random() > 0.5 else -1.0
                yaw += side_sign * math.pi / 2.0
                sidef = np.array([math.cos(yaw), math.sin(yaw), 0.0])
                new_x, new_y = clamp_bounds(pos[0] + speed_track[i] * sidef[0],
                                            pos[1] + speed_track[i] * sidef[1])
                if collides(new_x, new_y):
                    # give up on moving this frame; stay in place
                    new_x, new_y = pos[0], pos[1]
            pos = np.array([new_x, new_y, eye_height])
            target = pos + forward
            poses.append((pos.copy(), target.copy(), 0.0))

        # Coverage check: every object visible in at least one frame
        missed = []
        for o in scene.objects:
            if not any(frame_has_object(p[0], p[1], p[2], o) for p in poses):
                missed.append(o.object_id)
        if not missed:
            return poses
        # else: retry

    # Fallback: return last attempt even if coverage is incomplete
    return poses


def sample_trajectory(
    scene: Scene,
    cfg: dict[str, Any],
    rng: random.Random,
    traj_idx: int,
) -> list[Pose]:
    mode = str(cfg.get("trajectory", {}).get("mode", "orbit")).lower()
    if mode == "orbit":
        return _sample_orbit_trajectory(scene, cfg, rng, traj_idx)
    if mode == "free6dof":
        return _sample_free6dof_trajectory(scene, cfg, rng, traj_idx)
    if mode == "person_walk":
        return _sample_person_walk_trajectory(scene, cfg, rng, traj_idx)
    raise ValueError(f"unknown trajectory mode: {mode!r} (expected 'orbit', 'free6dof', or 'person_walk')")


# ---------------------------------------------------------------------------
# Drawables
# ---------------------------------------------------------------------------


def _shadow_drawable(
    obj: Object3D,
    cfg: dict[str, Any],
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    image_size: int,
):
    size_world = float(cfg["sizes"][obj.size])
    poly_world = _shadow_polygon_world(obj, size_world)
    proj = _project_points(poly_world, R, t, K)
    if proj is None:
        return None
    projected, mean_depth = proj
    hull = _convex_hull_2d(projected)
    if len(hull) < 3 or not _polygon_touches_image(hull, image_size):
        return None
    poly = [(float(u), float(v)) for (u, v) in hull]

    def draw_img(d):
        d.polygon(poly, fill=SHADOW_RGB)

    return mean_depth, draw_img, None


def _object_drawable(
    obj: Object3D,
    cfg: dict[str, Any],
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    f: float,
    image_size: int,
):
    size_world = float(cfg["sizes"][obj.size])
    rgb = tuple(cfg["colors"][obj.color])
    mid = obj.object_id + 1

    if obj.shape == "sphere":
        proj = project(np.array(obj.centroid, dtype=np.float64), R, t, K)
        if proj is None:
            return None
        u, v, depth = proj
        r_px = f * size_world / depth
        if r_px < 0.5:
            return None
        bbox = (u - r_px, v - r_px, u + r_px, v + r_px)
        if (u + r_px < -2 or u - r_px > image_size + 2
                or v + r_px < -2 or v - r_px > image_size + 2):
            return None

        def draw_img(d):
            d.ellipse(bbox, fill=rgb)

        def draw_mask(d):
            d.ellipse(bbox, fill=mid)

        return depth, draw_img, draw_mask

    samples = _object_silhouette_samples(obj, size_world)
    proj = _project_points(samples, R, t, K)
    if proj is None:
        return None
    projected, mean_depth = proj
    hull = _convex_hull_2d(projected)
    if len(hull) < 3 or not _polygon_touches_image(hull, image_size):
        return None
    poly = [(float(u), float(v)) for (u, v) in hull]

    def draw_img(d):
        d.polygon(poly, fill=rgb)

    def draw_mask(d):
        d.polygon(poly, fill=mid)

    return mean_depth, draw_img, draw_mask


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------


def _draw_frame(
    scene: Scene,
    cfg: dict[str, Any],
    eye: np.ndarray,
    target: np.ndarray,
    roll: float = 0.0,
) -> tuple[Image.Image, Image.Image, list[list[float]], list[list[float]]]:
    image_size = int(cfg["image_size"])
    bg = int(cfg["background_gray"])
    fov_deg = float(cfg.get("fov_degrees", 60.0))

    f = image_size / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cx_px = cy_px = image_size / 2.0
    K = np.array([[f, 0.0, cx_px], [0.0, f, cy_px], [0.0, 0.0, 1.0]])
    R, t = look_at(eye, target, up=np.array([0.0, 0.0, 1.0]), roll=roll)

    shadows = []
    objects = []
    for obj in scene.objects:
        sd = _shadow_drawable(obj, cfg, R, t, K, image_size)
        if sd is not None:
            shadows.append(sd)
        od = _object_drawable(obj, cfg, R, t, K, f, image_size)
        if od is not None:
            objects.append(od)

    # Painter order inside each pass; two passes keep shadows strictly behind
    # all objects (so a near object can occlude a far object's shadow but a
    # near object's shadow never occludes a far object's body).
    shadows.sort(key=lambda x: -x[0])
    objects.sort(key=lambda x: -x[0])

    img = Image.new("RGB", (image_size, image_size), (bg, bg, bg))
    mask = Image.new("L", (image_size, image_size), 0)
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)

    for _depth, fn_img, _fn_mask in shadows:
        fn_img(draw_img)
    for _depth, fn_img, fn_mask in objects:
        fn_img(draw_img)
        if fn_mask is not None:
            fn_mask(draw_mask)

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
    for i, (eye, target, roll) in enumerate(poses):
        img, mask, K, E = _draw_frame(scene, cfg, eye, target, roll=roll)
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
            "trajectory_mode": str(cfg.get("trajectory", {}).get("mode", "orbit")).lower(),
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
