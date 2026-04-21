"""Activation extraction pipeline.

Given a VLM wrapper and a directory of rendered scenes, runs forward passes,
maps each object's segmentation mask onto the visual-token grid, pools the
hidden states of tokens whose patch overlaps the mask, and writes one
``(parquet, npy)`` pair per layer.

On-disk layout:
    <out_dir>/layer_<LL>.parquet   # metadata rows
    <out_dir>/layer_<LL>.npy       # (N_rows, hidden_dim) float32

Metadata columns: scene_id, object_id, frame_id, layer, vec_row,
centroid_x, centroid_y, centroid_z.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from PIL import Image

from .models import ForwardOut, VLMWrapper
from .scene import Scene


@dataclass
class ExtractConfig:
    overlap_threshold: float = 0.3
    layers: list[int] | None = None


def mask_to_patch_coverage(
    mask: np.ndarray,
    grid_hw: tuple[int, int],
    object_ids: list[int],
) -> dict[int, np.ndarray]:
    """Downsample an (H, W) object-id mask onto a patch grid.

    The mask PNG stores object_id + 1 (with 0 reserved for background), so we
    add 1 when testing equality. Returns one (gh, gw) float32 map per
    object_id giving the fraction of each patch's pixels belonging to it.
    """
    H, W = mask.shape
    gh, gw = grid_hw
    ph, pw = H // gh, W // gw
    if ph == 0 or pw == 0:
        raise ValueError(f"mask {H}x{W} too small for grid {gh}x{gw}")
    mask = mask[: gh * ph, : gw * pw]
    reshaped = mask.reshape(gh, ph, gw, pw)
    out: dict[int, np.ndarray] = {}
    for oid in object_ids:
        pixel_id = oid + 1
        out[oid] = (reshaped == pixel_id).astype(np.float32).mean(axis=(1, 3))
    return out


def pool_object_vector(
    visual_hidden: np.ndarray,     # (gh, gw, D) one frame, one layer
    coverage: np.ndarray,          # (gh, gw) in [0, 1]
    threshold: float,
) -> np.ndarray | None:
    sel = coverage >= threshold
    if not sel.any():
        return None
    vecs = visual_hidden[sel]
    weights = coverage[sel]
    return (vecs * weights[:, None]).sum(axis=0) / weights.sum()


def extract_scene(
    wrapper: VLMWrapper,
    scene: Scene,
    scene_dir: Path,
    prompt: str,
    cfg: ExtractConfig,
) -> Iterator[tuple[dict, np.ndarray]]:
    """Yield (metadata_row, hidden_vector) for every (object, frame, layer).

    Each frame is processed as an independent image. Use this for Tier A
    or for the "infinite shuffle" ablation where temporal context is removed.
    """
    object_ids = [o.object_id for o in scene.objects]
    for frame in scene.frames:
        img = Image.open(scene_dir / frame.image_path).convert("RGB")
        mask = np.array(Image.open(scene_dir / frame.mask_path))

        out = wrapper.forward(img, prompt)
        img_h, img_w = wrapper.image_input_hw(out)
        mask_resized = np.array(
            Image.fromarray(mask).resize((img_w, img_h), Image.NEAREST)
        )

        _, gh, gw = out.grid
        coverage = mask_to_patch_coverage(mask_resized, (gh, gw), object_ids)
        visual_positions = out.extras.get("visual_positions")
        start, end = out.visual_token_range

        layer_ids = (
            cfg.layers if cfg.layers is not None else list(range(len(out.hidden_states)))
        )
        for layer_idx in layer_ids:
            hs = out.hidden_states[layer_idx][0]           # (T, D)
            if visual_positions is not None:
                visual = hs[visual_positions].float().cpu().numpy()
            else:
                visual = hs[start:end].float().cpu().numpy()   # (gh*gw, D)
            visual = visual.reshape(gh, gw, -1)
            for oid in object_ids:
                vec = pool_object_vector(visual, coverage[oid], cfg.overlap_threshold)
                if vec is None:
                    continue
                obj = scene.objects[oid]
                yield (
                    {
                        "scene_id": scene.scene_id,
                        "object_id": oid,
                        "frame_id": frame.frame_id,
                        "layer": int(layer_idx),
                        "centroid_x": float(obj.centroid[0]),
                        "centroid_y": float(obj.centroid[1]),
                        "centroid_z": float(obj.centroid[2]),
                    },
                    vec.astype(np.float32),
                )


def extract_scene_video(
    wrapper: "VLMWrapper",
    scene: Scene,
    scene_dir: Path,
    prompt: str,
    cfg: ExtractConfig,
) -> Iterator[tuple[dict, np.ndarray]]:
    """Yield (metadata_row, hidden_vector) by processing all frames as a video.

    Qwen2.5-VL applies a temporal patch merger of size ``temporal_patch_size``
    (=2 by default), so an N-frame input produces N/temporal_patch_size
    temporal tokens. Per temporal token we average the source frames' masks
    so each "temporal slot" gets a fused coverage map. The metadata
    ``frame_id`` field then encodes the temporal-token index, not an input
    frame index.
    """
    object_ids = [o.object_id for o in scene.objects]
    if not scene.frames:
        return

    frame_paths = [
        f"file://{(scene_dir / f.image_path).resolve()}" for f in scene.frames
    ]
    masks_raw = [np.array(Image.open(scene_dir / f.mask_path)) for f in scene.frames]

    out = wrapper.forward(frame_paths, prompt)
    img_h, img_w = wrapper.image_input_hw(out)
    masks_resized = [
        np.array(Image.fromarray(m).resize((img_w, img_h), Image.NEAREST))
        for m in masks_raw
    ]

    t_post, gh, gw = out.grid
    tps = wrapper.temporal_patch_size() if hasattr(wrapper, "temporal_patch_size") else 2
    expected_in = t_post * tps
    if len(masks_resized) != expected_in:
        raise RuntimeError(
            f"frame count {len(masks_resized)} does not match T*tps={expected_in} "
            f"(t_post={t_post}, temporal_patch_size={tps})"
        )

    # Per temporal token: average the tps source-frame coverage maps.
    coverage_per_t: list[dict[int, np.ndarray]] = []
    for t in range(t_post):
        slot = masks_resized[t * tps : (t + 1) * tps]
        slot_covs = [mask_to_patch_coverage(m, (gh, gw), object_ids) for m in slot]
        merged: dict[int, np.ndarray] = {}
        for oid in object_ids:
            merged[oid] = np.mean(np.stack([sc[oid] for sc in slot_covs]), axis=0)
        coverage_per_t.append(merged)

    # Two modes for visual-token extraction from the raw sequence:
    #   (a) contiguous range  — most models (Qwen2.5-VL, LLaVA-OV post-strip)
    #   (b) gathered positions — models that interleave frame markers between
    #       patch runs (e.g. InternVL3 wraps each frame in <img>...</img>)
    visual_positions = out.extras.get("visual_positions")
    start, end = out.visual_token_range
    layer_ids = (
        cfg.layers if cfg.layers is not None else list(range(len(out.hidden_states)))
    )
    for layer_idx in layer_ids:
        hs = out.hidden_states[layer_idx][0]                    # (T_seq, D)
        if visual_positions is not None:
            visual = hs[visual_positions].float().cpu().numpy()
        else:
            visual = hs[start:end].float().cpu().numpy()        # (t_post*gh*gw, D)
        visual = visual.reshape(t_post, gh, gw, -1)
        for t in range(t_post):
            for oid in object_ids:
                vec = pool_object_vector(
                    visual[t], coverage_per_t[t][oid], cfg.overlap_threshold
                )
                if vec is None:
                    continue
                obj = scene.objects[oid]
                yield (
                    {
                        "scene_id": scene.scene_id,
                        "object_id": oid,
                        "frame_id": int(t),  # temporal-token index
                        "layer": int(layer_idx),
                        "centroid_x": float(obj.centroid[0]),
                        "centroid_y": float(obj.centroid[1]),
                        "centroid_z": float(obj.centroid[2]),
                    },
                    vec.astype(np.float32),
                )


def run_extraction(
    wrapper: VLMWrapper,
    data_root: Path,
    out_dir: Path,
    prompt: str,
    cfg: ExtractConfig,
    tier: str | None = None,
    limit: int | None = None,
    mode: str = "image",
) -> None:
    """Run extraction over all scenes under ``data_root``.

    ``mode`` selects how multi-frame scenes are processed:
        "image"  — each frame is a separate forward pass (Tier A; or Tier B
                   "infinite shuffle" ablation since temporal context is gone)
        "video"  — all frames are bundled into a single video forward so the
                   model sees them with M-RoPE temporal positions (Tier B/C)
    """
    if mode not in ("image", "video"):
        raise ValueError(f"mode must be 'image' or 'video', got {mode!r}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted(
        d for d in Path(data_root).iterdir() if d.is_dir() and (d / "scene.json").exists()
    )
    if limit is not None:
        scene_dirs = scene_dirs[:limit]

    try:
        from tqdm import tqdm
        it = tqdm(scene_dirs, desc=f"extracting ({mode})")
    except ImportError:
        it = scene_dirs

    rows_by_layer: dict[int, list[dict]] = defaultdict(list)
    vecs_by_layer: dict[int, list[np.ndarray]] = defaultdict(list)

    extractor = extract_scene_video if mode == "video" else extract_scene

    for d in it:
        scene = Scene.load(d)
        if tier is not None and scene.tier != tier:
            continue
        for row, vec in extractor(wrapper, scene, d, prompt, cfg):
            layer = row["layer"]
            rows_by_layer[layer].append(row)
            vecs_by_layer[layer].append(vec)

    for layer, rows in rows_by_layer.items():
        df = pd.DataFrame(rows)
        df["vec_row"] = np.arange(len(rows), dtype=np.int64)
        df.to_parquet(out_dir / f"layer_{layer:02d}.parquet")
        np.save(out_dir / f"layer_{layer:02d}.npy", np.stack(vecs_by_layer[layer]))
