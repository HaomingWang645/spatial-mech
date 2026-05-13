#!/usr/bin/env python
"""Text-output activation-steering (plan §6.1, stronger version).

Upgrades the probe-on-probe test to a text-output test. For each steered
scene, we prompt the model with a world-frame comparison question:

    "In this scene, which object has a larger world-frame x-coordinate
     (further toward positive x)? Reply with one color word: {colorA} or
     {colorB}?"

and read the logit of the first answer token for each candidate. The signal
is the logit gap

    score = logit(first_token(" " + colorA)) − logit(first_token(" " + colorB))

If the model's spatial subspace is causally used for VERBAL answers, then
steering v_x with +α on object A's tokens at L_steer should make the model
more likely to say ``colorA`` (score ↑). The null-space control should
produce no consistent shift.

Each (scene, pair) contributes one score per (direction, α). We order each
pair so that A is the object on the LOWER ground-truth x, so a +α shift
that works pushes A's perceived-x higher and gives a more positive score
(baseline is negative: A < B in ground truth).

Usage:
  python scripts/activation_steering_text.py \
      --scenes data/tier_c_free6dof \
      --activations data/activations/tier_c_free6dof_qwen25vl_7b \
      --model-config configs/models/qwen25vl.yaml \
      --steer-layer 12 --t-min 4 --axis x \
      --alpha-values='-3,-1.5,0,1.5,3' \
      --n-scenes 5 --n-pairs-per-scene 2 \
      --out data/steering/tier_c_free6dof_7b_text_L12
"""
import _bootstrap  # noqa: F401

import argparse
import json
import random
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import Ridge

from spatial_subspace.extract import mask_to_patch_coverage
from spatial_subspace.labels import per_scene_normalized_coords
from spatial_subspace.models import Qwen25VLWrapper
from spatial_subspace.scene import Scene
from spatial_subspace.utils import ensure_dir, load_yaml, set_seed


# ---------------------------------------------------------------------------
# Probe + steering direction (mirrors activation_steering.py)
# ---------------------------------------------------------------------------


def fit_probe(activ_dir: Path, layer: int, t_min: int, alpha: float):
    meta = pd.read_parquet(activ_dir / f"layer_{layer:02d}.parquet")
    vecs = np.load(activ_dir / f"layer_{layer:02d}.npy")
    if t_min is not None:
        meta = meta[meta["frame_id"] >= t_min].reset_index(drop=True)
    pooled, lbls, scenes = [], [], []
    for (sid, oid), g in meta.groupby(["scene_id", "object_id"], sort=False):
        pooled.append(vecs[g["vec_row"].to_numpy()].mean(axis=0))
        row = g.iloc[0]
        lbls.append([row["centroid_x"], row["centroid_y"], row["centroid_z"]])
        scenes.append(sid)
    pooled = np.stack(pooled)
    lbls = np.asarray(lbls, dtype=np.float64)
    scenes = np.asarray(scenes)
    y = np.zeros_like(lbls, dtype=np.float64)
    for s in np.unique(scenes):
        m = scenes == s
        y[m] = per_scene_normalized_coords(lbls[m])
    mdl = Ridge(alpha=alpha).fit(pooled, y)
    return mdl, mdl.coef_.astype(np.float64), np.asarray(mdl.intercept_, dtype=np.float64)


def steering_directions(W: np.ndarray) -> np.ndarray:
    gram = W @ W.T
    return np.linalg.solve(gram, np.eye(3)) @ W  # (3, D)


def null_space_direction(W: np.ndarray, rng: np.random.Generator, target_norm: float) -> np.ndarray:
    r = rng.standard_normal(W.shape[1])
    gram = W @ W.T
    Wpinv = W.T @ np.linalg.inv(gram)
    r_null = r - Wpinv @ (W @ r)
    return r_null * (target_norm / np.linalg.norm(r_null))


# ---------------------------------------------------------------------------
# Visual-token position lookup
# ---------------------------------------------------------------------------


def object_token_positions(
    scene: Scene,
    scene_dir: Path,
    wrapper: Qwen25VLWrapper,
    visual_start: int,
    grid: tuple[int, int, int],
    threshold: float,
    t_min: int,
    object_id: int,
) -> list[int]:
    t_post, gh, gw = grid
    tps = wrapper.temporal_patch_size()
    img_h = gh * wrapper.patch_pixels()
    img_w = gw * wrapper.patch_pixels()
    masks = [np.array(Image.open(scene_dir / f.mask_path)) for f in scene.frames]
    masks_resized = [
        np.array(Image.fromarray(m).resize((img_w, img_h), Image.NEAREST)) for m in masks
    ]
    positions: list[int] = []
    for t in range(t_post):
        if t < t_min:
            continue
        covs = [
            mask_to_patch_coverage(masks_resized[t * tps + k], (gh, gw), [object_id])
            for k in range(tps)
        ]
        merged = np.mean(np.stack([c[object_id] for c in covs]), axis=0)
        ii, jj = np.where(merged >= threshold)
        base = visual_start + t * gh * gw
        for i, j in zip(ii, jj):
            positions.append(int(base + i * gw + j))
    return positions


# ---------------------------------------------------------------------------
# Prompting + logit extraction
# ---------------------------------------------------------------------------


def make_comparison_prompt(colorA: str, colorB: str, axis: str) -> str:
    axis_desc = {
        "x": "world-frame x-coordinate (further toward positive x)",
        "y": "world-frame y-coordinate (further toward positive y)",
        "z": "world-frame z-coordinate (higher up)",
    }[axis]
    return (
        f"This video shows a 3D scene filmed by an orbiting camera. "
        f"Which object has a larger {axis_desc}: "
        f"the {colorA} object or the {colorB} object? "
        f"Reply with one color word: {colorA} or {colorB}."
    )


def color_first_token(processor, color: str) -> int:
    # Prepend a space so we get the tokenizer's natural post-whitespace variant.
    ids = processor.tokenizer.encode(" " + color, add_special_tokens=False)
    if not ids:
        raise RuntimeError(f"empty tokenization for color {color!r}")
    return int(ids[0])


def logit_gap(logits_last: np.ndarray, tok_a: int, tok_b: int) -> float:
    return float(logits_last[tok_a] - logits_last[tok_b])


def run_forward_with_logits(
    wrapper: Qwen25VLWrapper,
    scene: Scene,
    scene_dir: Path,
    prompt: str,
    *,
    intervention=None,
) -> dict:
    handle = None
    if intervention is not None:
        lyr, positions, delta = intervention
        handle = wrapper.install_intervention(lyr, positions, delta)
    try:
        frame_paths = [
            f"file://{(scene_dir / f.image_path).resolve()}" for f in scene.frames
        ]
        out = wrapper.forward(frame_paths, prompt)
        logits_last = out.extras.get("logits_last")
        if logits_last is None:
            raise RuntimeError("wrapper did not return logits_last in extras")
    finally:
        if handle is not None:
            handle.remove()
    return {
        "logits_last": logits_last,
        "visual_start": out.visual_token_range[0],
        "grid": out.grid,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", required=True)
    p.add_argument("--activations", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steer-layer", type=int, required=True)
    p.add_argument("--t-min", type=int, default=4)
    p.add_argument("--alpha-probe", type=float, default=100.0)
    p.add_argument("--alpha-values", type=str, default="-3,-1.5,0,1.5,3")
    p.add_argument("--axis", choices=["x", "y", "z"], default="x")
    p.add_argument("--n-scenes", type=int, default=5)
    p.add_argument("--n-pairs-per-scene", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = ensure_dir(args.out)
    activ_dir = Path(args.activations)
    scenes_dir = Path(args.scenes)
    axis_idx = {"x": 0, "y": 1, "z": 2}[args.axis]

    print(f"[probe] fitting Ridge at L{args.steer_layer} (t≥{args.t_min}, alpha={args.alpha_probe})")
    _, W_steer, _ = fit_probe(activ_dir, args.steer_layer, args.t_min, args.alpha_probe)
    v_all = steering_directions(W_steer)
    v_steer = v_all[axis_idx]
    v_norm = float(np.linalg.norm(v_steer))
    v_perp = null_space_direction(W_steer, rng, v_norm)
    print(f"[probe] ||v_{args.axis}||={v_norm:.3f}  W @ v_perp = {W_steer @ v_perp}")

    # Pick scenes and in-scene pairs where object A has the lower ground-truth axis value
    scene_ids = sorted(d.name for d in scenes_dir.iterdir() if d.is_dir() and (d / "scene.json").exists())
    random.Random(args.seed).shuffle(scene_ids)
    scene_ids = scene_ids[: args.n_scenes]
    alpha_values = [float(a) for a in args.alpha_values.split(",")]

    print(f"[model] loading {args.model_config}")
    mcfg = load_yaml(args.model_config)
    wrapper = Qwen25VLWrapper(
        hf_id=mcfg["hf_id"],
        torch_dtype=mcfg.get("torch_dtype", "bfloat16"),
        device=mcfg.get("device", "cuda"),
        device_map=mcfg.get("device_map"),
    )
    prompt_template = mcfg["prompt"]  # kept for extract-consistency; unused here
    del prompt_template

    try:
        results: list[dict] = []
        for sidx, sid in enumerate(scene_ids):
            scene_dir = scenes_dir / sid
            scene = Scene.load(scene_dir)
            # Pair up objects with DISTINCT colors; order A = smaller axis value
            pairs = []
            for oa, ob in combinations(range(len(scene.objects)), 2):
                a, b = scene.objects[oa], scene.objects[ob]
                if a.color == b.color:
                    continue
                if a.centroid[axis_idx] == b.centroid[axis_idx]:
                    continue
                if a.centroid[axis_idx] < b.centroid[axis_idx]:
                    pairs.append((oa, ob))
                else:
                    pairs.append((ob, oa))
            random.Random(args.seed ^ hash(sid) & 0xFFFF).shuffle(pairs)
            pairs = pairs[: args.n_pairs_per_scene]

            for (oa, ob) in pairs:
                A, B = scene.objects[oa], scene.objects[ob]
                prompt = make_comparison_prompt(A.color, B.color, args.axis)
                tok_a = color_first_token(wrapper.processor, A.color)
                tok_b = color_first_token(wrapper.processor, B.color)
                gt_gap = float(A.centroid[axis_idx] - B.centroid[axis_idx])
                print(f"\n[{sidx+1}/{len(scene_ids)}] {sid}  "
                      f"A={A.color} obj{oa} (axis={A.centroid[axis_idx]:+.2f})  "
                      f"B={B.color} obj{ob} (axis={B.centroid[axis_idx]:+.2f})  "
                      f"gt Δaxis(A−B)={gt_gap:+.2f}")

                # Baseline forward
                base = run_forward_with_logits(wrapper, scene, scene_dir, prompt)
                base_gap = logit_gap(base["logits_last"], tok_a, tok_b)
                vstart = base["visual_start"]
                grid = base["grid"]
                posA = object_token_positions(
                    scene, scene_dir, wrapper, vstart, grid, args.threshold, args.t_min, oa
                )
                results.append({
                    "scene_id": sid, "obj_a": int(oa), "obj_b": int(ob),
                    "colorA": A.color, "colorB": B.color,
                    "gt_axis_gap": gt_gap,
                    "direction": "baseline", "alpha": 0.0,
                    "logit_gap": base_gap,
                    "n_positions": len(posA),
                })
                print(f"    baseline logit_gap({A.color}−{B.color}) = {base_gap:+.3f}  "
                      f"n_steer_tokens={len(posA)}")
                if not posA:
                    continue

                for direction_name, v in [("v_axis", v_steer), ("v_perp", v_perp)]:
                    for alpha in alpha_values:
                        if alpha == 0.0:
                            continue
                        fwd = run_forward_with_logits(
                            wrapper, scene, scene_dir, prompt,
                            intervention=(args.steer_layer, posA, alpha * v),
                        )
                        gap = logit_gap(fwd["logits_last"], tok_a, tok_b)
                        dgap = gap - base_gap
                        results.append({
                            "scene_id": sid, "obj_a": int(oa), "obj_b": int(ob),
                            "colorA": A.color, "colorB": B.color,
                            "gt_axis_gap": gt_gap,
                            "direction": direction_name, "alpha": float(alpha),
                            "logit_gap": gap,
                            "d_logit_gap": dgap,
                            "n_positions": len(posA),
                        })
                        print(f"    [{direction_name} α={alpha:+.2f}]  gap={gap:+.3f}  "
                              f"Δgap={dgap:+.3f}")
    finally:
        wrapper.close()

    df = pd.DataFrame(results)
    df.to_parquet(out_dir / "steering_text.parquet")
    (out_dir / "steering_text.json").write_text(
        json.dumps(results, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    )

    agg = (
        df[df.direction != "baseline"]
        .groupby(["direction", "alpha"])
        .agg(
            mean_dgap=("d_logit_gap", "mean"),
            std_dgap=("d_logit_gap", "std"),
            n=("d_logit_gap", "count"),
        )
        .reset_index()
    )
    agg.to_csv(out_dir / "steering_text_agg.csv", index=False)
    print("\n=== Aggregate Δ(logit_gap) by direction, α ===")
    print(agg.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for d, color in [("v_axis", "C0"), ("v_perp", "C3")]:
        sub = agg[agg.direction == d].sort_values("alpha")
        lab = f"v_{args.axis}  (in S)" if d == "v_axis" else "v_perp  (null-space of W)"
        ax.errorbar(sub.alpha, sub.mean_dgap, yerr=sub.std_dgap,
                    fmt="-o", color=color, capsize=3, label=lab)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel(f"steering α  (normalized {args.axis} coordinate units)")
    ax.set_ylabel(f"Δ logit gap  (colorA − colorB), relative to α=0 baseline")
    ax.set_title(
        f"Text-output steering — free6dof Tier C — "
        f"steer L{args.steer_layer} (axis={args.axis})\n"
        f"positive α pushes A's perceived axis-value higher; expect score ↑ for v_axis, 0 for v_perp"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "steering_text.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_dir}/{{steering_text.png, steering_text_agg.csv, steering_text.json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
