#!/usr/bin/env python
"""Multi-model text-output activation steering.

Generalization of `activation_steering_text.py` that dispatches on the
`family` field of the model YAML (qwen25vl / internvl3 / llava_onevision)
so the same experiment can be run on all three open VLMs.

Per (scene, pair) trial:
  1. Fit a Ridge probe on residual-stream activations at `--steer-layer`
     mapping pooled object-token activations -> per-scene-normalized 3D
     coordinates. Take v_axis = pseudo-inverse row for the chosen axis.
  2. Build a perpendicular control v_perp via projection onto null-space
     of W (probe coefficients), rescaled to ||v_axis||.
  3. Forward the (frames, comparison-prompt) once with no intervention to
     get the baseline logit-gap between {colorA, colorB}.
  4. For each (direction in {v_axis, v_perp}, alpha) inject delta=alpha*v
     onto object-A's residual at `--steer-layer` and re-decode the same
     prompt; record the new logit-gap.

Outputs:
  steering_text.parquet            (one row per (scene, pair, direction, alpha))
  steering_text_agg.csv            (per (direction, alpha) mean d-gap)
  steering_text.png                (per-direction d-gap vs alpha)
  flip_summary.json                (per-direction flip rate vs baseline argmax)

Example:
  python scripts/activation_steering_text_multi.py \
      --scenes data/tier_c_free6dof \
      --activations data/activations/tier_c_free6dof_qwen25vl_7b \
      --model-config configs/models/qwen25vl.yaml \
      --steer-layer 17 --t-min 4 --axis x \
      --alpha-values=-3,-1.5,0,1.5,3 \
      --n-scenes 10 --n-pairs-per-scene 3 \
      --out data/steering/tier_c_free6dof_qwen7b_text_L17_x
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
from spatial_subspace.models import (
    Qwen25VLWrapper,
    InternVL3Wrapper,
    LlavaOnevisionWrapper,
)
from spatial_subspace.scene import Scene
from spatial_subspace.utils import ensure_dir, load_yaml, set_seed


WRAPPERS = {
    "qwen25vl":        Qwen25VLWrapper,
    "internvl3":       InternVL3Wrapper,
    "llava_onevision": LlavaOnevisionWrapper,
}


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


def object_token_positions(scene, scene_dir, wrapper, visual_start, grid,
                           threshold, t_min, object_id):
    t_post, gh, gw = grid
    tps = wrapper.temporal_patch_size()
    img_h = gh * wrapper.patch_pixels()
    img_w = gw * wrapper.patch_pixels()
    masks = [np.array(Image.open(scene_dir / f.mask_path)) for f in scene.frames]
    masks_resized = [
        np.array(Image.fromarray(m).resize((img_w, img_h), Image.NEAREST)) for m in masks
    ]
    positions = []
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
    ids = processor.tokenizer.encode(" " + color, add_special_tokens=False)
    if not ids:
        raise RuntimeError(f"empty tokenization for color {color!r}")
    return int(ids[0])


def logit_gap(logits_last: np.ndarray, tok_a: int, tok_b: int) -> float:
    return float(logits_last[tok_a] - logits_last[tok_b])


def run_forward(wrapper, scene, scene_dir, prompt, *, intervention=None) -> dict:
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
    p.add_argument("--n-scenes", type=int, default=10)
    p.add_argument("--n-pairs-per-scene", type=int, default=3)
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

    scene_ids = sorted(d.name for d in scenes_dir.iterdir()
                       if d.is_dir() and (d / "scene.json").exists())
    random.Random(args.seed).shuffle(scene_ids)
    scene_ids = scene_ids[: args.n_scenes]
    alpha_values = [float(a) for a in args.alpha_values.split(",")]

    print(f"[model] loading {args.model_config}")
    mcfg = load_yaml(args.model_config)
    family = mcfg.get("family")
    if family not in WRAPPERS:
        raise SystemExit(f"unsupported family {family!r}; got config {args.model_config}")
    Wrapper = WRAPPERS[family]
    wrapper = Wrapper(
        hf_id=mcfg["hf_id"],
        torch_dtype=mcfg.get("torch_dtype", "bfloat16"),
        device=mcfg.get("device", "cuda"),
        device_map=mcfg.get("device_map"),
    )
    print(f"[model] family={family}  layer={args.steer_layer}")

    try:
        results = []
        for sidx, sid in enumerate(scene_ids):
            scene_dir = scenes_dir / sid
            scene = Scene.load(scene_dir)
            pairs = []
            for oa, ob in combinations(range(len(scene.objects)), 2):
                a, b = scene.objects[oa], scene.objects[ob]
                if a.color == b.color or a.centroid[axis_idx] == b.centroid[axis_idx]:
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
                      f"A={A.color}#{oa}  B={B.color}#{ob}  Δaxis(A−B)={gt_gap:+.2f}")

                base = run_forward(wrapper, scene, scene_dir, prompt)
                base_gap = logit_gap(base["logits_last"], tok_a, tok_b)
                vstart = base["visual_start"]
                grid = base["grid"]
                posA = object_token_positions(
                    scene, scene_dir, wrapper, vstart, grid,
                    args.threshold, args.t_min, oa,
                )
                results.append({
                    "scene_id": sid, "obj_a": int(oa), "obj_b": int(ob),
                    "colorA": A.color, "colorB": B.color,
                    "gt_axis_gap": gt_gap,
                    "direction": "baseline", "alpha": 0.0,
                    "logit_gap": base_gap,
                    "d_logit_gap": 0.0,
                    "n_positions": len(posA),
                })
                print(f"    baseline gap({A.color}−{B.color}) = {base_gap:+.3f}  n_steer_tok={len(posA)}")
                if not posA:
                    continue

                for direction_name, v in [("v_axis", v_steer), ("v_perp", v_perp)]:
                    for alpha in alpha_values:
                        if alpha == 0.0:
                            continue
                        fwd = run_forward(
                            wrapper, scene, scene_dir, prompt,
                            intervention=(args.steer_layer, posA, alpha * v),
                        )
                        gap = logit_gap(fwd["logits_last"], tok_a, tok_b)
                        results.append({
                            "scene_id": sid, "obj_a": int(oa), "obj_b": int(ob),
                            "colorA": A.color, "colorB": B.color,
                            "gt_axis_gap": gt_gap,
                            "direction": direction_name, "alpha": float(alpha),
                            "logit_gap": gap,
                            "d_logit_gap": gap - base_gap,
                            "n_positions": len(posA),
                        })
                        print(f"    [{direction_name} α={alpha:+.2f}] gap={gap:+.3f}  Δ={gap-base_gap:+.3f}")
    finally:
        wrapper.close()

    df = pd.DataFrame(results)
    df.to_parquet(out_dir / "steering_text.parquet")
    (out_dir / "steering_text.json").write_text(
        json.dumps(results, indent=2,
                   default=lambda x: float(x) if isinstance(x, np.floating) else x)
    )

    # Δ logit-gap aggregate
    interv = df[df.direction != "baseline"]
    agg = (interv.groupby(["direction", "alpha"])
                 .agg(mean_dgap=("d_logit_gap", "mean"),
                      std_dgap=("d_logit_gap", "std"),
                      n=("d_logit_gap", "count"))
                 .reset_index())
    agg.to_csv(out_dir / "steering_text_agg.csv", index=False)
    print("\n=== Aggregate Δ(logit_gap) ===")
    print(agg.to_string(index=False))

    # Answer-flip rate: per-trial baseline argmax vs intervention argmax.
    # baseline argmax = sign(logit_gap) > 0  → predicts colorA
    # The interesting flip is "did the predicted answer change relative to baseline?"
    flip_rows = []
    for (sid, oa, ob), g in df.groupby(["scene_id", "obj_a", "obj_b"]):
        base_pred = (g[g.direction == "baseline"].iloc[0].logit_gap > 0)
        for _, row in g[g.direction != "baseline"].iterrows():
            flip = bool((row.logit_gap > 0) != base_pred)
            flip_rows.append({
                "scene_id": sid, "obj_a": oa, "obj_b": ob,
                "direction": row.direction, "alpha": row.alpha,
                "flip": flip,
            })
    flip_df = pd.DataFrame(flip_rows)
    flip_summary = (flip_df.groupby(["direction", "alpha"])
                           .agg(flip_rate=("flip", "mean"),
                                n=("flip", "count"))
                           .reset_index())
    flip_summary.to_csv(out_dir / "flip_summary.csv", index=False)
    (out_dir / "flip_summary.json").write_text(
        flip_summary.to_json(orient="records", indent=2))
    print("\n=== Flip rate (predicted answer differs from baseline) ===")
    print(flip_summary.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for d, color in [("v_axis", "C0"), ("v_perp", "C3")]:
        sub = agg[agg.direction == d].sort_values("alpha")
        lbl = f"v_{args.axis}  (axis subspace)" if d == "v_axis" else "v_perp  (null-space ctrl)"
        axes[0].errorbar(sub.alpha, sub.mean_dgap, yerr=sub.std_dgap,
                         fmt="-o", color=color, capsize=3, label=lbl)
    axes[0].axhline(0, color="gray", lw=0.6); axes[0].axvline(0, color="gray", lw=0.6)
    axes[0].set_xlabel(f"α  ({args.axis} units)")
    axes[0].set_ylabel("Δ logit gap (colorA − colorB)")
    axes[0].set_title("Δ logit gap")
    axes[0].grid(alpha=0.3); axes[0].legend()

    for d, color in [("v_axis", "C0"), ("v_perp", "C3")]:
        sub = flip_summary[flip_summary.direction == d].sort_values("alpha")
        axes[1].plot(sub.alpha, sub.flip_rate, "-o", color=color,
                     label=f"v_{args.axis}" if d == "v_axis" else "v_perp")
    axes[1].set_xlabel(f"α  ({args.axis} units)")
    axes[1].set_ylabel("Answer-flip rate vs baseline")
    axes[1].set_title("Answer flip rate")
    axes[1].grid(alpha=0.3); axes[1].legend()
    axes[1].set_ylim(-0.02, 1.02)

    fig.suptitle(f"{family} L{args.steer_layer} axis={args.axis}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "steering_text.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_dir}/{{steering_text.png, steering_text_agg.csv, flip_summary.csv, *.parquet}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
