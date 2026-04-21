#!/usr/bin/env python
"""Q2 / H3 activation-steering experiment (plan §6.1).

For a chosen object o in a rendered scene, inject Δ = α·v into the hidden
states of o's visual tokens at a chosen "steer" LM layer L_steer. v is a
direction inside the spatial subspace S at L_steer derived from the Q1
probe's Ridge weights: v_x is the minimum-norm hidden-state direction whose
linear-probe readout is (1, 0, 0). v_y and v_z are defined analogously.

After the intervention propagates through layers L_steer..L_readout, we pool
the object's tokens at L_readout and apply that layer's probe to get the
model's "final" spatial readout for the object. The causal test: does the
readout shift in the direction of v?

Controls (plan §6.1):
  - v_perp: a random direction inside the orthogonal complement of S (the
    null-space of W_steer), normalised to the same ||.|| as v_x. Expectation:
    zero effect on the readout. This is the decisive control.

We report two quantities vs α and per direction:
  - Δ_readout = readout(α) − readout(0) in the probed axis.
  - Δ_readout.L2 across the 3-vec output.

Assumes a free6dof scene directory compatible with extract_scene_video.

Usage:
  python scripts/activation_steering.py \
      --scenes data/tier_c_free6dof \
      --activations data/activations/tier_c_free6dof_qwen25vl_7b \
      --model-config configs/models/qwen25vl.yaml \
      --steer-layer 12 --readout-layer 27 --t-min 4 \
      --alpha-values "-0.3,-0.15,0,0.15,0.3" \
      --n-scenes 5 --n-objects-per-scene 3 \
      --out data/steering/tier_c_free6dof_7b_L12
"""
import _bootstrap  # noqa: F401

import argparse
import json
import math
import random
from collections import defaultdict
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
# Probe & steering math
# ---------------------------------------------------------------------------


def fit_probe(activ_dir: Path, layer: int, t_min: int, alpha: float) -> tuple[Ridge, np.ndarray, np.ndarray]:
    """Fit the Q1 latter-frames Ridge probe at ``layer``. Returns (model, W, b).

    Targets are per-scene normalized coords; the per-(scene, object) summary
    is the mean over the activation rows at temporal tokens >= t_min.
    """
    meta = pd.read_parquet(activ_dir / f"layer_{layer:02d}.parquet")
    vecs = np.load(activ_dir / f"layer_{layer:02d}.npy")
    if t_min is not None:
        meta = meta[meta["frame_id"] >= t_min].reset_index(drop=True)

    pooled = []
    lbls = []
    for (sid, oid), g in meta.groupby(["scene_id", "object_id"], sort=False):
        pooled.append(vecs[g["vec_row"].to_numpy()].mean(axis=0))
        row = g.iloc[0]
        lbls.append([row["centroid_x"], row["centroid_y"], row["centroid_z"]])
    pooled = np.stack(pooled)
    lbls = np.asarray(lbls, dtype=np.float64)

    # Scene grouping for per-scene normalization
    scenes = np.asarray([s for (s, _) in meta.groupby(["scene_id", "object_id"], sort=False).groups.keys()])
    y = np.zeros_like(lbls, dtype=np.float64)
    for s in np.unique(scenes):
        mask = scenes == s
        y[mask] = per_scene_normalized_coords(lbls[mask])

    model = Ridge(alpha=alpha).fit(pooled, y)
    return model, model.coef_.astype(np.float64), np.asarray(model.intercept_, dtype=np.float64)


def steering_directions(W: np.ndarray) -> np.ndarray:
    """Return (3, D): rows are min-norm v_x, v_y, v_z such that W @ v = e_i."""
    gram = W @ W.T
    alphas = np.linalg.solve(gram, np.eye(3))
    return alphas @ W  # (3, D)


def null_space_direction(W: np.ndarray, rng: np.random.Generator, target_norm: float) -> np.ndarray:
    r = rng.standard_normal(W.shape[1])
    gram = W @ W.T
    Wpinv = W.T @ np.linalg.inv(gram)
    r_null = r - Wpinv @ (W @ r)
    n = float(np.linalg.norm(r_null))
    if n < 1e-12:
        raise RuntimeError("null-space projection gave near-zero vector")
    return r_null * (target_norm / n)


# ---------------------------------------------------------------------------
# Object → token-position lookup (mirrors extract_scene_video)
# ---------------------------------------------------------------------------


def object_token_positions(
    scene: Scene,
    scene_dir: Path,
    wrapper: Qwen25VLWrapper,
    visual_start: int,
    grid: tuple[int, int, int],
    threshold: float,
    t_min: int | None,
    object_id: int,
) -> tuple[list[int], list[int]]:
    """Positions in the input sequence of patches whose mask-coverage for
    ``object_id`` exceeds ``threshold``. Returns (positions, t_indices).
    """
    t_post, gh, gw = grid
    tps = wrapper.temporal_patch_size()
    img_h, img_w = gh * wrapper.patch_pixels(), gw * wrapper.patch_pixels()

    masks = [np.array(Image.open(scene_dir / f.mask_path)) for f in scene.frames]
    masks_resized = [
        np.array(Image.fromarray(m).resize((img_w, img_h), Image.NEAREST)) for m in masks
    ]
    positions: list[int] = []
    tidx: list[int] = []
    for t in range(t_post):
        if t_min is not None and t < t_min:
            continue
        pair_covs = [
            mask_to_patch_coverage(masks_resized[t * tps + k], (gh, gw), [object_id])
            for k in range(tps)
        ]
        merged = np.mean(np.stack([c[object_id] for c in pair_covs]), axis=0)
        ii, jj = np.where(merged >= threshold)
        base = visual_start + t * gh * gw
        for i, j in zip(ii, jj):
            positions.append(int(base + i * gw + j))
            tidx.append(int(t))
    return positions, tidx


def pool_at_layer(
    hidden_state: np.ndarray,       # (T_seq, D) at a given layer
    token_positions: list[int],
) -> np.ndarray:
    return hidden_state[np.asarray(token_positions, dtype=np.int64)].mean(axis=0)


# ---------------------------------------------------------------------------
# One-shot forward + intervention
# ---------------------------------------------------------------------------


def run_forward(
    wrapper: Qwen25VLWrapper,
    scene: Scene,
    scene_dir: Path,
    prompt: str,
    *,
    intervention: tuple[int, list[int], np.ndarray] | None = None,
) -> dict:
    """Returns dict with ``hidden_states`` (list of (T_seq, D) np arrays),
    ``visual_start``, ``grid``. If ``intervention`` is given as
    ``(layer_idx, positions, delta_vec)``, installs a pre-hook for that layer.
    """
    handle = None
    if intervention is not None:
        lyr, positions, delta = intervention
        handle = wrapper.install_intervention(lyr, positions, delta)
    try:
        frame_paths = [
            f"file://{(scene_dir / f.image_path).resolve()}" for f in scene.frames
        ]
        out = wrapper.forward(frame_paths, prompt)
        hidden = [hs[0].float().cpu().numpy() for hs in out.hidden_states]
    finally:
        if handle is not None:
            handle.remove()
    return {
        "hidden": hidden,
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
    p.add_argument("--readout-layer", type=int, required=True)
    p.add_argument("--t-min", type=int, default=4)
    p.add_argument("--alpha-probe", type=float, default=100.0,
                   help="Ridge alpha for fitting the probe used as steering oracle")
    p.add_argument("--alpha-values", type=str, default="-0.3,-0.15,0,0.15,0.3")
    p.add_argument("--axis", choices=["x", "y", "z"], default="x")
    p.add_argument("--n-scenes", type=int, default=5)
    p.add_argument("--n-objects-per-scene", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = ensure_dir(args.out)
    activ_dir = Path(args.activations)
    scenes_dir = Path(args.scenes)

    print(f"[probe] fitting probes at L{args.steer_layer} and L{args.readout_layer} "
          f"(t≥{args.t_min}, alpha={args.alpha_probe})")
    _steer_model, W_steer, b_steer = fit_probe(activ_dir, args.steer_layer, args.t_min, args.alpha_probe)
    _readout_model, W_read, b_read = fit_probe(activ_dir, args.readout_layer, args.t_min, args.alpha_probe)

    v_all = steering_directions(W_steer)  # (3, D)
    axis_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    v_steer = v_all[axis_idx]
    v_norm = float(np.linalg.norm(v_steer))
    v_perp = null_space_direction(W_steer, rng, v_norm)
    # Sanity: W_steer @ v_steer ≈ e_axis;  W_steer @ v_perp ≈ 0
    assert np.allclose(W_steer @ v_steer, np.eye(3)[axis_idx], atol=1e-6), "v_steer misaligned"
    assert np.linalg.norm(W_steer @ v_perp) < 1e-6, "v_perp not in null-space"
    print(f"[probe] ||v_steer||={v_norm:.3f}  W_steer.shape={W_steer.shape}  "
          f"W_steer @ v_perp = {W_steer @ v_perp}")

    # Pick scenes
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
    prompt = mcfg["prompt"]

    try:
        results: list[dict] = []
        for sidx, sid in enumerate(scene_ids):
            scene_dir = scenes_dir / sid
            scene = Scene.load(scene_dir)
            object_ids = [o.object_id for o in scene.objects][: args.n_objects_per_scene]
            print(f"\n[scene {sidx+1}/{len(scene_ids)}] {sid}  "
                  f"objects={object_ids}  n_frames={len(scene.frames)}")

            # Baseline forward (no intervention) once per scene
            base = run_forward(wrapper, scene, scene_dir, prompt)
            vstart = base["visual_start"]
            grid = base["grid"]

            # Per-object: compute token positions and baseline readout
            per_obj: dict[int, dict] = {}
            for oid in object_ids:
                positions, _ = object_token_positions(
                    scene, scene_dir, wrapper, vstart, grid,
                    args.threshold, args.t_min, oid,
                )
                if not positions:
                    continue
                base_vec_steer = pool_at_layer(base["hidden"][args.steer_layer], positions)
                base_vec_read = pool_at_layer(base["hidden"][args.readout_layer], positions)
                base_readout = W_read @ base_vec_read + b_read
                base_steer_readout = W_steer @ base_vec_steer + b_steer
                per_obj[oid] = {
                    "positions": positions,
                    "base_readout": base_readout,
                    "base_steer_readout": base_steer_readout,
                }
                results.append({
                    "scene_id": sid, "object_id": int(oid),
                    "direction": "baseline", "alpha": 0.0,
                    "readout": base_readout.tolist(),
                    "steer_readout": base_steer_readout.tolist(),
                    "n_positions": len(positions),
                })

            # Intervened forwards: for each (direction, alpha), one forward
            # that steers all target objects simultaneously is cheaper than
            # one-per-object, but then the perturbation per token changes.
            # Sticking to one-per-object-per-α for clean attribution.
            for direction_name, v in [("v_axis", v_steer), ("v_perp", v_perp)]:
                for alpha in alpha_values:
                    if alpha == 0.0:
                        continue  # baseline already recorded
                    for oid in object_ids:
                        if oid not in per_obj:
                            continue
                        positions = per_obj[oid]["positions"]
                        delta = alpha * v
                        fwd = run_forward(
                            wrapper, scene, scene_dir, prompt,
                            intervention=(args.steer_layer, positions, delta),
                        )
                        int_vec_read = pool_at_layer(fwd["hidden"][args.readout_layer], positions)
                        int_vec_steer = pool_at_layer(fwd["hidden"][args.steer_layer], positions)
                        readout = W_read @ int_vec_read + b_read
                        steer_readout = W_steer @ int_vec_steer + b_steer
                        results.append({
                            "scene_id": sid, "object_id": int(oid),
                            "direction": direction_name, "alpha": float(alpha),
                            "readout": readout.tolist(),
                            "steer_readout": steer_readout.tolist(),
                            "n_positions": len(positions),
                        })
                        print(f"  [{direction_name} α={alpha:+.2f} obj={oid}] "
                              f"Δread_{args.axis}={readout[axis_idx]-per_obj[oid]['base_readout'][axis_idx]:+.3f}  "
                              f"Δsteer_{args.axis}={steer_readout[axis_idx]-per_obj[oid]['base_steer_readout'][axis_idx]:+.3f}")
    finally:
        wrapper.close()

    # ---- Save raw results ----
    df = pd.DataFrame(results)
    df.to_parquet(out_dir / "steering_results.parquet")
    (out_dir / "steering_results.json").write_text(
        json.dumps(results, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    )

    # ---- Compute readout shifts vs baseline ----
    baseline_map = {
        (r["scene_id"], r["object_id"]): (np.asarray(r["readout"]), np.asarray(r["steer_readout"]))
        for r in results if r["direction"] == "baseline"
    }
    rows = []
    for r in results:
        if r["direction"] == "baseline":
            continue
        br, bsr = baseline_map[(r["scene_id"], r["object_id"])]
        rr = np.asarray(r["readout"])
        sr = np.asarray(r["steer_readout"])
        rows.append({
            "scene_id": r["scene_id"],
            "object_id": r["object_id"],
            "direction": r["direction"],
            "alpha": r["alpha"],
            "dread_x": float(rr[0] - br[0]),
            "dread_y": float(rr[1] - br[1]),
            "dread_z": float(rr[2] - br[2]),
            "dread_l2": float(np.linalg.norm(rr - br)),
            "dsteer_x": float(sr[0] - bsr[0]),
            "dsteer_y": float(sr[1] - bsr[1]),
            "dsteer_z": float(sr[2] - bsr[2]),
            "dsteer_l2": float(np.linalg.norm(sr - bsr)),
        })
    shifts = pd.DataFrame(rows)
    shifts.to_parquet(out_dir / "steering_shifts.parquet")

    # ---- Aggregate + plot ----
    axis_col = f"dread_{args.axis}"
    agg = shifts.groupby(["direction", "alpha"]).agg(
        mean_shift=(axis_col, "mean"),
        std_shift=(axis_col, "std"),
        n=(axis_col, "count"),
        mean_l2=("dread_l2", "mean"),
    ).reset_index()
    agg.to_csv(out_dir / "steering_agg.csv", index=False)
    print("\n=== Aggregate readout shift on axis", args.axis, "===")
    print(agg.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    for d, color in [("v_axis", "C0"), ("v_perp", "C3")]:
        sub = agg[agg.direction == d].sort_values("alpha")
        label = f"v_{args.axis} (in S)" if d == "v_axis" else "v_perp (null-space of W)"
        axes[0].errorbar(sub.alpha, sub.mean_shift, yerr=sub.std_shift,
                         fmt="-o", color=color, capsize=3, label=label)
    lims = [min(alpha_values), max(alpha_values)]
    axes[0].plot(lims, lims, "--", color="gray", lw=0.9, label="ideal slope 1")
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].axvline(0, color="gray", lw=0.5)
    axes[0].set_xlabel(f"steering α  (units of normalized {args.axis} coord)")
    axes[0].set_ylabel(f"Δ readout_{args.axis} at L{args.readout_layer}  "
                       f"(relative to α=0)")
    axes[0].set_title(f"Q2/H3: readout shift from L{args.steer_layer} intervention")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    for d, color in [("v_axis", "C0"), ("v_perp", "C3")]:
        sub = agg[agg.direction == d].sort_values("alpha")
        label = f"v_{args.axis} (in S)" if d == "v_axis" else "v_perp (null-space of W)"
        axes[1].plot(sub.alpha, sub.mean_l2, "-o", color=color, label=label)
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_xlabel(f"steering α")
    axes[1].set_ylabel(f"||Δ readout||₂")
    axes[1].set_title("Readout shift magnitude (3-vec L2)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        f"Activation steering — free6dof Tier C — "
        f"steer L{args.steer_layer} → readout L{args.readout_layer} "
        f"(axis={args.axis})"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "steering.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_dir}/{{steering.png, steering_results.json, steering_agg.csv}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
