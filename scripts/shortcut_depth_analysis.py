#!/usr/bin/env python
"""Depth-probe shortcut diagnostic (V1 + V3 from the discussion).

Question: does the per-object depth probe succeed by reading the object's
apparent pixel size (a trivial monocular cue, since r_px = f · size_world /
depth) rather than by encoding 3D depth?

Two tests:

  V1. Apparent-size baseline — fit a 4-feature linear regressor
      [1/r_px, r_px, log(r_px), sqrt(r_px)] → depth and report its R². This is
      the *ceiling* of what the shortcut alone can achieve.

  V3. Residual probe — subtract the V1 baseline's depth prediction from the
      true depth to get depth_residual, then fit the VLM linear probe on
      hidden_state → depth_residual. R² > 0 means the hidden states carry
      depth information ABOVE and beyond the apparent-size shortcut.

Usage:
  python scripts/shortcut_depth_analysis.py \
      --activations data/activations/tier_c_free6dof_qwen25vl_7b \
      --scenes      data/tier_c_free6dof \
      --out         data/probes/tier_c_free6dof/qwen25vl_7b_shortcut \
      --temporal-patch-size 2 --t-min 4
"""
import _bootstrap  # noqa: F401

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from spatial_subspace.labels import object_depth_in_camera
from spatial_subspace.scene import Scene


def _base_scene_id(sid: str) -> str:
    if "_t" in sid:
        head, tail = sid.rsplit("_t", 1)
        if tail.isdigit():
            return head
    return sid


def _compute_r_px_and_depth(
    meta: pd.DataFrame,
    scenes_dir: Path,
    tps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """For each row in meta, return (r_px, depth).

    r_px is the square-root of the object's mask pixel count at the source
    frame(s) fused for that temporal token (averaged when tps > 1). depth is
    the z-coordinate of the object's world centroid in the camera frame at
    the mid-point input frame of that temporal token.
    """
    # Pre-load every scene's masks + extrinsics + centroids. Scenes are
    # loaded once even if many rows reference them.
    unique_sids = meta["scene_id"].unique().tolist()
    cache: dict[str, dict] = {}
    for sid in unique_sids:
        scene = Scene.load(scenes_dir / sid)
        extrs = [np.asarray(f.camera.extrinsics, dtype=np.float64) for f in scene.frames]
        centroids = {o.object_id: np.asarray(o.centroid, dtype=np.float64)
                     for o in scene.objects}
        mask_paths = [scenes_dir / sid / f.mask_path for f in scene.frames]
        cache[sid] = {"extrs": extrs, "centroids": centroids, "mask_paths": mask_paths}

    # Memoize mask → per-object pixel counts so we scan each file once.
    mask_pixcount: dict[tuple[str, int], dict[int, int]] = {}

    def _pixcounts(sid: str, frame_idx: int) -> dict[int, int]:
        key = (sid, frame_idx)
        if key not in mask_pixcount:
            m = np.array(Image.open(cache[sid]["mask_paths"][frame_idx]))
            unique, counts = np.unique(m, return_counts=True)
            mask_pixcount[key] = dict(zip(unique.tolist(), counts.tolist()))
        return mask_pixcount[key]

    n = len(meta)
    r_px = np.zeros(n, dtype=np.float32)
    depth = np.zeros(n, dtype=np.float32)
    sids = meta["scene_id"].to_numpy()
    oids = meta["object_id"].to_numpy()
    ts = meta["frame_id"].to_numpy()

    for i in range(n):
        sid = sids[i]
        t = int(ts[i])
        oid = int(oids[i])
        extrs = cache[sid]["extrs"]
        # Camera frame for depth: use the mid-point of the tps input frames.
        f_curr = min(t * tps + (tps - 1), len(extrs) - 1)
        depth[i] = object_depth_in_camera(cache[sid]["centroids"][oid], extrs[f_curr])
        # r_px: sum pixel counts over the tps source frames, then sqrt, then
        # divide by sqrt(tps) so that a constant mask gives the single-frame
        # radius rather than a tps-inflated value.
        total_px = 0
        for k in range(tps):
            fi = min(t * tps + k, len(cache[sid]["mask_paths"]) - 1)
            total_px += _pixcounts(sid, fi).get(oid + 1, 0)  # object id in mask is oid+1
        r_px[i] = float(np.sqrt(max(total_px / max(tps, 1), 0.0)))
    return r_px, depth


def _rpx_features(r_px: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """4-feature non-linear projection of r_px so that ridge regression can
    capture any reasonable monotone r_px → depth relationship (including the
    physics-exact 1/r_px form for a fixed world size)."""
    r = np.clip(r_px, eps, None)
    return np.stack([1.0 / r, r, np.log(r), np.sqrt(r)], axis=1).astype(np.float32)


def _fit_baseline(
    r_px_tr: np.ndarray,
    depth_tr: np.ndarray,
    r_px_te: np.ndarray,
    depth_te: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Fit r_px → depth ridge. Returns (R² on test, residual depth on test)."""
    X_tr = _rpx_features(r_px_tr)
    X_te = _rpx_features(r_px_te)
    model = Ridge(alpha=1.0).fit(X_tr, depth_tr)
    pred_te = model.predict(X_te)
    resid_te = depth_te - pred_te
    return float(r2_score(depth_te, pred_te)), resid_te, model


def _split(base_ids: np.ndarray, train_frac: float, seed: int) -> tuple[set, set]:
    unique = np.array(sorted(set(base_ids)))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    n_tr = int(train_frac * len(unique))
    return set(unique[order[:n_tr]]), set(unique[order[n_tr:]])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--activations", required=True)
    p.add_argument("--scenes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--temporal-patch-size", type=int, default=2)
    p.add_argument("--t-min", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--alpha", type=float, default=10.0)
    args = p.parse_args()

    activ = Path(args.activations)
    scenes_dir = Path(args.scenes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    layers = sorted(int(f.stem.split("_")[1]) for f in activ.glob("layer_*.parquet"))
    if not layers:
        raise SystemExit(f"no layer_*.parquet under {activ}")

    meta0 = pd.read_parquet(activ / f"layer_{layers[0]:02d}.parquet")
    n_tokens = int(meta0.frame_id.max()) + 1
    t_min = args.t_min if args.t_min is not None else n_tokens // 2
    base_ids_all = np.array([_base_scene_id(s) for s in meta0.scene_id])
    train_base, test_base = _split(base_ids_all, args.train_frac, args.seed)
    print(f"[setup] n_tokens={n_tokens}  t_min={t_min}  "
          f"base scenes: {len(train_base)} tr / {len(test_base)} te")

    # Compute per-row (r_px, depth) once.
    print(f"[rpx] computing apparent radii + depths for {len(meta0)} rows ...")
    r_px, depth = _compute_r_px_and_depth(meta0, scenes_dir, args.temporal_patch_size)

    # Filter to latter frames.
    sids = meta0["scene_id"].to_numpy()
    ts = meta0["frame_id"].to_numpy()
    base = np.array([_base_scene_id(s) for s in sids])
    mask = (ts >= t_min)
    tr_row = mask & np.isin(base, list(train_base))
    te_row = mask & np.isin(base, list(test_base))
    print(f"[data] rows: {len(meta0)}  after t≥{t_min}: "
          f"{int(tr_row.sum())} tr / {int(te_row.sum())} te")

    # --- V1 baseline ---
    baseline_r2, depth_resid_te, _ = _fit_baseline(
        r_px[tr_row], depth[tr_row], r_px[te_row], depth[te_row],
    )
    print(f"[V1] apparent-size baseline R² = {baseline_r2:.3f}")

    # The residual target on *train* side (so we can fit per-layer probes on residuals).
    # Fit a fresh baseline on train alone to produce train residuals with zero peek into test.
    _r2_ignore, _resid_te_ignore, baseline_model = _fit_baseline(
        r_px[tr_row], depth[tr_row], r_px[tr_row], depth[tr_row]
    )
    depth_pred_tr = baseline_model.predict(_rpx_features(r_px[tr_row]))
    depth_resid_tr = depth[tr_row] - depth_pred_tr

    # --- per-layer probes ---
    results: list[dict] = []
    for layer in layers:
        vecs = np.load(activ / f"layer_{layer:02d}.npy")
        X_tr = vecs[tr_row]
        X_te = vecs[te_row]
        y_tr = depth[tr_row]
        y_te = depth[te_row]

        # VLM probe on raw depth (matches fit_probes_camera_depth output)
        m_raw = Ridge(alpha=args.alpha).fit(X_tr, y_tr)
        pred_raw = m_raw.predict(X_te)
        r2_raw = float(r2_score(y_te, pred_raw))

        # VLM probe on residual (V3)
        m_res = Ridge(alpha=args.alpha).fit(X_tr, depth_resid_tr)
        pred_res = m_res.predict(X_te)
        r2_res_depth = float(r2_score(depth_resid_te, pred_res))

        row = {
            "layer": int(layer),
            "r2_vlm_depth": r2_raw,
            "r2_vlm_residual": r2_res_depth,
        }
        results.append(row)
        print(f"[L{layer:02d}]  VLM→depth R²={r2_raw:+.3f}  "
              f"VLM→residual R²={r2_res_depth:+.3f}")

    summary = {
        "baseline_r2": baseline_r2,
        "t_min": int(t_min),
        "temporal_patch_size": int(args.temporal_patch_size),
        "n_train": int(tr_row.sum()),
        "n_test": int(te_row.sum()),
        "baseline_target_std": float(depth[te_row].std(ddof=1)),
        "per_layer": results,
    }
    (out / "shortcut_analysis.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] {out / 'shortcut_analysis.json'}")

    # --- plot ---
    Ls = np.array([r["layer"] for r in results])
    r2_raw = np.array([r["r2_vlm_depth"] for r in results])
    r2_res = np.array([r["r2_vlm_residual"] for r in results])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(Ls, r2_raw, "-o", color="C0", label=f"VLM probe → depth  (max {r2_raw.max():.3f})")
    ax.axhline(baseline_r2, color="C2", linestyle="--",
               label=f"apparent-size baseline (r_px features only) = {baseline_r2:.3f}")
    ax.plot(Ls, r2_res, "-s", color="C3",
            label=f"VLM probe → depth residual (V3, max {r2_res.max():.3f})")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("LM decoder layer")
    ax.set_ylabel("R²")
    ax.set_title(
        f"Depth shortcut analysis — {activ.name}\n"
        f"gap (VLM − baseline) = signal beyond apparent size"
    )
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "shortcut_analysis.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out / 'shortcut_analysis.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
