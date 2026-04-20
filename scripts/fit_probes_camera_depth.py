#!/usr/bin/env python
"""Probe camera motion and per-object depth from Tier C (free6dof) activations.

Alternative to the Q1 probe (world 3D coordinates). Two targets, both derived
from per-frame camera extrinsics stored in each scene.json:

  (1) cam_delta[(scene, t)]
      6-DoF relative camera pose between consecutive temporal-token pairs,
      encoded as ``[tx, ty, tz, rx, ry, rz]`` (axis-angle). Undefined at t=0.
      Scene-level target → fit on the mean over visible-object vectors at
      (scene, t).

  (2) depth[(scene, obj, t)]
      z-coordinate of the object's world centroid in the camera frame at
      temporal token t. Per-(scene, obj, t) target → fit on the pooled
      per-object vector directly.

Only the latter half of temporal tokens is probed (default t >= n_tokens // 2)
so the model has had enough context to integrate multi-view information before
we ask it to report the camera delta or the object depth.

Usage:
  python scripts/fit_probes_camera_depth.py \
      --activations data/activations/tier_c_free6dof_qwen25vl_7b \
      --scenes data/tier_c_free6dof \
      --out data/probes/tier_c_free6dof_camera_depth \
      --t-min 4 --temporal-patch-size 2
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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from spatial_subspace.labels import camera_delta_6d, object_depth_in_camera
from spatial_subspace.scene import Scene


def _base_scene_id(sid: str) -> str:
    if "_t" in sid:
        head, _tail = sid.rsplit("_t", 1)
        if _tail.isdigit():
            return head
    return sid


def _load_scene_cache(scenes_dir: Path, scene_ids: list[str]) -> dict[str, dict]:
    """Per scene_id: {'extrinsics': [np.ndarray 4x4, ...], 'centroids': [np...]}.

    Reads scene.json once per scene_id.
    """
    cache: dict[str, dict] = {}
    for sid in scene_ids:
        scene = Scene.load(scenes_dir / sid)
        extrs = [np.asarray(f.camera.extrinsics, dtype=np.float64) for f in scene.frames]
        centroids = {o.object_id: np.asarray(o.centroid, dtype=np.float64) for o in scene.objects}
        cache[sid] = {"extrinsics": extrs, "centroids": centroids}
    return cache


def _build_labels(
    meta: pd.DataFrame,
    cache: dict[str, dict],
    tps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (cam_delta[N,6], depth[N], valid[N] bool).

    ``valid`` is False for rows with t=0 (cam_delta undefined). ``cam_delta``
    at invalid rows is set to zero and should be ignored downstream.
    """
    n = len(meta)
    cam_delta = np.zeros((n, 6), dtype=np.float32)
    depth = np.zeros(n, dtype=np.float32)
    valid = np.ones(n, dtype=bool)

    sids = meta.scene_id.to_numpy()
    oids = meta.object_id.to_numpy()
    ts = meta.frame_id.to_numpy()  # temporal-token index

    for i in range(n):
        sid = sids[i]
        t = int(ts[i])
        oid = int(oids[i])
        extrs = cache[sid]["extrinsics"]
        f_curr = min(t * tps, len(extrs) - 1)
        # Depth is defined for every t.
        depth[i] = object_depth_in_camera(cache[sid]["centroids"][oid], extrs[f_curr])
        # Cam-delta needs a previous temporal token.
        if t == 0:
            valid[i] = False
            continue
        f_prev = (t - 1) * tps
        cam_delta[i] = camera_delta_6d(extrs[f_prev], extrs[f_curr]).astype(np.float32)
    return cam_delta, depth, valid


def _aggregate_by_scene_t(
    vecs: np.ndarray,
    sids: np.ndarray,
    ts: np.ndarray,
    cam_delta: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean-pool per-object vectors to one vector per (scene, t).

    Returns (agg_vecs, agg_labels, agg_base_scene, agg_t).
    """
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i in range(len(sids)):
        if not valid[i]:
            continue
        groups[(sids[i], int(ts[i]))].append(i)

    agg_vecs: list[np.ndarray] = []
    agg_labels: list[np.ndarray] = []
    agg_base: list[str] = []
    agg_t: list[int] = []
    for (sid, t), idx in groups.items():
        if not idx:
            continue
        agg_vecs.append(vecs[idx].mean(axis=0))
        # All rows in a (scene, t) group share the same cam_delta label.
        agg_labels.append(cam_delta[idx[0]])
        agg_base.append(_base_scene_id(sid))
        agg_t.append(t)
    return (
        np.stack(agg_vecs) if agg_vecs else np.zeros((0, vecs.shape[1])),
        np.stack(agg_labels) if agg_labels else np.zeros((0, 6)),
        np.asarray(agg_base),
        np.asarray(agg_t, dtype=np.int64),
    )


def _split(base_ids: np.ndarray, train_frac: float, seed: int) -> tuple[set, set]:
    unique = np.array(sorted(set(base_ids)))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    n_tr = int(train_frac * len(unique))
    return set(unique[order[:n_tr]]), set(unique[order[n_tr:]])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--activations", required=True, help="layer_NN.parquet/.npy dir")
    p.add_argument("--scenes", required=True, help="rendered data dir (scene.json per scene_id)")
    p.add_argument("--out", required=True)
    p.add_argument("--temporal-patch-size", type=int, default=2,
                   help="Qwen2.5-VL default = 2 (each temporal token covers 2 input frames)")
    p.add_argument("--t-min", type=int, default=None,
                   help="Probe only rows with t >= t_min (default = n_tokens // 2)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--min-train", type=int, default=30)
    p.add_argument("--min-test", type=int, default=10)
    args = p.parse_args()

    activ = Path(args.activations)
    scenes_dir = Path(args.scenes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    layers = sorted(int(f.stem.split("_")[1]) for f in activ.glob("layer_*.parquet"))
    if not layers:
        raise SystemExit(f"no layer_*.parquet files under {activ}")

    # Scene cache + temporal-token range from first layer's meta.
    meta0 = pd.read_parquet(activ / f"layer_{layers[0]:02d}.parquet")
    scene_cache = _load_scene_cache(scenes_dir, sorted(set(meta0.scene_id)))
    n_tokens = int(meta0.frame_id.max()) + 1
    t_min = args.t_min if args.t_min is not None else n_tokens // 2
    print(f"[setup] n_tokens={n_tokens}  t_min={t_min}  scenes={len(scene_cache)}")

    base_ids_all = np.array([_base_scene_id(s) for s in meta0.scene_id])
    train_base, test_base = _split(base_ids_all, args.train_frac, args.seed)
    print(f"[setup] base scenes: {len(train_base)} train / {len(test_base)} test")

    results: list[dict] = []
    for layer in layers:
        meta = pd.read_parquet(activ / f"layer_{layer:02d}.parquet")
        vecs = np.load(activ / f"layer_{layer:02d}.npy")
        cam_delta, depth, valid = _build_labels(meta, scene_cache, args.temporal_patch_size)

        sids = meta.scene_id.to_numpy()
        ts = meta.frame_id.to_numpy()
        base = np.array([_base_scene_id(s) for s in sids])

        # ---- Depth probe (per-row, per-object) ----
        mask_row = (ts >= t_min)  # latter frames only
        tr_row = mask_row & np.isin(base, list(train_base))
        te_row = mask_row & np.isin(base, list(test_base))
        depth_result = None
        if tr_row.sum() >= args.min_train and te_row.sum() >= args.min_test:
            model_d = Ridge(alpha=args.alpha).fit(vecs[tr_row], depth[tr_row])
            pred_d = model_d.predict(vecs[te_row])
            err_d = depth[te_row] - pred_d  # signed residual (truth − pred)
            depth_result = {
                "r2": float(r2_score(depth[te_row], pred_d)),
                "mae": float(np.mean(np.abs(err_d))),
                "rmse": float(np.sqrt(np.mean(err_d ** 2))),
                "error_mean": float(err_d.mean()),
                "error_std": float(err_d.std(ddof=1)),
                "error_var": float(err_d.var(ddof=1)),
                "target_std": float(depth[te_row].std(ddof=1)),
                "n_train": int(tr_row.sum()),
                "n_test": int(te_row.sum()),
            }

        # ---- Cam-delta probe (aggregated per (scene, t)) ----
        agg_X, agg_y, agg_base, agg_t = _aggregate_by_scene_t(
            vecs, sids, ts, cam_delta, valid
        )
        mask_agg = (agg_t >= t_min)
        tr_agg = mask_agg & np.isin(agg_base, list(train_base))
        te_agg = mask_agg & np.isin(agg_base, list(test_base))
        cam_result = None
        if tr_agg.sum() >= args.min_train and te_agg.sum() >= args.min_test:
            model_c = Ridge(alpha=args.alpha).fit(agg_X[tr_agg], agg_y[tr_agg])
            pred_c = model_c.predict(agg_X[te_agg])
            y_te = agg_y[te_agg]
            err_c = y_te - pred_c  # (N, 6) signed residuals per component
            r2_per = r2_score(y_te, pred_c, multioutput="raw_values")
            comp_names = ["tx", "ty", "tz", "rx", "ry", "rz"]
            cam_result = {
                "r2_overall": float(r2_score(y_te, pred_c, multioutput="uniform_average")),
                "r2_translation": float(np.mean(r2_per[:3])),
                "r2_rotation": float(np.mean(r2_per[3:])),
                "r2_components": {
                    name: float(r2_per[i]) for i, name in enumerate(comp_names)
                },
                "mae_translation": float(np.mean(np.abs(err_c[:, :3]))),
                "mae_rotation": float(np.mean(np.abs(err_c[:, 3:]))),
                "error_mean_components": {
                    name: float(err_c[:, i].mean()) for i, name in enumerate(comp_names)
                },
                "error_std_components": {
                    name: float(err_c[:, i].std(ddof=1)) for i, name in enumerate(comp_names)
                },
                "error_var_components": {
                    name: float(err_c[:, i].var(ddof=1)) for i, name in enumerate(comp_names)
                },
                "target_std_components": {
                    name: float(y_te[:, i].std(ddof=1)) for i, name in enumerate(comp_names)
                },
                "error_std_translation": float(np.linalg.norm(err_c[:, :3], axis=1).std(ddof=1)),
                "error_mean_translation_norm": float(np.linalg.norm(err_c[:, :3], axis=1).mean()),
                "error_std_rotation": float(np.linalg.norm(err_c[:, 3:], axis=1).std(ddof=1)),
                "error_mean_rotation_norm": float(np.linalg.norm(err_c[:, 3:], axis=1).mean()),
                "n_train": int(tr_agg.sum()),
                "n_test": int(te_agg.sum()),
            }

        results.append({
            "layer": int(layer),
            "depth": depth_result,
            "cam_delta": cam_result,
        })
        d_r2 = depth_result["r2"] if depth_result else float("nan")
        d_em = depth_result["error_mean"] if depth_result else float("nan")
        d_es = depth_result["error_std"] if depth_result else float("nan")
        d_ts = depth_result["target_std"] if depth_result else float("nan")
        c_r2 = cam_result["r2_overall"] if cam_result else float("nan")
        c_tr = cam_result["r2_translation"] if cam_result else float("nan")
        c_ro = cam_result["r2_rotation"] if cam_result else float("nan")
        c_es_t = cam_result["error_std_translation"] if cam_result else float("nan")
        c_es_r = cam_result["error_std_rotation"] if cam_result else float("nan")
        print(
            f"[L{layer:02d}]  depth R²={d_r2:.3f} err µ={d_em:+.3f} σ={d_es:.3f}"
            f" (y σ={d_ts:.3f})  |  cam R²={c_r2:.3f}"
            f"  trans R²={c_tr:.3f} σ={c_es_t:.3f}  rot R²={c_ro:.3f} σ={c_es_r:.3f}"
        )

    (out / "camera_depth_probes.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out}/camera_depth_probes.json")

    # ---- Plot ----
    Ls = np.array([r["layer"] for r in results])
    depth_r2 = np.array([r["depth"]["r2"] if r["depth"] else np.nan for r in results])
    depth_mu = np.array([r["depth"]["error_mean"] if r["depth"] else np.nan for r in results])
    depth_sg = np.array([r["depth"]["error_std"] if r["depth"] else np.nan for r in results])
    depth_ty = np.array([r["depth"]["target_std"] if r["depth"] else np.nan for r in results])

    cam_r2 = np.array([r["cam_delta"]["r2_overall"] if r["cam_delta"] else np.nan for r in results])
    cam_tr = np.array([r["cam_delta"]["r2_translation"] if r["cam_delta"] else np.nan for r in results])
    cam_ro = np.array([r["cam_delta"]["r2_rotation"] if r["cam_delta"] else np.nan for r in results])

    comp_names = ["tx", "ty", "tz", "rx", "ry", "rz"]
    # residual σ / target σ per component — a [0, 1+] "normalized spread" where
    # 0 means perfect and 1 means predict-mean-baseline.
    norm_spread = {
        c: np.array([
            r["cam_delta"]["error_std_components"][c] / max(r["cam_delta"]["target_std_components"][c], 1e-9)
            if r["cam_delta"] else np.nan
            for r in results
        ])
        for c in comp_names
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # --- (0,0) Depth R² ---
    ax = axes[0, 0]
    ax.plot(Ls, depth_r2, "-o", color="C0", label="per-object depth")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("LM decoder layer")
    ax.set_ylabel("R²")
    ax.set_title(f"Depth probe R²  (t ≥ {t_min})")
    ax.grid(alpha=0.3)
    ax.legend()

    # --- (0,1) Cam-motion R² ---
    ax = axes[0, 1]
    ax.plot(Ls, cam_r2, "-o", color="C1", label="cam Δ (6d)")
    ax.plot(Ls, cam_tr, "--s", color="C2", label="cam Δ translation")
    ax.plot(Ls, cam_ro, "--^", color="C3", label="cam Δ rotation")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("LM decoder layer")
    ax.set_ylabel("R²")
    ax.set_title(f"Camera-motion probe R²  (t ≥ {t_min})")
    ax.grid(alpha=0.3)
    ax.legend()

    # --- (1,0) Depth residual: bias ± σ, with target σ reference ---
    ax = axes[1, 0]
    ax.fill_between(Ls, depth_mu - depth_sg, depth_mu + depth_sg,
                    color="C0", alpha=0.2, label="bias ± 1σ residual")
    ax.plot(Ls, depth_mu, "-o", color="C0", label="residual mean (bias)")
    # target-σ band as a grey reference: mean ± target σ (what predict-mean would look like)
    ax.fill_between(Ls, -depth_ty, depth_ty, color="gray", alpha=0.08,
                    label="±1 target σ (predict-mean baseline)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("LM decoder layer")
    ax.set_ylabel("depth residual  (world units)")
    ax.set_title("Depth probe residual: signed bias + spread")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # --- (1,1) Cam-motion normalized residual spread (σ_err / σ_target) per component ---
    ax = axes[1, 1]
    colors_t = ["#1f77b4", "#2ca02c", "#17becf"]    # tx, ty, tz
    colors_r = ["#d62728", "#ff7f0e", "#9467bd"]    # rx, ry, rz
    for c, col in zip(["tx", "ty", "tz"], colors_t):
        ax.plot(Ls, norm_spread[c], "-o", color=col, label=c, markersize=4)
    for c, col in zip(["rx", "ry", "rz"], colors_r):
        ax.plot(Ls, norm_spread[c], "--^", color=col, label=c, markersize=4)
    ax.axhline(1.0, color="gray", lw=0.8, linestyle=":",
               label="predict-mean baseline")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlabel("LM decoder layer")
    ax.set_ylabel("residual σ / target σ")
    ax.set_title("Cam Δ per-component residual spread (normalized)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig.suptitle(
        "Tier C free6dof — probing camera motion + per-object depth "
        "(latter half of temporal tokens)",
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out / "camera_depth_probes.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}/camera_depth_probes.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
