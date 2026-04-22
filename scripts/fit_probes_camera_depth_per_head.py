#!/usr/bin/env python
"""Per-head camera-motion + per-object-depth probes (Tier C free6dof).

Reuses the label construction from ``scripts/fit_probes_camera_depth.py``
(``camera_delta_6d`` + ``object_depth_in_camera`` from extrinsics), but runs
the Ridge probe **separately for every (layer, head)** on the per-head
pooled vectors written by ``scripts/extract_per_head.py``.

Output:
    <out>/per_head.json / .parquet    one row per (layer, head)
    <out>/heatmaps.png                layer × head R² heatmaps for
                                       depth, cam_trans, cam_rot
    <out>/layer_best_heads.csv        top-k heads per (layer, target)
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


def _fast_ridge_predict(X_tr: np.ndarray, y_tr: np.ndarray,
                        X_te: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge via the smaller normal equation.

    For wide matrices where ``N < D`` (e.g. N≈1.5k, D=5120 for cam-delta
    aggregation), solving in sample space is O(N^3) instead of O(D^3).
    Identity used: ``(X^T X + αI_D)^{-1} X^T y  =  X^T (X X^T + αI_N)^{-1} y``.
    For tall matrices this routine falls back to the D×D solve, which is
    still the cholesky path but without the sklearn overhead.
    """
    X_tr = np.ascontiguousarray(X_tr, dtype=np.float32)
    X_te = np.ascontiguousarray(X_te, dtype=np.float32)
    y_tr = np.asarray(y_tr, dtype=np.float32)
    # Center target(s) — Ridge does not regularize the bias.
    if y_tr.ndim == 1:
        y_mean = y_tr.mean()
    else:
        y_mean = y_tr.mean(axis=0, keepdims=True)
    y_c = y_tr - y_mean

    N, D = X_tr.shape
    if N < D:
        K = X_tr @ X_tr.T                          # (N, N)
        K.flat[:: N + 1] += alpha                  # + α I in-place
        a = np.linalg.solve(K, y_c)                # (N, ...) coefficients in sample basis
        pred_te = X_te @ (X_tr.T @ a) + y_mean     # (N_te, ...)
    else:
        A = X_tr.T @ X_tr                          # (D, D)
        A.flat[:: D + 1] += alpha
        B = X_tr.T @ y_c                           # (D, ...)
        beta = np.linalg.solve(A, B)               # (D, ...)
        pred_te = X_te @ beta + y_mean
    return pred_te


def _base_scene_id(sid: str) -> str:
    if "_t" in sid:
        head, tail = sid.rsplit("_t", 1)
        if tail.isdigit():
            return head
    return sid


def _load_scene_cache(scenes_dir: Path, scene_ids: list[str]) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    for sid in scene_ids:
        s = Scene.load(scenes_dir / sid)
        cache[sid] = {
            "extrinsics": [np.asarray(f.camera.extrinsics, dtype=np.float64) for f in s.frames],
            "centroids": {o.object_id: np.asarray(o.centroid, dtype=np.float64) for o in s.objects},
        }
    return cache


def _build_labels(meta: pd.DataFrame, cache: dict[str, dict], tps: int):
    n = len(meta)
    cam_delta = np.zeros((n, 6), dtype=np.float32)
    depth = np.zeros(n, dtype=np.float32)
    valid = np.ones(n, dtype=bool)
    sids, oids, ts = meta.scene_id.to_numpy(), meta.object_id.to_numpy(), meta.frame_id.to_numpy()
    for i in range(n):
        extrs = cache[sids[i]]["extrinsics"]
        t = int(ts[i])
        f_curr = min(t * tps, len(extrs) - 1)
        depth[i] = object_depth_in_camera(cache[sids[i]]["centroids"][int(oids[i])], extrs[f_curr])
        if t == 0:
            valid[i] = False
            continue
        f_prev = (t - 1) * tps
        cam_delta[i] = camera_delta_6d(extrs[f_prev], extrs[f_curr]).astype(np.float32)
    return cam_delta, depth, valid


def _aggregate_by_scene_t(vecs, sids, ts, cam_delta, valid):
    """Vecs has shape (N, n_heads, D). Returns (agg_vecs (M, n_heads, D), ...) per (scene, t)."""
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
        agg_labels.append(cam_delta[idx[0]])
        agg_base.append(_base_scene_id(sid))
        agg_t.append(t)
    return (
        np.stack(agg_vecs) if agg_vecs else np.zeros((0, vecs.shape[1], vecs.shape[2])),
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


COMP_NAMES = ["tx", "ty", "tz", "rx", "ry", "rz"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--activations", required=True,
                   help="directory containing head_layer_LL.parquet/.npy")
    p.add_argument("--scenes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--temporal-patch-size", type=int, default=2)
    p.add_argument("--t-min", type=int, default=None,
                   help="default = n_tokens // 2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--min-train", type=int, default=30)
    p.add_argument("--min-test", type=int, default=10)
    p.add_argument("--top-k", type=int, default=5,
                   help="number of top heads per (layer, target) to report")
    p.add_argument("--title-model", default="Qwen2.5-VL",
                   help="model name to show in plot titles")
    args = p.parse_args()

    activ = Path(args.activations)
    scenes_dir = Path(args.scenes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    layers = sorted(int(f.stem.split("_")[2]) for f in activ.glob("head_layer_*.parquet"))
    if not layers:
        raise SystemExit(f"no head_layer_*.parquet files under {activ}")
    print(f"[setup] layers found: {layers}")

    meta0 = pd.read_parquet(activ / f"head_layer_{layers[0]:02d}.parquet")
    cache = _load_scene_cache(scenes_dir, sorted(set(meta0.scene_id)))
    n_tokens = int(meta0.frame_id.max()) + 1
    t_min = args.t_min if args.t_min is not None else n_tokens // 2
    base_ids_all = np.array([_base_scene_id(s) for s in meta0.scene_id])
    train_base, test_base = _split(base_ids_all, args.train_frac, args.seed)
    print(f"[setup] n_tokens={n_tokens}  t_min={t_min}  "
          f"train={len(train_base)}  test={len(test_base)}  "
          f"scenes={len(cache)}  rows/layer={len(meta0)}")

    rows: list[dict] = []
    for layer in layers:
        meta = pd.read_parquet(activ / f"head_layer_{layer:02d}.parquet")
        vecs = np.load(activ / f"head_layer_{layer:02d}.npy")  # (N, H, D)
        n_heads = vecs.shape[1]
        cam_delta, depth, valid = _build_labels(meta, cache, args.temporal_patch_size)

        sids = meta.scene_id.to_numpy()
        ts = meta.frame_id.to_numpy()
        base = np.array([_base_scene_id(s) for s in sids])

        # Depth: per-row masks.
        mask_row = ts >= t_min
        tr_row = mask_row & np.isin(base, list(train_base))
        te_row = mask_row & np.isin(base, list(test_base))
        # Cam-delta: aggregate per (scene, t) first.
        agg_X, agg_y, agg_base_a, agg_t = _aggregate_by_scene_t(
            vecs, sids, ts, cam_delta, valid
        )
        mask_agg = agg_t >= t_min
        tr_agg = mask_agg & np.isin(agg_base_a, list(train_base))
        te_agg = mask_agg & np.isin(agg_base_a, list(test_base))

        # One big fp16→fp32 cast per layer instead of 40× per-head casts
        vecs_tr_row = vecs[tr_row].astype(np.float32) if tr_row.sum() >= args.min_train else None
        vecs_te_row = vecs[te_row].astype(np.float32) if te_row.sum() >= args.min_test else None
        agg_tr = agg_X[tr_agg].astype(np.float32) if tr_agg.sum() >= args.min_train else None
        agg_te = agg_X[te_agg].astype(np.float32) if te_agg.sum() >= args.min_test else None

        for h in range(n_heads):
            row: dict = {"layer": int(layer), "head": int(h)}

            if vecs_tr_row is not None and vecs_te_row is not None:
                md = Ridge(alpha=args.alpha).fit(vecs_tr_row[:, h, :], depth[tr_row])
                pd_ = md.predict(vecs_te_row[:, h, :])
                y_te_d = depth[te_row]
                err = y_te_d - pd_
                row["depth_r2"] = float(r2_score(y_te_d, pd_))
                row["depth_mae"] = float(np.mean(np.abs(err)))
                row["depth_err_std"] = float(err.std(ddof=1))
            else:
                row["depth_r2"] = float("nan")

            if agg_tr is not None and agg_te is not None:
                mc = Ridge(alpha=args.alpha).fit(agg_tr[:, h, :], agg_y[tr_agg])
                pc = mc.predict(agg_te[:, h, :])
                y_te = agg_y[te_agg]
                r2_per = r2_score(y_te, pc, multioutput="raw_values")
                row["cam_r2_overall"] = float(r2_score(y_te, pc, multioutput="uniform_average"))
                row["cam_r2_translation"] = float(np.mean(r2_per[:3]))
                row["cam_r2_rotation"] = float(np.mean(r2_per[3:]))
                for i, c in enumerate(COMP_NAMES):
                    row[f"cam_r2_{c}"] = float(r2_per[i])
            else:
                row["cam_r2_overall"] = float("nan")
                row["cam_r2_translation"] = float("nan")
                row["cam_r2_rotation"] = float("nan")
                for c in COMP_NAMES:
                    row[f"cam_r2_{c}"] = float("nan")

            rows.append(row)

        row_ns = [r for r in rows if r["layer"] == layer]
        best_d = max(row_ns, key=lambda r: r.get("depth_r2", -1))
        best_c = max(row_ns, key=lambda r: r.get("cam_r2_overall", -1))
        print(f"[L{layer:02d}] best depth head={best_d['head']:>2} "
              f"R²={best_d['depth_r2']:+.3f}   "
              f"best cam head={best_c['head']:>2} "
              f"R²={best_c['cam_r2_overall']:+.3f}  "
              f"(trans={best_c['cam_r2_translation']:+.3f} rot={best_c['cam_r2_rotation']:+.3f})",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(out / "per_head.parquet")
    df.to_json(out / "per_head.json", orient="records", indent=2)

    # ---- Heatmaps ----
    targets = [
        ("depth_r2", "Depth R² (per-object)"),
        ("cam_r2_translation", "Cam Δ translation R²"),
        ("cam_r2_rotation", "Cam Δ rotation R²"),
        ("cam_r2_overall", "Cam Δ overall R²"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for (col, title), ax in zip(targets, axes.flat):
        pivot = df.pivot(index="layer", columns="head", values=col)
        vmax = float(max(0.1, np.nanquantile(pivot.values, 0.98)))
        vmin = float(min(0.0, np.nanquantile(pivot.values, 0.02)))
        im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                       cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xlabel("head index")
        ax.set_ylabel("LM decoder layer")
        ax.set_title(title + f"  (best={np.nanmax(pivot.values):.3f}, "
                     f"floor={np.nanmin(pivot.values):.3f})")
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle(
        f"Per-head residual-stream decomposition — {args.title_model} Tier C free6dof "
        "(Ridge cam-Δ + depth, latter half of temporal tokens)",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out / "heatmaps.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- Component-level heatmap for cam-delta ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i, c in enumerate(COMP_NAMES):
        ax = axes.flat[i]
        piv = df.pivot(index="layer", columns="head", values=f"cam_r2_{c}")
        vmax = float(max(0.1, np.nanquantile(piv.values, 0.98)))
        vmin = float(min(0.0, np.nanquantile(piv.values, 0.02)))
        im = ax.imshow(piv.values, aspect="auto", origin="lower",
                       cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xlabel("head"); ax.set_ylabel("layer")
        ax.set_title(f"cam Δ {c}  (best={np.nanmax(piv.values):.3f})")
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle("Per-head R² — individual cam-Δ components", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "heatmaps_components.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- Top-k heads per (layer, target) as CSV ----
    top = []
    for target, _ in targets + [(f"cam_r2_{c}", f"cam_r2_{c}") for c in COMP_NAMES]:
        for layer in layers:
            sub = df[df.layer == layer].sort_values(target, ascending=False).head(args.top_k)
            for rank, (_, r) in enumerate(sub.iterrows(), start=1):
                top.append({
                    "target": target,
                    "layer": int(layer),
                    "rank": int(rank),
                    "head": int(r["head"]),
                    "r2": float(r[target]),
                })
    pd.DataFrame(top).to_csv(out / "layer_best_heads.csv", index=False)

    print(f"\nsaved to {out}/:")
    for name in ("per_head.parquet", "per_head.json",
                 "heatmaps.png", "heatmaps_components.png",
                 "layer_best_heads.csv"):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
