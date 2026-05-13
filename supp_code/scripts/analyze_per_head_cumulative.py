#!/usr/bin/env python
"""Per-layer cumulative top-k head analysis.

Given the per-head probe metrics from ``fit_probes_camera_depth_per_head.py``,
for each layer selects heads in descending R² order and fits a ridge probe on
the *concatenation* of their pooled vectors (shape (N, k · D)). Answers "how
much of the layer's probe signal lives in the top-k heads?".

Also computes:
  - R² of sum-of-all-heads at each layer (attention-only residual write).
  - Participation ratio of the R² distribution across heads: large =
    information is spread, small = it lives in a few heads.

Output:
    <out>/cumulative.json / .parquet
    <out>/cumulative.png                per-layer R² vs k for depth / cam
    <out>/attention_vs_fulllayer.png    sum-of-heads R² compared against
                                        the pre-existing full-layer probe
                                        (if ``--full-layer-probe`` given)
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from spatial_subspace.labels import camera_delta_6d, object_depth_in_camera
from spatial_subspace.scene import Scene


def _fast_ridge_predict(X_tr, y_tr, X_te, alpha):
    """Closed-form ridge via the smaller normal equation.

    For N < D (common when k heads are concatenated), solves in N-sample space
    as ``X^T (X X^T + αI_N)^{-1} y``. For N > D, falls back to the D×D normal
    equation. Centers the target(s) but not X.
    """
    X_tr = np.ascontiguousarray(X_tr, dtype=np.float32)
    X_te = np.ascontiguousarray(X_te, dtype=np.float32)
    y_tr = np.asarray(y_tr, dtype=np.float32)
    y_mean = y_tr.mean(axis=0, keepdims=True) if y_tr.ndim > 1 else np.float32(y_tr.mean())
    y_c = y_tr - y_mean
    N, D = X_tr.shape
    if N < D:
        K = X_tr @ X_tr.T
        K.flat[:: N + 1] += alpha
        a = np.linalg.solve(K, y_c)
        pred = X_te @ (X_tr.T @ a) + y_mean
    else:
        A = X_tr.T @ X_tr
        A.flat[:: D + 1] += alpha
        B = X_tr.T @ y_c
        beta = np.linalg.solve(A, B)
        pred = X_te @ beta + y_mean
    return pred


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


def _labels(meta: pd.DataFrame, cache: dict, tps: int):
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
        cam_delta[i] = camera_delta_6d(extrs[(t - 1) * tps], extrs[f_curr]).astype(np.float32)
    return cam_delta, depth, valid


def _split(base_ids: np.ndarray, train_frac: float, seed: int) -> tuple[set, set]:
    unique = np.array(sorted(set(base_ids)))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    n_tr = int(train_frac * len(unique))
    return set(unique[order[:n_tr]]), set(unique[order[n_tr:]])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--per-head", required=True, help="dir containing per_head.json")
    p.add_argument("--activations", required=True, help="dir containing head_layer_LL.npy")
    p.add_argument("--scenes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--full-layer-probe", default=None,
                   help="optional path to the existing full-layer camera_depth_probes.json "
                        "for side-by-side comparison")
    p.add_argument("--ks", default="1,2,3,5,8,14,28")
    p.add_argument("--temporal-patch-size", type=int, default=2)
    p.add_argument("--t-min", type=int, default=None)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=10.0)
    args = p.parse_args()

    per_head_df = pd.read_parquet(Path(args.per_head) / "per_head.parquet")
    activ = Path(args.activations)
    scenes_dir = Path(args.scenes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ks = [int(x) for x in args.ks.split(",")]

    layers = sorted(per_head_df.layer.unique().tolist())
    meta0 = pd.read_parquet(activ / f"head_layer_{layers[0]:02d}.parquet")
    cache = _load_scene_cache(scenes_dir, sorted(set(meta0.scene_id)))
    n_tokens = int(meta0.frame_id.max()) + 1
    t_min = args.t_min if args.t_min is not None else n_tokens // 2
    base_all = np.array([_base_scene_id(s) for s in meta0.scene_id])
    train_base, test_base = _split(base_all, args.train_frac, args.seed)
    print(f"[setup] layers={layers}  t_min={t_min}  n_train_base={len(train_base)}")

    rows: list[dict] = []
    for L in layers:
        meta = pd.read_parquet(activ / f"head_layer_{L:02d}.parquet")
        vecs = np.load(activ / f"head_layer_{L:02d}.npy")  # (N, H, D)
        cam_delta, depth, valid = _labels(meta, cache, args.temporal_patch_size)
        sids = meta.scene_id.to_numpy()
        ts = meta.frame_id.to_numpy()
        base = np.array([_base_scene_id(s) for s in sids])

        # ---- Depth: per-row ----
        mask_row = ts >= t_min
        tr_row = mask_row & np.isin(base, list(train_base))
        te_row = mask_row & np.isin(base, list(test_base))

        # Rank heads by depth_r2 at this layer.
        subd = per_head_df[per_head_df.layer == L].sort_values("depth_r2", ascending=False)
        depth_order = subd["head"].tolist()
        subc = per_head_df[per_head_df.layer == L].sort_values("cam_r2_overall", ascending=False)
        cam_order = subc["head"].tolist()

        def fit_concat(vecs_NHd: np.ndarray, heads: list[int], X_mask_tr, X_mask_te, y):
            X = vecs_NHd[:, heads, :].reshape(vecs_NHd.shape[0], -1).astype(np.float32)
            if X_mask_tr.sum() < 30 or X_mask_te.sum() < 10:
                return float("nan")
            p = _fast_ridge_predict(X[X_mask_tr], y[X_mask_tr], X[X_mask_te], args.alpha)
            if y.ndim == 1:
                return float(r2_score(y[X_mask_te], p))
            return float(r2_score(y[X_mask_te], p, multioutput="uniform_average"))

        # Depth cumulative.
        for k in ks:
            k_use = min(k, len(depth_order))
            r2 = fit_concat(vecs, depth_order[:k_use], tr_row, te_row, depth)
            rows.append({"layer": int(L), "target": "depth",
                         "k": int(k), "heads_used": k_use, "r2": r2})

        # Sum of all heads (attention-only residual write).
        vecs_sum = vecs.sum(axis=1, dtype=np.float32)  # (N, D)
        if tr_row.sum() >= 30 and te_row.sum() >= 10:
            p_sum = _fast_ridge_predict(vecs_sum[tr_row], depth[tr_row],
                                         vecs_sum[te_row], args.alpha)
            r2 = float(r2_score(depth[te_row], p_sum))
        else:
            r2 = float("nan")
        rows.append({"layer": int(L), "target": "depth",
                     "k": 0, "heads_used": -1, "r2": r2, "how": "sum_all"})

        # Cam-delta: aggregate per (scene, t) first.
        from collections import defaultdict
        groups: dict[tuple[str, int], list[int]] = defaultdict(list)
        for i in range(len(sids)):
            if not valid[i]:
                continue
            groups[(sids[i], int(ts[i]))].append(i)
        agg_idx, agg_labels, agg_base_arr, agg_t = [], [], [], []
        for (sid, t), idx in groups.items():
            agg_idx.append(idx)
            agg_labels.append(cam_delta[idx[0]])
            agg_base_arr.append(_base_scene_id(sid))
            agg_t.append(t)

        def agg_vecs_for_heads(heads):
            # Mean over objects at (scene, t); concatenate per chosen head.
            H = len(heads)
            out_arr = np.zeros((len(agg_idx), H * vecs.shape[2]), dtype=np.float32)
            for i, ii in enumerate(agg_idx):
                pooled = vecs[ii][:, heads, :].mean(axis=0).astype(np.float32)
                out_arr[i] = pooled.reshape(-1)
            return out_arr

        agg_base_arr = np.asarray(agg_base_arr)
        agg_t = np.asarray(agg_t, dtype=np.int64)
        agg_labels = np.stack(agg_labels) if agg_labels else np.zeros((0, 6))
        mask_agg = agg_t >= t_min
        tr_agg = mask_agg & np.isin(agg_base_arr, list(train_base))
        te_agg = mask_agg & np.isin(agg_base_arr, list(test_base))

        for k in ks:
            k_use = min(k, len(cam_order))
            heads = cam_order[:k_use]
            X = agg_vecs_for_heads(heads)
            if tr_agg.sum() < 30 or te_agg.sum() < 10:
                r2 = float("nan")
            else:
                p = _fast_ridge_predict(X[tr_agg], agg_labels[tr_agg],
                                        X[te_agg], args.alpha)
                r2 = float(r2_score(agg_labels[te_agg], p, multioutput="uniform_average"))
            rows.append({"layer": int(L), "target": "cam",
                         "k": int(k), "heads_used": k_use, "r2": r2})

        # Sum of all heads for cam.
        X_all = agg_vecs_for_heads(list(range(vecs.shape[1])))
        X_sum = X_all.reshape(X_all.shape[0], vecs.shape[1], vecs.shape[2]).sum(axis=1)
        if tr_agg.sum() >= 30 and te_agg.sum() >= 10:
            p_cs = _fast_ridge_predict(X_sum[tr_agg], agg_labels[tr_agg],
                                       X_sum[te_agg], args.alpha)
            r2 = float(r2_score(agg_labels[te_agg], p_cs,
                                multioutput="uniform_average"))
        else:
            r2 = float("nan")
        rows.append({"layer": int(L), "target": "cam",
                     "k": 0, "heads_used": -1, "r2": r2, "how": "sum_all"})

        print(f"[L{L:02d}] depth top-1={next(r['r2'] for r in rows if r['layer']==L and r['target']=='depth' and r['k']==1):+.3f}  "
              f"top-3={next(r['r2'] for r in rows if r['layer']==L and r['target']=='depth' and r['k']==3):+.3f}  "
              f"sum={next(r['r2'] for r in rows if r['layer']==L and r['target']=='depth' and r.get('how')=='sum_all'):+.3f}  |  "
              f"cam top-1={next(r['r2'] for r in rows if r['layer']==L and r['target']=='cam' and r['k']==1):+.3f}  "
              f"top-3={next(r['r2'] for r in rows if r['layer']==L and r['target']=='cam' and r['k']==3):+.3f}  "
              f"sum={next(r['r2'] for r in rows if r['layer']==L and r['target']=='cam' and r.get('how')=='sum_all'):+.3f}")

    df = pd.DataFrame(rows)
    df.to_parquet(out / "cumulative.parquet")
    df.to_json(out / "cumulative.json", orient="records", indent=2)

    # ---- Plot: cumulative top-k R² vs k ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for target, ax in zip(["depth", "cam"], axes):
        sub = df[(df.target == target) & (df.k > 0)]
        for L in layers:
            s = sub[sub.layer == L].sort_values("k")
            ax.plot(s.k, s.r2, "-o", alpha=0.6, lw=1.0, ms=3, label=f"L{L:02d}")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 28])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "28"])
        ax.set_xlabel("top-k heads used")
        ax.set_ylabel("Ridge R²")
        ax.set_title(f"{target} cumulative top-k R² per layer")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=6, ncol=4, loc="lower right")
    fig.suptitle(
        "Per-head attribution — cumulative top-k R² across layers",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out / "cumulative.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot: sum-of-heads vs full-layer probe (if provided) ----
    if args.full_layer_probe and Path(args.full_layer_probe).exists():
        full = json.loads(Path(args.full_layer_probe).read_text())
        full_df = pd.DataFrame([
            {"layer": r["layer"],
             "depth_full": r["depth"]["r2"] if r["depth"] else float("nan"),
             "cam_full": r["cam_delta"]["r2_overall"] if r["cam_delta"] else float("nan")}
            for r in full
        ])
        sum_df = df[df.how.eq("sum_all")] if "how" in df.columns else df[df.k == 0]
        depth_sum = sum_df[sum_df.target == "depth"].set_index("layer").r2
        cam_sum = sum_df[sum_df.target == "cam"].set_index("layer").r2

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].plot(full_df.layer, full_df.depth_full, "-o", color="C0", lw=2,
                     label="full layer output h_{L+1}")
        axes[0].plot(depth_sum.index, depth_sum.values, "--s", color="C1", lw=2,
                     label="sum of head contributions Σ_h c_{L,h}")
        axes[0].set_xlabel("Layer")
        axes[0].set_ylabel("Depth Ridge R²")
        axes[0].set_title("Depth probe: attention write vs full layer")
        axes[0].axhline(0, color="gray", lw=0.6)
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=8)

        axes[1].plot(full_df.layer, full_df.cam_full, "-o", color="C0", lw=2,
                     label="full layer output h_{L+1}")
        axes[1].plot(cam_sum.index, cam_sum.values, "--s", color="C1", lw=2,
                     label="sum of head contributions Σ_h c_{L,h}")
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("Cam-Δ Ridge R² (overall)")
        axes[1].set_title("Cam-Δ probe: attention write vs full layer")
        axes[1].axhline(0, color="gray", lw=0.6)
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=8)
        fig.suptitle(
            "Attention-only residual write vs full layer output", y=1.02,
        )
        fig.tight_layout()
        fig.savefig(out / "attention_vs_fulllayer.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"\nsaved to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
