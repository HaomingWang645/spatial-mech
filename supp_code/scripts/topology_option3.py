#!/usr/bin/env python
"""Option 3 topology tests — per-scene object-rep layout vs. true 3D layout.

Reads per-layer activation parquets/npys from an extraction directory, pools
each object's rep across its temporal-chunk occurrences within one scene,
and computes four parameter-free topology-preservation metrics against the
scene's ground-truth 3D object positions:

    (a) RSA — Spearman rank correlation of pairwise distances
    (b) Dirichlet energy ratio vs. permutation null on a kNN scene graph
    (c) k-NN overlap (local neighborhood preservation)
    (d) Spectral-embedding cosine similarity (Theorem 5.1 analog)

Each metric is computed per scene and aggregated across scenes. A
permutation null (shuffle object↔position labels within each scene) gives a
z-score per scene.

Output:
    <out_dir>/layer_metrics.parquet   # per (scene, layer, metric)
    <out_dir>/summary.parquet         # per layer: mean, stderr, z-score across scenes
    <out_dir>/pca_examples.npz        # PCA-top-3 of H_s per (scene, layer) for a
                                      # sampled subset (used by visualization).

Example:
    python scripts/topology_option3.py \
        --activations data/activations/tier_c_free6dof_qwen25vl_7b \
        --out data/probes/topology_option3/qwen25vl_7b_f16 \
        --knn-k 2 --n-permutations 200 --pca-example-scenes 8
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Per-scene pooling — average per-object rep across temporal chunks
# ---------------------------------------------------------------------------


def pool_scene_reps(df: pd.DataFrame, vecs: np.ndarray) -> dict[str, dict]:
    """Group by scene, then by object_id, averaging across frame_id.

    Returns ``{scene_id: {"object_ids": [...], "H": (n, D), "P": (n, 3)}}``.
    Scenes with fewer than 3 distinct objects are dropped (topology on n<3
    is undefined).
    """
    out: dict[str, dict] = {}
    for sid, gs in df.groupby("scene_id", sort=False):
        per_obj_vec: dict[int, list[np.ndarray]] = defaultdict(list)
        per_obj_pos: dict[int, tuple[float, float, float]] = {}
        for _, row in gs.iterrows():
            oid = int(row["object_id"])
            per_obj_vec[oid].append(vecs[int(row["vec_row"])])
            per_obj_pos[oid] = (
                float(row["centroid_x"]),
                float(row["centroid_y"]),
                float(row["centroid_z"]),
            )
        if len(per_obj_vec) < 3:
            continue
        object_ids = sorted(per_obj_vec.keys())
        H = np.stack(
            [np.mean(np.stack(per_obj_vec[oid], axis=0), axis=0) for oid in object_ids]
        ).astype(np.float32)
        P = np.array([per_obj_pos[oid] for oid in object_ids], dtype=np.float32)
        out[sid] = {"object_ids": object_ids, "H": H, "P": P}
    return out


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def pairwise_dist_upper(X: np.ndarray) -> np.ndarray:
    """Upper-triangle (i<j) of pairwise Euclidean distances."""
    n = X.shape[0]
    i, j = np.triu_indices(n, k=1)
    d = np.linalg.norm(X[i] - X[j], axis=1)
    return d


def knn_graph(P: np.ndarray, k: int) -> np.ndarray:
    """Symmetric kNN adjacency: A[i,j]=1 if j is in the k nearest of i OR vice versa."""
    n = P.shape[0]
    k = min(k, n - 1)
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn_idx = np.argsort(d, axis=1)[:, :k]
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in nn_idx[i]:
            A[i, j] = 1.0
            A[j, i] = 1.0
    return A


def dirichlet_energy(H: np.ndarray, A: np.ndarray) -> float:
    """E_G(H) = sum_{(i,j): A_{ij}=1 and i<j} ||h_i - h_j||^2."""
    n = H.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                total += float(np.sum((H[i] - H[j]) ** 2))
    # Divide by edge count (normalized = mean squared gap over edges)
    n_edges = max(1.0, float(A[np.triu_indices(n, 1)].sum()))
    return total / n_edges


def knn_overlap(H: np.ndarray, P: np.ndarray, k: int) -> float:
    n = H.shape[0]
    k = min(k, n - 1)
    dP = np.linalg.norm(P[:, None] - P[None], axis=-1)
    dH = np.linalg.norm(H[:, None] - H[None], axis=-1)
    np.fill_diagonal(dP, np.inf)
    np.fill_diagonal(dH, np.inf)
    nP = np.argsort(dP, axis=1)[:, :k]
    nH = np.argsort(dH, axis=1)[:, :k]
    overlaps = [len(set(nP[i]) & set(nH[i])) / k for i in range(n)]
    return float(np.mean(overlaps))


def spectral_embedding(A: np.ndarray, k: int = 2) -> np.ndarray:
    """Return the k eigenvectors of the Laplacian for the 2nd..(k+1)-th smallest
    eigenvalues (skipping the trivial constant one)."""
    D = np.diag(A.sum(axis=1))
    L = D - A
    w, V = np.linalg.eigh(L)
    # Skip eigenvalue 0 (constant eigenvector). Return next k.
    return V[:, 1 : 1 + k]  # shape (n, k)


def spectral_cos(H: np.ndarray, A: np.ndarray, k: int = 2) -> list[float]:
    """|cos(PC_i, z^(i+1))| for i=1..k."""
    n = H.shape[0]
    if n < k + 2:
        return [float("nan")] * k
    Hc = H - H.mean(axis=0, keepdims=True)
    # PCA via SVD
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    # Top-k left singular vectors in the sample space = HcV
    # We actually want the sample-side principal component scores, which are HcV[:, :k].
    # Equivalent to top eigenvectors of Hc Hc^T.
    _U, _S, _Vt = np.linalg.svd(Hc, full_matrices=False)
    PC = _U[:, :k] * _S[:k]  # (n, k) scores
    Z = spectral_embedding(A, k=k)
    out = []
    for i in range(k):
        a = PC[:, i]
        b = Z[:, i]
        na = float(np.linalg.norm(a) * np.linalg.norm(b))
        if na < 1e-12:
            out.append(float("nan"))
        else:
            out.append(abs(float(np.dot(a, b)) / na))
    return out


# ---------------------------------------------------------------------------
# Per-scene metric bundle
# ---------------------------------------------------------------------------


@dataclass
class SceneMetrics:
    rsa: float
    energy: float
    energy_null_mean: float
    energy_null_std: float
    knn_overlap: float
    knn_null_mean: float
    knn_null_std: float
    spectral_cos_1: float
    spectral_cos_2: float


def compute_scene_metrics(
    H: np.ndarray,
    P: np.ndarray,
    k: int = 2,
    n_permutations: int = 200,
    rng: np.random.Generator | None = None,
) -> SceneMetrics:
    rng = rng or np.random.default_rng(0)
    n = H.shape[0]

    # (a) RSA
    rsa = float(spearmanr(pairwise_dist_upper(H), pairwise_dist_upper(P)).statistic)

    # (b) Dirichlet energy on kNN(P)
    A = knn_graph(P, k=k)
    E = dirichlet_energy(H, A)

    # (c) k-NN overlap
    ov = knn_overlap(H, P, k=k)

    # (d) Spectral cosine (top 2)
    spec = spectral_cos(H, A, k=2)

    # Null: permute row order of H (equivalent to permuting object↔position labels)
    null_E = np.empty(n_permutations, dtype=np.float64)
    null_ov = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        Hp = H[perm]
        null_E[i] = dirichlet_energy(Hp, A)
        null_ov[i] = knn_overlap(Hp, P, k=k)

    return SceneMetrics(
        rsa=rsa,
        energy=E,
        energy_null_mean=float(null_E.mean()),
        energy_null_std=float(null_E.std()) or 1.0,
        knn_overlap=ov,
        knn_null_mean=float(null_ov.mean()),
        knn_null_std=float(null_ov.std()) or 1.0,
        spectral_cos_1=spec[0],
        spectral_cos_2=spec[1],
    )


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def load_layer(act_dir: Path, layer: int) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(act_dir / f"layer_{layer:02d}.parquet")
    vecs = np.load(act_dir / f"layer_{layer:02d}.npy", mmap_mode="r")
    return df, np.asarray(vecs)


def list_layers(act_dir: Path) -> list[int]:
    out = []
    for p in sorted(act_dir.glob("layer_*.parquet")):
        out.append(int(p.stem.split("_")[-1]))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--activations", required=True, help="Directory of layer_LL.parquet/npy")
    p.add_argument("--out", required=True)
    p.add_argument("--knn-k", type=int, default=2)
    p.add_argument("--n-permutations", type=int, default=200)
    p.add_argument("--layers", type=str, default=None, help="comma-separated; default=all")
    p.add_argument("--limit-scenes", type=int, default=None)
    p.add_argument("--pca-example-scenes", type=int, default=8,
                   help="How many scenes to dump PCA-top-3 coords for (used by visualizer)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    act_dir = Path(args.activations)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_layers = list_layers(act_dir)
    if args.layers is not None:
        keep = {int(x) for x in args.layers.split(",")}
        all_layers = [l for l in all_layers if l in keep]
    if not all_layers:
        sys.exit(f"no layers found in {act_dir}")

    print(f"[topology_option3] act_dir={act_dir}")
    print(f"[topology_option3] layers: {all_layers}")
    print(f"[topology_option3] out_dir={out_dir}")

    # Determine scene list from layer 0 (consistent across layers for Qwen-style extraction)
    df0, _ = load_layer(act_dir, all_layers[0])
    scenes = list(dict.fromkeys(df0["scene_id"].tolist()))  # preserve order
    if args.limit_scenes is not None:
        scenes = scenes[: args.limit_scenes]
    print(f"[topology_option3] n_scenes: {len(scenes)}")

    rows = []
    pca_examples: dict = {}
    # Prefer scenes with more objects for the PCA-example set (topology visually
    # clearer with 6-8 objects than with 3). Rank scenes by object count at
    # layer 0, break ties by scene id ordering.
    n_obj_by_scene = df0.groupby("scene_id")["object_id"].nunique().to_dict()
    scenes_ranked = sorted(scenes, key=lambda s: (-n_obj_by_scene.get(s, 0), s))
    chosen_example_scenes = set(scenes_ranked[: args.pca_example_scenes])

    # For each layer, pool reps and compute metrics per scene
    for layer in all_layers:
        df, vecs = load_layer(act_dir, layer)
        df = df[df["scene_id"].isin(scenes)]
        pooled = pool_scene_reps(df, vecs)
        rng = np.random.default_rng(args.seed + layer)

        for sid in scenes:
            if sid not in pooled:
                continue
            H = pooled[sid]["H"]
            P = pooled[sid]["P"]
            sm = compute_scene_metrics(
                H, P,
                k=args.knn_k,
                n_permutations=args.n_permutations,
                rng=rng,
            )
            rows.append({
                "scene_id": sid,
                "layer": layer,
                "n_objects": H.shape[0],
                "rsa": sm.rsa,
                "energy": sm.energy,
                "energy_null_mean": sm.energy_null_mean,
                "energy_null_std": sm.energy_null_std,
                "energy_ratio": sm.energy / max(sm.energy_null_mean, 1e-12),
                "energy_z": (sm.energy - sm.energy_null_mean) / sm.energy_null_std,
                "knn_overlap": sm.knn_overlap,
                "knn_null_mean": sm.knn_null_mean,
                "knn_null_std": sm.knn_null_std,
                "knn_z": (sm.knn_overlap - sm.knn_null_mean) / sm.knn_null_std,
                "spectral_cos_1": sm.spectral_cos_1,
                "spectral_cos_2": sm.spectral_cos_2,
            })

            if sid in chosen_example_scenes:
                Hc = H - H.mean(axis=0, keepdims=True)
                U, S, _ = np.linalg.svd(Hc, full_matrices=False)
                PCs = (U[:, :3] * S[:3]).astype(np.float32)
                pca_examples[(sid, layer)] = {
                    "pc": PCs,
                    "P": P,
                    "object_ids": np.array(pooled[sid]["object_ids"]),
                }

        print(f"[layer {layer:02d}] "
              f"mean RSA={np.mean([r['rsa'] for r in rows if r['layer']==layer]):.3f}  "
              f"mean E/Enull={np.mean([r['energy_ratio'] for r in rows if r['layer']==layer]):.3f}  "
              f"mean kNN-overlap={np.mean([r['knn_overlap'] for r in rows if r['layer']==layer]):.3f}  "
              f"mean cos1={np.nanmean([r['spectral_cos_1'] for r in rows if r['layer']==layer]):.3f}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_parquet(out_dir / "layer_metrics.parquet")

    # Summary per layer
    summary_rows = []
    for layer in all_layers:
        sub = metrics_df[metrics_df["layer"] == layer]
        if len(sub) == 0:
            continue
        summary_rows.append({
            "layer": layer,
            "n_scenes": len(sub),
            "rsa_mean": float(sub["rsa"].mean()),
            "rsa_stderr": float(sub["rsa"].std() / math.sqrt(len(sub))),
            "energy_ratio_mean": float(sub["energy_ratio"].mean()),
            "energy_ratio_stderr": float(sub["energy_ratio"].std() / math.sqrt(len(sub))),
            "energy_z_mean": float(sub["energy_z"].mean()),
            "knn_overlap_mean": float(sub["knn_overlap"].mean()),
            "knn_overlap_stderr": float(sub["knn_overlap"].std() / math.sqrt(len(sub))),
            "knn_z_mean": float(sub["knn_z"].mean()),
            "spectral_cos_1_mean": float(sub["spectral_cos_1"].mean()),
            "spectral_cos_2_mean": float(sub["spectral_cos_2"].mean()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_parquet(out_dir / "summary.parquet")

    # PCA examples for visualization
    if pca_examples:
        keys = list(pca_examples.keys())
        np.savez_compressed(
            out_dir / "pca_examples.npz",
            scene_layers=np.array([f"{k[0]}||{k[1]}" for k in keys]),
            **{f"pc_{i}": pca_examples[k]["pc"] for i, k in enumerate(keys)},
            **{f"P_{i}": pca_examples[k]["P"] for i, k in enumerate(keys)},
            **{f"oid_{i}": pca_examples[k]["object_ids"] for i, k in enumerate(keys)},
        )

    (out_dir / "run_info.json").write_text(json.dumps({
        "activations": str(act_dir),
        "knn_k": args.knn_k,
        "n_permutations": args.n_permutations,
        "n_scenes": len(scenes),
        "layers": all_layers,
    }, indent=2))
    print(f"[topology_option3] wrote metrics to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
