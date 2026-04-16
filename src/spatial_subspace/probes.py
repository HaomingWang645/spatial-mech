"""Q1 probes: linear, pairwise, MLP, PCA-then-linear (plan §5.1).

All probes expose a uniform ``fit_*`` function that takes train/val matrices
and returns a ``FitResult`` with R² and any probe-specific extras. The
splits module provides the per-scene / cross-trajectory splitters called out
in plan §5.2 (per-object split is a trivial variant and can be swapped in
by the caller).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

from .metrics import procrustes_error


@dataclass
class FitResult:
    model: Any
    r2: float
    extras: dict[str, Any] = field(default_factory=dict)


def fit_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alpha: float = 1.0,
) -> FitResult:
    model = Ridge(alpha=alpha).fit(X, y)
    pred = model.predict(X_val)
    return FitResult(
        model=model,
        r2=float(r2_score(y_val, pred, multioutput="uniform_average")),
        extras={
            "procrustes": float(procrustes_error(pred, y_val)),
            "pred": pred,
        },
    )


def fit_mlp_probe(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden: int = 256,
    max_iter: int = 200,
    random_state: int = 0,
) -> FitResult:
    model = MLPRegressor(
        hidden_layer_sizes=(hidden,),
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
    ).fit(X, y)
    pred = model.predict(X_val)
    return FitResult(
        model=model,
        r2=float(r2_score(y_val, pred, multioutput="uniform_average")),
        extras={"pred": pred},
    )


def fit_pca_linear(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    k: int,
    alpha: float = 1.0,
) -> FitResult:
    k = min(k, X.shape[0], X.shape[1])
    pca = PCA(n_components=k).fit(X)
    Xp = pca.transform(X)
    Xv = pca.transform(X_val)
    model = Ridge(alpha=alpha).fit(Xp, y)
    pred = model.predict(Xv)
    return FitResult(
        model=model,
        r2=float(r2_score(y_val, pred, multioutput="uniform_average")),
        extras={"pca": pca, "k": int(k), "pred": pred},
    )


def effective_rank(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    ks: list[int],
    target_fraction: float = 0.95,
) -> int:
    full = fit_linear_probe(X, y, X_val, y_val).r2
    for k in sorted(ks):
        r = fit_pca_linear(X, y, X_val, y_val, k=k).r2
        if r >= target_fraction * full:
            return int(k)
    return int(ks[-1])


def fit_pairwise_distance_probe(
    vecs: np.ndarray,             # (N, D) object vectors
    coords: np.ndarray,           # (N, 3)
    scene_of: np.ndarray,         # (N,) integer scene id
    train_scenes: np.ndarray,
    test_scenes: np.ndarray,
    alpha: float = 1.0,
) -> FitResult:
    """Ridge from ``[h_i, h_j]`` concatenation to per-scene-normalized distance.

    The label is ``d_ij / max_j d_ij`` within each scene (plan §3.7), so
    the probe is invariant to global scale and scene size.
    """

    def build(scenes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for s in scenes:
            idx = np.where(scene_of == s)[0]
            if len(idx) < 2:
                continue
            pairs = np.array(list(combinations(idx, 2)))
            d = np.linalg.norm(
                coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=-1
            )
            dmax = float(d.max())
            if dmax < 1e-8:
                continue
            feat = np.concatenate(
                [vecs[pairs[:, 0]], vecs[pairs[:, 1]]], axis=1
            )
            Xs.append(feat)
            ys.append(d / dmax)
        if not Xs:
            return np.zeros((0, vecs.shape[1] * 2)), np.zeros((0,))
        return np.concatenate(Xs), np.concatenate(ys)

    Xtr, ytr = build(train_scenes)
    Xte, yte = build(test_scenes)
    model = Ridge(alpha=alpha).fit(Xtr, ytr)
    pred = model.predict(Xte)
    return FitResult(
        model=model,
        r2=float(r2_score(yte, pred)) if len(yte) else float("nan"),
        extras={
            "spearman": float(spearmanr(yte, pred).statistic) if len(yte) > 1 else float("nan"),
            "n_train_pairs": int(len(ytr)),
            "n_test_pairs": int(len(yte)),
        },
    )


def scene_split(
    scene_ids: np.ndarray, train_frac: float = 0.8, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Split unique scenes into train/test sets (plan §5.2 per-scene split)."""
    rng = np.random.default_rng(seed)
    unique = np.unique(scene_ids)
    order = rng.permutation(len(unique))
    unique = unique[order]
    n_train = int(train_frac * len(unique))
    return unique[:n_train], unique[n_train:]


def object_split(
    object_ids: np.ndarray, train_frac: float = 0.8, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole object instances (plan §5.2 per-object split)."""
    rng = np.random.default_rng(seed)
    unique = np.unique(object_ids)
    order = rng.permutation(len(unique))
    unique = unique[order]
    n_train = int(train_frac * len(unique))
    return unique[:n_train], unique[n_train:]
