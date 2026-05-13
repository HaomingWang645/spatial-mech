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
    max_epochs: int = 300,
    patience: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    dropout: float = 0.1,
    device: str | None = None,
    random_state: int = 0,
) -> FitResult:
    """2-layer MLP probe trained on GPU (when available) with cross-val early stop.

    Upgraded from ``sklearn.MLPRegressor`` to PyTorch for two reasons:
      - sklearn's default early-stopping diverged badly at late layers in
        prior reports (R² collapsing far below zero). Here we carve out a
        10% val fold from ``(X, y)`` for early-stopping patience, leaving
        ``(X_val, y_val)`` purely held out.
      - GPU training turns the dominant wall-clock cost of the Q1 sweep
        (MLP at each layer × each model) from minutes to seconds.

    Features are standardized on the train fold before fitting.
    """
    import torch
    import torch.nn as nn

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    rng = np.random.default_rng(random_state)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    if y_val.ndim == 1:
        y_val = y_val[:, None]

    n = X.shape[0]
    n_val_in = max(int(round(0.1 * n)), 1)
    perm = rng.permutation(n)
    val_idx = perm[:n_val_in]
    tr_idx = perm[n_val_in:]

    mu = X[tr_idx].mean(axis=0, keepdims=True)
    sd = X[tr_idx].std(axis=0, keepdims=True) + 1e-6
    def _normalize(a: np.ndarray) -> np.ndarray:
        return (a - mu) / sd

    X_tr = torch.from_numpy(_normalize(X[tr_idx])).to(dev)
    y_tr = torch.from_numpy(y[tr_idx]).to(dev)
    X_ev_in = torch.from_numpy(_normalize(X[val_idx])).to(dev)
    y_ev_in = torch.from_numpy(y[val_idx]).to(dev)
    X_te = torch.from_numpy(_normalize(X_val)).to(dev)
    y_te = torch.from_numpy(y_val).to(dev)

    in_dim = X.shape[1]
    out_dim = y.shape[1]
    torch.manual_seed(random_state)
    net = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    bad = 0

    n_tr = X_tr.shape[0]
    for epoch in range(max_epochs):
        net.train()
        idx = torch.randperm(n_tr, device=dev)
        for s in range(0, n_tr, batch_size):
            sel = idx[s : s + batch_size]
            pred = net(X_tr[sel])
            loss = ((pred - y_tr[sel]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()

        net.eval()
        with torch.no_grad():
            val_pred = net(X_ev_in)
            val_loss = float(((val_pred - y_ev_in) ** 2).mean().item())
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred_te = net(X_te).cpu().numpy()

    y_te_np = y_val
    pred = pred_te
    if pred.shape[1] == 1 and y_te_np.shape[1] == 1:
        pred = pred.ravel()
        y_te_np = y_te_np.ravel()
    return FitResult(
        model=net,
        r2=float(r2_score(y_te_np, pred, multioutput="uniform_average")),
        extras={
            "pred": pred,
            "best_val_mse": best_val,
            "epochs_used": epoch + 1,
            "device": str(dev),
        },
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
