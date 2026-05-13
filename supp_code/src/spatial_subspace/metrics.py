"""Probe evaluation metrics (plan §5.3)."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import spearmanr


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum())
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def procrustes_align(
    pred: np.ndarray, true: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Best rigid + scale alignment of ``pred`` to ``true`` (Kabsch + scale).

    Returns (aligned_pred, scale, rotation, translation). For each sample row:
        true_hat = scale * pred_centered @ R + mu_true
    """
    assert pred.shape == true.shape
    mu_p = pred.mean(axis=0)
    mu_t = true.mean(axis=0)
    P = pred - mu_p
    T = true - mu_t
    M = P.T @ T
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    det = float(np.sign(np.linalg.det(U @ Vt)))
    D = np.eye(U.shape[1])
    D[-1, -1] = det
    R = U @ D @ Vt
    var_p = float((P ** 2).sum())
    trace = float((S * np.diag(D)).sum())
    scale = trace / var_p if var_p > 1e-12 else 1.0
    aligned = scale * P @ R + mu_t
    translation = mu_t - scale * mu_p @ R
    return aligned, float(scale), R, translation


def procrustes_error(pred: np.ndarray, true: np.ndarray) -> float:
    aligned, *_ = procrustes_align(pred, true)
    return float(np.sqrt(((aligned - true) ** 2).sum(axis=1).mean()))


def pairwise_distance_spearman(pred: np.ndarray, true: np.ndarray) -> float:
    assert pred.shape == true.shape
    if pred.shape[0] < 2:
        return float("nan")
    idx = np.array(list(combinations(range(pred.shape[0]), 2)))
    dp = np.linalg.norm(pred[idx[:, 0]] - pred[idx[:, 1]], axis=-1)
    dt = np.linalg.norm(true[idx[:, 0]] - true[idx[:, 1]], axis=-1)
    return float(spearmanr(dp, dt).statistic)
