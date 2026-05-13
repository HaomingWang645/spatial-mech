#!/usr/bin/env python
"""Extract orthonormal nuisance directions (color + shape) for residualization.

Reads the base-model probe features at L17, fits multinomial logistic-regression
probes for color and for shape, takes the resulting weight vectors,
orthonormalizes them via a thin QR, and saves W ∈ R^(d × k).

The training-time Dirichlet loss can then project away these directions
before computing the energy: H_resid = H @ (I - W W^T), see Theorem 2 of
the theory_draft.md.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("res_basis")


def fit_probe_directions(
    H: np.ndarray, labels: np.ndarray, target_name: str
) -> np.ndarray:
    """Fit a multinomial LR probe and return its (orthonormalized) weight rows.

    Returns a matrix of shape (n_classes, d) of unit-norm weight vectors,
    one per class — the LR multinomial direction for each class is the
    "discriminating direction" for that class.
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(H)
    clf = LogisticRegression(
        multi_class="multinomial", solver="lbfgs",
        max_iter=2000, C=0.1, n_jobs=-1,
    )
    clf.fit(X, labels)
    train_acc = clf.score(X, labels)
    logger.info("  %s probe: %d classes, train acc=%.4f", target_name,
                len(clf.classes_), train_acc)
    # Compose with scaler to recover directions in the un-standardized H space.
    # The transformation is X = (H - mu) / std. Logits = X @ W^T = (H - mu) @ diag(1/std) @ W^T.
    # So in H-space, directions are W @ diag(1/std). We absorb the std rescaling
    # into the directions (we'll renormalize anyway).
    W_in_H_space = clf.coef_ / scaler.scale_  # (n_classes, d)
    # Normalize each row
    norms = np.linalg.norm(W_in_H_space, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    W_unit = W_in_H_space / norms
    return W_unit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe-npz", type=Path, required=True,
                   help="e.g. reports/probe_features/qwen_base.npz")
    p.add_argument("--out", type=Path, required=True,
                   help="Output .npz with key 'W' shape (d, k)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    d = np.load(args.probe_npz, allow_pickle=True)
    H = d["H"].astype(np.float32)            # (n_obj, d)
    colors = d["colors"]                       # (n_obj,) strings
    shapes = d["shapes"]                       # (n_obj,) strings
    logger.info("Loaded H of shape %s; %d colors, %d shapes",
                H.shape, len(set(colors)), len(set(shapes)))

    # Fit probes
    W_color = fit_probe_directions(H, colors, "color")
    W_shape = fit_probe_directions(H, shapes, "shape")

    # Stack and orthonormalize via thin QR
    W_stack = np.vstack([W_color, W_shape])   # (k_total, d)
    logger.info("  combined nuisance basis: %d candidate directions",
                W_stack.shape[0])
    # Q, R = qr of W_stack^T → Q has orthonormal columns spanning row(W_stack)
    Q, R = np.linalg.qr(W_stack.T)            # Q: (d, k_total), R: (k_total, k_total)
    # Drop near-zero rank columns (rank deficiency from co-linear class directions)
    rank_threshold = 1e-6
    diag_R = np.abs(np.diag(R))
    keep = diag_R > rank_threshold * diag_R.max()
    Q_clean = Q[:, keep]                       # (d, k_eff)
    logger.info("  effective rank: %d / %d", Q_clean.shape[1], W_stack.shape[0])

    # Verify orthonormality
    err = np.max(np.abs(Q_clean.T @ Q_clean - np.eye(Q_clean.shape[1])))
    logger.info("  orthonormality residual: %.2e", err)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, W=Q_clean.astype(np.float32),
             k_color=W_color.shape[0], k_shape=W_shape.shape[0],
             source=str(args.probe_npz))
    logger.info("Saved W of shape (%d, %d) to %s",
                Q_clean.shape[0], Q_clean.shape[1], args.out)


if __name__ == "__main__":
    main()
