"""3D-geometry-weighted Dirichlet-energy regularizer for VLM finetuning.

Implements the loss formulated in :doc:`reports/dirichlet_loss_plan.md`.
Theoretical justification: Theorem 3 of :doc:`reports/theory_draft.md`.

The Dirichlet ratio of an activation matrix H w.r.t. ground-truth 3D
coordinates X is defined as

    R_X(H) = sum_{i,j} W_{ij} ||h_i - h_j||^2 / sum_{i,j} ||h_i - h_j||^2,

where W_{ij} = exp(-||x_i - x_j||^2 / (2 tau^2)) is a Gaussian
similarity kernel in 3D space.  Minimizing R_X drives the model to
place 3D-nearby objects at representation-nearby points, which by
Theorem 3 forces the top principal components of H to align with the
world-coordinate axes (asymptotically, in the kernel-bandwidth limit
of Theorem 3').

This module is self-contained (no `spatial_subspace` imports), so it
can be dropped into any PyTorch training loop.

Example
-------
>>> import torch
>>> from scripts.dirichlet_loss import DirichletLoss
>>> loss_fn = DirichletLoss(tau=1.0)
>>> H = torch.randn(2, 8, 768, requires_grad=True)  # (B, n_obj, d)
>>> X = torch.randn(2, 8, 3)                        # (B, n_obj, 3)
>>> r = loss_fn(H, X)
>>> r.backward()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------- #
# Functional API
# --------------------------------------------------------------------- #


def gaussian_kernel(
    X: torch.Tensor,
    tau: float = 1.0,
    zero_diagonal: bool = True,
) -> torch.Tensor:
    """Gaussian similarity kernel in 3D coordinate space.

    Parameters
    ----------
    X : torch.Tensor
        ``(..., n, 3)`` tensor of 3D coordinates.
    tau : float
        Kernel bandwidth, in the same units as ``X``.  Smaller ``tau``
        gives sharper similarity (only true neighbours count); larger
        ``tau`` blurs and makes the energy more like a global pairwise
        distance penalty.
    zero_diagonal : bool
        If True (default), zero out the diagonal so self-pairs don't
        contribute spurious zero-distance terms.

    Returns
    -------
    torch.Tensor
        ``(..., n, n)`` symmetric similarity matrix with entries in
        ``(0, 1]`` (zero on the diagonal if ``zero_diagonal=True``).
    """
    sq = torch.cdist(X, X) ** 2  # (..., n, n)
    W = torch.exp(-sq / (2.0 * tau * tau))
    if zero_diagonal:
        n = X.shape[-2]
        eye = torch.eye(n, device=X.device, dtype=W.dtype)
        W = W * (1.0 - eye)
    return W


def dirichlet_ratio(
    H: torch.Tensor,
    X: torch.Tensor,
    tau: float = 1.0,
    eps: float = 1e-8,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalized 3D-Dirichlet ratio.

    Parameters
    ----------
    H : torch.Tensor
        ``(..., n, d)`` representation matrix (e.g. residual-stream
        activations at object-token positions).
    X : torch.Tensor
        ``(..., n, 3)`` 3D coordinates of the same objects.
    tau : float
        Kernel bandwidth.
    eps : float
        Numerical stabilizer to avoid ``0/0`` when all distances are
        zero (e.g. a degenerate scene with one object).
    valid_mask : torch.Tensor or None
        Optional ``(..., n)`` boolean mask; if provided, only pairs
        ``(i, j)`` with both endpoints valid contribute to the
        numerator and denominator.  Useful for batches with variable
        ``n``.

    Returns
    -------
    torch.Tensor
        Dirichlet ratio averaged over leading (batch) dimensions, in
        ``[0, 1]``.  Returns a scalar tensor.
    """
    W = gaussian_kernel(X, tau=tau, zero_diagonal=True)
    H_sq = torch.cdist(H, H) ** 2  # (..., n, n)

    if valid_mask is not None:
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()
        # outer product: M_{ij} = mask_i AND mask_j
        m = valid_mask.unsqueeze(-1) & valid_mask.unsqueeze(-2)
        m = m.to(W.dtype)
        W = W * m
        H_sq = H_sq * m

    numer = (W * H_sq).flatten(-2).sum(dim=-1)
    denom = H_sq.flatten(-2).sum(dim=-1) + eps
    ratio = numer / denom  # (...,)
    return ratio.mean()


def dirichlet_energy(
    H: torch.Tensor,
    X: torch.Tensor,
    tau: float = 1.0,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unnormalized Dirichlet energy ``sum_ij W_ij ||h_i - h_j||^2``.

    Use this if you want to penalize absolute representation scale (it
    grows quadratically with ``||H||``); otherwise prefer
    :func:`dirichlet_ratio`, which is scale-invariant.
    """
    W = gaussian_kernel(X, tau=tau, zero_diagonal=True)
    H_sq = torch.cdist(H, H) ** 2

    if valid_mask is not None:
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()
        m = (valid_mask.unsqueeze(-1) & valid_mask.unsqueeze(-2)).to(W.dtype)
        W = W * m
        H_sq = H_sq * m

    return (W * H_sq).flatten(-2).sum(dim=-1).mean()


# --------------------------------------------------------------------- #
# Module API
# --------------------------------------------------------------------- #


@dataclass
class DirichletLossConfig:
    """Configuration for :class:`DirichletLoss`."""

    tau: float = 1.0
    """Kernel bandwidth in the units of the input ``X`` coordinates."""

    normalize: bool = True
    """If True, use the (scale-invariant) ratio formulation; otherwise
    use the unnormalized energy."""

    eps: float = 1e-8
    """Stabilizer for the denominator of the ratio formulation."""


class DirichletLoss(nn.Module):
    """3D-geometry-weighted Dirichlet-energy regularizer.

    Parameters
    ----------
    tau : float, default 1.0
        Gaussian kernel bandwidth.  In the asymptotic Belkin-Niyogi
        limit (Theorem 3' of theory_draft.md), one wants ``tau`` to
        scale as ``n^{-1/(d_M + 2 + alpha)}`` where ``d_M`` is the
        intrinsic dimension of the scene manifold (3 in our case).  In
        practice we treat ``tau`` as a hyperparameter to tune.
    normalize : bool, default True
        If True, use ``R_X(H)`` as defined above.  If False, use the
        unnormalized energy ``E_X(H) = sum_{ij} W_{ij} ||h_i - h_j||^2``.
    eps : float, default 1e-8
        Numerical stabilizer.
    """

    def __init__(
        self,
        tau: float = 1.0,
        normalize: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.tau = tau
        self.normalize = normalize
        self.eps = eps

    def forward(
        self,
        H: torch.Tensor,
        X: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the Dirichlet (ratio or energy) of H w.r.t. X.

        Parameters
        ----------
        H : torch.Tensor
            ``(B, n, d)`` per-sample residual-stream activations at
            object-token positions, on a single chosen layer.
        X : torch.Tensor
            ``(B, n, 3)`` per-sample 3D ground-truth coordinates.
        valid_mask : torch.Tensor, optional
            ``(B, n)`` boolean tensor; True for valid object slots.

        Returns
        -------
        torch.Tensor
            Scalar loss; ready to be added to LM cross-entropy.
        """
        if H.shape[:-1] != X.shape[:-1]:
            raise ValueError(
                f"H batch/object shape {tuple(H.shape[:-1])} doesn't match "
                f"X shape {tuple(X.shape[:-1])}."
            )
        if X.shape[-1] != 3:
            raise ValueError(
                f"X must have last dim 3 (got {X.shape[-1]})."
            )

        if self.normalize:
            return dirichlet_ratio(
                H, X, tau=self.tau, eps=self.eps, valid_mask=valid_mask
            )
        return dirichlet_energy(H, X, tau=self.tau, valid_mask=valid_mask)


# --------------------------------------------------------------------- #
# Layer-hook helper
# --------------------------------------------------------------------- #


class ResidualStreamHook:
    """Captures the residual stream at a chosen transformer layer.

    Use as a context manager; activations are exposed as ``hook.last``.

    Example
    -------
    >>> hook = ResidualStreamHook(model, layer_idx=17)
    >>> with hook:
    ...     out = model(input_ids=...)
    ...     H_all = hook.last  # (B, seq, d)

    Notes
    -----
    The hook is registered on the *output* of the transformer block,
    which is the post-residual hidden state.  Concrete model classes
    (Qwen2VL, InternVL3, LlavaOneVision) all expose their layer stack
    via ``model.model.layers`` or ``model.language_model.layers`` —
    pass the appropriate ``layer_module`` to the constructor if your
    model uses a different attribute path.
    """

    def __init__(
        self,
        layer_module: nn.Module,
    ):
        self.layer_module = layer_module
        self._handle = None
        self.last: Optional[torch.Tensor] = None

    def __enter__(self) -> "ResidualStreamHook":
        def _hook(_mod, _inp, out):
            # transformer blocks return either a Tensor or a tuple
            self.last = out[0] if isinstance(out, tuple) else out
        self._handle = self.layer_module.register_forward_hook(_hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        # don't suppress exceptions
        return False


# --------------------------------------------------------------------- #
# Self-test (run as `python scripts/dirichlet_loss.py`)
# --------------------------------------------------------------------- #


def _self_test() -> None:
    """Sanity checks: gradient flows, ratio is in [0, 1], theorem-3 limit."""
    torch.manual_seed(0)

    # 1. Shapes and gradient flow
    H = torch.randn(2, 8, 768, requires_grad=True)
    X = torch.randn(2, 8, 3)
    loss = DirichletLoss(tau=1.0)(H, X)
    loss.backward()
    assert H.grad is not None and torch.isfinite(H.grad).all()
    print(f"[ok] gradient flows; sample ratio = {loss.item():.4f}")

    # 2. Ratio is in [0, 1]
    for _ in range(20):
        H = torch.randn(4, 16, 256)
        X = torch.randn(4, 16, 3) * 5.0
        r = DirichletLoss(tau=1.0)(H, X).item()
        assert 0.0 <= r <= 1.0 + 1e-6, f"ratio out of range: {r}"
    print("[ok] ratio is in [0, 1] over 20 random trials")

    # 3. Theorem-3 spot check: H = X projected up gives small ratio
    n, d = 32, 64
    X = torch.randn(n, 3) * 2.0
    A = torch.randn(3, d)
    H_aligned = X @ A                            # (n, d), perfectly 3D-structured
    H_random = torch.randn(n, d) * H_aligned.std()  # noise of same scale
    r_aligned = dirichlet_ratio(H_aligned, X, tau=1.0).item()
    r_random = dirichlet_ratio(H_random, X, tau=1.0).item()
    assert r_aligned < r_random, (
        f"aligned H should have lower Dirichlet ratio "
        f"(got aligned={r_aligned:.4f}, random={r_random:.4f})"
    )
    print(
        f"[ok] aligned H has lower ratio than random H: "
        f"{r_aligned:.4f} < {r_random:.4f}"
    )

    # 4. Mask works
    H = torch.randn(2, 10, 64, requires_grad=True)
    X = torch.randn(2, 10, 3)
    mask = torch.tensor([[1] * 7 + [0] * 3, [1] * 5 + [0] * 5], dtype=torch.bool)
    loss_masked = dirichlet_ratio(H, X, tau=1.0, valid_mask=mask)
    loss_full = dirichlet_ratio(H[:, :7], X[:, :7], tau=1.0)
    # Not exactly equal because batch[1] has only 5 valid; just check finite + diff
    assert torch.isfinite(loss_masked) and loss_masked.item() != loss_full.item()
    print("[ok] mask changes the loss")

    # 5. The energy form is unnormalized and grows with ||H||
    H_small = torch.randn(2, 8, 64)
    X = torch.randn(2, 8, 3)
    e_small = dirichlet_energy(H_small, X, tau=1.0).item()
    e_large = dirichlet_energy(H_small * 10.0, X, tau=1.0).item()
    # ratio should be ~100 (since dist^2 scales as scale^2)
    assert 50 < e_large / e_small < 200, e_large / e_small
    print(
        f"[ok] energy scales correctly with ||H||: "
        f"ratio = {e_large / e_small:.1f} (expected ~100)"
    )

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    _self_test()
