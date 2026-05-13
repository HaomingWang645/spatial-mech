#!/usr/bin/env python
"""Pilot experiments for the Dirichlet-loss training plan.

Four lightweight experiments validating the loss formulation before
launching a full VLM finetuning sweep.

  Pilot 1 (synthetic):   Verify Theorem 3 -- minimizing Dirichlet ratio
                         from random init recovers world-coordinate axes.
  Pilot 2 (layer scan):  Across layers of a pretrained VLM, the
                         Dirichlet ratio and the top-3-PC ↔ X alignment
                         R² are inversely correlated -- the loss
                         function is a faithful diagnostic.
  Pilot 3 (refinement):  At the peak layer, optimizing a small
                         perturbation δ to minimize the Dirichlet ratio
                         simultaneously raises the 3D-alignment R² --
                         the gradient is informative on real
                         pretrained activations.
  Pilot 4 (shuffle):     Same as Pilot 3 but with X coords shuffled
                         across objects -- expected to give no
                         improvement (control).

Usage
-----
    CUDA_VISIBLE_DEVICES=4 python scripts/dirichlet_pilot.py \\
        --out-dir reports/dirichlet_pilot

Output
------
    reports/dirichlet_pilot/
        pilot1_history.csv, pilot1.png
        pilot2_layer_scan.csv, pilot2.png
        pilot3_results.csv, pilot4_results.csv (shuffle control)
        summary.json
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (sys.path setup)

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dirichlet_loss import dirichlet_ratio  # noqa: E402


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def alignment_top3_with_X(H: torch.Tensor, X: torch.Tensor) -> float:
    """R² of predicting X from the top-3 left singular vectors of H.

    1.0 = top-3 PCs perfectly recover (a rotation of) X.
    ~0 = top-3 PCs are unrelated to X (random projection).

    This is the empirical analog of Theorem 1's "subspace recovery"
    statement: it measures the alignment between the top-3 PC subspace
    of H and the 3D coordinate space of X.
    """
    Hc = H - H.mean(0)
    Xc = X - X.mean(0)
    U, S, _ = torch.linalg.svd(Hc, full_matrices=False)
    top3 = U[:, :3] * S[:3]  # PC scores, (n, 3)
    A = torch.linalg.lstsq(top3, Xc).solution  # (3, 3)
    pred = top3 @ A
    ss_res = (pred - Xc).pow(2).sum()
    ss_tot = Xc.pow(2).sum() + 1e-8
    return float(1.0 - ss_res / ss_tot)


def load_scene_activations(act_dir: Path, layer: int) -> dict:
    """Load per-scene (H, X) for a given layer.

    For each scene, average the per-frame activations and 3D
    coordinates per object, giving an (n_obj, d) representation
    matrix and an (n_obj, 3) coordinate matrix.
    """
    parquet = pd.read_parquet(act_dir / f"layer_{layer:02d}.parquet")
    npy = np.load(act_dir / f"layer_{layer:02d}.npy", mmap_mode="r")

    scenes = {}
    for scene_id, sub in parquet.groupby("scene_id"):
        objs, coords = [], []
        for _obj_id, obj_sub in sub.groupby("object_id"):
            vec_rows = obj_sub["vec_row"].values
            avg_act = np.asarray(npy[vec_rows]).mean(axis=0)
            avg_coord = obj_sub[["centroid_x", "centroid_y", "centroid_z"]].mean().values
            objs.append(avg_act)
            coords.append(avg_coord)
        if len(objs) < 4:  # need at least 4 objects for meaningful PCA-3D
            continue
        scenes[scene_id] = {
            "H": np.stack(objs).astype(np.float32),
            "X": np.stack(coords).astype(np.float32),
        }
    return scenes


# --------------------------------------------------------------------- #
# Pilot 1: synthetic recovery
# --------------------------------------------------------------------- #


def laplacian_eigvecs(X: torch.Tensor, tau: float, k: int) -> torch.Tensor:
    """Return the (k+1) smallest-eigenvalue eigenvectors of the
    Gaussian-kernel Laplacian induced by X, EXCLUDING the constant
    z^(1).  Result has shape (n, k).

    These are the "z^(2), ..., z^(k+1)" of Theorem 3 — the predicted
    optimum of the Dirichlet-energy minimization problem.
    """
    from dirichlet_loss import gaussian_kernel
    W = gaussian_kernel(X, tau=tau, zero_diagonal=True)
    deg = W.sum(dim=-1)
    L = torch.diag(deg) - W
    L = L.cpu().double()  # eigh more stable in float64
    evals, evecs = torch.linalg.eigh(L)
    # smallest k+1 eigenvalues; drop index 0 (constant z^(1))
    return evecs[:, 1:k + 1].to(X.device, dtype=X.dtype)


def pilot1_synthetic(out_dir: Path, device: str) -> list[dict]:
    """Theorem 3 verification on two synthetic geometries.

    *Setup A* (random X in R^3): X are i.i.d. Gaussian.  This is in
    the "small-n / finite-bandwidth" regime where the Belkin-Niyogi
    limit (Theorem 3') hasn't kicked in -- the Laplacian eigenvectors
    z^(2), z^(3), z^(4) are NOT the coordinate functions, they are
    something more localized.  The right test of Theorem 3 here is:
    does the optimizer recover the top-3 Laplacian eigenvectors?
    (NOT X itself.)

    *Setup B* (X on a 4x4x4 grid): the Laplacian eigenvectors of a
    regular grid graph ARE (asymptotically) the coordinate functions,
    so on this structured X both recoveries should hold.

    For each setup we measure:
      - R²(H, top-3 Laplacian eigvecs):  what Theorem 3 directly predicts.
      - R²(H, X):                          what we ultimately want for VLMs.
    """
    print("\n=== Pilot 1: Synthetic recovery (Theorem 3 verification) ===")
    sigma_target = torch.tensor([3.0, 2.0, 1.0], device=device)
    tau = 2.0  # bandwidth roughly matched to the spread of X

    def run_setup(X: torch.Tensor, name: str) -> list[dict]:
        n = X.shape[0]
        print(f"\n  [{name}]  n={n}")
        # Compute the theoretically optimal H (Theorem 3 prediction):
        # top-3 Laplacian eigenvectors.
        Z = laplacian_eigvecs(X, tau=tau, k=3)  # (n, 3)
        r2_Z_X = alignment_top3_with_X(Z, X)
        dr_Z = float(dirichlet_ratio(Z * sigma_target.unsqueeze(0), X, tau=tau))
        print(f"  predicted optimum (Laplacian eigvecs): R²(Z, X)={r2_Z_X:.4f}  dr={dr_Z:.4f}")

        # Initialize U on the Stiefel manifold and optimize
        torch.manual_seed(0)
        U_init = torch.randn(n, 3, device=device)
        Q_init, _ = torch.linalg.qr(U_init)
        U = Q_init.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([U], lr=0.05)

        history = []
        log_steps = sorted(set(list(range(0, 100, 5)) + list(range(100, 2001, 50))))
        for step in range(2001):
            opt.zero_grad()
            H = U * sigma_target.unsqueeze(0)
            loss_d = dirichlet_ratio(H, X, tau=tau)
            loss_d.backward()
            opt.step()
            with torch.no_grad():
                U_svd, _, V_svd = torch.linalg.svd(U, full_matrices=False)
                U.data = U_svd @ V_svd

            if step in log_steps:
                with torch.no_grad():
                    H = U * sigma_target.unsqueeze(0)
                    history.append({
                        "setup": name,
                        "step": step,
                        "dirichlet_ratio": float(loss_d),
                        "R2_with_Z": alignment_top3_with_X(H, Z),
                        "R2_with_X": alignment_top3_with_X(H, X),
                        "predicted_optimum_R2_Z_X": r2_Z_X,
                    })
        f = history[-1]
        print(
            f"  step {f['step']}: dr={f['dirichlet_ratio']:.4f}  "
            f"R²(H, Z_predicted)={f['R2_with_Z']:.4f}  "
            f"R²(H, X)={f['R2_with_X']:.4f}"
        )
        return history

    # Setup A: random X (the realistic small-bandwidth regime)
    torch.manual_seed(0)
    X_A = torch.randn(64, 3, device=device) * 2.0
    h_A = run_setup(X_A, "random_X")

    # Setup B: X on a 4x4x4 grid (Belkin-Niyogi-friendly structure)
    grid = torch.linspace(-2, 2, 4, device=device)
    xx, yy, zz = torch.meshgrid(grid, grid, grid, indexing="ij")
    X_B = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # (64, 3)
    h_B = run_setup(X_B, "grid_X")

    history = h_A + h_B
    df = pd.DataFrame(history)
    df.to_csv(out_dir / "pilot1_history.csv", index=False)

    # Plot: 2x2 panels
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for col_i, setup in enumerate(["random_X", "grid_X"]):
        sub = df[df["setup"] == setup]
        axes[0, col_i].plot(sub["step"], sub["dirichlet_ratio"], "-",
                            color="tab:purple")
        axes[0, col_i].set_xlabel("step")
        axes[0, col_i].set_ylabel("Dirichlet ratio")
        axes[0, col_i].set_title(f"{setup}: loss decreases")
        axes[0, col_i].grid(alpha=0.3)
        axes[1, col_i].plot(sub["step"], sub["R2_with_Z"], "-",
                            color="tab:green",
                            label="R²(H, Z_predicted) — Theorem 3 prediction")
        axes[1, col_i].plot(sub["step"], sub["R2_with_X"], "-",
                            color="tab:orange",
                            label="R²(H, X) — direct recovery")
        axes[1, col_i].axhline(
            sub["predicted_optimum_R2_Z_X"].iloc[0], color="gray",
            linestyle="--", alpha=0.6, label="R²(Z, X) — predicted optimum")
        axes[1, col_i].set_xlabel("step")
        axes[1, col_i].set_ylabel("Alignment R²")
        axes[1, col_i].set_ylim([-0.05, 1.05])
        axes[1, col_i].set_title(f"{setup}: alignment")
        axes[1, col_i].legend(fontsize=8, loc="lower right")
        axes[1, col_i].grid(alpha=0.3)
    fig.suptitle(
        "Pilot 1: minimizing Dirichlet ratio under the Stiefel "
        "spectral constraint recovers the top-3 Laplacian eigenvectors\n"
        "(which equal X for grid-structured input but not for random Gaussian X)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "pilot1.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return history

    final = history[-1]
    print(
        f"  step {final['step']:5d}: dr={final['dirichlet_ratio']:.4f}  "
        f"R²={final['top3_X_alignment_R2']:.4f}  H_var={final['H_var']:.3f}"
    )

    df = pd.DataFrame(history)
    df.to_csv(out_dir / "pilot1_history.csv", index=False)

    # Plot: side-by-side curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(df["step"], df["dirichlet_ratio"], "-", color="tab:purple")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("Dirichlet ratio")
    axes[0].set_title("Loss decreases as expected")
    axes[0].grid(alpha=0.3)
    axes[1].plot(df["step"], df["top3_X_alignment_R2"], "-", color="tab:green")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("Top-3 PC ↔ X alignment R²")
    axes[1].set_title("PCA-3D recovery emerges")
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].grid(alpha=0.3)
    fig.suptitle(
        "Pilot 1: Random-init H driven by Dirichlet loss alone "
        "recovers the 3D coordinate axes in its top-3 PCs"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "pilot1.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return history


# --------------------------------------------------------------------- #
# Pilot 2: layer scan
# --------------------------------------------------------------------- #


def pilot2_layer_scan(act_dir: Path, model_slug: str, out_dir: Path) -> tuple[pd.DataFrame, float]:
    print(f"\n=== Pilot 2: Layer scan ({model_slug}) ===")
    n_layers = len(list(act_dir.glob("layer_*.npy")))
    print(f"  {n_layers} layers")

    rows = []
    for layer in range(n_layers):
        scenes = load_scene_activations(act_dir, layer)
        keys = list(scenes.keys())[:60]  # subset for speed
        ratios, aligns = [], []
        for k in keys:
            H = torch.from_numpy(scenes[k]["H"])
            X = torch.from_numpy(scenes[k]["X"])
            ratios.append(float(dirichlet_ratio(H, X, tau=1.0)))
            aligns.append(alignment_top3_with_X(H, X))
        rows.append({
            "layer": layer,
            "n_scenes": len(ratios),
            "dirichlet_ratio": float(np.mean(ratios)),
            "top3_X_alignment_R2": float(np.mean(aligns)),
        })
        if layer % 4 == 0:
            print(
                f"  L{layer:02d}: dr={rows[-1]['dirichlet_ratio']:.4f}  "
                f"R²={rows[-1]['top3_X_alignment_R2']:.4f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pilot2_layer_scan.csv", index=False)

    corr = float(np.corrcoef(df["dirichlet_ratio"], df["top3_X_alignment_R2"])[0, 1])
    print(f"  Pearson correlation = {corr:+.3f} (expected: large negative)")

    # Plot: dual-axis layer scan
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(df["layer"], df["dirichlet_ratio"], "o-", color="tab:purple",
             label="Dirichlet ratio")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Dirichlet ratio (lower = more 3D-smooth)", color="tab:purple")
    ax1.tick_params(axis="y", labelcolor="tab:purple")
    ax2 = ax1.twinx()
    ax2.plot(df["layer"], df["top3_X_alignment_R2"], "s-", color="tab:green",
             label="Top-3 PC ↔ X R²")
    ax2.set_ylabel("Top-3 PC ↔ X R²", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    fig.suptitle(
        f"Pilot 2: Layer-wise Dirichlet ratio and 3D alignment "
        f"({model_slug})\nPearson corr = {corr:+.3f}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "pilot2.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return df, corr


# --------------------------------------------------------------------- #
# Pilot 3 / 4: perturbation experiment
# --------------------------------------------------------------------- #


def pilot3_perturbation(
    act_dir: Path,
    layer: int,
    out_dir: Path,
    device: str,
    shuffled: bool = False,
    n_scenes: int = 60,
    n_steps: int = 500,
) -> pd.DataFrame:
    """Optimize H + delta to minimize Dirichlet ratio at the peak layer.

    Pilot 3 (shuffled=False): use true X.  Both Dirichlet ratio and
    alignment-with-true-X are evaluated against the same X.

    Pilot 4 (shuffled=True): optimize toward a *shuffled* X, but
    evaluate alignment against the *true* X.  This is the proper
    control: if the loss were "any 3D structure works", alignment
    would be unchanged or randomly affected.  If the loss truly
    requires the correct labels, alignment with the true X should
    *decrease* under shuffled-X optimization.
    """
    name = "Pilot 4 (shuffle)" if shuffled else "Pilot 3 (refinement)"
    print(f"\n=== {name} at layer {layer} ===")
    scenes = load_scene_activations(act_dir, layer)

    rows = []
    keys = list(scenes.keys())[:n_scenes]
    rng = torch.Generator(device="cpu").manual_seed(0)
    for k in keys:
        H_orig = torch.from_numpy(scenes[k]["H"]).to(device)
        X_true = torch.from_numpy(scenes[k]["X"]).to(device)

        # The X we optimize against (could be shuffled).
        if shuffled:
            perm = torch.randperm(X_true.shape[0], generator=rng).to(device)
            X_opt = X_true[perm]
        else:
            X_opt = X_true

        # initial metrics — always w.r.t. the *true* X
        with torch.no_grad():
            r_init = float(dirichlet_ratio(H_orig, X_true, tau=1.0))
            a_init = alignment_top3_with_X(H_orig, X_true)

        # optimize a small additive perturbation (against X_opt)
        delta = torch.zeros_like(H_orig, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=0.01)
        for _ in range(n_steps):
            opt.zero_grad()
            H_pert = H_orig + delta
            loss_d = dirichlet_ratio(H_pert, X_opt, tau=1.0)
            loss_reg = 0.001 * delta.pow(2).mean()
            (loss_d + loss_reg).backward()
            opt.step()

        # final metrics — always w.r.t. the *true* X
        with torch.no_grad():
            H_final = H_orig + delta
            r_final = float(dirichlet_ratio(H_final, X_true, tau=1.0))
            a_final = alignment_top3_with_X(H_final, X_true)
            delta_pct = float((delta.norm() / H_orig.norm()) * 100)

        rows.append({
            "scene_id": k,
            "r_init": r_init, "r_final": r_final, "r_delta": r_final - r_init,
            "a_init": a_init, "a_final": a_final, "a_delta": a_final - a_init,
            "delta_norm_pct": delta_pct,
        })

    df = pd.DataFrame(rows)
    suffix = "_shuffled" if shuffled else ""
    df.to_csv(out_dir / f"pilot3{suffix}_results.csv", index=False)

    print(
        f"  Dirichlet ratio (vs TRUE X)  mean={df['r_delta'].mean():+.4f}  "
        f"(init {df['r_init'].mean():.4f} → final {df['r_final'].mean():.4f})"
    )
    print(
        f"  Alignment R² (vs TRUE X)     mean={df['a_delta'].mean():+.4f}  "
        f"(init {df['a_init'].mean():.4f} → final {df['a_final'].mean():.4f})"
    )
    print(f"  ‖δ‖/‖H‖ mean={df['delta_norm_pct'].mean():.2f}%")
    return df


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("reports/dirichlet_pilot"))
    parser.add_argument(
        "--act-dir",
        type=Path,
        default=Path("data/activations/tier_c_free6dof_f32_internvl3_8b"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = "cuda:0"
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    t0 = time.time()
    h1 = pilot1_synthetic(args.out_dir, device)
    df2, corr = pilot2_layer_scan(args.act_dir, args.act_dir.name, args.out_dir)

    peak_layer = int(df2.loc[df2["dirichlet_ratio"].idxmin(), "layer"])
    print(f"\nPeak layer (lowest Dirichlet ratio): L{peak_layer}")

    df3 = pilot3_perturbation(args.act_dir, peak_layer, args.out_dir, device, shuffled=False)
    df4 = pilot3_perturbation(args.act_dir, peak_layer, args.out_dir, device, shuffled=True)

    # Pilot 1 summary: pull final values for both setups
    df_h1 = pd.DataFrame(h1)
    p1_random = df_h1[df_h1["setup"] == "random_X"].iloc[-1].to_dict()
    p1_grid = df_h1[df_h1["setup"] == "grid_X"].iloc[-1].to_dict()
    summary = {
        "model": args.act_dir.name,
        "peak_layer": peak_layer,
        "pilot1_random_dirichlet": p1_random["dirichlet_ratio"],
        "pilot1_random_R2_with_Z": p1_random["R2_with_Z"],
        "pilot1_random_R2_with_X": p1_random["R2_with_X"],
        "pilot1_grid_dirichlet": p1_grid["dirichlet_ratio"],
        "pilot1_grid_R2_with_Z": p1_grid["R2_with_Z"],
        "pilot1_grid_R2_with_X": p1_grid["R2_with_X"],
        "pilot2_layer_correlation": corr,
        "pilot3_mean_dirichlet_delta": float(df3["r_delta"].mean()),
        "pilot3_mean_alignment_delta": float(df3["a_delta"].mean()),
        "pilot3_mean_delta_pct": float(df3["delta_norm_pct"].mean()),
        "pilot4_mean_dirichlet_delta": float(df4["r_delta"].mean()),
        "pilot4_mean_alignment_delta": float(df4["a_delta"].mean()),
        "pilot4_mean_delta_pct": float(df4["delta_norm_pct"].mean()),
        "wall_time_seconds": time.time() - t0,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
