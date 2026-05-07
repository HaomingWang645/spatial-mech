#!/usr/bin/env python
"""§4.1 supplementary figure: per-patch residual-stream activations as a
grid heatmap.

For visualization only — synthetic activation values are drawn at each
ViT patch position to make the spatial structure visible.  The purpose
is to show:

    "for every ViT patch p in frame t, the layer-l residual stream
     produces a value h_{p,t} (averaged over d dims) that we then
     coverage-weighted-pool"

Layout:
  Left:  one rendered frame with the patch grid drawn on top.
  Right: an n x n heatmap; each cell = average activation magnitude
         (over d dims) at that patch.

Outputs: figures/fig_sec41_patch_activations.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

SCENE_DIR = Path("data/tier_c_free6dof_f32/s_002abd79ae_t1")
OUT_PNG   = Path("figures/fig_sec41_patch_activations.png")
OUT_PDF   = Path("figures/fig_sec41_patch_activations.pdf")

GRID_H = 32
GRID_W = 32
N_DIMS = 64   # number of synthetic dims to average over


def synthetic_activations(gh, gw, n_dims, seed=0):
    """Generate (gh, gw, n_dims) synthetic activation vectors.

    Each dim k = 0..n_dims-1 is a smooth 2D field over the patch grid:
    a sum of low-frequency sinusoids at random phases plus a localized
    Gaussian bump that mimics object-token sensitivity.  We will then
    average across the dim axis to produce a single per-patch scalar.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(
        np.linspace(0, np.pi, gh),
        np.linspace(0, np.pi, gw),
        indexing="ij",
    )
    out = np.zeros((gh, gw, n_dims), dtype=np.float32)
    for k in range(n_dims):
        f = 0.0
        for _ in range(4):
            fx, fy = rng.integers(1, 4, size=2)
            phx, phy = rng.uniform(0, 2 * np.pi, size=2)
            amp     = rng.uniform(0.4, 1.2)
            f = f + amp * np.cos(fx * xx + phx) * np.sin(fy * yy + phy)
        # localized object bumps at a few clustered locations
        for _ in range(2):
            cy, cx = rng.uniform(0.2, 0.8, size=2) * np.array([gh, gw])
            sigma  = rng.uniform(2.0, 4.0)
            gauss  = np.exp(-((np.arange(gh)[:, None] - cy) ** 2 +
                              (np.arange(gw)[None, :] - cx) ** 2)
                            / (2 * sigma ** 2))
            f = f + 1.2 * gauss
        out[..., k] = f
    out = (out - out.mean(axis=(0, 1), keepdims=True)) / (
        out.std(axis=(0, 1), keepdims=True) + 1e-6)
    return out


def main():
    # Pick a representative frame
    sc = json.load(open(SCENE_DIR / "scene.json"))
    fr_id = 6
    img = np.array(Image.open(SCENE_DIR / sc["frames"][fr_id]["image_path"]))
    Hpx, Wpx, _ = img.shape

    # Synthesize activations on the visualization grid, then AVERAGE
    # over the dim axis to get a single scalar per patch -> n x n heatmap.
    A_full = synthetic_activations(GRID_H, GRID_W, N_DIMS, seed=2)
    A_avg  = A_full.mean(axis=-1)              # (gh, gw)
    print(f"Activation grid: shape={A_avg.shape}  "
          f"(averaged over {N_DIMS} dims)")

    # ============== figure ==============
    fig = plt.figure(figsize=(13.0, 6.4), dpi=140)
    gs = GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.0, 1.0],
        wspace=0.15, left=0.04, right=0.985,
        top=0.90, bottom=0.10,
    )

    # ----- Panel A: rendered frame with patch grid overlay -----
    axA = fig.add_subplot(gs[0, 0])
    axA.imshow(img)
    px = Hpx / GRID_H
    py = Wpx / GRID_W
    for k in range(GRID_H + 1):
        axA.axhline(k * px - 0.5, color="white", linewidth=0.4, alpha=0.55)
    for k in range(GRID_W + 1):
        axA.axvline(k * py - 0.5, color="white", linewidth=0.4, alpha=0.55)
    axA.set_title(f"(A)  one frame  +  ViT patch grid ({GRID_H}×{GRID_W})",
                  fontsize=14, weight="bold", pad=8)
    axA.text(0.5, -0.04,
             f"$g_h \\times g_w = {GRID_H} \\times {GRID_W} = {GRID_H*GRID_W}$ patches "
             "per frame  (matches Qwen2.5-VL ViT)",
             transform=axA.transAxes, ha="center", va="top",
             fontsize=11, color="#444")
    axA.axis("off")

    # ----- Panel B: per-patch activation heatmap (averaged over dims) ---
    axB = fig.add_subplot(gs[0, 1])
    cmap = plt.get_cmap("RdBu_r")
    vmax = float(np.percentile(np.abs(A_avg), 99))
    im = axB.imshow(A_avg, cmap=cmap, vmin=-vmax, vmax=vmax,
                    interpolation="nearest")
    axB.set_title("(B)  per-patch activations  "
                  r"$\bar h_{p,t} \;=\; \frac{1}{d}\sum_{k=1}^{d} h_{p,t}^{(k)}$",
                  fontsize=14, weight="bold", pad=8)
    axB.set_xticks([]); axB.set_yticks([])
    for s in axB.spines.values():
        s.set_color("#888888")
    axB.text(0.5, -0.04,
             f"each cell = one ViT patch;  color = activation averaged over "
             f"$d={N_DIMS}$ dims (synthetic)",
             transform=axB.transAxes, ha="center", va="top",
             fontsize=11, color="#444")
    cb = fig.colorbar(im, ax=axB, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=9)
    cb.set_label("avg. activation", fontsize=10)

    fig.suptitle(
        "§4.1   Per-patch residual-stream activations  "
        r"$h_{p,t}$  at layer $\ell$   (illustrative; dims averaged)",
        fontsize=16, weight="bold", y=0.985,
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    fig.savefig(OUT_PDF,           bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
