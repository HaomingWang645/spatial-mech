#!/usr/bin/env python
"""§4.1 supplementary figure: per-object residual-stream activations as
an m x 1 heatmap.

For visualization only -- synthetic activation values are drawn for each
object in the scene to make the per-object structure visible.  The
purpose is to show:

    "for every object o in scene s, the layer-l residual stream
     produces a vector h^{(s,o)}_{l} of d dims; averaging over d
     gives one scalar per object, which we display as an m x 1
     heatmap."

Layout:
  Left:  one rendered frame with every object's GT mask outlined and
         labeled o_0, o_1, ... o_{m-1}.
  Right: an m x 1 heatmap; each cell = average activation magnitude
         (over d dims) for that object.

Outputs: figures/fig_sec41_object_activations.{png,pdf}
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
OUT_PNG   = Path("figures/fig_sec41_object_activations.png")
OUT_PDF   = Path("figures/fig_sec41_object_activations.pdf")

N_DIMS = 64   # number of synthetic dims to average over

# Distinct outline colors for object masks (cycled if m > len)
OUTLINE_COLORS = [
    "#FFEC3D", "#FF6B6B", "#4DABF7", "#69DB7C", "#F783AC",
    "#FFB347", "#9775FA", "#63E6BE", "#FFD43B", "#FF8787",
]


def synthetic_object_activations(m, n_dims, seed=0):
    """Generate (m, n_dims) synthetic per-object activation vectors.

    Each object's vector is a mix of a shared base direction plus a
    per-object signature, scaled by a random magnitude.  Averaging
    across dims gives a per-object scalar that varies cleanly across
    the m objects."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size=(n_dims,))
    H = np.zeros((m, n_dims), dtype=np.float32)
    for i in range(m):
        sig = rng.normal(0.0, 1.0, size=(n_dims,))
        amp = rng.uniform(0.4, 1.6)
        bias = rng.uniform(-0.6, 0.6)
        H[i] = amp * (0.6 * base + sig) + bias
    H = (H - H.mean(axis=1, keepdims=True)) / (
        H.std(axis=1, keepdims=True) + 1e-6)
    # add a small per-object DC offset so the dim-mean varies
    H = H + rng.uniform(-0.8, 0.8, size=(m, 1)).astype(np.float32)
    return H


def main():
    sc = json.load(open(SCENE_DIR / "scene.json"))
    objects = sc["objects"]
    m = len(objects)

    # Pick a representative frame: prefer one where most objects are visible.
    best_fr, best_n = 0, -1
    for fr in sc["frames"]:
        msk = np.array(Image.open(SCENE_DIR / fr["mask_path"]))
        n_vis = sum(int((msk == o["object_id"] + 1).sum() > 0) for o in objects)
        if n_vis > best_n:
            best_n, best_fr = n_vis, fr["frame_id"]
    fr_id = best_fr
    img  = np.array(Image.open(SCENE_DIR / sc["frames"][fr_id]["image_path"]))
    mask = np.array(Image.open(SCENE_DIR / sc["frames"][fr_id]["mask_path"]))
    print(f"Picked frame {fr_id}: {best_n}/{m} objects visible")

    # Synthesize per-object activations, then average over dims -> m x 1 col.
    H_obj = synthetic_object_activations(m, N_DIMS, seed=3)
    a_obj = H_obj.mean(axis=1)                # (m,)
    print(f"Per-object activation vector: shape={a_obj.shape} "
          f"(averaged over {N_DIMS} dims)")

    # ============== figure ==============
    fig = plt.figure(figsize=(13.0, 7.4), dpi=140)
    gs = GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.0, 0.18],
        hspace=0.28, left=0.07, right=0.93,
        top=0.86, bottom=0.10,
    )

    # ----- Panel A: rendered frame + per-object mask outlines + labels -----
    axA = fig.add_subplot(gs[0, 0])
    # square the frame inside its row
    axA.set_aspect("equal")
    axA.imshow(img)
    for i, o in enumerate(objects):
        obj_mask = (mask == o["object_id"] + 1).astype(np.uint8)
        if obj_mask.sum() == 0:
            continue
        col = OUTLINE_COLORS[i % len(OUTLINE_COLORS)]
        axA.contour(obj_mask, levels=[0.5], colors=[col], linewidths=2.2)
        ys, xs = np.where(obj_mask > 0)
        cy, cx = float(ys.mean()), float(xs.mean())
        axA.text(
            cx, cy, f"o$_{{{i}}}$",
            ha="center", va="center", fontsize=13, weight="bold",
            color="black",
            bbox=dict(facecolor=col, edgecolor="black", linewidth=0.8,
                      boxstyle="round,pad=0.18", alpha=0.92),
        )
    axA.set_title(f"(A)  one frame  +  per-object GT masks  (m = {m})",
                  fontsize=14, weight="bold", pad=8)
    axA.axis("off")

    # ----- Panel B: per-object activation heatmap (1 x m, dims averaged) ---
    axB = fig.add_subplot(gs[1, 0])
    cmap = plt.get_cmap("RdBu_r")
    vmax = float(np.percentile(np.abs(a_obj), 99))
    if vmax < 1e-6:
        vmax = float(np.abs(a_obj).max() + 1e-6)
    axB.imshow(
        a_obj[None, :], cmap=cmap, vmin=-vmax, vmax=vmax,
        aspect="auto", interpolation="nearest",
    )
    axB.set_yticks([])
    axB.set_xticks(range(m))
    axB.set_xticklabels([f"o$_{{{i}}}$" for i in range(m)], fontsize=12)
    axB.tick_params(axis="x", length=0, pad=6)
    # outline each cell in its object's panel-A color so the link to (A)
    # is unambiguous.
    for i in range(m):
        col = OUTLINE_COLORS[i % len(OUTLINE_COLORS)]
        rect = mpatches.Rectangle(
            (i - 0.5, -0.5), 1.0, 1.0,
            facecolor="none", edgecolor=col, linewidth=2.2,
        )
        axB.add_patch(rect)
    for s in axB.spines.values():
        s.set_color("#888888")
    axB.set_title("(B)  per-object activations  "
                  r"$\bar h^{(s,o)}_{\ell} = "
                  r"\frac{1}{d}\sum_{k=1}^{d}\, h^{(s,o)\,(k)}_{\ell}$"
                  f"   (dim-averaged, $d={N_DIMS}$, synthetic)",
                  fontsize=13, weight="bold", pad=8)

    fig.suptitle(
        "§4.1   Per-object residual-stream activations  "
        r"$h^{(s,o)}_{\ell}$  at layer $\ell$   (illustrative; dims averaged)",
        fontsize=16, weight="bold", y=0.97,
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    fig.savefig(OUT_PDF,           bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
