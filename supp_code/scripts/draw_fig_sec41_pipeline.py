#!/usr/bin/env python
"""§4.1 illustrative figure built from REAL data.

Pipeline:
  3D scene + camera trajectory + rendered frames
  -> vision encoder patch grid + GT object mask
  -> per-patch coverage c_{o,p,t}, kappa-filter, weighted pool
  -> per-frame object token h_{l,t}^{(s,o)}
  -> temporal mean -> per-object representation h_l^{(s,o)}
  -> stack m objects -> per-scene activation matrix H_l^{(s)}

All quantities use ACTUAL data:
  - rendered frame & mask:  data/tier_c_free6dof_f32/<scene>/{frames,masks}/
  - per-patch coverage:     computed from the real mask at the model's
                            14x14 patch size (448/14 = 32x32 grid)
  - stacked H matrix:       reports/probe_features/qwen_base.npz (Qwen2.5-VL-7B L17)

Outputs: figures/fig_sec41_pipeline.{png,pdf}
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

SCENE_ID  = "s_002abd79ae_t1"
SCENE_DIR = Path(f"data/tier_c_free6dof_f32/{SCENE_ID}")
H_NPZ     = Path("reports/probe_features/qwen_base.npz")
OUT_PNG   = Path("figures/fig_sec41_pipeline.png")
OUT_PDF   = Path("figures/fig_sec41_pipeline.pdf")

PATCH_SIZE = 14   # Qwen2.5-VL ViT
KAPPA      = 0.30


def coverage_map(mask: np.ndarray, target_id_plus1: int, patch: int) -> np.ndarray:
    """Per-patch coverage c_{o,p} = |patch_p ∩ mask_o| / |patch_p|."""
    H, W = mask.shape
    gh, gw = H // patch, W // patch
    obj = (mask == target_id_plus1).astype(np.float32)
    return obj.reshape(gh, patch, gw, patch).mean(axis=(1, 3))


def best_frame_object(sc):
    """Find (frame_id, object_id) with the largest visible target."""
    best = (0, 0, -1)
    for fr in sc["frames"]:
        m = np.array(Image.open(SCENE_DIR / fr["mask_path"]))
        for o in sc["objects"]:
            c = coverage_map(m, o["object_id"] + 1, PATCH_SIZE)
            n = int((c >= KAPPA).sum())
            if 18 <= n <= 90 and n > best[2]:
                best = (fr["frame_id"], o["object_id"], n)
    if best[2] < 0:
        # fallback: max over anything
        for fr in sc["frames"]:
            m = np.array(Image.open(SCENE_DIR / fr["mask_path"]))
            for o in sc["objects"]:
                c = coverage_map(m, o["object_id"] + 1, PATCH_SIZE)
                n = int((c >= KAPPA).sum())
                if n > best[2]:
                    best = (fr["frame_id"], o["object_id"], n)
    return best


def main():
    sc = json.load(open(SCENE_DIR / "scene.json"))

    fr_id, obj_id, n_above = best_frame_object(sc)
    target = next(o for o in sc["objects"] if o["object_id"] == obj_id)
    print(f"Picked frame {fr_id}, object {obj_id} ({target['color']} {target['shape']}), "
          f"{n_above} patches above kappa={KAPPA}")

    img = np.array(Image.open(SCENE_DIR / sc["frames"][fr_id]["image_path"]))
    mask = np.array(Image.open(SCENE_DIR / sc["frames"][fr_id]["mask_path"]))
    cov = coverage_map(mask, obj_id + 1, PATCH_SIZE)
    Hpx, Wpx = mask.shape
    gh, gw = Hpx // PATCH_SIZE, Wpx // PATCH_SIZE

    # Compute zoom bbox around the target's footprint (with margin)
    rows_with_cov = np.where(cov.sum(1) > 0)[0]
    cols_with_cov = np.where(cov.sum(0) > 0)[0]
    if len(rows_with_cov) and len(cols_with_cov):
        r0, r1 = rows_with_cov.min(), rows_with_cov.max()
        c0, c1 = cols_with_cov.min(), cols_with_cov.max()
    else:
        r0, r1, c0, c1 = 0, gh - 1, 0, gw - 1
    pad = 2
    r0 = max(r0 - pad, 0); r1 = min(r1 + pad, gh - 1)
    c0 = max(c0 - pad, 0); c1 = min(c1 + pad, gw - 1)

    # Stacked-H matrix
    Hnpz = np.load(H_NPZ, allow_pickle=True)
    H_all  = Hnpz["H"]
    sids   = Hnpz["scene_ids"]
    colors = Hnpz["colors"]
    shapes = Hnpz["shapes"]
    sid = SCENE_ID if SCENE_ID in sids else np.unique(sids)[0]
    sel = sids == sid
    H_scene    = H_all[sel]
    obj_colors = colors[sel]
    obj_shapes = shapes[sel]
    print(f"H_scene: scene={sid} m={H_scene.shape[0]} d={H_scene.shape[1]}")

    # ============== figure ==============
    fig = plt.figure(figsize=(20.0, 11.5), dpi=130)
    gs = GridSpec(
        2, 4, figure=fig,
        height_ratios=[1.05, 0.95],
        width_ratios=[1.0, 1.05, 1.05, 1.05],
        hspace=0.42, wspace=0.30,
        left=0.035, right=0.985, top=0.93, bottom=0.06,
    )

    # ----- Panel A: BEV / scene ----------------------------------------
    axA = fig.add_subplot(gs[0, 0])
    bev = np.array(Image.open(SCENE_DIR / "bev.png"))
    axA.imshow(bev)
    axA.set_title("(A)  3D scene + camera trajectory",
                  fontsize=15, weight="bold", pad=8)
    axA.text(0.5, -0.06,
             f"BEV; T = {len(sc['frames'])} frames rendered along trajectory\n"
             f"scene  {sid}  ({len(sc['objects'])} objects)",
             transform=axA.transAxes, ha="center", va="top",
             fontsize=11, color="#333333")
    axA.axis("off")

    # ----- Panel B: rendered frame + patch grid + GT mask --------------
    axB = fig.add_subplot(gs[0, 1])
    axB.imshow(img)
    for k in range(gw + 1):
        axB.axvline(k * PATCH_SIZE - 0.5, color="white", linewidth=0.3, alpha=0.55)
    for k in range(gh + 1):
        axB.axhline(k * PATCH_SIZE - 0.5, color="white", linewidth=0.3, alpha=0.55)
    obj_mask = (mask == obj_id + 1).astype(np.uint8)
    axB.contour(obj_mask, levels=[0.5], colors=["#FFEC3D"], linewidths=2.4)
    # show zoom rectangle
    rect = mpatches.Rectangle(
        (c0 * PATCH_SIZE - 0.5, r0 * PATCH_SIZE - 0.5),
        (c1 - c0 + 1) * PATCH_SIZE, (r1 - r0 + 1) * PATCH_SIZE,
        facecolor="none", edgecolor="#FF3030", linewidth=2.0, linestyle="--",
    )
    axB.add_patch(rect)
    axB.set_title(f"(B)  rendered frame  +  ViT patch grid ({gh}×{gw})",
                  fontsize=15, weight="bold", pad=8)
    axB.text(0.5, -0.06,
             f"target: {target['color']} {target['shape']}  "
             f"(GT mask outlined yellow);  red box → zoom in (C)",
             transform=axB.transAxes, ha="center", va="top",
             fontsize=11, color="#333333")
    axB.axis("off")

    # ----- Panel C: zoomed coverage map with kappa outline + values ----
    axC = fig.add_subplot(gs[0, 2])
    cov_zoom = cov[r0:r1+1, c0:c1+1]
    im = axC.imshow(cov_zoom, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    nh, nw = cov_zoom.shape
    above_zoom = cov_zoom >= KAPPA
    for r in range(nh):
        for c in range(nw):
            if above_zoom[r, c]:
                rect2 = mpatches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor="none", edgecolor="#FF3030", linewidth=2.0,
                )
                axC.add_patch(rect2)
            v = cov_zoom[r, c]
            if v > 0.02:
                axC.text(c, r, f"{v:.2f}", ha="center", va="center",
                         fontsize=8.5,
                         color="white" if v < 0.55 else "black")
    axC.set_title(f"(C)  per-patch coverage  $c_{{o,p,t}}$  (zoom on target)",
                  fontsize=15, weight="bold", pad=8)
    axC.text(0.5, -0.06,
             f"red outline: $c \\geq \\kappa = {KAPPA}$  "
             f"({n_above} patches kept of {gh*gw} total)",
             transform=axC.transAxes, ha="center", va="top",
             fontsize=11, color="#333333")
    axC.set_xticks([]); axC.set_yticks([])
    cb = fig.colorbar(im, ax=axC, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=10)
    cb.set_label("coverage", fontsize=11)

    # ----- Panel D: pooling formula + per-frame token strip ------------
    axD = fig.add_subplot(gs[0, 3])
    axD.axis("off")
    axD.set_title("(D)  coverage-weighted pool",
                  fontsize=15, weight="bold", pad=8)
    axD.text(
        0.5, 0.78,
        r"$h^{(s,o)}_{\ell,t} \;=\;"
        r"\dfrac{\sum_{p\,:\,c_{o,p,t}\geq\kappa}\, c_{o,p,t}\, h_{p,t}}"
        r"{\sum_{p\,:\,c_{o,p,t}\geq\kappa}\, c_{o,p,t}}$",
        transform=axD.transAxes, ha="center", va="center", fontsize=20,
    )
    axD.text(
        0.5, 0.50,
        "$h_{p,t}$  =  layer-$\\ell$ residual stream\n"
        "                 at patch $p$, frame $t$",
        transform=axD.transAxes, ha="center", va="center",
        fontsize=12, color="#333333",
    )
    # mock per-frame token vector strip from real H
    obj_idx_in_H = list(zip(obj_colors, obj_shapes)).index(
        (target["color"], target["shape"])) if (target["color"], target["shape"]) in \
        list(zip(obj_colors, obj_shapes)) else 0
    vec = H_scene[obj_idx_in_H, :180]
    axD.imshow(
        vec[None, :], aspect="auto",
        extent=(0.06, 0.94, 0.10, 0.24),
        transform=axD.transAxes, cmap="RdBu_r",
        vmin=-np.abs(vec).max(), vmax=np.abs(vec).max(),
    )
    axD.text(0.50, 0.32,
             f"$h^{{(s,o)}}_{{\\ell,t}} \\in \\mathbb{{R}}^{{d}}$  (d={H_scene.shape[1]})",
             transform=axD.transAxes, ha="center", va="center", fontsize=12)

    # ----- Panel E: per-frame object tokens (4 thumbnails) -------------
    axE = fig.add_subplot(gs[1, 0])
    axE.axis("off")
    axE.set_title("(E)  per-frame object tokens",
                  fontsize=15, weight="bold", pad=8)
    n_show = 4
    sel_fr = np.linspace(0, len(sc["frames"]) - 1, n_show).astype(int)
    sub_w = 0.215; sub_h = 0.55
    for i, fid in enumerate(sel_fr):
        th = np.array(Image.open(SCENE_DIR / sc["frames"][fid]["image_path"]))
        x0 = 0.025 + i * (sub_w + 0.01)
        axE.imshow(th, extent=(x0, x0 + sub_w, 0.32, 0.32 + sub_h),
                   transform=axE.transAxes, aspect="auto")
        axE.text(x0 + sub_w / 2, 0.29, f"$t={fid+1}$",
                 transform=axE.transAxes, ha="center", va="top", fontsize=11)
    axE.text(0.5, 0.95,
             f"$T={len(sc['frames'])}$ frames per scene  →  $T/\\tau$ kept",
             transform=axE.transAxes, ha="center", va="top",
             fontsize=11, color="#333333")
    axE.text(0.5, 0.16,
             r"$\{ h^{(s,o)}_{\ell,t} \}_{t=1}^{T/\tau}$",
             transform=axE.transAxes, ha="center", va="center", fontsize=15)

    # ----- Panel F: temporal mean --------------------------------------
    axF = fig.add_subplot(gs[1, 1])
    axF.axis("off")
    axF.set_title("(F)  temporal mean", fontsize=15, weight="bold", pad=8)
    axF.text(0.5, 0.62,
             r"$h^{(s,o)}_{\ell} \;=\; \dfrac{1}{T/\tau}\sum_{t=1}^{T/\tau} h^{(s,o)}_{\ell,t}$",
             transform=axF.transAxes, ha="center", va="center", fontsize=20)
    axF.annotate("", xy=(0.92, 0.4), xytext=(0.08, 0.4),
                 xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#444444"))
    axF.text(0.5, 0.27,
             r"per-object representation  $h^{(s,o)}_{\ell} \in \mathbb{R}^{d}$",
             transform=axF.transAxes, ha="center", va="center",
             fontsize=12, color="#333333")

    # ----- Panel G: stacked H matrix as heatmap ------------------------
    axG = fig.add_subplot(gs[1, 2:])
    cap = 240
    Hsub = H_scene[:, :cap]
    vmax = float(np.percentile(np.abs(Hsub), 99))
    axG.imshow(Hsub, aspect="auto", cmap="RdBu_r",
               vmin=-vmax, vmax=vmax, interpolation="nearest")
    axG.set_yticks(range(H_scene.shape[0]))
    axG.set_yticklabels(
        [f"o$_{i}$  ({obj_colors[i]} {obj_shapes[i]})"
         for i in range(H_scene.shape[0])],
        fontsize=12,
    )
    axG.set_xlabel(f"residual-stream dim (showing first {cap} of d={H_scene.shape[1]})",
                   fontsize=12)
    axG.set_title(
        "(G)  stack $m$ objects  →  per-scene activation matrix  "
        f"$H^{{(s)}}_{{\\ell}} \\in \\mathbb{{R}}^{{m \\times d}}$  "
        f"(scene {sid}, m={H_scene.shape[0]})",
        fontsize=15, weight="bold", pad=8,
    )
    axG.tick_params(axis="x", labelsize=10)

    fig.suptitle(
        "§4.1   Per-object representation via mask-driven, coverage-weighted token pooling",
        fontsize=19, weight="bold", y=0.985, ha="center",
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight")
    fig.savefig(OUT_PDF,           bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
