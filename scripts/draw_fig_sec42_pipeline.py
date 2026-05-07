#!/usr/bin/env python
"""§4.2 illustrative figure built from REAL data.

Pipeline:
  Linear-representation hypothesis:
      h^(s,o) = u_id(o) + u_sp(x_o^s) + eps^(s,o)

  STEP 1.  Cross-scene averaging cancels u_sp (positions are uniform random)
           and yields the identity prototype  hbar^(o) ~ u_id(o).

  STEP 2.  SVD of stacked prototypes yields the orthonormal identity basis
           W_l in R^{d x k}; the spatial-feature projector P_perp = I - W W^T
           removes the identity component.  PCA on h_tilde reveals the 3D
           spatial structure that was hidden under color/shape clustering.

All quantities use ACTUAL data:
  - per-object activations:   reports/probe_features/qwen_base.npz
                              (H: 201 x 3584, Qwen2.5-VL-7B L17)
  - 3 example scenes (BEV):   data/tier_c_free6dof_f32/<scene>/bev.png
  - cross-scene averaging:    computed in this script
  - basis W:                  computed via SVD on H_bar
  - PCA scatters:             computed in this script (before vs after P_perp)

Outputs: figures/fig_sec42_pipeline.{png,pdf}
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

H_NPZ      = Path("reports/probe_features/qwen_base.npz")        # baseline: fit W on this
H_DEMO_NPZ = Path("reports/probe_features/qwen_lam1_seed0.npz")  # demo:    apply P_perp to this
TIER_C     = Path("data/tier_c_free6dof_f32")
OUT_PNG    = Path("figures/fig_sec42_pipeline.png")
OUT_PDF    = Path("figures/fig_sec42_pipeline.pdf")

# Display hex codes for the 8 colors in the dataset
COLOR_HEX = {
    "yellow":  "#F4C430",
    "magenta": "#E040A0",
    "cyan":    "#3FC0D0",
    "green":   "#3FA040",
    "orange":  "#F08030",
    "blue":    "#3060D0",
    "red":     "#D04040",
    "purple":  "#8040C0",
}


def cross_scene_average(H, colors, shapes, scene_ids):
    """Compute identity prototypes by averaging activations across scenes."""
    keys = sorted(set(zip(colors.tolist(), shapes.tolist())))
    Hbar, key_list, n_per = [], [], []
    for k in keys:
        sel = (colors == k[0]) & (shapes == k[1])
        if sel.sum() < 3:   # need at least 3 scenes for stable mean
            continue
        Hbar.append(H[sel].mean(axis=0))
        key_list.append(k)
        n_per.append(int(sel.sum()))
    return np.stack(Hbar, axis=0), key_list, n_per


def pca2(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T, S


def main():
    npz = np.load(H_NPZ, allow_pickle=True)
    H        = npz["H"].astype(np.float32)
    coords   = npz["coords"].astype(np.float32)
    colors   = npz["colors"]
    shapes   = npz["shapes"]
    scene_ids = npz["scene_ids"]
    print(f"H = {H.shape}  ({len(np.unique(scene_ids))} scenes, "
          f"{len(np.unique(colors))} colors, {len(np.unique(shapes))} shapes)")

    # -------- STEP 1: cross-scene averaging --------
    Hbar, keys, n_per = cross_scene_average(H, colors, shapes, scene_ids)
    print(f"H_bar (identity prototypes): {Hbar.shape}  K={len(keys)}")
    for k, n in zip(keys, n_per):
        print(f"  prototype {k}: averaged over {n} scenes")

    # -------- STEP 2: SVD on prototypes -> identity basis W --------
    K, d = Hbar.shape
    Hbar_c = Hbar - Hbar.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Hbar_c, full_matrices=False)
    # paper uses k = (number of identities) - 1 (after centering); here we set
    # k = number of independent prototype directions retained (rank-K-1).
    k_keep = min(K - 1, len(S))
    W = Vt[:k_keep].T              # (d, k)
    print(f"W shape = {W.shape}  (k_keep={k_keep})  singular values: {S[:6].round(2)}")

    # -------- spatial-feature projection on the full H --------
    P_perp = np.eye(d, dtype=np.float32) - W @ W.T

    # Demo H: load a model that DOES encode spatial structure (post-Dirichlet)
    # so the projector's effect is visible.  W is still fit on the baseline.
    demo = np.load(H_DEMO_NPZ, allow_pickle=True)
    H_demo = demo["H"].astype(np.float32)
    # Sanity-check the rows align (same probe features across checkpoints):
    assert (demo["scene_ids"] == scene_ids).all()
    assert (demo["colors"]    == colors).all()
    assert (demo["shapes"]    == shapes).all()

    H_tilde = H_demo @ P_perp.T
    print(f"H_tilde = {H_tilde.shape}  (demo: {H_DEMO_NPZ.name})")

    # -------- BEFORE/AFTER PCA scatters --------
    # BEFORE: full-population PCA on raw demo-H — should still show identity
    #         clusters because Dirichlet doesn't strip identity completely.
    Y_before, _ = pca2(H_demo)

    # AFTER: within-scene PCA on H_tilde — subtracting the scene centroid
    # isolates the spatial-variation component the projector exposes.
    H_resid = H_tilde.copy()
    for sid in np.unique(scene_ids):
        m = scene_ids == sid
        H_resid[m] = H_tilde[m] - H_tilde[m].mean(axis=0, keepdims=True)
    Y_after, _ = pca2(H_resid)

    # 3D-position color map for the AFTER scatter: project x,y onto angle
    # so each object gets a 2D coord -> hue.  Use the (x,y) location as a
    # 2D color where x maps to red-blue and y maps to green-purple.
    x = coords[:, 0]
    y = coords[:, 1]

    # ============== figure ==============
    fig = plt.figure(figsize=(20.0, 12.0), dpi=130)
    gs = GridSpec(
        3, 4, figure=fig,
        height_ratios=[0.16, 0.95, 1.10],
        width_ratios=[1.0, 1.0, 1.0, 1.0],
        hspace=0.45, wspace=0.30,
        left=0.04, right=0.985, top=0.94, bottom=0.05,
    )

    # ----- LRH banner (row 0, all columns) -----
    axLRH = fig.add_subplot(gs[0, :])
    axLRH.axis("off")
    axLRH.text(
        0.5, 0.70,
        r"$h^{(s,o)}_{\ell} \;=\; u_{\mathrm{id}}(o) \;+\; "
        r"u_{\mathrm{sp}}\!\left(x_o^{(s)}\right) \;+\; \varepsilon^{(s,o)}$",
        transform=axLRH.transAxes, ha="center", va="center", fontsize=22,
    )
    # annotation labels under the equation terms
    axLRH.text(0.32, 0.30, "identity\n(color, shape)",
               transform=axLRH.transAxes, ha="center", va="top",
               fontsize=11, color="#A04040", style="italic")
    axLRH.text(0.49, 0.30, "spatial\n(3D position)",
               transform=axLRH.transAxes, ha="center", va="top",
               fontsize=11, color="#3060D0", style="italic")
    axLRH.text(0.64, 0.30, "noise",
               transform=axLRH.transAxes, ha="center", va="top",
               fontsize=11, color="#666666", style="italic")
    axLRH.text(
        0.5, 0.04,
        "Linear-representation hypothesis (LRH)",
        transform=axLRH.transAxes, ha="center", va="center",
        fontsize=13, color="#666666", style="italic",
    )

    # ============================================================
    # STEP 1: cross-scene averaging
    # ============================================================
    # Choose one identity (color, shape) that has >=3 scenes; pick (red, cube)
    # if available else the first available
    pick = None
    for prefer in [("red", "cube"), ("blue", "cube"), ("yellow", "cube"),
                   ("magenta", "cylinder")]:
        sel = (colors == prefer[0]) & (shapes == prefer[1])
        if sel.sum() >= 3:
            pick = prefer
            break
    if pick is None:
        pick = keys[0]
    sel_pick = (colors == pick[0]) & (shapes == pick[1])
    sids_with_pick = list(np.unique(scene_ids[sel_pick]))[:3]
    print(f"Showing identity {pick}: scenes {sids_with_pick}")

    # row 1, cols 0..2: 3 BEV scenes for that identity
    for i, sid in enumerate(sids_with_pick):
        ax = fig.add_subplot(gs[1, i])
        bev_path = TIER_C / sid / "bev.png"
        if bev_path.exists():
            ax.imshow(np.array(Image.open(bev_path)))
        else:
            ax.text(0.5, 0.5, "(no BEV)", transform=ax.transAxes,
                    ha="center", va="center", color="#888")
        ax.axis("off")
        # find coord of the picked identity in this scene
        sel_in_sc = (scene_ids == sid) & sel_pick
        if sel_in_sc.sum() >= 1:
            xy = coords[sel_in_sc][0]
            label_pos = f"$x={xy[0]:+.2f},\\ y={xy[1]:+.2f}$"
        else:
            label_pos = ""
        ax.set_title(
            f"scene  {sid}",
            fontsize=12, weight="bold", pad=6,
        )
        ax.text(0.5, -0.05,
                f"{pick[0]} {pick[1]}  at  {label_pos}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=11, color=COLOR_HEX.get(pick[0], "#333"))

    # row 1, col 3: cross-scene averaging formula
    axAvg = fig.add_subplot(gs[1, 3])
    axAvg.axis("off")
    axAvg.set_title("STEP 1.  cross-scene averaging  →  identity prototype",
                    fontsize=14, weight="bold", pad=8)
    axAvg.text(
        0.5, 0.78,
        r"$\bar h^{(o)}_{\ell} \;=\; "
        r"\dfrac{1}{|\mathcal{S}_o|}\sum_{s \in \mathcal{S}_o} h^{(s,o)}_{\ell}$",
        transform=axAvg.transAxes, ha="center", va="center", fontsize=20,
    )
    axAvg.text(
        0.5, 0.50,
        f"$\\quad\\to\\; u_{{\\mathrm{{id}}}}(o)$",
        transform=axAvg.transAxes, ha="center", va="center", fontsize=18,
    )
    axAvg.text(
        0.5, 0.34,
        "positions $x_o^{(s)}$ are uniformly random\n"
        "$\\Rightarrow$ $u_{\\mathrm{sp}}$ averages to 0,\n"
        "identity prototype remains.",
        transform=axAvg.transAxes, ha="center", va="center",
        fontsize=12, color="#333333",
    )
    # show the prototype as a vector strip from REAL data
    Hbar_idx = keys.index(pick) if pick in keys else 0
    proto_vec = Hbar[Hbar_idx, :220]
    vmax_p = float(np.abs(proto_vec).max())
    axAvg.imshow(
        proto_vec[None, :], aspect="auto",
        extent=(0.06, 0.94, 0.06, 0.18),
        transform=axAvg.transAxes, cmap="RdBu_r",
        vmin=-vmax_p, vmax=vmax_p,
    )
    axAvg.text(0.50, 0.225,
               f"$\\bar h^{{(o)}}_{{\\ell}} \\in \\mathbb{{R}}^{{d}}$  (real, "
               f"avg of {n_per[Hbar_idx]} scenes; first 220 dims shown)",
               transform=axAvg.transAxes, ha="center", va="center",
               fontsize=11, color="#333")

    # ============================================================
    # STEP 2: SVD -> W -> P_perp -> PCA before/after
    # ============================================================
    # row 2, col 0: SVD spectrum + W panel
    axSVD = fig.add_subplot(gs[2, 0])
    axSVD.axis("off")
    axSVD.set_title("STEP 2.  identity-attribute basis  $W$",
                    fontsize=14, weight="bold", pad=8)
    # singular value bars
    axSVD.text(
        0.5, 0.94,
        f"SVD($\\bar H_{{\\ell}}^{{T}}$)  =  $U \\Sigma V^{{T}}$",
        transform=axSVD.transAxes, ha="center", va="center",
        fontsize=14,
    )
    # mini bar of singular values
    sv = S / S.max()
    bar_x = np.linspace(0.1, 0.9, len(sv))
    bar_w = (0.9 - 0.1) / (len(sv) + 1)
    for i, v in enumerate(sv):
        keep = i < k_keep
        axSVD.add_patch(mpatches.Rectangle(
            (bar_x[i] - bar_w/2, 0.55), bar_w, 0.30 * v,
            facecolor="#3060D0" if keep else "#CCCCCC",
            edgecolor="#222222", linewidth=0.6,
            transform=axSVD.transAxes,
        ))
    axSVD.text(0.50, 0.50, f"singular values  (keep top {k_keep} → $W \\in \\mathbb{{R}}^{{d \\times k}}$)",
               transform=axSVD.transAxes, ha="center", va="center", fontsize=11,
               color="#333")
    # W matrix as a heatmap
    Wshow = W[:200, :].copy()
    vw = float(np.percentile(np.abs(Wshow), 99))
    axSVD.imshow(Wshow, aspect="auto",
                 extent=(0.30, 0.70, 0.07, 0.40),
                 transform=axSVD.transAxes, cmap="RdBu_r",
                 vmin=-vw, vmax=vw)
    axSVD.text(0.50, 0.04, f"$W$  (first 200 of $d={d}$ rows  ×  $k={W.shape[1]}$ cols)",
               transform=axSVD.transAxes, ha="center", va="center",
               fontsize=11, color="#333")

    # row 2, col 1: P_perp formula + extracted activation
    axPP = fig.add_subplot(gs[2, 1])
    axPP.axis("off")
    axPP.set_title("spatial-feature projector",
                   fontsize=14, weight="bold", pad=8)
    axPP.text(
        0.5, 0.78,
        r"$P_{\perp} \;=\; I_d - W\,W^{T}$",
        transform=axPP.transAxes, ha="center", va="center", fontsize=20,
    )
    axPP.text(
        0.5, 0.55,
        r"$\tilde h^{(s,o)}_{\ell} \;=\; P_{\perp}\, h^{(s,o)}_{\ell}"
        r"\;\approx\; u_{\mathrm{sp}}(x_o^{(s)})$",
        transform=axPP.transAxes, ha="center", va="center", fontsize=16,
    )
    axPP.text(
        0.5, 0.36,
        "removes the identity component\n"
        "→ the spatial code is now isolated",
        transform=axPP.transAxes, ha="center", va="center",
        fontsize=12, color="#333",
    )
    # show one real h vs h_tilde strip
    idx_demo = int(np.where(sel_pick)[0][0])
    h_orig    = H_demo[idx_demo, :220]
    h_extract = H_tilde[idx_demo, :220]
    vm = max(np.abs(h_orig).max(), np.abs(h_extract).max())
    axPP.imshow(h_orig[None, :], aspect="auto",
                extent=(0.06, 0.94, 0.20, 0.27),
                transform=axPP.transAxes, cmap="RdBu_r", vmin=-vm, vmax=vm)
    axPP.text(0.50, 0.18,  r"$h^{(s,o)}_{\ell}$",
              transform=axPP.transAxes, ha="center", va="top", fontsize=11)
    axPP.imshow(h_extract[None, :], aspect="auto",
                extent=(0.06, 0.94, 0.06, 0.13),
                transform=axPP.transAxes, cmap="RdBu_r", vmin=-vm, vmax=vm)
    axPP.text(0.50, 0.04, r"$\tilde h^{(s,o)}_{\ell} = P_{\perp}\,h^{(s,o)}_{\ell}$",
              transform=axPP.transAxes, ha="center", va="top", fontsize=11)

    # row 2, col 2: BEFORE PCA scatter (colored by IDENTITY/color)
    axB = fig.add_subplot(gs[2, 2])
    pt_colors = np.array([COLOR_HEX.get(c, "#888") for c in colors])
    shape_marker = {"cube": "s", "cylinder": "o", "sphere": "^"}
    for shp in np.unique(shapes):
        mk = shape_marker.get(shp, "o")
        sel = shapes == shp
        axB.scatter(Y_before[sel, 0], Y_before[sel, 1],
                    c=pt_colors[sel], marker=mk, s=46,
                    edgecolors="#202020", linewidths=0.5, alpha=0.92)
    axB.set_title("BEFORE:  PCA on $h^{(s,o)}_{\\ell}$ (all scenes)\n"
                  "— clustered by identity (shape / color)",
                  fontsize=13, weight="bold", pad=6)
    axB.set_xlabel("PC 1", fontsize=11)
    axB.set_ylabel("PC 2", fontsize=11)
    axB.tick_params(labelsize=9)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)

    # row 2, col 3: AFTER — within-scene 2D PCA on H_tilde, plotted for the
    # scene with the most objects, with a side-by-side ground-truth layout.
    axA = fig.add_subplot(gs[2, 3])
    # pick the scene with the most objects
    sid_counts = {sid: int((scene_ids == sid).sum())
                  for sid in np.unique(scene_ids)}
    best_sid = max(sid_counts, key=sid_counts.get)
    sel_best = scene_ids == best_sid
    Hts = H_tilde[sel_best]
    cs  = colors[sel_best]
    shs = shapes[sel_best]
    xy  = coords[sel_best, :2]
    # within-scene PCA
    Y_one, _ = pca2(Hts)
    # match orientation of PC plot to physical (x, y) by Procrustes-style flip
    # (just a visual nicety: flip axes if it improves the alignment)
    def best_flip(A, B):
        best = (1, 1, np.inf)
        for sx in (-1, 1):
            for sy in (-1, 1):
                Ap = A * np.array([sx, sy])
                # normalize both
                Ap = Ap - Ap.mean(0); Ap /= np.linalg.norm(Ap) + 1e-9
                Bn = B - B.mean(0); Bn /= np.linalg.norm(Bn) + 1e-9
                err = np.linalg.norm(Ap - Bn)
                if err < best[2]:
                    best = (sx, sy, err)
        return best[0], best[1]
    sx, sy = best_flip(Y_one, xy)
    Y_one[:, 0] *= sx
    Y_one[:, 1] *= sy

    # plot
    for i in range(len(cs)):
        mk = shape_marker.get(shs[i], "o")
        col = COLOR_HEX.get(cs[i], "#888")
        axA.scatter(Y_one[i, 0], Y_one[i, 1],
                    c=col, marker=mk, s=180,
                    edgecolors="#202020", linewidths=0.9)
        axA.annotate(
            f"{cs[i][0]}{shs[i][0]}",
            (Y_one[i, 0], Y_one[i, 1]),
            xytext=(8, 6), textcoords="offset points",
            fontsize=10, color="#101010",
        )
    axA.set_title("AFTER:  within-scene PCA on  $\\tilde h^{(s,o)}_{\\ell}$\n"
                  f"— recovers 3D layout (scene {best_sid}, m={Hts.shape[0]})",
                  fontsize=13, weight="bold", pad=6)
    axA.set_xlabel("PC 1", fontsize=11)
    axA.set_ylabel("PC 2", fontsize=11)
    axA.tick_params(labelsize=9)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    # show the ground-truth physical layout as a faint inset
    inset = axA.inset_axes([0.04, 0.04, 0.32, 0.30])
    for i in range(len(cs)):
        mk = shape_marker.get(shs[i], "o")
        col = COLOR_HEX.get(cs[i], "#888")
        inset.scatter(xy[i, 0], xy[i, 1], c=col, marker=mk, s=46,
                      edgecolors="#202020", linewidths=0.6)
    inset.set_title("GT 3D (x, y)", fontsize=8.5, pad=2)
    inset.set_xticks([]); inset.set_yticks([])
    inset.set_aspect("equal")
    inset.spines["top"].set_color("#888888")
    inset.spines["right"].set_color("#888888")
    inset.spines["left"].set_color("#888888")
    inset.spines["bottom"].set_color("#888888")

    # mini legends
    # color legend for (B)
    leg_handles = [mpatches.Patch(color=COLOR_HEX[c], label=c)
                   for c in ["red", "blue", "yellow", "green", "magenta",
                             "cyan", "orange", "purple"]
                   if c in np.unique(colors)]
    axB.legend(handles=leg_handles, fontsize=8, loc="upper right",
               frameon=False, ncol=2, handlelength=1.0, columnspacing=0.6,
               labelspacing=0.2)
    axA.text(0.02, 0.97,
             "labels: shape×color  (e.g. bs = blue sphere)",
             transform=axA.transAxes, va="top", ha="left",
             fontsize=9.5, color="#333", style="italic")

    fig.suptitle(
        "§4.2   Linear spatial feature extraction via cross-scene averaging",
        fontsize=19, weight="bold", y=0.985, ha="center",
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight")
    fig.savefig(OUT_PDF,           bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
