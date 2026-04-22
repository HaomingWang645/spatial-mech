# Experiment plan: graph-topology tests for 3D spatial representations in VLMs

## 1. Motivation

Our linear-probe results (R² for normalized 3D coords) reach 0.92 on synthetic Tier C
but collapse to ≤0 on real Tier D video. The collapse could mean either:

- (a) VLMs genuinely lack a spatial code in real video, or
- (b) The code exists but is **nonlinear**, so a Ridge readout can't find it.

Linear probes cannot distinguish (a) from (b). Park et al. (ICLR 2025, "In-Context
Learning of Representations") give a toolkit that does — they test whether token
representations respect the **topology** of a ground-truth graph using
parameter-free metrics (PCA layout + Dirichlet energy). That is strictly more
permissive than a linear fit: it catches structure even when distances are warped
nonlinearly.

This plan ports their toolkit to our setting and tests three distinct design
choices. We implement **Option 3** (the most faithful analog of their setup)
first; Options 1 and 2 are specified for later.

## 2. The three experimental designs

| Option | Node = | Instances averaged | What averaging cancels | What survives (hypothesis) |
|---|---|---|---|---|
| **1** | World-frame 3D position bin `(i,j,k)` | All (scene, object) pairs landing in bin | Object identity, scene content, camera pose | Content-agnostic world-frame position code |
| **2** | Camera-pose bin (extrinsic xyz binned) | All frames across scenes whose camera falls in bin | Scene content, object identity | Ego-pose / localization code |
| **3** | (Object, scene) pair | The N_frames reps of that object in that scene | Viewpoint variation across frames | Per-scene object-instance code whose topology mirrors the scene's 3D layout |

### Mapping to the ICLR paper

The paper's nodes are **tokens** in one specific graph context; they average
across multiple random-walk visits of the same token within that fixed context.
Option 3 is the direct port: one scene = one graph; per-frame reps of an object
= the "visits"; average frames → one rep per object; check if those reps form
the scene's 3D layout.

Options 1 and 2 are stronger extensions (cross-scene averaging) and are
described in §6 but not implemented in this pass.

## 3. Option 3 in detail (implement first)

### 3.1 Data

**Re-render Tier C free6dof at multiple frame counts** so we can sweep the
context-length axis (analog of their Fig. 4 context-scaling curve):

- `N_frames ∈ {8, 16, 32, 64}` (must be even for Qwen's temporal patch merger
  of size 2, so the LM sees `{4, 8, 16, 32}` temporal chunks respectively).
- 200 base scenes × 2 trajectories each = 400 rendered scenes per frame count.
- Same object layouts (re-render over existing `data/scenes_3d/`), same
  rendering config, only the `n_frames` value changes.
- Output: `data/tier_c_free6dof_f{N}/`.

Storage: each frame is a 448×448 PNG (~150 KB) + mask. 64-frame run ≈
400 × 64 × 2 × 150 KB ≈ 8 GB. Feasible.

### 3.2 Extraction

Run the existing `scripts/extract_activations.py --mode video` on each
`(model, N_frames)` combination:

- **Models:** Qwen2.5-VL-7B, LLaVA-OV-7B, InternVL3-8B (three distinct
  architectures at a comparable 7–8B scale for a fair cross-family comparison).
- Qwen2.5-VL-32B optionally as a scale ablation.
- For each layer ℓ, the extractor writes `layer_LL.parquet` with
  `(scene_id, object_id, frame_id [= temporal chunk], layer, centroid_xyz)` and
  a matching `layer_LL.npy` of shape `(N_rows, D)`.

**Parallel launch**: one process per `(model, N_frames)` pinned to one GPU
via `CUDA_VISIBLE_DEVICES`. With 8 H100s we can run 8 jobs in parallel. Large
models (32B) use 2× GPUs via `device_map=auto`.

### 3.3 Per-scene aggregation

For each `(model, N_frames)` extraction and each layer ℓ:

```
for scene s in scenes:
    for object o in s.objects:
        h_{o,s}^ℓ = mean over temporal-chunks t of vec(o, s, t, ℓ)
    stack into H_s^ℓ  ∈ R^(n_s × D)   where n_s ∈ [3, 8]
    record  P_s     ∈ R^(n_s × 3)    true 3D centroids
```

### 3.4 Topology metrics (per scene, linearity-free)

Four metrics of increasing permissiveness. Each is computed per scene and
aggregated across scenes (mean, median, stderr, z-score vs null).

**(a) Pairwise-distance rank correlation (RSA / Spearman)**
```
D_H = pairwise distances of H_s^ℓ    (n_s choose 2)
D_P = pairwise distances of P_s       (n_s choose 2)
rho_s = spearman(D_H[triu], D_P[triu])
```
Tests monotonic distance relation — neighbors in 3D tend to be closer in rep space.

**(b) Dirichlet energy ratio vs. permutation null**
```
G_s = kNN-graph(P_s, k=2)
E_s = sum_{(i,j) in edges(G_s)} ||h_i - h_j||^2
E_null = mean over 200 permutations of (object -> position) labels, recompute E
ratio_s = E_s / E_null            # <1 means neighbors in G are closer than chance
```
Tests only local smoothness w.r.t. the scene graph.

**(c) k-NN overlap**
```
for object i:
    nn_P_i = top-k NN of i in P_s      (k=2)
    nn_H_i = top-k NN of i in H_s^ℓ
    overlap_i = |nn_P_i ∩ nn_H_i| / k
overlap_s = mean over objects
```
Pure topological preservation of local neighborhood.

**(d) Spectral-embedding cosine (Theorem 5.1 from the paper)**
```
L = Laplacian(G_s)
z2, z3 = eigenvectors for 2nd, 3rd smallest eigenvalues
PC1, PC2 = top PCA components of H_s^ℓ (after centering)
cos_s = (|cos(PC1, z2)| + |cos(PC2, z3)|) / 2
```
Tests whether the leading PCA axes of H match the graph's spectral embedding,
which is the specific linearity guaranteed by their Theorem 5.1 when reps
minimize Dirichlet energy.

### 3.5 Null models

For each metric, compute both:
- **Permutation null**: shuffle the object→position assignment within each scene
- **Random-rep null**: sample `H_s^ℓ` from Gaussian with matching mean/covariance

Report z-scores `(metric - null_mean) / null_std` — makes metrics comparable
across scene sizes.

### 3.6 Sweep axes

For each metric we get a 3-way grid:

| Axis | Values |
|---|---|
| Layer ℓ | all decoder layers of each model (~28 for 7B, ~32 for 32B) |
| N_frames | 8, 16, 32, 64 |
| Model | Qwen-7B, LLaVA-OV-7B, InternVL3-8B (+ optional Qwen-32B) |

This is `≈ 28 × 4 × 3 = 336` cells per metric per scene, aggregated over
200–400 scenes. Dominant cost is extraction, not analysis.

### 3.7 Visualizations (paper-figure analogs)

Produce the following PNGs in `figures/topology_option3/`:

1. **Per-scene PCA grid** (Fig. 9 analog): pick 4–6 representative scenes; for
   each, plot PCA-top-2 of object reps at layers {0, ℓ/4, ℓ/2, 3ℓ/4, ℓ-1},
   overlay the true 3D k-NN edges. Visually compare scene shape vs. rep shape.
   One grid per model.
2. **Layer curves** (Fig. 4 analog): `normalized Dirichlet energy` and `RSA`
   vs. `layer index`, one curve per `N_frames` value, with permutation-null
   band shaded. One panel per model, shared axes.
3. **Frame-count emergence curve**: for the best layer per model, plot metric
   vs. `N_frames` — does it improve with more frames, or saturate?
4. **Model comparison bar**: best metric value + layer-of-best across models,
   for N_frames=64.
5. **Spectral-cosine heatmap** (Table 2 analog): `|cos(PC_k, z^(k+1))|` as a
   `layers × k` heatmap per model.
6. **Scene examples with GT overlay** (Fig. 1/2 analog): 6 scenes at the best
   layer for Qwen-7B, showing true BEV layout side-by-side with PCA of reps.

### 3.8 Expected outcomes and interpretation matrix

What we'd conclude from each pattern across the four metrics:

| R² (prev) | RSA | Dirichlet | k-NN | Conclusion |
|---|---|---|---|---|
| ✓ (0.9) | ✓ | ✓ | ✓ | Tier C baseline: rep space affine-isometric to 3D |
| ✗ | ✓ | ✓ | ✓ | **Strong reframe**: signal is nonlinear but topologically preserved |
| ✗ | ✗ | ✓ | ✓ | Only local adjacency — distances globally scrambled |
| ✗ | ✗ | ✗ | ✓ | Only immediate-neighbor identity preserved |
| ✗ | ✗ | ✗ | ✗ | Genuinely no spatial structure in the reps |

If Tier C shows row 1 across all metrics, the pipeline is working. If Tier D
(future, not in this pass) shows row 2, our negative result from
`tier_d_arkitscenes_cam_motion` is reframed as "nonlinear topology preserved"
rather than "no spatial signal."

## 4. Compute plan

Available: 8× H100 (4× NVL 94GB + 4× PCIe 80GB).

**Rendering (CPU-bound):** 400 scenes × 64 frames × 448² ≈ 1 hour per
`N_frames` setting. Run sequentially: 4 settings × 1 h = 4 h. Fork across cores
with `multiprocessing`.

**Extraction (GPU-bound):** per (model, N_frames, 400 scenes):

| Model | GPU | Time/scene (64f) | Total per setting |
|---|---|---|---|
| Qwen-7B | 1 GPU | ~8 s | ~55 min |
| LLaVA-OV-7B | 1 GPU | ~10 s | ~70 min |
| InternVL3-8B | 1 GPU | ~12 s | ~80 min |
| Qwen-32B | 2 GPUs | ~25 s | ~170 min |

Launch 8 processes in parallel across 8 GPUs:
- GPUs 0, 2, 3: Qwen-7B at N_frames ∈ {8, 16, 32, 64}? No — one job per GPU.
- Strategy: for each model, sweep N_frames sequentially on one GPU; three GPUs
  cover three models in parallel. 32B gets 2 GPUs. Remaining GPUs reserved.

Wall clock: ~5 h for all extractions.

**Analysis (CPU-bound):** topology metrics over extracted reps, parallel per
(model, N_frames). ~15 min total.

**Visualization:** ~10 min.

Total wall clock from scratch: ~10 hours. If existing 16-frame extractions are
reused, cut to ~6 hours.

## 5. Deliverables

- `reports/experiment_plan_topology.md` (this file)
- `scripts/topology_option3.py` — metrics computation, reading existing
  activation parquets
- `scripts/visualize_topology.py` — figures
- `scripts/render_tier_c_long.sh` — orchestration wrapper for frame-count sweep
- `scripts/extract_topology_all_models.sh` — parallel launcher across GPUs
- `data/probes/topology_option3/{model}__f{N}/metrics.parquet`
- `figures/topology_option3/*.png`
- `reports/tier_c_topology_option3.md` — results write-up with tables + figure refs

## 6. Options 1 and 2 plans (not implemented)

### 6.1 Option 1 — World-frame position binning

**Setup**: discretize normalized world coords to a 4×4 BEV grid → 16 bins. For
each bin `(i, j)`, collect every (scene, object) whose centroid lands there
(~1875 instances per bin given ~30000 total object placements). Average those
reps at each layer ℓ:
```
h_bin(i,j)^ℓ = mean over all (s, o) with bin(o, s) = (i, j) of h_{o,s}^ℓ
```

**Analysis**: stack 16 bin reps → `H ∈ R^(16 × D)`. PCA top-2 should recover a
4×4 lattice. Dirichlet energy vs. 4-connected grid adjacency.

**What's different from Option 3**: averaging across *scenes* cancels content,
isolates a **scene-agnostic world-frame position code**. Strictly stronger
claim — if it holds, VLMs have a canonical position representation; if it
fails, they only encode relative spatial layout within each scene.

**Scripts to add**: `scripts/topology_option1_binning.py`, reusing the
extraction outputs.

### 6.2 Option 2 — Camera-pose binning

**Setup**: discretize camera extrinsics (xyz of the camera in world coords)
into a 4×4 BEV grid → 16 bins. For each bin, collect every frame across
scenes whose camera pose falls there. Per frame, use **mean-pooled visual
tokens** (not object-masked — we care about the frame-global rep, which
encodes ego-pose).
```
h_bin(i,j)^ℓ = mean over all frames with camera in bin (i,j)
             of mean-pool(visual tokens at layer ℓ)
```

**Analysis**: same topology metrics vs. the 4-connected grid adjacency of
camera-position bins.

**What's different**: tests an **ego-localization code** — does the model
know "where the camera is in the world". Complementary to both Option 1
(where objects are) and your existing camera-Δ 6-DoF probes (relative
motion between frames).

**New infrastructure**: extraction needs to change — instead of object-masked
pooling, mean-pool the whole visual-token grid per frame. Write
`scripts/extract_frame_mean.py`.

### 6.3 Why Option 3 first

- Fully reuses existing extraction infrastructure (per-object-per-frame reps
  already in parquets for Qwen-7B/32B/72B, LLaVA-OV, InternVL3).
- Most faithful to the ICLR paper's methodology.
- Per-scene analysis decouples from any scene-distribution assumption (Option
  1 assumes bin population is homogeneous; Option 2 assumes camera-pose
  distribution is well-sampled).
- The Tier D rescue story works cleanly: if per-scene topology is preserved
  on real video even when linear probes fail, that changes the narrative
  without needing Options 1/2.

## 7. Success criteria

- **Pipeline works on Tier C (sanity):** ≥3 of 4 metrics significantly above
  their permutation nulls (z > 3) at some layer for Qwen-7B. Matches expected
  R² = 0.92 baseline.
- **Frame-count emergence detectable:** at least one metric shows monotonic
  improvement with N_frames for at least one model.
- **Model differences quantified:** best per-metric value and best-layer
  ranked across models, with paired stderr.
- **Figures reproduce paper-style panels:** Fig. 9 (per-layer PCA grid),
  Fig. 4 (layer × metric curve), Fig. 1 (scene PCA vs. ground truth).

If Tier C passes cleanly, a follow-up pass runs the identical pipeline on
`data/activations/tier_d_*` to test the nonlinear-rescue hypothesis for the
ARKitScenes failure.
