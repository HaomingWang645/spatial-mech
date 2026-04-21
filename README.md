# x-spatial-manual

Probing the 3D **spatial subspace** of video VLMs — does the model build an internal geometric representation of a scene that mirrors its true 3D layout, and does it actually *use* that representation when answering questions?

This repository is the implementation and experimental record for the plan in [`VLM_3D_Spatial_Subspace_Experiment_Plan_1.pdf`](VLM_3D_Spatial_Subspace_Experiment_Plan_1.pdf). It contains:

1. A tiered synthetic-scene generator (single BEV → fragmented BEV video → perspective ego-video, sharing one canonical 3D ground truth across tiers).
2. A model-agnostic activation-extraction pipeline with a Qwen2.5-VL 7B / 32B / 72B wrapper that hooks every language-model decoder layer and maps object segmentation masks onto the visual-token grid.
3. Probe families (linear ridge, PCA-then-linear, MLP upper bound, pairwise distance) with per-scene / per-object / cross-trajectory splits.
4. Camera-motion + per-object-depth probes that target the *extrinsic*-derived label (cam Δ, object depth), not just world coordinates.
5. A causal test: forward-pre-hook activation steering along a probe direction, evaluated both at the probe readout and against the model's text output (logit gap between two color names in a spatial comparison question).
6. Three run-level analyses ([reports/](reports/)) with headline figures.

Everything below refers to a single model family (Qwen2.5-VL) because it is the only family currently wrapped, but the extraction pipeline is model-agnostic and adding a second wrapper is a one-file job against the `VLMWrapper` protocol in [src/spatial_subspace/models.py](src/spatial_subspace/models.py).

---

## 1. Motivation

Recent interpretability work shows that LLMs and VLMs map abstract concepts onto low-dimensional geometric structures inside their hidden-state space, and that VLMs use linearly-extractable "Spatial IDs" to bind 2D image positions to object tokens. Both lines of work stop at 2D and at intermediate layers. The experiment plan asks:

- **Q1**  Does a low-dimensional *spatial subspace* exist in the hidden states of a video VLM, in which the projected object-token activations recover the true 3D layout up to an affine transform?
- **Q2**  Does the model actually *use* this subspace to make decisions, or is it an epiphenomenal correlate?
- **Q3**  Does subspace fidelity correlate with downstream 3D scene-understanding performance?
- **Q4**  Can subspace correctness be used as an auxiliary fine-tuning loss, and does it beat pure NTP on 3D reasoning tasks at matched compute?
- **Q5**  Can we give a Dirichlet-energy / manifold-learning account of when and why such a subspace must emerge?

The core hypotheses (H1–H6) and the monocular scale-ambiguity argument that motivates the label conventions are laid out in the PDF — read sections 1–3 of the plan for the scientific setup and this README for the code.

---

## 2. Headline findings so far

From the three run reports in [reports/](reports/):

| | Tier A single BEV | Tier B fragmented BEV video | Tier C perspective ego-video (orbit) |
|---|---|---|---|
| Best linear probe R² | **0.917** @ L0 | 0.476 @ L18 | **0.917** @ L12 |
| Best Procrustes | 0.082 @ L0 | 0.220 @ L18 | 0.112 @ L12 |
| Best pairwise ρ | 0.446 @ L0 | 0.418 @ L9 | **0.519** @ L20 |
| Layer profile | high plateau → decay | low bowl → mid peak → decay | climb → broad plateau |
| Cross-trajectory R² (H2) | — | — | **0.42 @ L27 vs 0.94 same-traj** |

Three main takeaways:

1. **The spatial subspace migrates deeper as the task gets harder.** Complete BEV → visual encoder. Fragmented BEV → mid-LM. Perspective video → mid-LM, higher ceiling.
2. **Within a single video forward, the subspace builds up across temporal tokens.** At every layer, per-`t` probe R² rises monotonically with the temporal-token index — at L22, R² goes from −0.04 at `t=0` to +0.43 at `t=6`. This is cross-temporal attention assembling the global scene, not a static property of middle layers.
3. **H2 fails.** The linear probe trained on trajectory 0 produces *negative* R² on trajectory 1 at early layers (−0.36 at L0) and only climbs to +0.42 at the last layer. The spatial code is camera-frame-dependent; de-rotation toward world frame is a gradient through depth, not a phase change, and it never finishes.

Latest ongoing work (ongoing as of 2026-04-21):

- Free-6DoF Tier C (visibility-repaired free-drift trajectory with roll) extracted on 7B, 32B, and 72B; latter-frames Q1 probes saved under [data/probes/tier_c_free6dof_qwen25vl_{7b,32b,72b}_q1_latter/](data/probes/).
- Camera-motion + per-object-depth probes ([scripts/fit_probes_camera_depth.py](scripts/fit_probes_camera_depth.py), results in [data/probes/tier_c_orbit_camera_depth/](data/probes/tier_c_orbit_camera_depth/)).
- First steering sweep: 7B, steer L12 → readout L27, axis x ([data/steering/tier_c_free6dof_7b_L12_x/](data/steering/tier_c_free6dof_7b_L12_x/)).

---

## 3. Repository layout

```
x-spatial-manual/
├── VLM_3D_Spatial_Subspace_Experiment_Plan_1.pdf   ← the plan
├── pyproject.toml                                   ← package metadata
├── configs/
│   ├── base.yaml                                    ← shared paths / seed
│   ├── tier_a.yaml                                  ← single BEV
│   ├── tier_b.yaml                                  ← fragmented BEV video
│   ├── tier_c.yaml                                  ← perspective orbit (default)
│   ├── tier_c_free6dof.yaml                         ← perspective free-6DoF (with visibility repair)
│   └── models/
│       ├── qwen25vl.yaml                            ← 7B, single-GPU
│       ├── qwen25vl_32b.yaml                        ← 32B, device_map=auto
│       └── qwen25vl_72b.yaml                        ← 72B, device_map=auto
├── src/spatial_subspace/
│   ├── scene.py                                     ← Scene / Object3D / Frame / Camera / QAItem dataclasses, JSON I/O
│   ├── render/
│   │   ├── common.py                                ← 3D scene sampler (unique (shape, color), pairwise rejection)
│   │   ├── qa.py                                    ← rule-based QA generator
│   │   ├── tier_a.py                                ← orthographic BEV + NW-sun shadows (PIL)
│   │   ├── tier_b.py                                ← AR(1) panning BEV crops
│   │   └── tier_c.py                                ← pinhole rasterizer, orbit + free6dof trajectories
│   ├── models.py                                    ← VLMWrapper protocol + Qwen25VLWrapper
│   ├── extract.py                                   ← mask→patch coverage, per-object pooling, image/video extraction
│   ├── labels.py                                    ← normalized coords / pairwise / rank; camera Δ-6D, object depth
│   ├── probes.py                                    ← ridge / MLP / PCA-linear / pairwise, scene/object splits
│   ├── metrics.py                                   ← R², Kabsch+scale Procrustes, pairwise Spearman
│   ├── datasets.py                                  ← scene directory iterator
│   ├── utils.py                                     ← YAML + seeding + path helpers
│   └── __init__.py
├── scripts/
│   ├── _bootstrap.py                                ← adds ./src to sys.path
│   ├── generate_scenes.py                           ← sample canonical 3D scenes (tier-agnostic)
│   ├── render_tier_{a,b,c}.py                       ← thin wrappers over src/spatial_subspace/render/*.main()
│   ├── extract_activations.py                       ← VLM forward pass, per-layer dump
│   ├── fit_probes_q1.py                             ← Q1 linear/PCA/MLP/pairwise per layer
│   ├── fit_probes_camera_depth.py                   ← extrinsic-derived cam Δ + per-object depth probes
│   ├── cross_trajectory_probe.py                    ← H2 same/cross/both protocols
│   ├── probe_temporal_dynamics.py                   ← per-(layer, t) probe — "does later give a better read?"
│   ├── activation_steering.py                       ← Δ = α·v injection; probe-readout evaluation
│   ├── activation_steering_text.py                  ← same Δ, but read the logit gap between two color tokens
│   ├── visualize_q1.py                              ← layer dynamics + reconstruction examples
│   ├── visualize_compare_tiers.py                   ← multi-tier overlay
│   └── sanity_tier_a.py                             ← framework self-test (no VLM)
├── tests/                                           ← pytest: render A/B/C, extract, labels, metrics, probes, scene
├── data/
│   ├── scenes_3d/                                   ← ~5000 canonical 3D scenes (scene.json only)
│   ├── tier_a/  tier_b/  tier_c/  tier_c_free6dof/  ← rendered per-tier output
│   ├── activations/<run_name>/                      ← layer_LL.parquet + layer_LL.npy
│   ├── probes/<run_name>/                           ← q1_probes.json / cross_trajectory.json / camera_depth_probes.json
│   └── steering/<run_name>/                         ← steering_results.parquet, steering.png, *.csv
├── figures/                                         ← headline PNGs checked into the repo
├── logs/                                            ← extraction / probe / steering stdout captures
└── reports/
    ├── tier_b_analysis.md                           ← fragmented-BEV result write-up
    ├── tier_b_temporal_analysis.md                  ← per-(layer, t) cross-temporal build-up
    └── tier_c_analysis.md                           ← Tier C + H2 cross-trajectory failure
```

---

## 4. Install

Python ≥ 3.10, Linux, NVIDIA GPU (one for 7B; two or more for 32B / 72B via `device_map=auto`).

```bash
# Lightweight install (probing + analysis only, no VLM)
pip install -e .

# Full install with Qwen2.5-VL support
pip install -e ".[vlm]"

# Dev (adds pytest + ruff)
pip install -e ".[vlm,dev]"
```

`qwen_vl_utils` is the processor helper that handles the video pathway (`process_vision_info`); it is pulled in by the `vlm` extra.

Scripts are runnable from the repo root without installing because each one imports [scripts/_bootstrap.py](scripts/_bootstrap.py), which prepends `./src` to `sys.path`. For library use from another project, prefer `pip install -e .`.

---

## 5. End-to-end example: Tier A

The full path from an empty repo to a Q1 probe plot for Tier A on Qwen2.5-VL-7B is:

```bash
# 1. Sample 1000 canonical 3D scenes (tier-agnostic ground truth, no frames).
python scripts/generate_scenes.py \
    --config configs/tier_a.yaml \
    --out data/scenes_3d --n-scenes 1000

# 2. Render a BEV image for each 3D scene.
python scripts/render_tier_a.py \
    --config configs/tier_a.yaml \
    --scenes-in data/scenes_3d \
    --out data/tier_a

# 3. Extract per-decoder-layer hidden states for each visible object.
python scripts/extract_activations.py \
    --data-root data/tier_a \
    --out-dir   data/activations/tier_a_qwen25vl_7b \
    --model-config configs/models/qwen25vl.yaml \
    --tier A --mode image

# 4. Fit Q1 probes across all 28 LM decoder layers.
python scripts/fit_probes_q1.py \
    --activations data/activations/tier_a_qwen25vl_7b \
    --out          data/probes/tier_a_qwen25vl_7b \
    --pairwise --mlp

# 5. Plot layer dynamics + reconstruction examples.
python scripts/visualize_q1.py \
    --probes      data/probes/tier_a_qwen25vl_7b \
    --activations data/activations/tier_a_qwen25vl_7b \
    --out figures/tier_a \
    --example-layer 0
```

You should see `linear_r2 ≈ 0.91–0.92` at layer 0 (see Tier A row of the table above).

---

## 6. Data pipeline

The design principle is **one canonical 3D scene, many tiers**. A `Scene` produced by [`generate_3d_scene`](src/spatial_subspace/render/common.py) contains only object positions, colors, shapes, sizes, and QA — no frames. Each tier's renderer reads the same `Scene` and adds its own frames, so probe results across tiers compare representations of *identical underlying worlds*.

### 6.1 Canonical 3D scenes — `scripts/generate_scenes.py`

[src/spatial_subspace/render/common.py](src/spatial_subspace/render/common.py) samples scenes subject to two invariants:

- **Unique (shape, color) pairs.** Sampled without replacement from the Cartesian product, so every object in a scene has a distinct `(shape, color)` pair — uniqueness is guaranteed, not attempted.
- **No horizontal overlap.** Per-pair rejection sampling: `d(i, j) ≥ r_i + r_j + min_gap` where the radius is `cfg["sizes"][obj.size]`. This fixes a subtle bug where the old global `min_separation` could be smaller than the sum of two max radii.

Every object rests on the floor at `floor_z + size`, so `bbox_min.z == floor_z` and `bbox_max.z == floor_z + 2*size`. The scene is genuinely 3D (non-zero extent in z); Tier A just projects it orthographically for BEV.

Each scene gets a rule-based QA set ([src/spatial_subspace/render/qa.py](src/spatial_subspace/render/qa.py)): relative-position questions (`left/right/front/behind`) for every pair and distance-order questions for every 3-tuple, shuffled and capped at 10 per scene.

### 6.2 Tier A — single BEV

[src/spatial_subspace/render/tier_a.py](src/spatial_subspace/render/tier_a.py): top-down orthographic projection rendered with PIL — pixel-exact masks, ~1000× faster than Blender, and sufficient because the probe target is geometry. Objects are drawn as circles (sphere, cylinder) or squares (cube); cubes and spheres at the same radius share the same silhouette from above, which is fine for position probing.

Shadows are simulated with a fixed oblique sun (`SUN_AZ_DEG=135°` NW, `SUN_ELEV_DEG=55°`) so the scene has a floor reference; shadows are painted *before* objects so the occlusion order is correct.

### 6.3 Tier B — fragmented BEV video

[src/spatial_subspace/render/tier_b.py](src/spatial_subspace/render/tier_b.py): same orthographic projection, but each frame is a **crop** of the world whose center follows an AR(1) random walk

```
v ← momentum · v + N(0, step_sigma²)
c ← c + v
```

with the crop window optionally clamped to the working volume. The crop side is half of the working-volume diagonal, so each frame shows ≈ 25% of the scene area. Each object is visible only in some frames; the model must integrate across frames to recover the layout.

The `temporal_shuffle` flag permutes the frame order (plan §3.2 ablation) while keeping the original chronological `frame_id` values, so downstream analysis can recover either order.

### 6.4 Tier C — perspective ego-video (orbit + free6dof)

[src/spatial_subspace/render/tier_c.py](src/spatial_subspace/render/tier_c.py) is a hand-rolled PIL rasterizer with painter-algorithm occlusion. Pixel-exact masks, no OpenGL dependency, millisecond per frame. Two things are done properly:

- **Silhouettes per shape.** Cubes project to the 2D convex hull of their 8 corners (a hexagonal silhouette under typical oblique views); cylinders use rim samples on top and bottom circles; spheres use the small-angle approximation `r_px = f · size / depth`.
- **Ground shadows.** Each object's ground footprint plus its shadow-shifted copy are projected to the image and filled in dark grey before any objects are painted.

Two trajectory modes:

| Mode | What | Config block |
|---|---|---|
| `orbit`   | 1-DoF planar arc; fixed altitude; camera always looks at scene centroid. One trajectory per `traj_idx`, enabling the H2 cross-trajectory test. | `trajectory: { mode: orbit, radii, altitudes, look_at_z, arc_degrees }` |
| `free6dof` | Smooth-noise drift on orbit radius, altitude, eye xyz, look-at xy, look-at z, and camera roll around the forward axis. Each frame runs through `_repair_visibility` — a blend-toward-nearest-object loop that guarantees ≥1 object remains on-screen. | `trajectory: { mode: free6dof, free6dof: { ... } }` |

Both modes index into the same `traj_idx` namespace so `render_tier_c.py --trajectories-per-scene 2` produces `<scene>_t0` and `<scene>_t1` under either mode. `Scene.extras["base_scene_id"]` preserves the cross-trajectory grouping key.

---

## 7. Activation extraction

[src/spatial_subspace/extract.py](src/spatial_subspace/extract.py) + [src/spatial_subspace/models.py](src/spatial_subspace/models.py).

### 7.1 Qwen25VLWrapper

Wraps `transformers`' `Qwen2_5_VLForConditionalGeneration`. On construction it registers a forward hook on **every** LM decoder layer; `forward()` runs one pass and exposes a `ForwardOut` with:

- `hidden_states[L]` — `(B, T, D)` post-layer tensor for layer `L`.
- `visual_token_range` — `(start, end)` index slice into the token sequence where `<|image_pad|>` / `<|video_pad|>` tokens live, located by `(input_ids == token_id).nonzero()`.
- `grid` — `(T_video, H_tok, W_tok)` post-merger visual-token grid. For images `T_video = 1`; for videos it is already `n_input_frames / temporal_patch_size` because the processor applies the temporal merger before the LM sees it. `H_tok = H_raw / merge_size`, `W_tok = W_raw / merge_size`.
- `extras` — at minimum the `input_ids`, an `is_video` flag, and (when logits are requested) the last-position logits as a CPU numpy array for the text-output steering script.

Decoder-layer location is robust to multiple `transformers` versions (see `_locate_layers`): we try a short list of paths including the pre-4.50 flat layout and the post-4.50 nested `Qwen2_5_VLModel → Qwen2_5_VLTextModel → layers` layout.

`install_intervention(layer_idx, token_positions, delta)` registers a forward-*pre*-hook that adds `delta` (shape `(D,)`) to the hidden states at the given positions before that layer executes. Pre-hook semantics match the plan (§6.1): the perturbation enters the residual stream *before* the layer transforms it, and the capture hooks fire on the layer's output so downstream captures reflect the post-intervention activations. `.remove()` on the returned handle fully undoes the intervention.

### 7.2 Mask → patch coverage → pooled object vector

`mask_to_patch_coverage(mask, (gh, gw), object_ids)` downsamples an `(H, W)` object-id mask (stored as `object_id + 1`, 0 = background) onto the `(gh, gw)` patch grid by reshaping to `(gh, ph, gw, pw)` and averaging over the `(ph, pw)` sub-blocks. The result is one `(gh, gw)` float-32 map per object giving the fraction of each patch's pixels belonging to that object.

`pool_object_vector(visual_hidden, coverage, threshold)` keeps only patches with `coverage ≥ threshold` (default 0.3) and returns a coverage-weighted mean of their hidden vectors. Returns `None` if no patch passes — those rows are dropped, not zero-filled.

### 7.3 Image vs video extraction

- `extract_scene` (`--mode image`) processes each `Frame` as an independent forward pass. Correct for Tier A; for multi-frame tiers it is the "infinite shuffle" ablation (temporal context removed).
- `extract_scene_video` (`--mode video`) bundles all frames into one video forward so the model sees them with M-RoPE temporal positions. Qwen2.5-VL's temporal patch merger (size 2 by default) collapses N input frames into N/2 temporal tokens; per temporal token we **average the source frames' coverage maps** so each temporal slot gets a fused mask. The on-disk `frame_id` then encodes the *temporal-token index*, not an input-frame index.

### 7.4 On-disk layout

One `(parquet, npy)` pair per layer per run:

```
data/activations/<run_name>/
├── layer_00.parquet    scene_id / object_id / frame_id / layer / vec_row / centroid_{x,y,z}
├── layer_00.npy        (N_rows, D) float32
├── layer_01.parquet
├── layer_01.npy
└── ...
```

`vec_row` is the index into the `.npy` array for that metadata row. Rows are written in iteration order; groups `[scene_id, object_id]` and temporal tokens are kept contiguous so an `object_summary` grouper is cheap.

### 7.5 Running on bigger models

`qwen25vl_32b.yaml` and `qwen25vl_72b.yaml` set `device_map: auto` so `from_pretrained` shards weights across all visible GPUs; input tensors still live on `cuda:0`. Only the `device_map` key differs from the 7B config. Runs so far are captured in [logs/extract_tier_c_free6dof_{7b,32b,72b}.log](logs/).

---

## 8. Probing — Q1

[src/spatial_subspace/probes.py](src/spatial_subspace/probes.py) exposes four probe families, all with a uniform `FitResult(model, r2, extras)` return. Driver scripts apply them in the way the plan calls for.

### 8.1 Label conventions — label normalization and the monocular scale ambiguity

[src/spatial_subspace/labels.py](src/spatial_subspace/labels.py) implements plan §3.7:

- `per_scene_normalized_coords(coords)` — centroid-subtract then divide by the bbox diagonal. Answers "does the model encode metric layout up to overall scale and origin?".
- `normalized_pairwise_distances(coords)` — `d_ij / max_ij d_ij` within a scene. Answers "does the model encode the relational geometry, independent of frame and scale?".
- `distance_rank_order(coords)` — Spearman-compatible, maximally invariant. "Does the model at least encode the topology?".

Raw absolute coordinates are available from `scene.json` and used only for (a) Procrustes-aligned reconstruction error and (b) the scale-ambiguity diagnostic from plan §3.7.1 (paired scenes with `2×` rescale).

Also in `labels.py`: `rotation_to_axis_angle` (Rodrigues), `camera_delta_6d(E_prev, E_curr)` (relative cam pose as `[tx, ty, tz, rx, ry, rz]`), and `object_depth_in_camera(p_world, E)` — the labels used by [`scripts/fit_probes_camera_depth.py`](scripts/fit_probes_camera_depth.py).

### 8.2 `scripts/fit_probes_q1.py`

Per-layer:

1. Aggregate per-`(scene, object)` "object summary" representations by mean-pooling over the activation rows (matches plan §4). The optional `--t-min k` flag restricts the mean to temporal tokens `t ≥ k` — used for Tier C free6dof runs where early temporal tokens have not yet integrated enough views.
2. Build per-scene normalized labels.
3. 80/20 per-scene split (deterministic by seed).
4. Fit:
   - `fit_linear_probe` — Ridge with `alpha=1` by default, closed-form.
   - `fit_pca_linear` at `k ∈ {2, 4, 8, 16, 32, 64}` — PCA on the training rows, then Ridge in PC space. Used for the *effective-rank* analysis (smallest `k*` such that PCA-k R² ≥ 95% of full-feature R²).
   - `fit_mlp_probe` (optional, `--mlp`) — one-hidden-layer MLP with `sklearn` early stopping. Upper bound on the information accessible nonlinearly.
   - `fit_pairwise_distance_probe` (optional, `--pairwise`) — Ridge from `[h_i, h_j]` concatenation to `d_ij / d_max(scene)` per-scene-normalized pairwise distance. Reports R² and Spearman ρ.

Outputs `q1_probes.json` and `q1_probes.parquet` in `--out`.

### 8.3 `scripts/visualize_q1.py`

Two figures per run:

- `q1_layer_dynamics.png` — 4 panels: full linear R² + PCA-k, Procrustes, pairwise Spearman, effective rank `k*` on log₂ axis.
- `q1_reconstruction_examples.png` — for a chosen layer, a grid of test scenes with ground-truth (o) and Procrustes-aligned probe prediction (×) overlaid; segment lines connect matched points.

`scripts/visualize_compare_tiers.py` overlays multiple runs (`--tier LABEL=path` repeats) so Tier A vs B vs C goes on one canvas ([figures/compare_tier_a_b_c.png](figures/compare_tier_a_b_c.png)).

### 8.4 Cross-trajectory probe (H2) — `scripts/cross_trajectory_probe.py`

Scene IDs of the form `<base>_t<k>` are split into `base_scene_id` and `traj_idx`. Three protocols per layer:

- `same_traj` — train on `(train_base, t=0)`, test on `(test_base, t=0)`. Standard cross-scene baseline.
- `cross_traj` — train on `(train_base, t=0)`, test on `(train_base, t=1)`. Same scenes, different trajectory. **This is the H2 test.**
- `cross_both` — train on `(train_base, t=0)`, test on `(test_base, t=1)`. Strict: held-out scene *and* held-out trajectory.

Outputs `cross_trajectory.json`, `cross_trajectory.parquet`, and a 2-panel R²/Procrustes plot.

### 8.5 Temporal dynamics — `scripts/probe_temporal_dynamics.py`

For each `(layer, t)`, fits a fresh ridge on per-temporal-token rows at exactly that `t`. The complementary `summary_to_t` mode fits the ridge on per-`(scene, object)` summaries and evaluates on per-`t` test rows.

This is the script behind [reports/tier_b_temporal_analysis.md](reports/tier_b_temporal_analysis.md): it confirmed that the mid-layer "bowl" reported in `tier_b_analysis.md` is driven by **causal cross-temporal attention building a global scene representation as more frames stream in**, not by a static property of middle layers.

### 8.6 Camera motion + per-object depth — `scripts/fit_probes_camera_depth.py`

Alternative to the world-frame Q1 probe: probes *extrinsic*-derived labels that a camera-frame representation should encode well.

- **Depth probe** — per-`(scene, obj, t)`. Target is `object_depth_in_camera(centroid_world, E[t·tps])`. Ridge fit on the per-object pooled vector directly.
- **Cam-delta probe** — per-`(scene, t)`. Target is `camera_delta_6d(E[(t-1)·tps], E[t·tps])` for `t ≥ 1`. Ridge fit on the mean-pooled per-object vector at that `(scene, t)` slot.

Both restricted to `t ≥ t_min` (default `n_tokens // 2`). Reports R² overall / translation / rotation, per-component residual stats, and a 4-panel figure with bias ± σ bands.

---

## 9. Causal tests — Q2

Two scripts, both in [scripts/](scripts/), both built on `Qwen25VLWrapper.install_intervention`.

### 9.1 `activation_steering.py`

Implements plan §6.1 in its probe-on-probe form. At the steer layer `L_steer`:

- Fit a Ridge probe (latter-frames) and recover its weight matrix `W_steer ∈ R^{3×D}`.
- Compute `v_x = W_steer.T @ (W_steer @ W_steer.T)⁻¹ @ e_x` — the min-norm hidden-state direction whose linear readout is `(1, 0, 0)`. `v_y`, `v_z` analogously. Pre-checked: `W_steer @ v_axis ≈ e_axis`.
- `v_perp` — a random vector projected onto the null space of `W_steer`, normalized to `||v_perp|| = ||v_axis||`. Pre-checked: `W_steer @ v_perp ≈ 0`. This is the *decisive* control.

For each `(direction, α)` combination: run the full forward with `install_intervention(L_steer, obj_positions, α · v)`, then pool the object's tokens at `L_readout` and apply that layer's probe `W_read`. Signal: the shift in the probe readout on the chosen axis relative to the baseline forward. Emits `steering_results.parquet`, `steering_shifts.parquet`, `steering_agg.csv`, and a 2-panel figure with the ideal-slope-1 diagonal overlaid.

### 9.2 `activation_steering_text.py`

Upgrades the probe-on-probe test to a text-output test. For each scene, build a comparison prompt like

> *This video shows a 3D scene filmed by an orbiting camera. Which object has a larger world-frame x-coordinate: the red object or the green object? Reply with one color word: red or green.*

The objects in the pair are ordered so A has the *lower* ground-truth x-value (so a steering direction that works pushes A's perceived-x higher and makes the model more likely to answer `colorA`). The signal is the **logit gap**

```
score = logits[first_token(" " + colorA)] − logits[first_token(" " + colorB)]
```

Same `v_axis` vs `v_perp` controls; report `Δ(logit_gap)` relative to the `α=0` baseline. A positive mean `Δ` for `v_axis` with a zero mean for `v_perp` is direct causal evidence that the spatial subspace drives verbal spatial answers.

---

## 10. Reports

| File | What it says |
|---|---|
| [reports/tier_b_analysis.md](reports/tier_b_analysis.md) | Tier A → B shift: spatial subspace migrates from L0 to L18 when views are fragmented. Capacity ceiling is lower under partial views, but the *form* (linear accessibility) peaks mid-stack. Effective rank stays small throughout. |
| [reports/tier_b_temporal_analysis.md](reports/tier_b_temporal_analysis.md) | The mid-layer peak is cross-temporal attention build-up: at every layer, R² at `t=0` is chance and climbs sharply with `t`. Biggest gap at L22 (Δ ≈ +0.47 between `t=0` and `t=6`). |
| [reports/tier_c_analysis.md](reports/tier_c_analysis.md) | Perspective video hits Tier-A-level probe R² (0.917) but at L12 instead of L0, and the code is preserved deeper into the stack. H2 fails: `cross_traj` R² is −0.36 at L0, crosses zero at L13, maxes at +0.42 at L27 — the model's spatial code is camera-frame-dependent and only partially de-rotates through depth. |

The experiment plan itself is [VLM_3D_Spatial_Subspace_Experiment_Plan_1.pdf](VLM_3D_Spatial_Subspace_Experiment_Plan_1.pdf).

---

## 11. Tests

```bash
pytest -q
```

Covers (all without loading a VLM):

- Canonical 3D scene sampling: unique `(shape, color)` pairs, no horizontal overlap, genuine 3D (non-zero z).
- Tier A / B / C renders: file layout, scene round-trip, per-frame visibility, temporal shuffle, two trajectories differ, free6dof visibility repair, painter-algorithm camera math (`look_at`, `project`).
- Mask→patch coverage on a 28×28 → 2×2 grid with hand-checked expected values; pooling respects the coverage threshold.
- Label normalization: unit-ball, centroid-zero, max-pairwise = 1, rank-order is a permutation, `rotation_to_axis_angle` identity and known-z rotation, `camera_delta_6d` zero for identical extrinsics, `object_depth_in_camera` matches projection depth.
- Metrics: R² perfect on identity, Procrustes recovers rigid + scale transforms, pairwise Spearman ≈ 1 on identity.
- Synthetic probe sanity: if the first 3 feature dims are the target plus isotropic noise, linear probe R² > 0.9, PCA-8 R² > 0.8.

[scripts/sanity_tier_a.py](scripts/sanity_tier_a.py) is the end-to-end framework self-test: render 50 Tier A scenes, fabricate hidden states whose first 3 dims are the per-scene normalized coords plus noise, fit the full probe stack, assert `R² ≥ 0.9`. No GPU, no VLM — this catches wiring regressions in seconds.

---

## 12. Configuration reference

### 12.1 Base scene controls — all tiers inherit

```yaml
min_objects: 3
max_objects: 8
floor_z: 0.0
min_gap: 0.2              # edge-to-edge gap (d_ij ≥ r_i + r_j + min_gap)
shapes: [cube, sphere, cylinder]
colors:                   # RGB tuples used by the PIL rasterizer
  red:    [220, 50, 50]
  ...
sizes:                    # each object's centroid z = floor_z + size; bbox z ∈ [floor_z, 2·size]
  small:  0.4
  medium: 0.6
  large:  0.8
working_volume:
  x: [-4.0, 4.0]
  y: [-4.0, 4.0]
image_size: 448           # multiple of 28 for Qwen2.5-VL's 14×14 patch × 2×2 merge
background_gray: 155
```

### 12.2 Tier B additions

```yaml
n_frames: 16              # 8 / 16 / 32 are the plan variants
trajectory:
  window_world_size: 4.0  # crop side in world units (half of working volume → ¼ of area)
  step_sigma: 1.2         # std-dev of per-frame center delta
  momentum: 0.55          # AR(1) coefficient on velocity
  zoom_sigma: 0.0         # > 0 to vary crop size per frame
  out_of_bounds: clamp
temporal_shuffle: false   # plan §3.2 ablation — permutes rendered frames
```

### 12.3 Tier C additions

```yaml
fov_degrees: 60.0
trajectory:
  mode: orbit             # or "free6dof"
  # Indexed by traj_idx so each scene can be rendered under multiple
  # independent trajectories (H2 test).
  radii:      [8.0, 9.0]
  altitudes:  [3.5, 4.5]
  look_at_z:  0.5
  arc_degrees: 180.0

  free6dof:
    base_radii:     [7.5, 8.5, 9.5, 10.5]
    base_altitudes: [2.8, 3.8, 4.8, 5.8]
    arc_degrees:    220.0
    n_modes:        3            # smooth-noise frequency basis per dim
    eye_jitter:       1.2
    radius_jitter:    1.0
    altitude_jitter:  0.8
    target_jitter:    1.5
    target_z_jitter:  0.4
    roll_max_degrees: 20.0
    visibility_min_radius_px: 2.0   # object counts as visible if apparent radius ≥ this
    visibility_margin_px:     0.0   # … and its disc touches the image (+margin)
    repair_max_iters:         8     # blend-toward-anchor iterations when no object visible
```

### 12.4 Model config

```yaml
name: Qwen2.5-VL-72B
hf_id: Qwen/Qwen2.5-VL-72B-Instruct
torch_dtype: bfloat16
device: cuda              # input-tensor placement
device_map: auto          # shard weights across all visible GPUs (omit for single-GPU 7B)
patch_size: 14
merge_size: 2             # each visual token covers 28×28 px
prompt: "Describe the spatial layout of the objects in this image."
```

---

## 13. Reproducibility & seeding

- `spatial_subspace.utils.set_seed` seeds Python `random`, `numpy`, `PYTHONHASHSEED`, and (if importable) `torch` + CUDA.
- Every script accepts `--seed` (default 0). Split constructors in `probes.py` take `seed` directly.
- Per-scene RNGs in `render_tier_b.py` / `render_tier_c.py` are derived from `args.seed ^ hash((scene_id, traj_idx))` so trajectories are deterministic given `(seed, scene_id, traj_idx)` — adding trajectories to an existing run is stable.
- Activation extraction is deterministic given the same model weights and inputs; `transformers` sometimes yields tiny floating-point drift across minor versions, which rarely matters at probe-R² precision.

---

## 14. Known caveats, in roughly decreasing order of how much they matter

1. **Only Qwen2.5-VL is wrapped.** The plan calls for ≥3 open models. Adding LLaVA-Video-7B / InternVL3-8B is a single-file job against the `VLMWrapper` protocol — expose `forward`, `patch_pixels`, `image_input_hw`, `close`, and optionally `install_intervention` + `temporal_patch_size`. The mid-layer-peak / H2-failure findings could be Qwen-architecture-specific and need to be checked.
2. **Hand-rolled rasterizer simplification.** Tier C uses filled circles / convex hulls / oblique shadows — no per-material lighting, no textures, no explicit floor grid. Fine for position probing, but richer visuals (Kubric/Blender) could change the picture. The plan suggests Kubric for Tier C going forward.
3. **Z is bounded by object size.** All objects rest on the floor so `z ∈ {0.4, 0.6, 0.8}` — effectively a discrete encoding of size, not a free vertical position. Probe R² is dominated by x/y recovery; z is not stress-tested. A `z_jitter` config field for stacked or floating objects is a small extension to `common.py`.
4. **Tier D (real video) not yet implemented.** ScanNet / EmbodiedScan integration is next on the list — the probing scripts work unchanged as long as the adapter produces `Scene` objects with ground-truth 3D centroids and per-frame masks.
5. **`sklearn.MLPRegressor` diverges at late layers.** R² goes strongly negative; not reliable as the linearity ceiling. A torch-based implementation with proper weight decay and early stopping is the fix.
6. **Scale-ambiguity diagnostic (plan §3.7.1) not yet implemented.** Paired `(a)` / `(b)` / `(c)` scenes at `1×` and `2×` rescales; predicted by the plan, needed to verify that the per-scene-normalized labels are not losing real signal.
7. **Q4 auxiliary-loss fine-tuning and Q5 Dirichlet-energy theory not yet implemented.** Those are the dominant compute costs in the plan (§10.2: ~10k A100-h for Q4) and have not been attempted yet.

---

## 15. References

- The plan: [VLM_3D_Spatial_Subspace_Experiment_Plan_1.pdf](VLM_3D_Spatial_Subspace_Experiment_Plan_1.pdf)
- Qwen2.5-VL model family: Qwen2.5-VL-{7B, 32B, 72B}-Instruct on Hugging Face, served via `Qwen2_5_VLForConditionalGeneration` in `transformers`.
- CLEVR (Johnson et al., 2017) — synthetic-scene inspiration for the Tier A/B/C scene generator. The per-object-segmentation + JSON-metadata recipe is adapted from the original CLEVR generator.
- Kubric — recommended replacement for the hand-rolled rasterizer in Tier C when better visuals are needed.

---

## 16. License

No license file has been added yet. Until one is, treat this repository as "all rights reserved" — do not redistribute, and check with the author before building on it externally. A permissive license (likely MIT or Apache-2.0) will be added before any public release.
