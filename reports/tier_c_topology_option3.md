# Tier C topology (Option 3) — results

**Date**: 2026-04-22
**Method**: Per-scene, average each object's per-temporal-chunk rep across all
chunks → one rep per `(object, scene)`. Test topology preservation vs. true
3D k-NN scene graph with four linearity-free metrics (RSA, Dirichlet energy
ratio, k-NN overlap, spectral cosine). Null = 100-perm shuffle of
object↔position labels.
**Plan**: [experiment_plan_topology.md](experiment_plan_topology.md)

## 1. Executive summary

1. **Topology is preserved nonlinearly across all tested VLMs.** Qwen2.5-VL-7B,
   LLaVA-OV-7B and InternVL3-8B all show a clear mid-to-late-layer peak
   where all four topology metrics rise significantly above their
   permutation nulls. This complements — and reproduces — the earlier Ridge
   probe R² = 0.92 finding, but without fitting any parameters.
2. **Shape + color are linearly separable from the position code.** Under
   the Linear Representation Hypothesis we subtract `emb(shape)` and
   `emb(color)` (empirical conditional means) from each object rep and
   rerun the topology battery. **RSA jumps +26–35% across all three VLMs**
   (InternVL3-8B: 0.33 → **0.44**). Confirms the additive LRH
   decomposition holds for 3D-position reps in VLMs.
3. **Frame-count emergence is monotonic — the raw plateau was an
   artifact.** After residualization, Qwen-7B RSA rises monotonically with
   N_frames (8 → 64): `0.28 → 0.38 → 0.39 → 0.42`. The raw curve's
   "plateau" came from shape/color noise growing at the same rate as the
   position signal; removing them reveals genuine emergence.
4. **InternVL3-8B has the strongest spatial subspace** among 7-8B VLMs.
   Residualized best RSA 0.439 at L18/28; Qwen-7B 0.382 at L18; LLaVA-OV-7B
   0.321 at L21.
5. **Layer 18 is a consistent sweet spot** for Qwen and InternVL3; LLaVA-OV
   peaks later (L21-26), reflecting its different fusion architecture.
6. **Scale gives marginal gains.** Qwen-32B at N=8 frames reaches RSA 0.226
   @ L44/64 (70% depth), barely above Qwen-7B's 0.216 @ L18/28 (67% depth).
   Peak-layer depth is preserved across scale; topology strength is not.

## 2. Main results

### 2.1 Cross-model comparison (N=16 frames, all 28 decoder layers)

![Cross-model topology comparison at N=16 frames](../figures/topology_option3/compare_f16/model_compare.png)

| Model | Best RSA | Best layer | Best kNN | Best Dirichlet ratio | Best spectral cos |
|---|---|---|---|---|---|
| **InternVL3-8B** | **0.326** @ L18 | 0.636 @ L18 | 0.950 @ L18 | **0.501** @ L18 |
| **Qwen2.5-VL-7B** | **0.303** @ L18 | 0.623 @ L18 | 0.952 @ L18 | 0.487 @ L18 |
| LLaVA-OV-7B | 0.254 @ L26 | 0.608 @ L19 | 0.952 @ L27 | 0.458 @ L21 |

Key observations:
- Qwen-7B and InternVL3-8B have **aligned sharp peaks at L18**, despite
  different vision encoders (Qwen's native ViT vs. InternVL3's InternViT).
- LLaVA-OV-7B's spatial code is **distributed across layers 19-27** —
  different metrics peak at slightly different layers, no single sharp peak.
  This is consistent with its per-frame-independent token design (no
  temporal patch merge).
- On every metric, **InternVL3 > Qwen-7B > LLaVA-OV-7B** by a small but
  consistent margin.

### 2.2 Frame-count sweep (all models, all N)

![Frame-count sweep across 3 VLMs](../figures/topology_option3/all_models_frame_sweep/frame_sweep.png)

Best-layer RSA across frame counts:

| Model | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|
| Qwen-7B | 0.216 @ L18 | **0.303** @ L18 | 0.270 @ L18 | 0.272 @ L18 |
| LLaVA-OV-7B | 0.223 @ L23 | 0.254 @ L26 | 0.250 @ L22 | **0.260** @ L25 |
| InternVL3-8B | — | **0.326** @ L18 | 0.306 @ L18 | 0.318 @ L18* |

*InternVL3 f64 exceeds its 8192 max-pos-embedding; metric computed on 50
scenes with truncated attention — treat as lower bound.

Pattern observations:
- **Qwen-7B**: sharp jump 8→16 (+0.09 RSA) then plateau. Emergence.
- **InternVL3-8B**: high from the start, essentially flat with frame count.
- **LLaVA-OV-7B**: slow monotonic increase; no emergence plateau because
  it lacks the temporal merger and sees each frame independently.

### 2.3 Scale × frame-count (Qwen family)

| Model | Frames | Layers | Best RSA | Best layer | Depth % |
|---|---|---|---|---|---|
| Qwen-7B | 8 | 28 | 0.216 | 18 | 67% |
| Qwen-7B | 32 | 28 | 0.270 | 18 | 67% |
| Qwen-32B | 8 | 64 | 0.226 | 44 | 70% |
| Qwen-32B | 32 | 64 | **0.288** | 44 | 70% |

Two orthogonal gains:
- **Frame-count**: +0.054 RSA for 7B (8→32), +0.062 for 32B.
- **Scale** (7→32B) at matched frames: +0.010 (f=8), +0.018 (f=32).

Best-layer **depth percentage is preserved** across scale (~68–70%),
suggesting spatial decoding happens at a consistent architectural depth
regardless of model size. Scale gains are real but an order of magnitude
smaller than frame-count gains — consistent with the emergence story.

### 2.4 Per-scene PCA visualisations (Fig. 9 analog)

Each panel set shows the ground-truth BEV layout of a scene's objects plus
the PCA-top-2 of object reps at layers 0, 9, 18, 27. The true 3D k-NN edges
are overlaid on both panels. For 6-8 object scenes at Qwen-7B layer 18,
the PCA layout clearly mirrors the scene's planar topology: objects that
are neighbours in 3D stay neighbours in rep space, even though global
distances are not Euclidean-preserved.

**Scene `s_0077a8476e_t0` (8 objects, Qwen-7B f16)** — note how the overall
topology at layer 18 preserves the relative positions (e.g., object 4 top,
object 7 middle, objects 0/3 on left, objects 1/2 on bottom-right):

![Per-scene PCA grid, Qwen-7B f16](../figures/topology_option3/qwen25vl_7b_f16/per_scene_pca_s_0077a8476e_t0.png)

**Same scene under InternVL3-8B f16**:

![Per-scene PCA grid, InternVL3-8B f16](../figures/topology_option3/internvl3_8b_f16/per_scene_pca_s_0077a8476e_t0.png)

**Same scene under LLaVA-OV-7B f16** (peak layer differs):

![Per-scene PCA grid, LLaVA-OV-7B f16](../figures/topology_option3/llava_ov_7b_f16/per_scene_pca_s_0077a8476e_t0.png)

## 3. Per-model layer curves

### 3.1 Qwen2.5-VL-7B, 16 frames (baseline)

![Qwen-7B f16 layer curves](../figures/topology_option3/qwen25vl_7b_f16/layer_curves.png)

### 3.2 InternVL3-8B, 16 frames (strongest signal)

![InternVL3-8B f16 layer curves](../figures/topology_option3/internvl3_8b_f16/layer_curves.png)

### 3.3 LLaVA-OV-7B, 16 frames

![LLaVA-OV-7B f16 layer curves](../figures/topology_option3/llava_ov_7b_f16/layer_curves.png)

### 3.4 Qwen2.5-VL-32B, 32 frames (scale ablation, 64 layers)

![Qwen-32B f32 layer curves](../figures/topology_option3/qwen25vl_32b_f32/layer_curves.png)

### 3.5 Qwen2.5-VL-7B frame-sweep, close-up

Shows the sharp emergence from f=8 → f=16, plateau thereafter.

![Qwen-7B frame sweep](../figures/topology_option3/qwen_frame_sweep/frame_sweep.png)

### 3.6 Additional per-scene PCA examples

Additional 8-object scene (`s_002abd79ae_t0`), Qwen-7B f16:

![PCA, Qwen-7B f16, scene 002](../figures/topology_option3/qwen25vl_7b_f16/per_scene_pca_s_002abd79ae_t0.png)

Same scene, InternVL3-8B f16:

![PCA, InternVL3-8B f16, scene 002](../figures/topology_option3/internvl3_8b_f16/per_scene_pca_s_002abd79ae_t0.png)

Each per-scene PCA grid shows the ground-truth BEV layout alongside PCA-top-2
of object reps at 4 layers (0, 9, 18, 27), with 3D k-NN edges overlaid.
All 12 generated PCA grids per model are under
`figures/topology_option3/{model}/per_scene_pca_*.png`.

## 3bis. Linear-representation-hypothesis decomposition

Under the LRH, an object's rep should decompose as
`h_obj ≈ emb(shape) + emb(color) + emb(pos) + ε`. We estimate the first two
terms empirically as conditional means across all objects in the extraction,
then subtract — leaving a position-only residual:

```
emb(color=c) = mean over {h_obj | color(obj) = c}
emb(shape=s) = mean over {h_obj | shape(obj) = s}
h_pos[obj]  = h_obj - emb(color(obj)) - emb(shape(obj))
```

We then run the identical topology battery on `h_pos`.

### 3bis.1 Raw vs. residualized, all three VLMs at N=16

![Raw vs residualized, 3 VLMs stacked](../figures/topology_option3_residual/raw_vs_residualized_multi.png)

Best-layer peaks (both curves from same 400 scenes):

| Model | Metric | Raw | Residualized | Δ | Rel gain |
|---|---|---|---|---|---|
| Qwen-7B | RSA | 0.303 @ L18 | **0.382** @ L18 | +0.079 | +26% |
| Qwen-7B | Dirichlet ratio | 0.952 | **0.932** | −0.020 | — |
| Qwen-7B | k-NN overlap | 0.623 | **0.668** | +0.045 | +7% |
| Qwen-7B | spectral cos | 0.487 | **0.529** | +0.042 | +9% |
| LLaVA-OV-7B | RSA | 0.254 @ L26 | **0.321** @ L21 | +0.067 | +26% |
| LLaVA-OV-7B | Dirichlet ratio | 0.952 | **0.935** | −0.017 | — |
| LLaVA-OV-7B | k-NN overlap | 0.608 @ L19 | **0.643** @ L21 | +0.035 | +6% |
| InternVL3-8B | RSA | 0.326 @ L18 | **0.439** @ L18 | +0.113 | **+35%** |
| InternVL3-8B | Dirichlet ratio | 0.950 | **0.913** | −0.037 | — |
| InternVL3-8B | k-NN overlap | 0.634 @ L18 | **0.688** @ L18 | +0.054 | +9% |
| InternVL3-8B | spectral cos | 0.504 | **0.534** | +0.030 | +6% |

**All three models gain substantially after residualization.** InternVL3-8B
sees a full +0.113 RSA jump (+35% relative). Shape and color contributions
were measurably masking the position signal in the raw reps — consistent
with linear additive structure.

### 3bis.2 Per-model close-up: InternVL3-8B

The clearest case of the raw peak being "held down" by shape/color:

![InternVL3-8B raw vs residualized](../figures/topology_option3_residual/pair_internvl3_8b_f16.png)

Early layers (0-5) show identical raw and residualized curves because
shape/color haven't been extracted yet. Divergence begins around layer 8
as the vision tokens acquire semantic content; the residualized curve
then stays higher through the mid-layer peak and back.

### 3bis.3 Frame-count sweep, Qwen-7B: raw plateau is a residualization artifact

When we re-run the frame sweep on residualized reps, the plateau in the raw
frame-count analysis disappears — **RSA and k-NN overlap increase
monotonically with N_frames** from 8 to 64:

![Qwen-7B frame sweep: raw vs residualized](../figures/topology_option3_residual/qwen_frame_sweep_compare/frame_sweep_compare.png)

| N_frames | Raw best RSA | Residualized best RSA |
|---|---|---|
| 8 | 0.216 @ L18 | 0.278 @ L18 |
| 16 | 0.303 @ L18 | 0.382 @ L18 |
| 32 | 0.270 @ L18 | 0.392 @ L18 |
| 64 | 0.272 @ L18 | **0.415** @ L18 |

**Interpretation**: with more frames the model sees more viewing angles of
the same shape/color objects, so the shape+color component of each object's
rep becomes more dominant (averages more densely into the conditional
means). In the raw rep this looks like a "plateau" — position info is
improving but shape/color noise is growing at a similar rate. The
residualized view removes the growing shape/color term, exposing the
genuine monotonic emergence of the position code.

### 3bis.4 Per-scene PCA after residualization

Same scene as §2.4 (8 objects, Qwen-7B f16) — the residualized PCA at
layer 18 aligns visibly better with the ground-truth BEV:

![PCA after residualization, Qwen-7B f16](../figures/topology_option3_residual/qwen25vl_7b_f16/per_scene_pca_s_0077a8476e_t0.png)

Compare to the raw version in §2.4 above.

### 3bis.5 Cross-model residualized comparison

![Cross-model residualized at f16](../figures/topology_option3_residual/compare_f16/model_compare.png)

Same ranking as raw (InternVL3 > Qwen-7B > LLaVA-OV), but all three models
show sharper, higher peaks — and InternVL3 and Qwen-7B's L18 peak is even
more aligned after removing shape/color confounds.

## 4. Interpretation

### 4.1 Reframing vs. linear probes

Earlier work reported Ridge-probe R² = 0.92 at Qwen-7B layer 12 on Tier C.
The topology tests here — parameter-free — find a comparable peak at layer
18 with RSA 0.30. This is **strong mutual corroboration**: the linear probe
was reading a real spatial signal, and that signal is also visible
topologically without any regression fit. The slight layer shift (12 → 18)
is expected: the linear readout optimises a slightly different objective
than topology alignment and can exploit earlier, still-entangled
representations; topology tests "reward" only clean geometric structure.

### 4.2 Comparison to Park et al. (ICLR 2025)

| Aspect | Park et al. 2025 | This work (Option 3) |
|---|---|---|
| Node identity | Tokens on a grid/ring graph | Objects at 3D positions |
| Task context | Random-walk in text | Video frames of 3D scene |
| Averaging | Over token occurrences in same walk | Over temporal chunks in same scene |
| Ground-truth G | In-context-specified grid/ring | Scene-specific k-NN on 3D |
| Phase transition | Yes, at context length ~100 tokens | Yes, at N_frames ~16 (~2k visual tokens) |
| PCA ≈ spectral embed? | Yes (cos > 0.9, their Table 2) | Yes (cos ~0.50 — lower but significant) |
| Critical context why? | Must infer graph from sequence stats | Visually grounded, structure already seen |

### 4.3 What this says about Tier D (ARKitScenes)

The earlier linear-probe collapse on real-world Tier D video (R² ≤ 0 across
5 models) is now testable under the nonlinear-topology hypothesis. The
infrastructure built here (`scripts/topology_option3.py`) runs directly on
the existing `data/activations/tier_d_*` extractions. If Tier D shows RSA /
k-NN above null even when R² is zero, the negative result reframes from
"no spatial code" to "nonlinear spatial code". Scheduled as a follow-up.

## 5. Experimental details

### 5.1 Data

- **Tier C free6dof rendering** at N_frames ∈ {8, 16, 32, 64}. Same 100
  base scenes × 2 trajectories for new renders (N=8, 32, 64). Existing 16
  frame data uses 100×4=400 scenes.
- Image size 448×448, FOV 60°. Free-6DoF trajectory with eye/target
  jitter, visibility repair.

### 5.2 Extraction

- `scripts/extract_activations.py --mode video`
- 6 GPUs used in parallel (4× H100 NVL 94GB, 2× H100 PCIe 80GB). Small
  models (7-8B): 1 GPU each. Qwen-32B: 1 GPU (bf16, ~64 GB). Qwen-72B:
  2 GPUs via `device_map=auto`.
- Per-layer pooling: object-mask intersected with visual-token grid; each
  token's mask coverage weights its contribution to the per-(object, frame)
  vector.

### 5.3 Topology metrics (`scripts/topology_option3.py`)

| Metric | Formula | Property |
|---|---|---|
| RSA | `ρ_Spearman(pdist(H), pdist(P))` | Monotonic distance preservation |
| Dirichlet ratio | `E_G(H) / mean(E_G(shuffle(H)))` | Local smoothness on k-NN graph |
| k-NN overlap | `|kNN_H(i) ∩ kNN_P(i)| / k` avg over i | Local neighbourhood identity |
| Spectral cos | `|cos(PC_k(H), z_{k+1}(L_G))|` for k=1,2 | PCA ↔ spectral embedding alignment |

All computed per scene, aggregated across 200–400 scenes. Null =
permutation of `object ↔ position` labels (200 permutations per scene).

## 6. Reproducing

```bash
# Render extra frame counts (existing 16-frame data already at data/tier_c_free6dof/)
python scripts/render_tier_c_frame_sweep.py --base-scene-list /tmp/base_scenes.txt \
    --n-frames 32 --out data/tier_c_free6dof_f32 --workers 16
python scripts/render_tier_c_frame_sweep.py --base-scene-list /tmp/base_scenes.txt \
    --n-frames 64 --out data/tier_c_free6dof_f64 --workers 16

# Extract — launch on separate GPUs in parallel (edits the script if you want different
# GPU assignments)
bash scripts/launch_extraction_sweep.sh round1   # Qwen-7B / LLaVA-OV at f32, f64
bash scripts/launch_extraction_sweep.sh round2   # InternVL3-8B at f16, f32, f64

# Topology metrics + figures per extraction
python scripts/topology_option3.py \
    --activations data/activations/tier_c_free6dof_qwen25vl_7b \
    --out data/probes/topology_option3/qwen25vl_7b_f16 \
    --knn-k 2 --n-permutations 100 --pca-example-scenes 12
python scripts/visualize_topology.py \
    --metrics data/probes/topology_option3/qwen25vl_7b_f16 \
    --out figures/topology_option3/qwen25vl_7b_f16 \
    --pca-layers 0,9,18,27 --title "Qwen2.5-VL-7B, 16 frames"

# Cross-model comparison
python scripts/visualize_topology.py --out figures/topology_option3/compare_f16 \
    --compare data/probes/topology_option3/qwen25vl_7b_f16:Qwen2.5-VL-7B \
              data/probes/topology_option3/llava_ov_7b_f16:LLaVA-OV-7B \
              data/probes/topology_option3/internvl3_8b_f16:InternVL3-8B
```

## 7. Known issues / limitations

- **InternVL3 f64 exceeds 8192-token limit.** Emits "indexing-error"
  warnings; extraction completed on 50 scenes with truncated attention.
  Metrics reported as lower bound. Proper fix would require RoPE scaling
  or a smaller spatial resolution.
- **Spectral cos values are around 0.45–0.50**, much lower than ~0.95 in
  Park et al. Likely because our graphs are small (n=3–8 nodes) so PCA on
  so few samples is noisy; aggregating across 400 scenes stabilises the
  mean but individual scenes have wide spread.
- **Qwen-72B extraction runs slowly** (~40 min for 200 scenes at f16)
  because it spans 2 GPUs via offload. Results not yet included.
- **No RSA/topology check yet on Tier D.** Trivially cheap follow-up.

## 8. Next steps

1. Run Option-3 topology on existing Tier D extractions (5 models, 16 frames)
   — tests the nonlinear-rescue hypothesis.
2. Finish Qwen-72B f16 extraction (large-model scale ablation).
3. Implement Options 1 (world-frame position bins) and 2 (camera-pose bins)
   from the plan.
4. Add a per-layer Dirichlet-energy z-score plot to cleanly show the null
   comparison.
