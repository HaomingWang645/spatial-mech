# Dirichlet-Loss Training: Detailed Plan

**Goal.** Add a 3D-geometry-weighted Dirichlet-energy regularizer to the
language-modelling loss during VLM finetuning, and demonstrate that it
(a) lowers the layer-wise Dirichlet ratio, (b) raises residualized-RSA
at the peak layer, and (c) improves downstream spatial-VQA accuracy.

The goal is to convert the diagnostic measurement (Dirichlet ratio,
residualized RSA) of our analysis paper into an active training signal —
in the same spirit as Kang et al.'s spatial-ID loss (ICLR 2026), but
with theoretical backing from our Theorem 3.

---

## 1. Motivation

### 1.1. Why a representation-level loss?

Standard finetuning supervises only the LM output. The residual stream
is regulated only indirectly through the gradient of the LM loss back
through layers. If the model finds a non-3D shortcut that achieves low
LM loss, it will use it — leading to the depth-shortcut behaviour
documented in §5.7 of the analysis report. A representation-level loss
that *directly* shapes the residual stream's geometry is the natural
counter-measure.

### 1.2. Why specifically the Dirichlet ratio?

Three reasons:

1. **It's exactly what we measure.** The Dirichlet ratio
   $\widetilde{\mathcal{R}}_X(H) = \mathcal{E}_X(H)/\sum_{ij}\|h_i - h_j\|^2$
   appears throughout the analysis report as the diagnostic for "is the
   representation 3D-smooth at this layer?". Optimizing it directly
   closes the loop between measurement and intervention.

2. **Theorem 3 gives a guarantee.** Among all $H$ matrices of fixed
   spectral profile, the minimizer of $\mathcal{E}_X(H)$ has its top-3
   PCs equal to the eigenmaps of the scene-induced Laplacian. By the
   Belkin–Niyogi limit (Theorem 3′), these eigenmaps converge to the
   world-coordinate axes. So minimizing the Dirichlet ratio during
   training provably trains the model to encode world coordinates as
   its top-3 PCs at the chosen layer.

3. **It works without "extracting" reference vectors.** Unlike Kang et
   al.'s cosine-sim-to-spatial-IDs loss, which requires running an
   extraction pipeline beforehand on a synthetic grid, the Dirichlet
   loss is computable from scratch given only $H$ and ground-truth 3D
   coordinates $X$.

### 1.3. Why now, in this paper?

Section 8 of Kang et al. 2026 closes with: *"future work could include
expanded use cases such as explicit temporal guidance at large scale."*
A 3D-Dirichlet loss is exactly that — a generalization of their
2D-grid-style supervision to continuous 3D scenes, with formal
guarantees they don't have. It positions our paper as both an
*analytical* extension and a *constructive* improvement over Kang.

---

## 2. Comparison with existing approaches

| Method | What it supervises | Where signal comes from | Theoretical backing | Continuous 3D? |
|---|---|---|---|---|
| **Output-level supervision** (SpatialVLM, Chen et al. 2024) | Output token logits on synthetic 3D-QA | Manual QA generation | None | Yes |
| **Cosine-sim to extracted spatial IDs** (Kang et al. 2026) | Single layer's object-token activations | Empirically extracted IDs from a 4×4 grid | None | No (grid-bound) |
| **Activation distillation** (Hinton et al. 2015) | Match teacher's residuals | Strong teacher model | None | Yes |
| **Contrastive scene-pair loss** (e.g., DINO-style) | Same-scene attractive, cross-scene repulsive | Self-supervision via augmentations | Some (InfoNCE) | Yes |
| **Graph Laplacian regularization** (Belkin & Niyogi 2006, classical SSL) | Smoothness of predictor on graph | Geometric structure of unlabeled data | Spectral graph theory | Yes (in feature space, not 3D) |
| **Our Dirichlet loss** | Residual stream at chosen layer | 3D scene coords via Gaussian kernel | **Theorem 3** | **Yes (native)** |

The closest existing method is Belkin–Niyogi's classical Laplacian
regularization for semi-supervised learning. We borrow the
mathematical structure but apply it in a different context: the
"graph" is induced by the 3D scene geometry of a video, and we
regularize the residual stream of an LLM rather than a learned
classifier.

---

## 3. Theoretical guarantees

### 3.1. Forward direction (Theorem 3 from theory_draft.md)

Under spectral-profile constraints on $H$, the minimizer of
$\mathcal{E}_X(H)$ has top-3 PCs equal to the second through fourth
eigenvectors of the scene-Laplacian $L$. With Gaussian kernel of
appropriate bandwidth, these eigenvectors are (asymptotically) the
world-coordinate axes.

**Implication for training.** Training with the Dirichlet loss will
*provably* drive the chosen layer's residual stream toward the
"world-coordinates-in-top-3-PCs" configuration that we measure
empirically.

### 3.2. New: trade-off Pareto bound

We add one new theorem to be proved during the paper write-up, giving
a Pareto bound on (LM loss, Dirichlet ratio):

**Theorem 5** (Pareto trade-off, sketch). *Let $\mathcal{L}(\theta)$
denote the LM loss and $\mathcal{R}(\theta) :=
\widetilde{\mathcal{R}}_X(H^{(\ell)}(\theta))$ the Dirichlet ratio at
layer $\ell$ as a function of model parameters $\theta$. Suppose both
are $C^1$ and Lipschitz on a compact parameter region. Define the
Pareto frontier $\mathcal{P} := \{(\mathcal{L}(\theta),
\mathcal{R}(\theta)) : \theta \text{ is Pareto-optimal}\}$. Then
training with $\mathcal{L} + \lambda \mathcal{R}$ for varying $\lambda
\geq 0$ traces out $\mathcal{P}$ continuously.*

This gives the user a principled way to choose $\lambda$: pick the
point on the Pareto frontier that achieves the LM loss within $\epsilon$
of unregularized training, with the smallest Dirichlet ratio.

### 3.3. Sample-complexity bound (using Theorem 4)

Theorem 4 gives a sample-complexity bound on subspace recovery as a
function of frame count $T$. This translates directly into a guarantee
on the *minimum* training data needed for the Dirichlet loss to
identify the correct subspace: training with $T < T^*$ frames per
scene cannot, in expectation, distinguish the true 3D subspace from
noise, so the Dirichlet loss will be uninformative below this
threshold.

This predicts: the training improvement from Dirichlet loss should be
larger when training data has more frames per scene, plateauing once
$T \gtrsim T^*$.

---

## 4. Experiment setup

### 4.1. Models

**Primary**: Qwen2.5-VL-3B (smallest of the Qwen-VL family with mature
HF support; 28 transformer layers; image+video).

**Secondary** (if compute permits): InternVL3-2B, Qwen2.5-VL-7B.

Reason for starting at 3B: small enough to LoRA-finetune on a single
80GB A100, large enough to be a real model. Once the pipeline works,
scale up.

### 4.2. Datasets

| Stage | Dataset | Purpose |
|---|---|---|
| Train | `tier_c_free6dof` (existing) | Synthetic 3D scenes with ground-truth coords; 200 scenes × ~5 objects × 32 temporal chunks |
| Eval (in-dist) | `tier_c_free6dof` held-out 50 scenes | Spatial VQA accuracy |
| Eval (transfer 1) | COCO-Spatial (Kamath et al. 2023) | Direct comparison to Kang et al.'s evaluation |
| Eval (transfer 2) | ARKitScenes camera-motion VQA | Real-world transfer |
| Probing | All extracted activations | Verify representational change |

Training samples are constructed as `(image, question, answer, X_object_coords)` tuples, where `X_object_coords` is the matrix of 3D coordinates for the objects mentioned in the question.

### 4.3. Training conditions

Four conditions, all using identical LoRA finetuning setups:

1. **Baseline**: standard LM cross-entropy loss.
2. **Kang-style**: LM + cosine-sim-to-extracted-IDs loss at layer L
   (Kang et al. 2026, §4.3 setup; replicated for our continuous 3D
   case using the natural extension).
3. **Dirichlet-light**: LM + 0.1 × Dirichlet ratio at layer L.
4. **Dirichlet-heavy**: LM + 1.0 × Dirichlet ratio at layer L.

Layer L is chosen as the empirical peak of residualized RSA in the
base model — for Qwen-2.5-VL-3B that's likely layer ~17–18 (≈ 65% depth).

### 4.4. Metrics

| Metric | What it tells us | Where measured |
|---|---|---|
| Training LM loss | Sanity check that LM still learns | Per step |
| Training Dirichlet ratio at L | The loss is doing its job | Per step |
| Held-out residualized RSA at L | Representational change is real | After training |
| Held-out Dirichlet ratio at L | Generalization of the regularization | After training |
| Held-out spatial VQA accuracy | Downstream payoff | After training |
| COCO-Spatial accuracy | Transfer to real images | After training |
| ARKitScenes accuracy | Transfer to real-world videos | After training |

### 4.5. Hyperparameter sweep

- $\lambda \in \{0, 0.01, 0.1, 0.3, 1.0, 3.0\}$ — to characterize the
  Pareto frontier of Theorem 5.
- Layer choice: single (L=peak), multi-layer with decay weights (e.g.,
  $\{0.5, 1.0, 0.5\}$ at $\{L-3, L, L+3\}$).
- Kernel bandwidth $\tau \in \{0.3, 1.0, 3.0\}$ in units of scene-scale
  standard deviation.
- LoRA rank ∈ {8, 16, 32}.

Each sweep point: 1 epoch on free6DoF (~4k steps), single A100. ≈ 1
hour per run. Total compute: ~30 hours for the full sweep.

### 4.6. Predictions (what we expect to see)

If the theory is right:

1. **Training Dirichlet ratio at L drops** monotonically as $\lambda$
   increases — the loss does its job.
2. **Residualized RSA at L rises** monotonically with $\lambda$ — the
   geometry actually aligns with world coords.
3. **VQA accuracy improves** for $\lambda \in [0.1, 1.0]$, then degrades
   for $\lambda > 1.0$ (over-regularization conflicts with LM
   objective; this is the Pareto trade-off of Theorem 5).
4. **Sweet-spot $\lambda$ is consistent** across model scales (a
   reassuring sign that this is a real effect, not a hyperparameter
   accident).

If the theory is wrong (or our model of the model is wrong):

- Training Dirichlet ratio drops but residualized RSA doesn't move →
  the loss optimizes the surface but not the underlying structure.
- VQA accuracy doesn't improve → 3D structure isn't actually used by
  the model for VQA. This would actually be an *interesting negative
  result* that informs the causal-ablation experiment.

### 4.7. Ablations

- **Dirichlet vs Kang-style** at matched compute. Direct head-to-head.
- **Dirichlet at L vs Dirichlet at random layer**. If the loss only
  helps at the empirical peak layer, that's evidence the peak is
  genuinely the "3D layer".
- **Dirichlet on shuffled 3D coords**. Should give *no improvement* —
  control for "any regularization helps".
- **Dirichlet with random-direction kernel** (instead of 3D-coord
  kernel). Same control.

### 4.8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Loss is too noisy and destabilizes training | Medium | Start with $\lambda = 0.01$, use gradient clipping, batch-normalize the loss |
| Dirichlet drops but VQA doesn't improve | Medium | Frame as ablation: shows 3D not causally used (counterpart to causal experiment) |
| Improvement is within run-to-run variance | Medium | Run each condition 3 times, report mean ± std |
| LoRA rank insufficient for Dirichlet to take effect | Low | Sweep rank up to 32; full-finetune comparison |
| Pretrained model already minimizes Dirichlet | Low | Empirically Dirichlet ratio is ≈ 0.92 (not 0); plenty of room |

---

## 5. Implementation

The implementation is split into two files:

- **[scripts/dirichlet_loss.py](../scripts/dirichlet_loss.py)** — pure-PyTorch loss module. Self-contained, has no dependencies beyond PyTorch.
- **[scripts/train_with_dirichlet.py](../scripts/train_with_dirichlet.py)** — training-script template using HuggingFace `Trainer`, with object-token-position extraction, forward hook at layer L, and combined LM+Dirichlet loss.

See those files for code; this section gives a high-level overview of
how they fit together.

### 5.1. Architecture

```
[VLM forward pass]
    ↓
[Forward hook at layer L]  →  H ∈ R^(B, seq, d)
    ↓
[Object-token extraction]  →  H_obj ∈ R^(B, n_obj, d)
    ↓
[Pair with X_obj from batch]  →  X_obj ∈ R^(B, n_obj, 3)
    ↓
[DirichletLoss(H_obj, X_obj)]  →  scalar
    ↓
[loss = LM_loss + λ · dirichlet_loss]
    ↓
[backward]
```

### 5.2. Object-token-position extraction

Per training sample, we need to know which token positions correspond
to which scene objects. This information is already in the
free6DoF dataset structure (each scene JSON has an objects list with
canonical names). The extraction logic mirrors what
`scripts/extract_activations.py` already does.

### 5.3. Loss computation

Given:
- `H_obj`: (B, n_obj, d) — object-token activations at layer L
- `X_obj`: (B, n_obj, 3) — corresponding 3D coordinates

Compute per-sample:
- $W_{ij} = \exp(-\|x_i - x_j\|^2 / 2\tau^2)$, with diagonal zeroed
- $D_{ij} = \|h_i - h_j\|^2$
- $\widetilde{\mathcal{R}} = \frac{\sum_{ij} W_{ij} D_{ij}}{\sum_{ij} D_{ij}}$

Then average across the batch. Total loss is `lm_loss + lambda *
dirichlet_loss.mean()`.

### 5.4. Training command

After implementation:

```
python scripts/train_with_dirichlet.py \
    --model qwen2.5-vl-3b \
    --layer 17 \
    --lambda 0.1 \
    --tau 1.0 \
    --train-data data/tier_c_free6dof \
    --epochs 1 \
    --output-dir checkpoints/dirichlet_lambda0.1
```

---

## 6. Timeline

| Week | Milestone |
|---|---|
| 1 | Implement `dirichlet_loss.py` + `train_with_dirichlet.py`; run baseline training (λ=0) and verify pipeline |
| 2 | First sweep over λ on Qwen-2.5-VL-3B |
| 3 | Ablations (random-3D control, layer-position control, Kang-style head-to-head) |
| 4 | Real-world transfer (COCO-Spatial, ARKitScenes); writeup |

Total: 4 weeks. Compute: ~80 A100-hours (well within a single
small-cluster allocation).

---

## 7. What this gets us in the paper

Section 6 of the planned paper ("Training with Dirichlet Loss")
becomes:

1. **Motivation** (½ page): the residualized-RSA peak is a
   *measurement*. Can we make it an *objective*?
2. **Method** (½ page): the loss formulation, derived from Theorem 3.
3. **Experiments** (1 page): the four-condition comparison, two
   transfer evals, λ sweep, ablations.
4. **Discussion** (½ page): when does Dirichlet loss win over Kang-style?
   When does either fail? (Probably: both fail when 3D is irrelevant
   to the question, and Dirichlet wins when the scene has continuous
   geometry that doesn't map to a discrete grid.)

Combined with the causal-ablation experiments (separate plan), this
lifts the paper from "we measured 3D structure" to "we measured it,
proved it, and trained better models with it" — which is the
ICLR-spotlight bar.
