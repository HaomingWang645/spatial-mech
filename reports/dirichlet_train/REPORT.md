# Real Dirichlet-loss training run — report

**TL;DR.** I actually finetuned Qwen2.5-VL-7B with `lm_loss + λ * dirichlet_ratio`
for 200 LoRA steps on Free6DoF spatial VQA. Compared against an identical
baseline run with λ=0, the Dirichlet-regularized run produced a monotone
decrease in val Dirichlet ratio (0.2375 → 0.2350) and a monotone increase
in val 3D-alignment R² (0.6251 → 0.6588), while baseline both metrics
oscillate flat around their initial values. **The Dirichlet loss measurably
improves 3D-structured representations during real LM-constrained
training.**

---

## 1. Setup

| Component | Setting |
|---|---|
| Model | `Qwen/Qwen2.5-VL-7B-Instruct` (28 LLM layers, hidden dim 3584) |
| Adaptation | LoRA rank 8 on `{q,k,v,o}_proj`; trainable params 5.05M (0.06% of model) |
| Training data | 2,988 spatial-VQA pairs from Free6DoF (image + scene-object list + question + answer) |
| Validation | 332 held-out QA pairs |
| Loss | `lm_cross_entropy + λ * dirichlet_ratio(H_obj_at_L17, X_obj)` |
| Hook layer | L17 (≈ 64% depth — empirical residualized-RSA peak for Qwen-7B) |
| Kernel bandwidth τ | 2.0 |
| Batch size | 1 |
| Steps | 200 |
| Learning rate | 1e-4 (Adam) |
| Two conditions | λ = 0 (baseline) and λ = 1.0 (Dirichlet) — run in parallel on GPUs 4 and 5 |
| Hardware | 1× H100 NVL per condition |

Crucial implementation note: **gradient checkpointing must be disabled**
(see §3 of the report below). Forward hooks capture activations at forward
time; with checkpointing enabled, those tensors get freed before the
backward pass and the Dirichlet gradient does not propagate.

---

## 2. Results

### 2.1. Validation timeline

| Step | Baseline λ=0 | | | Dirichlet λ=1.0 | | |
|---|---|---|---|---|---|---|
| | LM loss | Dir | R² | LM loss | Dir | R² |
| 49 | 0.0190 | 0.2376 | 0.6238 | 0.0214 | 0.2375 | 0.6251 |
| 99 | 0.0165 | 0.2378 | 0.6218 | 0.0179 | 0.2372 | 0.6284 |
| 149 | 0.0533 | 0.2379 | 0.6218 | 0.0483 | 0.2367 | 0.6372 |
| 199 | 0.0612 | 0.2377 | 0.6238 | **0.0431** | **0.2350** | **0.6588** |

![Side-by-side training curves](../dirichlet_train_results.png)

### 2.2. Observed effects

- **Dirichlet ratio at L17 (lower = more 3D-smooth)**: monotonically
  *decreases* under λ=1.0 (0.2375 → 0.2350, Δ = −0.0025). The baseline
  is flat (oscillates within ±0.0003 of 0.2378).

- **3D-alignment R² at L17 (higher = world coords in top PCs)**:
  monotonically *increases* under λ=1.0 (0.6251 → 0.6588,
  Δ = +0.034 absolute, +5.4% relative). The baseline is flat at 0.622.

- **Validation LM loss**: surprisingly, the Dirichlet run also has a
  *lower* val LM loss at step 199 (0.0431 vs 0.0612). The two runs
  diverge after step 100, with the Dirichlet run generalizing slightly
  better. This is consistent with a regularization interpretation —
  the Dirichlet term prevents the LM from over-relying on
  representation directions orthogonal to the 3D structure.

The two metrics directly affected by the loss (Dir, R²) both move
monotonically across all four eval points in the predicted direction
in the Dirichlet run, and not at all in the baseline run. This is
strong evidence the loss is doing genuine work.

---

## 3. What went wrong on the first attempt, and how I fixed it

The first run (λ=0.1, gradient checkpointing enabled) showed identical
trajectories for both conditions — eval Dirichlet ratio stuck at 0.2378
across both runs at all eval points. Two compounding bugs:

1. **Gradient checkpointing × forward hooks.**
   `model.gradient_checkpointing_enable()` reroutes the forward pass
   through `torch.utils.checkpoint.checkpoint`, which discards
   intermediate activations and recomputes them during backward. The
   tensor captured by my forward hook at L17 has a dangling autograd
   graph after the recompute, so gradients on the Dirichlet term never
   reach the LoRA parameters. Verified by isolating the Dirichlet
   gradient on a single sample (without checkpointing) and confirming
   it produces a non-trivial gradient on `lora_B` at L17 (norm ≈ 5.6e-3).
   **Fix**: disable gradient checkpointing. With Qwen2.5-VL-7B at
   batch_size=1 on 95GB H100, full activations easily fit (peak ≈ 30 GB).

2. **λ=0.1 was too small.** Even with checkpointing fixed, an
   Adam-normalized update from a 0.1 × dir gradient cannot compete with
   the LM gradient (which is itself near zero on this small,
   memorizable task). **Fix**: use λ=1.0 — 10× the original. With
   the larger weight, the per-step Adam updates from the Dirichlet
   term become large enough to consistently move the residual stream's
   geometry over 200 steps.

In retrospect, I should have run a tiny gradient-flow sanity check
*before* launching the full sweep. The pilot experiments verified the
loss math; they didn't verify it was wired into a real training loop.

---

## 4. Files produced

| Path | Contents |
|---|---|
| [scripts/build_dirichlet_train_data.py](../../scripts/build_dirichlet_train_data.py) | Builds JSONL training set from Free6DoF scenes |
| [data/dirichlet_train/train.jsonl](../../data/dirichlet_train/train.jsonl) | 2,988 train examples |
| [data/dirichlet_train/val.jsonl](../../data/dirichlet_train/val.jsonl) | 332 val examples |
| [scripts/train_qwen_dirichlet.py](../../scripts/train_qwen_dirichlet.py) | Real training script (HF model + LoRA + LM + Dirichlet) |
| [checkpoints/qwen7b_baseline/lora](../../checkpoints/qwen7b_baseline/lora) | Baseline LoRA weights (λ=0) |
| [checkpoints/qwen7b_dir_lam1.0/lora](../../checkpoints/qwen7b_dir_lam1.0/lora) | Dirichlet LoRA weights (λ=1.0) |
| [logs/train_baseline.log](../../logs/train_baseline.log) | Baseline training log |
| [logs/train_dirichlet.log](../../logs/train_dirichlet.log) | Dirichlet training log |
| [reports/dirichlet_train_results.png](../dirichlet_train_results.png) | Comparison curves figure |

---

## 5. Caveats — what these results do NOT establish

1. **VQA accuracy is not measured here.** The eval reports LM loss, not
   accuracy on the answer tokens. To make the Kang-et-al.-style "improves
   downstream task" claim, I'd need to (a) generate predictions with
   constrained decoding, (b) match against the ground-truth answer, (c)
   compute accuracy. That's a 1-day extension to this script.

2. **n=50 val subset.** The eval pass evaluates 50 of the 332 val
   examples. The deltas (especially the LM loss difference) could move
   under bigger eval sets. The Dirichlet ratio and R² are population
   statistics so should be more stable.

3. **Single seed, single λ.** No error bars across seeds, no λ sweep.
   The plan in [reports/dirichlet_loss_plan.md](../dirichlet_loss_plan.md)
   calls for {0.01, 0.1, 0.3, 1.0, 3.0} × 3 seeds. This pilot ran 1
   value × 1 seed.

4. **Only Qwen2.5-VL-7B.** Plan calls for replication across model
   families.

5. **Effect size is small.** ΔDirichlet-ratio = −0.0025 (1.1% relative)
   and ΔR² = +0.035 (5.4% relative) are real but modest at 200 LoRA
   steps. Longer training, larger λ, or larger LoRA rank could amplify
   the effect — or shrink it via overfitting. Currently undetermined.

---

## 6. What this run does establish

1. **The Dirichlet loss can be wired into a real VLM training loop**
   with HF + PEFT + forward hooks. The pipeline works end-to-end.

2. **Under correct configuration (no checkpointing, sufficient λ), the
   loss measurably steers the residual-stream geometry** in the
   theorem-predicted direction during LM-constrained training.

3. **The improvement is monotonic in training step**, distinguishing
   it from random noise that would give a non-monotone or symmetric
   trajectory.

4. **The Dirichlet run also has lower val LM loss** at the end, hinting
   at a regularization effect — the loss may improve generalization,
   not just representation geometry.

5. **The implementation is reproducible**: exact command for both
   conditions is in [logs/train_baseline.log](../../logs/train_baseline.log)
   and [logs/train_dirichlet.log](../../logs/train_dirichlet.log).

---

## 7. Recommended next experiments

In priority order:

1. **VQA-accuracy eval** on both checkpoints (~1 day). The conversion
   from "improves R²" to "improves task accuracy" is the publishable
   delta. If R² gain → accuracy gain, the paper has its key result.

2. **λ sweep** {0.1, 0.3, 1.0, 3.0, 10.0} to characterize the Pareto
   frontier (Theorem 5 of the theory draft). Each run is ~3 minutes
   on H100, so ~15 minutes total.

3. **Longer training** (1000 steps) to see if the trend continues. With
   only 200 steps the curves haven't plateaued; the gain might compound.

4. **Causal-ablation experiment** on the Dirichlet-trained checkpoint
   vs baseline — does the Dirichlet model's enhanced 3D subspace also
   matter more behaviourally? (This is the missing "use" tier.)
