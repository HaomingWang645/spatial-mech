# Why is there a significant base → λ=0 improvement? Are the v8 numbers an artifact?

A close reading of the v8 numbers (especially Qwen's `object_abs_distance` going from 0.000 → 0.315 and InternVL's going from 0.464 → 0.972) raises a concern that the v8 headline gains may be partly a **scoring-methodology artifact** rather than spatial-reasoning improvement. This document lays out the concern, what's likely real, what's likely artifact, and what experiment would settle it.

---

## What "λ=0" actually does

Set the Dirichlet weight $\lambda = 0$, but **LoRA training still runs for 500 steps with LM cross-entropy** on the Free6DoF synthetic dataset. So "λ=0" is *not* the base model — it is

> base model + 500 steps of LM-only LoRA on synthetic spatial data, **no geometric regularizer**.

The base → λ=0 gap is therefore the contribution of *LM-only synthetic-data finetuning*, with the Dirichlet loss multiplied by zero.

## What is almost certainly real

| Effect | Plausible mechanism | Evidence in v8 |
|---|---|---|
| Format calibration | Free6DoF answers are bare letters ("A", "B") and bare scalars in meters — the same format VSI-Bench expects. After LoRA training, the model emits answers in this format; the base model often emits prose. | The +9.8pp Qwen overall gain is consistent with most of the lift coming from this. |
| Question grounding | Free6DoF has questions phrased like *"what is the distance between X and Y?"* — same syntax as VSI-Bench. The model learns the register. | Same. |
| Numeric range calibration | Free6DoF distances are in 0.5–10 m, room sizes in 5–60 m². VSI-Bench's GT distributions overlap. After LoRA, the model emits numbers in the right magnitude bracket. | The 0% → 30% jumps on Qwen `abs_distance` and `room_size` are explained almost entirely by this. |

This is **honest LM-only transfer learning**. But notice: it is *not* learning spatial reasoning — it is learning the *answer-format distribution* that downstream scoring rewards.

## What is likely a scoring-methodology artifact

Look at these specific cells of the v8 table:

| Task | base | λ=0 |
|---|---|---|
| Qwen `object_abs_distance` | **0.000** | 0.315 |
| Qwen `room_size_estimation` | **0.000** | 0.292 |
| InternVL `object_abs_distance` | 0.464 | **0.785** |

Two red flags:

### Flag 1 — A 0.0% baseline on a 4-way scoring is suspicious

Random chance is 25%. Getting *exactly 0%* on a 4-candidate task with $n = 834$ items means the model is *systematically* picking AGAINST the GT. That requires a consistent, deterministic preference for one specific distractor.

The numeric scoring in `scripts/eval_vsi_batched.py` (and `eval_vsi.py`) builds distractors via [eval_vsi.py:33](scripts/eval_vsi.py#L33):

```python
perturbations = [v * 0.5, v * 1.5, v * 2.0, v * 0.25]
```

For GT $= 4$, the candidate strings are `"4"`, `"2"`, `"6"`, `"8"`. The model picks whichever has highest mean log-prob. *Base Qwen* likely systematically favours one of these distractors — most likely the smallest token (`"1"`, `"2"`) because short, frequent token sequences have higher unconditional log-prob in the absence of strong context. So before any finetuning, the scoring rule effectively measures "does the model emit the smallest plausible number?" — to which the answer is "yes, almost always," and the GT (rarely the smallest) loses.

After 500 LM-only LoRA steps on Free6DoF, the model becomes *calibrated to the right magnitude range* and starts preferring the GT-magnitude answer. The 0% → 30% jump is therefore a **format/calibration shift**, not "learning to estimate distance."

### Flag 2 — InternVL's 97.2% at λ=3 is too clean

`object_abs_distance` rises 78.5% → 97.0% (λ=0.3) → 87.9% (λ=1) → 97.2% (λ=3). Going from "decently calibrated" (the LoRA-only baseline) to "picks GT 809/834 times" via the Dirichlet axis is a 19pp lift on a numeric task. Possible explanations:

1. **Real:** Dirichlet really does refine the metric-distance encoding (Theorem 5: sample-complexity reduction in the metric subspace).
2. **Artifact:** distractor-based scoring saturates for any model whose numeric output distribution is centered on the GT magnitude with low variance — once the model is well-calibrated, it consistently beats $\{0.5\times, 1.5\times, 2\times\}$ distractors regardless of the actual answer being right.

We cannot distinguish (1) from (2) from the existing data. **A 97% accuracy under distractor-ranking scoring is consistent with a model that always emits a number close to the GT magnitude even when its actual point-prediction is wildly wrong**, because the perturbed distractors are all *farther* from the GT in log-space.

## What this means for the v8 conclusions

| Claim | Status |
|---|---|
| Qwen `rel_direction_medium` +6.7pp at λ=1 (over LoRA-only) | **Real.** MC-letter scoring; model picks among "front-left", "back-right", etc. — no numeric range issue. |
| Qwen base → λ=0 overall +9.8pp | **Mostly format adaptation**, not spatial reasoning. |
| InternVL `abs_distance` 46.4% → 97.2% (base → λ=3) | **Suspect.** Could be real metric-encoding gain, but could also be distractor-scoring saturation. |
| InternVL `room_size` 32.6% → 70.1% (base → λ=3) | **Suspect** for same reason. |
| InternVL overall +9.4pp (base → λ=3) | **Partly real** (some MC tasks improve), partly suspect (driven by numeric tasks). |

The general principle: **the v4–v7 reports were right to emphasize λ=0 vs λ=3 comparisons rather than base vs λ=3.** v8's choice to report base columns is informative but introduces a confounder for any reader who naively reads "+50pp on abs_distance."

## How to settle it: free-form generation evaluation

The clean test is to evaluate numeric questions by **free-form generation** (no distractors, no log-prob ranking). The standard VSI-Bench metric for numeric questions is **MRA** (Mean Relative Accuracy):

$$
\mathrm{MRA}(\hat y, y) \;=\; \frac{1}{|\Theta|} \sum_{\theta \in \Theta} \mathbb{1}\!\left[\frac{|\hat y - y|}{\max(|\hat y|, |y|)} \leq \theta\right],
$$

where $\Theta = \{0.5, 0.4, 0.3, 0.2, 0.1, 0.05\}$ — the prediction is "correct" at threshold $\theta$ if its relative error is at most $\theta$. The final MRA score averages accuracy over the six thresholds.

Under this protocol:

- The model generates a number directly — no comparison to perturbed distractors.
- A model that always outputs "1.0" gets near-zero MRA on a benchmark with diverse GTs.
- A model that emits the *correct* GT magnitude is rewarded at every threshold.
- Distractor-scoring saturation cannot inflate the score.

If InternVL at λ=3 still gets MRA ≥ 0.7 on `abs_distance` under generation-eval, the v8 finding is real. If MRA drops to ≤ 0.4, the 97% in v8 was largely a scoring artifact.

A generation-eval re-run on the most-suspect cells (Qwen + InternVL × base + λ=0 + λ=3, on the four numeric tasks) is what's needed.
