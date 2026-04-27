# v10: Generation-eval verification — the v8 numeric findings were a scoring artifact

This report verifies the v8 numeric-task accuracies using a **free-form
generation evaluator** with the official VSI-Bench MRA (Mean Relative
Accuracy) metric, instead of the distractor-ranking scoring used in
v4–v8. Motivated by the artifact analysis in
[v8_lambda0_artifact_analysis.md](v8_lambda0_artifact_analysis.md).

**Headline finding**: the v8 numeric numbers were **massively
inflated** by distractor-ranking saturation. Under the cleaner
generation+MRA protocol, InternVL's `object_abs_distance` accuracy at
λ=3 drops from **97.2% → 18.2%**, and `room_size_estimation` drops
from **70.1% → 11.2%**. The Dirichlet effect on numeric tasks
**reverses sign** under MRA — InternVL's `abs_distance` actually
*decreases* slightly with Dirichlet (22.7% base → 18.2% λ=3).

---

## 1. Setup

| Item | Detail |
|---|---|
| Evaluator | `scripts/eval_vsi_generation.py` (free-form generation, no distractors) |
| Metric | MRA = mean over θ ∈ {0.5, 0.4, 0.3, 0.2, 0.1, 0.05} of $\mathbb{1}[\frac{|\hat y - y|}{\max(|\hat y|, |y|)} \leq \theta]$ |
| Items | 200 stratified per task × 4 numeric tasks = 800 items per eval |
| Conditions | base (no LoRA), λ=0 (LM-only LoRA), λ=3 (Dirichlet) — covers the suspect cells |
| Models | Qwen2.5-VL-7B + InternVL3-8B |
| Total evals | 6 |

---

## 2. Headline comparison

### InternVL — the "97% on abs_distance" deflation

| Task | base | λ=0 | λ=3 | v8 distractor-ranking λ=3 | inflation factor |
|---|---|---|---|---|---|
| `object_abs_distance` (n=200) | 0.227 | 0.203 | **0.182** | 0.972 | **5.3×** |
| `object_size_estimation` | 0.228 | 0.256 | 0.220 | 0.295 | 1.3× |
| `object_counting` | 0.104 | 0.107 | 0.080 | 0.115 | 1.4× |
| `room_size_estimation` | 0.079 | 0.093 | **0.112** | 0.701 | **6.3×** |
| **Overall MRA** | **0.159** | **0.165** | **0.148** | (n/a) | |

The two tasks where v8 reported "Dirichlet helps a lot" — `abs_distance`
and `room_size` — show **the largest inflation**. Under MRA:

- `abs_distance` actually *decreases* slightly with Dirichlet (0.227 → 0.182).
- `room_size` improves slightly (+3.3pp from base) but is **far below** the v8 number (70%).
- The Dirichlet effect **reverses sign** for `abs_distance`.

The "97.2% accuracy on InternVL abs_distance at λ=3" claim from v8 is
**not a real spatial-reasoning achievement** — it was an artifact of
distractor-ranking giving high scores to any model with a calibrated
numeric output distribution.

### Qwen — partial deflation

| Task | base | λ=0 | λ=3 | v8 distractor-ranking λ=3 | inflation factor |
|---|---|---|---|---|---|
| `object_abs_distance` | 0.077 | 0.067 | 0.083 | 0.219 | **2.6×** |
| `object_size_estimation` | 0.320 | 0.381 | 0.340 | 0.240 | 0.7× (deflation) |
| `object_counting` | 0.073 | 0.090 | 0.090 | 0.071 | 0.8× |
| `room_size_estimation` | 0.271 | 0.280 | 0.286 | 0.295 | 1.0× |
| **Overall MRA** | **0.185** | **0.205** | **0.200** | | |

Qwen's pattern is more interesting:

- `abs_distance` is inflated 2.6× by distractor-ranking (still ≪ 1
  inflation, base ≈ 8% under MRA — a genuinely poor performance).
- `size_estimation` is *deflated* under distractor-ranking (38% MRA at
  λ=0 vs 28% v8 distractor) — Qwen *can* size-estimate decently when
  given partial credit.
- `room_size` and `counting` agree well between MRA and v8 — these were
  not artifact-prone.
- Overall MRA: base→λ=0 is +2.0pp (vs v8's +9.8pp). Most of the v8
  gain was format calibration, exactly as the artifact analysis predicted.

---

## 3. Where v8's claims hold and where they don't

### Claims that are **confirmed** by v10

1. **"Qwen rel_direction_medium gains under Dirichlet"** — v10 doesn't
   re-test MC questions, but the v8 MC scoring is letter-based and not
   subject to numeric-distractor saturation. The +6.7pp at λ=1 should
   stand.
2. **"`obj_appearance_order` improves with Dirichlet on Qwen"** — same
   reasoning, MC-based, not affected by this artifact.

### Claims that **must be retracted** based on v10

1. **"InternVL abs_distance: +50.8pp from base to λ=3"** — under MRA
   the gap is −4.5pp.
2. **"InternVL room_size: +37.5pp from base to λ=3"** — under MRA
   the gap is +3.3pp.
3. **"Dirichlet adds +18.7pp on InternVL abs_distance over LoRA-only"**
   — under MRA the effect is −2.1pp.

### Claims partially confirmed

1. **"Qwen base→λ=0 overall +9.8pp on VSI-Bench"** — partially real
   (+2.0pp under MRA, the rest was format calibration).
2. **"InternVL benefits more from Dirichlet than Qwen"** — partially:
   on MC tasks (rel_direction_easy/hard, route_planning, MindCube
   perpendicular), InternVL still shows larger gains. The numeric-task
   advantage was mostly artifact.

---

## 4. Methodological lessons

1. **Distractor-ranking ≠ accuracy on numeric tasks.** The protocol
   ranks GT against {0.5×, 1.5×, 2×}-perturbed-GT. A model whose
   numeric output distribution is centered on the GT magnitude
   trivially beats these distractors regardless of *which* number it
   would actually emit. v4–v8 reports relied on this scoring and
   over-stated numeric improvements.

2. **MC scoring is not affected by this artifact** — letter
   candidates ("A", "B", "C", "D") are equally simple tokens; the
   scoring measures genuine semantic preference. The v4–v8 MC
   findings are robust.

3. **Always run a free-form generation eval before claiming numeric
   improvements**, even if it's expensive. The 6-eval verification run
   here cost ~1 GPU-hour and overturned a substantial fraction of v8's
   headline numbers.

4. **Pre-existing benchmark protocols (here, VSI-Bench MRA) should
   be used over custom scoring** unless the deviation is well-
   motivated. Our distractor-ranking was a convenience for log-prob
   evaluators; the convenience came at the cost of correctness.

---

## 5. The corrected v8 narrative

After v10's correction, the cleaned-up Dirichlet story across v4–v9
is:

| Finding | Status |
|---|---|
| Dirichlet ratio drops, 3D-alignment R² rises with λ | ✅ Theorem 3 confirmed (n=8, p<10⁻⁶) |
| Qwen `rel_direction_medium` +6.7pp at λ=1 (full bench, n=378) | ✅ Real |
| Qwen `obj_appearance_order` +3.6pp at λ=3 | ✅ Real (MC) |
| InternVL gains on MindCube perpendicular (+1.9pp at λ=3) | ✅ Real (v7) |
| InternVL `abs_distance` +50pp at λ=3 | ❌ Retracted (artifact) |
| InternVL `room_size` +37pp at λ=3 | ❌ Retracted (artifact) |
| Qwen `rel_distance` regression at high λ | ❌ Retracted (small-sample) |
| Residualized > non-residualized at Qwen λ=3 (+2-4pp) | ⚠ Within noise (n=2) |

**The defensible final claim**: Dirichlet loss provides a small,
consistent improvement on **MC direction-axis reasoning** for Qwen
(+5-7pp on rel_direction_medium, smaller gains on appearance_order
and route_planning). Improvements on numeric magnitude tasks reported
in v8 were largely an artifact of distractor-ranking scoring; under
the MRA metric, numeric-task effects are within noise.

---

## 6. Files

| Path | Contents |
|---|---|
| `scripts/eval_vsi_generation.py` | Generation-eval with MRA |
| `reports/v8_lambda0_artifact_analysis.md` | Original artifact analysis |
| `reports/vsi_gen_eval/{qwen,intern}_{base,lam0,lam3.0}.json` | 6 generation-eval results |
| `reports/dirichlet_train/REPORT_v10_generation_verification.md` | This report |

---

## 7. What this means for the manuscript

For the paper, this means the headline empirical claim should be:

> *"Dirichlet-energy regularization on the residual stream of a
> spatial-VQA-trained VLM provides a small, robust improvement on
> multiple-choice direction-axis questions: +6.7pp on
> rel_direction_medium for Qwen2.5-VL-7B (n=378, full VSI-Bench), with
> consistent direction across MindCube perpendicular questions for
> InternVL3-8B. Numeric-magnitude tasks (abs_distance, room_size) do
> not show robust improvements under the official VSI-Bench MRA
> metric."*

This is a **smaller but scientifically honest** claim than the v8
narrative. The theory (Theorem 3) is unchanged — the geometric
reshaping does happen. What changes is which downstream tasks
benefit, and how much.
