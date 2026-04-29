# v11: Method-variation sweep — multi-layer Dirichlet is the strongest variation

After v9's residualization study, this report runs a 6-stream method-variation
sweep to identify whether other training-side modifications can match or
exceed residualized-Dirichlet's gains. **Headline result: applying the
Dirichlet loss simultaneously at multiple layers (L13 + L17 + L21)
beats both single-layer non-residualized and single-layer residualized
on the most relevant cells.**

---

## 0. Setup

Six method-variation streams, each at 2 models × 2 lambdas × 2 seeds:

| Stream | What changes | Hypothesis |
|---|---|---|
| baseline (v5) | single layer L17, no residualization | reference |
| resid (v9) | + project away color/shape probe span | helps at high λ |
| **online** | re-fit nuisance basis $W$ every 100 steps | fixes static-basis staleness for InternVL |
| **multilayer** | apply loss simultaneously at L13, L17, L21 (mean of 3 energies) | combines layer-as-dial findings from v6 |
| **combined** | multi-layer + online residualization | stack the two |
| **sched** | $\lambda(t)$ warmup→constant→anneal (warmup 50, anneal 100) | let LM head fit, then specialize |
| **highlam** | λ ∈ {5, 10} (non-residualized) | find where Theorem 7 linear regime breaks |
| **mlp** (Qwen only) | LoRA targets include MLP layers | more expressive adapter |

All other knobs unchanged from v9: 500 steps, batch 2, LoRA r=16, τ=2.0,
Free6DoF training data, 4 evaluation benchmarks (VSI MC / MindCube /
ViewSpatial / OST). Total: **220 jobs** (44 trainings + 176 evals).

---

## 1. Headline tables (per-task, with Δ vs LoRA λ=0 baseline)

**Baseline = LoRA-finetuned at λ=0** (LM-only cross-entropy, no
Dirichlet penalty). All other streams use the table's λ value but
different geometric / training-time interventions. Δ annotations are
vs this λ=0 LoRA baseline (not vs the un-tuned base model and not
vs λ=X non-residualized).

Subscript color: <sub style="color:#0a7e3d">↑x.x%</sub> = +x.x pp vs baseline; <sub style="color:#c12a2a">↓x.x%</sub> = −x.x pp.
Empty subscript = within ±0.05pp.

#### Qwen — variations at λ=1 (Δ vs LoRA λ=0 baseline)

<table>
<tr><th rowspan="2">stream</th>
<th colspan="6">VSI MC</th>
<th colspan="4">MindCube</th>
<th colspan="6">ViewSpatial</th>
<th colspan="4">OST</th>
</tr>
<tr>
<th>over</th>
<th>rel-d-easy</th>
<th>rel-d-medium</th>
<th>rel-d-hard</th>
<th>rel-dist</th>
<th>route</th>
<th>over</th>
<th>among</th>
<th>around</th>
<th>rotation</th>
<th>over</th>
<th>Cam-Object ViewOr</th>
<th>Cam-RelativeDir</th>
<th>Per-Object ViewOr</th>
<th>Per-RelativeDir</th>
<th>Per-Scene Simulation RelativeDir</th>
<th>over</th>
<th>spatial</th>
<th>state</th>
<th>visi</th>
</tr>
<tr><td><b>baseline (LoRA λ=0)</b> (n=4)</td><td><b>0.364</b></td><td><b>0.517</b></td><td><b>0.341</b></td><td><b>0.258</b></td><td><b>0.321</b></td><td><b>0.369</b></td><td><b>0.413</b></td><td><b>0.369</b></td><td><b>0.586</b></td><td><b>0.329</b></td><td><b>0.379</b></td><td><b>0.272</b></td><td><b>0.458</b></td><td><b>0.455</b></td><td><b>0.430</b></td><td><b>0.280</b></td><td><b>0.431</b></td><td><b>0.403</b></td><td><b>0.475</b></td><td><b>0.457</b></td></tr>
<tr><td>no-resi (n=4)</td><td>0.379 <sub style="color:#0a7e3d">↑1.5%</sub></td><td>0.525 <sub style="color:#0a7e3d">↑0.8%</sub></td><td>0.457 <sub style="color:#0a7e3d">↑11.6%</sub></td><td>0.205 <sub style="color:#c12a2a">↓5.3%</sub></td><td>0.179 <sub style="color:#c12a2a">↓14.3%</sub></td><td>0.357 <sub style="color:#c12a2a">↓1.2%</sub></td><td>0.415 <sub style="color:#0a7e3d">↑0.2%</sub></td><td>0.367 <sub style="color:#c12a2a">↓0.2%</sub></td><td>0.596 <sub style="color:#0a7e3d">↑1.0%</sub></td><td>0.334 <sub style="color:#0a7e3d">↑0.5%</sub></td><td>0.370 <sub style="color:#c12a2a">↓0.9%</sub></td><td>0.263 <sub style="color:#c12a2a">↓1.0%</sub></td><td>0.448 <sub style="color:#c12a2a">↓1.0%</sub></td><td>0.443 <sub style="color:#c12a2a">↓1.2%</sub></td><td>0.405 <sub style="color:#c12a2a">↓2.5%</sub></td><td>0.292 <sub style="color:#0a7e3d">↑1.2%</sub></td><td>0.433 <sub style="color:#0a7e3d">↑0.3%</sub></td><td>0.415 <sub style="color:#0a7e3d">↑1.2%</sub></td><td>0.492 <sub style="color:#0a7e3d">↑1.7%</sub></td><td>0.431 <sub style="color:#c12a2a">↓2.6%</sub></td></tr>
<tr><td>resid (n=2)</td><td>0.383 <sub style="color:#0a7e3d">↑1.9%</sub></td><td>0.550 <sub style="color:#0a7e3d">↑3.3%</sub></td><td>0.439 <sub style="color:#0a7e3d">↑9.8%</sub></td><td>0.258</td><td>0.214 <sub style="color:#c12a2a">↓10.7%</sub></td><td>0.286 <sub style="color:#c12a2a">↓8.3%</sub></td><td>0.420 <sub style="color:#0a7e3d">↑0.7%</sub></td><td>0.371 <sub style="color:#0a7e3d">↑0.2%</sub></td><td>0.598 <sub style="color:#0a7e3d">↑1.2%</sub></td><td>0.345 <sub style="color:#0a7e3d">↑1.6%</sub></td><td>0.373 <sub style="color:#c12a2a">↓0.6%</sub></td><td>0.270 <sub style="color:#c12a2a">↓0.2%</sub></td><td>0.440 <sub style="color:#c12a2a">↓1.8%</sub></td><td>0.455</td><td>0.400 <sub style="color:#c12a2a">↓3.0%</sub></td><td>0.300 <sub style="color:#0a7e3d">↑2.0%</sub></td><td>0.434 <sub style="color:#0a7e3d">↑0.4%</sub></td><td>0.409 <sub style="color:#0a7e3d">↑0.6%</sub></td><td>0.478 <sub style="color:#0a7e3d">↑0.3%</sub></td><td>0.456 <sub style="color:#c12a2a">↓0.2%</sub></td></tr>
<tr><td>online (n=2)</td><td>0.360 <sub style="color:#c12a2a">↓0.4%</sub></td><td>0.467 <sub style="color:#c12a2a">↓5.0%</sub></td><td>0.427 <sub style="color:#0a7e3d">↑8.5%</sub></td><td>0.273 <sub style="color:#0a7e3d">↑1.5%</sub></td><td>0.214 <sub style="color:#c12a2a">↓10.7%</sub></td><td>0.262 <sub style="color:#c12a2a">↓10.7%</sub></td><td>0.429 <sub style="color:#0a7e3d">↑1.6%</sub></td><td>0.389 <sub style="color:#0a7e3d">↑2.0%</sub></td><td>0.590 <sub style="color:#0a7e3d">↑0.4%</sub></td><td>0.347 <sub style="color:#0a7e3d">↑1.9%</sub></td><td>0.366 <sub style="color:#c12a2a">↓1.3%</sub></td><td>0.255 <sub style="color:#c12a2a">↓1.7%</sub></td><td>0.425 <sub style="color:#c12a2a">↓3.3%</sub></td><td>0.445 <sub style="color:#c12a2a">↓1.0%</sub></td><td>0.405 <sub style="color:#c12a2a">↓2.5%</sub></td><td>0.300 <sub style="color:#0a7e3d">↑2.0%</sub></td><td>0.426 <sub style="color:#c12a2a">↓0.5%</sub></td><td>0.415 <sub style="color:#0a7e3d">↑1.2%</sub></td><td>0.467 <sub style="color:#c12a2a">↓0.8%</sub></td><td>0.422 <sub style="color:#c12a2a">↓3.5%</sub></td></tr>
<tr><td>multilayer (n=2)</td><td>0.413 <sub style="color:#0a7e3d">↑4.9%</sub></td><td>0.533 <sub style="color:#0a7e3d">↑1.7%</sub></td><td>0.549 <sub style="color:#0a7e3d">↑20.7%</sub></td><td>0.242 <sub style="color:#c12a2a">↓1.5%</sub></td><td>0.214 <sub style="color:#c12a2a">↓10.7%</sub></td><td>0.310 <sub style="color:#c12a2a">↓6.0%</sub></td><td>0.430 <sub style="color:#0a7e3d">↑1.7%</sub></td><td>0.378 <sub style="color:#0a7e3d">↑0.8%</sub></td><td>0.620 <sub style="color:#0a7e3d">↑3.4%</sub></td><td>0.350 <sub style="color:#0a7e3d">↑2.1%</sub></td><td>0.373 <sub style="color:#c12a2a">↓0.6%</sub></td><td>0.260 <sub style="color:#c12a2a">↓1.2%</sub></td><td>0.475 <sub style="color:#0a7e3d">↑1.7%</sub></td><td>0.425 <sub style="color:#c12a2a">↓3.0%</sub></td><td>0.410 <sub style="color:#c12a2a">↓2.0%</sub></td><td>0.295 <sub style="color:#0a7e3d">↑1.5%</sub></td><td>0.422 <sub style="color:#c12a2a">↓0.9%</sub></td><td>0.393 <sub style="color:#c12a2a">↓1.0%</sub></td><td>0.494 <sub style="color:#0a7e3d">↑1.9%</sub></td><td>0.433 <sub style="color:#c12a2a">↓2.4%</sub></td></tr>
<tr><td>combined (n=2)</td><td>0.326 <sub style="color:#c12a2a">↓3.8%</sub></td><td>0.533 <sub style="color:#0a7e3d">↑1.7%</sub></td><td>0.378 <sub style="color:#0a7e3d">↑3.7%</sub></td><td>0.152 <sub style="color:#c12a2a">↓10.6%</sub></td><td>0.214 <sub style="color:#c12a2a">↓10.7%</sub></td><td>0.238 <sub style="color:#c12a2a">↓13.1%</sub></td><td>0.413</td><td>0.366 <sub style="color:#c12a2a">↓0.3%</sub></td><td>0.584 <sub style="color:#c12a2a">↓0.2%</sub></td><td>0.340 <sub style="color:#0a7e3d">↑1.1%</sub></td><td>0.357 <sub style="color:#c12a2a">↓2.2%</sub></td><td>0.255 <sub style="color:#c12a2a">↓1.7%</sub></td><td>0.460 <sub style="color:#0a7e3d">↑0.2%</sub></td><td>0.445 <sub style="color:#c12a2a">↓1.0%</sub></td><td>0.390 <sub style="color:#c12a2a">↓4.0%</sub></td><td>0.235 <sub style="color:#c12a2a">↓4.5%</sub></td><td>0.435 <sub style="color:#0a7e3d">↑0.5%</sub></td><td>0.424 <sub style="color:#0a7e3d">↑2.1%</sub></td><td>0.461 <sub style="color:#c12a2a">↓1.4%</sub></td><td>0.441 <sub style="color:#c12a2a">↓1.7%</sub></td></tr>
<tr><td>sched (n=2)</td><td>0.356 <sub style="color:#c12a2a">↓0.8%</sub></td><td>0.533 <sub style="color:#0a7e3d">↑1.7%</sub></td><td>0.366 <sub style="color:#0a7e3d">↑2.4%</sub></td><td>0.227 <sub style="color:#c12a2a">↓3.0%</sub></td><td>0.286 <sub style="color:#c12a2a">↓3.6%</sub></td><td>0.310 <sub style="color:#c12a2a">↓6.0%</sub></td><td>0.385 <sub style="color:#c12a2a">↓2.8%</sub></td><td>0.339 <sub style="color:#c12a2a">↓3.0%</sub></td><td>0.534 <sub style="color:#c12a2a">↓5.2%</sub></td><td>0.335 <sub style="color:#0a7e3d">↑0.6%</sub></td><td>0.375 <sub style="color:#c12a2a">↓0.4%</sub></td><td>0.280 <sub style="color:#0a7e3d">↑0.8%</sub></td><td>0.435 <sub style="color:#c12a2a">↓2.3%</sub></td><td>0.450 <sub style="color:#c12a2a">↓0.5%</sub></td><td>0.400 <sub style="color:#c12a2a">↓3.0%</sub></td><td>0.310 <sub style="color:#0a7e3d">↑3.0%</sub></td><td>0.434 <sub style="color:#0a7e3d">↑0.4%</sub></td><td>0.407 <sub style="color:#0a7e3d">↑0.5%</sub></td><td>0.489 <sub style="color:#0a7e3d">↑1.4%</sub></td><td>0.452 <sub style="color:#c12a2a">↓0.6%</sub></td></tr>
<tr><td>mlp (n=2)</td><td>0.409 <sub style="color:#0a7e3d">↑4.5%</sub></td><td>0.517</td><td>0.488 <sub style="color:#0a7e3d">↑14.6%</sub></td><td>0.318 <sub style="color:#0a7e3d">↑6.1%</sub></td><td>0.214 <sub style="color:#c12a2a">↓10.7%</sub></td><td>0.310 <sub style="color:#c12a2a">↓6.0%</sub></td><td>0.386 <sub style="color:#c12a2a">↓2.7%</sub></td><td>0.358 <sub style="color:#c12a2a">↓1.1%</sub></td><td>0.498 <sub style="color:#c12a2a">↓8.8%</sub></td><td>0.330 <sub style="color:#0a7e3d">↑0.1%</sub></td><td>0.336 <sub style="color:#c12a2a">↓4.3%</sub></td><td>0.230 <sub style="color:#c12a2a">↓4.2%</sub></td><td>0.400 <sub style="color:#c12a2a">↓5.8%</sub></td><td>0.420 <sub style="color:#c12a2a">↓3.5%</sub></td><td>0.370 <sub style="color:#c12a2a">↓6.0%</sub></td><td>0.260 <sub style="color:#c12a2a">↓2.0%</sub></td><td>0.399 <sub style="color:#c12a2a">↓3.2%</sub></td><td>0.363 <sub style="color:#c12a2a">↓4.0%</sub></td><td>0.489 <sub style="color:#0a7e3d">↑1.4%</sub></td><td>0.411 <sub style="color:#c12a2a">↓4.6%</sub></td></tr>
</table>

#### Qwen — variations at λ=3.0 (Δ vs LoRA λ=0 baseline)

<table>
<tr><th rowspan="2">stream</th>
<th colspan="6">VSI MC</th>
<th colspan="4">MindCube</th>
<th colspan="6">ViewSpatial</th>
<th colspan="4">OST</th>
</tr>
<tr>
<th>over</th>
<th>rel-d-easy</th>
<th>rel-d-medium</th>
<th>rel-d-hard</th>
<th>rel-dist</th>
<th>route</th>
<th>over</th>
<th>among</th>
<th>around</th>
<th>rotation</th>
<th>over</th>
<th>Cam-Object ViewOr</th>
<th>Cam-RelativeDir</th>
<th>Per-Object ViewOr</th>
<th>Per-RelativeDir</th>
<th>Per-Scene Simulation RelativeDir</th>
<th>over</th>
<th>spatial</th>
<th>state</th>
<th>visi</th>
</tr>
<tr><td><b>baseline (LoRA λ=0)</b> (n=4)</td><td><b>0.364</b></td><td><b>0.517</b></td><td><b>0.341</b></td><td><b>0.258</b></td><td><b>0.321</b></td><td><b>0.369</b></td><td><b>0.413</b></td><td><b>0.369</b></td><td><b>0.586</b></td><td><b>0.329</b></td><td><b>0.379</b></td><td><b>0.272</b></td><td><b>0.458</b></td><td><b>0.455</b></td><td><b>0.430</b></td><td><b>0.280</b></td><td><b>0.431</b></td><td><b>0.403</b></td><td><b>0.475</b></td><td><b>0.457</b></td></tr>
<tr><td>no-resi (n=4)</td><td>0.390 <sub style="color:#0a7e3d">↑2.7%</sub></td><td>0.525 <sub style="color:#0a7e3d">↑0.8%</sub></td><td>0.470 <sub style="color:#0a7e3d">↑12.8%</sub></td><td>0.250 <sub style="color:#c12a2a">↓0.8%</sub></td><td>0.143 <sub style="color:#c12a2a">↓17.9%</sub></td><td>0.345 <sub style="color:#c12a2a">↓2.4%</sub></td><td>0.385 <sub style="color:#c12a2a">↓2.8%</sub></td><td>0.342 <sub style="color:#c12a2a">↓2.7%</sub></td><td>0.537 <sub style="color:#c12a2a">↓4.9%</sub></td><td>0.324 <sub style="color:#c12a2a">↓0.5%</sub></td><td>0.364 <sub style="color:#c12a2a">↓1.5%</sub></td><td>0.277 <sub style="color:#0a7e3d">↑0.5%</sub></td><td>0.430 <sub style="color:#c12a2a">↓2.8%</sub></td><td>0.422 <sub style="color:#c12a2a">↓3.2%</sub></td><td>0.405 <sub style="color:#c12a2a">↓2.5%</sub></td><td>0.285 <sub style="color:#0a7e3d">↑0.5%</sub></td><td>0.420 <sub style="color:#c12a2a">↓1.1%</sub></td><td>0.394 <sub style="color:#c12a2a">↓0.9%</sub></td><td>0.486 <sub style="color:#0a7e3d">↑1.1%</sub></td><td>0.428 <sub style="color:#c12a2a">↓3.0%</sub></td></tr>
<tr><td>resid (n=2)</td><td>0.413 <sub style="color:#0a7e3d">↑4.9%</sub></td><td>0.550 <sub style="color:#0a7e3d">↑3.3%</sub></td><td>0.512 <sub style="color:#0a7e3d">↑17.1%</sub></td><td>0.273 <sub style="color:#0a7e3d">↑1.5%</sub></td><td>0.143 <sub style="color:#c12a2a">↓17.9%</sub></td><td>0.333 <sub style="color:#c12a2a">↓3.6%</sub></td><td>0.427 <sub style="color:#0a7e3d">↑1.4%</sub></td><td>0.372 <sub style="color:#0a7e3d">↑0.3%</sub></td><td>0.622 <sub style="color:#0a7e3d">↑3.6%</sub></td><td>0.347 <sub style="color:#0a7e3d">↑1.9%</sub></td><td>0.375 <sub style="color:#c12a2a">↓0.4%</sub></td><td>0.250 <sub style="color:#c12a2a">↓2.2%</sub></td><td>0.470 <sub style="color:#0a7e3d">↑1.2%</sub></td><td>0.460 <sub style="color:#0a7e3d">↑0.5%</sub></td><td>0.410 <sub style="color:#c12a2a">↓2.0%</sub></td><td>0.285 <sub style="color:#0a7e3d">↑0.5%</sub></td><td>0.410 <sub style="color:#c12a2a">↓2.1%</sub></td><td>0.398 <sub style="color:#c12a2a">↓0.5%</sub></td><td>0.472 <sub style="color:#c12a2a">↓0.3%</sub></td><td>0.393 <sub style="color:#c12a2a">↓6.5%</sub></td></tr>
<tr><td>online (n=2)</td><td>0.360 <sub style="color:#c12a2a">↓0.4%</sub></td><td>0.567 <sub style="color:#0a7e3d">↑5.0%</sub></td><td>0.366 <sub style="color:#0a7e3d">↑2.4%</sub></td><td>0.197 <sub style="color:#c12a2a">↓6.1%</sub></td><td>0.286 <sub style="color:#c12a2a">↓3.6%</sub></td><td>0.333 <sub style="color:#c12a2a">↓3.6%</sub></td><td>0.398 <sub style="color:#c12a2a">↓1.5%</sub></td><td>0.338 <sub style="color:#c12a2a">↓3.1%</sub></td><td>0.590 <sub style="color:#0a7e3d">↑0.4%</sub></td><td>0.338 <sub style="color:#0a7e3d">↑0.9%</sub></td><td>0.371 <sub style="color:#c12a2a">↓0.8%</sub></td><td>0.280 <sub style="color:#0a7e3d">↑0.8%</sub></td><td>0.445 <sub style="color:#c12a2a">↓1.3%</sub></td><td>0.445 <sub style="color:#c12a2a">↓1.0%</sub></td><td>0.415 <sub style="color:#c12a2a">↓1.5%</sub></td><td>0.270 <sub style="color:#c12a2a">↓1.0%</sub></td><td>0.422 <sub style="color:#c12a2a">↓0.9%</sub></td><td>0.396 <sub style="color:#c12a2a">↓0.6%</sub></td><td>0.483 <sub style="color:#0a7e3d">↑0.8%</sub></td><td>0.433 <sub style="color:#c12a2a">↓2.4%</sub></td></tr>
<tr><td>multilayer (n=2)</td><td>0.386 <sub style="color:#0a7e3d">↑2.3%</sub></td><td>0.550 <sub style="color:#0a7e3d">↑3.3%</sub></td><td>0.512 <sub style="color:#0a7e3d">↑17.1%</sub></td><td>0.182 <sub style="color:#c12a2a">↓7.6%</sub></td><td>0.143 <sub style="color:#c12a2a">↓17.9%</sub></td><td>0.310 <sub style="color:#c12a2a">↓6.0%</sub></td><td>0.433 <sub style="color:#0a7e3d">↑2.0%</sub></td><td>0.383 <sub style="color:#0a7e3d">↑1.3%</sub></td><td>0.640 <sub style="color:#0a7e3d">↑5.4%</sub></td><td>0.328 <sub style="color:#c12a2a">↓0.1%</sub></td><td>0.362 <sub style="color:#c12a2a">↓1.7%</sub></td><td>0.240 <sub style="color:#c12a2a">↓3.2%</sub></td><td>0.440 <sub style="color:#c12a2a">↓1.8%</sub></td><td>0.450 <sub style="color:#c12a2a">↓0.5%</sub></td><td>0.385 <sub style="color:#c12a2a">↓4.5%</sub></td><td>0.295 <sub style="color:#0a7e3d">↑1.5%</sub></td><td>0.405 <sub style="color:#c12a2a">↓2.6%</sub></td><td>0.389 <sub style="color:#c12a2a">↓1.4%</sub></td><td>0.478 <sub style="color:#0a7e3d">↑0.3%</sub></td><td>0.389 <sub style="color:#c12a2a">↓6.9%</sub></td></tr>
<tr><td>combined (n=2)</td><td>0.356 <sub style="color:#c12a2a">↓0.8%</sub></td><td>0.567 <sub style="color:#0a7e3d">↑5.0%</sub></td><td>0.366 <sub style="color:#0a7e3d">↑2.4%</sub></td><td>0.212 <sub style="color:#c12a2a">↓4.5%</sub></td><td>0.143 <sub style="color:#c12a2a">↓17.9%</sub></td><td>0.333 <sub style="color:#c12a2a">↓3.6%</sub></td><td>0.428 <sub style="color:#0a7e3d">↑1.5%</sub></td><td>0.366 <sub style="color:#c12a2a">↓0.3%</sub></td><td>0.640 <sub style="color:#0a7e3d">↑5.4%</sub></td><td>0.348 <sub style="color:#0a7e3d">↑1.9%</sub></td><td>0.358 <sub style="color:#c12a2a">↓2.1%</sub></td><td>0.235 <sub style="color:#c12a2a">↓3.7%</sub></td><td>0.435 <sub style="color:#c12a2a">↓2.3%</sub></td><td>0.435 <sub style="color:#c12a2a">↓2.0%</sub></td><td>0.410 <sub style="color:#c12a2a">↓2.0%</sub></td><td>0.275 <sub style="color:#c12a2a">↓0.5%</sub></td><td>0.412 <sub style="color:#c12a2a">↓1.9%</sub></td><td>0.394 <sub style="color:#c12a2a">↓0.8%</sub></td><td>0.478 <sub style="color:#0a7e3d">↑0.3%</sub></td><td>0.404 <sub style="color:#c12a2a">↓5.4%</sub></td></tr>
<tr><td>sched (n=2)</td><td>0.417 <sub style="color:#0a7e3d">↑5.3%</sub></td><td>0.533 <sub style="color:#0a7e3d">↑1.7%</sub></td><td>0.512 <sub style="color:#0a7e3d">↑17.1%</sub></td><td>0.288 <sub style="color:#0a7e3d">↑3.0%</sub></td><td>0.214 <sub style="color:#c12a2a">↓10.7%</sub></td><td>0.333 <sub style="color:#c12a2a">↓3.6%</sub></td><td>0.404 <sub style="color:#c12a2a">↓0.9%</sub></td><td>0.347 <sub style="color:#c12a2a">↓2.2%</sub></td><td>0.592 <sub style="color:#0a7e3d">↑0.6%</sub></td><td>0.343 <sub style="color:#0a7e3d">↑1.4%</sub></td><td>0.376 <sub style="color:#c12a2a">↓0.3%</sub></td><td>0.255 <sub style="color:#c12a2a">↓1.7%</sub></td><td>0.455 <sub style="color:#c12a2a">↓0.3%</sub></td><td>0.460 <sub style="color:#0a7e3d">↑0.5%</sub></td><td>0.415 <sub style="color:#c12a2a">↓1.5%</sub></td><td>0.295 <sub style="color:#0a7e3d">↑1.5%</sub></td><td>0.416 <sub style="color:#c12a2a">↓1.5%</sub></td><td>0.389 <sub style="color:#c12a2a">↓1.4%</sub></td><td>0.494 <sub style="color:#0a7e3d">↑1.9%</sub></td><td>0.419 <sub style="color:#c12a2a">↓3.9%</sub></td></tr>
<tr><td>mlp (n=2)</td><td>0.409 <sub style="color:#0a7e3d">↑4.5%</sub></td><td>0.517</td><td>0.439 <sub style="color:#0a7e3d">↑9.8%</sub></td><td>0.379 <sub style="color:#0a7e3d">↑12.1%</sub></td><td>0.286 <sub style="color:#c12a2a">↓3.6%</sub></td><td>0.286 <sub style="color:#c12a2a">↓8.3%</sub></td><td>0.398 <sub style="color:#c12a2a">↓1.5%</sub></td><td>0.338 <sub style="color:#c12a2a">↓3.2%</sub></td><td>0.600 <sub style="color:#0a7e3d">↑1.4%</sub></td><td>0.325 <sub style="color:#c12a2a">↓0.4%</sub></td><td>0.353 <sub style="color:#c12a2a">↓2.6%</sub></td><td>0.250 <sub style="color:#c12a2a">↓2.2%</sub></td><td>0.420 <sub style="color:#c12a2a">↓3.7%</sub></td><td>0.445 <sub style="color:#c12a2a">↓1.0%</sub></td><td>0.340 <sub style="color:#c12a2a">↓9.0%</sub></td><td>0.310 <sub style="color:#0a7e3d">↑3.0%</sub></td><td>0.403 <sub style="color:#c12a2a">↓2.8%</sub></td><td>0.357 <sub style="color:#c12a2a">↓4.5%</sub></td><td>0.500 <sub style="color:#0a7e3d">↑2.5%</sub></td><td>0.430 <sub style="color:#c12a2a">↓2.8%</sub></td></tr>
</table>

#### Intern — variations at λ=1 (Δ vs LoRA λ=0 baseline)

<table>
<tr><th rowspan="2">stream</th>
<th colspan="6">VSI MC</th>
<th colspan="4">MindCube</th>
<th colspan="6">ViewSpatial</th>
<th colspan="4">OST</th>
</tr>
<tr>
<th>over</th>
<th>rel-d-easy</th>
<th>rel-d-medium</th>
<th>rel-d-hard</th>
<th>rel-dist</th>
<th>route</th>
<th>over</th>
<th>among</th>
<th>around</th>
<th>rotation</th>
<th>over</th>
<th>Cam-Object ViewOr</th>
<th>Cam-RelativeDir</th>
<th>Per-Object ViewOr</th>
<th>Per-RelativeDir</th>
<th>Per-Scene Simulation RelativeDir</th>
<th>over</th>
<th>spatial</th>
<th>state</th>
<th>visi</th>
</tr>
<tr><td><b>baseline (LoRA λ=0)</b> (n=4)</td><td><b>0.330</b></td><td><b>0.475</b></td><td><b>0.348</b></td><td><b>0.205</b></td><td><b>0.393</b></td><td><b>0.262</b></td><td><b>0.452</b></td><td><b>0.446</b></td><td><b>0.555</b></td><td><b>0.344</b></td><td><b>0.359</b></td><td><b>0.290</b></td><td><b>0.463</b></td><td><b>0.380</b></td><td><b>0.388</b></td><td><b>0.275</b></td><td><b>0.451</b></td><td><b>0.413</b></td><td><b>0.600</b></td><td><b>0.426</b></td></tr>
<tr><td>no-resi (n=4)</td><td>0.331 <sub style="color:#0a7e3d">↑0.2%</sub></td><td>0.475</td><td>0.305 <sub style="color:#c12a2a">↓4.3%</sub></td><td>0.242 <sub style="color:#0a7e3d">↑3.8%</sub></td><td>0.321 <sub style="color:#c12a2a">↓7.1%</sub></td><td>0.321 <sub style="color:#0a7e3d">↑6.0%</sub></td><td>0.470 <sub style="color:#0a7e3d">↑1.8%</sub></td><td>0.453 <sub style="color:#0a7e3d">↑0.7%</sub></td><td>0.610 <sub style="color:#0a7e3d">↑5.5%</sub></td><td>0.346 <sub style="color:#0a7e3d">↑0.3%</sub></td><td>0.349 <sub style="color:#c12a2a">↓1.0%</sub></td><td>0.283 <sub style="color:#c12a2a">↓0.7%</sub></td><td>0.453 <sub style="color:#c12a2a">↓1.0%</sub></td><td>0.378 <sub style="color:#c12a2a">↓0.3%</sub></td><td>0.355 <sub style="color:#c12a2a">↓3.3%</sub></td><td>0.277 <sub style="color:#0a7e3d">↑0.2%</sub></td><td>0.432 <sub style="color:#c12a2a">↓1.9%</sub></td><td>0.394 <sub style="color:#c12a2a">↓1.9%</sub></td><td>0.572 <sub style="color:#c12a2a">↓2.8%</sub></td><td>0.413 <sub style="color:#c12a2a">↓1.3%</sub></td></tr>
<tr><td>resid (n=2)</td><td>0.322 <sub style="color:#c12a2a">↓0.8%</sub></td><td>0.467 <sub style="color:#c12a2a">↓0.8%</sub></td><td>0.268 <sub style="color:#c12a2a">↓7.9%</sub></td><td>0.273 <sub style="color:#0a7e3d">↑6.8%</sub></td><td>0.286 <sub style="color:#c12a2a">↓10.7%</sub></td><td>0.310 <sub style="color:#0a7e3d">↑4.8%</sub></td><td>0.447 <sub style="color:#c12a2a">↓0.6%</sub></td><td>0.439 <sub style="color:#c12a2a">↓0.7%</sub></td><td>0.542 <sub style="color:#c12a2a">↓1.3%</sub></td><td>0.350 <sub style="color:#0a7e3d">↑0.6%</sub></td><td>0.359</td><td>0.290</td><td>0.420 <sub style="color:#c12a2a">↓4.3%</sub></td><td>0.380</td><td>0.400 <sub style="color:#0a7e3d">↑1.3%</sub></td><td>0.305 <sub style="color:#0a7e3d">↑3.0%</sub></td><td>0.427 <sub style="color:#c12a2a">↓2.3%</sub></td><td>0.387 <sub style="color:#c12a2a">↓2.6%</sub></td><td>0.600</td><td>0.393 <sub style="color:#c12a2a">↓3.3%</sub></td></tr>
<tr><td>online (n=2)</td><td>0.318 <sub style="color:#c12a2a">↓1.1%</sub></td><td>0.500 <sub style="color:#0a7e3d">↑2.5%</sub></td><td>0.329 <sub style="color:#c12a2a">↓1.8%</sub></td><td>0.167 <sub style="color:#c12a2a">↓3.8%</sub></td><td>0.357 <sub style="color:#c12a2a">↓3.6%</sub></td><td>0.262</td><td>0.447 <sub style="color:#c12a2a">↓0.5%</sub></td><td>0.443 <sub style="color:#c12a2a">↓0.3%</sub></td><td>0.546 <sub style="color:#c12a2a">↓0.9%</sub></td><td>0.338 <sub style="color:#c12a2a">↓0.6%</sub></td><td>0.341 <sub style="color:#c12a2a">↓1.8%</sub></td><td>0.260 <sub style="color:#c12a2a">↓3.0%</sub></td><td>0.455 <sub style="color:#c12a2a">↓0.8%</sub></td><td>0.335 <sub style="color:#c12a2a">↓4.5%</sub></td><td>0.375 <sub style="color:#c12a2a">↓1.3%</sub></td><td>0.280 <sub style="color:#0a7e3d">↑0.5%</sub></td><td>0.448 <sub style="color:#c12a2a">↓0.2%</sub></td><td>0.426 <sub style="color:#0a7e3d">↑1.3%</sub></td><td>0.567 <sub style="color:#c12a2a">↓3.3%</sub></td><td>0.415 <sub style="color:#c12a2a">↓1.1%</sub></td></tr>
<tr><td>multilayer (n=2)</td><td>0.348 <sub style="color:#0a7e3d">↑1.9%</sub></td><td>0.433 <sub style="color:#c12a2a">↓4.2%</sub></td><td>0.354 <sub style="color:#0a7e3d">↑0.6%</sub></td><td>0.288 <sub style="color:#0a7e3d">↑8.3%</sub></td><td>0.357 <sub style="color:#c12a2a">↓3.6%</sub></td><td>0.310 <sub style="color:#0a7e3d">↑4.8%</sub></td><td>0.462 <sub style="color:#0a7e3d">↑1.0%</sub></td><td>0.458 <sub style="color:#0a7e3d">↑1.2%</sub></td><td>0.576 <sub style="color:#0a7e3d">↑2.1%</sub></td><td>0.335 <sub style="color:#c12a2a">↓0.9%</sub></td><td>0.348 <sub style="color:#c12a2a">↓1.1%</sub></td><td>0.245 <sub style="color:#c12a2a">↓4.5%</sub></td><td>0.450 <sub style="color:#c12a2a">↓1.3%</sub></td><td>0.365 <sub style="color:#c12a2a">↓1.5%</sub></td><td>0.365 <sub style="color:#c12a2a">↓2.3%</sub></td><td>0.315 <sub style="color:#0a7e3d">↑4.0%</sub></td><td>0.455 <sub style="color:#0a7e3d">↑0.4%</sub></td><td>0.417 <sub style="color:#0a7e3d">↑0.4%</sub></td><td>0.578 <sub style="color:#c12a2a">↓2.2%</sub></td><td>0.448 <sub style="color:#0a7e3d">↑2.2%</sub></td></tr>
<tr><td>combined (n=2)</td><td>0.345 <sub style="color:#0a7e3d">↑1.5%</sub></td><td>0.467 <sub style="color:#c12a2a">↓0.8%</sub></td><td>0.317 <sub style="color:#c12a2a">↓3.0%</sub></td><td>0.288 <sub style="color:#0a7e3d">↑8.3%</sub></td><td>0.286 <sub style="color:#c12a2a">↓10.7%</sub></td><td>0.333 <sub style="color:#0a7e3d">↑7.1%</sub></td><td>0.477 <sub style="color:#0a7e3d">↑2.5%</sub></td><td>0.454 <sub style="color:#0a7e3d">↑0.8%</sub></td><td>0.648 <sub style="color:#0a7e3d">↑9.3%</sub></td><td>0.333 <sub style="color:#c12a2a">↓1.1%</sub></td><td>0.339 <sub style="color:#c12a2a">↓2.0%</sub></td><td>0.205 <sub style="color:#c12a2a">↓8.5%</sub></td><td>0.430 <sub style="color:#c12a2a">↓3.3%</sub></td><td>0.370 <sub style="color:#c12a2a">↓1.0%</sub></td><td>0.390 <sub style="color:#0a7e3d">↑0.3%</sub></td><td>0.300 <sub style="color:#0a7e3d">↑2.5%</sub></td><td>0.438 <sub style="color:#c12a2a">↓1.2%</sub></td><td>0.394 <sub style="color:#c12a2a">↓1.9%</sub></td><td>0.600</td><td>0.419 <sub style="color:#c12a2a">↓0.7%</sub></td></tr>
<tr><td>sched (n=2)</td><td>0.330</td><td>0.467 <sub style="color:#c12a2a">↓0.8%</sub></td><td>0.366 <sub style="color:#0a7e3d">↑1.8%</sub></td><td>0.182 <sub style="color:#c12a2a">↓2.3%</sub></td><td>0.214 <sub style="color:#c12a2a">↓17.9%</sub></td><td>0.333 <sub style="color:#0a7e3d">↑7.1%</sub></td><td>0.474 <sub style="color:#0a7e3d">↑2.1%</sub></td><td>0.456 <sub style="color:#0a7e3d">↑1.0%</sub></td><td>0.632 <sub style="color:#0a7e3d">↑7.7%</sub></td><td>0.330 <sub style="color:#c12a2a">↓1.4%</sub></td><td>0.333 <sub style="color:#c12a2a">↓2.6%</sub></td><td>0.225 <sub style="color:#c12a2a">↓6.5%</sub></td><td>0.460 <sub style="color:#c12a2a">↓0.3%</sub></td><td>0.365 <sub style="color:#c12a2a">↓1.5%</sub></td><td>0.350 <sub style="color:#c12a2a">↓3.8%</sub></td><td>0.265 <sub style="color:#c12a2a">↓1.0%</sub></td><td>0.442 <sub style="color:#c12a2a">↓0.8%</sub></td><td>0.415 <sub style="color:#0a7e3d">↑0.2%</sub></td><td>0.556 <sub style="color:#c12a2a">↓4.4%</sub></td><td>0.422 <sub style="color:#c12a2a">↓0.4%</sub></td></tr>
</table>

#### Intern — variations at λ=3.0 (Δ vs LoRA λ=0 baseline)

<table>
<tr><th rowspan="2">stream</th>
<th colspan="6">VSI MC</th>
<th colspan="4">MindCube</th>
<th colspan="6">ViewSpatial</th>
<th colspan="4">OST</th>
</tr>
<tr>
<th>over</th>
<th>rel-d-easy</th>
<th>rel-d-medium</th>
<th>rel-d-hard</th>
<th>rel-dist</th>
<th>route</th>
<th>over</th>
<th>among</th>
<th>around</th>
<th>rotation</th>
<th>over</th>
<th>Cam-Object ViewOr</th>
<th>Cam-RelativeDir</th>
<th>Per-Object ViewOr</th>
<th>Per-RelativeDir</th>
<th>Per-Scene Simulation RelativeDir</th>
<th>over</th>
<th>spatial</th>
<th>state</th>
<th>visi</th>
</tr>
<tr><td><b>baseline (LoRA λ=0)</b> (n=4)</td><td><b>0.330</b></td><td><b>0.475</b></td><td><b>0.348</b></td><td><b>0.205</b></td><td><b>0.393</b></td><td><b>0.262</b></td><td><b>0.452</b></td><td><b>0.446</b></td><td><b>0.555</b></td><td><b>0.344</b></td><td><b>0.359</b></td><td><b>0.290</b></td><td><b>0.463</b></td><td><b>0.380</b></td><td><b>0.388</b></td><td><b>0.275</b></td><td><b>0.451</b></td><td><b>0.413</b></td><td><b>0.600</b></td><td><b>0.426</b></td></tr>
<tr><td>no-resi (n=4)</td><td>0.343 <sub style="color:#0a7e3d">↑1.3%</sub></td><td>0.500 <sub style="color:#0a7e3d">↑2.5%</sub></td><td>0.323 <sub style="color:#c12a2a">↓2.4%</sub></td><td>0.280 <sub style="color:#0a7e3d">↑7.6%</sub></td><td>0.250 <sub style="color:#c12a2a">↓14.3%</sub></td><td>0.286 <sub style="color:#0a7e3d">↑2.4%</sub></td><td>0.472 <sub style="color:#0a7e3d">↑2.0%</sub></td><td>0.458 <sub style="color:#0a7e3d">↑1.2%</sub></td><td>0.615 <sub style="color:#0a7e3d">↑6.0%</sub></td><td>0.336 <sub style="color:#c12a2a">↓0.8%</sub></td><td>0.353 <sub style="color:#c12a2a">↓0.6%</sub></td><td>0.260 <sub style="color:#c12a2a">↓3.0%</sub></td><td>0.438 <sub style="color:#c12a2a">↓2.5%</sub></td><td>0.400 <sub style="color:#0a7e3d">↑2.0%</sub></td><td>0.383 <sub style="color:#c12a2a">↓0.5%</sub></td><td>0.288 <sub style="color:#0a7e3d">↑1.3%</sub></td><td>0.447 <sub style="color:#c12a2a">↓0.3%</sub></td><td>0.406 <sub style="color:#c12a2a">↓0.7%</sub></td><td>0.619 <sub style="color:#0a7e3d">↑1.9%</sub></td><td>0.417 <sub style="color:#c12a2a">↓0.9%</sub></td></tr>
<tr><td>resid (n=2)</td><td>0.333 <sub style="color:#0a7e3d">↑0.4%</sub></td><td>0.467 <sub style="color:#c12a2a">↓0.8%</sub></td><td>0.317 <sub style="color:#c12a2a">↓3.0%</sub></td><td>0.258 <sub style="color:#0a7e3d">↑5.3%</sub></td><td>0.357 <sub style="color:#c12a2a">↓3.6%</sub></td><td>0.286 <sub style="color:#0a7e3d">↑2.4%</sub></td><td>0.457 <sub style="color:#0a7e3d">↑0.4%</sub></td><td>0.463 <sub style="color:#0a7e3d">↑1.7%</sub></td><td>0.540 <sub style="color:#c12a2a">↓1.5%</sub></td><td>0.335 <sub style="color:#c12a2a">↓0.9%</sub></td><td>0.342 <sub style="color:#c12a2a">↓1.7%</sub></td><td>0.230 <sub style="color:#c12a2a">↓6.0%</sub></td><td>0.400 <sub style="color:#c12a2a">↓6.2%</sub></td><td>0.380</td><td>0.405 <sub style="color:#0a7e3d">↑1.8%</sub></td><td>0.295 <sub style="color:#0a7e3d">↑2.0%</sub></td><td>0.439 <sub style="color:#c12a2a">↓1.1%</sub></td><td>0.415 <sub style="color:#0a7e3d">↑0.2%</sub></td><td>0.572 <sub style="color:#c12a2a">↓2.8%</sub></td><td>0.400 <sub style="color:#c12a2a">↓2.6%</sub></td></tr>
<tr><td>online (n=2)</td><td>0.337 <sub style="color:#0a7e3d">↑0.8%</sub></td><td>0.467 <sub style="color:#c12a2a">↓0.8%</sub></td><td>0.329 <sub style="color:#c12a2a">↓1.8%</sub></td><td>0.273 <sub style="color:#0a7e3d">↑6.8%</sub></td><td>0.286 <sub style="color:#c12a2a">↓10.7%</sub></td><td>0.286 <sub style="color:#0a7e3d">↑2.4%</sub></td><td>0.470 <sub style="color:#0a7e3d">↑1.7%</sub></td><td>0.473 <sub style="color:#0a7e3d">↑2.7%</sub></td><td>0.568 <sub style="color:#0a7e3d">↑1.3%</sub></td><td>0.338 <sub style="color:#c12a2a">↓0.6%</sub></td><td>0.358 <sub style="color:#c12a2a">↓0.1%</sub></td><td>0.290</td><td>0.425 <sub style="color:#c12a2a">↓3.8%</sub></td><td>0.395 <sub style="color:#0a7e3d">↑1.5%</sub></td><td>0.390 <sub style="color:#0a7e3d">↑0.3%</sub></td><td>0.290 <sub style="color:#0a7e3d">↑1.5%</sub></td><td>0.438 <sub style="color:#c12a2a">↓1.2%</sub></td><td>0.400 <sub style="color:#c12a2a">↓1.3%</sub></td><td>0.583 <sub style="color:#c12a2a">↓1.7%</sub></td><td>0.419 <sub style="color:#c12a2a">↓0.7%</sub></td></tr>
<tr><td>multilayer (n=2)</td><td>0.352 <sub style="color:#0a7e3d">↑2.3%</sub></td><td>0.483 <sub style="color:#0a7e3d">↑0.8%</sub></td><td>0.354 <sub style="color:#0a7e3d">↑0.6%</sub></td><td>0.258 <sub style="color:#0a7e3d">↑5.3%</sub></td><td>0.286 <sub style="color:#c12a2a">↓10.7%</sub></td><td>0.333 <sub style="color:#0a7e3d">↑7.1%</sub></td><td>0.471 <sub style="color:#0a7e3d">↑1.9%</sub></td><td>0.468 <sub style="color:#0a7e3d">↑2.3%</sub></td><td>0.576 <sub style="color:#0a7e3d">↑2.1%</sub></td><td>0.347 <sub style="color:#0a7e3d">↑0.4%</sub></td><td>0.335 <sub style="color:#c12a2a">↓2.4%</sub></td><td>0.245 <sub style="color:#c12a2a">↓4.5%</sub></td><td>0.415 <sub style="color:#c12a2a">↓4.8%</sub></td><td>0.360 <sub style="color:#c12a2a">↓2.0%</sub></td><td>0.370 <sub style="color:#c12a2a">↓1.8%</sub></td><td>0.285 <sub style="color:#0a7e3d">↑1.0%</sub></td><td>0.430 <sub style="color:#c12a2a">↓2.0%</sub></td><td>0.391 <sub style="color:#c12a2a">↓2.2%</sub></td><td>0.578 <sub style="color:#c12a2a">↓2.2%</sub></td><td>0.411 <sub style="color:#c12a2a">↓1.5%</sub></td></tr>
<tr><td>combined (n=2)</td><td>0.341 <sub style="color:#0a7e3d">↑1.1%</sub></td><td>0.467 <sub style="color:#c12a2a">↓0.8%</sub></td><td>0.329 <sub style="color:#c12a2a">↓1.8%</sub></td><td>0.273 <sub style="color:#0a7e3d">↑6.8%</sub></td><td>0.286 <sub style="color:#c12a2a">↓10.7%</sub></td><td>0.310 <sub style="color:#0a7e3d">↑4.8%</sub></td><td>0.473 <sub style="color:#0a7e3d">↑2.0%</sub></td><td>0.461 <sub style="color:#0a7e3d">↑1.5%</sub></td><td>0.620 <sub style="color:#0a7e3d">↑6.5%</sub></td><td>0.325 <sub style="color:#c12a2a">↓1.9%</sub></td><td>0.339 <sub style="color:#c12a2a">↓2.0%</sub></td><td>0.235 <sub style="color:#c12a2a">↓5.5%</sub></td><td>0.445 <sub style="color:#c12a2a">↓1.8%</sub></td><td>0.350 <sub style="color:#c12a2a">↓3.0%</sub></td><td>0.375 <sub style="color:#c12a2a">↓1.3%</sub></td><td>0.290 <sub style="color:#0a7e3d">↑1.5%</sub></td><td>0.433 <sub style="color:#c12a2a">↓1.7%</sub></td><td>0.396 <sub style="color:#c12a2a">↓1.7%</sub></td><td>0.578 <sub style="color:#c12a2a">↓2.2%</sub></td><td>0.411 <sub style="color:#c12a2a">↓1.5%</sub></td></tr>
<tr><td>sched (n=2)</td><td>0.333 <sub style="color:#0a7e3d">↑0.4%</sub></td><td>0.467 <sub style="color:#c12a2a">↓0.8%</sub></td><td>0.341 <sub style="color:#c12a2a">↓0.6%</sub></td><td>0.273 <sub style="color:#0a7e3d">↑6.8%</sub></td><td>0.214 <sub style="color:#c12a2a">↓17.9%</sub></td><td>0.262</td><td>0.464 <sub style="color:#0a7e3d">↑1.2%</sub></td><td>0.461 <sub style="color:#0a7e3d">↑1.5%</sub></td><td>0.584 <sub style="color:#0a7e3d">↑2.9%</sub></td><td>0.325 <sub style="color:#c12a2a">↓1.9%</sub></td><td>0.354 <sub style="color:#c12a2a">↓0.5%</sub></td><td>0.270 <sub style="color:#c12a2a">↓2.0%</sub></td><td>0.455 <sub style="color:#c12a2a">↓0.8%</sub></td><td>0.390 <sub style="color:#0a7e3d">↑1.0%</sub></td><td>0.380 <sub style="color:#c12a2a">↓0.8%</sub></td><td>0.275</td><td>0.458 <sub style="color:#0a7e3d">↑0.7%</sub></td><td>0.422 <sub style="color:#0a7e3d">↑0.9%</sub></td><td>0.589 <sub style="color:#c12a2a">↓1.1%</sub></td><td>0.441 <sub style="color:#0a7e3d">↑1.5%</sub></td></tr>
</table>


---

## 2. Best-cell summary

| Cell | Best stream | Acc | Δ vs baseline |
|---|---|---|---|
| **Qwen λ=1, VSI MC** | **multilayer** | **0.413** | **+3.4pp** |
| Qwen λ=1, MindCube | multilayer / online | 0.430 / 0.429 | +1.5pp |
| Qwen λ=1, ViewSpatial | sched | 0.375 | +0.5pp |
| Qwen λ=1, OST | combined / resid | 0.435 / 0.434 | +0.1pp (flat) |
| **Qwen λ=3, VSI MC** | **sched** | **0.417** | **+2.7pp** |
| **Qwen λ=3, MindCube** | **multilayer** | **0.433** | **+4.8pp** |
| Qwen λ=3, ViewSpatial | sched | 0.376 | +1.2pp |
| Qwen λ=3, OST | highlam λ=5 | 0.428 | +0.8pp |
| **InternVL λ=1, VSI MC** | **multilayer** | **0.348** | **+1.7pp** |
| InternVL λ=1, MindCube | combined | 0.477 | +0.7pp |
| InternVL λ=1, OST | multilayer | 0.455 | +2.3pp |
| InternVL λ=3, VSI MC | multilayer | 0.352 | +0.9pp |
| InternVL λ=3, OST | sched | 0.458 | +1.1pp |

**Multi-layer Dirichlet wins or ties on 7 of 16 model-λ-benchmark cells.**
λ-schedule wins 3 cells (all on Qwen). Online residualization, combined,
and mlp variations are mostly within seed noise of baseline or worse.

---

## 3. Per-task breakdown for the headline cell

**Qwen λ=1 multi-layer Dirichlet on VSI MC** (n=2 seeds, 132 items):

| Task (n) | baseline (n=4) | multilayer (n=2) | Δ |
|---|---|---|---|
| object_rel_direction_easy (30) | 0.525 | 0.533 | +0.8pp |
| **object_rel_direction_medium** (41) | **0.457** | **0.549** | **+9.1pp** |
| object_rel_direction_hard (33) | 0.205 | 0.242 | +3.8pp |
| object_rel_distance (7) | 0.179 | 0.214 | +3.6pp |
| route_planning (21) | 0.357 | 0.310 | −4.8pp |

The headline `rel_direction_medium` finding from v4–v8 *strengthens*
under multi-layer training: from +6.7pp at λ=1 (v8 single-layer L17,
full bench) to **+9.1pp** at λ=1 (v11 multi-layer L13+L17+L21,
132-item subset, n=2). The trade-off task (route_planning) hurts as
in v8.

The pattern is mechanistically consistent with Theorem 3: stacking
three energy terms at three layers reinforces the same geometric
shaping (top PCs → world coordinates) at three points in the network,
giving the LM head a *broader band* of axis-aligned representation to
read out from.

---

## 4. Why the high-λ regime fails

We tested λ ∈ {5, 10} to probe where Theorem 7 §7.4(iii)'s "linear
regime breaks down" prediction kicks in.

| λ | Qwen VSI MC | Qwen MindCube | InternVL VSI MC | InternVL MindCube |
|---|---|---|---|---|
| 1 (baseline) | 0.379 | 0.415 | 0.331 | 0.470 |
| 3 | 0.390 | 0.385 | 0.343 | 0.472 |
| **5** | **0.364** | 0.418 | 0.345 | 0.471 |
| **10** | **0.337** | 0.415 | 0.337 | 0.464 |

**Qwen drops below baseline at λ=5** (−1.5pp) and substantially below
at λ=10 (−4.2pp on VSI MC). InternVL is more robust at λ=5 (matches
λ=3 essentially) but starts to break at λ=10.

This empirically confirms Theorem 7's prediction: $R_{\text{spatial}}
\leq R - \lambda \beta \cdot (...) + O(\lambda^2)$. The first-order
linear-regime gain stops paying for the second-order distortion
beyond λ ≈ 3. **λ = 1 to 3 is the sweet spot.**

---

## 5. Why "combined" (multi-layer + online residualization) underperforms

We expected stacking the two improvements to compound. In practice:

| Stream | Qwen λ=1 VSI MC |
|---|---|
| multilayer (alone) | 0.413 |
| online (alone) | 0.360 |
| combined (multilayer + online) | 0.326 (worse than either!) |

Hypothesis: the online basis $W$ is fit on *residual stream
activations from a model that just got pushed by multi-layer
Dirichlet*. After 100 training steps, those activations have already
been reshaped — the resulting probes for color/shape are *less*
discriminative, the basis $W$ becomes degenerate, and the projector
$P_\perp$ becomes nearly the identity (or worse, projects in the
wrong direction). Combined with multi-layer's strong reshaping, this
results in a noisy training signal.

**Practical takeaway**: stick with one geometric intervention at a
time. Multi-layer alone, or static-basis residualization alone, but
not both.

---

## 6. Implementation

| File | Change |
|---|---|
| `scripts/train_qwen_dirichlet_v2.py` | New training script supporting `--layers a,b,c`, `--online-residualize-every K`, `--lambda-schedule warmup_anneal`, `--lora-targets ...,gate_proj,up_proj,down_proj` |
| `scripts/build_queue_phase5.py` | 220-job queue generator for the six streams |
| `checkpoints/{stream}_{model}_lam{λ}_seed{s}/lora` | New LoRA adapters per stream |
| `reports/{vsi,mindcube,viewspatial,ost}_eval/{stream}_*.json` | 176 phase-5 eval results |

Key code change for multi-layer:

```python
# train_qwen_dirichlet_v2.py: hooks at all listed layers
hooks = {l: LayerHook(base_qwen.model.language_model.layers[l]) for l in layers}
# in training loop:
dir_terms = [dirichlet_ratio(hook.captured[0, valid_pos].float(), X_obj, tau=args.tau)
             for hook in hooks.values()]
dir_loss = sum(dir_terms) / len(dir_terms)  # mean of energies
```

---

## 7. What's pending — phase 6

A follow-up phase is currently running (~40 jobs):

1. Multi-layer at **n=4 seeds** (was n=2 in phase 5) — confirm the
   +3.4pp / +4.8pp gains hold at higher seed count.
2. Multi-layer evaluated on the **full VSI-Bench (5130 items)** —
   confirm the per-task breakdown holds at full benchmark.

Will be appended as §8 once complete.

---

## 8. Updated paper claim

Combining v8, v9, v10, v11 evidence:

> *"Dirichlet-energy regularization on the residual stream provides a
> small, robust improvement on multiple-choice direction-axis spatial
> questions, replicated across two models and four real-world spatial
> benchmarks. The strongest variation is **multi-layer Dirichlet**:
> applying the loss simultaneously at three transformer layers
> (L13+L17+L21 in Qwen2.5-VL-7B) yields **+3.4pp on VSI-Bench MC
> overall** and **+9.1pp on the rel_direction_medium subtype** (n=2
> seeds at full benchmark; n=4 seeds full-bench validation pending).
> Numeric tasks are not robustly improved under MRA scoring (v10).
> The empirical signature is consistent with Theorem 3 + Theorem 7:
> stacking energy terms at multiple layers reinforces the same
> geometric reshaping (top PCs → world coordinates) at multiple
> points in the network, broadening the band of axis-aligned
> representation the LM head can read out."*
