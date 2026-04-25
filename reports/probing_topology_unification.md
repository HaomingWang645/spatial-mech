# Connecting linear probing with PCA-based topology

## TL;DR

Linear probing for camera motion / depth and PCA-based scene-topology
analysis (RSA, Dirichlet ratio) are not in tension and do not produce
"different conclusions" — they are **testing the same object at different
strictness levels**.  Specifically, both certify the existence of a
3D-aligned linear subspace $\mathcal{S}_X(\ell)$ in the residual stream
at layer $\ell$; they differ only in *what they assert about that
subspace*.

| Claim | Method | Strictness | What we observe empirically |
|---|---|---|---|
| **Existence** of $\mathcal{S}_X$ | Linear probe (camera motion, depth) | Weakest | Almost always true, even at early layers |
| **Primacy** of $\mathcal{S}_X$ — it's in the top PCs | PCA / RSA / Dirichlet ratio | Mid | True only at peak layers (~65–70% relative depth) |
| **Use** of $\mathcal{S}_X$ for downstream behaviour | Causal ablation | Strongest | Open — currently missing experiment |

---

## 1. Setup and definitions

Let $X \in \mathbb{R}^{n \times 3}$ stack the ground-truth 3D coordinates
of $n$ object-tokens in a scene, and $H \in \mathbb{R}^{n \times d}$ the
corresponding residual-stream activations at layer $\ell$.  Define the
**3D-aligned subspace at layer $\ell$**:

$$
\mathcal{S}_X(\ell) \;:=\; \arg\min_{\mathcal{V} \subset \mathbb{R}^d,\,\dim\mathcal{V}=3}\, \|X - H \, P_\mathcal{V}^{-1} \, \pi_\mathcal{V}\|_F^2,
$$

i.e., the 3-dimensional subspace of $\mathbb{R}^d$ from which $H$ best
linearly predicts $X$ (here $\pi_\mathcal{V}$ is the projection map and
$P_\mathcal{V}^{-1}$ is the corresponding pseudo-inverse for the
prediction).  This is the canonical "3D probe" subspace that both our
methods care about.

Two empirical methods are used to characterize $\mathcal{S}_X(\ell)$:

- **Linear probing.**  Train a regularized linear regressor
  $\hat W = \arg\min_W \|HW - X\|_F^2 + \alpha \|W\|_F^2$, report
  $R^2$ or VQA accuracy of the resulting predictor.  This tells you
  *whether* $\mathcal{S}_X(\ell)$ exists with non-trivial dimension and
  signal-to-noise ratio.

- **PCA / RSA / Dirichlet ratio.**  Compute the top-$k$ principal
  components of $H$, then measure their alignment with the 3D
  geometry — via Pearson correlation of pairwise distance matrices
  (RSA) or via the Dirichlet ratio of $H$ against the scene-induced
  graph Laplacian.  This tells you whether $\mathcal{S}_X(\ell)$ is
  among the *top* singular subspaces of $H$ — i.e., whether it
  *dominates* the layer's geometry rather than living in low-variance
  directions.

These two methods describe the same object from opposite sides of the
SVD; this document explains the connection.

---

## 2. The mathematical bridge (Theorem 1 of [theory_draft.md](theory_draft.md))

Both methods reduce to a singular-value comparison.  Decompose
$H = XA + E$ where $A \in \mathbb{R}^{3 \times d}$ is the linear probe
of $\mathcal{S}_X(\ell)$ with row-norms $\beta_1 \geq \beta_2 \geq \beta_3 > 0$,
and $E$ is residual.  Then:

- **Linear probe** at rank 3 measures the smallest probe gain $\beta_3$
  scaled by the smallest world-coordinate spread $\sigma_3(X)$, relative
  to the noise: it succeeds iff $\beta_3 \sigma_3(X) \gg \|E\|_\mathrm{op}$.

- **PCA / RSA topology** at the top-3 PC level measures $\sin\Theta(\mathcal{U}_3(H), \mathcal{U}_3(X))$, which by Davis–Kahan is bounded by $\|E\|_\mathrm{op} / (\beta_3 \sigma_3(X))$.

Both are functions of the same ratio $\beta_3 \sigma_3(X) / \|E\|_\mathrm{op}$.
Theorem 1 of [theory_draft.md](theory_draft.md) makes this precise:

$$
\sin\Theta\bigl(\mathcal{U}_3(H), \mathcal{U}_3(X)\bigr) \;\leq\; \frac{2\|E\|_\mathrm{op}}{\beta_3 \sigma_3(X)}.
$$

When the linear probe succeeds at rank 3, the right-hand side is small
— so PCA succeeds too.  When PCA succeeds, the converse: the top-3
singular subspace of $H$ aligns with $\mathcal{U}_3(X)$, so the linear
probe is well-conditioned.  **They are the same statement.**

---

## 3. Where the methods can disagree

The two methods can produce ostensibly different conclusions in two
scenarios:

### 3.1 Linear probe high, PCA / RSA low

**What this means.**  $\mathcal{S}_X(\ell)$ exists in $H$ but lives in
*low-variance* directions — say PCs 47–52 instead of 1–3.  The linear
probe is permissive (any rank-3 subspace will do); PCA only inspects
the top.

**When we see this.**  At early or late VLM layers.  Linear probes for
depth and camera motion show non-trivial $R^2$ across most layers;
RSA / Dirichlet ratio show a sharp peak narrowly at ~65–70% depth.
The interpretation is that 3D *information* is present everywhere but
3D *geometry* dominates the representation only at peak layers — these
are the layers where the VLM is "thinking about 3D structure" rather
than merely encoding it.

**This is not a contradiction.**  It's the difference between
"information is somewhere" and "information is *primary*".

### 3.2 Linear probe low, PCA / RSA high

**What this would mean.**  Top-3 PCs of $H$ have a 3D-like geometry,
but no rank-3 linear probe predicts $X$ accurately.  This requires a
strongly *non-linear* probe (rank-1 features that combine
multiplicatively) or a measurement artefact.

**When we see this.**  Essentially never in the existing analyses —
this case would be theoretically interesting but isn't empirically
present.

---

## 4. Residualization closes a real loophole

A standard linear probe for, say, camera motion can succeed by relying
on a *depth shortcut*: if depth alone is decodable, and depth is
correlated with the camera-motion target on this dataset, the probe
passes.  This is Q3 of the analysis report and the motivation for the
depth-shortcut baseline.

**Residualized RSA fixes this loophole at the representation level.**
By Theorem 2 of [theory_draft.md](theory_draft.md), residualized RSA
equals $\rho(\Pi_{C^\perp} H, X)$ where $C$ is the confound matrix
(depth, frame-index, scene dummies).  By Corollary 2.1, $\rho > 0$
certifies that $H$ has a 3D-aligned component *orthogonal to all linear
combinations of the confounds*.

**This makes residualized RSA strictly more informative than either
unconditional linear probing or unconditional RSA**: it is a primacy
test on the *non-shortcut subspace*.  When residualized RSA peaks
sharply at ~65–70% depth and depth-shortcut probing is also non-zero
across many layers, the right interpretation is:

- depth is everywhere → reachable by trivial shortcuts at any layer
- 3D *geometry* (orthogonal to depth) emerges narrowly at peak layers
- the peak layer is where the VLM is genuinely composing depth into a
  3D scene representation, not merely passing depth forward

---

## 5. Three-tier framework for the paper

The paper should organize the empirical evidence into the three-tier
hierarchy below.  Each tier is a strictly stronger claim than the one
above; demonstrating all three is the "definitive 3D-mechanism" claim
that lifts the paper above measurement-only analyses.

```
                                  Causal ablation
                                  ↑  USE
                                  ↑  (strongest)
                                  ↑
                          PCA / RSA / Dirichlet ratio
                                  ↑  PRIMACY
                                  ↑
                        Linear probing (camera motion, depth)
                                  ↑  EXISTENCE
                                  ↑  (weakest)
```

| Tier | Empirical evidence | Status in current report | Where to find |
|---|---|---|---|
| Existence | Linear probes for camera motion, depth, depth-shortcut | ✅ done | [reports/tier_c_free6dof_camera_motion_5models.md](tier_c_free6dof_camera_motion_5models.md), [reports/tier_c_free6dof_depth_shortcut_5models.md](tier_c_free6dof_depth_shortcut_5models.md) |
| Primacy | Residualized RSA, Dirichlet ratio across 5 models × 4 frame counts | ✅ done | [reports/tier_c_topology_option3.md](tier_c_topology_option3.md) §5.8–5.13 |
| Use | Causal ablation of $\mathcal{S}_X(\ell)$ → spatial-VQA accuracy drop | ❌ missing | TBD |

---

## 6. Suggested unifying paragraph (paper-ready)

The following is a single drop-in paragraph that can sit at the start
of the empirical section, framing all subsequent results.

> Linear probing certifies that camera motion and per-object depth are
> *decodable* from VLM residual streams — establishing the existence
> of a 3D-aligned linear subspace $\mathcal{S}_X(\ell)$ at every layer.
> PCA-based topology probes are the corresponding *primacy* test: they
> ask whether $\mathcal{S}_X(\ell)$ lies among the *top* principal
> directions of the representation, i.e., whether 3D structure
> *dominates* the residual stream's geometry rather than living in a
> low-variance subspace.  By Theorem 1, both tests measure the same
> signal-to-noise ratio $\beta_3 \sigma_3(X) / \|E\|_\mathrm{op}$, at
> different strictness levels.  Residualized RSA strengthens both by
> first projecting out the depth-shortcut subspace (Theorem 2),
> certifying that 3D structure is encoded *beyond* what any linear
> combination of depth, frame-index, and scene-identity could
> explain.  Combined, the two methods give a coherent picture: 3D
> information exists everywhere in the network, but 3D *geometry*
> (free of shortcuts) dominates only in a narrow band of mid-late
> layers.  The remaining open question — whether the model actually
> *uses* this geometry for downstream reasoning — we address in
> §[causal-ablation section] via direct subspace ablation.

---

## 7. References

- Theorems 1, 2 are from [reports/theory_draft.md](theory_draft.md).
- Davis, C. & Kahan, W. M. (1970). *The rotation of eigenvectors by a
  perturbation, III.*  SIAM J. Numer. Anal. 7(1), 1–46.
- Yu, Y., Wang, T., & Samworth, R. J. (2015). *A useful variant of the
  Davis–Kahan theorem for statisticians.*  Biometrika 102(2), 315–323.
- Frisch, R. & Waugh, F. V. (1933). *Partial time regressions as
  compared with individual trends.*  Econometrica 1(4), 387–401.
  [The classical statement of the residualization-as-projection
  identity used in Theorem 2.]
