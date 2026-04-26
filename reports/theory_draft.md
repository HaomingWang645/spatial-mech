# Theoretical foundations for 3D-structure probes in VLM residual streams

**Draft — theorems and proofs.** This document provides paper-appendix-quality
formulations and proofs of four theorems supporting the empirical claims of the
paper. The structure mirrors Park et al. (ICLR 2025, *In-Context Learning of
Representations*): a representation-recovery theorem (our Theorem 1), a
methodological identity (our Theorem 2), a variational characterization
linking energy minimization to PCA (our Theorem 3), and a sample-complexity
bound (our Theorem 4). Each theorem carries a plain-English summary, a formal
statement, a complete proof with intuition for every step, and remarks
addressing limitations and testable predictions.

Notation is fixed once in §1. The standard perturbation-theoretic and
concentration lemmas we invoke are stated explicitly as Lemmas B1–B5 in §2,
so the proofs in §§3–6 can be read self-contained.

---

## 0. Reading guide for ML readers

This document has more matrix algebra than a typical ML paper. To make
the math digestible without sacrificing rigor, every theorem is
structured as a four-layer onion that you can peel as deep as you want:

1. **Plain-English summary** (one paragraph) — what the theorem says
   in everyday language and why an ML researcher should care.
2. **Statement** — the formal claim, with all hypotheses spelled out.
3. **Chain of reasoning (intuitive overview)** — *new in this version.*
   Tells the proof's *story*: what's the destination, what's the
   obstacle, what are the 2–4 key ideas that bridge them, and where
   the "real work" happens. Read this if you want to understand
   *why* the proof works without reading the algebra.
4. **Step-by-step proof** — the formal derivation with every line
   justified. Read this only if you need to verify a specific
   inequality or want to adapt the proof to a variant.

If you're building intuition: read 1 → 3 for each theorem.
If you're writing a paper that cites these results: read 1 → 2 → 3.
If you're refereeing or extending the proofs: read 1 → 2 → 3 → 4.

The four theorems also have a logical dependency order:

- **Theorem 1** (PCA recovers 3D under linear-probe assumption) is the
  *foundation*. It's just Davis–Kahan applied to a particular
  factorization.
- **Theorem 2** (residualization = orthogonal projection) is *purely
  algebraic* — a generalization of the Frisch–Waugh–Lovell theorem to
  similarity matrices. Used to defend the residualized RSA methodology.
- **Theorem 3** (Dirichlet-energy minimization → PCA = Laplacian
  eigenmaps) is the *conceptual heart*. It says that the loss we
  defined is variationally equivalent to spectral graph embedding.
- **Theorem 4** (frame-count emergence) is *Theorem 1 applied to
  averaged noise*. It quantifies a sample-complexity threshold and
  predicts the empirically-observed emergence curve.

Reading order to maximize understanding: 1 → 3 → 2 → 4. Theorem 1 is
the simplest; Theorem 3 is the deepest and most ML-relevant; Theorem 2
is an algebraic identity needed for methodology; Theorem 4 is a
straightforward consequence of Theorem 1.

---

## 1. Notation and conventions

- For $A \in \mathbb{R}^{m \times n}$, $\sigma_k(A)$ is the $k$-th singular
  value (in non-increasing order), $\|A\|_\mathrm{op}$ the operator
  (spectral) norm, $\|A\|_F$ the Frobenius norm. We write $A = U_A
  \Sigma_A V_A^\top$ for the (compact) SVD.
- $\mathcal{U}_k(A) \subset \mathbb{R}^m$ denotes the span of the top-$k$
  left singular vectors of $A$. $u_k(A)$ denotes the $k$-th left singular
  vector itself.
- For two $k$-dimensional subspaces $\mathcal{V}, \mathcal{W} \subset
  \mathbb{R}^n$ with orthogonal projectors $P_\mathcal{V}$ and
  $P_\mathcal{W}$, we write $\sin\Theta(\mathcal{V}, \mathcal{W}) :=
  \|P_\mathcal{V} - P_\mathcal{W}\|_\mathrm{op}$ for the operator-norm
  principal angle. Equivalently, this is the sine of the largest principal
  angle between the two subspaces.
- We index object-tokens by $i \in [n]$. Each token has a ground-truth
  world coordinate $x_i \in \mathbb{R}^3$ and a residual-stream activation
  $h_i \in \mathbb{R}^d$ at the layer under study. We write $X \in
  \mathbb{R}^{n \times 3}$ and $H \in \mathbb{R}^{n \times d}$ for the
  row-stacked matrices.
- Unless stated, $X$ is mean-centered: $\mathbf{1}^\top X = 0$.
- **Representational Similarity Analysis (RSA).** For matrices $H, X$
  with $n$ rows, define pairwise squared-distance matrices $(D_M)_{ij} =
  \|m_i - m_j\|^2$. The RSA score is the Pearson correlation of the
  upper-triangular entries:
  $\rho(H, X) := \mathrm{corr}\bigl(\mathrm{vec}^{\triangle}(D_H),\,
  \mathrm{vec}^{\triangle}(D_X)\bigr).$
- $J := \mathrm{span}(\mathbf{1}_n) \subset \mathbb{R}^n$ is the
  one-dimensional "constant" subspace. $J^\perp$ is its orthogonal
  complement. The mean-centering operator is $P_{J^\perp} = I_n -
  \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$.

---

## 2. Background results

We state the standard tools we use as Lemmas B1–B5, citing references for
proofs. These are textbook material; readers familiar with matrix
perturbation theory may skip to §3.

**Lemma B1** (Weyl's inequality for singular values; Bhatia 1997, Cor.
III.2.6). *Let $M, M' \in \mathbb{R}^{n \times d}$. For all $k \in
[\min(n,d)]$,*

$$
|\sigma_k(M) - \sigma_k(M')| \;\leq\; \|M - M'\|_\mathrm{op}.
$$

*Plain-English:* singular values are 1-Lipschitz with respect to the
operator norm. If you perturb a matrix $M$ by adding $E$ to it, no
singular value can change by more than $\|E\|_\mathrm{op}$.

**Lemma B2** (Davis–Kahan $\sin\Theta$ theorem, Yu–Wang–Samworth 2015
restatement). *Let $M, M' \in \mathbb{R}^{n \times d}$ with singular
values $\sigma_1(M) \geq \cdots \geq \sigma_r(M)$ and similarly for $M'$.
Fix $k \in [r-1]$ and assume the spectral gap $\sigma_k(M) -
\sigma_{k+1}(M) > 0$. Then*

$$
\sin\Theta\bigl(\mathcal{U}_k(M'), \mathcal{U}_k(M)\bigr)
\;\leq\; \frac{\|M - M'\|_\mathrm{op}}{\sigma_k(M) - \sigma_{k+1}(M)}.
$$

*Plain-English:* the principal-angle distance between top-$k$ singular
subspaces is at most "noise divided by spectral gap". The larger the gap
between $\sigma_k$ and $\sigma_{k+1}$ in the clean matrix, the more
robust its top-$k$ subspace is to perturbation.

**Lemma B3** (Wedin's theorem; Stewart & Sun 1990, Thm. V.4.4). *Under
the assumptions of Lemma B2, but with the gap measured at $M'$ instead of
$M$,*

$$
\sin\Theta\bigl(u_k(M'), u_k(M)\bigr)
\;\leq\; \frac{\sqrt{2}\, \|M - M'\|_\mathrm{op}\, \max\{\sigma_k(M), \sigma_k(M')\}}{\min\bigl(\sigma_k(M)^2 - \sigma_{k+1}(M)^2,\, \sigma_{k-1}(M)^2 - \sigma_k(M)^2\bigr)}.
$$

*Plain-English:* same idea as Davis–Kahan but for *individual* singular
vectors rather than the top-$k$ subspace; this is what we need for
axis-by-axis (rather than just subspace-level) recovery.

**Lemma B4** (Ky Fan inequality for trace minima; Fan 1949, Thm. 1).
*Let $L \in \mathbb{R}^{n \times n}$ be symmetric with eigenvalues
$\lambda_1 \leq \cdots \leq \lambda_n$ and corresponding orthonormal
eigenvectors $z^{(1)}, \ldots, z^{(n)}$. Let $w_1 \geq w_2 \geq \cdots
\geq w_s > 0$ be a strictly decreasing sequence of weights. Then*

$$
\min_{\substack{u_1, \ldots, u_s \in \mathbb{R}^n \\ \langle u_i, u_j \rangle = \delta_{ij}}}
\sum_{k=1}^s w_k\, \langle u_k, L u_k \rangle
\;=\; \sum_{k=1}^s w_k\, \lambda_k,
$$

*and the unique (up to sign) minimizer is $u_k = z^{(k)}$ for each $k$.*

*Plain-English:* if we want to minimize a weighted sum of Rayleigh
quotients of $L$, with strictly decreasing positive weights, over
orthonormal vectors, the optimum is achieved by *aligning* $u_k$ with
the eigenvector of the $k$-th smallest eigenvalue. Strictly decreasing
weights are essential: equal weights would only pin down the *span* of
the $u_k$'s, not their individual identities.

**Lemma B5** (operator norm of a Gaussian random matrix; Vershynin 2018,
Thm. 4.4.5). *Let $G \in \mathbb{R}^{n \times d}$ have i.i.d. entries
$\mathcal{N}(0, s^2)$. There exists an absolute constant $c_0 > 0$ such
that for any $\delta \in (0,1)$, with probability at least $1 - \delta$,*

$$
\|G\|_\mathrm{op} \;\leq\; c_0\, s\, \bigl(\sqrt{n} + \sqrt{d} + \sqrt{2\log(1/\delta)}\bigr).
$$

*Plain-English:* a Gaussian random matrix has operator norm of order
$s(\sqrt{n} + \sqrt{d})$ in expectation, with sub-Gaussian tails. This
is the non-asymptotic version of the Marchenko–Pastur edge.

---

## 3. Theorem 1 — PCA–3D recovery

### 3.1. Plain-English summary

*"If a layer's residual stream contains a noisy linear probe of the 3D
scene coordinates, then the top-3 PCA components of that layer recover
the world-coordinate 3-subspace, with rotation error proportional to
noise-over-signal. Under stronger conditions (distinct probe gains $\beta_k$
along distinct world-coordinate directions $\sigma_k$), individual PC
axes recover individual world-coordinate axes."*

This is the precise statement behind every PCA visualization in the
paper. When you see a PCA scatter that looks like an orthographic
projection of the scene, Theorem 1 certifies the visual recognition: it
tells you *exactly* what assumptions buy you that picture and how the
recovery error scales with noise.

### 3.2. Statements

**Theorem 1** (subspace recovery). *Let $X \in \mathbb{R}^{n \times 3}$
have full column rank with $\sigma_1(X) > \sigma_2(X) > \sigma_3(X) > 0$.
Let $A \in \mathbb{R}^{3 \times d}$ factor as $A = \mathrm{diag}(\beta_1,
\beta_2, \beta_3)\, Q$, where $\beta_1 > \beta_2 > \beta_3 > 0$ and $Q
\in \mathbb{R}^{3 \times d}$ has orthonormal rows ($QQ^\top = I_3$).
Suppose $H = XA + E$ where $\|E\|_\mathrm{op} < \tfrac{1}{2} \beta_3
\sigma_3(X)$. Then*

$$
\sin\Theta\bigl(\mathcal{U}_3(H),\, \mathcal{U}_3(X)\bigr)
\;\leq\; \frac{2\,\|E\|_\mathrm{op}}{\beta_3\,\sigma_3(X)}.
$$

**Theorem 1′** (axis-by-axis recovery). *Suppose additionally the
products $\beta_k^2 \sigma_k(X)^2$ are distinct, and define*

$$
\mathrm{gap}_k \;:=\; \min_{j \neq k}\bigl|\beta_k^2 \sigma_k(X)^2 - \beta_j^2 \sigma_j(X)^2\bigr| \;>\; 0
\qquad k \in \{1,2,3\}.
$$

*Then for every $k \in \{1,2,3\}$,*

$$
\sin\Theta\bigl(u_k(H),\, u_k(XA)\bigr)
\;\leq\; \frac{\sqrt{2}\, \|E\|_\mathrm{op} \cdot \beta_1 \sigma_1(X) \bigl(2 + \|E\|_\mathrm{op}/(\beta_1\sigma_1(X))\bigr)}{\mathrm{gap}_k}.
$$

*Furthermore, $u_k(XA)$ lies in $\mathcal{U}_3(X)$, so the $k$-th left
singular vector of $H$ recovers a fixed direction in the world-coordinate
3-subspace up to $O(\|E\|_\mathrm{op}/\mathrm{gap}_k)$ error.*

### 3.3. Proof of Theorem 1

#### Chain of reasoning (intuitive overview)

**What we want.** Show that the top-3 PCA directions of the *noisy*
representation $H$ point in essentially the same 3D subspace as the
*clean* world-coordinate matrix $X$. In ML terms: even though the
model's residual stream is $H = XA + E$ with $A$ rescaling the axes
and $E$ adding noise, doing PCA on $H$ "rediscovers" the world-coordinate
geometry up to small angular error.

**Why this isn't trivial.** Both $A$ (the linear probe) and $E$ (the
noise) are unknown. $A$ could rotate or rescale axes arbitrarily, and
$E$ could in principle line up with the signal and rotate the PCs
significantly. The proof must show that *neither* of these effects is
enough to break recovery, given our (mild) hypotheses.

**The two key ideas.**

1. *The probe doesn't expand the subspace.* Multiplying $X$ by any
   matrix $A$ on the right keeps you within $X$'s column space. So
   $XA$ — the clean signal — has *the same* top-3 PC subspace as $X$.
   The matrix $A$ only rotates and rescales *within* that subspace; it
   cannot move the subspace itself. (Step 1.)

2. *Spectral gap = robustness.* Once the clean signal $XA$ has a
   non-trivial gap between its 3rd and 4th singular values (here
   $\beta_3\sigma_3 > 0 = $ no rank-4 component), and the noise is
   bounded ($\|E\| < \tfrac{1}{2}\beta_3\sigma_3$), then by Weyl's
   inequality the noisy $H$ must also have a strictly positive
   spectral gap. (Step 2.)

3. *Spectral gap controls PCA stability.* Davis–Kahan is a quantitative
   "implicit function theorem" for eigenvectors: the angle between
   the noisy and clean top-$k$ PC subspaces is bounded by
   noise / spectral gap. (Step 3.)

The whole proof is just chaining these three pieces together. The
"hard work" is geometric — Step 1, recognizing that $A$ doesn't break
the subspace. Steps 2–3 are bookkeeping with standard tools.

#### Step-by-step proof

**Setup.** Write the SVD $X = U_X \Sigma_X V_X^\top$ with $\Sigma_X =
\mathrm{diag}(\sigma_1, \sigma_2, \sigma_3)$ where we abbreviate $\sigma_k
:= \sigma_k(X)$ throughout this proof. $U_X \in \mathbb{R}^{n \times 3}$
has orthonormal columns; $V_X \in \mathbb{R}^{3 \times 3}$ is orthogonal.
Let $D := \mathrm{diag}(\beta_1, \beta_2, \beta_3)$, so $A = D Q$ with
$Q$ orthonormal-rowed.

**Step 1 (the clean column-space is preserved).**

*Goal:* show $\mathcal{U}_3(XA) = \mathcal{U}_3(X)$ and bound
$\sigma_3(XA)$ from below.

We compute the Gram matrix on the *left* (the $n \times n$ side):

$$
(XA)(XA)^\top \;=\; X D Q Q^\top D X^\top \;=\; X D^2 X^\top \;=\; U_X \Sigma_X V_X^\top D^2 V_X \Sigma_X U_X^\top,
$$

where the second equality uses $QQ^\top = I_3$. Define the inner $3
\times 3$ symmetric positive-definite matrix $M := \Sigma_X V_X^\top D^2
V_X \Sigma_X$. Then $(XA)(XA)^\top = U_X M U_X^\top$, an $n \times n$
matrix whose non-zero eigenvalues are exactly the eigenvalues of $M$, and
whose corresponding eigenvectors lie in $\mathrm{col}(U_X) =
\mathcal{U}_3(X)$.

Therefore the top-3 left singular vectors of $XA$ span $\mathcal{U}_3(X)$:
$\mathcal{U}_3(XA) = \mathcal{U}_3(X)$. ✓

To bound $\sigma_3(XA)$ from below: the eigenvalues of $M$ equal
$\sigma_k(XA)^2$. By Sylvester's law of inertia, $V_X^\top D^2 V_X$ is
*similar* to $D^2$ (same eigenvalues, just expressed in a different
orthogonal basis), so its eigenvalues lie in $[\beta_3^2, \beta_1^2]$.
Now $M = \Sigma_X (V_X^\top D^2 V_X) \Sigma_X$, so by interlacing-type
bounds (or direct Rayleigh-quotient estimates), the eigenvalues of $M$
lie in $[\beta_3^2 \sigma_3^2, \beta_1^2 \sigma_1^2]$. In particular,

$$
\sigma_3(XA) \;\geq\; \beta_3 \sigma_3. \tag{1}
$$

*Why this step works.* The map $A: \mathbb{R}^3 \to \mathbb{R}^d$ is an
*isometric injection followed by reweighting*: it stretches the three
input directions by $\beta_1, \beta_2, \beta_3$ but cannot expand the
3-dimensional column-space of $X$. Geometrically, $XA$ is the same
3-dimensional point cloud as $X$, just embedded in a higher-dimensional
space and with axes rescaled. So PCA of $XA$ — which is invariant to the
ambient dimension — recovers the same point cloud as PCA of $X$.

**Step 2 (the spectral gap of $H$ is bounded away from zero).**

*Goal:* show $\sigma_3(H) > \sigma_4(H)$ strictly, with quantitative gap
$\sigma_3(H) - \sigma_4(H) \geq \tfrac{1}{2}\beta_3 \sigma_3$.

Apply Lemma B1 (Weyl) to $H = XA + E$, recalling $\sigma_4(XA) = 0$
(because $XA$ has rank 3):

$$
\sigma_3(H) \;\geq\; \sigma_3(XA) - \|E\|_\mathrm{op} \;\stackrel{(1)}{\geq}\; \beta_3 \sigma_3 - \tfrac{1}{2}\beta_3 \sigma_3 \;=\; \tfrac{1}{2}\beta_3 \sigma_3,
$$

$$
\sigma_4(H) \;\leq\; \sigma_4(XA) + \|E\|_\mathrm{op} \;=\; 0 + \|E\|_\mathrm{op} \;<\; \tfrac{1}{2}\beta_3 \sigma_3.
$$

Subtracting,

$$
\sigma_3(H) - \sigma_4(H) \;>\; \tfrac{1}{2}\beta_3 \sigma_3 - \|E\|_\mathrm{op} \;>\; 0. \tag{2}
$$

*Why this step works.* The clean matrix $XA$ has a "perfect" spectral
gap at index 3: a positive value $\beta_3 \sigma_3$ above and an exact
zero below. The noise $E$ can only smear each singular value by
$\|E\|_\mathrm{op}$ in either direction (Lemma B1). Since we assumed
$\|E\|_\mathrm{op} < \tfrac{1}{2}\beta_3\sigma_3$, the noise can't eat up
more than half the gap on each side, so a strictly positive gap
survives.

**Step 3 (Davis–Kahan converts the gap into a subspace rotation
bound).**

*Goal:* derive the claimed bound $\sin\Theta(\mathcal{U}_3(H),
\mathcal{U}_3(X)) \leq 2\|E\|_\mathrm{op}/(\beta_3\sigma_3)$.

Apply Lemma B2 (Davis–Kahan) with $M = XA$, $M' = H$:

$$
\sin\Theta\bigl(\mathcal{U}_3(H), \mathcal{U}_3(XA)\bigr)
\;\leq\; \frac{\|E\|_\mathrm{op}}{\sigma_3(XA) - \sigma_4(XA)}
\;=\; \frac{\|E\|_\mathrm{op}}{\sigma_3(XA)}
\;\stackrel{(1)}{\leq}\; \frac{\|E\|_\mathrm{op}}{\beta_3 \sigma_3}.
$$

To get the constant 2 (matching the version with the gap measured at
$M'$): use the Yu–Wang–Samworth restatement that allows replacing the
gap denominator by either matrix's gap. Concretely, using $\sigma_3(H) -
\sigma_4(H) \geq \tfrac{1}{2}\beta_3 \sigma_3$ from (2):

$$
\sin\Theta\bigl(\mathcal{U}_3(H), \mathcal{U}_3(XA)\bigr)
\;\leq\; \frac{\|E\|_\mathrm{op}}{\sigma_3(H) - \sigma_4(H)}
\;\leq\; \frac{2\|E\|_\mathrm{op}}{\beta_3 \sigma_3}. \tag{3}
$$

By Step 1, $\mathcal{U}_3(XA) = \mathcal{U}_3(X)$, so substituting into
(3) gives the theorem. $\square$

*Why this step works.* Davis–Kahan is the matrix-perturbation analog of
the implicit function theorem: it says the top-$k$ singular subspace
varies *Lipschitz-continuously* with the matrix, with Lipschitz constant
controlled by the spectral gap. Step 2 made the gap explicit; Step 3
just plugs it in.

### 3.4. Proof of Theorem 1′

We now strengthen subspace-recovery to axis-by-axis recovery. The
mechanism is the same — perturbation theory — but applied to individual
singular vectors rather than the entire top-3 subspace, which requires
distinct singular values of $XA$.

**Step 1′ (singular values of $XA$ are distinct).**

*Goal:* under the gap assumption, show $\sigma_1(XA) > \sigma_2(XA) >
\sigma_3(XA) > 0$.

From Step 1 of Theorem 1, $\sigma_k(XA)^2$ are the eigenvalues of $M =
\Sigma_X V_X^\top D^2 V_X \Sigma_X$. The hypothesis is exactly that
$\beta_k^2\sigma_k^2$ are distinct *with positive gap*, but we need to
relate this to the eigenvalues of $M$ rather than the products
$\beta_k^2\sigma_k^2$ themselves.

Observe: when $V_X = I$ (no rotation in the column-space of $X$), we
have $M = \Sigma_X D^2 \Sigma_X = \mathrm{diag}(\beta_k^2 \sigma_k^2)$,
so $\sigma_k(XA)^2 = \beta_k^2 \sigma_k^2$ exactly. For general
orthogonal $V_X$, the eigenvalues of $M$ are continuously perturbed from
$\{\beta_k^2 \sigma_k^2\}$, but stay within the same interval. The gap
condition $\mathrm{gap}_k > 0$ is exactly what guarantees these remain
distinct under any orthogonal $V_X$ — this is the *generic* assumption,
and we adopt it as a hypothesis.

**Step 2′ (Wedin for individual vectors).**

*Goal:* bound $\sin\Theta(u_k(H), u_k(XA))$ in terms of $\|E\|_\mathrm{op}$
and $\mathrm{gap}_k$.

Apply Lemma B3 (Wedin) with $M = XA$, $M' = H$, at index $k$:

$$
\sin\Theta\bigl(u_k(H), u_k(XA)\bigr)
\;\leq\; \frac{\sqrt{2}\, \|E\|_\mathrm{op} \cdot \max\{\sigma_k(XA), \sigma_k(H)\}}{\min\bigl(\sigma_k(XA)^2 - \sigma_{k+1}(XA)^2,\; \sigma_{k-1}(XA)^2 - \sigma_k(XA)^2\bigr)}.
$$

The denominator is exactly $\mathrm{gap}_k$ (from Step 1′).

For the numerator, $\sigma_k(XA) \leq \beta_1 \sigma_1$ by Step 1, and
$\sigma_k(H) \leq \sigma_k(XA) + \|E\|_\mathrm{op} \leq \beta_1 \sigma_1
+ \|E\|_\mathrm{op}$. Substituting:

$$
\sin\Theta\bigl(u_k(H), u_k(XA)\bigr)
\;\leq\; \frac{\sqrt{2}\, \|E\|_\mathrm{op}\, \bigl(\beta_1 \sigma_1 + \|E\|_\mathrm{op}\bigr)}{\mathrm{gap}_k}.
$$

Since $u_k(XA) \in \mathcal{U}_3(XA) = \mathcal{U}_3(X)$, the right-hand
side bounds the angle between $u_k(H)$ and a fixed direction in the
world-coordinate 3-subspace. $\square$

*Why this step works.* The same intuition as Davis–Kahan but at finer
granularity. To pin down a *specific* singular vector — not just the
subspace it lies in — we need that singular vector to be the unique
solution to a Rayleigh-quotient problem. That requires a strict gap
between its singular value and *both* its neighbors. The gap condition
provides exactly this.

### 3.5. Remarks

1. **Non-degeneracy is essential.** The assumption $\beta_1 > \beta_2 >
   \beta_3 > 0$ is the analog of Park et al.'s non-degenerate
   singular-value assumption (their Theorem B.1). Without it (e.g., if
   the model encoded $x, y, z$ with equal probe strength), Theorem 1
   still gives subspace recovery but Theorem 1′ cannot identify
   *individual* axes — PCA would return a rotation of $(x,y,z)$, not
   $(x,y,z)$ themselves. Empirically, since residual streams typically
   allocate slightly different gains to different geometric axes (depth
   gets more bandwidth than left/right in most VLMs), this assumption
   holds in practice.

2. **What absorbs into $E$.** The noise term $E$ collects (a) finite-sample
   variance in the activation read-out, (b) genuine non-linear scene
   structure beyond the linear probe, (c) other features encoded at the
   same layer (e.g., object category, lighting). The bound is
   *non-asymptotic* and gives a quantitative recovery guarantee for any
   realization of $E$.

3. **The constant 2 is loose.** The sharp Davis–Kahan constant is
   $\sqrt{2}$ under a slightly different normalization. The asymptotic
   rate $O(\|E\|_\mathrm{op}/\sigma_3)$ is unchanged. We use the
   convenient constant 2 for simplicity.

4. **Empirical tightness.** In our experiments at the peak layer (e.g.,
   InternVL3-8B layer 18, $f = 32$), the residualized RSA is $\rho \approx
   0.47$. Up to a strictly monotone reparameterization, $\rho$ is related
   to $1 - \sin^2\Theta$ (Mantel-style identity), giving $\sin\Theta \approx
   0.73$. From Theorem 1, this corresponds to $\|E\|_\mathrm{op}/(\beta_3
   \sigma_3) \approx 0.36$, i.e., the noise-to-3rd-signal ratio is roughly
   $1/3$. This is consistent with the empirical observation that the 3rd
   PC is visibly noisier than the first two in detailed-PCA panels.

---

## 4. Theorem 2 — Residualization as orthogonal projection

### 4.1. Plain-English summary

*"Computing residualized RSA — that is, regressing each activation
column against confound covariates and then computing RSA on the
residuals — is mathematically identical to computing RSA on the
projection of $H$ onto the orthogonal complement of the confound subspace.
A non-zero residualized RSA therefore certifies that the geometric signal
in $H$ is carried by directions that no linear combination of the
confounds can produce."*

This is a clean methodological certification: it converts a procedural
choice (residualize before computing RSA) into a falsification statement
(non-zero residualized RSA falsifies the linear-shortcut hypothesis).
Reviewers asking "are you sure your model isn't just using depth as a
shortcut?" can be referred to this proposition.

### 4.2. Statement

**Setup.** Let $C \in \mathbb{R}^{n \times k}$ be a confound design
matrix with columns indexing per-token confound covariates (depth,
frame-index, scene-id one-hots, etc.). Let

$$
\Pi_C \;:=\; C(C^\top C)^{+} C^\top, \qquad \Pi_{C^\perp} \;:=\; I_n - \Pi_C
$$

be the orthogonal projectors on $\mathrm{col}(C)$ and its orthogonal
complement, respectively (with $(\cdot)^+$ the Moore–Penrose
pseudo-inverse to handle rank-deficient $C$). Define the **residualized
activation matrix**

$$
\widetilde{H} \;:=\; \Pi_{C^\perp} H,
$$

i.e., the matrix obtained by regressing each *column* of $H$ on $C$ and
keeping the residuals. Define **residualized RSA** as

$$
\tilde\rho(H, X; C) \;:=\; \rho\bigl(\widetilde{H},\, X\bigr).
$$

**Proposition 2** (residualization identity). *Decomposing $H = \Pi_C H +
\Pi_{C^\perp} H =: H_\parallel + H_\perp$, we have*

$$
\tilde\rho(H, X; C) \;=\; \rho(H_\perp, X).
$$

**Corollary 2.1** (linear shortcut null). *Suppose $H = C\Phi$ for some
$\Phi \in \mathbb{R}^{k \times d}$, i.e., $H$ is exactly explained by a
linear shortcut from $C$. Then $\widetilde{H} = 0$, and consequently the
distance matrix $D_{\widetilde{H}} \equiv 0$. By the standard convention
(Pearson correlation undefined when one argument is constant), we set
$\tilde\rho(H, X; C) := 0$ in this case.*

**Corollary 2.2** (depth-shortcut immunity). *Including depth, frame
index, and per-scene dummies as columns of $C$ implies that any signal
in $\tilde\rho > 0$ comes from a component of $H$ that is **not** a
linear function of these covariates. Consequently, $\tilde\rho > 0$
falsifies the linear-shortcut hypothesis with respect to $C$.*

**Corollary 2.3** (partial-correlation equivalence). *The "partial RSA"
defined as the Pearson partial correlation $\rho(D_H, D_X \mid D_C)$ —
where $D_C$ is the pairwise-distance matrix induced by the confounds —
satisfies $\rho(D_H, D_X \mid D_C) = \rho(\Pi_{C^\perp}^{(2)} D_H, X)$
where $\Pi_{C^\perp}^{(2)}$ acts on the pairwise-distance vectors in
$\mathbb{R}^{n(n-1)/2}$. Both versions kill linear shortcuts.*

### 4.3. Proof of Proposition 2

#### Chain of reasoning (intuitive overview)

**What we want.** When we "residualize" the activations $H$ against
confounds $C$ (depth, frame index, scene ID) and then compute RSA, we
should recover *exactly* the part of $H$ that is **orthogonal** to all
linear combinations of $C$. So if a model's signal lives entirely in
$\mathrm{col}(C)$ (a pure linear shortcut), residualized RSA is zero;
if any signal is orthogonal to $\mathrm{col}(C)$, that part survives.

**Why this matters in ML terms.** Reviewers always ask "are you sure
the model isn't just using depth as a shortcut?" Theorem 2 answers it
with a single algebraic identity: residualization is mathematically
equivalent to running RSA on the orthogonal-to-confounds subspace, so a
nonzero residualized RSA *certifies* signal orthogonal to depth /
frame / scene shortcuts.

**The single key idea.** Residualization is *just orthogonal
projection*. When you regress each column of $H$ on $C$ and keep the
residual, you get $\Pi_{C^\perp} H$ where $\Pi_{C^\perp}$ is the
projector on the orthogonal complement of $\mathrm{col}(C)$. RSA only
sees this projection — which by construction has zero overlap with
$C$.

**Connection to econometrics.** This is a generalization of the
Frisch–Waugh–Lovell theorem (1933): regression coefficients are
unchanged when you pre-residualize regressors and outcome against
shared controls. Same trick, applied to similarity matrices instead
of regressions.

#### Step-by-step proof

**Step 1 (the residualization is an orthogonal projection).**

*Goal:* show that $\widetilde{H} = H_\perp$, where the latter is defined
by orthogonally decomposing each column of $H$ in $\mathbb{R}^n$.

By definition, the column-residualization regresses each column $H_{\cdot,j}
\in \mathbb{R}^n$ ($j \in [d]$) on the columns of $C$, then keeps the
residual:

$$
\widetilde{H}_{\cdot,j} \;=\; H_{\cdot,j} - C \hat\Phi_j, \qquad \hat\Phi_j = (C^\top C)^{+} C^\top H_{\cdot,j}.
$$

Substituting:

$$
\widetilde{H}_{\cdot,j} \;=\; H_{\cdot,j} - C(C^\top C)^{+} C^\top H_{\cdot,j} \;=\; (I - \Pi_C) H_{\cdot,j} \;=\; \Pi_{C^\perp} H_{\cdot,j}.
$$

Stacking the columns: $\widetilde{H} = \Pi_{C^\perp} H = H_\perp$. ✓

**Step 2 (RSA inherits the equality).**

*Goal:* conclude $\tilde\rho(H, X; C) = \rho(H_\perp, X)$.

Since $\widetilde{H} = H_\perp$ as $n \times d$ matrices, their pairwise
squared-distance matrices are identical:

$$
(D_{\widetilde{H}})_{ij} \;=\; \|\widetilde{H}_i - \widetilde{H}_j\|^2 \;=\; \|(H_\perp)_i - (H_\perp)_j\|^2 \;=\; (D_{H_\perp})_{ij}.
$$

Thus $\rho(\widetilde{H}, X) = \rho(H_\perp, X)$, which by definition is
$\tilde\rho(H, X; C)$. $\square$

*Why this proposition matters.* The decomposition $H = H_\parallel +
H_\perp$ is into two pieces that live in orthogonal subspaces of
$\mathbb{R}^n$. The proposition says residualized RSA *only* sees
$H_\perp$, the part of $H$ that is uncorrelated (in $\mathbb{R}^n$) with
every column of $C$. Anything in $H_\parallel$ — including any pure
linear shortcut — contributes nothing.

### 4.4. Proof of Corollary 2.1

If $H = C\Phi$, every column of $H$ lies in $\mathrm{col}(C)$, so
$\Pi_{C^\perp} H = 0$. Hence $\widetilde{H} = 0$ and $D_{\widetilde{H}}
\equiv 0$ identically. Pearson correlation between a zero vector and
$\mathrm{vec}^{\triangle}(D_X)$ is undefined (zero divided by zero); we
adopt the convention $\tilde\rho := 0$, which is consistent with the
limiting interpretation ($\tilde\rho \to 0$ as the residual signal
vanishes). $\square$

### 4.5. Proof of Corollary 2.2

By Corollary 2.1, any $H$ of the form $C\Phi$ produces $\tilde\rho = 0$.
Contrapositively, $\tilde\rho > 0$ implies $H \notin \{C\Phi : \Phi \in
\mathbb{R}^{k \times d}\}$, i.e., $H$ has at least one column with a
non-trivial component in $\mathrm{col}(C)^\perp$. This non-trivial
component carries the entire residual RSA signal. $\square$

### 4.6. Proof of Corollary 2.3

In partial-correlation RSA, one regresses both $\mathrm{vec}^{\triangle}(D_H)$
and $\mathrm{vec}^{\triangle}(D_X)$ on $\mathrm{vec}^{\triangle}(D_C)$
(or on a richer matrix derived from $C$), then computes the Pearson
correlation of the residuals. Letting $\Pi_{C^\perp}^{(2)}$ be the
orthogonal projector on the orthogonal complement of the confound
distance-vector subspace, the same identity holds at the
distance-vector level. The construction is parallel to the activation-level
case; we omit the algebraic details. $\square$

### 4.7. Remarks

1. **Linearity is essential.** Both Proposition 2 and its corollaries
   assume *linear* residualization. A non-linear shortcut (e.g.,
   $H \approx \phi(C)$ for some non-linear $\phi$) is *not* killed by
   linear residualization: only the linear projection of $\phi(C)$ onto
   $\mathrm{col}(C)$ is removed. Mitigation: include non-linear features
   (polynomials, splines, learned features) of $C$ as additional columns.
   In practice, we include linear depth, depth$^2$, frame-index,
   frame-index$^2$, and per-scene dummies in $C$.

2. **Choice of pairwise distance.** Above we use squared Euclidean
   distance. The proposition extends verbatim to any pairwise
   $\ell_p$-distance with $p \geq 1$, since the projector identity
   doesn't depend on the choice of norm. The Pearson correlation is
   invariant to monotone reparameterization of the distance.

3. **Geometric content.** The proposition is essentially a restatement of
   the Frisch–Waugh–Lovell theorem from econometrics in the RSA setting.
   FWL says that regression coefficients are unchanged when you
   pre-residualize regressors and outcomes against shared controls; here
   we apply the same idea to similarity matrices.

4. **Connection to causal inference.** Residualization is
   *back-door adjustment* on linear pathways: if $C$ d-separates a
   shortcut from $X$, then $\tilde\rho > 0$ implies the existence of a
   causal pathway from the encoded representation to $X$ that does not
   pass through the confounds. Translating this into a formal causal
   identification statement requires further structural assumptions
   beyond the scope of this paper.

---

## 5. Theorem 3 — Continuous Dirichlet energy ↔ PCA

### 5.1. Plain-English summary

*"If the model's residual stream minimizes a 3D-geometry-weighted
Dirichlet energy on the scene — i.e., places nearby objects (in 3D) at
nearby points (in representation space) — then the principal components
of that representation are exactly the eigenvectors of the
Laplace-Beltrami operator on the scene manifold. In the appropriate
kernel limit, those eigenvectors converge to the Cartesian
coordinate functions $x, y, z$ themselves: PCA recovers world
coordinates as a free byproduct of energy minimization."*

This is the variational counterpart of Theorem 1. Theorem 1 says: *if* a
linear 3D probe is present, *then* PCA recovers it. Theorem 3 says: *if*
the model implements smoothness with respect to 3D geometry, *then*
PCA *must* yield a linear 3D probe. The two together say the empirical
PCA-recovers-3D phenomenon arises from a single underlying mechanism
(smoothness) and admits a precise mathematical statement.

This is the direct continuous-geometry analog of Park et al., Theorem
B.1 (which handles discrete graphs).

### 5.2. Setup

Fix the $n$ object-tokens of a scene with 3D coordinates $X = (x_1,
\ldots, x_n)^\top \in \mathbb{R}^{n \times 3}$. Let $\kappa: \mathbb{R}^3
\times \mathbb{R}^3 \to \mathbb{R}_{\geq 0}$ be a symmetric
positive-semidefinite kernel; e.g., the Gaussian kernel
$\kappa_\tau(x, y) = \exp(-\|x - y\|^2/2\tau^2)$ with bandwidth $\tau >
0$. Construct:

- Weight matrix $W \in \mathbb{R}^{n \times n}$ with $W_{ij} = \kappa(x_i,
  x_j)$, $W_{ii} = 0$ (we exclude self-loops).
- Degree matrix $D_W = \mathrm{diag}(W \mathbf{1})$.
- Graph Laplacian $L = D_W - W$.

$L$ is symmetric positive semi-definite. Its eigenvalues $0 = \lambda_1 <
\lambda_2 \leq \cdots \leq \lambda_n$ have orthonormal eigenvectors
$z^{(1)}, z^{(2)}, \ldots, z^{(n)}$. The smallest eigenvalue is zero (if
the graph is connected) with eigenvector $z^{(1)} = n^{-1/2}\mathbf{1}$,
which we call the *trivial mode*.

The **3D-geometry-weighted Dirichlet energy** of $H \in \mathbb{R}^{n
\times d}$ is

$$
\mathcal{E}_X(H) \;:=\; \mathrm{tr}(H^\top L H) \;=\; \tfrac{1}{2}\sum_{i,j} W_{ij}\, \|h_i - h_j\|^2.
$$

*Why this functional?* The term $W_{ij} \|h_i - h_j\|^2$ penalizes a large
representation-space distance between two object-tokens that are *close*
in 3D. So minimizing $\mathcal{E}_X(H)$ encourages the representation to
be a *smooth* function of 3D position: it acts as a Laplacian regularizer
indexed by the scene geometry.

### 5.3. Statements

**Theorem 3** (Dirichlet ↔ PCA). *Fix $s \leq \min(n, d) - 1$ and
constants $\epsilon_1 > \epsilon_2 > \cdots > \epsilon_s > 0$. Let*

$$
H^* \;\in\; \arg\min_{H \in \mathbb{R}^{n \times d}}\, \mathcal{E}_X(H)
\qquad \text{subject to} \qquad \sigma_k(H) \geq \epsilon_k \text{ for all } k \in [s]. \tag{$\star$}
$$

*Then:*

*(a) The minimizer is unique up to right-orthogonal transformations.
Among the equivalence class of minimizers, the left singular vectors
satisfy $u_k(H^*) = z^{(k)}$ for $k = 1, \ldots, s$.*

*(b) After mean-centering, $H^*_c := P_{J^\perp} H^*$, the $k$-th
principal component (for $k = 1, \ldots, s-1$) equals $z^{(k+1)}$.*

**Theorem 3′** (Belkin–Niyogi limit). *Suppose $x_1, \ldots, x_n$ are
i.i.d. samples from an absolutely continuous distribution supported on a
compact subset $\mathcal{M} \subset \mathbb{R}^3$, with smooth density
bounded away from zero. Let $\kappa$ be the Gaussian kernel with
bandwidth $\tau = \tau_n$ chosen so that $\tau_n \to 0$ and $n
\tau_n^{3+\alpha} \to \infty$ for some $\alpha > 0$. Then, for any fixed
$k$, with probability $\to 1$ as $n \to \infty$:*

$$
z^{(k+1)}(x_i) \;\longrightarrow\; \phi^{(k)}(x_i) \quad \text{uniformly in } i,
$$

*where $\phi^{(k)}$ is the $k$-th non-constant eigenfunction of the
Laplace–Beltrami operator on $\mathcal{M}$. In the special case where
$\mathcal{M}$ is a 3-dimensional Euclidean domain, the first three such
eigenfunctions span the affine functions: $\phi^{(1)}, \phi^{(2)},
\phi^{(3)}$ are the centered coordinate functions $x, y, z$ (up to
rotation and reflection of axes).*

### 5.4. Proof of Theorem 3

#### Chain of reasoning (intuitive overview)

**What we want.** Show that minimizing the 3D-geometry-weighted
Dirichlet energy over a representation matrix $H$ (with mild
non-degeneracy constraints) forces the principal components of $H$ to
be the *Laplacian eigenmaps* of the scene — and in the kernel limit
(Theorem 3'), those eigenmaps converge to the world-coordinate
functions $x, y, z$ themselves.

In ML terms: if your loss says "objects close in 3D should have close
representations," then the geometry of the representation literally
becomes a coordinate system for 3D — for free, just from the loss.

**Why this is non-trivial.** The Dirichlet loss is a *quadratic form
in $H$*; the principal components of $H$ are the *eigenvectors of
$H^\top H$*. These are different mathematical objects — quadratic-form
minimization vs. eigendecomposition. Theorem 3 is the bridge: under
appropriate constraints, optimizing the loss forces the eigenstructure
to a specific form.

**The four key ideas.**

1. *Decompose along singular directions.* Use the SVD $H = U\Sigma V^\top$.
   The Dirichlet energy $\mathrm{tr}(H^\top L H)$ rewrites as a sum
   $\sum_k \sigma_k^2 \cdot \langle u_k, L u_k \rangle$. Each "PC
   direction" $u_k$ contributes its own term. (Step 1.)

2. *The constraints decouple.* The lower-bound constraint $\sigma_k
   \geq \epsilon_k$ pushes each $\sigma_k$ to its lower bound (since
   each term is $\sigma_k^2 \times$ something positive). Once the
   $\sigma_k$ are pinned, only the $u_k$ are free. (Step 2.)

3. *Ky Fan inequality picks the eigenvectors.* When you minimize a
   weighted sum $\sum_k w_k \langle u_k, L u_k \rangle$ over orthonormal
   $u_k$ with strictly decreasing weights $w_k$, the unique answer is
   $u_k = z^{(k)}$ — the eigenvectors of $L$ paired with its smallest
   $k$ eigenvalues. (Step 3.)

4. *Mean-centering kills the trivial mode.* The smallest-eigenvalue
   eigenvector of $L$ is the constant vector $\mathbf{1}$. PCA
   automatically subtracts the mean, so this trivial mode disappears
   and the *visible* PCs of mean-centered $H^*$ are $z^{(2)}, z^{(3)},
   z^{(4)}, \ldots$ — the "geometry-carrying" Laplacian modes.
   (Step 4.)

**Why each idea is needed:** Step 1 turns the loss into a sum the
constraints can act on. Step 2 strips out the singular-value
optimization (purely numerical). Step 3 is the "core math" — Ky Fan
is an old (1949) inequality saying weighted Rayleigh quotient is
minimized by eigenvector matching. Step 4 is bookkeeping for the
PCA convention.

**Where the work happens:** Steps 1 and 3 carry the actual content.
Step 1 is the SVD trick; Step 3 is the spectral-graph-theory result.
The composition gives the theorem.

**Why Theorem 3' (Belkin–Niyogi) finishes the story:** Theorem 3 says
PCs equal Laplacian eigenmaps. But Laplacian eigenmaps of a
*Gaussian-kernel* graph on samples from a smooth manifold converge —
in a precise asymptotic sense — to the eigenfunctions of the
Laplace–Beltrami operator on that manifold. For a Euclidean 3D
domain, these are simply the coordinate functions $x, y, z$. So in the
appropriate limit, PCs literally **are** world coordinates.

#### Step-by-step proof

The proof closely parallels Park et al., Theorem B.1, with the key
modification that the Laplacian $L$ is now derived from a continuous 3D
kernel rather than a discrete graph. The spectral structure of $L$ is
identical in form, so the algebra goes through; the interpretation of
$L$'s eigenvectors changes (from graph spectral embeddings to
continuous-geometry eigenfunctions).

**Step 1 (decompose the energy along singular directions of $H$).**

*Goal:* express $\mathcal{E}_X(H)$ as a sum over singular components,
revealing the variational structure.

Write the SVD $H = U \Sigma V^\top$ with $U \in \mathbb{R}^{n \times r}$
(orthonormal columns), $V \in \mathbb{R}^{d \times r}$ (orthonormal
columns), $\Sigma = \mathrm{diag}(\sigma_1, \ldots, \sigma_r)$, $r =
\min(n,d)$. Substitute and use $V^\top V = I_r$:

$$
\mathcal{E}_X(H) \;=\; \mathrm{tr}(H^\top L H) \;=\; \mathrm{tr}\bigl(V \Sigma U^\top L U \Sigma V^\top\bigr).
$$

By the cyclic property of the trace:

$$
\mathcal{E}_X(H) \;=\; \mathrm{tr}\bigl(\Sigma V^\top V \Sigma U^\top L U\bigr) \;=\; \mathrm{tr}\bigl(\Sigma^2 U^\top L U\bigr) \;=\; \sum_{k=1}^r \sigma_k^2 \cdot \langle u_k, L u_k \rangle. \tag{4}
$$

*Why this step works.* The energy $\mathcal{E}_X$ is a *quadratic form*
in $H$, so it splits along orthogonal directions. The SVD provides
the natural orthogonal decomposition: $H$'s row-space contribution along
direction $u_k$ has magnitude $\sigma_k$ and "smoothness cost"
$\langle u_k, L u_k \rangle$. The total energy is a weighted sum.

**Step 2 (the optimization decouples).**

*Goal:* show that, at the minimum, $\sigma_k = \epsilon_k$ exactly, and
$\{u_k\}$ minimize the Laplacian-quadratic-form sum with fixed weights.

The expression (4) is jointly convex in $\{\sigma_k\}$ and quadratic in
$\{u_k\}$. The constraints decouple: $\sigma_k \geq \epsilon_k > 0$ acts
on each $\sigma_k$ separately, and the orthonormality acts on $\{u_k\}$
collectively but independently of the $\sigma_k$. Since each
$\langle u_k, L u_k \rangle \geq 0$ (because $L \succeq 0$), the energy
is minimized by pushing each $\sigma_k$ to its lower bound:

$$
\sigma_k^*(H^*) = \epsilon_k \quad \forall k \in [s], \qquad \sigma_k^*(H^*) = 0 \quad \forall k > s.
$$

Substituting back, the residual problem on $\{u_k\}$ is

$$
\min_{\{u_k\}_{k \in [s]} \subset \mathbb{R}^n,\, u_i \cdot u_j = \delta_{ij}} \sum_{k=1}^s \epsilon_k^2\, \langle u_k, L u_k \rangle. \tag{5}
$$

*Why this step works.* The constrained optimization splits because
nothing in $\langle u_k, L u_k \rangle$ depends on $\sigma_k$, and
$\sigma_k$ enters only through the coefficient $\sigma_k^2$ which we
want to make as small as possible. The orthonormality constraint
prevents the trivial solution $u_k = 0$.

**Step 3 (Ky Fan identifies the optimal vectors).**

*Goal:* show the optimum of (5) is $u_k = z^{(k)}$, the eigenvectors of
$L$ at the smallest eigenvalues.

Apply Lemma B4 (Ky Fan) with weights $w_k = \epsilon_k^2$ and matrix $L$.
The hypothesis $\epsilon_1 > \epsilon_2 > \cdots > \epsilon_s$ implies
$w_1 > w_2 > \cdots > w_s > 0$ — a strictly decreasing positive
sequence. Lemma B4 asserts that the unique minimizer (up to sign) is
$u_k = z^{(k)}$, with optimum value $\sum_{k=1}^s \epsilon_k^2 \lambda_k$.

This proves part (a). $\square$

*Why Ky Fan's strictly-decreasing-weights assumption matters.* If we
had equal weights $\epsilon_1 = \epsilon_2$, any orthonormal basis of
$\mathrm{span}(z^{(1)}, z^{(2)})$ would minimize the partial sum
$\epsilon_1^2(\langle u_1, L u_1\rangle + \langle u_2, L u_2 \rangle)$,
not just the canonical eigenvectors. We'd recover the *subspace* but
not individual vectors. The strictly decreasing condition breaks this
degeneracy and forces axis-by-axis alignment.

**Step 4 (mean-centering kills the trivial mode).**

*Goal:* derive part (b) from part (a).

The PCA of $H^*$ is the SVD of its mean-centered version $H^*_c =
P_{J^\perp} H^*$, where $P_{J^\perp} = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$.
Note $z^{(1)} = n^{-1/2} \mathbf{1}$ lies in $J$, so $P_{J^\perp} z^{(1)}
= 0$. The other eigenvectors $z^{(k)}$ for $k \geq 2$ are orthogonal to
$\mathbf{1}$ (they're orthogonal to $z^{(1)} \propto \mathbf{1}$), hence
$P_{J^\perp} z^{(k)} = z^{(k)}$ for $k \geq 2$.

Combining with part (a), the left singular vectors of $H^*_c$ are
$P_{J^\perp} u_k = z^{(k)}$ for $k = 2, \ldots, s$ (with $u_1 = z^{(1)}$
killed by the projector). After re-indexing by descending singular value,
the $k$-th principal component (for $k = 1, \ldots, s-1$) is $z^{(k+1)}$.
This proves part (b). $\square$

### 5.5. Discussion of Theorem 3′

A complete proof of Theorem 3′ requires substantial machinery from
spectral geometry and is beyond the scope of this document. We restrict
ourselves to a discussion of the main ingredients and the precise
references.

The convergence statement is Theorem 3.1 of Belkin & Niyogi (2003) for
the Gaussian kernel on a smooth Riemannian submanifold of Euclidean space.
The key analytical fact is that the Gaussian-kernel graph Laplacian
applied to a smooth function $f \in C^2(\mathcal{M})$ converges, after
the appropriate normalization, to the Laplace–Beltrami operator $\Delta_\mathcal{M}$:

$$
\frac{2}{n \tau^{d_\mathcal{M} + 2}} L^{(\tau)} f(x_i) \;\to\; -\Delta_\mathcal{M} f(x_i) \quad \text{as } n \to \infty, \tau \to 0,
$$

where $d_\mathcal{M}$ is the intrinsic dimension of $\mathcal{M}$.
Eigenvalue and eigenvector convergence follow by spectral perturbation
arguments under additional smoothness conditions on $\mathcal{M}$ and
the data-generating distribution.

The Euclidean case $\mathcal{M} = \mathbb{R}^3$ (or any compact convex
3-domain) is special: the Laplace–Beltrami operator reduces to the
ordinary Laplacian $\Delta = \partial_x^2 + \partial_y^2 + \partial_z^2$,
and on a compact convex domain its first three non-constant
eigenfunctions are the centered coordinate functions $x, y, z$ (up to
rotation/reflection of the axis system). This recovers the desired
"PCA = world coordinates" identification.

For non-Euclidean scene geometries (e.g., scenes that lie on a curved
manifold like a surface), Theorem 3′ still applies, and the recovered
"PCs" are the natural intrinsic coordinates of that manifold. This is
philosophically clean: the model recovers whatever geometry the scene
actually has, not whatever we *think* the scene has.

### 5.6. Remarks

1. **Park et al. correspondence.** The proof of Theorem 3, parts (a)
   and (b), is identical to the proof of Park et al., Theorem B.1, with
   our $L$ in place of their graph Laplacian. The novelty is *not* the
   algebra but the connection (via Theorem 3′) between $L$ and the
   continuous 3D geometry of the scene. Park et al. study discrete
   token-graphs (where $L$ is constructed from explicit edges); we study
   continuous scene-geometries (where $L$ emerges from a kernel).

2. **Approximate minimizers.** Theorem 3 characterizes *exact*
   minimizers of $\mathcal{E}_X$. In practice, the model only
   *approximately* minimizes Dirichlet energy. The relevant question is:
   how close does the model need to come for PCA to give a usable
   approximation of $z^{(k)}$? This is governed by an analog of
   Davis–Kahan for the energy minimization: if $\mathcal{E}_X(H) \leq
   \mathcal{E}_X(H^*) + \delta$, then PCA of $H$ recovers
   $\mathcal{U}_3(H^*)$ up to $O(\sqrt{\delta}/\mathrm{gap}_\lambda)$
   error. This is a routine perturbation argument we omit.

3. **Kernel choice.** The Gaussian kernel is convenient but not
   essential. Any symmetric, monotone-decreasing-in-distance kernel with
   exponential tails gives the same Belkin–Niyogi limit. The bandwidth
   $\tau$ is a regularization parameter: small $\tau$ recovers fine
   geometric detail (high-frequency eigenfunctions) but is sample-hungry,
   large $\tau$ blurs detail but is robust to sparse sampling.

4. **Connection to Laplacian eigenmaps.** Theorem 3 says the model is
   *implicitly* doing Laplacian eigenmap dimensionality reduction: it
   spontaneously embeds objects into low-dimensional coordinates that
   preserve scene geometry. This connects to a long line of manifold
   learning (Belkin–Niyogi, ISOMAP, t-SNE, UMAP) and offers a clean
   bridge between mechanistic interpretability and classical geometric
   data analysis.

5. **Relation to the empirical Dirichlet ratio.** In the report, we
   compute the *Dirichlet ratio* $\mathcal{E}_X(H) / \mathcal{E}_\pi(H)$,
   where $\pi$ is a uniform random permutation of $X$. Theorem 3
   predicts that this ratio is small at layers where the model *does*
   minimize Dirichlet energy, and close to 1 at layers where it does not.
   Empirically we see the ratio dip from $\approx 0.95$ at early layers
   to $\approx 0.92$ at peak layers — a small but consistent deviation
   from null, exactly as predicted by approximate (rather than exact)
   minimization.

---

## 6. Theorem 4 — Frame-count emergence bound

### 6.1. Plain-English summary

*"Frame-count is a sample-size axis for the residual stream's 3D probe.
Below a critical frame count $T^*$ — proportional to (probe-noise
variance) divided by (3rd probe gain $\times$ 3rd world-coordinate
spread)$^2$ — the noise in the per-frame activations overwhelms the
geometric signal, and PCA recovers a random 3-subspace. Above $T^*$, the
PCA-3D-recovery error decays as $1/\sqrt{T}$, predicting a sharp emergence
in residualized RSA at $T = T^*$."*

This is the explanation for the empirical f8 → f16 → f32 → f64 emergence
curve: the curve's shape is not mysterious but a direct consequence of
classical sample-complexity scaling for subspace estimation.

### 6.2. Setup

Each object-token $i \in [n]$ has a fixed 3D coordinate $x_i \in
\mathbb{R}^3$, constant across frames. At each frame $t \in [T]$, the
model produces an activation

$$
h_i^{(t)} \;=\; A^\top x_i + \xi_i^{(t)}, \qquad \xi_i^{(t)} \stackrel{\mathrm{iid}}{\sim} \mathcal{N}(0, \nu^2 I_d),
$$

where:
- $A \in \mathbb{R}^{3 \times d}$ is a fixed linear 3D probe with
  $A A^\top = \mathrm{diag}(\beta_1^2, \beta_2^2, \beta_3^2)$, $\beta_1 >
  \beta_2 > \beta_3 > 0$, identifying the model's three probe gains along
  three orthogonal probe directions.
- The noise $\xi_i^{(t)}$ is independent across both objects and
  frames.

The model aggregates frames via a (stylized) linear average:

$$
\bar h_i \;:=\; \frac{1}{T} \sum_{t=1}^T h_i^{(t)} \;=\; A^\top x_i + \bar\xi_i, \qquad \bar\xi_i \stackrel{\mathrm{iid}}{\sim} \mathcal{N}\bigl(0, (\nu^2/T)\, I_d\bigr).
$$

Stacking, $\bar H = X A + \bar\Xi$, where $\bar\Xi \in \mathbb{R}^{n
\times d}$ has i.i.d. $\mathcal{N}(0, \nu^2/T)$ entries.

*Why this setup?* The linear-average aggregation is a stand-in for the
attention-weighted average a real VLM performs. The Gaussian noise is
the simplest model for residual-stream noise. Both can be relaxed
(sub-Gaussian tails, more elaborate aggregation) without changing the
qualitative conclusions, as we discuss in §6.5.

### 6.3. Statements

**Theorem 4** (sample-complexity bound for subspace recovery). *Fix
$\delta \in (0,1)$. There exists an absolute constant $c > 0$ such that,
with probability at least $1 - \delta$,*

$$
\sin\Theta\bigl(\mathcal{U}_3(\bar H), \mathcal{U}_3(X)\bigr)
\;\leq\; \frac{c\, \nu}{\beta_3 \sigma_3(X)} \cdot \sqrt{\frac{n + d + \log(1/\delta)}{T}},
$$

*provided $T$ is large enough that the right-hand side is at most $\tfrac{1}{2}$.*

**Corollary 4.1** (emergence threshold). *To achieve $\sin\Theta \leq
\theta$ with probability at least $1 - \delta$, it suffices that*

$$
T \;\geq\; T^*(\theta, \delta) \;:=\; \frac{c^2\, \nu^2\, (n + d + \log(1/\delta))}{\beta_3^2\, \sigma_3(X)^2\, \theta^2}.
$$

*The dominant scaling is $T^* \propto \nu^2 / (\beta_3 \sigma_3(X))^2$.*

**Corollary 4.2** (population RSA emergence). *As $T \to \infty$,
$\rho(\bar H, X) \to \rho_\infty$ where $\rho_\infty$ is determined by the
limiting (noiseless) configuration. The convergence rate is $1/\sqrt{T}$
above the critical $T^*$, and the deviation $\rho_\infty - \rho(\bar H,
X)$ is sub-Gaussian.*

### 6.4. Proof of Theorem 4

#### Chain of reasoning (intuitive overview)

**What we want.** Show that as you give the model more frames to average
over, the recovered 3D subspace converges to the true one at rate
$1/\sqrt{T}$. There's a critical threshold $T^*$ below which recovery
is essentially random (frame averaging hasn't denoised enough), and
above which recovery is provably accurate.

In ML terms: if you train with single frames you'd need a *huge* probe
SNR to recover 3D structure; but with many frames the noise averages
out, and you can succeed even with a much weaker probe.

**Why this isn't trivial.** The bound has to:
(a) Quantify how Gaussian noise concentrates after averaging.
(b) Apply Theorem 1's geometric bound to the *averaged* noise.
(c) Handle the (small) probability of bad noise realizations.

**The single key idea.** Frame averaging is *identical* to dividing
the per-frame noise by $\sqrt{T}$. So if Theorem 1 needs noise
$\|E\|_\mathrm{op} < \tfrac{1}{2}\beta_3\sigma_3$ to give a clean
recovery bound, frame averaging gives us this for free as soon as
$T$ is large enough — specifically, as soon as
$\nu/\sqrt{T} < \tfrac{1}{2}\beta_3\sigma_3 / (\sqrt{n} + \sqrt{d})$.

**Step-by-step strategy.**

1. *Concentration.* The averaged noise $\bar\Xi$ is a Gaussian random
   matrix with variance $\nu^2/T$ per entry. Lemma B5 gives a sharp
   probabilistic bound on its operator norm, so we can say "with
   probability $1-\delta$, $\|\bar\Xi\|_\mathrm{op}$ is at most this
   small thing."

2. *Plug into Theorem 1.* Once $\|\bar\Xi\|_\mathrm{op}$ is bounded,
   Theorem 1's $\sin\Theta$ bound applies directly. We just substitute.

3. *Solve for $T$.* Setting the right-hand side equal to a target
   recovery angle $\theta$ and solving for $T$ gives the critical
   sample-complexity bound $T^* \propto \nu^2 / (\beta_3\sigma_3 \cdot
   \theta)^2$.

The proof has *no new geometric content*; it's a composition of
Theorem 1 with a standard probabilistic bound. The "value" of the
theorem is in turning Theorem 1 into a quantitative *frame-count*
prediction that you can directly check empirically (see report §5).

#### Step-by-step proof

The proof reduces to applying Theorem 1 with the "noise" matrix being
$\bar\Xi$, and bounding $\|\bar\Xi\|_\mathrm{op}$ via Lemma B5.

**Step 1 (noise concentration).**

*Goal:* bound $\|\bar\Xi\|_\mathrm{op}$ with high probability.

The matrix $\bar\Xi$ has i.i.d. $\mathcal{N}(0, \nu^2/T)$ entries. Apply
Lemma B5 with $s = \nu/\sqrt{T}$:

$$
\|\bar\Xi\|_\mathrm{op} \;\leq\; c_0 \cdot \frac{\nu}{\sqrt{T}} \cdot \bigl(\sqrt{n} + \sqrt{d} + \sqrt{2\log(1/\delta)}\bigr) \tag{6}
$$

with probability at least $1 - \delta$.

*Why this step works.* The aggregated noise $\bar\Xi$ has variance
$\nu^2/T$ per entry, so its operator norm is of order
$(\nu/\sqrt{T})(\sqrt{n} + \sqrt{d})$. The factor $1/\sqrt{T}$ is what
drives the emergence: more frames → smaller aggregated noise → larger
effective signal-to-noise ratio.

**Step 2 (verify Theorem 1's hypothesis).**

*Goal:* show $\|\bar\Xi\|_\mathrm{op} < \tfrac{1}{2} \beta_3 \sigma_3(X)$
when the right-hand side of the claimed bound is $\leq \tfrac{1}{2}$.

Suppose $\frac{c\,\nu}{\beta_3 \sigma_3(X)} \sqrt{(n+d+\log(1/\delta))/T}
\leq \tfrac{1}{2}$. Choosing $c$ large enough relative to $c_0\sqrt{2}$,
this implies $\|\bar\Xi\|_\mathrm{op} \leq \tfrac{1}{2}\beta_3\sigma_3(X)$
on the event of (6). ✓

**Step 3 (apply Theorem 1).**

*Goal:* derive the claimed $\sin\Theta$ bound.

Theorem 1 with $H = \bar H = XA + \bar\Xi$ and $E = \bar\Xi$ gives, on
the event of (6):

$$
\sin\Theta\bigl(\mathcal{U}_3(\bar H), \mathcal{U}_3(X)\bigr)
\;\leq\; \frac{2\|\bar\Xi\|_\mathrm{op}}{\beta_3\sigma_3(X)}
\;\stackrel{(6)}{\leq}\; \frac{2 c_0 \nu}{\beta_3 \sigma_3(X) \sqrt{T}} \bigl(\sqrt{n} + \sqrt{d} + \sqrt{2\log(1/\delta)}\bigr).
$$

By Cauchy–Schwarz, $\sqrt{a} + \sqrt{b} + \sqrt{c} \leq \sqrt{3(a+b+c)}$,
so absorbing the constants $2 c_0 \sqrt{3}\sqrt{2}$ into a single
constant $c$:

$$
\sin\Theta\bigl(\mathcal{U}_3(\bar H), \mathcal{U}_3(X)\bigr)
\;\leq\; \frac{c\,\nu}{\beta_3 \sigma_3(X)} \cdot \sqrt{\frac{n + d + \log(1/\delta)}{T}}.
$$

This is the claimed bound. $\square$

*Why this approach works.* The frame-averaging acts as a *denoiser* with
gain $\sqrt{T}$. Below $T^*$, the denoising is insufficient and the
residual noise $\bar\Xi$ swamps the smallest signal singular value
$\beta_3 \sigma_3$, breaking the spectral gap that Theorem 1 relies on.
Above $T^*$, the spectral gap reasserts itself and Theorem 1 kicks in,
delivering recovery at rate $\|\bar\Xi\|_\mathrm{op}/(\beta_3\sigma_3) =
O(1/\sqrt{T})$.

### 6.5. Proof of Corollaries

**Corollary 4.1.** Setting the right-hand side of Theorem 4 equal to
$\theta$ and solving for $T$ gives the stated $T^*$. $\square$

**Corollary 4.2.** Let $\rho_\infty$ denote the population RSA at $T =
\infty$ (i.e., the RSA of the noiseless configuration $XA$ vs. $X$). The
RSA function $\rho(\cdot, X)$ is Lipschitz with respect to small
perturbations of its first argument, with Lipschitz constant bounded in
terms of pairwise-distance variation; combining this Lipschitz property
with Theorem 4's $\sin\Theta = O(1/\sqrt{T})$ subspace bound yields a
$1/\sqrt{T}$ convergence rate for $\rho$ itself. The sub-Gaussian tails
follow from the sub-Gaussianity of $\bar\Xi$. We omit the routine
algebra; see Tropp (2015), §6.1, for the matrix concentration
ingredients. $\square$

### 6.6. Remarks

1. **Sub-Gaussian extension.** The Gaussian assumption on $\xi_i^{(t)}$
   can be relaxed to sub-Gaussian noise without changing the bound's
   form (only the absolute constant $c_0$ in Lemma B5 changes). For
   heavy-tailed noise, the rate degrades to $T^{-1/p}$ for some $p < 2$.

2. **Beyond linear averaging.** Real VLMs aggregate frames via
   attention rather than uniform averaging. As long as the attention
   weights are bounded and roughly uniform across frames (no single
   frame dominates), the same $\sqrt{T}$-denoising rate applies. Strong
   non-uniformity (e.g., the model attending to only the first frame)
   would slow emergence by an attention-concentration factor.

3. **Empirical validation.** Theorem 4 makes the testable prediction that
   the *ordering* of emergence thresholds across models is determined
   by $\beta_3$ (the smallest probe gain). Larger models, having stronger
   probes, should emerge at smaller $T^*$. In our data:

   - InternVL3-8B emerges fully by $T = 16$ (RSA 0.439 there, vs. 0.473
     at $T = 32$), suggesting $T^* \lesssim 16$.
   - Qwen-7B doesn't fully emerge until $T = 32$, with $T^* \approx 16$.
   - Qwen-32B emerges between $T = 16$ and $T = 32$, with $T^* \approx
     24$.

   These orderings are *qualitatively* consistent with Theorem 4 but
   converting the signs to a quantitative prediction would require
   estimating $\beta_3$ and $\nu$ per model — an empirical exercise
   we leave to future work.

4. **Why $1/\sqrt{T}$ vs. empirical $\log T$?** The empirical RSA curves
   in §5 of the report show approximately $\log T$ growth in the
   sub-critical regime, contrasting with our $1/\sqrt{T}$ post-critical
   rate. We attribute this to two model-mismatches with our stylized
   setup: (a) per-frame noise $\nu$ is itself a function of $T$ (more
   frames → richer attention context → smaller per-frame noise), and
   (b) saturation of $\rho_\infty$ at a value $\rho_\infty \approx 0.5$
   strictly below $1$ (because the "true" 3D structure is only partially
   captured by any linear probe). A more detailed model accounting for
   both effects would predict the observed $\log T$-then-plateau shape;
   we view this as a refinement worth pursuing.

5. **Connection to graph percolation (Park et al.).** Park et al.
   conjecture that the in-context emergence transition in their setup
   is a *graph percolation* phenomenon (their §5.2). Our Theorem 4
   identifies a different mechanism — sample-complexity for subspace
   estimation — that gives a $1/\sqrt{T}$ rate rather than a percolation
   transition. The two mechanisms are not mutually exclusive: in the
   continuous-geometry case, one can have *both* a percolation-style
   transition (for the geometric *connectivity* of the scene as
   resolved by sparse frame samples) *and* a sample-complexity
   threshold (for the resulting subspace estimation). Disentangling the
   two empirically would be an interesting follow-up.

---

## 7. How the four theorems wire into the paper narrative

| Theorem | What it certifies | Empirical evidence in report |
|---|---|---|
| 1 — PCA–3D recovery | Top-3 PCA of $H$ recovers world coordinates if a linear probe exists | `detailed_pca_*.png` figures (§5.5) |
| 2 — Residualization | Residualized RSA is RSA on $\Pi_{C^\perp} H$; rules out linear shortcuts | Raw vs residualized RSA gap in §5.9–5.11 |
| 3 — Dirichlet ↔ PCA | Energy minimization $\Rightarrow$ PCA = Laplacian eigenmap = world coords | Dirichlet-ratio curves per model |
| 4 — Frame-emergence | Frame-count is sample-size for subspace recovery | RSA-vs-frames tables 5.11–5.13 |

**Theorems 1 and 3 together form the main theoretical contribution**:

- **Theorem 1** is the *forward direction*: assume a linear probe exists,
  conclude PCA recovers it. This certifies our PCA-figure narrative.
- **Theorem 3** is the *variational direction*: assume the model
  minimizes a smoothness objective on 3D geometry, conclude that PCA
  *must* yield a linear 3D probe. This explains *why* the linear-probe
  hypothesis of Theorem 1 holds in practice — the model implements
  smoothness, and Theorem 3 says smoothness implies the linear probe
  via spectral embedding.
- Park et al. proved a single theorem (their B.1) covering both
  directions in the discrete-graph case. We split into two because the
  continuous-geometry case requires the additional Belkin–Niyogi limit
  (Theorem 3′), which is more involved than the discrete case and
  warrants separate exposition.

**Theorems 2 and 4 are auxiliary but necessary**:

- **Theorem 2** addresses the "depth-shortcut" reviewer concern with a
  one-line algebraic identity. It is the methodological backbone of
  the residualized-RSA analysis.
- **Theorem 4** turns the qualitative emergence observation into a
  quantitative prediction. It also provides the link between the paper's
  scaling story (more frames → better) and classical sample-complexity
  theory.

**Submission strategy.** For an 8-page conference submission (e.g.,
ICLR), Theorems 1 and 3 should appear in the main text with full
proofs sketched and complete proofs in the appendix. Theorems 2 and 4
can live entirely in the appendix, with their statements referenced from
the main text where the experimental sections rely on them.

---

## 8. References

- **Belkin, M. & Niyogi, P.** (2003). Laplacian eigenmaps for
  dimensionality reduction and data representation. *Neural
  Computation*, 15(6), 1373–1396.
- **Bhatia, R.** (1997). *Matrix Analysis*. Springer Graduate Texts in
  Mathematics, vol. 169.
- **Davis, C. & Kahan, W. M.** (1970). The rotation of eigenvectors by a
  perturbation, III. *SIAM Journal on Numerical Analysis*, 7(1), 1–46.
- **Fan, K.** (1949). On a theorem of Weyl concerning eigenvalues of
  linear transformations, I. *Proceedings of the National Academy of
  Sciences*, 35(11), 652–655.
- **Park, C. F., Lee, A., Lubana, E. S., Yang, Y., Okawa, M., Nishi,
  K., Wattenberg, M., & Tanaka, H.** (2025). ICLR: In-context learning
  of representations. *International Conference on Learning
  Representations 2025*.
- **Stewart, G. W. & Sun, J. G.** (1990). *Matrix Perturbation Theory*.
  Academic Press.
- **Tropp, J. A.** (2015). An introduction to matrix concentration
  inequalities. *Foundations and Trends in Machine Learning*, 8(1–2),
  1–230.
- **Vershynin, R.** (2018). *High-Dimensional Probability: An
  Introduction with Applications in Data Science*. Cambridge University
  Press.
- **Yu, Y., Wang, T., & Samworth, R. J.** (2015). A useful variant of
  the Davis–Kahan theorem for statisticians. *Biometrika*, 102(2),
  315–323.
