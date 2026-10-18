# Landscape scheme: almost every gradient trajectory ends at a global optimum

Started 2026-10-18 at the user's request. Status: active. Work solo and follow
AGENTS.md. Only Lean results without `sorry` count as proved: check
`dt show` or the `INDEX.md` status before you use one. Literature, hand
arguments and numerics are hypotheses until they are proved in Lean.

## Objective and constraints

The goal is one reusable proof pattern for gradient descent and for gradient
flows. It has three inputs and a bridge:

- **(A) Avoidance:** from almost every start, the trajectory converges to no
  strict saddle.
- **(B) Convergence:** every trajectory converges. This follows from the
  Łojasiewicz inequality at the critical points.
- **(C) Landscape:** every critical point is a global optimum or a strict
  saddle.
- **(D) Bridge:** from a manuscript's model to the energy that (A)–(C) speak
  about.

From these the pattern concludes that almost every trajectory converges to a
global optimum. It is then applied to the phase model of grokking
(`Transformer.PhaseModel`) and to the particle models of arXiv:2312.10794v5
(`Transformer.Perspective`).

Constraints:

- **No conditional results on unproved claims.** Do not add a theorem whose
  hypothesis is unproved in Lean. That excludes `strict_saddle` until M3.1
  proves it (arXiv:2507.22857 answers `o:strictsaddle` on paper) and every
  other unproved landscape claim.
- **Łojasiewicz enters only as a hypothesis.** Łojasiewicz's theorem
  (lojasiewicz1963propriete) is proved in the literature, not in Lean. It may
  enter only as an explicit hypothesis, as `hloj` does in
  `SaddleAvoidance.ae_exists_tendsto_isLocalMin`.
- **Never weaken a paper's statement under its name.** Partial regimes get
  their own names, and the paper's theorem stays `sorry`.
- **Work through the board.** At the user's request, the board was created
  at the root on 2026-10-18 with tasks #1–#31. After M0.1, #13 and #15 were
  removed and #32–#37 added. Every title starts with its milestone code;
  list them with `agent-kanban list --tag landscape`. Claim a task before
  working on it.

## Proved on 2026-10-18

| Layer | Lean | Status |
| --- | --- | --- |
| A, gradient descent | `SaddleAvoidance.ae_not_tendsto_of_isStrictSaddle`; global form `measure_tendsto_symmetricUnstable_eq_zero` | proved |
| B, gradient descent | `SaddleAvoidance.exists_tendsto_of_isLojasiewiczAt`, which converges from a Łojasiewicz limit point taken modulo translations `Λ`; `isLojasiewiczAt_of_injective` | proved |
| A + B, gradient descent | `SaddleAvoidance.ae_exists_tendsto_isLocalMin`, with hypotheses `hcrit`, `hloj` and bounded orbits | proved |
| B, flows | `Normalization.lojasiewicz_modulated_of_local_gradient_inequality`, which needs a compact `K`, an analytic `E` and the local inequality at every critical point; `analytic_gradient_inequality_of_nondegenerate` | proved |
| B, degenerate analytic case | `Normalization.lojasiewicz_modulated` | sorry |
| Flows | `GlobalFlow.IsFlowField.contDiff_flow`, `hasFDerivAt_flow` (the variational equation), `flow_reverse` | proved |
| C, phase model | `PhaseModel.exists_sub_characterPhase_mem_periods_of_isLocalMin` and `isStrictSaddle_of_divisionEnergy_lt` for `V < n²/8` | proved |
| C, `𝖤_0` on spheres | `Perspective.yury_lemma` | proved |
| p:beta0 | `Perspective.beta0_consensus`, stated after its proof in `AppendixA_Saddle`, from `almost_sure_consensus_beta0`, via an explicit inequality for `𝖤_0` and a cone argument at the saddles | proved |
| Bounds, periodic energies (M1.4) | `SaddleAvoidance.exists_lipschitzWith_gradient_of_periodic`: a `C²` function invariant under a set of translations spanning `E` has a bounded gradient, a bounded Hessian and a globally Lipschitz gradient; `exists_lipschitzWith_gradient_of_isCompact` for a compact set of representatives | proved |
| B, phase model (M3.4a) | `PhaseModel.injective_fderiv_gradient_divisionEnergy_univ` and `isLojasiewiczAt_divisionEnergy_univ`: the Hessian of `V_full` is injective at the characters up to periods, and Łojasiewicz holds there with exponent `1/2`; `isLojasiewiczAt_divisionEnergy_of_mem_span` at every zero of a generating `V_S` | proved |
| B, under a translation symmetry (M2.1) | `SaddleAvoidance.isLojasiewiczAt_of_invariant`: if `f` is invariant under the translations by a subspace `W` and its Hessian at a critical point is injective on `Wᗮ`, Łojasiewicz holds there with exponent `1/2`; the Hessian of an invariant `f` maps into `Wᗮ` (`mem_orthogonal_of_hasFDerivAt_gradient`) | proved |
| B, glue to Normalization (M2.3) | `SaddleAvoidance.exists_isLojasiewiczAt_of_local_gradient_inequality`: the local inequality in the form of `hlocal` of `lojasiewicz_modulated_of_local_gradient_inequality` gives `IsLojasiewiczAt` with an exponent `0 ≤ a < 1`, the form of `hloj`; `exists_isLojasiewiczAt_of_analyticAt_low_dimension` at every critical point of an analytic function on `EucSpace N`, `N ≤ 1` | proved |
| C, reduction on the circle (M3.1a) | `Perspective.sum_sin_mul_exp_eq_zero_of_isCritical`: a critical point of `𝖤_β` is stationary (C1) for the kernel `sin x · e^{β (cos x - 1)}`; `cutStable_of_torusHessianNonPos`: a point with no positive second derivative is cut-stable (C2) for every `S`, so a negative cut sum makes a strict saddle (`isStrictSaddleTorusEnergy_of_cut_neg`); synchronized angles are global maxima (`isGlobalMaxTorusEnergy_of_synchronized`) | proved |
| B, flows (M2.2) | `SaddleAvoidance.exists_tendsto_gradientFlow_of_isLojasiewiczAt`: a gradient flow converges when a point at which the inequality holds is a limit point of it taken modulo translations `Λ`; `tendsto_gradientFlow_of_isLojasiewiczAt` for a cluster point. It reuses `Normalization.converges_of_displacement_bound` | proved |
| A + B, global minimizers (M4.1) | `SaddleAvoidance.ae_exists_tendsto_isMinOn`: if every critical point is a global minimizer or a strict saddle, given `hloj` and a limit point modulo `Λ`, almost every orbit of gradient descent converges to a global minimizer; `ae_exists_tendsto_isMinOn_of_span_eq_top` needs no limit point when `Λ` spans `E` | proved |
| A, time-one map of a gradient flow (M1.2) | `SaddleAvoidance.gradientFlow`, the flow of `-∇f` for `f` of class `C²` with `∇f` bounded and Lipschitz (`isFlowField_neg_gradient`): `contDiff_gradientFlow` and `det_fderiv_gradientFlow_ne_zero` for `t ≥ 0`; at a critical point the derivative is `exp (-t ∇²f z)` (`hasFDerivAt_gradientFlow_of_gradient_eq_zero`); `IsStrictSaddle.isSymmetricUnstable_gradientFlow` for the time-one map | proved |
| C, spheres for `d ≥ 3` (M3.3) | `Perspective.exists_secondDerivEBetaAt_pos_of_nonTrivial`: for `d ≥ 3` and `β > 0`, every critical point of `𝖤_β` with two distinct tokens has a positive second derivative along a normalized line `s ↦ (x_i + s v_i)/‖x_i + s v_i‖`; `synchronized_of_secondDerivEBetaAt_nonpos`, `thm:spheres` of arXiv:2405.18273; `secondDerivEBetaAt_eq_two_mul_hessQuad`, the `key` of `hessian_at_critical_intrinsic` | proved |
| C, criterion on the circle (M3.1c) | `Perspective.taumax_sharp`: for a `2π`-periodic `C³` kernel with `f' < 0` off `[-τ, τ]`, a stationary (C1) configuration that is cut-stable (C2) for every `S` and not synchronized has `4 (1/τ + 1/τ_max) f'(0) ≤ ∫_{-π}^{π} \|f'''\|₊` (`taumax`, `lemma:taumax`); `synchronized_of_circleGapMax_lt` (`cor:main`) and `synchronized_of_integral_le` (`thm:main`), with `f'(0) > 0` added (`zero_kernel_counterexample`) | proved |
| A, gradient descent from a compact sublevel set (M1.6) | `SaddleAvoidance.exists_pos_measure_sublevel_tendsto_strictSaddle_eq_zero`: for `f` of class `C²` with `{f ≤ c}` compact, every step size below some `η₀ > 0` keeps gradient descent in `{f ≤ c}` (`mapsTo_gradientStep_sublevel`), and the starts in `{f ≤ c}` converging to a strict saddle form a null set; `measure_inter_tendsto_strictSaddle_eq_zero` on any forward-invariant set with a Hessian bound there, from `measure_inter_tendsto_symmetricUnstable_eq_zero` | proved |
| D, angles and the circle (M5.1) | `Perspective.usa_anglesToSphere`: solutions of `eq:onangles` give solutions of USA on `SphereTuple 2 n`; `eq_map_angleCube`: every `UniformTuple 2 n P` is the image of the normalized Lebesgue measure on `[0, 2π)ⁿ`; `measure_eq_zero_of_volume_preimage`: a set whose preimage in angles is Lebesgue-null is `P`-null, measurable or not, so `ae_of_ae_volume` carries an almost-everywhere statement in angles to `P`, and `measure_image_eq_zero` takes a `2π`-periodic null set to a `P`-null set | proved |
| C, phase model over finite abelian groups (M3.4d) | `PhaseModel.Abelian.exists_sub_characterPhase_mem_periods_of_isLocalMin`, `isStrictSaddle_of_divisionEnergy_lt` and `ae_tendsto_characterPhase`: the landscape below `n²/8` of `V_full` for a finite abelian group `G` of order `n ≥ 4`, with the characters `AddChar G ℂ` and their phases `arg χ(x)`; Parseval's identity and the cubic identity over the characters (`sum_normSq_fourier`, `sum_normSq_mul_re_fourier`) | proved |
| A, gradient flows (M1.3) | `SaddleAvoidance.ae_not_tendsto_gradientFlow_of_isStrictSaddle`: for `f` of class `C²` with `∇f` bounded and Lipschitz and `ν` absolutely continuous, `ν`-almost every gradient flow converges to no strict saddle; the flow is a one-parameter group (`gradientFlow_gradientFlow`) whose integer times iterate the time-one map (`iterate_gradientFlow_one`); ascent, the flow of `-f`, almost never converges to a critical point with a positive Hessian eigenvalue (`measure_tendsto_gradientAscent_strictSaddle_eq_zero`, the Euclidean analog of `l:nosaddleconv`) | proved |
| A + B, gradient flows (M4.2) | `SaddleAvoidance.ae_exists_tendsto_gradientFlow_isMinOn`: if every critical point is a global minimizer or a strict saddle, given `hloj` and a limit point modulo `Λ`, almost every trajectory of the gradient flow converges to a global minimizer; a convergent trajectory ends at a critical point (`gradient_eq_zero_of_tendsto_gradientFlow`); `ae_exists_tendsto_gradientFlow_isMinOn_of_span_eq_top` needs neither a limit point nor bounds on `∇f` when `Λ` spans `E` | proved |
| C, the kernel `f_β` for `0 < β < 1` (M3.1f) | `Perspective.τ_mul_integral_posPart_f_β'''_lt_four_of_lt_one`: `τ_β ∫_{-π}^{π} \|f_β'''\|₊ < 4` for `0 < β < 1`, the bound of `lemma:beta_0.75`, `lemma:beta_0.5`, `lemma:beta_0.33` and `lemma:beta_final`, with other constants: `-A ≤ f_β'' ≤ U` on `[0, π]` from pieces of `cos x` (`f_β''_mem_Icc_of_isChain`), `τ_β ≤ π/2 - cos τ_β` (`τ_β_star_two_le`), on nine intervals of `β` | proved |
| A, symmetrizable derivatives and flows in a metric (M1.5) | `SaddleAvoidance.measure_tendsto_symmetrizableUnstable_eq_zero`: the starts converging to fixed points whose derivative is conjugate to a symmetric map with an eigenvalue of modulus greater than one form a null set; `ae_not_tendsto_metricGradientFlow_of_isStrictSaddle`: for `f` of class `C²` and `G` of class `C¹` with `G ∇f` bounded and Lipschitz and `G` symmetric positive definite at the strict saddles, `ν`-almost every trajectory of `ẋ = -G(x) ∇f(x)` converges to no strict saddle | proved |

## Milestones

### M0 — Sources and board

Task: M0.1 #1, done on 2026-10-18.

| Item | Deliverable | Status |
| --- | --- | --- |
| M0.1 | Read arXiv:2507.22857 and arXiv:2405.18273. For each, record which `β` and `d` make every non-maximal critical point of `𝖤_β` a strict saddle, and how. | done; findings below |

**Answer.** In the regimes below, every non-maximal critical point of `𝖤_β`
is a strict saddle. These are results on paper and stay hypotheses until M3
proves them in Lean. So `o:strictsaddle` is answered: yes, for every
`β > 0`. Both papers take convergence from Łojasiewicz's theorem. Their
landscape arguments are elementary.

| `d` | `β` | Source | Proof route |
| --- | --- | --- | --- |
| 2 | every `β ≥ -0.16`, so every `β > 0` | arXiv:2507.22857, `cor:conjecture` (§2), proved in §5 from `thm:main` | cut-stability and integration by parts on the gaps (§3) |
| `≥ 2` | `0 < β ≤ 1` | arXiv:2405.18273, `cor:smallbeta` from `thm:taylorphi` (§4) | Schur product theorem, a rank argument, `lem:hemisphere` |
| `≥ 2` | `β ≥ n²/π²` | arXiv:2405.18273, `cor:largebeta` from `thm:largebeta` (App. E): the survey's argument, repeated | cut directions `1_S`, a spanning tree with edges `≤ π/n`, `lem:hemisphere` |
| `≥ 3` | every `β > 0` | arXiv:2405.18273, `cor:spheres` from `thm:spheres` (Markdahl et al., Thm 13; direct proof in App. C) | every particle moved toward one common direction |

**arXiv:2507.22857v1** (Polyanskiy, Rigollet, Yao, *Synchronization of
mean-field models on the circle*):

- **`thm:main` (§2)** is about `ẋ_i = -Σ_j f(x_i - x_j)` on `𝕋ⁿ` (S1).
  Take `τ ∈ (0, π]` with `f' < 0` off `[-τ, τ]`. If
  `τ ∫_{-π}^{π} |f'''|₊ ≤ 4 (1 + τ/2π) f'(0)`, every stationary point is
  locally unstable or synchronized. **`cor:main` (§3)** replaces `2π` by
  any `M` above the largest gap `τ_max` of the stable, stationary,
  non-synchronized points.
- **The proof (§3)** uses stationarity (C1), `Σ_j f(x_i - x_j) = 0`, and
  cut-stability (C2), `Σ_{i ∈ S, j ∉ S} f'(x_i - x_j) ≥ 0`.
  - `lemma:twoconsecutive_general`: at most one gap exceeds `τ`.
  - `lemma:taumax`: `⟨1, |f'''|₊⟩ ≥ min(8/τ, 4/τ + 4/τ_max) f'(0)`. It
    integrates `φ'' = Σ_j f'''(· - x_j)` against
    `Ψ_i(x) = (a_i/2)(x - θ_i)(θ_{i+1} - x)` on each gap, twice by parts.
    By (C1), `∫ φ` over each gap is `0`.
- **Two source fixes.**
  - `thm:main` and `cor:main` need `f'(0) > 0`. Counterexample: `f ≡ 0`
    with `τ = π` meets the hypotheses, yet every configuration is stationary
    and none is unstable.
  - (C2) is stated only for sets `S` of coincident particles, but
    `lemma:twoconsecutive_general` applies it to an arc. At a point with
    non-positive Hessian, §4 (`remark:negative_semidef`) gives
    `Σ_{i,j} (c_i - c_j)² f'(x_i - x_j) ≥ 0` for every `c ∈ ℝⁿ`, so (C2)
    holds for every `S`. Formalize (C2) for every `S`.
- **§4.** The direction `c = 1_S` is the one of
  `secondDeriv_torusEnergy_block`. The second derivative of `𝖤_β` along
  `1_S` is `-n⁻² Σ_{i ∈ S, j ∉ S} g_β(θ_i - θ_j)`, where
  `g_β_2d β = e^β f_β'`.
- **§5, `cor:conjecture`.** `f_β(x) = sin x e^{β(cos x - 1)}` satisfies the
  criterion for every `β ≥ -0.16`, with
  `τ = arccos((√(1 + 4β²) - 1)/(2β))`, which is `τ_β_star 2 β`. The cases:
  - `β > 1/3` (App. A): `lemma:positiveregion`, `lemma:tau1`,
    `corollary:tau2`, `lemma:beta1` (`β ≥ 1`), `lemma:beta_0.75`,
    `lemma:beta_0.5`, `lemma:beta_0.33`;
  - `0 < β ≤ 1/3` (App. B): `lemma:positiveregion2`, `lemma:beta_final`;
  - `β = 0`: the Kuramoto model;
  - `-0.16 ≤ β < 0` (App. C): `M = π`.

  `cor:negativebeta`: for `β < -2/3`, global synchronization fails when
  `3 ∣ n` or `n ≥ C_β`.
- **The constants of App. A–B** are stated without proof ("straightforward
  to verify"). A floating-point check on 2026-10-18 found all of them true,
  but several only by margins between `10⁻⁴` and `10⁻³`:
  - `max g¹ = 0.37490` against `0.375`;
  - `min g¹ = -1.38012` against `-1.381`;
  - `g²(0.18) = -0.24677` against `-0.247`;
  - `(1 + 3β) e^{β² - β} π/2 ≤ 1.9995` against `2` for `β ≤ 0.148`.

  The criterion itself has room: `τ ∫ |f_β'''|₊ ≤ 3.51` against `4` for
  `β ∈ [10⁻³, 10³]`, rising towards `3.51` as `β → ∞`. So the Lean proofs
  may use their own, looser constants. One typo: in `lemma:beta_0.5`,
  `√0.75 τ(√0.75)` should read `√0.75 τ(0.75)`; both are below `0.88`. A
  floating-point check is not a proof.

**arXiv:2405.18273v2** (Criscitiello, Rebjock, McRae, Boumal,
*Synchronization on circles and spheres with nonlinear interactions*) treats
`½ Σ w_ij φ(x_iᵀx_j)` on `(𝕊^{d-1})ⁿ`:

- **`thm:spheres`** (§1, Markdahl et al., Thm 13). Assume `d ≥ 3`, a
  connected graph, and `φ' > 0`, `φ'' ≥ 0` on `[-1, 1]`. Then critical
  points with negative semidefinite Hessian are the synchronized global
  maxima; `cor:spheres` applies this to `φ_β` for every `β > 0`. The direct
  proof (App. C, `eq:socpphiconvex`, `eq:almostthere`) moves every particle
  toward one random `γ`: `Ẋ = Proj_X(γ𝟙ᵀ)` and
  `𝔼[ẊᵀẊ] = (d - 2)𝟙𝟙ᵀ + (XᵀX)^{⊙2}`. With the first-order condition this
  gives `Σ w_ij φ'(x_iᵀx_j)(1 - x_iᵀx_j)² ≤ 0`. In Lean, a sum over an
  orthonormal basis replaces the expectation.
- **`thm:taylorphi`** (§1, proved in §4): `d ≥ 2`, `φ' > 0`, `φ'' ≥ 0`,
  `w_ij = q_i q_j` with `q > 0`, and Taylor coefficients
  `a_0 > a_1 ≥ a_2 ≥ ⋯` of `φ'`. The proof uses the Schur product theorem,
  the directions `α = Xᵀu`, a rank argument and `lem:hemisphere` (§4,
  proved in App. B).
- **`thm:largebeta`** (§1, proved in App. E): `d = 2`, a connected graph,
  `φ' > 0`, and `h(t) = tφ' - (1 - t²)φ'' < 0` for `-1 ≤ t < cos(π/n)`.
  The cut directions `1_S` grow a spanning tree with edges `≤ π/n`, so the
  particles lie in a half circle, and `lem:hemisphere` concludes. App. E
  credits the argument to the survey.
- **`lem:socpcircles`** (§3): for `d = 2` the Hessian is `-L(M)`, with
  `m_ij = w_ij h(x_iᵀx_j)`.
- **`thm:circles`** (§1, proved in §5): for `d = 2`, the complete graph and
  `φ_{ε,τ}(t) = ε log(1 + e^{(t - τ)/ε})`, which has `φ', φ'' > 0`, some
  `ε`, `τ` give spurious local maxima for every `n ≥ 5`. So on the circle,
  convexity of `φ` does not decide the landscape.
- Line 198 cites arXiv:2507.22857 for `β ≥ -0.16` on the circle.

**Lean status** on 2026-10-18 (`dt show`, `INDEX.md`):

- Proved and ready to use:
  - for the circle: `secondDeriv_torusEnergy_block`, `taylor3_inequality`
    (which is (C2) for every `S` at a point with non-positive Hessian),
    `hasDerivAt_torusEnergy`, `τ_β_star_spec`, `τ_β_star_unique` and
    `cosTauStar_pos`;
  - for `d ≥ 3`: `secondDeriv_selfEnergy`, `dr1_skew_inequality`,
    `claim_yury` and `hessian_at_critical_intrinsic`. The internal `key` of
    the last one says that the second derivative along any curve with
    velocity `v` is `2 · hessQuad X v`.
- Mathlib: `intervalIntegral.integral_mul_deriv_eq_deriv_mul`,
  `intervalIntegral.integral_eq_sub_of_hasDerivAt`, and
  `Matrix.PosSemidef.hadamard` (the Schur product theorem).
- Nothing else above is in Lean.

### M1 — Layer A for flows (general, `Transformer.SaddleAvoidance`)

Tasks: M1.1 #2, M1.2 #3, M1.3 #4, M1.4 #5, M1.5 #6, M1.6 #7.

| Item | Deliverable | Depends on |
| --- | --- | --- |
| M1.1 | **Linear flow.** For a constant symmetric `A`, `J τ = exp (τ A)` solves `J' = A J` with `J 0 = id`; `exp A` is symmetric; an eigenvalue `μ` of `A` gives the eigenvalue `e^μ` of `exp A`. Mathlib: `hasDerivAt_exp_smul_const`, `IsSelfAdjoint.exp`. | — |
| M1.2 | **Time-one map, done.** `Φ = flow (-∇f) 0 1` for `f` of class `C²` with `∇f` bounded and Lipschitz, which is `IsFlowField`. `Φ` is `C¹` (`contDiff_flow`) with an invertible derivative (backward flow, `flow_reverse`). At a critical point, `Φ z = z` and `DΦ z = exp (-∇²f z)`, obtained by feeding `J τ = exp (-τ ∇²f z)` to `hasFDerivAt_flow`. A strict saddle is therefore `IsSymmetricUnstable Φ z`. | M1.1 |
| M1.3 | **Avoidance for flows, done.** `ae_not_tendsto_gradientFlow_of_isStrictSaddle`: for `ν`-almost every `x`, `t ↦ gradientFlow f t x` converges to no strict saddle, because integer times are iterates of `Φ` (`iterate_gradientFlow_one`). Ascent is the same statement for `-f` (`measure_tendsto_gradientAscent_strictSaddle_eq_zero`). | M1.2 |
| M1.4 | **Periodic energies, done.** `exists_lipschitzWith_gradient_of_periodic`: if `f` is `C²` and invariant under a set of translations spanning `E`, such as a lattice, then `∇f` is bounded and globally Lipschitz. This supplies the hypotheses of M1.2 and of the descent theorems for `𝖤_β` on angles and for `V_full`. | — |
| M1.5 | **Symmetrizable derivatives, done.** `IsSymmetrizableUnstable`: the derivative is conjugate to a symmetric map by a linear isomorphism; `measure_tendsto_symmetrizableUnstable_eq_zero` extends `measure_tendsto_symmetricUnstable_eq_zero` to it. For `ẋ = -G(x) ∇f(x)` with `G` of class `C¹` (`metricGradientFlow`), the time-one map is `C¹` with an invertible derivative, `exp (-G(z) ∇²f(z))` at a critical point; where `G(z)` is symmetric positive definite, `G(z) ∇²f(z)` is conjugate to `S ∇²f(z) S`, `S² = G(z)` (`exists_sqrt`), so a strict saddle is `IsSymmetrizableUnstable` (`IsStrictSaddle.isSymmetrizableUnstable_metricGradientFlow`), and `measure_tendsto_metricGradientFlow_strictSaddle_eq_zero` follows. SA and the sphere bridge M5.3 need it. | M1.3 |
| M1.6 | **Optional, for networks, done.** Gradient descent with a compact forward-invariant sublevel set in place of a global Lipschitz constant. | — |

### M2 — Layer B, convergence

Tasks: M2.1 #8, M2.2 #9, M2.3 #10, M2.4 #11.

| Item | Deliverable | Depends on |
| --- | --- | --- |
| M2.1 | **Łojasiewicz under a symmetry, done.** `isLojasiewiczAt_of_invariant`: if `f` is invariant under translations by a subspace `W` and its Hessian at the critical point `z` is injective on `Wᗮ`, then `IsLojasiewiczAt f z (1/2)`. Proof: restrict to `z + Wᗮ` and apply `isLojasiewiczAt_of_injective`. It covers the synchronized states of `𝖤_β`, where `W = ℝ𝟙`. | — |
| M2.2 | **Convergence for flows, done.** The flow analogue of `exists_tendsto_of_isLojasiewiczAt`, with translations `Λ`: a limit point taken modulo `Λ` at which the inequality holds is the limit. First check what the proof of `lojasiewicz_modulated_of_local_gradient_inequality` already gives; reuse it rather than duplicate it. | — |
| M2.3 | **Glue to Normalization, done.** Convert the form `\|E y - E z\| ^ α ≤ k ‖∇E y‖` used by the `hlocal` hypothesis of `lojasiewicz_modulated_of_local_gradient_inequality` into `IsLojasiewiczAt`, so the hard core plugs in once proved. | — |
| M2.4 | **Hard core, not this plan's work.** Łojasiewicz at degenerate critical points of analytic functions: the remaining input of `Normalization.lojasiewicz_modulated`. The survey proves convergence of every trajectory by Łojasiewicz's theorem. For `β = 0` the repository replaced it by an explicit inequality (`AppendixA_Lojasiewicz`); for `β > 0` no replacement is known. So `beta_tiny`, `beta_tiny_circle`, `beta_interval` and `boumal_clustering` cannot close before M2.4. | Normalization |

### M3 — Layer C, landscape

Tasks: M3.1 #12, with parts M3.1a–f #32–#37; M3.3 #14; M3.4a #16; M3.4b
#17; M3.4c #18; M3.4d #19.

M3.2 (`β ≤ 1` and `β ≥ n²/π²` on the torus, #13) and M3.5 (`β ≤ c/n` on
spheres, #15) were dropped on 2026-10-18. M3.1 covers every `β > 0` on the
circle, and M3.3 covers every `β > 0` for `d ≥ 3`.

M3.1 follows arXiv:2507.22857. Its parts are new modules of
`Transformer.Perspective`, with their own names and citing the paper;
arXiv:2507.22857 is not formalized as a paper of its own. M3.1 is pure
landscape and needs neither A nor B: it is the only Perspective `sorry` that
can close without M2.4.

| Item | Deliverable | Depends on |
| --- | --- | --- |
| M3.1a | **Reduction, done.** Criticality (`IsCriticalTorusEnergy`) gives (C1) through `hasDerivAt_torusEnergy`. A point with no positive second derivative satisfies `TorusHessianNonPos`, so `taylor3_inequality` gives (C2) for every `S`. Synchronized angles are global maxima. | — |
| M3.1b | **Gaps.** Order the distinct positions as `θ_1 < ⋯ < θ_K` in `[0, 2π)` and take the gaps `Δ_k`. If (C2) holds for every `S` and `f' < 0` off `[-τ, τ]`, at most one gap exceeds `τ` (`lemma:twoconsecutive_general`). | — |
| M3.1c | **The criterion for a general `f`, done.** For `f` `2π`-periodic, `C³`, with `f'(0) > 0`: first `lemma:taumax`, then `cor:main` and `thm:main`. A stationary configuration that is cut-stable for every `S` is synchronized. | M3.1b |
| M3.1d | **The kernel `f_β`.** `f_β' < 0` off `[-τ_β, τ_β]`, where `τ_β = τ_β_star 2 β`. The sign regions of `f_β'''` come from the quartic `p(cos x)` (`lemma:positiveregion`, `lemma:positiveregion2`). `∫ \|f_β'''\|₊` is expressed through `f_β''` at the sign changes. | — |
| M3.1e | **`β ≥ 1`, done:** `τ_β ∫ \|f_β'''\|₊ < 4` (`τ_mul_integral_posPart_f_β'''_lt_four`, `Perspective.SyncKernelBetaOne`). `lemma:beta1` holds for all `a, b ∈ [0, π]`, from `-(8/5)√β ≤ f_β'' ≤ (2/5)√β` on `[0, π]`. `corollary:tau2`, `τ_β < 1/√β` for every `β > 0`, comes from `cos u ≤ 1 - u²/2 + u⁴/24` in place of `lemma:tau1`. | M3.1d |
| M3.1f | **`0 < β < 1`, done:** the same bound (`lemma:beta_0.75`, `lemma:beta_0.5`, `lemma:beta_0.33`, `lemma:beta_final`). Fallback for `0 < β ≤ 1`: `thm:taylorphi` of arXiv:2405.18273, which needs the second derivative along every direction, not only along `1_S`. | M3.1d |
| M3.1 | **`Perspective.strict_saddle`.** By M3.1a, a non-maximal critical point is not synchronized. M3.1c with M3.1e–f then gives an `S` with `Σ_{i ∈ S, j ∉ S} g_β(θ_i - θ_j) < 0`. `secondDeriv_torusEnergy_block` turns `1_S` into a direction of positive second derivative. Rewrite the docstring: the survey leaves the problem open, and arXiv:2507.22857 answers it. | M3.1a, M3.1c, M3.1e, M3.1f |
| M3.3 | **`d ≥ 3`, done.** The non-synchronized critical points of `𝖤_β` on `(𝕊^{d-1})^n` are strict saddles, as a new theorem with its own name (arXiv:2405.18273, App. C). Export the `key` of `hessian_at_critical_intrinsic` (second derivative `= 2 · hessQuad X v`). Sum `hessQuad X (Proj_X(e_k 𝟙ᵀ))` over an orthonormal basis `e_k`. The curve `s ↦ (x_i + s v_i)/‖x_i + s v_i‖` has velocity `v`. | — |
| M3.4a | **Characters are nondegenerate, done.** The Hessian of `V_full` at a character is injective, since the rows `e_x - e_y - e_{x-y}` span `ℝⁿ`. This gives Łojasiewicz with exponent `1/2` at the characters. | — |
| M3.4b | **Research: `V_full` above `n²/8`, searched.** `scripts/phase_minima.py` (inside `python/`: `uv run --locked python ../scripts/phase_minima.py`, 44 min; seeded by `n`, so `--largest 8` reproduces rows 2–8) runs 4000 steps of gradient descent of step `1/(3n)` for each `2 ≤ n ≤ 30`. It starts from 400 uniform phases, 400 points of `{0, π}ⁿ`, 400 of `(2π/m)ℤⁿ` for `m ∈ {3, 4, 5, 6, 8}`, 400 mixtures of two characters, and 5 copies of each non-character critical point that Newton's method reaches from 200 uniform starts; every start but the uniform ones is moved by `10⁻²`. All 1600–2355 runs per `n` end at a character up to periods (`V < 10⁻⁸`), so no local minimizer with `V ≥ n²/8` was found. No Newton critical point has a positive semidefinite Hessian. The lowest non-character critical values found are `2, 4, 8, 12, 16, 24, 32, 36, 48, 60, 84` for `n = 2, …, 12`, all above `n²/8`; Newton's method is not exhaustive, finds few points for `n ≥ 18` and none for `n = 25` and `27 ≤ n ≤ 30`. **Hypothesis, not proved:** for `n ≤ 30`, every local minimizer of `V_full` is the phases of a character up to periods. | lab |
| M3.4c | **Research: partial tables `V_S`.** The analogy is Kuramoto on Erdős–Rényi graphs just above connectivity (abdalla2022expander, cited in arXiv:2312.10794v5 §6). A comment in that survey's TeX source (line 908) says the reference proves only that a convergent trajectory ends synchronized. So this is an analogy and a hypothesis. | — |
| M3.4d | **Finite abelian groups, done.** `PhaseModel.Abelian` generalizes `Fourier` and `Landscape`, with the chain from `Division` to `Characters` they rest on, from `ℤ/n` to a finite abelian group `G`, with the characters `AddChar G ℂ`. | — |

### M4 — The scheme theorems (general)

Tasks: M4.1 #20, M4.2 #21, M4.3 #22.

| Item | Deliverable | Depends on |
| --- | --- | --- |
| M4.1 | **Gradient descent, done.** `ae_exists_tendsto_isMinOn`: if every critical point is a global minimizer or a strict saddle, given `hloj` and a limit point modulo `Λ`, then almost every orbit converges to a global minimizer. Periodic corollary: when `Λ` is a lattice spanning `E`, every orbit has a limit point modulo `Λ`, so that hypothesis is free. | M3.4a for the example |
| M4.2 | **Gradient flows, done.** `ae_exists_tendsto_gradientFlow_isMinOn`: the same statement for gradient flows, whose limits are critical (`gradient_eq_zero_of_tendsto_gradientFlow`); `ae_exists_tendsto_gradientFlow_isMinOn_of_span_eq_top` needs no limit point when `Λ` spans `E`. | M1.3, M2.2 |
| M4.3 | **`𝖤_β` on angles (USA).** For every `β > 0`, almost every start converges to a synchronized state, given `hloj` at the non-synchronized critical points. M2.1 covers the synchronized ones. | M3.1, M4.2, M1.4, M2.1 |

### M5 — Bridges (D)

Tasks: M5.1 #23, M5.2 #24, M5.3 #25, M5.4 #26.

| Item | Deliverable | Depends on |
| --- | --- | --- |
| M5.1 | **Angles and the circle, done.** `θ ↦ (cos θ_i, sin θ_i)` carries `angularUSA` to USA on `SphereTuple 2 n`. `UniformTuple 2 n P` is the image of normalized Lebesgue measure on `[0, 2π)ⁿ`; the uniform law is unique (`57993faf`). A `2π`-periodic Lebesgue-null set therefore maps to a `P`-null set. | — |
| M5.2 | **SA in angles.** SA in angles is a gradient flow in the metric `diag(Z_i)`. | M1.5 |
| M5.3 | **Spheres for `d ≥ 3`, without manifolds.** On `(ℝ^d∖0)^n`, the field `F_i(x) = ‖x_i‖² ∇_i(𝖤_β ∘ normalize)(x)` has the sphere flow on every slice. Radially invariant Lebesgue-null sets correspond to uniform-null sets (`Measure.measurePreserving_homeomorphUnitSphereProd`). | M1.5 |
| M5.4 | **Research: grokking network to phase model.** No Lean until a statement is fixed on paper. | — |

## Order of work

1. M0.1: done on 2026-10-18.
2. M1.1 → M1.2 → M1.3, then M1.4.
3. M2.1, M2.3, M3.4a, then M4.1.
4. M2.2 → M4.2.
5. M3.1a, M3.1b → M3.1c, and M3.1d → M3.1e, M3.1f. These are independent
   chains, so separate agents can work on them. Then M3.1, and M3.3.
6. M5.1 → M4.3.
7. M1.5 → M5.2, M5.3.

The research items M2.4, M3.4b, M3.4c and M5.4 wait for an argument on paper.

## Target `sorry`s

| `sorry` | Task | Closed by | Blocked on |
| --- | --- | --- | --- |
| `Perspective.strict_saddle` | #12 | M3.1, with parts M3.1a–f (#32–#37) | — |
| `Perspective.beta_tiny_circle` | #27 | M3.1, M4.3, M5.1, M5.2 | M2.4 |
| `Perspective.beta_tiny` | #28 | M2.1, M3.1 (`d = 2`), M3.3 (`d ≥ 3`), M4.2, M5.1–M5.3 | M2.4 |
| `Perspective.beta_interval` | #29 | M2.1, M3.1 (`d = 2`), M3.3 (`d ≥ 3`), M4.2, M5.1–M5.3 | M2.4 |
| `Perspective.boumal_clustering` | #30 | M2.1, M3.3, M4.2, M5.3 | M2.4 |
| `Normalization.lojasiewicz_modulated` | #11 | M2.4 | Normalization line |
| `Perspective.beta0_consensus` | #31 | outside the scheme, done | — |

`Perspective.beta0_consensus` follows from `almost_sure_consensus_beta0`.
`Section3_SmallBeta` comes before `AppendixA_Saddle` in the import order, so the
statement moved to the end of `AppendixA_Saddle`, where it is proved.

The other `sorry`s of `INDEX.md` have no tasks yet.
