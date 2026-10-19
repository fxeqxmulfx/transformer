# Landscape scheme: almost every gradient trajectory ends at a global optimum

Started 2026-10-18 at the user's request. Status: active. Work solo and follow
AGENTS.md. Only Lean results without `sorry` count as proved: check
`dt show` or the `INDEX.md` status before you use one. Literature, hand
arguments and numerics are hypotheses until they are proved in Lean.

Active extension from 2026-10-19: [Landscape explanation of modular arithmetic
generalization](landscape_grokking_plan.md). It asks for a bridge to the actual
Rotate90 browser model, analytic SiLU/SwiGLU variants, and an explanation of
the gap between representability and gradient access for a scalar MLP. M0-M8
below remain the ledger of the framework's existing results.

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
(`Transformer.Landscape.PhaseModel`) and to the particle models of arXiv:2312.10794v5
(`Transformer.Perspective`).

Constraints:

- **No conditional results on unproved claims.** Do not add a theorem whose
  hypothesis is unproved in Lean. That excludes every unproved landscape
  claim. `strict_saddle` is proved (M3.1), so it may be used.
- **Łojasiewicz enters as a hypothesis, discharged for analytic energies.**
  The general theorems take the Łojasiewicz inequality as an explicit
  hypothesis, as `hloj` does in `Landscape.SaddleAvoidance.ae_exists_tendsto_isLocalMin`.
  Łojasiewicz's theorem (lojasiewicz1963propriete) is now proved in Lean
  (M2.4, `Landscape.SaddleAvoidance.exists_isLojasiewiczAt_of_analyticAt`), so an
  analytic energy satisfies `hloj` at every point.
- **Never weaken a paper's statement under its name.** Partial regimes get
  their own names, and the paper's theorem stays `sorry`.
- **Work through the board.** At the user's request, the board was created
  at the root on 2026-10-18 with tasks #1–#31. After M0.1, #13 and #15 were
  removed and #32–#37 added. M6, the reuse of the scheme, added #38–#47 the
  same day (`--tag reuse`). M7, one directory for the framework, and M8,
  conditions along the orbit, added #49–#53 the same day (`--tag layout`,
  `--tag orbit`). Every title starts with its milestone code;
  list them with `agent-kanban list --tag landscape`. Claim a task before
  working on it.

## Proved on 2026-10-18

| Layer | Lean | Status |
| --- | --- | --- |
| A, gradient descent | `Landscape.SaddleAvoidance.ae_not_tendsto_of_isStrictSaddle`; global form `measure_tendsto_symmetricUnstable_eq_zero` | proved |
| B, gradient descent | `Landscape.SaddleAvoidance.exists_tendsto_of_isLojasiewiczAt`, which converges from a Łojasiewicz limit point taken modulo translations `Λ`; `isLojasiewiczAt_of_injective` | proved |
| A + B, gradient descent | `Landscape.SaddleAvoidance.ae_exists_tendsto_isLocalMin`, with hypotheses `hcrit`, `hloj` and bounded orbits | proved |
| B, flows | `Landscape.Lojasiewicz.lojasiewicz_modulated_of_local_gradient_inequality`, which needs a compact `K`, an analytic `E` and the local inequality at every critical point; `analytic_gradient_inequality_of_nondegenerate` | proved |
| B, degenerate analytic case (M2.4) | `Landscape.Lojasiewicz.lojasiewicz_modulated`: an analytic energy that is not locally constant has compact monomial charts at every point (`hasCompactMonomialCharts_of_analyticAt`, from local uniformization `Landscape.Uniformization.hasMonomialCharts_of_analyticAt`), so the local inequality holds at every critical point (`analytic_gradient_inequality_of_compact_monomial_charts`) | proved |
| Flows | `Landscape.GlobalFlow.IsFlowField.contDiff_flow`, `hasFDerivAt_flow` (the variational equation), `flow_reverse` | proved |
| C, phase model | `Landscape.PhaseModel.exists_sub_characterPhase_mem_periods_of_isLocalMin` and `isStrictSaddle_of_divisionEnergy_lt` for `V < n²/8` | proved |
| C, `𝖤_0` on spheres | `Perspective.yury_lemma` | proved |
| p:beta0 | `Perspective.beta0_consensus`, stated after its proof in `AppendixA_Saddle`, from `almost_sure_consensus_beta0`, via an explicit inequality for `𝖤_0` and a cone argument at the saddles | proved |
| Bounds, periodic energies (M1.4) | `Landscape.SaddleAvoidance.exists_lipschitzWith_gradient_of_periodic`: a `C²` function invariant under a set of translations spanning `E` has a bounded gradient, a bounded Hessian and a globally Lipschitz gradient; `exists_lipschitzWith_gradient_of_isCompact` for a compact set of representatives | proved |
| B, phase model (M3.4a) | `Landscape.PhaseModel.injective_fderiv_gradient_divisionEnergy_univ` and `isLojasiewiczAt_divisionEnergy_univ`: the Hessian of `V_full` is injective at the characters up to periods, and Łojasiewicz holds there with exponent `1/2`; `isLojasiewiczAt_divisionEnergy_of_mem_span` at every zero of a generating `V_S` | proved |
| B, under a translation symmetry (M2.1) | `Landscape.SaddleAvoidance.isLojasiewiczAt_of_invariant`: if `f` is invariant under the translations by a subspace `W` and its Hessian at a critical point is injective on `Wᗮ`, Łojasiewicz holds there with exponent `1/2`; the Hessian of an invariant `f` maps into `Wᗮ` (`mem_orthogonal_of_hasFDerivAt_gradient`) | proved |
| B, glue to Normalization (M2.3) | `Landscape.SaddleAvoidance.exists_isLojasiewiczAt_of_local_gradient_inequality`: the local inequality in the form of `hlocal` of `lojasiewicz_modulated_of_local_gradient_inequality` gives `IsLojasiewiczAt` with an exponent `0 ≤ a < 1`, the form of `hloj`; `exists_isLojasiewiczAt_of_analyticAt_low_dimension` at every critical point of an analytic function on `EucSpace N`, `N ≤ 1` | proved |
| C, reduction on the circle (M3.1a) | `Perspective.sum_sin_mul_exp_eq_zero_of_isCritical`: a critical point of `𝖤_β` is stationary (C1) for the kernel `sin x · e^{β (cos x - 1)}`; `cutStable_of_torusHessianNonPos`: a point with no positive second derivative is cut-stable (C2) for every `S`, so a negative cut sum makes a strict saddle (`isStrictSaddleTorusEnergy_of_cut_neg`); synchronized angles are global maxima (`isGlobalMaxTorusEnergy_of_synchronized`) | proved |
| B, flows (M2.2) | `Landscape.SaddleAvoidance.exists_tendsto_gradientFlow_of_isLojasiewiczAt`: a gradient flow converges when a point at which the inequality holds is a limit point of it taken modulo translations `Λ`; `tendsto_gradientFlow_of_isLojasiewiczAt` for a cluster point. It reuses `Landscape.Lojasiewicz.converges_of_displacement_bound` | proved |
| A + B, global minimizers (M4.1) | `Landscape.SaddleAvoidance.ae_exists_tendsto_isMinOn`: if every critical point is a global minimizer or a strict saddle, given `hloj` and a limit point modulo `Λ`, almost every orbit of gradient descent converges to a global minimizer; `ae_exists_tendsto_isMinOn_of_span_eq_top` needs no limit point when `Λ` spans `E` | proved |
| A, time-one map of a gradient flow (M1.2) | `Landscape.SaddleAvoidance.gradientFlow`, the flow of `-∇f` for `f` of class `C²` with `∇f` bounded and Lipschitz (`isFlowField_neg_gradient`): `contDiff_gradientFlow` and `det_fderiv_gradientFlow_ne_zero` for `t ≥ 0`; at a critical point the derivative is `exp (-t ∇²f z)` (`hasFDerivAt_gradientFlow_of_gradient_eq_zero`); `IsStrictSaddle.isSymmetricUnstable_gradientFlow` for the time-one map | proved |
| C, spheres for `d ≥ 3` (M3.3) | `Perspective.exists_secondDerivEBetaAt_pos_of_nonTrivial`: for `d ≥ 3` and `β > 0`, every critical point of `𝖤_β` with two distinct tokens has a positive second derivative along a normalized line `s ↦ (x_i + s v_i)/‖x_i + s v_i‖`; `synchronized_of_secondDerivEBetaAt_nonpos`, `thm:spheres` of arXiv:2405.18273; `secondDerivEBetaAt_eq_two_mul_hessQuad`, the `key` of `hessian_at_critical_intrinsic` | proved |
| C, criterion on the circle (M3.1c) | `Perspective.taumax_sharp`: for a `2π`-periodic `C³` kernel with `f' < 0` off `[-τ, τ]`, a stationary (C1) configuration that is cut-stable (C2) for every `S` and not synchronized has `4 (1/τ + 1/τ_max) f'(0) ≤ ∫_{-π}^{π} \|f'''\|₊` (`taumax`, `lemma:taumax`); `synchronized_of_circleGapMax_lt` (`cor:main`) and `synchronized_of_integral_le` (`thm:main`), with `f'(0) > 0` added (`zero_kernel_counterexample`) | proved |
| A, gradient descent from a compact sublevel set (M1.6) | `Landscape.SaddleAvoidance.exists_pos_measure_sublevel_tendsto_strictSaddle_eq_zero`: for `f` of class `C²` with `{f ≤ c}` compact, every step size below some `η₀ > 0` keeps gradient descent in `{f ≤ c}` (`mapsTo_gradientStep_sublevel`), and the starts in `{f ≤ c}` converging to a strict saddle form a null set; `measure_inter_tendsto_strictSaddle_eq_zero` on any forward-invariant set with a Hessian bound there, from `measure_inter_tendsto_symmetricUnstable_eq_zero` | proved |
| D, angles and the circle (M5.1) | `Perspective.usa_anglesToSphere`: solutions of `eq:onangles` give solutions of USA on `SphereTuple 2 n`; `eq_map_angleCube`: every `UniformTuple 2 n P` is the image of the normalized Lebesgue measure on `[0, 2π)ⁿ`; `measure_eq_zero_of_volume_preimage`: a set whose preimage in angles is Lebesgue-null is `P`-null, measurable or not, so `ae_of_ae_volume` carries an almost-everywhere statement in angles to `P`, and `measure_image_eq_zero` takes a `2π`-periodic null set to a `P`-null set | proved |
| C, phase model over finite abelian groups (M3.4d) | `Landscape.PhaseModel.Abelian.exists_sub_characterPhase_mem_periods_of_isLocalMin`, `isStrictSaddle_of_divisionEnergy_lt` and `ae_tendsto_characterPhase`: the landscape below `n²/8` of `V_full` for a finite abelian group `G` of order `n ≥ 4`, with the characters `AddChar G ℂ` and their phases `arg χ(x)`; Parseval's identity and the cubic identity over the characters (`sum_normSq_fourier`, `sum_normSq_mul_re_fourier`) | proved |
| A, gradient flows (M1.3) | `Landscape.SaddleAvoidance.ae_not_tendsto_gradientFlow_of_isStrictSaddle`: for `f` of class `C²` with `∇f` bounded and Lipschitz and `ν` absolutely continuous, `ν`-almost every gradient flow converges to no strict saddle; the flow is a one-parameter group (`gradientFlow_gradientFlow`) whose integer times iterate the time-one map (`iterate_gradientFlow_one`); ascent, the flow of `-f`, almost never converges to a critical point with a positive Hessian eigenvalue (`measure_tendsto_gradientAscent_strictSaddle_eq_zero`, the Euclidean analog of `l:nosaddleconv`) | proved |
| A + B, gradient flows (M4.2) | `Landscape.SaddleAvoidance.ae_exists_tendsto_gradientFlow_isMinOn`: if every critical point is a global minimizer or a strict saddle, given `hloj` and a limit point modulo `Λ`, almost every trajectory of the gradient flow converges to a global minimizer; a convergent trajectory ends at a critical point (`gradient_eq_zero_of_tendsto_gradientFlow`); `ae_exists_tendsto_gradientFlow_isMinOn_of_span_eq_top` needs neither a limit point nor bounds on `∇f` when `Λ` spans `E` | proved |
| C, the kernel `f_β` for `0 < β < 1` (M3.1f) | `Perspective.τ_mul_integral_posPart_f_β'''_lt_four_of_lt_one`: `τ_β ∫_{-π}^{π} \|f_β'''\|₊ < 4` for `0 < β < 1`, the bound of `lemma:beta_0.75`, `lemma:beta_0.5`, `lemma:beta_0.33` and `lemma:beta_final`, with other constants: `-A ≤ f_β'' ≤ U` on `[0, π]` from pieces of `cos x` (`f_β''_mem_Icc_of_isChain`), `τ_β ≤ π/2 - cos τ_β` (`τ_β_star_two_le`), on nine intervals of `β` | proved |
| A, symmetrizable derivatives and flows in a metric (M1.5) | `Landscape.SaddleAvoidance.measure_tendsto_symmetrizableUnstable_eq_zero`: the starts converging to fixed points whose derivative is conjugate to a symmetric map with an eigenvalue of modulus greater than one form a null set; `ae_not_tendsto_metricGradientFlow_of_isStrictSaddle`: for `f` of class `C²` and `G` of class `C¹` with `G ∇f` bounded and Lipschitz and `G` symmetric positive definite at the strict saddles, `ν`-almost every trajectory of `ẋ = -G(x) ∇f(x)` converges to no strict saddle | proved |
| C, partial tables (M3.4c) | `Landscape.PhaseModel.not_forall_exists_sub_characterPhase_mem_periods_of_isLocalMin`: the landscape of `V_full` below `n²/8` fails for partial tables generating the full one, by `exists_mem_span_isLocalMin_divisionEnergy_eq_three` for `n = 5`; twisted states are local minimizers of generating `V_S` (`isLocalMin_divisionEnergy_twisted`); a critical point of a phase energy with positive cosines and spanning rows is a local minimizer (`isLocalMin_phaseEnergy_of_cos_pos`) | proved |
| C, `o:strictsaddle` on the circle (M3.1) | `Perspective.strict_saddle`: for every `β > 0`, a critical point of `𝖤_β` on `𝕋^n` that is not a global maximum is a strict saddle. A critical point that is not a strict saddle is stationary (C1) and cut-stable (C2) for `f_β`, which meets `thm:main` of arXiv:2507.22857 with `τ = τ_β`, so it is synchronized and a global maximum | proved |
| D, SA in angles (M5.2) | `Perspective.ae_not_tendsto_angularSA_of_isStrictSaddle`: for `β ≠ 0` and Lebesgue-almost every start `θ₀`, no solution of SA in angles from `θ₀` converges to a critical point of `𝖤_β` with a line of positive second derivative; such a point is a strict saddle of `-𝖤_β` on `EucSpace n` (`isStrictSaddle_of_isStrictSaddleTorusEnergy`). SA in angles (`angularSA`) gives SA on `(𝕊¹)ⁿ` (`sa_anglesToSphere`), is the gradient flow of `n² 𝖤_β` for the metric `diag(Z_i)` (`hasDerivAt_torusEnergy_of_angularSA`), and its solutions are exactly the trajectories of `metricGradientFlow` for `G = diag(n²/Z_i)` (`angularSA_metricGradientFlow`, `toLp_eq_metricGradientFlow`), whose field is bounded and Lipschitz by `Landscape.SaddleAvoidance.exists_lipschitzWith_of_periodic` | proved |
| A + B + C, `𝖤_β` on angles (M4.3) | `Perspective.ae_tendsto_synchronized_of_angularUSA`: for every `β > 0` and Lebesgue-almost every `θ₀ ∈ ℝⁿ`, every solution of `eq:onangles` from `θ₀` converges to one synchronized state; every critical point of `-𝖤_β` is a global minimizer or a strict saddle (`isMinOn_or_isStrictSaddle_neg_eucTorusEnergy`, by `strict_saddle`), and `-𝖤_β` is analytic, so it satisfies the Łojasiewicz inequality at every point (`exists_isLojasiewiczAt_neg_eucTorusEnergy`, by M2.4); with exponent `1/2` wherever only the common rotations have zero second derivative (`isLojasiewiczAt_neg_eucTorusEnergy`): at the synchronized states (`isLojasiewiczAt_of_synchronized`) and, for two tokens, at every critical point (`isLojasiewiczAt_of_isCritical_two`) | proved |
| A + B, descent curves | `Landscape.SaddleAvoidance.IsDescentCurve.exists_tendsto_of_isLojasiewiczAt`: a curve along which `f` decreases at least at the rate `c₁ ‖∇f‖²` while it moves at most at the speed `c₂ ‖∇f‖`, with a Łojasiewicz limit point up to `f`-invariant translations, converges, to a critical point (`IsDescentCurve.gradient_eq_zero_of_tendsto`); gradient flows and the flows `ẋ = -G(x) ∇f(x)` with `c₁ ≤ G ≤ c₂` are such curves (`isDescentCurve_of_gradientFlow`, `isDescentCurve_of_metric`) | proved |
| A + B + C + D, clustering on the circle (#27) | `Perspective.clustering_circle`: for every `β ≥ 0` and every uniform tuple law on `(𝕊¹)ⁿ`, almost every initial tuple clusters, for SA and for USA; `beta_tiny_circle` is the case `β ≤ 1` of the survey. Every solution of SA in angles descends `-𝖤_β` at rates `n e^{∓|β|}` and converges to a critical point (`exists_tendsto_angularSA`), so for `β > 0` almost every one synchronizes (`ae_tendsto_synchronized_of_angularSA`); USA and SA on the sphere have unique solutions (`USA_unique`, `SA_unique`), which follow their angles (`exists_tendsto_of_angles`); `β = 0` is `beta0_consensus` | proved |
| D, spheres without manifolds (M5.3) | `Perspective.ae_iff_ae_volume`: for every `d ≥ 1` and every uniform tuple law `P` on `(𝕊^{d-1})^n`, a property holds `P`-almost surely exactly when it holds at `x/‖x‖` for Lebesgue-almost every `x ∈ (ℝ^d)^n`; for sets, `measure_eq_zero_iff_volume_sphereCone` and `volume_eq_zero_iff_measure`, by polar coordinates (`measurePreserving_tuplePolar`) and `shellLaw_eq_surfaceMeasure`. `usa_iff_gradient`, `sa_iff_gradient`: for `β ≠ 0`, USA and SA are the slices `‖x_i‖ = r_i` of the flows `ẋ_i = n⁻¹ ‖x_i‖² ∇_i g(x)` and `ẋ_i = ‖x_i‖² Z_i⁻¹ ∇_i g(x)` on `(ℝ^d)^n`, `g(x) = 𝖤_β(x/‖x‖)` (`hasGradientAt_selfEnergy_update`); a solution of either from nonzero vectors keeps its norms, and its directions solve the sphere flow (`usa_of_radialField`, `sa_of_radialField`) | proved |
| B, Łojasiewicz's theorem on every finite-dimensional space (M6.1) | `Landscape.SaddleAvoidance.exists_isLojasiewiczAt_of_analyticAt`: a function analytic at a point of a finite-dimensional real inner product space satisfies the Łojasiewicz inequality there with an exponent `0 ≤ a < 1`, by the `EucSpace` case through the isometry `(stdOrthonormalBasis ℝ E).repr`; `exists_isLojasiewiczAt_of_analyticOnNhd` gives `hloj` for a function analytic everywhere; `IsLojasiewiczAt` is invariant under linear isometries (`IsLojasiewiczAt.comp_linearIsometryEquiv_symm`, `isLojasiewiczAt_comp_linearIsometryEquiv_symm_iff`) and depends only on the germ of `f` (`IsLojasiewiczAt.congr_of_eventuallyEq`, `isLojasiewiczAt_congr`) | proved |
| A + B, analytic energies without `hloj` (M6.2) | `Landscape.SaddleAvoidance.exists_tendsto_of_analyticOnNhd`: for `f` analytic with an `L`-Lipschitz gradient, `0 < η` and `η L < 1`, an orbit of gradient descent with a limit point up to `f`-invariant translations converges, to a critical point; every orbit when the translations span `E` (`exists_tendsto_of_span_eq_top_of_analyticOnNhd`), every bounded orbit (`exists_tendsto_of_isBounded_of_analyticOnNhd`). The same for every solution of `ẋ = -∇f(x)`, with no bound on `∇f` (`exists_tendsto_gradientFlow_of_analyticOnNhd`, `exists_tendsto_gradientFlow_of_span_eq_top_of_analyticOnNhd`, `exists_tendsto_gradientFlow_of_isBounded_of_analyticOnNhd`). Analytic forms of the scheme, with `hf : AnalyticOnNhd ℝ f univ` in place of `ContDiff ℝ 2 f` and `hloj`: `ae_exists_tendsto_isMinOn_of_analyticOnNhd`, `ae_exists_tendsto_isMinOn_of_span_eq_top_of_analyticOnNhd`, `ae_exists_tendsto_isLocalMin_of_analyticOnNhd`, `ae_exists_tendsto_gradientFlow_isMinOn_of_analyticOnNhd`, `ae_exists_tendsto_gradientFlow_isMinOn_of_span_eq_top_of_analyticOnNhd` | proved |
| C, second-order form (M6.5) | `Landscape.SaddleAvoidance.isMinOn_or_isStrictSaddle_of_inner_nonneg`: if every critical point of a `C²` function at which the Hessian is positive semidefinite is a global minimizer, every critical point is a global minimizer or a strict saddle, the hypothesis `hcrit` of the landscape theorems; `isLocalMin_or_isStrictSaddle_of_inner_nonneg` for local minimizers, and `isMinOn_or_isStrictSaddle_of_inner_nonneg_of_lt` and `isLocalMin_or_isStrictSaddle_of_inner_nonneg_of_lt` for the critical points below a level `c`. The case split is `or_isStrictSaddle_iff`, which also gives the converse: a critical point is a strict saddle exactly when it has a direction of negative curvature (`isStrictSaddle_iff_exists_inner_neg`) | proved |
| Symmetries (M6.10) | `Landscape.SaddleAvoidance.gradient_apply_of_invariant`: a linear isometry `g` of `E` onto itself with `f ∘ g = f` has `∇f(g x) = g ∇f(x)`; it commutes with gradient steps and their iterates (`gradientStep_apply_of_invariant`, `iterate_gradientStep_apply_of_invariant`) and, for `∇f` bounded and Lipschitz, with the gradient flow (`gradientFlow_apply_of_invariant`). The subspace `fixedSubspace S` fixed by a set `S` of such isometries is closed and mapped into itself by the gradient, gradient descent and the gradient flow (`mapsTo_gradient_fixedSubspace`, `mapsTo_gradientStep_fixedSubspace`, `mapsTo_gradientFlow_fixedSubspace`); a point of it where the derivative of `f` vanishes along it is critical (`gradient_eq_zero_of_mem_fixedSubspace`, symmetric criticality); for differentiable `f` the gradient of `f` restricted to it is `∇f` (`coe_gradient_restrict`), and gradient descent of the restriction is gradient descent of `f` (`coe_iterate_gradientStep_restrict`) | proved |
| A, flows on an open set (M6.7) | `Landscape.SaddleAvoidance.ae_not_tendsto_metricGradient_of_isOpen`: for `f` of class `C²` and `G` of class `C¹` on an open `U`, `G` symmetric positive definite on `U`, and `ν` absolutely continuous, for `ν`-almost every start no solution of `ẋ = -G(x) ∇f(x)` on `[0, ∞)` that stays in `U` converges to a strict saddle in `U`; `ae_not_tendsto_metricGradient_of_isCompact` for the solutions that stay in a compact `K ⊆ U`, which converge to no strict saddle at all (an open `R` with compact closure in `U` is the case `K = closure R`). Near `K` a cutoff makes `f` and `G` global (`exists_cutoff`, `contDiff_smul_of_tsupport_subset`), so that `ae_not_tendsto_metricGradientFlow_of_isStrictSaddle` applies, and strict saddles depend only on the germ of `f` (`IsStrictSaddle.congr_of_eventuallyEq`); countably many compact sets catch every compact subset of `U` (`exists_isCompact_cover`). `Perspective.ae_not_tendsto_of_cutoff` now follows from it, without a cutoff of its own | proved |
| A + B, compact sublevel sets (M6.6) | `Landscape.SaddleAvoidance.exists_pos_ae_sublevel_exists_tendsto_isLocalMin`: for `f` of class `C²` with `{f ≤ c}` compact, if every critical point in `{f ≤ c}` is a local minimizer or a strict saddle and satisfies the Łojasiewicz inequality with an exponent `0 ≤ a < 1`, then for some `η₀ > 0` and every `0 < η < η₀`, gradient descent from almost every start in `{f ≤ c}` converges to a local minimizer in `{f ≤ c}`, with no global Lipschitz constant; `exists_pos_ae_sublevel_exists_tendsto_isLocalMin_of_analyticOnNhd` for analytic `f`, without the Łojasiewicz hypothesis. Convergence comes from `exists_pos_tendsto_of_mapsTo`: for small steps, orbits from a compact set `K` that gradient descent maps into itself converge to a critical point in `K`, given the inequality at the critical points in `K`. Their orbits are those of a cutoff of `f` with a Lipschitz gradient (`exists_lipschitzWith_gradient_eqOn`, `iterate_gradientStep_eq_of_eqOn`) | proved |
| Rates (M6.9) | `Landscape.SaddleAvoidance.exists_le_mul_pow_of_isLojasiewiczAt`: an orbit of gradient descent converging to `w`, where the Łojasiewicz inequality holds with `a ≤ 1/2`, converges geometrically: `f x_k - f w` and `‖x_k - w‖` are at most `C b^k` with `0 < b < 1`; with `1/2 < a < 1`, `f x_k - f w ≤ C (k + 1)^{-1/(2a-1)}` and `‖x_k - w‖ ≤ C (k + 1)^{-(1-a)/(2a-1)}` (`exists_le_mul_rpow_of_isLojasiewiczAt`). The same for descent curves, among them gradient flows and the flows `ẋ = -G(x) ∇f(x)`, with `C e^{-λt}`, `λ > 0`, and `t + 1` for `k + 1` (`IsDescentCurve.exists_le_mul_exp_of_isLojasiewiczAt`, `IsDescentCurve.exists_le_mul_rpow_of_isLojasiewiczAt`; for `ẋ = -∇f(x)`, `exists_le_mul_exp_gradientFlow_of_isLojasiewiczAt` and `exists_le_mul_rpow_gradientFlow_of_isLojasiewiczAt`). The limit need not be a local minimizer, and `a ≤ 1/2` may be `0` or negative (`IsLojasiewiczAt.mono` raises the exponent) | proved |
| A + B + C below a level (M6.3) | `Landscape.SaddleAvoidance.ae_exists_tendsto_isMinOn_of_lt`: if every critical point `z` with `f z < c` is a global minimizer or a strict saddle and satisfies the Łojasiewicz inequality, almost every start of gradient descent with `f x₀ < c` and a limit point modulo `Λ` converges to a global minimizer; `ae_exists_tendsto_isMinOn_of_lt_of_span_eq_top` needs no limit point when `Λ` spans `E`. For gradient flows `ae_exists_tendsto_gradientFlow_isMinOn_of_lt` and `ae_exists_tendsto_gradientFlow_isMinOn_of_lt_of_span_eq_top`; for analytic `f` the four forms ending in `_of_analyticOnNhd`, without `hloj`. The limit is a critical point no higher than the start and not a strict saddle (`ae_exists_tendsto_not_isStrictSaddle_of_lt`, `ae_exists_tendsto_gradientFlow_not_isStrictSaddle_of_lt`): `f` does not rise along the orbit, and a bound along a sequence holds at its limit points modulo `Λ` (`le_of_frequently_norm_add_sub_lt`). For `2 - cos x - cos² x` and `c = 2` the hypotheses hold, while the local minimizer `π` at the level `2` breaks `hcrit` of the global theorems | proved |
| A + B + C, phase models (M6.4) | `Landscape.PhaseModel.ae_exists_tendsto_characterPhase`: for `n ≥ 4`, `0 < η` with `9 n² η < 1` and `ν` absolutely continuous, from `ν`-almost every start with `V_full < n²/8` gradient descent converges, to the phases of a character up to periods; `Landscape.PhaseModel.Abelian.ae_exists_tendsto_characterPhase` over every finite abelian group of order `n`. The phase energy is analytic (`Landscape.PhaseModel.analyticOnNhd_phaseEnergy`), so every orbit of gradient descent on a phase energy with integer rows and `η Σ_r ‖a_r‖² < 1` converges, to a critical point (`Landscape.PhaseModel.exists_tendsto_gradient_phaseEnergy_eq_zero` and its abelian twin, through `Landscape.PhaseModel.Abelian.span_periods`); `ae_tendsto_characterPhase` now follows from these theorems | proved |
| The cyclic phase model from the abelian one (M6.8) | `Landscape.PhaseModel.bijective_cyclicChar`: the characters of `Fin n` are the `cyclicChar n j : x ↦ e^{2πijx/n}`, `j < n`, whose phases `arg e^{2πijx/n}` differ from `characterPhase n j` by a period (`Landscape.PhaseModel.characterPhase_sub_mem_periods`). The theorems of `Landscape.PhaseModel.Coordinates`, `Basin`, `Characters`, `Orthogonality`, `Fourier` and `Landscape` keep their names and statements and follow from those of `Landscape.PhaseModel.Abelian` at `G = Fin n`, which no longer imports them; the facts on the circle both use (`exp_ofReal_add_int_mul_two_pi_mul_I`, `conj_exp_ofReal_mul_I`, `one_sub_cos_add_le`, `one_sub_cos_sub_le`) are in `Landscape.PhaseModel.Basic` | proved |
| A, a Hessian bound along the orbit (M8.1) | `Landscape.SaddleAvoidance.le_sub_of_gradientStep_of_segment`: for `f` of class `C²` with `‖∇²f‖ ≤ L` on the segment from `x` to its gradient step, the step lowers `f` by at least `η (1 - η L) ‖∇f(x)‖²`, by `le_add_inner_add_mul_norm_sq_of_convex` on the segment; along an orbit satisfying (S), step `k` lowers `f` by at least `η (1 - η L) ‖∇f(x_k)‖²` (`iterate_succ_le_sub_of_segment`), and `f` decreases when `0 ≤ η` and `η L ≤ 1` (`antitone_iterate_of_segment`). `measure_iterate_norm_le_tendsto_strictSaddle_eq_zero`: for `0 < η`, `η L < 1` and every Haar measure, the starts whose orbit has `‖∇²f(x_k)‖ ≤ L` at every `k` and converges to a strict saddle form a null set, by `measure_inter_tendsto_strictSaddle_eq_zero` on this forward-invariant set; under (S) (`measure_segment_tendsto_strictSaddle_eq_zero`), and almost surely for a law `ν ≪ μ` (`ae_not_tendsto_strictSaddle_of_segment`) | proved |
| B, Łojasiewicz convergence from a decrease (M8.2) | `Landscape.SaddleAvoidance.exists_tendsto_of_isLojasiewiczAt_of_le_sub`: for `0 < η` and `κ > 0`, an orbit of gradient descent each of whose steps lowers `f` by at least `η κ ‖∇f(x_k)‖²` converges if it has a limit point `z` up to translations leaving `f` invariant, where `f` is continuous and satisfies the Łojasiewicz inequality with `0 ≤ a < 1`; to `z` when `z` is a limit point (`tendsto_of_isLojasiewiczAt_of_le_sub`); `z` is critical when `∇f` is continuous (`gradient_eq_zero_of_frequently_of_le_sub`). The step bound, the trap and the finite length take the same decrease, with `m (1 - a) κ` in place of `m (1 - a)(1 - η L)` (`mul_norm_gradientStep_sub_le_of_le_sub`, `mul_norm_iterate_succ_sub_le_of_le_sub`, `iterate_mem_ball_of_le_sub`, `mul_norm_iterate_sub_le_of_le_sub`). There is no `hL`, no `η L < 1` and no differentiability; the present theorems follow with `κ = 1 - η L`, the decrease of every step of an orbit (`iterate_succ_le_sub`). The examples take `‖x‖² / 2` with `η = 3/2`, beyond `η L < 1`, where a step lowers `f` by `η (1 - η / 2) ‖∇f‖²` (`half_mul_norm_sq_gradientStep`) | proved |
| C, the scheme along the orbit (M8.3) | `Landscape.SaddleAvoidance.exists_tendsto_or_tendsto_norm_of_segment`: for `f` of class `C²`, `0 < η` and `η L < 1`, an orbit with `‖∇²f‖ ≤ L` on the segment of every step (S) converges to a critical point `w` with `f w ≤ f x`, or `‖x_k‖ → ∞`, if `f` satisfies the Łojasiewicz inequality with `0 ≤ a < 1` at the critical points no higher than the start. `ae_exists_tendsto_not_isStrictSaddle_or_tendsto_norm_of_lt_of_segment`: for a Haar measure `μ`, `ν ≪ μ` and the Łojasiewicz inequality at the critical points below `c`, `ν`-almost every `x` with `f x < c` whose orbit satisfies (S) converges to a critical point `w` with `f w ≤ f x` that is not a strict saddle, or `‖x_k‖ → ∞`; to a global minimizer if every critical point below `c` is one or a strict saddle (`ae_exists_tendsto_isMinOn_or_tendsto_norm_of_lt_of_segment`). For analytic `f`, with neither the Łojasiewicz hypothesis nor `c`: `ae_exists_tendsto_not_isStrictSaddle_or_tendsto_norm_of_segment_of_analyticOnNhd` and `ae_exists_tendsto_isMinOn_or_tendsto_norm_of_segment_of_analyticOnNhd`. There is no `hL`, no `hcl` and no compact sublevel set. The examples: for the double well `x⁴/4 - x²/2`, `η = 1/10` and `L = 23/4`, every orbit from `[-3/2, 3/2]` satisfies (S) (`norm_fderiv_gradient_doubleWell_le_of_mem_Icc`) although the Hessian is unbounded (`not_bddAbove_norm_fderiv_gradient_doubleWell`), and almost every start there converges to a global minimizer `±1`; for `-x` every orbit satisfies (S) with `L = 0` and `‖x_k‖ → ∞` | proved |
| Rates along the orbit (M8.4) | `Landscape.SaddleAvoidance.exists_le_mul_pow_of_isLojasiewiczAt_of_le_sub`, `exists_le_mul_rpow_of_isLojasiewiczAt_of_le_sub` and the estimates behind them, `exists_rate_estimates_of_isLojasiewiczAt_of_le_sub`: the rates of M6.9 for an orbit each of whose steps lowers `f` by at least `η κ ‖∇f(x_k)‖²`, `0 < η`, `κ > 0`, converging to `w`, where `f` is continuous and satisfies the Łojasiewicz inequality; the present theorems follow with `κ = 1 - η L`. Under (S), for `f` of class `C²`, `0 < η` and `η L < 1`, with the decrease of M8.1: `exists_rate_estimates_of_isLojasiewiczAt_of_segment`, `exists_le_mul_pow_of_isLojasiewiczAt_of_segment`, `exists_le_mul_rpow_of_isLojasiewiczAt_of_segment`. The examples take `‖x‖² / 2`, with `η = 3/2` for the decrease and `η = 1/2`, `L = 1` for (S) | proved |

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

### M1 — Layer A for flows (general, `Transformer.Landscape.SaddleAvoidance`)

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
| M2.4 | **Hard core, done.** Łojasiewicz at degenerate critical points of analytic functions, the remaining input of `Landscape.Lojasiewicz.lojasiewicz_modulated`. Local uniformization (`Landscape.Uniformization.hasMonomialCharts_of_analyticAt`), by induction on the dimension and on the order in the last variable through Weierstrass preparation, ramified Newton charts and the principalization of `(y, t^p)` by monomial maps, gives every analytic energy that is not locally constant compact monomial charts (`hasCompactMonomialCharts_of_analyticAt`); `analytic_gradient_inequality_of_compact_monomial_charts` derives the local inequality from them. `beta_tiny`, `beta_tiny_circle`, `beta_interval` and `boumal_clustering` no longer wait for it. | Normalization |

### M3 — Layer C, landscape

Tasks: M3.1 #12, with parts M3.1a–f #32–#37; M3.3 #14; M3.4a #16; M3.4b
#17; M3.4c #18; M3.4d #19.

M3.2 (`β ≤ 1` and `β ≥ n²/π²` on the torus, #13) and M3.5 (`β ≤ c/n` on
spheres, #15) were dropped on 2026-10-18. M3.1 covers every `β > 0` on the
circle, and M3.3 covers every `β > 0` for `d ≥ 3`.

M3.1 follows arXiv:2507.22857. Its parts are new modules of
`Transformer.Perspective`, with their own names and citing the paper;
arXiv:2507.22857 is not formalized as a paper of its own. M3.1 is pure
landscape and needs neither A nor B.

| Item | Deliverable | Depends on |
| --- | --- | --- |
| M3.1a | **Reduction, done.** Criticality (`IsCriticalTorusEnergy`) gives (C1) through `hasDerivAt_torusEnergy`. A point with no positive second derivative satisfies `TorusHessianNonPos`, so `taylor3_inequality` gives (C2) for every `S`. Synchronized angles are global maxima. | — |
| M3.1b | **Gaps.** Order the distinct positions as `θ_1 < ⋯ < θ_K` in `[0, 2π)` and take the gaps `Δ_k`. If (C2) holds for every `S` and `f' < 0` off `[-τ, τ]`, at most one gap exceeds `τ` (`lemma:twoconsecutive_general`). | — |
| M3.1c | **The criterion for a general `f`, done.** For `f` `2π`-periodic, `C³`, with `f'(0) > 0`: first `lemma:taumax`, then `cor:main` and `thm:main`. A stationary configuration that is cut-stable for every `S` is synchronized. | M3.1b |
| M3.1d | **The kernel `f_β`.** `f_β' < 0` off `[-τ_β, τ_β]`, where `τ_β = τ_β_star 2 β`. The sign regions of `f_β'''` come from the quartic `p(cos x)` (`lemma:positiveregion`, `lemma:positiveregion2`). `∫ \|f_β'''\|₊` is expressed through `f_β''` at the sign changes. | — |
| M3.1e | **`β ≥ 1`, done:** `τ_β ∫ \|f_β'''\|₊ < 4` (`τ_mul_integral_posPart_f_β'''_lt_four`, `Perspective.SyncKernelBetaOne`). `lemma:beta1` holds for all `a, b ∈ [0, π]`, from `-(8/5)√β ≤ f_β'' ≤ (2/5)√β` on `[0, π]`. `corollary:tau2`, `τ_β < 1/√β` for every `β > 0`, comes from `cos u ≤ 1 - u²/2 + u⁴/24` in place of `lemma:tau1`. | M3.1d |
| M3.1f | **`0 < β < 1`, done:** the same bound (`lemma:beta_0.75`, `lemma:beta_0.5`, `lemma:beta_0.33`, `lemma:beta_final`). Fallback for `0 < β ≤ 1`: `thm:taylorphi` of arXiv:2405.18273, which needs the second derivative along every direction, not only along `1_S`. | M3.1d |
| M3.1 | **`Perspective.strict_saddle`, done.** For every `β > 0`, by contradiction. A critical point that is not a strict saddle is stationary (C1) and cut-stable (C2) for every `S` (M3.1a). `f_β` meets the criterion of `thm:main` with `τ = τ_β` (M3.1e for `β ≥ 1`, M3.1f for `0 < β < 1`), so M3.1c synchronizes the point, and synchronized angles are global maxima. The hypothesis `2 ≤ n` is dropped: the survey states none, and the proof needs none. | M3.1a, M3.1c, M3.1e, M3.1f |
| M3.3 | **`d ≥ 3`, done.** The non-synchronized critical points of `𝖤_β` on `(𝕊^{d-1})^n` are strict saddles, as a new theorem with its own name (arXiv:2405.18273, App. C). Export the `key` of `hessian_at_critical_intrinsic` (second derivative `= 2 · hessQuad X v`). Sum `hessQuad X (Proj_X(e_k 𝟙ᵀ))` over an orthonormal basis `e_k`. The curve `s ↦ (x_i + s v_i)/‖x_i + s v_i‖` has velocity `v`. | — |
| M3.4a | **Characters are nondegenerate, done.** The Hessian of `V_full` at a character is injective, since the rows `e_x - e_y - e_{x-y}` span `ℝⁿ`. This gives Łojasiewicz with exponent `1/2` at the characters. | — |
| M3.4b | **Research: `V_full` above `n²/8`, searched.** `scripts/phase_minima.py` (inside `python/`: `uv run --locked python ../scripts/phase_minima.py`, 44 min; seeded by `n`, so `--largest 8` reproduces rows 2–8) runs 4000 steps of gradient descent of step `1/(3n)` for each `2 ≤ n ≤ 30`. It starts from 400 uniform phases, 400 points of `{0, π}ⁿ`, 400 of `(2π/m)ℤⁿ` for `m ∈ {3, 4, 5, 6, 8}`, 400 mixtures of two characters, and 5 copies of each non-character critical point that Newton's method reaches from 200 uniform starts; every start but the uniform ones is moved by `10⁻²`. All 1600–2355 runs per `n` end at a character up to periods (`V < 10⁻⁸`), so no local minimizer with `V ≥ n²/8` was found. No Newton critical point has a positive semidefinite Hessian. The lowest non-character critical values found are `2, 4, 8, 12, 16, 24, 32, 36, 48, 60, 84` for `n = 2, …, 12`, all above `n²/8`; Newton's method is not exhaustive, finds few points for `n ≥ 18` and none for `n = 25` and `27 ≤ n ≤ 30`. **Hypothesis, not proved:** for `n ≤ 30`, every local minimizer of `V_full` is the phases of a character up to periods. | lab |
| M3.4c | **Partial tables `V_S`, counterexample done.** A generating `S` is not enough: `Landscape.PhaseModel.not_forall_exists_sub_characterPhase_mem_periods_of_isLocalMin`. For `n = 5`, 15 of the 25 pairs generate the full table over `ℤ`, so the zeros of `V_S` are the characters, yet `V_S` has a local minimizer of energy `3 < 5²/8` that is not a character (`exists_mem_span_isLocalMin_divisionEnergy_eq_three`). The energy `3` is above `|S|/8 = 15/8`, so this refutes the threshold `n²/8` carried over unchanged, not a threshold scaled with `|S|`. The mechanism is a twisted state (`isLocalMin_divisionEnergy_twisted`). Take integer phases `j` and `ψ = (π/3) j`. If every defect `j_x - j_y - j_{x-y}` of `S` is `0` or `±1` modulo `6` and the rows of defect `1` and `-1` have the same sum, then `ψ` is critical and every cosine there is positive. So `ψ` is a local minimizer when `S` generates. Open, a hypothesis only: whether random `S` avoid twisted states and other spurious minimizers with high probability as `n → ∞`, and at which density. The analogy is Kuramoto on Erdős–Rényi graphs just above connectivity (abdalla2022expander, cited in arXiv:2312.10794v5 §6). A comment in that survey's TeX source (line 908) says that reference proves only that a convergent trajectory ends synchronized. | — |
| M3.4d | **Finite abelian groups, done.** `Landscape.PhaseModel.Abelian` generalizes `Fourier` and `Landscape`, with the chain from `Division` to `Characters` they rest on, from `ℤ/n` to a finite abelian group `G`, with the characters `AddChar G ℂ`. | — |

### M4 — The scheme theorems (general)

Tasks: M4.1 #20, M4.2 #21, M4.3 #22.

| Item | Deliverable | Depends on |
| --- | --- | --- |
| M4.1 | **Gradient descent, done.** `ae_exists_tendsto_isMinOn`: if every critical point is a global minimizer or a strict saddle, given `hloj` and a limit point modulo `Λ`, then almost every orbit converges to a global minimizer. Periodic corollary: when `Λ` is a lattice spanning `E`, every orbit has a limit point modulo `Λ`, so that hypothesis is free. | M3.4a for the example |
| M4.2 | **Gradient flows, done.** `ae_exists_tendsto_gradientFlow_isMinOn`: the same statement for gradient flows, whose limits are critical (`gradient_eq_zero_of_tendsto_gradientFlow`); `ae_exists_tendsto_gradientFlow_isMinOn_of_span_eq_top` needs no limit point when `Λ` spans `E`. | M1.3, M2.2 |
| M4.3 | **`𝖤_β` on angles (USA), done.** `Perspective.ae_tendsto_synchronized_of_angularUSA`: for every `β > 0` and Lebesgue-almost every `θ₀ ∈ ℝⁿ`, every solution of `eq:onangles` from `θ₀` converges to one synchronized state. The solutions are the gradient flow of `-𝖤_β` run at speed `n` (`angularUSA_eq_gradientFlow`, `angularUSA_gradientFlow`). `-𝖤_β` is analytic, so M2.4 gives `hloj` at every point (`exists_isLojasiewiczAt_neg_eucTorusEnergy`); M2.1 gives it with exponent `1/2` at the synchronized states (`isLojasiewiczAt_of_synchronized`), and for two tokens at every critical point (`isLojasiewiczAt_of_isCritical_two`). | M3.1, M4.2, M1.4, M2.1 |

### M5 — Bridges (D)

Tasks: M5.1 #23, M5.2 #24, M5.3 #25, M5.4 #26.

| Item | Deliverable | Depends on |
| --- | --- | --- |
| M5.1 | **Angles and the circle, done.** `θ ↦ (cos θ_i, sin θ_i)` carries `angularUSA` to USA on `SphereTuple 2 n`. `UniformTuple 2 n P` is the image of normalized Lebesgue measure on `[0, 2π)ⁿ`; the uniform law is unique (`57993faf`). A `2π`-periodic Lebesgue-null set therefore maps to a `P`-null set. | — |
| M5.2 | **SA in angles, done.** SA in angles is `eq:onangles` with `n` replaced by `Z_i` (`angularSA`) and gives SA on the circle (`sa_anglesToSphere`). It is the gradient flow of `n² 𝖤_β`, the energy of §3.4, in the metric `diag(Z_i)` (`hasDerivAt_torusEnergy_of_angularSA`), and `θ̇ = -G(θ) ∇f(θ)` with `f = -𝖤_β` and `G = diag(n²/Z_i)` on `EucSpace n` (`angularSA_iff`). Periodicity makes `G ∇f` bounded and Lipschitz, so M1.5 applies: for Lebesgue-almost every start no solution converges to a critical point with `IsStrictSaddleTorusEnergy` (`ae_not_tendsto_angularSA_of_isStrictSaddle`). | M1.5 |
| M5.3 | **Spheres without manifolds, done, for every `d ≥ 1`.** On `(ℝ^d)^n`, the field `F_i(x) = ‖x_i‖ V(x/‖x‖)_i` has the sphere flow of `V` on every slice `‖x_i‖ = r_i > 0` (`hasDerivAt_radialField_iff`). For USA it is `n⁻¹ ‖x_i‖² ∇_i(𝖤_β ∘ normalize)(x)`, for SA `‖x_i‖² Z_i⁻¹ ∇_i(𝖤_β ∘ normalize)(x)` (`usa_iff_gradient`, `sa_iff_gradient`). A set of configurations is null for the uniform law exactly when its cone is Lebesgue-null (`measure_eq_zero_iff_volume_sphereCone`, `ae_iff_ae_volume`, from `Measure.measurePreserving_homeomorphUnitSphereProd`). M1.5 wants an inner product space and `G ∇f` bounded and Lipschitz on all of it, so the flow still has to move to `PiLp 2` and be cut off outside the invariant shells `ε ≤ ‖x_i‖ ≤ ε⁻¹`: `g` is singular where some `x_i = 0`, and `F` grows linearly. | M1.5 |
| M5.4 | **Research: grokking network to phase model, stated.** The clock network below reduces a phase code of Rotate90 Dense to `V_S`: exactly, one phase model per plane, with a linear readout; with the softmax, to `V_S + W_S` at first order in the logit scale. A hypothesis until proved in Lean. | — |

#### M5.4, stated: the clock network

**Hypothesis until proved in Lean.** Items 1–4 below are exact statements about the
clock network, argued on paper and not yet in Lean. The claim that the grokking
network trains like the clock network is the content of the hypotheses (H0)–(H4).
Its measured training does not satisfy them (last paragraph).

*The network.* Rotate90 Dense is the `a·b⁻¹ mod 97` network of the grokking_abmod
lab (`docs/rotate90-mathematics.md` there, §1.1 and §2.3; its §8.4 introduces `V_S`
as a model of the network's code). It computes `x0 = E_a + J E_b`,
`x1 = x0 + ReLU(N(x0) A)` and `logit_q = ⟪N(x1), E_q⟫`, and it is trained by AdamW
with weight decay on the cross-entropy. Plane `p` of `ℝ^W` is read as `ℂ`, so `J`
multiplies each plane by `i`. The nonzero residues are indexed by their discrete
logarithms `x ∈ ℤ/n`, `n = 96`. The row `(a, b)` is then the pair `(x, y)` of
`divisionRow`, with answer `x - y`.

*Hypotheses.*

- (H0) Only the rows and answers with nonzero residues: the token `0` is dropped.
- (H1) A phase code: plane `p` of `E_x` is `c_p e^{iψᵖ_x}`, with `c_p ≠ 0` fixed and
  the phases `ψᵖ ∈ ℝⁿ` free.
- (H2) A product block: on the row `(x, y)`, plane `p` of `N(x1)` is
  `r_p c_p e^{i(ψᵖ_x - ψᵖ_y)}`, with `r_p > 0` fixed.
- (H3) No competition between answers: a row costs `κ - logit_{x-y}`, with
  `κ = Σ_p κ_p`, in place of `log Σ_z e^{logit_z} - logit_{x-y}`.
- (H4) Gradient descent or the gradient flow on the phases, in place of AdamW with
  batches and decay.

Under (H0)–(H2) the logits are the clock readout
`logit_z(x, y) = Σ_p κ_p cos(ψᵖ_z - ψᵖ_x + ψᵖ_y)` with `κ_p = r_p |c_p|²`.

1. **Linear readout.** Under (H3) the training loss is `Σ_p κ_p V_S(ψᵖ)`, with
   `V_S = divisionEnergy S`: one phase model per plane, uncoupled. Under (H4),
   gradient descent of step `η` on `ψᵖ` is gradient descent on `V_S` of step `η κ_p`
   (`gradientStep`), so `PhaseModel` applies plane by plane. For instance, on the
   full table with `n ≥ 4` and `9 n² η κ_p < 1`, `ae_exists_tendsto_characterPhase`
   applies: from almost every start below `n²/8`, the orbit converges, to a character
   up to periods.
2. **Cartesian parameters.** Take `Eᵖ ∈ (ℂ∖0)ⁿ` as parameters and put `Eᵖ_x / |Eᵖ_x|`
   in place of `e^{iψᵖ_x}`. The loss is then `κ_p V_S(arg Eᵖ)`. It does not change
   when any `Eᵖ_x` is scaled by a positive number, so its gradient is tangent to the
   circles. The gradient flow therefore keeps every radius `ρ_x = |Eᵖ_x|`, and its
   angles solve `ψ̇_x = -(κ_p / ρ_x²) ∂_x V_S(ψ)`. In `φ_x = ρ_x ψ_x / √κ_p` this is
   the gradient flow of `φ ↦ V_S(√κ_p φ / ρ)`. Critical points, local minimizers,
   strict saddles and Lebesgue-null sets of starts correspond under `E ↦ arg E`.
3. **Softmax readout.** Without (H3), the cross-entropy is
   `|S| log n + Σ_p κ_p (V_S + W_S - |S|)(ψᵖ) + R`. Jensen's inequality and Hoeffding's
   lemma give `0 ≤ R ≤ |S| κ² / 2`, and `∇R = O(κ²)`. Here
   `W_S(ψ) = (1/n) Σ_{(x,y) ∈ S} Σ_z cos(ψ_z - ψ_x + ψ_y)` is the mean logit. To first
   order in `κ`, the cross-entropy is a sum of the energies `V_S + W_S`, one per
   plane, not of `V_S`.
4. **What `W_S` changes.**
   - On the full table, `V_full + W_full = n² - (1/n) Σ_{k ≠ 0} |û_k|² Re û_k`, in the
     notation of `Landscape.PhaseModel.Fourier`. This is `≥ 0`, and it vanishes exactly at the
     characters with `k ≠ 0`, up to periods: the competition removes the trivial
     character. When `V_full + W_full < n²`, some `j ≠ 0` has
     `n Σ_x (1 - cos(ψ_x - 2πjx/n)) ≤ V_full + W_full`. This is the Fourier step of
     the landscape (`exists_mul_sum_one_sub_cos_le`) without the trivial character.
   - On a partial table, take the character `k` with `2k ≠ 0` in `ℤ/n`. There
     `∂_z (V_S + W_S) = -(1/n) Im(e^{2πikz/n} · conj ĥ_S(k))`, with
     `ĥ_S(k) = Σ_{(x,y) ∈ S} e^{2πik(x-y)/n}`. The character is therefore a critical
     point only when the answers of `S` have a vanishing `k`-th Fourier coefficient.

Items 1–3 are provable now on top of `PhaseModel`, and item 4 on top of its
`Fourier` identities.

*What the reduction does not cover.* (H1), (H2) and (H4) describe a family that the
lab's training does not stay on. In trained networks `χ(a/b)` explains 19–33% of the
variance of a plane of `N(x1)`, and 8–10 of the 48 frequencies are used (§2.4 of the
note). In its small-step limit AdamW with decay follows `-F(θ) - wd·θ` (§7.3 of the
note), which is not a gradient flow. Items 1–4 are statements about the clock network, not about
the lab's trajectories.

### M6 — Reuse (general, `Transformer.Landscape.SaddleAvoidance`)

Tasks: M6.1 #38, M6.2 #39, M6.3 #40, M6.4 #41, M6.5 #42, M6.6 #43, M6.7 #44,
M6.8 #45, M6.9 #46, M6.10 #47.

Added on 2026-10-18 after a review of how the four models use the scheme. Only
`𝖤_β` on angles uses it directly (M4.3). `Landscape.PhaseModel.ae_tendsto_characterPhase` and
its abelian twin assemble the conclusion by hand, and assume that the orbit
converges. The spheres take `Landscape.Lojasiewicz.lojasiewicz_modulated_of_finiteDimensional`
for convergence, because Łojasiewicz's theorem in `SaddleAvoidance` was stated on
`EucSpace N` only before M6.1. For avoidance they wrote their own cutoff
(`Perspective.ae_not_tendsto_of_cutoff`), because the flow theorems wanted global
bounds; since M6.7 it follows from
`Landscape.SaddleAvoidance.ae_not_tendsto_metricGradient_of_isCompact`. After M6 a new model brings three inputs. The first is analyticity, usually by
`fun_prop` as in `Perspective.analyticAt_neg_eucTorusEnergy`. The second is a Lipschitz
constant or a compact sublevel set. The third is its landscape.

| Item | Deliverable | Depends on |
| --- | --- | --- |
| M6.1 | **Łojasiewicz's theorem on every finite-dimensional space, done.** Generalize `Landscape.SaddleAvoidance.exists_isLojasiewiczAt_of_analyticAt` from `EucSpace N` to every finite-dimensional real inner product space `E`, keeping its name and its callers. Route: `IsLojasiewiczAt` is invariant under linear isometries (a new lemma, from `Landscape.Lojasiewicz.gradient_comp_linearIsometryEquiv_symm`); apply the `EucSpace` case through `(stdOrthonormalBasis ℝ E).repr`, as `lojasiewicz_modulated_of_finiteDimensional` does. Also: `IsLojasiewiczAt` depends only on the germ of `f` (`Filter.EventuallyEq.gradient`), and `AnalyticOnNhd ℝ f univ` gives `hloj` at every point. The `LocalInequality` row of the `SaddleAvoidance` docstring still says dimension at most one; fix it. | — |
| M6.2 | **The scheme without `hloj` for analytic energies, done.** (a) Let `f` be analytic, with `∇f` `L`-Lipschitz, and let `ηL < 1`. Then every orbit of gradient descent with a limit point modulo `f`-invariant translations converges to a critical point. So does every orbit when the translations span `E`, and every bounded orbit. (b) The same for gradient flows. (c) Analytic forms of `ae_exists_tendsto_isMinOn`, `ae_exists_tendsto_isMinOn_of_span_eq_top`, `ae_exists_tendsto_isLocalMin`, `ae_exists_tendsto_gradientFlow_isMinOn` and `ae_exists_tendsto_gradientFlow_isMinOn_of_span_eq_top`. In them `hf : AnalyticOnNhd ℝ f univ` replaces both `hf : ContDiff ℝ 2 f` (by `AnalyticOnNhd.contDiff`) and `hloj` (by M6.1). Names end in `_of_analyticOnNhd`. | M6.1 |
| M6.3 | **Landscape below a level `c`, done.** `ae_exists_tendsto_isMinOn_of_lt`: suppose every critical point `z` with `f z < c` is a global minimizer or a strict saddle and satisfies the Łojasiewicz inequality. Then for almost every start with `f x₀ < c`, the orbit converges to a global minimizer, given a limit point modulo `Λ`; that limit point is free when `Λ` spans `E`. Route: `f` decreases along the orbit (`antitone_iterate`), so a limit point modulo `Λ` has `f < c`. It is critical (`gradient_eq_zero_of_frequently`), the orbit converges (`exists_tendsto_of_isLojasiewiczAt`), and the limit is no strict saddle (`ae_not_tendsto_of_isStrictSaddle`). Also the flow analogue, and analytic forms without `hloj`. This is the form of most landscape theorems: the landscape is benign below a level. | M6.1 |
| M6.4 | **Phase models: convergence becomes a conclusion, done.** First, `Landscape.PhaseModel.analyticOnNhd_phaseEnergy`. By M6.2, every orbit of gradient descent on a phase energy with integer rows and `η Σ_r ‖a_r‖² < 1` converges to a critical point. Then `Landscape.PhaseModel.ae_exists_tendsto_characterPhase` and `Landscape.PhaseModel.Abelian.ae_exists_tendsto_characterPhase`: for almost every start with `V_full < n²/8`, the orbit converges, and its limit is the phases of a character up to periods (M6.3 with `c = n²/8`). The present `ae_tendsto_characterPhase` assume convergence; keep them or derive them from the new theorems. | M6.2, M6.3 |
| M6.5 | **The landscape hypothesis in second-order form, done.** Suppose every critical point with a positive semidefinite Hessian is a global minimizer (or a local minimizer, or either only below `c`). Then every critical point is a global minimizer (resp. a local minimizer) or a strict saddle. The proof is a case split through `isStrictSaddle_of_inner_neg`. Papers state landscapes this way: no spurious second-order critical points. | — |
| M6.6 | **The scheme on a compact sublevel set, done.** Let `f` be `C²` with `{f ≤ c}` compact, and ask `hcrit` and `hloj` only on `{f ≤ c}`. Then there is `η₀ > 0` such that for `0 < η < η₀`, almost every start in `{f ≤ c}` converges to a local minimizer. Add an analytic form without `hloj`. No global Lipschitz constant is needed; polynomials of degree above two and networks lack one, while regularized losses have compact sublevel sets. Avoidance is M1.6 (`exists_pos_measure_sublevel_tendsto_strictSaddle_eq_zero`). For convergence, prove either a local trap or a cutoff. The cutoff: take `χ f`, with `χ` a smooth bump equal to `1` near `{f ≤ c}` (`ContDiffBump`, as in `Perspective.SphereCutoff`). Its orbits from `{f ≤ c}` are those of `f`, and its gradient is globally Lipschitz, so `tendsto_of_isLojasiewiczAt` applies to it. | M6.1 |
| M6.7 | **Avoidance for flows on an open set, done.** Move the cutoff of `Perspective.ae_not_tendsto_of_cutoff` into `SaddleAvoidance`, for a general metric. Let `f` be `C²` and `G` be `C¹` on an open `U`, with `G` symmetric positive definite on `U`, and let `R` be open with compact closure in `U`. Then almost no start has a trajectory of `ẋ = -G(x) ∇f(x)` that stays in `R` and converges to a strict saddle in `R`. The same follows for a countable cover of `U` by such sets `R`. | — |
| M6.8 | **The cyclic phase model as the abelian one at `G = Fin n`, done.** Derive the theorems of `Landscape.PhaseModel.Coordinates`, `Basin`, `Characters`, `Orthogonality`, `Fourier` and `Landscape` (about 940 lines) from `Landscape.PhaseModel.Abelian` (`Fin.addCommGroup`), keeping their names and statements. This needs `EucSpace n` identified with `EuclideanSpace ℝ (Fin n)`, and `characterPhase n j` with the phases of a character up to periods. A later task over a group then builds on `Abelian` alone. | M6.4 |
| M6.9 | **Rates, done.** Let `IsLojasiewiczAt f w a` hold at the limit `w`. For `a ≤ 1/2`, `‖x_k - w‖` and `f x_k - f w` decay geometrically; for `1/2 < a < 1`, they decay as `k^{-(1-a)/(2a-1)}`. Prove this for gradient descent and for flows. The lab can then compare measured rates with the theory. | — |
| M6.10 | **Symmetries, done.** Let `g` be a linear isometry with `f ∘ g = f`. Then `∇f (g x) = g ∇f(x)`, gradient steps and flows commute with `g`, and the fixed subspace of a group of such isometries is invariant. This is a building block for bridges (D). An almost-everywhere statement on an invariant subspace is about that subspace's own Lebesgue measure, and the subspace is null in `E`. So a bridge also needs the subspace to attract. | — |

### M7 — One directory for the framework

Tasks: M7.1 #49.

Added on 2026-10-18 at the user's request. The framework and what it imports
lie in eight directories among some fifty others, most of them papers:
`SaddleAvoidance`, whose name covers only layer A, its model `PhaseModel`,
`GlobalFlow`, and Łojasiewicz's theorem for analytic functions, which sits in
the paper directory `Normalization` with its foundations `Uniformization`,
`AnalyticPreparation`, `RealAnalyticGerms` and `Sturm`. What is proved is hard
to find: while M8 was planned, the descent lemma on a segment was taken for
new, though `Descent` and `Sublevel` have it.

| Item | Deliverable | Depends on |
| --- | --- | --- |
| M7.1 | **Move, without changing a statement.** Everything the framework imports, except `Transformer.Basic`, moves with `git mv` into `src/Transformer/Landscape/`, one subdirectory per part. `SaddleAvoidance` (layers A to C, the scheme theorems, the rates) and `PhaseModel`, with `Abelian`. `GlobalFlow`. `Lojasiewicz`: the 91 modules of `Normalization` that prove Łojasiewicz's gradient inequality for analytic functions and the convergence of modulated gradient flows (`lem: loj`, App. D.1 of arXiv:2510.22026); `Normalization/Lojasiewicz.lean` becomes `Lojasiewicz/Modulated.lean`. `Basic` and `TimeChange` use the paper's dynamics `NA` and stay, with the paper's other modules. `Uniformization`, `AnalyticPreparation`, `RealAnalyticGerms` and `Sturm`, on which that theorem rests. Namespaces follow the directories (`Transformer.Landscape.SaddleAvoidance`, `Transformer.Landscape.Lojasiewicz`, …). `SaddleAvoidance` stays one directory: `IsStrictSaddle`, `IsSymmetricUnstable`, `IsDescentCurve` and `IsLojasiewiczAt` have lemmas in other modules than their own, and dot notation such as `hz.isSymmetricUnstable` needs them in one namespace; its map groups the modules by layer instead. `Landscape.lean` holds only imports and the map: for each subdirectory, what it proves and its main theorems; `Lojasiewicz.lean` gets a map of its own. Every importer changes its imports, names and `open` lines: the papers `Normalization`, `Perspective`, `Clusters`, `Interpolation`, `Causal`, `Kinetic`, `Imbalance` and `ContinuityEq` (27 modules), `src/Transformer.lean`, `scripts/phase_minima.py`, `scripts/AnalyticPreparationAxioms.lean`, the notices in `third_party/`, `EXPERIMENT_PLAN.md` and the Lean names in this plan. No statement or proof changes beyond names, namespaces, imports and `open` lines. | — |

### M8 — Conditions along the orbit

Tasks: M8.1 #50, M8.2 #51, M8.3 #52, M8.4 #53.

Added on 2026-10-18 for losses whose gradient is not globally Lipschitz and
whose sublevel sets need not be compact, such as a perceptron without weight
decay. The scheme so far asks for one of two things: a global Lipschitz
constant `hL` (M4, M6.3), or a compact sublevel set and small steps (M6.6). It
also asks for a limit point, `hcl`. M8 asks instead for a condition that a run
can check along its own orbit `x_k = (gradientStep f η)^[k] x`:

> (S) `‖∇²f(y)‖ ≤ L` at every `y` of every segment `[x_k, x_{k+1}]`, with
> `η L < 1`.

In place of `hcl`, the conclusion gets a second alternative: `‖x_k‖ → ∞`.

The descent lemma of `Descent` has `L` where the usual one has `L/2`. So (S)
asks for `η ‖∇²f‖ < 1`, half the edge-of-stability threshold `η λ_max = 2`.
Avoidance needs `η λ ≠ 1` at every eigenvalue anyway, and (S) gives that. M8
is about gradient descent only, not AdamW, schedules or batches.

The pieces exist:

- avoidance on a forward-invariant set, `measure_inter_tendsto_strictSaddle_eq_zero` (M1.6);
- the descent lemma on a convex set, `le_add_inner_add_mul_norm_sq_of_convex`.

`hL` enters the Łojasiewicz convergence only through the sufficient decrease,
via `le_sub_of_gradientStep` and `antitone_iterate`.

| Item | Deliverable | Depends on |
| --- | --- | --- |
| M8.1 | **One step at a time, done.** (a) Sufficient decrease from a bound on the segment of one step: if `‖∇²f‖ ≤ L` on `segment ℝ x (gradientStep f η x)`, then `f (gradientStep f η x) ≤ f x - η (1 - η L) ‖∇f x‖²`, by `le_add_inner_add_mul_norm_sq_of_convex` on the segment. So `f` decreases along every orbit satisfying (S). (b) Avoidance: take `0 < η`, `η L < 1` and any Haar measure. The starts whose orbit has `‖∇²f(x_k)‖ ≤ L` at every `k` and converges to a strict saddle form a null set. These starts form a forward-invariant set, so `measure_inter_tendsto_strictSaddle_eq_zero` applies. (b) needs the bound only at the points `x_k`, not on the segments. | M7.1 |
| M8.2 | **Łojasiewicz convergence from a decrease, done.** New forms of `mul_norm_gradientStep_sub_le`, `mul_norm_iterate_succ_sub_le`, `iterate_mem_ball`, `mul_norm_iterate_sub_le`, `exists_tendsto_of_isLojasiewiczAt`, `tendsto_of_isLojasiewiczAt` and `gradient_eq_zero_of_frequently`. In them a decrease along the orbit, `f x_{k+1} ≤ f x_k - η κ ‖∇f x_k‖²` for every `k` with `κ > 0`, replaces `hL` and `η L < 1`. The constant `m (1 - a)(1 - η L)` becomes `m (1 - a) κ`. Where `hL` also gave the continuity of `∇f`, that continuity becomes a hypothesis. The present theorems keep their names and statements and follow with `κ = 1 - η L`; no proof is duplicated. Done so; the convergence forms ask, in place of `Differentiable ℝ f`, only that `f` be continuous at `z`. | M7.1 |
| M8.3 | **The scheme along the orbit, done.** Let `f` be `C²`, `0 < η`, `η L < 1`, `μ` a Haar measure and `ν ≪ μ`. Let `f` satisfy the Łojasiewicz inequality, with an exponent `0 ≤ a < 1`, at every critical point below `c`. Take `ν`-almost every `x` with `f x < c` whose orbit satisfies (S). Then either the orbit converges to a critical point `w` with `f w ≤ f x` that is not a strict saddle, or `‖x_k‖ → ∞`. There is no `hL`, no `hcl` and no compact sublevel set. Route: by M8.1, `f` decreases along the orbit. An orbit with a cluster point converges by M8.2; if the cluster point is not critical, Łojasiewicz holds there anyway (`isLojasiewiczAt_of_gradient_ne_zero`). By M8.1 (b), for almost every start the limit is no strict saddle. An orbit without a cluster point leaves every closed ball (`IsCompact.exists_mapClusterPt_of_frequently`). Also: the `hcrit` form, whose first alternative ends at a global minimizer, and the analytic forms, with neither `hloj` nor a level. Two examples. For `x⁴/4 - x²/2` on `ℝ`, with `η = 1/10` and `L = 23/4`, every start in `[-3/2, 3/2]` satisfies (S), although the Hessian is unbounded. For `-x`, every orbit satisfies (S) with `L = 0` and `‖x_k‖ → ∞`. Done so; the dichotomy for one orbit (`exists_tendsto_or_tendsto_norm_of_segment`) asks the Łojasiewicz inequality only at the critical points no higher than its start. | M8.1, M8.2 |
| M8.4 | **Rates along the orbit, done.** The gradient-descent rates of M6.9 (`exists_rate_estimates_of_isLojasiewiczAt`, `exists_le_mul_pow_of_isLojasiewiczAt`, `exists_le_mul_rpow_of_isLojasiewiczAt`) with the decrease of M8.2 in place of `hL`, and their forms under (S). The present theorems keep their names and statements. Done so: the `_of_le_sub` forms ask, in place of `Differentiable ℝ f`, only that `f` be continuous at the limit; the present theorems follow with `κ = 1 - η L`, the `_of_segment` forms with the decrease of M8.1. | M8.1, M8.2 |

What M8 tells about a run of gradient descent from almost every start: the run
ends in one of four ways.

1. It converges to a critical point with no negative Hessian eigenvalue: a
   global minimizer, a spurious local minimizer or a degenerate saddle.
2. It is still descending when the budget ends. M8.3 has no time scale; M8.4
   gives the rates.
3. `‖x_k‖` grows without bound.
4. (S) fails at some step: `η ‖∇²f‖ ≥ 1` there, and M8 says nothing.

A run checks (S) only at the points it samples, not on every segment.

## Order of work

1. M0.1: done on 2026-10-18.
2. M1.1 → M1.2 → M1.3, then M1.4.
3. M2.1, M2.3, M3.4a, then M4.1.
4. M2.2 → M4.2.
5. M3.1a, M3.1b → M3.1c, and M3.1d → M3.1e, M3.1f. These are independent
   chains, so separate agents can work on them. Then M3.1, and M3.3.
6. M5.1 → M4.3.
7. M1.5 → M5.2, M5.3.
8. M6.1 → M6.2, M6.3 → M6.4 → M6.8, and M6.1 → M6.6. M6.5 and M6.7 are
   independent. M6.8, M6.9 and M6.10 have low priority.
9. M7.1 first: it moves every module of the framework, so nothing runs
   beside it. Then M8.1 and M8.2, which are independent, then M8.3 and M8.4.

M2.4 is done. The research item M3.4b waits for an argument on paper. M5.4 is
stated on paper and waits for Lean. M3.4c has a counterexample in Lean; its
question for random `S` is open.

## Target `sorry`s

| `sorry` | Task | Closed by | Blocked on |
| --- | --- | --- | --- |
| `Perspective.strict_saddle` | #12 | M3.1, with parts M3.1a–f (#32–#37), done | — |
| `Perspective.beta_tiny_circle` | #27 | M3.1, M4.3, M5.1, M5.2, descent curves, done | — |
| `Perspective.beta_tiny` | #28 | `clustering_circle` (`d = 2`) and `boumal_clustering` (`d ≥ 3`), done | — |
| `Perspective.beta_interval` | #29 | M2.1, M3.1 (`d = 2`), M3.3 (`d ≥ 3`), M4.2, M5.1–M5.3 | — |
| `Perspective.boumal_clustering` | #30 | M2.1, M3.3, M4.2, M5.3 | — |
| `Landscape.Lojasiewicz.lojasiewicz_modulated` | #11 | M2.4, done | — |
| `Perspective.beta0_consensus` | #31 | outside the scheme, done | — |

`Perspective.beta0_consensus` follows from `almost_sure_consensus_beta0`.
`Section3_SmallBeta` comes before `AppendixA_Saddle` in the import order, so the
statement moved to the end of `AppendixA_Saddle`, where it is proved.
`Perspective.beta_tiny_circle` moved to `CircleClustering`, after its proof;
it holds for every `β ≥ 0` (`clustering_circle`), and the source's `n ≥ 2` is
dropped.  `Perspective.beta_tiny` moved to `AppendixC_BetaTiny`, after its proof
by `clustering_of_nonneg`; its constant `C`, numerical in the source, is now
chosen before `d` and `n`.

The other `sorry`s of `INDEX.md` have no tasks yet.
