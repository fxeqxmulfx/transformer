# Hard proof tasks: 30 original candidates

This is an approximate difficulty ranking from the 175 `sorry` declarations in
the initial `INDEX.md` snapshot. Difficulty reflects missing mathematical theory, the breadth of
the claim, and the amount of Lean infrastructure needed. `REVIEW.md` was also
checked: its historical `FIXED` notes are not current defects, while its
remaining source-fidelity warnings are called out below. The theorem in
`task.md` is already assigned and is omitted here.

Each row is a separate task. The statement-level comparison with the cited
manuscript is recorded below the list. Resolve every flagged discrepancy
before proving the corresponding theorem. If the statement is false, prove a
counterexample and document a faithful correction or refutation as required
by `AGENTS.md`. Never close a row by using another `sorry` as a hidden premise.
Refresh `INDEX.md` before starting, since work on the project may change the
count.

## Status audit (2026-09-29)

The list and paper comparison below are the original snapshot; read this status
before treating a row as open. The current generated `INDEX.md` has 169 `sorry`.

- **#20:** the quantitative SA/USA result is proved in
  [`DirectProof.lean`](src/Transformer/Metastability/DirectProof.lean), with the
  corrected sign in the first bound of `eq: lambda.3`. The theorem quantifies
  over solutions; existence and uniqueness of those solutions are not proved
  by this theorem.
- **#28:** the paper's finite-dimensional unbiasedness claim is refuted, also
  inside the documented scale window, by
  [`not_mean_rhtInv_msEden_window`](src/Transformer/Quartet/Section3_EdenBias.lean).
- **#29:** the printed Fourier decay is refuted for every `β > 2` and every
  fixed `t` by
  [`not_uniform_decay_of_two_lt`](src/Transformer/Modes/Section5_PtBddDecayFalse.lean).
  The repaired `uniform_decay` for `0 < β < 2` still has `sorry`.
- **#30:** [`theorem_A`](src/Transformer/AMSGrad/Section4_TheoremAFinite.lean)
  is proved with the printed constants and the paper's finite-horizon
  assumptions on losses, steps, and schedule. The proof uses a schedule-free projection
  potential and bounds the correction `g_t - m_{t-1}` with Young's inequality.
  [`not_abel_printed_actual_run`](src/Transformer/AMSGrad/Section4_TheoremAActualAbelFalse.lean)
  still refutes the scalar step of the original proof on an admissible run;
  it does not refute the regret bound.
- **#12, #13, #15, #21:** there are corrections and auxiliary proofs, but each
  listed main theorem still has `sorry`.

| # | Declaration | Main obstacle |
| ---: | --- | --- |
| 1 | [`Transformer.Metastability.saddle_to_saddle_gradient_reparam`](src/Transformer/Metastability/OpenProblems.lean) | The manuscript leaves the specified gradient-based time change open; a general staircase limit and a genuine energy-gradient construction are missing. |
| 2 | [`Transformer.Metastability.saddle_to_saddle`](src/Transformer/Metastability/OpenProblems.lean) | Open staircase conjecture for the full SA flow, with one time reparametrization covering every plateau. |
| 3 | [`Transformer.Perspective.strict_saddle`](src/Transformer/Perspective/StrictSaddle.lean) | Open classification of all non-maximal critical points of the torus energy, including Hessian directions. |
| 4 | [`Transformer.Perspective.phase_transition_curve`](src/Transformer/Perspective/Section5_HighDCurve.lean) | Uniform-in-time concentration of all pairwise overlaps around the scalar limit, with both error branches and constants uniform in dimension. |
| 5 | [`Transformer.Causal.subspace_cluster`](src/Transformer/Causal/MainTheorem.lean) | Paper conjecture for arbitrary query/key matrices and a multidimensional top eigenspace. |
| 6 | [`Transformer.Causal.two_cluster`](src/Transformer/Causal/MainTheorem.lean) | Paper conjecture requiring almost-everywhere convergence to the two signs of the top eigenvector under arbitrary query/key matrices. |
| 7 | [`Transformer.Homogenized.large_beta_logistic_limit`](src/Transformer/Homogenized/LogisticLimit.lean) | Dimension-dependent convergence in law of entire overlap paths to a logistic SDE under common noise. |
| 8 | [`Transformer.Homogenized.poc_wellposedness`](src/Transformer/Homogenized/WellPosed.lean) | Conditional McKean–Vlasov existence and a superposition principle linking weak SPDE solutions to the SDE. |
| 9 | [`Transformer.Homogenized.propagation_of_chaos`](src/Transformer/Homogenized/PropagationChaos.lean) | Uniform-in-time expected particle error and Wasserstein estimates for a coupled system with common noise. |
| 10 | [`Transformer.Homogenized.exists_mckeanVlasovSolution`](src/Transformer/Homogenized/WellPosed.lean) | Construct a filtered probability space, common noise, and a nonlinear fixed-point solution. |
| 11 | [`Transformer.Homogenized.large_beta_metastability`](src/Transformer/Homogenized/Metastability.lean) | High-probability long-time overlap control from martingale concentration and Gaussian kernel asymptotics. |
| 12 | [`Transformer.Kinetic.lost_correlations`](src/Transformer/Kinetic/Correlations.lean) | Quantitative one- and two-point correlation expansions with Fourier estimates uniform in particle number. |
| 13 | [`Transformer.Kinetic.mean_field_limit`](src/Transformer/Kinetic/MeanField.lean) | Existence and uniqueness of the layered mean-field equation plus propagation of empirical measures. |
| 14 | [`Transformer.Perspective.boumal_clustering`](src/Transformer/Perspective/Section5_HighD.lean) | Almost-everywhere clustering for both SA and USA in every dimension at least three and at every nonnegative temperature. |
| 15 | [`Transformer.Interpolation.compression`](src/Transformer/Interpolation/Clustering.lean) | Simultaneous clustering of diffuse measures into distinct atomic targets while preserving separation of their geodesic convex hulls. |
| 16 | [`Transformer.Modes.mainResult_expectedModes`](src/Transformer/Modes/Section1_Main.lean) | Global two-sided asymptotics for the expected number of modes, assembling the Kac–Rice integral and tail estimates. |
| 17 | [`Transformer.Modes.kacRice`](src/Transformer/Modes/Section2_KacRice.lean) | General Kac–Rice formula quoted from external sources, with path regularity, joint densities, and crossing counts. |
| 18 | [`Transformer.Modes.edgeworth_three`](src/Transformer/Modes/Section3_BR.lean) | Weighted two-dimensional Edgeworth expansion with an order-`n⁻¹` density error; the cited probability theorem is not in the library. |
| 19 | [`Transformer.Homogenized.weak_error_modified`](src/Transformer/Homogenized/WeakError.lean) | High-order weak error for the modified SDE at grid times. `REVIEW.md` records a refutation of the source's all-time version. |
| 20 | [`Transformer.Metastability.metastability`](src/Transformer/Metastability/DirectProof.lean) | Proved in corrected form; see status audit above. |
| 21 | [`Transformer.FrankWolfe.metastability`](src/Transformer/FrankWolfe/Section5_Metastability.lean) | Corrected exit-time bound remains unproved. The chain estimate, union over bad-step subsets, and binomial-tail arithmetic are missing. |
| 22 | [`Transformer.Clusters.wellposed_contEq`](src/Transformer/Clusters/Section6_ContEq.lean) | Global existence and uniqueness of a nonlinear, compactly supported measure-valued continuity equation. |
| 23 | [`Transformer.Clusters.contEq_w2_stability`](src/Transformer/Clusters/Section6_ContEq.lean) | Quantitative Wasserstein stability for all solutions of that equation on bounded time intervals. |
| 24 | [`Transformer.Perceptron.any_d_generic_isFinitelyAtomic`](src/Transformer/Perceptron/HigherDim.lean) | Generic finite atomicity from real-analytic zero-set geometry and stationary measure conditions in any dimension. |
| 25 | [`Transformer.Perceptron.any_d_measure_support_eq_zero`](src/Transformer/Perceptron/HigherDim.lean) | Show stationary or strict-SOPD measures have support of zero spherical volume, including the non-analytic ReLU case. |
| 26 | [`Transformer.CRASP.rtfr_pes_depth_hierarchy`](src/Transformer/CRASP/EncodingHierarchy.lean) | Requires positive transformer constructions and three positional-encoding simulations; `REVIEW.md` notes that the manuscript argument does not cover `k = 0`. |
| 27 | [`Transformer.Causal.single_cluster`](src/Transformer/Causal/MainTheorem.lean) | Almost-everywhere convergence of causal attention to the first token for arbitrary query/key matrices. |
| 28 | [`Transformer.Quartet.mean_rhtInv_msEden`](src/Transformer/Quartet/Section3_EdenBias.lean) | **Refuted.** Exact finite-dimensional unbiasedness fails throughout the paper's scale window; the full-vector and GEMM biases are also proved. |
| 29 | [`Transformer.Modes.uniform_decay`](src/Transformer/Modes/Section5_PtBddFourier.lean) | **Refuted as printed** for every `β > 2` at each fixed location; the repaired `0 < β < 2` theorem still has `sorry`. |
| 30 | [`Transformer.AMSGrad.theorem_A`](src/Transformer/AMSGrad/Section4_TheoremAFinite.lean) | **Solved.** Printed constants proved for the paper's finite-horizon assumptions; the paper's critique of the original proof step remains valid. |

## Paper statement check

I compared the hypotheses, quantifier order, constants, and conclusions of
these declarations against the local TeX manuscripts. This is a statement
audit; proving equivalence of every auxiliary predicate remains part of each
proof task. **Needs repair** means the present Lean claim must be corrected
before it can be called a formalization of the cited claim. **Changed** means
the departure is already described in the Lean docstring, but the manuscript
does not literally state the Lean theorem.

| # | Manuscript | Result of comparison |
| ---: | --- | --- |
| 1 | [§6, reparametrization candidate](papers/arXiv-2410.06833v1/actual_main.tex#L2035) | **Changed.** The source asks whether its staircase question holds for the displayed gradient time change. Lean uses only SA, the rescaled energy `2β Eβ`, and an energy-derivative predicate in place of a constructed Riemannian gradient. The source also permits either SA or USA and omits the final plateau. |
| 2 | [§6, saddle-to-saddle problem](papers/arXiv-2410.06833v1/actual_main.tex#L1795) | **Changed.** The source permits SA or USA, any continuous nonnegative time change, the printed `Eβ`, and convergence before the last plateau. Lean restricts to SA, requires an increasing unbounded time change, rescales by `2β`, and includes the last plateau; its docstring explains why the printed formulation is trivial. |
| 3 | [§7, strict-saddle problem](papers/arXiv-2312.10794v5/survey.tex#L1298) | **Aligned.** For `β > 0` and `n ≥ 2`, Lean asks that a critical non-maximum have a direction with positive second derivative, corresponding to a positive Hessian eigenvalue. |
| 4 | [§6.2, phase-transition curve](papers/arXiv-2312.10794v5/survey.tex#L1040) | **Aligned.** `β ≥ 0`, `n ≥ 2`, the dimension threshold, uniform initial law, probability `1-2n²d⁻¹ᐟ⁶⁴`, both branches of the minimum, and `C, λ` chosen before `d` agree with the statement. |
| 5 | [§4, Conjecture thm2](papers/arXiv-2411.04990v2/main_arxiv.tex#L463) | **Changed to repair a source typo.** The prose says projection onto the top eigenspace `L`, while its displayed definition uses `Lᗮ`. Lean follows the prose; the eigenvalue, invariant complement, strict transverse bound, and almost-everywhere conclusion otherwise agree. |
| 6 | [§4, Conjecture thm1.5](papers/arXiv-2411.04990v2/main_arxiv.tex#L454) | **Aligned.** Distinct positive real eigenvalues, arbitrary `Q,K`, and almost-everywhere convergence of each token to `ξ` or `-ξ` are present. The Lean statement also has a vacuous `n = 0` instance. |
| 7 | [thm:large_beta_meta, second part](papers/arXiv-2604.01978v1/main.tex#L1023) | **Changed.** Lean encodes `d β⁻¹ᐟ² → 0` through effective temperature and path-law test functions, adds path continuity and integrability hypotheses, and asserts existence of a logistic limit but omits the source's uniqueness assertion. |
| 8 | [thm:PoC_wellposedness](papers/arXiv-2604.01978v1/main.tex#L1392) | **Changed.** The source uses a specified common Wiener process and pathwise uniqueness. Lean's martingale-problem language produces a space existentially and concludes uniqueness of finite-dimensional distributions instead. The SPDE superposition component is retained. |
| 9 | [prop:poc](papers/arXiv-2604.01978v1/main.tex#L2239) | **Aligned after encoding.** Lean uses `n+1` for positive particle count, `IsLUB` witnesses for the time suprema, and one constant chosen before `n`; the same particle and empirical-`W₂` estimate is asserted. |
| 10 | [prop:mckean_vlasov_common_noise](papers/arXiv-2604.01978v1/main.tex#L2189) | **Changed.** The source constructs an adapted continuous solution on a given filtered space carrying the noise. Lean asks for a solution on some space in martingale-problem form, since its predicate does not name that noise. |
| 11 | [thm:large_beta_meta, first part](papers/arXiv-2604.01978v1/main.tex#L1023) | **Changed.** The printed `C` is universal and its remainder lacks a factor `T`; Lean lets `C` depend on the density bounds, uses effective temperature, standard value scaling, and includes `T` in the accumulated remainder. These corrections are described in the Lean docstring. |
| 12 | [th:lost](papers/arXiv-2605.09213v1/main.tex#L374) | **Partly repaired, still open.** Lean now requires a probability profile and a smooth periodic test function. It asks for the cross-correlation estimate only for distinct token indices, because the same-cell version appears false but is not refuted in Lean. The correlations are still densities rather than the paper's measures or distributions; the theorem has `sorry`. |
| 13 | [th:mf-gpt(i)](papers/arXiv-2605.09213v1/main.tex#L329) | **Repaired encoding, still open.** The initial profile is a measurable cursor kernel, `β > 0`, and equality is almost everywhere in `σ`, matching the disintegration reading of the source. The main theorem still has `sorry`. |
| 14 | [thm:boumal](papers/arXiv-2312.10794v5/survey.tex#L855) | **Aligned.** The source fixes `n ≥ 2`, `d ≥ 3`, `β ≥ 0` and obtains almost-everywhere clustering for both SA and USA; Lean states both halves. |
| 15 | [prop:compression](papers/arXiv-2411.04551v3/paper.tex#L604) | **Aligned for `N ≥ 1`, still open.** Atomless inputs, disjoint geodesic convex hulls, distinct target atoms within their input hulls, common piecewise constant control, `W₂` accuracy, and final separation are present. Lean also permits `N = 0`, but now has a nondegenerate `N = 2` satisfiability example. The main theorem has `sorry`. |
| 16 | [thm:main-result, point 1](papers/arXiv-2412.09080v3/paper.tex#L324) | **Aligned.** `IsRegime` carries `nᶜ ≲ β ≲ n²⁻ᶜ` with both tending to infinity; the conclusion is the source's `Θ(√(β log β))` expected mode count. |
| 17 | [thm:kac-rice](papers/arXiv-2412.09080v3/paper.tex#L534) | **Aligned.** `IsKacRiceField` spells out all four source hypotheses: almost-sure `C¹`, finite second moments, continuous marginal and joint densities, and the modulus bound. The conclusion is the same formula as an extended nonnegative integral. |
| 18 | [thm:br](papers/arXiv-2412.09080v3/paper.tex#L757) | **Specialized.** The paper quotes arbitrary dimension `k` and order `s ≥ 2` under unspecified suitable conditions. Lean takes `k = 2`, `s = 3`, fourth moments, an integrable characteristic-function power, and exponential moments for its cumulant definition. |
| 19 | [paragraph after thm:weak_error_clean](papers/arXiv-2604.01978v1/main.tex#L637) | **Changed after refutation.** The source prints the improved `η·max(η,α)` rate uniformly over all times. Lean asserts it at grid times only; `not_weak_error_modified_printed` refutes the all-time version. |
| 20 | [thm:metastability](papers/arXiv-2410.06833v1/actual_main.tex#L297) | **Proved with a documented correction.** Lean now has both bounds on `λ`, explicit bounds on `T₁,T₂`, and SA or USA. The first printed `λ` bound has a sign error before its logarithm; Lean uses the bound needed to prove `T₁<T₂`. It quantifies over solutions and does not itself prove their existence or uniqueness. |
| 21 | [lem:metastab.1](papers/arXiv-2508.09628v1/main.tex#L876) | **Corrected, still open.** The printed exponent omits the polytope diameter and a binomial factor. Lean uses `M = ⌈ε/(γ·diam K)⌉`, an explicit smallness condition on `ρ + ε`, and `σ` fibres for the groups; its docstring records the changes. The theorem has `sorry`, and the printed bound has no Lean refutation yet. |
| 22 | [c:wellposedtransformers](papers/arXiv-2305.05465v6/arxiv.tex#L600) | **Aligned.** Compactly supported initial probability measure, global unique continuity-equation solution, and the paper's solution class are represented by `IsContEqSolution`. |
| 23 | [e:w2estimate](papers/arXiv-2305.05465v6/arxiv.tex#L604) | **Aligned up to ball convention.** `R,T > 0`, a constant depending on those and the fixed matrices, and the exponential `W₂` estimate agree. Lean permits support on the boundary of the closed radius-`R` ball; the paper writes `B(0,R)`. |
| 24 | [thm:any.d(ii)](papers/arXiv-2601.21366v2/preprint.tex#L604) | **Aligned.** Analytic activation with no nonzero roots, open dense parameter set, stationary measure, and finite atomic support agree. |
| 25 | [thm:any.d(i)](papers/arXiv-2601.21366v2/preprint.tex#L604) | **Aligned.** Lean includes both the nonanalytic ReLU-stationary and analytic strict-SOPD alternatives, with `β > 0` and zero spherical volume of the support. |
| 26 | [§4.5 and Appendix F](papers/arXiv-2506.16055v3/appendix.tex#L1143) | **Changed restriction and quantifiers.** The headline theorem does not mention rational angles, but Appendix F assumes them for sinusoidal embeddings and RoPE; Lean makes that premise explicit. Lean fixes each particular encoding `pe` and requires a positive construction using that same `pe`, while the headline says the transformers *can use* one of the three encoding types. Its `k = 0` case is in the headline but not covered by the manuscript's cited depth argument. |
| 27 | [thm1](papers/arXiv-2411.04990v2/main_arxiv.tex#L436) | **Changed to repair a boundary case.** The source has no stated dimension bound; Lean adds `d ≥ 2` because the claimed almost-everywhere clustering fails on `S⁰`. With `n ≥ 1` and `V = I`, the arbitrary `Q,K` and first-token limit agree. |
| 28 | [§3.3 corollary](papers/arXiv-2601.22813v2/main.tex#L341) | **Refuted.** Exact finite-dimensional unbiasedness fails even at the paper's clipping scale and throughout its documented scale window; `not_mean_rhtInv_msEden_window` proves this in Lean. |
| 29 | [eq:uniform-decay](papers/arXiv-2412.09080v3/paper.tex#L1255) | **Refuted as printed.** The constant cannot be uniform in `t`; more strongly, for every `β > 2` and fixed `t`, the asserted decay fails. Lean's repaired `0 < β < 2` statement still has `sorry`. |
| 30 | [Theorem A](papers/arXiv-1904.03590v4/paper.tex#L398) | **Proved.** The printed constants hold with assumptions on `f_t`, `α_t`, and `β₁,t` only for `1 ≤ t ≤ T`. Lean makes implicit `0 ≤ β₁,t`, `β₁ < 1`, `0 < β₂ < 1` explicit and uses `β₁/√β₂ < 1` because the formula divides by `1-γ`. The conclusion holds for every feasible comparator, including the paper's minimizer. The scalar step of the original proof remains refuted on an admissible run. |

## Other candidates from the original check

- [`energy_level_metastability`](src/Transformer/Metastability/OpenProblems.lean)
  originally used a vacuous predicate that allowed zero caps. `IsMetastable`
  now requires a nonempty cap cover and the quantitative hypotheses of the
  [§4 energy-level problem](papers/arXiv-2410.06833v1/actual_main.tex#L1440).
  The statement also requires a nonempty energy window and remains unproved.
- The former `universal_approximation_measure` had no matching theorem in the
  cited [survey §10](papers/arXiv-2312.10794v5/survey.tex#L1518).
  [`not_universal_approximation_measure`](src/Transformer/Perspective/Section9_ApproximationMeasure.lean)
  refutes that overbroad statement; it has been deleted.
- [`main_result`](src/Transformer/Interpolation/Main.lean) now follows
  [thm:main.result](papers/arXiv-2411.04551v3/paper.tex#L332) in asking for
  piecewise constant control without a universal `O(dN)` switch bound. The
  [standing assumption](papers/arXiv-2411.04551v3/paper.tex#L300) of distinct
  inputs and targets is explicit. The theorem remains unproved.
- [`layer_clustering`](src/Transformer/GPTMini/ClusteringTheorem.lean) transfers
  the continuous causal [thm1](papers/arXiv-2411.04990v2/main_arxiv.tex#L436)
  to a discrete Pre-LN recursion with RoPE, which that paper does not study.
  `REVIEW.md` correctly describes this as an unproved project conjecture.

For any completed row, build the affected module and then run `lake build`,
`lake env lean scripts/Axioms.lean`, and `python3 scripts/index.py`. The audit
must have zero proved declarations resting on `sorryAx` and zero extra axioms;
the `sorry` count must not rise. Do not commit unless explicitly requested.
