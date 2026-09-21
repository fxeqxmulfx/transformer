# Review

Audit of every proved theorem against its source: the Lean statement and the
definitions it rests on say what the paper says — no weakened hypothesis, no
strengthened assumption, no definition that does the work.  A refutation must
kill the claim the paper actually makes, not a misreading of it.  The proofs
themselves are vouched for by `lake build` and `scripts/Axioms.lean`.

Marks: `ok` checked and faithful · `fixed` corrected in a commit · `issue` open.

## 1. Refutations (78)

- ok `not_marked_of_clearKey` — Transformer/ALM/HullClear.lean:119 — own ALM lemma, no paper claim
- ok `not_eraseStep_of_lift` — Transformer/ALM/HullLift.lean:84 — own ALM lemma, no paper claim
- ok `not_eraseStep_of_marked` — Transformer/ALM/HullMark.lean:116 — own ALM lemma, no paper claim
- ok `not_tie_three` — Transformer/ALM/HullScan.lean:71 — own ALM lemma, no paper claim
- ok `not_isBinary_odd` — Transformer/ALM/ScoreWall.lean:84 — own ALM lemma, no paper claim
- ok `not_taylor` — Transformer/AMSGrad/Section2_Prelim.lean:97 — Lemma 2.3 as printed
- ok `not_red_ineq` — Transformer/AMSGrad/Section3_Optimal.lean:92 — red inequality of §3, Example 3.2 parameters match
- ok `not_cor_lower` — Transformer/AMSGrad/Section4_CounterRegret.lean:78 — literal lim R(T)/T=0 false (β₁=0 fits both settings); faithful upper half kept as cor_lambda/cor_inv
- ok `not_altList_succ_sublist` — Transformer/CRASP/Alternating.lean:186 — helper lemma (block count), not a refutation
- ok `not_altList_not_sublist_of_mem_altPlus` — Transformer/CRASP/Alternating.lean:212 — helper lemma for L_k = A_k∖B_k, not a refutation
- ok `not_recognizes_altPlus_of_clustered` — Transformer/CRASP/Collapse.lean:141 — own conjecture (not a paper claim), paper's thm:rtfr_to_TLCl carried as hypothesis
- fixed `not_minimalOneConstantOn_firstNotA` — Transformer/CRASP/CroppingUnsound.lean:91 — 064b44f: ConstantOn now over positions 1..|w| as in def:constant
- fixed `cropping_oneway_unsound` — Transformer/CRASP/Depth.lean:81 — 064b44f: faithful after ConstantOn fix; kills lem:cropping_oneway (PNPs free before I)
- fixed `cropping_oneway_right_unsound` — Transformer/CRASP/Depth.lean:96 — 064b44f: same, second half
- fixed `reduction_past_unsound` — Transformer/CRASP/Depth.lean:117 — 064b44f: now at exact depth 2 (topTwo); literal, rests on the middle's closed lower end ℙ(λ)
- fixed `reduction_unsound` — Transformer/CRASP/Depth.lean:133 — 064b44f: same, TLCP version (typo TLCP_k read as k-1)
- ok `not_definableL_altPlus` — Transformer/CRASP/LowerBound.lean:114 — thm:TLCl_depth lower half, k>0 as in paper
- ok `not_definable_altPlus_double` — Transformer/CRASP/LowerBoundTwoSided.lean:210 — thm:TLC_depth lower half; k=0 case added in 07d6b08
- ok `not_forall_closed_majTwo_of_definable` — Transformer/CRASP/MajTwoDepthOne.lean:159 — paper: closed φ defines L(φ); TLC_0 ∋ Q_b; Maj2 syntax/semantics/depth match def:MAJtwo, def:depth_MAJtwo
- ok `tr_paperBlockSize_unsound` — Transformer/CRASP/PositionalReductionEquiv.lean:148 — T_ρ atoms match appendix 1027-1037; at w=ε, r=M(Y+1)=2 gives T_2[Y Q_e]=⊥ but f(ε)=ee ⊨ Y Q_e; kills the proof's r, the lemma itself is proved with reach ≤ r
- fixed `not_solvesPrediction_altPlus` — Transformer/CRASP/Prediction.lean:91 — SolvesPrediction = def:prediction_task; corollary had an added 0<k, removed, k=0 proved (522cd3f)
- ok `not_lang_eq_restrict_startAB` — Transformer/CRASP/ReductionUnsound.lean:74 — helper for reduction refutation
- ok `nonstrict_subseq_formula_unsound` — Transformer/CRASP/Subsequence.lean:187 — appendix 148 formula with inclusive ◁# (neurips 364) accepts 'a' for σ1=σ2=a; kills the proof's formula, lemma proved with strict count
- ok `nonstrict_twoSided_formula_unsound` — Transformer/CRASP/SubsequenceTwoSided.lean:182 — appendix 157-162 at k=1, pattern aaa, inclusive ◁#/▷#; kills the proof's formula only
- fixed `cropping_unsound` — Transformer/CRASP/TLCDepth.lean:48 — 064b44f: same, Appendix D lem:cropping, I ⊆ [0,n] hypothesis present
- ok `pushY_unguarded_unsound` — Transformer/CRASP/YNormalFormEquiv.lean:123 — appendix 928-935 N^c rules give Y¬Q_a ↦ ¬Y Q_a; Y needs i>1 (902); disagree on 'a'. kills the proof's transform, thm:ynf proved with guards
- issue `not_forall_single_token_convergence` — Transformer/Causal/SingleToken.lean:124 — refutes an earlier misformalization (declared so), correct; but the survey's lemma1 itself is not stated anywhere — invisible debt, needs L'(V), L(V) via generalized eigenspaces
- issue [FIXED: PeriodicAngles hypotheses added, example now E = cos] Transformer/Causal/SequentialFlow.lean (sorried): `sequentialFlow_converges` is stated on ℝⁿ with no 2π-periodicity of E, Z; E_k = -φ_k, Z = 1 has no critical points and φ_k(t) = φ_k(0) + t diverges, so it is false as written
- ok `not_tendsto_zero_of_tendsto_common` — Transformer/Clusters/Section7_HigherDim.lean:87 — eq:P matches; kills literal x_i(t)->(1,0) with P->I for every Q,K; rescaled reading untouched, docstring says so
- ok `not_tendsto_id_of_tendsto_common` — Transformer/Clusters/Section7_HigherDim.lean:104 — eq:P matches; kills literal x_i(t)->(1,0) with P->I for every Q,K; rescaled reading untouched, docstring says so
- ok `not_configHull_subset_of_preconditioner` — Transformer/FrankWolfe/Section2_HullFailure.lean:86 — the remark's own example proved: P=diag(0.6,0.7)=(I+V)^{-1}V for V=diag(3/2,7/3), unique argmax x3, (0.4,0.7)∉K
- ok `not_rope_clustering_antipodalPair` — Transformer/GPTMini/Bridge/RoPENoClustering.lean:72 — kills the repo's former ∀X₀ rope_clustering, not a paper claim; declared as such; paper's a.e. statement not touched
- ok `not_forall_rope_clustering` — Transformer/GPTMini/Bridge/RoPENoClustering.lean:118 — kills the repo's former ∀X₀ rope_clustering, not a paper claim; declared as such; paper's a.e. statement not touched
- fixed `not_mean_field_clustering` — Transformer/GPTMini/MeanFieldRefutation.lean:85 — kills the repo's free-W₂ gpt-mini transfer, not the survey; docstring claimed W₂ inexpressible, corrected to point at thm:mfclust in MeanField.Clustering
- fixed `not_forall_overlapDrift_eq_simplexDrift` — Transformer/Homogenized/GramStability.lean:52 — witness moved to sigma_V^2=1/d (gaussHeadLaw, sigma_A=0); gap -2g(0)/d nonzero for every law
- fixed `not_forall_satisfying_MF_rate` — Transformer/Homogenized/MeanFieldLipschitz.lean:180 — K moved inside model data (weakest reading of O); witness sigma_A=0 admissible under ass:high_order_short
- ok `not_forall_clustering_to_atom` — Transformer/Interpolation/AtomClustering.lean:102 — kills the repo's free-Winf form, not the paper; repaired statement is clustering_to_atom with IsWinfToDirac
- ok `not_exists_ball_of_mass_of_dirac` — Transformer/Interpolation/BallDecomposition.lean:42 — shows the dropped a.c. hypothesis of cl: balls is necessary; restored in claim_balls
- ok `not_forall_monge` — Transformer/Interpolation/Main.lean:258 — kills the repo's free-W2/free-constant form; repaired and proved as monge (C=1)
- fixed `not_forall_wToBall` — Transformer/Interpolation/MassConcentration.lean — refuted only the free-W₂ strawman; replaced by `wToBall` proved for the real W₂ (C=1), and lem: mass.concentration.Q1 found false as printed (Cη rate): `not_massConcentrationQ1`, corrected to 2√η in `massConcentrationQ1_sqrt`
- ok `not_forall_Hartman_Grobman` — Transformer/Interpolation/Settling.lean:271 — kills the repo's former unrestricted form (every path, every omega), docstring says so; but see issue on Hartman_Grobman in section 2
- ok `not_exists_rate_at_simplex` — Transformer/MeanField/Equiangular.lean:75 — necessity of the basin condition; paper assumes rho0 in [0,1], so no paper claim is refuted
- ok `not_equiangular_local_rate_zero` — Transformer/MeanField/EquiangularRate.lean:159 — necessity of 1 <= n (n=0 kills the denominator); no paper claim refuted
- ok `not_forall_bakry_emery` — Transformer/Metastability/BakryEmery.lean:185 — kills the repo's former free gradNorm/gradHess form, docstring says so; bakry_emery restores the paper's hypotheses
- ok `not_quantitative_inequality_two_mul` — Transformer/Metastability/ExponentialFlow.lean:61 — 2c refuted under the charitable reading (PL along the path, flow through v); ascent/descent is E -> -E; docstring's sign note corrected (paper's lemma is ascent)
- ok `not_rho_diff_ineq_of_free` — Transformer/Metastability/MainTheorem.lean:159 — necessity of tying rho_q to the configuration (repo's free-rho_q form); no paper claim
- issue `not_forall_cap_exit` — Transformer/Metastability/MeanField.lean:182 — refutes only the free-(eta,V) strawman; the paper's claim is dropped from the books, yet eta_q(t)=mu(t)(cap), V_q(t) are definable from a solution mu of eq: mean.field.pde
- issue `not_forall_variance_small` — Transformer/Metastability/MeanField.lean:211 — same as not_forall_cap_exit: strawman refuted, paper's eq: v.small not on the books
- ok `not_otto_reznikoff` — Transformer/Metastability/OttoReznikoff.lean:104 — FIXED: thm: Otto result refuted as printed (E ≥ 0, HasGradientAt, true flow, v the H1-projection); fails at t = 0 for E = x²/2, N = {0}
- issue `not_forall_claim_one` — Transformer/Metastability/OttoReznikoff.lean:258 — refutes only the free-(Theta,r) strawman; claim: 1 dropped from the books, though expressible with hypotheses (indices of one cap, increasing angles, Theta not in slowManifold)
- ok `not_forall_reverse_PL_acceleration` — Transformer/Metastability/ReversePL.lean:142 — necessity of hchain; the faithful reverse_PL_acceleration (with chain rule) is proved
- issue `not_exact_time_scale` — Transformer/Metastability/Staircase.lean:127 — literal statement (u0 in [0,1], tex 1950) correctly refuted at u0=0, beta=e; but the substantive error (centre 1/(2c), not 2/c) and the corrected asymptotic lemma for u0 in (0,1] are not on the books
- ok `not_forall_modeSet_subset_upcrossingSet` — Transformer/Modes/Section2_Degenerate.lean:90 — tex 562 'i.e. t is a mode' read pathwise; mode = local max; (-1,1), beta=1 degenerate max; expectation identity kept sorried
- ok `not_integrableOn_tildeY` — Transformer/Modes/Section3_ErrorThird.lean:44 — a paper claim (tex 773 divergence), proved, not a refutation
- ok `not_exists_hermite_le_cube` — Transformer/Modes/Section3_Hermite.lean:68 — tex ~826 'Trivially |H^(k,3-k)(x)| <~ ||x||^3' for all x; H^(3,0)(e,0)=e^3-3e; corrected 3(|x|+|x|^3) proved
- ok `not_uniform_decay` — Transformer/Modes/Section5_PtBddFourier.lean:104 — tex: 'implicit constant depends only on beta' after a wlog t=0 reduction; G,G' match eq: Gt; refutes the t-uniform reading; fixed-t uniform_decay kept sorried
- ok `not_isDensityOf_one` — Transformer/Modes/Section5_PtBddOne.lean:64 — paper's own remark (n=1 no continuous density), proved stronger; not a refutation
- ok `not_synchronizes_const` — Transformer/Normalization/ClusteringLine.lean:51 — helper, not a refutation
- ok `not_clusters_from_uniform_one` — Transformer/Normalization/ClusteringLine.lean:67 — tex 421-437 states thm:convergence/corollary for every d (S^{d-1}); at d=1 tangent space 0; Q=K=V=I, n=2; corrected d>=2 forms kept
- ok `not_clusters_or_stalls_from_gaussian_one` — Transformer/Normalization/ClusteringLine.lean:155 — tex 421-437 states thm:convergence/corollary for every d (S^{d-1}); at d=1 tangent space 0; Q=K=V=I, n=2; corrected d>=2 forms kept
- ok `not_unconditional_synchronization_one` — Transformer/Normalization/ClusteringLine.lean:172 — tex 421-437 states thm:convergence/corollary for every d (S^{d-1}); at d=1 tangent space 0; Q=K=V=I, n=2; corrected d>=2 forms kept
- ok `not_ae_gaussian_pair` — Transformer/Normalization/Line.lean:154 — helper, not a refutation
- ok `not_forall_initial_velocity_small` — Transformer/Normalization/Rates.lean:122 — kills the repo's former free-sigma form; faithful initial_velocity_small against UniformTuple kept sorried
- ok `not_isAnalyticOnSphere_relu` — Transformer/Perceptron/Analytic.lean:142 — satisfiability witness for thm:circle / thm:any.d(i) hypotheses, not a refutation
- ok `not_subsingleton_sphereHyperplane_iff` — Transformer/Perceptron/Hyperplane.lean:99 — the paper's own equivalence (rem: ext (i)), proved with a!=0, d>=2 added; deviation recorded; not a refutation
- issue `not_alpha_at_one_over_n_of_free` — Transformer/Perspective/AppendixD_Alpha.lean:115 — strawman: tex 1885 pins x* as the common limit (in the cone of x_i(1/n), e:decompox*.step2); docstring's 'never says which x*' is wrong; alpha_at_one_over_n proves it for x*=x_0(1/n) instead — a different statement under the paper's name
- issue `not_forall_diff_ineq_alpha` — Transformer/Perspective/AppendixD_AlphaDeriv.lean:255 — shows necessity of hhull, but hhull (x* in convex hull of X(s) for all s) is the wrong hypothesis: paper has the cone (eta x*); on the sphere hull membership forces x* = particles, near-vacuous
- issue `not_forall_product_close_to_one` — Transformer/Perspective/AppendixD_Product.lean:190 — strawman: x* = antipode of a lone particle, whereas the paper's x* is the particles' limit (there alpha=1 and the bound holds); its docstring's claim to refute the survey's form is false
- issue `not_forall_ybeta_close_to_1` — Transformer/Perspective/AppendixD_Ybeta.lean:214 — tex 1869 'for any t>=0'; ybetaODE_SA = eq: ybeta with gamma(0)=0; n=1,beta=0 closed form; (also fails at n=2, beta=0: 1 > e^{2/3}/2); BUT the Lean witness is n=1 while thm: phase.transition.curve fixes n>=2 (tex 749-856): move the witness to n=2, gamma=tanh
- ok `not_forall_usa_analogue` — Transformer/Perspective/AppendixD_YbetaUSA.lean:182 — necessity of n>=2 (paper's standing assumption); refutes no paper claim
- issue `not_russian_trick_one` — Transformer/Perspective/RussianTrick.lean:150 — rests on Lean's 0⁻¹ = 0: on paper 1/(d-1) is undefined at d=1, and the cleared form -(d-1)I = sum B_j^2 holds at d=1; the survey's standing assumption is d>=2; docstring overclaims a refutation — reword as a d>=2 remark (russian_trick itself ok)
- issue `not_step2_decomposition` — Transformer/Perspective/Section5_Hemisphere.lean:196 — strawman (free x*); and step2_decomposition's hhull (x* in convex HULL) is stronger than tex 967 (eta x* in hull, eta in (0,1], i.e. the cone) and fails for any x* on the sphere off the particles
- issue `not_forall_step2_alpha_diff_ineq` — Transformer/Perspective/Section5_HighD.lean:221 — strawman (x* = antipode, n=1, d=1 outside d,n>=2); the faithful step 2 (x* the limit under the hemisphere hypothesis) is not on the books — hemisphere_clustering carries it as hstep
- ok `not_sharpConfiguration_antipodal` — Transformer/Perspective/Section8_CohnKumar.lean:138 — helper
- ok `not_cohn_kumar_dichotomy` — Transformer/Perspective/Section8_CohnKumar.lean:188 — tex 1387-1392: m>1 clause printed; n=2 antipodal pair minimises H_beta (2e^beta+2e^{beta<u,v>}); a 600-cell has 120 points so excepting any one finset is stronger; design clause untouched so free sigma harmless
- ok `not_torusHessianNonPos_of_strictSaddle` — Transformer/Perspective/StrictSaddle.lean:60 — definitional helper, not a refutation
- ok `not_reverse_of_heads_eq_zero` — Transformer/RASP/Compilation.lean:157 — a positive result (one head is necessary for reverse), not a refutation; indices-dependence handled (indices const per position)
- ok `not_forall_raspGeneralizationConjecture` — Transformer/RASPL/Conjecture.lean:141 — meta: justifies keeping the empirical conjecture a predicate of LengthGeneralizes; the paper claims nothing about all predicates
- ok `not_degPLt_zero` — Transformer/RASPL/MinDegree.lean:94 — helper, not a refutation: weights are sums of squares

## 2. Proved theorems, by module (511 modules, 2420 theorems)


### Transformer.ALM — —
- Directory note: 0 sorries. The cited source code (hull2d_cht.h, vm-rs/*, todo3.md, graph/core.py) is not in the repository, so code-line claims, measurements and shipped constants are unverifiable; statements are faithful to their own definitions. The recurring defect is satisfiability examples that restate the conclusion instead of witnessing the hypotheses.


- ok Transformer/ALM/Basic.lean (6): `score_self`, `score_gap`, `score_lt_of_ne`, `score_isGreatest`, `norm_sq_eq_sum`, `one_le_dist_sq_of_int`
- ok Transformer/ALM/BinSearch.lean (11): `bsearch_zero`, `bsearch_succ`, `bcount_zero`, `bcount_succ`, `le_bsearch`, `bsearch_le`, `bsearch_congr`, `bsearch_lt`, `bsearch_ge_of_lt`, `bcount_le_log`, `hull_bsearch_isGreatest`
- ok Transformer/ALM/BuildFinger.lean (13): `one_le_stepCost`, `searchCost_ge`, `searchCost_le`, `portCost_le_buildCost`, `portCost_le`, `searchCost_of_all_ends`, `portCost_of_all_ends`, `portPrices_nonempty`, `portPrices_ge`, `portPrices_le`, `portPrices_ratio`, `portCost_moves_on_the_paraboloid`, `stress_port_ratio`
- ok Transformer/ALM/BuildOrder.lean (10): `buildPrices_nonempty`, `buildPrices_ge`, `buildPrices_le`, `buildPrices_spread`, `buildPrices_ratio`, `buildPrices_nontrivial`, `log_two_stress`, `stress_spread_le`, `stress_le`, `buildPrices_paraboloid`
- ok Transformer/ALM/ClearKey.lean (5): `dot_marked`, `cleared_lt_live`, `marked_sup'_eq_live`, `unit_gap_unstorable_abs`, `the_marker_costs_the_grid`
- ok Transformer/ALM/ClearQuery.lean (8): `dot_marked_query`, `abs_dot_le_clearBound`, `marked_sup'_eq_live_at_query`, `pos_of_margin`, `marker_vanishes_at_zero_ordinate`, `live_lt_cleared_of_added_marker`, `the_section_4b_head_is_guarded`, `the_top_of_the_range_is_not_guarded`
- issue Transformer/ALM/CrossFilter.lean (5): `sign_of_error_lt`, `cross_filter_sound`, `crossProdTerms_sum`, `cross_sign_eq`, `cross_paths_agree` — example states the conclusion, not the hypotheses
- ok Transformer/ALM/CumSum.lean (3): `recip_isBinary_iff`, `round_trip_drift`, `affine_drift`
- ok Transformer/ALM/Dense.lean (4): `unpack_packed`, `packed_lt_buffer`, `rowFold_congr`, `apply_eq_rowMajor`
- ok Transformer/ALM/DensePad.lean (8): `blockCount_eq_ceilDiv`, `writeAt_lt_rows`, `dropped_ge_rows`, `writeAt_chunkOf`, `chunkOf_writeAt`, `rows_le_blockCount_mul`, `chunkOf_mem`, `blocks_of_buffer`
- issue Transformer/ALM/DotError.lean (4): `abs_add_le_dotTerms`, `dot_error_le`, `cmp_of_dot_guard`, `dotTerms_markKey_self` — example witnesses only part of the hypotheses (minor)
- ok Transformer/ALM/DriftMargin.lean (6): `qScore_intCast`, `qScore_drift`, `order_survives_drift`, `winner_survives_drift`, `lookup_survives_drift`, `symmetric_tie_broken_by_drift`
- ok Transformer/ALM/Duality.lean (12): `dot_eq_mul_lineEval`, `dot_le_dot_iff_of_pos`, `dot_le_dot_iff_of_neg`, `dot_of_snd_eq_zero`, `dot_le_dot_iff_of_snd_eq_zero`, `isGreatest_dot_iff_of_pos`, `isGreatest_dot_iff_of_neg`, `dot_lift`, `dot_lift_int`, `liftQuery_snd`, `liftQuery_snd_pos`, `lookup_reduces_to_upper_envelope`
- ok Transformer/ALM/Envelope.lean (5): `lineEval_interX`, `lineEval_le_of_slope_eq`, `dominated_iff_le_at_interX`, `interX_le_interX_iff`, `sup'_erase_of_le`
- issue Transformer/ALM/ExactDot.lean (4): `crossTerms_sum`, `crossTerms_sum_lineEval`, `dot_cmp_eq`, `dot_cmp_tie` — example is `dot x y = dot x y`, witnesses nothing
- ok Transformer/ALM/Expansion.lean (4): `grow_sum`, `sum_filter_ne_zero`, `push_sum`, `expansion_sum`
- issue Transformer/ALM/ExpansionSign.lean (3): `expansion_nil_sum`, `sign_of_top`, `expansion_sign_eq` — example is `[].sum = 0`, witnesses nothing
- ok Transformer/ALM/FloatGrid.lean (5): `fp_eval_exact_of_grid`, `fp_exact_of_grid`, `roundScore_grid`, `fp_walk_collects_of_grid`, `fp_walk_trichotomy_of_grid`
- ok Transformer/ALM/FloatHead.lean (6): `hullIndex_ans_one`, `embInt_one_apply`, `fpProbe_key_eq`, `fp_head_output`, `half_le_key_dist_mid`, `fp_head_output_of_int`
- issue Transformer/ALM/FloatHeadTie.lean (2): `fp_head_tie_resolves`, `fp_head_tie_resolves_latest` — example states the conclusion, not the hypotheses
- issue Transformer/ALM/FloatHull.lean (6): `cmp_of_sep`, `isect_liftKey_error`, `fp_no_spurious_erase`, `fp_query_branch`, `lineEval_eq_at_midpoint`, `fp_longDouble` — example states the conclusion, not the hypotheses
- issue Transformer/ALM/FloatIndex.lean (7): `fpSearch_eq`, `fpSearch_eq_of_bounded`, `fpProbe_le`, `fpProbe_eq_hullProbe`, `fp_hullIndex_key`, `fp_hullIndex_isGreatest`, `fp_hullIndex_longDouble` — example states the conclusion, not the hypotheses
- ok Transformer/ALM/FloatLattice.lean (3): `fpSearch_isGreatest_of_int`, `sortedKey_int`, `fpProbe_mem_argmaxSet_of_int`
- ok Transformer/ALM/FloatResolve.lean (3): `fp_query_resolve`, `fp_query_resolveLatest`, `fp_query_cost_total`
- ok Transformer/ALM/FloatTie.lean (5): `dot_liftQuery`, `fp_tie_sound`, `lineEval_liftKey_int`, `fp_tie_no_false_positive`, `fp_tie_no_false_negative`
- ok Transformer/ALM/FloatWalk.lean (5): `fpProbe_mem_fpTieSet`, `fpTieSet_nonempty`, `fp_walk_sound`, `fp_walk_collects`, `fp_walk_trichotomy`
- ok Transformer/ALM/GateGrid.lean (7): `gate_eq_ite`, `relu_isBinary`, `abs_gate_le`, `isBinary_gate`, `isBinary_add`, `the_ffn_wall`, `isBinary_gate_of_wall`
- ok Transformer/ALM/GeneralPosition.lean (3): `lt_max_of_lt_at_interX`, `erase_preserves_tieSet`, `liftKey_not_concurrent`
- ok Transformer/ALM/GridWitness.lean (14): `gridRatio_nonneg`, `offGrid_iff_one_lt_gridRatio`, `foldr_max_nonneg`, `foldr_max_append`, `le_foldr_max`, `foldr_max_le`, `runCount_append`, `runWorst_append`, `clean_merge`, `clean_iff_forall`, `clean_iff_worst_le_one`, `strict_guard_of_worst_lt_one`, `binade_of_worst_le_half`, `run_retrieval_of_worst_lt_one`
- ok Transformer/ALM/GuardSep.lean (8): `ulpOf_succ_exp`, `ulpOf_lt_one_iff`, `isExp_two_mul`, `clean_strict_at_every_scale`, `scaled_unit_sep`, `cmp_of_guard`, `retrieval_survives_the_guard`, `guard_le_no_inversion`
- ok Transformer/ALM/Hull.lean (9): `score_eq_lineEval`, `sScore_eq_lineEval`, `interX_liftKey`, `liftKey_not_dominated`, `le_of_step_lt`, `hslope_liftKey`, `hbp_liftKey`, `hull_isGreatest`, `hull_isGreatest_score`
- ok Transformer/ALM/HullBranch.lean (7): `planar_bsearch_of_pos`, `planar_bsearch_of_neg`, `dot_le_dot_iff_of_snd_eq_zero_neg`, `planar_argmax_of_snd_eq_zero`, `planar_argmax_of_snd_eq_zero_neg`, `planar_argmax_unique_of_snd_eq_zero`, `planar_argmax_unique_of_snd_eq_zero_neg`
- issue Transformer/ALM/HullBuild.lean (7): `foldl_stepState_count`, `pops_add_size`, `pops_le_length`, `buildCost_le`, `hullIndex_build_paid`, `hullAns_eq_bfAns`, `hullIndex_agrees_with_bruteForce` — minor: last example is a conclusion instance
- ok Transformer/ALM/HullCache.lean (5): `isChain_append_singleton_congr`, `breakTo_none`, `cacheOk_splice`, `cacheOk_splice_dropped`, `the_two_line_cache`
- ok Transformer/ALM/HullClear.lean (7): `clearKey_fst`, `clearKey_zero`, `lineEval_clearKey_le`, `sq_sub_le_of_le`, `eraseStep_of_clearKey`, `not_marked_of_clearKey`, `clear_bracket_of_shipped`
- issue Transformer/ALM/HullCost.lean (4): `hullProbe_mem_argmaxSet`, `hullQuery_cost_total`, `argmaxSet_trichotomy`, `hullQuery_collects` — minor: last example is a conclusion instance
- issue Transformer/ALM/HullCover.lean (5): `foldl_stepState_pops_of_no_erase`, `runState_of_no_erase`, `buildCost_of_no_erase`, `hull_covers_every_key`, `build_isGreatest_score` — build_isGreatest_score example states the conclusion
- issue Transformer/ALM/HullErase.lean (6): `sup'_erase_of_slope_eq`, `exists_ge_of_interX_le`, `sup'_erase_of_interX_le`, `erase_preserves_isGreatest`, `erase_slope_eq_preserves_isGreatest`, `erase_interX_le_preserves_isGreatest` — examples for erase_preserves_isGreatest, erase_slope_eq/interX_le state conclusions
- issue Transformer/ALM/HullHead.lean (3): `score_eq_of_mem_argmaxSet`, `argmaxTie_head_resolves`, `hullTie_head_resolves` — example states the conclusion inequality at Fin 2; hb, hc, hne, hmass not witnessed
- issue Transformer/ALM/HullHeadLatest.lean (1): `argmaxTie_head_resolves_latest` — same as HullHead: conclusion instance, hypotheses not witnessed
- issue Transformer/ALM/HullIndex.lean (8): `hullProbe_le`, `hullProbe_isGreatest`, `hullIdx_spec`, `hullIdx_isGreatest`, `hullProbe_cost`, `hullIdx_isGreatest_score`, `hullAns_isGreatest`, `hullIndex_query_paid` — minor: example is a conclusion instance (only hypothesis is Nonempty)
- issue Transformer/ALM/HullLift.lean (7): `liftKey_injective`, `lineEval_liftKey_self`, `lineEval_liftKey_lt_of_ne`, `not_eraseStep_of_lift`, `erasesTo_eq_of_lift`, `build_eq_of_lift`, `build_card_eq_keyCard` — minor: build_card example is a conclusion instance, ReflTransGen not witnessed
- ok Transformer/ALM/HullLines.lean (7): `bsearch_lines_isGreatest`, `lineEval_neg`, `interX_neg`, `dot_neg`, `parabLine_interX`, `parabLine_slope`, `parabLine_bp`
- ok Transformer/ALM/HullLower.lean (5): `lineEval_neg_liftKey`, `neg_liftKey_dominated`, `eraseStep_neg_liftKey`, `lower_envelope_eq_extremes`, `lower_two_lines_suffice`
- issue Transformer/ALM/HullMark.lean (10): `markKey_zero`, `markKey_fst`, `lineEval_markKey_self`, `lineEval_markKey_lt`, `Marked.subset`, `not_eraseStep_of_marked`, `erasesTo_eq_of_marked`, `build_eq_of_marked`, `build_card_eq_of_marked`, `marked_sep_of_shipped` — minor: build_card example is a conclusion instance, ReflTransGen not witnessed
- issue Transformer/ALM/HullMono.lean (4): `lineEval_le_of_interX_le`, `interX_lt_interX_iff`, `interX_lt_interX_widen_left`, `cached_lt_iff` — first example's conjunct does not match hB of widen_left (interX A D for interX B D)
- issue Transformer/ALM/HullNear.lean (9): `lineEval_markKey_eq`, `lineEval_markKey_lt_of_gap`, `sq_dist_gap_of_near_int`, `lineEval_markKey_lt_of_near`, `abs_lt_abs_of_near`, `Marked.offset_mem`, `isGreatest_of_near`, `the_shipped_window`, `the_window_is_needed` — isGreatest_of_near example states the conclusion
- issue Transformer/ALM/HullPrune.lean (4): `eraseStep_of_slope_eq`, `eraseStep_of_interX_le`, `erases_preserves_isGreatest`, `build_isGreatest_of_inserted` — examples state conclusions; ErasesTo/ReflTransGen hypothesis not witnessed
- issue Transformer/ALM/HullResolve.lean (8): `merge_empty_left`, `scanCombined_count`, `scanBest_eq`, `scanCombined_resolveAverage`, `scanCombined_resolveLatest`, `scanCombined_comm`, `scanCombined_resolveLatest_of_ne`, `tie_resolve` — minor: examples are conclusion instances (hypotheses trivially satisfiable)
- ok Transformer/ALM/HullScan.lean (12): `lineEval_liftKey`, `tie_iff_midpoint`, `not_tie_three`, `tie_adjacent`, `mem_argmaxSet`, `argmaxSet_tie`, `argmaxSet_card_le_two`, `argmaxSet_adjacent`, `scan_left_step`, `scan_right_step`, `scan_merge_count_le_one`, `argmaxSet_eq_pair`
- ok Transformer/ALM/HullSep.lean (5): `sq_dist_gap_of_sep`, `lineEval_markKey_lt_of_sep`, `nearest_fails_of_close`, `nearest_fails_of_close'`, `the_shipped_separation_floor`
- ok Transformer/ALM/HullSpace.lean (5): `lineEval_liftKey_le_sq`, `envelope_le_sq`, `envelope_eq_sq_of_mem`, `lineEval_midpoint_gt`, `card_le_of_envelope_eq`
- issue Transformer/ALM/HullSplice.lean (7): `breakOrd_pre`, `breakOrd_post`, `breakOrd_append_cons_cons`, `breakOrd_append_cons`, `breakOrd_splice`, `breakOrd_splice_dropped`, `the_two_line_order` — minor: examples show BreakOrd results, not hback/hkeep/hhi
- issue Transformer/ALM/HullTwin.lean (5): `lineEval_markKey_sub_twin`, `twin_later_wins`, `twin_not_greatest`, `the_measured_twins_are_invisible`, `the_longest_run_is_covered_too` — twin_not_greatest example states the conclusion
- ok Transformer/ALM/HullValue.lean (4): `argmaxSet_resolve`, `argmaxSet_resolveLatest`, `hullQuery_resolve`, `hullQuery_resolveLatest`
- issue Transformer/ALM/HullWall.lean (5): `markKey_eq_liftKey_of_wall`, `marks_tie_past_the_wall`, `lifted_of_marked_of_wall`, `the_shipped_spread_is_under_two`, `the_wasm_heads_are_past_the_wall` — marks_tie_past_the_wall example is a conclusion instance; lifted_of_marked_of_wall example witnesses only Marked 0, not hrep/hsq/hwall
- ok Transformer/ALM/IntGrid.lean (6): `step_add_one_le`, `key_add_one_le`, `mid_add_one_le`, `mid_half_int`, `half_int_le_of_lt_add_half`, `le_half_int_of_lt_add_half`
- ok Transformer/ALM/KeyOrder.lean (10): `mem_keySet`, `keyCard_pos`, `keyCard_le`, `sortedKey_of_le`, `sortedKey_of_ge`, `sortedKey_lt_succ`, `exists_sortedKey_eq`, `exists_eq_sortedKey`, `keyCard_eq_of_injective`, `exists_bound_sortedKey`
- ok Transformer/ALM/LatestClose.lean (2): `invLogPos_step_le`, `past_the_window_rounding_decides`
- ok Transformer/ALM/LatestWindow.lean (10): `log_shift_pos`, `invLogPos_zero`, `invLogPos_lt_invLogPos`, `invLogPos_nonneg`, `invLogPos_lt`, `writeScore_eq`, `latest_wins`, `one_le_sScore_sub`, `distinct_keys_keep_their_order`, `released_alpha_below_half`
- issue Transformer/ALM/Lattice.lean (4): `geom_pos_le`, `geom_sq_le`, `sum_exp_gap_le`, `softmax_winner_lengthfree` — no satisfiability example for sum_exp_gap_le / softmax_winner_lengthfree
- ok Transformer/ALM/LiftCompare.lean (12): `the_abscissa_and_its_half_are_both_storable`, `the_section_4b_key`, `lt_of_int_lt`, `lt_iff`, `dot_markKey_upper`, `dot_markKey_lower`, `upper_lt_iff`, `lower_lt_iff`, `sq_dist_le`, `sq_dist_fits`, `the_shipped_spread`, `the_section_4b_query`
- ok Transformer/ALM/LiftResidual.lean (10): `upper_sub`, `lower_sub`, `int_lt_of_drift`, `upper_lt_of_sq_lt`, `upper_lt_iff_of_sq_eq`, `upper_near_lt_iff`, `lower_near_lt_iff`, `the_shipped_window`, `above`, `the_section_4a_query`
- ok Transformer/ALM/LookupIndex.lean (2): `reduce_iff`, `bfAns_isGreatest`
- issue Transformer/ALM/MarkedPosition.lean (3): `markKey_not_concurrent`, `marked_not_concurrent`, `marked_erase_preserves_tieSet` — minor: no example for marked_not_concurrent, marked_erase_preserves_tieSet
- ok Transformer/ALM/OrthVectors.lean (11): `bit_nonneg`, `bit_sq`, `one_sub_bit_sq`, `ip_nonneg`, `orth_iff`, `inner_kvec_qvec`, `norm_sq_kvec`, `score_qvec_kvec`, `score_qvec_le`, `score_qvec_eq_iff_orth`, `argmax_decides_ov`
- issue Transformer/ALM/PlanarHead.lean (2): `planarAns_isGreatest`, `planar_head_argmax` — example takes N := -parabLine, which violates hslopeN/hbpN (docstring admits it); N-invariants not witnessed
- ok Transformer/ALM/Query.lean (6): `lineEval_sub`, `lineEval_le_iff_interX_le`, `lineEval_le_iff_le_interX`, `lineEval_mono_right`, `lineEval_anti_right`, `lowerBound_isGreatest`
- issue Transformer/ALM/QueryScale.lean (13): `scaleQuery_one`, `dot_scaleQuery_int`, `dot_smul_query`, `order_scale_invariant`, `maximizers_scale_invariant`, `gridScale_nonneg`, `gridScale_eq_zero_iff`, `gridScale_scaleQuery`, `onTheGrid_scaleQuery`, `onTheGrid_preserves_order`, `onTheGrid_score_isInt`, `dot_liftQuery_eq_qScore`, `rounded_normalization_keeps_the_winner` — rounded_normalization example is incoherent: hr at ε=1, hmargin at ε=1e-9, hwin absent
- ok Transformer/ALM/SAHead.lean (7): `must`, `inner_packBlocks`, `fstBlockLin_packBlocks`, `sndBlockLin_packBlocks`, `inner_liftQueryVec_liftKeyVec`, `inner_queryProj_keyProj`, `SAOutput_eq_softmax_head`
- issue Transformer/ALM/SAHeadValue.lean (2): `SAOutput_close_of_mass`, `sa_head_output_at_index` — minor: SAOutput_close_of_mass example witnesses only hw at ε=1, not hC
- issue Transformer/ALM/ScalarInt.lean (1): `softmax_winner_scalar_int` — minor: no satisfiability example (ScalarSharp's covers hinj)
- ok Transformer/ALM/ScalarSharp.lean (4): `the`, `sum_exp_sq_gap_le`, `softmax_winner_scalar_int_sharp`, `scalar_int_sharp_lt`
- ok Transformer/ALM/ScoreGap.lean (7): `keyGap_scale_free`, `keyGap_self`, `one_le_keyGap`, `one_le_keyGap_iff`, `retrieval_of_keyGap`, `rounding_decides_below`, `keyGap_writeScore`
- ok Transformer/ALM/ScoreGuard.lean (6): `ulpOf_pos`, `isExp_unique`, `isExp_mul`, `clean_at_every_scale`, `dirty_at_every_scale`, `guard_iff_below_wall`
- ok Transformer/ALM/ScoreWall.lean (11): `isBinary_intCast`, `isBinary_even`, `not_isBinary_odd`, `two_le_dist_of_isBinary`, `unit_gap_unstorable`, `key_coord_unstorable`, `sScore_eq_iScore`, `minimal_margin_unstorable`, `the_wall`, `the_first_unstorable_key`, `the_missing_unit`
- issue Transformer/ALM/Softmax.lean (3): `softmax_winner_ge`, `softmax_winner_sharp`, `softmax_winner_ge'` — no satisfiability examples
- ok Transformer/ALM/SoftmaxIndex.lean (8): `NNIndex.ans_eq_of_query_mem`, `hullIndex_ans_eq`, `score_gap_one_of_int`, `softmax_at_index_ge`, `softmax_at_hullIndex_int`, `score_embInt_int`, `one_le_score_gap_int`, `softmax_at_index_ge_of_untied`
- ok Transformer/ALM/SoftmaxLatest.lean (1): `softmax_head_resolves_latest`
- issue Transformer/ALM/SoftmaxLatestMass.lean (2): `softmax_head_resolves_latest_of_gap`, `softmax_head_resolves_latest_of_int` — example states the conclusion, not the hypotheses
- issue Transformer/ALM/SoftmaxMass.lean (3): `softmax_tie_mass`, `softmax_tie_mass_ge`, `softmax_head_resolves_average_of_gap` — all three examples state conclusions; htail/hgap not witnessed as such
- ok Transformer/ALM/SoftmaxTie.lean (4): `dist_weighted_sum_le_of_level`, `softmax_weight_eq_of_score_eq`, `softmax_output_close_level`, `softmax_head_resolves_average`
- issue Transformer/ALM/SoftmaxTieInt.lean (3): `sScore_tie_gap_one`, `softmax_tie_mass_int`, `softmax_head_resolves_average_of_int` — example states the conclusion, not the hypotheses
- ok Transformer/ALM/SoftmaxValue.lean (6): `dist_weighted_sum_le`, `softmax_weight_nonneg`, `softmax_weight_sum`, `softmax_output_close`, `head_output_at_index`, `head_output_at_index_untied`
- issue Transformer/ALM/SparseHead.lean (6): `sparseFold_eq_rowFold`, `firstMax_lt`, `firstMax_le`, `firstMax_first`, `firstMax_congr`, `firstMax_sparse_eq` — examples are conclusion instances (sparseFold = rowFold, firstMax values)
- ok Transformer/ALM/SparseSoftmax.lean (7): `sparseWeight_nonneg`, `sparseWeight_sum`, `dist_weighted_sum_sub_le`, `sparse_total_variation`, `sparse_softmax_output_close`, `softmax_mass_outside_le`, `sparse_softmax_output_close_of_gap`
- issue Transformer/ALM/Theta.lean (4): `sqNorm_zero`, `sqNorm_cast`, `theta1_le`, `box_sum_le` — minor: example is a conclusion instance at m = 0
- ok Transformer/ALM/TieBreak.lean (5): `sScore_eq_iff`, `sScore_lt_of_ne`, `argmax_unique_of_query_mem`, `resolve_eq_of_single`, `merge_comm`
- ok Transformer/ALM/TieHyperplane.lean (6): `score_eq_iff_norm_eq`, `score_eq_iff_inner_eq_zero`, `score_midpoint_eq`, `exists_score_ne`, `tie_eq_of_query_mem`, `score_eq_iff_int_one`
- ok Transformer/ALM/TieMeasure.lean (6): `tieLocus_eq_bisector`, `tieLocus_ne_top`, `volume_tieLocus`, `volume_tie_locus_family`, `ae_no_tie`, `ae_argmax_unique`
- ok Transformer/ALM/TieSet.lean (6): `mem_tieSet`, `tieSet_subset`, `eq_interX_of_lineEval_eq`, `concurrent_iff_collinear`, `interX_eq_of_concurrent`, `erase_keeps_the_value_and_drops_the_winner`
- ok Transformer/ALM/TreeBalance.lean (8): `toList_rotateLeft`, `toList_rotateRight`, `toList_paint`, `toList_run`, `toList_of_stepAt`, `toList_of_rebalances`, `lowerBound_eq_find_of_rebalances`, `the_rotation_moves_the_depth`
- issue Transformer/ALM/TreeQuery.lean (6): `lowerBound_nil`, `lbCount_nil`, `lowerBound_eq_find`, `lbCount_le_depth`, `lbCount_le_two_log`, `log_succ_bound` — minor: examples are conclusion instances; hmono and WF not witnessed
- ok Transformer/ALM/VectorInt.lean (6): `embInt_apply`, `score_gap_int`, `sum_exp_lattice_le`, `softmax_winner_int_sharp`, `score_embInt_one`, `softmax_winner_int_sharp_one`

### Transformer.AMSGrad — arXiv:1904.03590

- ok Transformer/AMSGrad/Section1_AMSGrad.lean (1): `isWeightedProj_boxProj`
- ok Transformer/AMSGrad/Section1_TheoremA.lean (1): `isOnlineConvex_zero`
- ok Transformer/AMSGrad/Section2_Prelim.lean (9): `fderiv_apply_eq_sum`, `convex_first_order`, `cauchy_schwarz`, `taylor_geom`, `not_taylor`, `taylor_deriv`, `harmonic_le`, `sum_inv_sqrt_le`, `sum_div_sum_le`
- ok Transformer/AMSGrad/Section2_Proj.lean (4): `posDef_comm`, `posDef_nonneg`, `proj_variational`, `mcm_str`
- ok Transformer/AMSGrad/Section3_Example.lean (9): `grad_linear`, `isOnlineConvex_exa`, `exa_state_one`, `exaY₁_mem`, `exa_x_two`, `exa_state_two`, `exa_x_two_lt`, `exa_sign_one`, `exa_sign_two`
- ok Transformer/AMSGrad/Section3_Issue.lean (4): `le_amsgradRule`, `prepare_lem`, `abel_eq`, `abel_le`
- ok Transformer/AMSGrad/Section3_Optimal.lean (5): `exaCoef_sum_le_101`, `exaCoef_sum_add_101`, `exaCoef_sum_pos`, `exa_optimal`, `not_red_ineq`
- ok Transformer/AMSGrad/Section3_Step.lean (6): `v_nonneg`, `m_eq_zero`, `m_eq_zero_of_vhat`, `proj_le`, `young`, `step_ineq`
- ok Transformer/AMSGrad/Section4_Corollary.lean (2): `cor_lambda`, `cor_inv`
- ok Transformer/AMSGrad/Section4_Counter.lean (2): `isOnlineConvex_sign`, `sign_hyp`
- ok Transformer/AMSGrad/Section4_CounterRegret.lean (5): `sum_blocks`, `sign_block_sum`, `sign_block_sign`, `sum_half_pow_le`, `not_cor_lower`
- ok Transformer/AMSGrad/Section4_CounterRun.lean (8): `blockSign_sq`, `sign_vhat`, `sign_step`, `clamp_sign`, `blockSign_block`, `sign_x_mem`, `sign_block`, `sign_term`
- ok Transformer/AMSGrad/Section4_Lemmas.lean (11): `state_x_mem`, `x_mem`, `v_le`, `vhat_succ`, `vhat_le_succ`, `vhat_nonneg`, `vt`, `sqrt_div_le`, `t_0_of_key`, `t_0_lambda`, `t_0_inv`
- ok Transformer/AMSGrad/Section4_MainLemma.lean (5): `geomSum_nonneg`, `sum_geomSum_div_sqrt_le`, `moment_le`, `term_le`, `mainlem`
- ok Transformer/AMSGrad/Section4_Rate.lean (3): `gnorm_le`, `rate_le`, `tendsto_rate`
- ok Transformer/AMSGrad/Section4_Telescope.lean (4): `telescope_le`, `eqmain_le_of`, `vt_div`, `eqmain_le`
- ok Transformer/AMSGrad/Section4_Terms.lean (2): `sum_sqrt_mul_pow_le`, `eqsecond_le`
- ok Transformer/AMSGrad/Section4_Theorem.lean (2): `mainthm_lambda`, `mainthm_inv`
- ok Transformer/AMSGrad/Section4_Third.lean (2): `eqthird_lambda_le`, `eqthird_inv_le`
- ok Transformer/AMSGrad/Section5_AdamX.lean (6): `le_adamXRule`, `adamX_vhat_one`, `adamX_vhat_succ`, `vtnew`, `vt2`, `adamX_eq_amsgrad`
- ok Transformer/AMSGrad/Section5_Bounds.lean (3): `vtnew_div`, `adamX_mono`, `eqthird_adamX_le`
- ok Transformer/AMSGrad/Section5_Corollary.lean (3): `ge_cor`, `bound_lambda`, `bound_inv`
- ok Transformer/AMSGrad/Section5_Sums.lean (2): `sum_lambda_le`, `sum_inv_le`
- issue Transformer/AMSGrad/Section5_Theorem.lean (1): `mainthm2` — mainthm2 is weaker than the paper's Theorem 5.1: second term has (1-β₁)² where the paper has (1-β₁); the paper's constant is neither proved nor refuted. Under non-increasing β_{1,t}, Lemma 5.2 gives √v̂_t ≤ G and the paper's constant follows; in general it is open — either prove it under that hypothesis or refute it

### Transformer.AdamBeyond — arXiv:1904.09237

- ok Transformer/AdamBeyond/AppendixG_Auxiliary.lean (7): `of`, `psd_comm`, `psd_nonneg`, `proj_variational_psd`, `proj_lemma`, `sum_div_sqrt_partial_le`, `proj_1d`
- ok Transformer/AdamBeyond/Section2_Adam.lean (8): `v_sum`, `m_sum`, `adagrad_vhat`, `gamma_succ`, `gamma_nonneg_of`, `gamma_sgd_nonneg`, `gamma_adagrad_nonneg`, `gamma_amsgrad_nonneg`
- ok Transformer/AdamBeyond/Section3_Counter.lean (7): `cex_term_ge`, `cex_regret`, `half_div_sqrt_hyp`, `counter_example_epsilon`, `counter_example`, `counter_example_const`, `gamma_adam_neg`
- ok Transformer/AdamBeyond/Section3_GenBlock.lean (5): `genX_succ'`, `genSlope_block`, `gen_inblock`, `genM_block_anti`, `genD_ge`
- ok Transformer/AdamBeyond/Section3_GenRegret.lean (5): `gen_start`, `gen_block_regret`, `gen_regret_blocks`, `gen_regret`, `gen_constants`
- ok Transformer/AdamBeyond/Section3_GenRun.lean (13): `abs_genSlope_le`, `isOnlineConvex_gen`, `genM_succ`, `genV_succ`, `genX_succ`, `genX_zero`, `genM_zero`, `genV_zero`, `genSlope_mem`, `gen_warmup`, `genX_le_one`, `neg_one_le_genX`, `gen_bounds`
- ok Transformer/AdamBeyond/Section3_GenStep.lean (4): `proj_1d_of_mono`, `geom_Icc`, `gen_block_sum`, `gen_block`
- ok Transformer/AdamBeyond/Section3_General.lean (1): `counter_example_gen`
- ok Transformer/AdamBeyond/Section3_Run.lean (6): `adamEpsRule_zero`, `isOnlineConvex_cex`, `cex_v`, `cex_x`, `block_arith`, `cex_block`
- ok Transformer/AdamBeyond/Section3_Stoch.lean (8): `exists_isBernoulliSeq`, `stoch_g`, `stoch_m_succ`, `stoch_v_succ`, `stochX_succ`, `stochX_one`, `stoch_m_eq`, `stoch_v_eq`
- ok Transformer/AdamBeyond/Section3_StochBound.lean (12): `coinGrad_le`, `coinGrad_le_ind`, `coinGrad_sq_le`, `sum_pow_sub_le`, `inv_sqrt_ge`, `stoch_T1_le`, `stoch_m_le`, `stoch_v_nonneg`, `stoch_v_ge`, `stoch_v_le_ind`, `stoch_m_le_ind`, `stoch_T2_le`
- ok Transformer/AdamBeyond/Section3_StochCoins.lean (11): `coin_factor`, `dependsOn_sum`, `dependsOn_comp₂`, `dependsOn_coin_mul`, `coin_measurable`, `coin_integrable`, `coin_indep`, `coin_integral_ind`, `coin_integral_ind_mul`, `coin_integral_not_mul`, `coin_integral_sum`
- ok Transformer/AdamBeyond/Section3_StochMean.lean (7): `coinM_dep`, `coinV_dep`, `coinMV_dep`, `stoch_int_T1`, `stoch_int_T2`, `stoch_int_v`, `stoch_int_inv`
- ok Transformer/AdamBeyond/Section3_StochStep.lean (3): `coin_step_split`, `stoch_const_le`, `stoch_step`
- ok Transformer/AdamBeyond/Section3_Stochastic.lean (1): `counter_example_stochastic`
- ok Transformer/AdamBeyond/Section4_AMSGrad.lean (2): `amsgrad_moment_sum`, `amsgrad_moment_sum_sqrt`
- ok Transformer/AdamBeyond/Section4_Abel.lean (2): `sum_Icc_two_sub`, `abel_beta_le`
- issue Transformer/AdamBeyond/Section4_Corollary.lean (3): `sum_lambda_sqrt_le`, `amsgrad_regret_lambda`, `amsgrad_regret_inv` — amsgrad_regret_lambda restores a factor d/α in Corollary 1's second term; the printed β₁D²G/((1-β₁)²(1-λ)²) does not follow from Theorem 4 but is not refuted either: a statement weaker than printed under the paper's name
- ok Transformer/AdamBeyond/Section4_Regret.lean (2): `amsgrad_regret_moment`, `amsgrad_regret`
- ok Transformer/AdamBeyond/Section5_AdamNC.lean (4): `adamNC_vhat_succ`, `adamNC_vhat_sum`, `adamNC_vhat_inv`, `adamNC_inv_cond`
- issue Transformer/AdamBeyond/Section5_Corollary.lean (4): `sqrt_mul_sqrt_vhat_inv`, `sqrt_vhat_inv_le`, `adamNC_regret_lambda`, `adamNC_regret_inv` — adamNC_regret_lambda restores d/α in Corollary 2's second term, as for Corollary 1; the printed constant is neither proved nor refuted
- ok Transformer/AdamBeyond/Section5_Lemma.lean (3): `sum_geomSum_le`, `m_sq_le_gnorm`, `adamNC_moment_sum`
- issue Transformer/AdamBeyond/Section5_Regret.lean (1): `adamNC_regret` — adamNC_regret takes condition 1 at α_t instead of the printed α_T, which is a stronger hypothesis (1/α_t ≤ 1/α_T); the printed Theorem 5 is neither proved nor refuted

### Transformer — arXiv:2106.06981

- ok Transformer/Basic.lean (4): `inner_proj_eq_zero`, `proj_smul_self`, `norm_proj_le`, `norm_proj_sub_proj_le` — projection API; norm_proj_sub_proj_le constants checked

### Transformer.CRASP — arXiv:2506.16055

- ok Transformer/CRASP/Affine.lean (3): `countP_bool_eq`, `Term.val_affine`, `Form.sat_eq_of_count_eq`
- ok Transformer/CRASP/Alternating.lean (15): `altList_succ`, `length_altList`, `altList_not_succ`, `altPlus_one`, `sublist_of_ne_cons`, `sublist_of_ne_replicate`, `altList_sublist_succ`, `altList_succ_of_both`, `altList_sublist_of_mem_altPlus`, `not_altList_succ_sublist`, `not_altList_not_sublist_of_mem_altPlus`, `cons_mem_altPlus_same`, `cons_mem_altPlus_flip`, `mem_altPlus_of_sublist`, `altPlus_eq`
- ok Transformer/CRASP/Basic.lean (18): `val_ofPos`, `depth_ofPos`, `past_ofPos`, `val_nsmul`, `depth_nsmul`, `past_nsmul`, `sat_or`, `depth_or`, `past_or`, `sat_le`, `depth_le`, `sat_isZero`, `depth_isZero`, `past_isZero`, `sat_atEnd`, `depth_atEnd`, `pnpFree_atEnd`, `dyck_mem`
- ok Transformer/CRASP/Blocks.lean (7): `altList_sublist_of_le`, `altList_not_sublist_iff`, `eq_of_mem_altPlus`, `take_mem_altPlus`, `getElem?_of_mem_altPlus`, `append_mem_altPlus`, `mem_altPlus_three`
- ok Transformer/CRASP/BoundedExists.lean (16): `countP_range'_add`, `val_countL_succ`, `val_countL_pos_iff`, `val_countR_eq_succ`, `val_countR_pos_iff`, `depth_exAt`, `depth_exBefore`, `past_exAt`, `past_exBefore`, `pnpFree_exAt`, `pnpFree_exBefore`, `depth_exAfter`, `pnpFree_exAfter`, `sat_exAt`, `sat_exBefore`, `sat_exAfter`
- ok Transformer/CRASP/Collapse.lean (2): `exists_mem_TLCl_of_clustered`, `not_recognizes_altPlus_of_clustered`
- ok Transformer/CRASP/Commutative.lean (4): `Term.val_countL_length_eq`, `Form.sat_length_eq_of_depth_le_one`, `Term.val_length_eq_of_depth_le_one`, `commutativeOnMiddle_of_mem_TLCP_one`
- ok Transformer/CRASP/Conjunctions.lean (10): `Form.sat_all`, `Form.depth_all_le`, `Form.past_all`, `Form.pnpFree_all`, `Form.past_any`, `Form.pnpFree_any`, `Form.sat_onStr`, `Form.depth_onStr`, `Form.past_onStr`, `Form.pnpFree_onStr`
- ok Transformer/CRASP/ConstantLayer.lean (5): `layer_of_const`, `act_add_of_const`, `layer_eq_of_fields`, `act_eq_of_fields`, `out_collapse`
- ok Transformer/CRASP/CroppingUnsound.lean (6): `Form.sat_firstNotA`, `Form.minimalOne_firstNotA`, `accommodating_one`, `two_le_of_prefixVec_mem`, `pnpsConstantOn_firstNotA`, `not_minimalOneConstantOn_firstNotA`
- ok Transformer/CRASP/Defs.lean (2): `sat_eq`, `depth_eq`
- ok Transformer/CRASP/Depth.lean (5): `cropping_oneway_unsound`, `cropping_oneway_right_unsound`, `reduction_past_unsound`, `reduction_unsound`, `definableL_altPlus`
- ok Transformer/CRASP/DepthZero.lean (5): `Form.exists_eq_pnp_of_mem_pnps`, `Term.exists_eq_pnp_of_mem_pnps`, `Term.val_eq_of_depth_eq_zero`, `Form.sat_eq_of_depth_eq_zero`, `Form.sat_eq_of_mem_pnps`
- ok Transformer/CRASP/Extensions.lean (5): `Form.depth_toX`, `Term.depth_toX`, `Form.sat_toX`, `Term.val_toX`, `Form.lang_toX`
- ok Transformer/CRASP/ExtensionsCounts.lean (10): `filter_range'_congr`, `length_filter_range'_strictL`, `length_filter_range'_strictR`, `length_filter_range'_all`, `covers_countL`, `covers_countR`, `exists_sat_neg_or`, `covers_countAll`, `covers_countLStrict`, `covers_countRStrict`
- ok Transformer/CRASP/ExtensionsElim.lean (5): `FormX.depth_elim_le`, `TermX.depth_pieces_le`, `FormX.sat_elim`, `TermX.covers_pieces`, `exists_form_of_formX`
- ok Transformer/CRASP/ExtensionsPieces.lean (14): `Form.sat_topAt`, `Form.depth_topAt`, `Form.sat_pos`, `Form.depth_pos`, `Form.sat_any`, `Form.depth_any_le`, `Term.val_addNat`, `Term.depth_addNat`, `sat_ltPieces`, `depth_ltPieces_le`, `covers_condPieces`, `depth_condPieces_le`, `covers_addPieces`, `depth_addPieces_le`
- ok Transformer/CRASP/FiniteFunction.lean (9): `depth_ite_neg`, `past_ite_neg`, `pnpFree_ite_neg`, `sat_ite_neg`, `sat_bitsFormula`, `depth_bitsFormula_le`, `past_bitsFormula`, `pnpFree_bitsFormula`, `finite_function`
- ok Transformer/CRASP/Fixed.lean (12): `ext`, `m_zero`, `val_zero`, `le_clamp`, `clamp_lt`, `clamp_eq_self`, `m_round`, `val_round`, `val_round_le`, `le_val_round`, `round_val`, `eq_of_abs_val_sub_lt`
- ok Transformer/CRASP/FixedBits.lean (4): `bit_succ`, `emod_pow_eq_of_parity`, `ext_of_bit`, `bit_zero`
- ok Transformer/CRASP/Frame.lean (8): `getElem?_frame`, `getElem?_frame_outside`, `Form.sat_eq_of_pnpFree_depth_eq_zero`, `length_eq_of_count`, `Form.constOnMiddle_of_depth_eq_zero`, `exists_middle`, `Form.mem_TLC_of_mem_countSubs`, `Term.mem_TLC_of_mem_countSubs`
- ok Transformer/CRASP/FrameAffine.lean (2): `Term.val_affine_frame`, `Form.sat_eq_of_take_count_eq`
- ok Transformer/CRASP/FrameBox.lean (5): `exists_half_mul_le`, `exists_box_lt`, `ConstOnBox.mono`, `ConstOnBox.neg`, `ConstOnBox.and`
- ok Transformer/CRASP/FrameCount.lean (6): `Term.val_countR_add_val_countL`, `Term.val_countL_add`, `Term.val_countL_middle`, `Term.val_countL_outside`, `Term.val_eq_of_constOnMiddle`, `Form.sat_eq_of_constOnMiddle`
- ok Transformer/CRASP/FrameShrink.lean (2): `Form.exists_constOnBox`, `exists_constOnBox_list`
- ok Transformer/CRASP/Indicator.lean (10): `ite_mem_TLCl`, `lt_add_one_mem_TLCl`, `ltIndR_mem_TLCl`, `ltInd_mem_TLCl`, `ltSum_mem_TLCl`, `sat_ite`, `_root_.Transformer.CRASP.Term.val_sum`, `sat_ltIndR`, `sat_ltInd`, `sat_ltSum`
- ok Transformer/CRASP/Locality.lean (4): `Form.mem_TLCl_of_mem_countSubs`, `Term.mem_TLCl_of_mem_countSubs`, `Form.sat_append`, `Term.val_append`
- ok Transformer/CRASP/LowerBound.lean (3): `exists_constOnStrip_altPlus`, `exists_models_iff_altPlus`, `not_definableL_altPlus`
- ok Transformer/CRASP/LowerBoundTwoSided.lean (5): `ConstOnBox.constOnMiddle`, `replicate_append_mem_altPlus`, `exists_constOnMiddle_altPlus`, `exists_models_iff_altPlusDouble`, `not_definable_altPlus_double`
- ok Transformer/CRASP/MajTwo.lean (7): `sat_top`, `sat_ex`, `sat_all`, `depth_ex`, `closed_closedTop`, `depth_closedTop`, `lang_closedTop`
- ok Transformer/CRASP/MajTwoCount.lean (13): `sum_Ico_boole_eq_length_filter`, `sum_Icc_le`, `sum_Icc_ge`, `sum_Icc_eq`, `_root_.Transformer.CRASP.Var.other_ne`, `_root_.Transformer.CRASP.Var.ne_other`, `sat_topv`, `mass_append`, `mass_map_neg`, `mass_replicate_topv`, `mass_replicate_lt`, `sat_majList`, `sat_cmpList`
- ok Transformer/CRASP/MajTwoDepthOne.lean (6): `swapPos_one`, `swapPos_two`, `getElem?_swapPos`, `sat_swap_of_depth_eq_zero`, `sat_eq_of_closed_depth_le_one`, `not_forall_closed_majTwo_of_definable`
- issue Transformer/CRASP/MajTwoEquiv.lean (3): `exists_majTwo_of_mem_TLC`, `exists_closed_majTwo`, `exists_closed_majTwo_of_definable` — majTwo_depth_hierarchy: the docstring says the second half is proved, but it calls the sorried definable_of_closed_majTwo
- ok Transformer/CRASP/MajTwoOfTLC.lean (7): `Form.freeIn_toMaj`, `Maj2.depth_majList_le`, `Form.depth_toMaj_le`, `Term.depth_toMajs_le`, `Term.val_eq_ones`, `Form.sat_toMaj`, `Term.mass_toMajs`
- ok Transformer/CRASP/Middle.lean (7): `countP_range'_eq_countP`, `getElem?_length_sub_one_append`, `parikh_append`, `perm_of_parikh_eq`, `parikh_eq_of_parikh_append_eq`, `prefixVec_mem_middle`, `Form.sat_middle`
- ok Transformer/CRASP/NeutralLetter.lean (4): `map_some_sublist_iff`, `PT.width_mapSome`, `PT.lang_mapSome`, `KPiecewiseTestable.preimage_reduceOption`
- ok Transformer/CRASP/Parikh.lean (1): `accommodating_trivial`
- ok Transformer/CRASP/PiecewiseTestable.lean (7): `kPiecewiseTestable_altPlus`, `PT.depth_toForm_le`, `PT.past_toForm`, `PT.pnpFree_toForm`, `PT.lang_toForm`, `definableL_of_kPiecewiseTestable`, `definable_of_kPiecewiseTestable`
- ok Transformer/CRASP/Positional.lean (2): `TLClMod_subset_TLClPos`, `TLClY_subset_TLClPos`
- ok Transformer/CRASP/PositionalDepth.lean (2): `exists_form_of_formP`, `definablePos_altPlusNeutral`
- ok Transformer/CRASP/PositionalEmbedding.lean (6): `Form.exists_formP`, `Term.exists_termP`, `DefinableL.exists_formP`, `DefinableL.definablePos`, `DefinableL.definableMod`, `DefinableL.definableY`
- issue Transformer/CRASP/PositionalHierarchy.lean (1): `alibi_window` — rtfr_pes_depth_hierarchy: the docstring says the negative half is proved, but it calls the three sorried simulations; they should be hypotheses. 0 < k is extra
- ok Transformer/CRASP/PositionalReduction.lean (2): `FormP.tr_mem_TLCl`, `TermP.tr_mem`
- ok Transformer/CRASP/PositionalReductionAtom.lean (5): `ofBool_mem_TLCl`, `sat_ofBool`, `FormP.period_pos`, `TermP.period_pos`, `FormP.atomTr_mem_TLCl`
- ok Transformer/CRASP/PositionalReductionAtomEquiv.lean (2): `getElem?_spread_sub`, `FormP.sat_atomTr`
- ok Transformer/CRASP/PositionalReductionCount.lean (3): `sum_map_ite_eq_countP`, `TermP.val_countL_blockEnd`, `TermP.val_countL_spread`
- ok Transformer/CRASP/PositionalReductionEquiv.lean (4): `FormP.sat_tr`, `TermP.val_tr`, `FormP.models_spread_iff`, `tr_paperBlockSize_unsound`
- ok Transformer/CRASP/Prediction.lean (4): `predictAltPlus_mem`, `solvesPrediction_predictAltPlus`, `not_solvesPrediction_altPlus`, `prediction_task_depth`
- ok Transformer/CRASP/ReductionUnsound.lean (2): `Form.sat_abab_eq_aabb`, `not_lang_eq_restrict_startAB`
- ok Transformer/CRASP/Shrink.lean (3): `exists_strip_lt`, `Form.exists_constOnStrip`, `exists_constOnStrip_list`
- ok Transformer/CRASP/Spread.lean (8): `spread_nil`, `reduceOption_spread`, `length_spread`, `getElem?_flatMap_block`, `getElem?_spread`, `FormP.sat_append`, `TermP.val_append`, `FormP.sat_spread_of_le`
- ok Transformer/CRASP/Strip.lean (6): `getElem?_append_length_add`, `ConstOnStrip.mono`, `ConstOnStrip.append`, `ConstOnStrip.neg`, `ConstOnStrip.and`, `Form.constOnStrip_of_depth_eq_zero`
- ok Transformer/CRASP/Subsequence.lean (12): `sublist_snoc_snoc`, `sublist_snoc_take`, `depth_subseqStrict`, `depth_subseqAt`, `past_subseqStrict`, `past_subseqAt`, `pnpFree_subseqStrict`, `pnpFree_subseqAt`, `sat_subseqStrict`, `sat_subseqAt`, `lang_subseqAt`, `nonstrict_subseq_formula_unsound`
- ok Transformer/CRASP/SubsequenceTwoSided.lean (9): `cons_sublist_drop`, `append_cons_sublist_iff`, `depth_subseqAfter`, `pnpFree_subseqAfter`, `depth_subseqTwoSided_le`, `pnpFree_subseqTwoSided`, `sat_subseqAfter`, `lang_subseqTwoSided`, `nonstrict_twoSided_formula_unsound`
- ok Transformer/CRASP/TLCDepth.lean (4): `cropping_unsound`, `altPlusDouble_succ`, `kPiecewiseTestable_altPlusDouble`, `definable_altPlusDouble`
- issue Transformer/CRASP/Transformers.lean (4): `self_mem_masked`, `length_bos`, `definableL_iff_recognizes`, `rtfr_depth_hierarchy` — rtfr_depth_hierarchy adds 0 < k, which the paper does not have; k = 0 holds too (a+ vs depth 0)
- ok Transformer/CRASP/YNormalForm.lean (7): `FormP.prevN_succ'`, `FormP.depth_prevN`, `YAtomic.prevN`, `FormP.depth_pushY`, `TermP.depth_pushY`, `FormP.yNormal_pushY`, `TermP.yNormalT_pushY`
- ok Transformer/CRASP/YNormalFormEquiv.lean (7): `FormP.sat_prevN`, `FormP.sat_guard`, `length_filter_range'_delay`, `FormP.sat_pushY`, `TermP.val_pushY`, `exists_yNormal`, `pushY_unguarded_unsound`

### Transformer.Causal — arXiv:2411.04990

- ok Transformer/Causal/Interaction.lean (7): `hasDerivAt_h_pot`, `h_pot_periodic`, `g_pot_periodic`, `h_pot_odd`, `g_pot_even`, `h_pot_nonneg`, `g_pot_nonpos`
- ok Transformer/Causal/InteractionBounds.lean (6): `cos_le_quartic`, `sin_sq_lt_sq`, `exp_factor_bounds`, `h_pot_bounds`, `g_pot_lower_bound_near`, `g_pot_lower_bound_gauss`
- ok Transformer/Causal/InteractionNumerics.lean (5): `exp_neg_le_four_div_sq`, `exp_neg_27_32_le_half`, `lin_lower_of_small`, `const_lower_of_small`, `quartic_exponent_le`
- ok Transformer/Causal/InteractionPeak.lean (1): `h_pot_unimodal`
- ok Transformer/Causal/InteractionRemark.lean (2): `g_pot_nonpos_core`, `interaction_window` — interaction_window now concludes the minus-sign form that fixed_centers uses; the printed N g form was automatic
- ok Transformer/Causal/InteractionWindow.lean (2): `interaction_inequalities_core`, `interaction_inequalities`
- issue [FIXED: UniformTuple σ; thm1 needs d ≥ 2 (false on 𝕊^0); thm1.5 given the paper's hypothesis on V; thm2 P_{L⊥} typo recorded] Transformer/Causal/MainTheorem.lean (2): `inner_mul_le_inner_of_abs_le`, `inner_mul_eq_inner_iff` — single_cluster, two_cluster, subspace_cluster: the paper's volume measure is a free parameter σ; with σ = Dirac at the equilibrium x₂ = -x₁, single_cluster is false
- issue [FIXED: replaced by Causal.FixedCenters.fixed_centers — frozen θ_j, a_j, h/g conditions with the proof's sign, 0 < m, a.e. under abs.-cont. μ₀] Transformer/Causal/Metastability.lean (2): `csa_const_one`, `renyi_count` — fixed_centers_convergence is not thm:fixed_centers (no frozen θ_j or a_j, no h/g hypotheses, δ in place of εβ^{-1/2}, every solution in place of a.e.) and is false at m = 0, n ≥ 1; renyi_count proves only the packing number

### Transformer.Causal.Packing — —

- ok Transformer/Causal/Packing/Basic.lean (2): `volume_ball_eucSpace`, `volume_closedBall_eucSpace`
- ok Transformer/Causal/Packing/Count.lean (4): `pow_sub_pow_le`, `le_pow_sub_pow`, `card_le_of_separatedOnSphere`, `card_ge_of_maximalSeparated`
- ok Transformer/Causal/Packing/Lower.lean (1): `maximal_volume_ledger`
- ok Transformer/Causal/Packing/Renyi.lean (2): `card_le_at_renyi_scale`, `exists_card_ge_at_renyi_scale`
- ok Transformer/Causal/Packing/Upper.lean (3): `separated_volume_ledger`, `card_le_of_separated`, `exists_maximalSeparated`

### Transformer.Causal — arXiv:2411.04990

- ok Transformer/Causal/ParkingCount.lean (2): `continuous_geoDist`, `strong_renyi_expected_count`
- issue Transformer/Causal/SingleToken.lean (3): `norm_eq_one_of_singleTokenODE`, `not_forall_single_token_convergence`, `single_token_convergence_trivial` — lemma1 itself is not stated (see section 1)

### Transformer.Clusters — arXiv:2305.05465

- ok Transformer/Clusters/Extremum.lean (4): `antitone_sup'_of_hasDerivAt`, `le_sup'_mul_exp_of_hasDerivAt`, `le_sqrt_add_one_mul_exp`, `hasDerivAt_norm_sq_rclike`
- ok Transformer/Clusters/Section10_ProjHull.lean (4): `isProjOnto_one`, `convex_image_tokenHull`, `image_tokenHull_subset`, `image_tokenHull_antitone`
- issue Transformer/Clusters/Section10_Remainder.lean (3): `projScore_self`, `scoreRemainder_self`, `softmax_le_exp_neg` — FIXED: refuted as `not_exists_bound_scoreRemainder` (Section10_RemainderFalse). exists_bound_scoreRemainder (e:boundrj) FALSE: d=2, V=diag(1,0), lam=1, F=span e1, G=span e2, A=Q=K=[[1,1/2],[1/2,1]], mu=0, constant z1=(0,1), z2=(0,2): r12=(1+a^2)-a^2 e^{2t}; paper silently uses pi_F(A e^{tV} z)=e^{lam t} pi_F(A z), needs A to respect F+G
- issue Transformer/Clusters/Section10_Step2.lean (2): `hasDerivAt_norm_sq_proj`, `candidatesThickening_subset` — cl:gamma'12 and Step 2' faithful to the paper, but the paper's route to Step 2' goes through the false e:boundrj; truth undecided
- ok Transformer/Clusters/Section12_Feedforward.lean (6): `actPointwise_apply`, `actPointwise_zero`, `actPointwise_id`, `relu_zero`, `mlpRescaledDynamics_id_iff`, `mlpRescaledDynamics_const`
- ok Transformer/Clusters/Section12_Generic.lean (4): `qkMatrix_one`, `sum_smul_single`, `inner_eq_sum_qkMatrix`, `attentionMatrix_congr_qkMatrix`
- ok Transformer/Clusters/Section12_MultiHead.lean (2): `multiHeadTransformer_one_iff`, `multiHeadTransformer_zero`
- ok Transformer/Clusters/Section1_Dynamics.lean (8): `attentionMatrix_pos`, `attentionMatrix_nonneg`, `sum_attentionMatrix`, `transformerDynamics_const`, `discreteTransformer_const`, `isPosDefOp_id`, `isPosDefQK_of_isAttentionRoot`, `isAttentionRoot_id`
- ok Transformer/Clusters/Section2_LowRank.lean (3): `pi_single_one_nonneg`, `sum_pi_single_one`, `isBooleanLimit_of_isBooleanRows`
- ok Transformer/Clusters/Section3_Discrete.lean (2): `discreteRescaled_const`, `discreteTransformer_iff_rescaled`
- ok Transformer/Clusters/Section3_IdCase.lean (3): `attentionMatrix_expTime_one`, `rescaledDynamics_one_iff`, `rescaledDynamics_one_const`
- ok Transformer/Clusters/Section3_Rescaled.lean (9): `expTime_zero`, `expTime_zero_map`, `expTime_one_apply`, `hasDerivAt_expTime`, `commute_smul_self`, `commute_expTime`, `expTime_neg_mul`, `rescaledDynamics_const`, `transformerDynamics_iff_rescaled`
- ok Transformer/Clusters/Section4_Codim.lean (1): `isUnstableSplitting_one`
- ok Transformer/Clusters/Section4_Hyperplanes.lean (5): `mem_affineShift_self`, `isGoodTriple_iff`, `span_singleton_eq_top`, `isGoodTripleWith_one`, `isGoodTriple_one`
- ok Transformer/Clusters/Section5_Mix.lean (2): `isPosDefQK_one`, `isGoodTripleMulti_one`
- ok Transformer/Clusters/Section6_ContEq.lean (1): `isContEqSolution_dirac`
- ok Transformer/Clusters/Section6_Kernel.lean (5): `isCarriedBy_dirac`, `attentionKernel_dirac_zero`, `ae_mem_closedBall`, `integrable_attentionWeight`, `attentionKernel_norm_le`
- ok Transformer/Clusters/Section6_KernelDeriv.lean (4): `scoreDual_apply`, `continuous_scoreDual`, `norm_scoreDual_le`, `attentionKernel_hasFDerivAt`
- ok Transformer/Clusters/Section6_KernelLip.lean (3): `abs_exp_sub_exp_le`, `abs_inner_le_of_mem_closedBall`, `attentionKernel_lipschitz_in_measure`
- issue Transformer/Clusters/Section6_WellPosed.lean (1): `isLocLipschitzCurve_const` — header says the rescaled well-posedness is stated independently because transformerDynamics_iff_rescaled is unproved; it is proved (stale docstring)
- ok Transformer/Clusters/Section7_Bounded.lean (3): `isBoundedToken_zero`, `tendsto_attention_of_tendsto_others`, `tendsto_row_isProbability`
- ok Transformer/Clusters/Section7_DistNonDec.lean (4): `softmax_monotone`, `inner_drift_sub_nonneg`, `norm_sub_monotone`, `ne_of_norm_sub_monotone`
- ok Transformer/Clusters/Section7_HigherDim.lean (3): `tendsto_attentionMatrix_of_tendsto_common`, `not_tendsto_zero_of_tendsto_common`, `not_tendsto_id_of_tendsto_common`
- ok Transformer/Clusters/Section7_LogSumExp.lean (5): `sum_rpow_mul_rpow_le`, `sum_exp_inner_pos`, `convexOn_logSumExp`, `transformerDynamics_one_iff`, `idNonrescaledDynamics_zero`
- ok Transformer/Clusters/Section7_Symmetric.lean (16): `norm_unit1`, `coord_smul_unit1`, `inner_smul_unit1`, `symSign_zero`, `symSign_one`, `symSign_two`, `symDrift_zero`, `attentionMatrix_symTriple`, `sum_exp_symSign_pos`, `sum_weighted_symSign`, `drift_symTriple`, `idNonrescaledDynamics_symTriple`, `isOrderedConfig_symTriple`, `isBoundedToken_symTriple`, `symInterior_ne_first`, `symInterior_ne_last`
- ok Transformer/Clusters/Section7_Unbounded.lean (3): `isOrderedConfig_subsingleton`, `idNonrescaledDynamics_single`, `exists_auxiliary_constant`
- ok Transformer/Clusters/Section8_Bounded.lean (3): `mul_exp_add_one_nonneg`, `inner_negIdDrift_neg`, `exists_bound_negIdDynamics`
- ok Transformer/Clusters/Section8_Energy.lean (2): `sum_exp_smul_eq_neg_smul_negIdDrift`, `integrableOn_sq_norm_negIdDrift`
- ok Transformer/Clusters/Section8_Origin.lean (4): `isIdentityQK_one`, `transformerDynamics_neg_one_iff`, `transformerDynamics_zero`, `attentionMatrix_tendsto_uniform`
- ok Transformer/Clusters/Section8_Polytope.lean (6): `mem_tokenHull`, `isCompact_tokenHull`, `idRescaledDynamics_const`, `tokenHull_antitone`, `mem_tokenHull_zero`, `exists_bound_idRescaled`
- ok Transformer/Clusters/Section8_Stationary.lean (2): `isStationaryConfig_zero`, `eq_zero_of_isStationaryConfig`
- ok Transformer/Clusters/Section9_Eigen.lean (6): `isEigenFunctional_one`, `le_maxCoord`, `minCoord_le`, `minCoord_le_maxCoord`, `hasDerivAt_eigenFunctional`, `abs_eigenFunctional_le`
- ok Transformer/Clusters/Section9_Fj.lean (3): `norm_sum_attentionMatrix_smul_le`, `minCoord_eq_neg`, `maxCoord_antitoneOn_minCoord_monotoneOn`
- ok Transformer/Clusters/Section9_Growth.lean (4): `isEigenFunctional_zero`, `norm_eigenFunctional_le`, `isProjContraction_zero`, `norm_proj_le`
- ok Transformer/Clusters/Section9_Hyperplanes.lean (4): `apply_smul_eq`, `tendsto_infDist_affineShift`, `finrank_ker_add_one`, `dist_tendsto_zero_of_tendsto_eigenFunctional`
- ok Transformer/Clusters/Section9_Limits.lean (5): `exists_tendsto_of_antitoneOn`, `exists_tendsto_of_monotoneOn`, `exists_tendsto_maxCoord_minCoord`, `bounded_of_coord_bounded`, `eq_sum_proj_single`

### Transformer.FrankWolfe — arXiv:2508.09628

- ok Transformer/FrankWolfe/Section2_Derivations.lean (2): `configHull_subset_of_isHardmaxStep`, `configHull_antitone` — lem:convHullDecreases as in §2.3; singleLeader (sorried) faithful, the paper's proof gap documented
- ok Transformer/FrankWolfe/Section2_HullFailure.lean (5): `diagTwo_apply`, `triangle_subset`, `triangle_nonneg`, `inner_single_one`, `not_configHull_subset_of_preconditioner` — the remark's counterexample, V exhibited
- ok Transformer/FrankWolfe/Section3_NegativeDefinite.lean (2): `isFrankWolfeStep_iff_isHardmaxStep`, `says` — reparametrization; fw_cluster (sorried) matches thm: fw.cluster with γ=2/(t+2)
- issue Transformer/FrankWolfe/Section4_Cells.lean (5): `mem_cell_iff`, `convex_cell`, `interior_cell_inter`, `iUnion_cell`, `cell_eq_vorCell_inter` — lem: cells ok. Sorried §5: (1) IsSAProcess reads eq: softmax.process literally, per target position: when x_j = x_j' (j≠j'), e.g. γ=1/2 after a swap, the constraint gives P = w_j instead of w_j + w_j', and the leftover mass may go anywhere, so first_phase/metastability quantify over non-processes (stronger than the paper, likely false). (2) metastability: radius r > 0 free instead of Cτ, and x_i^0 ∈ K dropped: stronger than lem: metastab.1, likely false for large r. (3) metastability for every β > 0 rather than β ≥ β_* — FIXED: IsSAProcess is now a transition kernel conditioned on the past (coinciding particles carry the sum of their weights; isSAProcess_single witnesses it); metastability takes radius Cτ with C > 1 outermost, τ bounded by the two geometric terms, x⁰ ∈ K, β ≥ β_*(γ), and its example witnesses every hypothesis including the process. Open: first_phase is vacuous at n = κ and its example does not witness the process.

### Transformer.GPTMini — arXiv:2512.01868

- issue Transformer/GPTMini/ClusteringTheorem.lean (sorried, no proved entries): project conjectures, not paper statements. `polynomial_rate` cites thm:preln-slow, which says only Var' = -Θ(Var/t) for continuous-time dynamics started in a narrow cone; the `C/L³` exponent, the global a.e. initial data and a `C` uniform over x₀ are all invented, and uniformity is likely false (starts near an unstable configuration linger arbitrarily long). `layer_clustering` transfers thm1 of 2411.04990 (continuous, on the sphere) to a discrete Pre-LN recursion with RoPE and XSA, with no argument that the bridges compose. The docstring's 'Block.attnSubLayer is still a placeholder' is stale. — FIXED (polynomial_rate): refuted as `not_polynomial_rate` (RateRefutation, via TwoTokens): two tokens, the first on its ray, the second started far out turns by O(√d/R) per layer, on an open ball of starts; the sorried statement is removed. The stale docstring is corrected. layer_clustering remains open.
- issue whole directory: `reference/model.py`, cited as the source everywhere, is not in the repository.

- ok Transformer/GPTMini/AttentionBounds.lean (6): `causalAttnWeights_zero_above`, `causalAttnWeights_nonneg`, `causalAttnWeights_bounds`, `attnOutput_norm_le`, `xsaProjection_norm_le`, `attentionHead_norm_le`
- ok Transformer/GPTMini/AttentionLipschitz.lean (3): `attnOutput_dist_le`, `norm_proj_sub_proj_le`, `xsaProjection_dist_le`
- ok Transformer/GPTMini/AttnSubLayerLipschitz.lean (2): `attnSubLayer_eq`, `attnSubLayer_dist_le`
- ok Transformer/GPTMini/Block.lean (3): `attnSubLayer_bounded`, `ffnSubLayer_bounded`, `blockForward_growth`
- ok Transformer/GPTMini/BlockLipschitz.lean (1): `ffnSubLayer_lipschitz`

### Transformer.GPTMini.Bridge — —

- ok Transformer/GPTMini/Bridge/ALMLookup.lean (8): `normL2_smul_of_pos`, `score_smul_key`, `score_smul_query`, `qknorm_indifferent_where_lookup_is_not`, `score_eq_inner_div`, `score_sub_eq_const_mul_lookup_sub`, `score_le_iff_lookup_le`, `score_le_iff_dist_le`
- issue Transformer/GPTMini/Bridge/CausalConnection.lean (2): `causalAttnWeights_matches_eq_csa`, `causalAttnWeights_eq_csa_coeff` — both theorems unfold causalAttnWeights and never mention Causal.CSA; the docstring's 'puts thm1 in scope' overclaims, and RoPE makes Q,K position-dependent, so no fixed Q,K of eq: csa matches
- issue Transformer/GPTMini/Bridge/RoPEAsTimeVarying.lean (5): `applyRope_add`, `applyRope_smul`, `ropeIsometry_apply`, `rope_timeParam_norm_preserved`, `rope_score_relative` — rope_timeParam indexes Q(t) by position, but TimeParam's t is ODE time (depth), shared by all tokens: 'RoPE attention is transformerODE at Q(t),K(t)' is false; lemmas themselves correct
- ok Transformer/GPTMini/Bridge/RoPENoClustering.lean (3): `transformerODE_const_antipodalPair`, `not_rope_clustering_antipodalPair`, `not_forall_rope_clustering`
- ok Transformer/GPTMini/Bridge/SphereResidence.lean (3): `rmsNorm_direction_on_sphere`, `toSphere_norm`, `token_sequence_on_sphere`
- ok Transformer/GPTMini/Bridge/XSAEquivalence.lean (2): `xsaProjection_eq_sphereProj`, `attentionHead_eq_sphereProj`

### Transformer.GPTMini — arXiv:2512.01868

- ok Transformer/GPTMini/CausalMHA.lean (2): `causalAttnWeights_row_sum`, `xsaProjection_orthogonal`
- ok Transformer/GPTMini/Config.lean (8): `head_dim_pos`, `group_size_pos`, `n_kv_heads_mul_group_size`, `kvHead_surjective`, `kvHead_val_eq`, `kvHead_val_eq_zero`, `exists_shared_kvHead`, `cache_eq_group_size_mul`
- ok Transformer/GPTMini/HeadLipschitz.lean (4): `attentionHead_eq`, `headLipschitz_nonneg`, `headLipschitz_mono`, `attentionHead_dist_le`
- ok Transformer/GPTMini/MeanFieldRefutation.lean (3): `attentionHead_one`, `preLNHead_one_ne_zero`, `not_mean_field_clustering`
- issue Transformer/GPTMini/Model.lean (1): `forward_total` — forward_total is decorative: ∃ y, f = y closed by rfl, true of every term; docstring's 'T ≤ max_seq_len' is not even a hypothesis

### Transformer.GPTMini.Properties — —

- ok Transformer/GPTMini/Properties/Causal.lean (1): `attnOutput_causal`
- issue Transformer/GPTMini/Properties/Entropy.lean (6): `entropy_le_log_card`, `log_le_entropy_of_le`, `softmaxEntropy_nonneg`, `softmaxEntropy_eq`, `softmaxEntropy_le_log_vocab`, `softmaxEntropy_lower_bound` — header promises H ≥ log V - C(L,‖W‖,α_max) via nonexistent forward_lipschitz_embedding; what is proved takes the logit bound M as hypothesis
- ok Transformer/GPTMini/Properties/Lipschitz.lean (2): `hidden_isStream`, `stream_lipschitz`
- ok Transformer/GPTMini/Properties/LipschitzConstants.lean (6): `attnLipschitz_nonneg`, `ffnLipschitz_nonneg`, `one_le_perBlockLipschitz`, `perBlockLipschitz_nonneg`, `endToEndLipschitz_nonneg`, `blockForward_lipschitz`
- ok Transformer/GPTMini/Properties/OutputSimplex.lean (6): `softmaxOutput_nonneg`, `softmaxOutput_denom_pos`, `softmaxOutput_pos`, `softmaxOutput_sum_one`, `softmaxOutput_le_one`, `softmaxOutput_is_distribution`
- ok Transformer/GPTMini/Properties/StreamGrowth.lean (3): `blockGrowth_nonneg`, `residual_stream_linear_growth`, `final_representation_norm_le`

### Transformer.GPTMini — arXiv:2512.01868

- issue Transformer/GPTMini/QKNorm.lean (5): `normL2_norm_le`, `score_bounded`, `partition_bounds`, `rmsNorm_eq_smul_normL2`, `rmsScore_eq_score` — normL2 = x/(‖x‖+eps) but F.normalize is x/max(‖x‖,eps): undocumented deviation; header's bound n⁻¹e^{±2α} should be e^{±2e^α} (AttentionBounds has it right)
- ok Transformer/GPTMini/QKNormLipschitz.lean (3): `normL2_lipschitz`, `abs_inner_sub_inner_le`, `score_lipschitz`
- ok Transformer/GPTMini/RMSNorm.lean (5): `rmsNorm_norm_eq_sqrt_d`, `rmsNormEps_norm_le`, `rmsNormEps_lipschitz`, `rmsNorm_pos_homog`, `continuous_rmsNorm`
- ok Transformer/GPTMini/ReLU2FFN.lean (12): `relu2_nonneg`, `relu2_of_pos`, `relu2_of_nonpos`, `relu2_le_sq`, `continuous_relu2`, `relu2Vec_coord_nonneg`, `relu2Vec_apply`, `relu2Vec_norm_bound`, `euclidean_norm_le_of_coord_le`, `abs_relu2_sub_le`, `relu2Vec_lipschitz`, `relu2FFN_lipschitz_on_ball`
- ok Transformer/GPTMini/Reshape.lean (15): `d_model_eq`, `qkv_disjoint`, `qkvSlice_apply`, `headSlice_apply`, `headMerge_apply`, `headSlice_headMerge`, `headMerge_headSlice`, `norm_comp_injective_le`, `qkvSlice_norm_le`, `qkvV_injective`, `qkvQ_injective`, `qkvK_injective`, `headSlice_norm_le`, `headMerge_norm_sq`, `headMerge_norm_le`
- ok Transformer/GPTMini/ReshapeDist.lean (3): `qkvSlice_dist_le`, `headSlice_dist_le`, `headMerge_dist_le`
- ok Transformer/GPTMini/RoPE.lean (11): `ropeSplit_symm_inl_inl`, `ropeSplit_symm_inl_inr`, `applyRope_apply`, `sum_split`, `sum_split'`, `applyRope_isometry`, `ropeCoord_sub`, `applyRope_sub`, `applyRope_dist`, `ropeAngle_sub`, `applyRope_relative`
- ok Transformer/GPTMini/SoftmaxStability.lean (3): `causalAttnWeights_le_mul`, `causalAttnWeights_l1_le`, `causalAttnWeights_l1_le_linear`
- ok Transformer/GPTMini/TotalVariation.lean (4): `l1_le_two`, `l1_le_of_le_mul`, `exp_two_mul_sub_one_le`, `l1_le_of_le_exp`

### Transformer.Homogenized — arXiv:2604.01978

- issue Transformer/Homogenized/Regimes.lean (sorried, no proved entries): `ballistic_regime` and `modified_regime` state the rate `Ce^{Ct_L}(t_L+1)max(η,α)` of cor:ode1/cor:ode2 without the printed leading `η` — weaker than printed; the docstring argues the printed rate is false but no counterexample theorem is proved. `diffusive_regime` is stated only at grid times `t = kη`. FIXED for the two deterministic corollaries: `not_ballistic_regime_printed` and `not_modified_regime_printed` (`RegimeRefutation`, on the rotation head of `RotationHead`) prove the printed rate false; the grid restriction of `diffusive_regime` is justified by `not_diffusive_regime_printed` (`WeakErrorRefutation`).
- issue Transformer/Homogenized/WeakError.lean (sorried, no proved entries): `weak_error_modified` restricted to grid times; the falsity of the sup-over-all-t version is argued in the docstring, not proved. FIXED: `not_weak_error_modified_printed` (`WeakErrorRefutation`) proves the sup-over-all-t version false.
- note whole directory: `IsItoSolution` and the martingale-problem solution concepts are weaker than the source's strong solutions; documented in their docstrings.

- ok Transformer/Homogenized/Barycenter.lean (7): `softWeight_pos`, `softWeight_eq_attnWeight`, `softBary_dirac`, `norm_softBary_le_one`, `baryCorr_self`, `baryCorr_comm`, `normalizeLayer_smul`
- ok Transformer/Homogenized/Basic.lean (12): `attnField_zero_beta`, `norm_normalizeLayer`, `normalizeLayer_of_norm_one`, `interpChain_natCast_mul`, `IsVarianceProxy.unique`, `valueMap_zero`, `attnField_zero_param`, `meanField_dirac_zero`, `fluct_dirac_zero`, `bField_dirac_zero`, `Gfield_dirac_zero`, `isVarianceProxy_dirac_zero`
- ok Transformer/Homogenized/CoupledPair.lean (2): `sphHess₂_zero`, `isCoupledPair_dirac_zero`
- ok Transformer/Homogenized/CoupledSystem.lean (2): `pocHess_zero`, `isPoCSystem_dirac_zero`
- ok Transformer/Homogenized/Defs.lean (1): `attnWeight_pos`
- ok Transformer/Homogenized/Frozen.lean (11): `hasVanishingField_dirac_zero`, `mvGenerator_of_vanishing`, `mvGenerator₂_of_vanishing`, `isMcKeanVlasovSolution_frozen`, `isCoupledPair_frozen`, `measurable_clampSphere`, `clampSphere_of_norm_eq_one`, `norm_clampSphere`, `map_clampSphere`, `isCoupledPair_uniformAmbient`, `exists_norm_eq_one`
- ok Transformer/Homogenized/GaussianEnsemble.lean (9): `entryVar_zero`, `entryVar_one`, `entryVar_two`, `gaussBlock_apply`, `measurable_gaussBlock`, `map_gaussBlock`, `isGaussianHeadLaw_gaussHeadLaw`, `stdSigmaV_sq`, `isGaussianHeadLaw_stdScaling`
- ok Transformer/Homogenized/GaussianInit.lean (3): `isGaussianHeadLaw_dirac_zero`, `alphaOf_gaussian`, `isDiffusiveSde_dirac_zero`
- ok Transformer/Homogenized/GaussianInterp.lean (1): `gaussInterp_zero_fun`
- ok Transformer/Homogenized/GaussianKernel.lean (3): `valueMap_smul`, `valueMap_integral`, `attnFieldOf_eq_valueMap_softBary`
- ok Transformer/Homogenized/Generator.lean (4): `sphHess_zero`, `noiseField_one_one`, `isItoSolution_dirac_zero`, `isBallisticFlow_dirac_zero`
- ok Transformer/Homogenized/GramStability.lean (1): `not_forall_overlapDrift_eq_simplexDrift` — refutation of D(R(γ))=b(γ) verified against eq:Dij_explicit_clean
- ok Transformer/Homogenized/HansonWright.lean (2): `matOpNorm_nonneg`, `matOpNorm_zero`
- ok Transformer/Homogenized/Logistic.lean (3): `logisticGenerator_one`, `logisticGenerator_neg_one`, `isLogisticSolution_one`
- ok Transformer/Homogenized/LogisticLimit.lean (1): `rescaledOverlap_zero`
- ok Transformer/Homogenized/McKeanVlasov.lean (2): `isWeakSpdeSolution_dirac_zero`, `isMcKeanVlasovSolution_dirac_zero`
- ok Transformer/Homogenized/MeanField.lean (7): `real_inner_proj_left`, `integral_empMeasure`, `attnFieldOf_empMeasure`, `meanFieldOf_empMeasure`, `fluctOf_empMeasure`, `Gfield_eq_GfieldOf`, `inner_covKernel_proj`
- issue Transformer/Homogenized/MeanFieldLipschitz.lean (7): `attnFieldOf_dirac`, `GfieldOf_dirac`, `valueMap_radMatrix`, `integral_valueMap_radLaw`, `norm_proj_sub_proj_sq`, `sqrt_integral_norm_GfieldOf_radLaw`, `not_forall_satisfying_MF_rate` — satisfying_MF rate refuted only via β↓0; corrected (1+β²) rate not stated
- ok Transformer/Homogenized/Metastability.lean (1): `effBeta_pos`
- ok Transformer/Homogenized/MvGenerator.lean (8): `isMartingaleOn_const`, `sphHess₁_zero`, `attnFieldOf_zero`, `meanFieldOf_dirac_zero`, `fluctOf_dirac_zero`, `GfieldOf_dirac_zero`, `mvGenerator_dirac_zero`, `spdeNoise_dirac_zero`
- ok Transformer/Homogenized/OneDim.lean (5): `mul_self_eq_one_of_norm_eq_one`, `proj_one`, `GfieldOf_one`, `hasVanishingField_one`, `isCoupledPair_uniformAmbient_one`
- ok Transformer/Homogenized/OverlapDrift.lean (2): `sphGenerator_overlap_diffusive`, `overlapDrift_self`
- ok Transformer/Homogenized/OverlapObservable.lean (7): `overlapForm_apply`, `hasFDerivAt_overlap`, `fderiv_overlap`, `fderiv_fderiv_overlap`, `iteratedFDeriv_two_overlap`, `sphHess_overlap`, `sphGenerator_overlap`
- ok Transformer/Homogenized/RademacherLaw.lean (7): `iIndepFun_of_const_of_ne`, `integral_fairCoin`, `radMatrix_apply`, `measurable_radMatrix`, `integral_radLaw`, `isHighOrderLaw_rademacher`, `hasHighOrderLaw_rademacher`
- ok Transformer/Homogenized/RandomChain.lean (4): `iIndepFun_of_unit`, `isHighOrderLaw_dirac_zero`, `hasHighOrderLaw_dirac_zero`, `isRandomChain_dirac_zero`
- ok Transformer/Homogenized/Simplex.lean (4): `attnProb_pos`, `sum_attnProb`, `isSimplexConfig_of_subsingleton`, `is`
- ok Transformer/Homogenized/SimplexBary.lean (5): `attnProb_eq_softWeight`, `softBary_empMeasure`, `inner_softBary_simplex`, `baryCorr_simplex`, `baryCorr_simplex_self`
- ok Transformer/Homogenized/SimplexDrift.lean (4): `feeds`, `overlapDrift_congr`, `overlapDrift_simplex`, `overlapDrift_simplex_sub_simplexDrift`
- ok Transformer/Homogenized/SlowMotion.lean (1): `isLowTemperature_uniformAmbient`
- ok Transformer/Homogenized/SmallBeta.lean (2): `meanOverlap_const`, `hasDerivAt_logistic`
- ok Transformer/Homogenized/SoftmaxDerivatives.lean (2): `hasFDerivAt_softmaxWeight`, `fderiv_softmaxWeight`
- ok Transformer/Homogenized/UniformLaw.lean (10): `measurable_radialProj`, `norm_radialProj`, `radialProj_comp_isometry`, `isOpen_puncturedBall`, `measurableSet_puncturedBall`, `ne_zero_of_mem_puncturedBall`, `measure_puncturedBall_pos`, `measure_puncturedBall_lt_top`, `uniformAmbient_apply`, `isUniformAmbient_uniformAmbient`

### Transformer.Interpolation — arXiv:2411.04551

- ok Transformer/Interpolation/AtomClustering.lean (4): `fullVF_diracProb_self`, `cauchyPB_const_diracProb`, `not_forall_clustering_to_atom`, `exists_dirac_close_of_diam_tendsto`
- ok Transformer/Interpolation/BallDecomposition.lean (1): `not_exists_ball_of_mass_of_dirac`
- ok Transformer/Interpolation/Basic.lean (3): `eq_of_mem_support_dirac`, `antipode_ne`, `basePoint_mem_positiveQuadrant`
- issue Transformer/Interpolation/Clustering.lean (2): `le_inner_barycenter`, `barycenter_ne_zero` — proved lemmas ok; FIXED compression: restated with Interpolation.W2, convG (new, Basic), x ∈ conv_g, distinct targets, ∀ T, every solution (and one exists). Was: compression (sorried) carries W₂ as a free function — false as written, like the refuted monge (W₂≡1, ε=1/2, any atomless μ₀), should be Interpolation.W2; also drops x_k^i ∈ conv_g supp μ_0^i and the distinct-target condition, and the switch count is only 'finite'. FIXED Disentanglement: prop: separation is false for equal data and is refuted in every d ≥ 1 (`not_separation`, SeparationFalse); restated with ∃ C universal, 1 ≤ N, pairwise distinct μ₀, convG, every solution; first_quadrant: C depends on μ₀ but not T, d ≥ 2 (S⁰: all fields vanish); induction_barycenter: ∃ C universal, supports in ℚ₁, one index j, geodesic ball, every solution; perturbation (moved to Perturbation.lean): distinct measures, both cases incl. the Lipschitz invertible flow map and identity.flow, witness in d = 2. Was: Disentanglement: separation and induction_barycenter take C : ℕ universally — C=0 forces K=0, PiecewiseConstant θ T 0 forces T=0, contradicting hT: both false as written (the bug fixed in targets_atoms); first_quadrant picks C after μ₀ and T (paper C=C(N)). FIXED NeuralODE: eq:neural.ode.sphere and cauchy.pb / characteristics (Basic.cauchyPB, IsCharacteristic) now in integrated form with an integrable integrand — piecewise-constant θ has no derivative at switches, so the differential form had no solution and every 'every solution' clause was vacuous; both propositions restated with ∃ C > 0 before the data, W U b piecewise constant through neuralParams with ≤ 6M (resp. 6) switches, the norm bound, ε > 0, existence and every solution. Was: NeuralODE: prop_interpolation_neural_ode / lem_induction_neural_ode — switches is a free ∃ unconnected to (W,U,b), W is not required piecewise constant, the norm bound CM/(T min ε_i) and ε_i>0 are dropped, and '∀ x solving neuralODESphere' is vacuous for (W,U,b) making the ODE unsolvable (e.g. W=1_ℚ(t)·I): trivially provable. FIXED BallTransport: in the source's order (parameters and flow map before μ₀) both lemmas are false whenever the mass must move — a bijection fixing the complement of S maps S onto S, and Dirac data then forces S ⊆ target (`not_two_balls`, `not_tubular_mass_movement`, BallTransportFalse, witnesses on the circle); restated with μ₀ first as in the proof, d ≥ 2 (S⁰: no mass moves), the flow map of eq:neural.ode.sphere (IsNeuralFlow), μ(T) = φ^T_# μ₀, existence and every solution, and the W₂ 'Furthermore' clause (collapseInto). Was: BallTransport: two_balls / tubular_mass_movement '∀ μ, cauchyPB θ μ →' is vacuous the same way, θ being free off [0,T) while cauchyPB asks all t∈ℝ
- ok Transformer/Interpolation/IdentityFlow.lean (2): `fullVF_eq_zero`, `identity_flow`
- issue Transformer/Interpolation/Main.lean (3): `isHole_antipode_diracProb`, `monge`, `not_forall_monge` — monge proved and faithful; targets_atoms, main_result ok; FIXED: lem: hyp.propagation refuted for every invertible representation of the flow maps (`not_hyp_propagation`, HypPropagationFalse): T^i may merge atoms. Was: hyp_propagation (sorried) takes Φ₁, Φ₃ as free functions, not the flow maps of prop: separation — false as written (Φ₁ ≡ const δ_x, Φ₃ = id, two distinct Dirac data)
- fixed Transformer/Interpolation/MassConcentration.lean (5): `wToBall`, `measurableSet_positiveQuadrant`, `antipode_notMem_positiveQuadrant`, `one_le_dist_antipode_of_mem`, `not_massConcentrationQ1` — cl: W.to.ball and lem: mass.concentration.Q1 as printed
- ok Transformer/Interpolation/MassConcentrationSqrt.lean (2): `diagPoint_mem_positiveQuadrant`, `massConcentrationQ1_sqrt` — paper's construction, rate 2√η, deviations in docstring
- issue Transformer/Interpolation/Settling.lean (4): `neuralODESphere_iff_perceptronField`, `norm_sub_le_of_contraction`, `Hartman_Grobman`, `not_forall_Hartman_Grobman` — Hartman_Grobman assumes hcontr (linear contraction along the trajectory), which is the conclusion in differential form: a Gronwall lemma under the paper's name. Paper: the specific ODE xdot=(<gamma,x>-eps/2)_+ Proj_x omega_+, x0 in S_+={<gamma,x> >= eps}; faithful form looks provable via u=<x,omega_+>
- ok Transformer/Interpolation/Wasserstein.lean (5): `transportCosts_nonneg`, `bddBelow_transportCosts`, `W2_nonneg`, `W2_le_of_coupling`, `W2_self`

### Transformer.Kinetic — arXiv:2605.09213

- ok Transformer/Kinetic/Accuracy.lean (2): `here`, `acc_def`
- ok Transformer/Kinetic/Codewords.lean (8): `torusDist_comm`, `torusDist_nonneg`, `torusDist_triangle`, `torusDist_le_abs`, `coe_add_int_mul_period`, `two_pi_div_le_torusDist_codeword`, `exists_codeword_close`, `isNearestCodeword_iff`
- ok Transformer/Kinetic/Correlations.lean (1): `torusConv_const`
- ok Transformer/Kinetic/Defs.lean (4): `hasDerivAt_wBeta`, `wBeta_periodic`, `wBetaDeriv_periodic`, `graphon_of_le`
- issue Transformer/Kinetic/Hardy.lean (2): `besselI_succ_zero`, `hardyProfile_zero` — u_shape faithful to thm:U-shape, but its satisfiability example does not exhibit a t meeting eq:affine-smallness (needs sup a_n < ∞, i.e. aCoeff_tendsto_zero, sorried)
- ok Transformer/Kinetic/MeanField.lean (1): `periodic_deriv`

### Transformer.MeanField — arXiv:2512.01868

- issue [FIXED global_clustering: UniformTuple, SA and USA, global existence; mfclust: L² density against the uniform law, Wasserstein.W2, eq:continuity USA field — moved to MeanField.GlobalRate] Transformer/MeanField/Clustering.lean (1): `exists_common_hemisphere_of_linearIndependent` — proved lemma ok; global_clustering (sorried) quantifies over a free reference measure σ — false as written (σ = δ at the antipodal pair, stationary for SA), should be Perspective.UniformTuple; only SA, paper also USA. meanField_exponential_rate (thm:mfclust) carries W₂ free and drops 'μ₀ has density f₀ ∈ L²' — false as written: μ₀ = ⅔δ_e + ⅓δ_{-e} has R₀ = 1/9 > 0 and is stationary for every β
- ok Transformer/MeanField/Equiangular.lean (3): `equiangularSA_const_simplex`, `not_exists_rate_at_simplex`, `equiangularSA_const_one`
- ok Transformer/MeanField/EquiangularLimit.lean (4): `tendsto_cos_of_tendsto`, `equiOutCos_seq_eq`, `equiDiag_add_equiOff_seq`, `long_context_phase_transition`
- ok Transformer/MeanField/EquiangularPhases.lean (8): `equiDiag_eq_one_div`, `equiDiag_seq_eq`, `tendsto_log_natAdd_two`, `tendsto_ratio_natAdd`, `ratio_natAdd_mem`, `tendsto_equiDiag_of_lt`, `tendsto_equiDiag_of_eq`, `tendsto_equiDiag_of_gt`
- ok Transformer/MeanField/EquiangularRate.lean (2): `equiangular_local_rate`, `not_equiangular_local_rate_zero`
- ok Transformer/MeanField/EquiangularWeights.lean (13): `equiNorm_pos`, `sum_exp_equiGram`, `sum_equiWeight`, `equiWeight_self`, `equiWeight_of_ne`, `sum_equiWeight_mul_equiGram`, `equiOutInner_eq`, `equiDiag_pos`, `equiOff_pos`, `equiDiag_add_equiOff`, `sum_equiWeight_sq`, `sum_equiWeight_mul`, `equiOutCos_of_ne`
- issue Transformer/MeanField/Noisy.lean (1): `inner_noisyDrift_eq_zero` — inner_noisyDrift_eq_zero ok; fokkerPlanck (a def, no theorem uses it) silently differs from the printed eq:Fokker: sign of the Laplacian (the source's printed sign is backward heat) and drift Perspective.vectorField (normalized SA) instead of the source's unnormalized ∫e^{β⟨·,y⟩}y dμ_t; deviation undocumented
- issue Transformer/MeanField/PairMerge.lean (4): `hardmaxPair_const`, `hardmaxPair_stationary`, `hardmaxPair_inner_hasDerivAt`, `hardmaxPair_eq_of_eq` — proved lemmas ok (hardmaxPair_eq_of_eq documents the source's 'merge in finite rescaled time' as wrong). Merging.agazzi_merge (sorried) likely false: clusterSA keeps self-attention, so the pair's rescaled speed is (α_j̄/α_ī)·e^{β(ρ(s)-ρ(0))}, which has mass-ratio factor and diverges once ρ rises — the β→∞ limit is not hardmaxPair; the survey's paraphrase of Bruno–Pasqualotto–Agazzi needs the original statement — FIXED: `not_agazzi_merge` (MeanField/MergingFalse.lean) refutes the survey's statement with its own α_j ≥ 0: α = (0,1), the heavy cluster is stationary under eq:SA but moves under hardmaxPair; closed-form trajectories in MergingPaths.lean; sorried agazzi_merge removed; positive-mass failure argued in the docstring, not formalized (original Bruno–Pasqualotto–Agazzi statement not in papers/)

### Transformer.Metastability — arXiv:2410.06833

- ok Transformer/Metastability/AlphaDist.lean (3): `norm_sub_le_of_mem_sphericalCap`, `αDist_le_of_orthogonal`, `isSeparated_of_near_orthogonal` — 4ε+4√ε replaces the proof step ε²+2ε (false: sup = sin 2θ at an orthogonal frame); deviation documented
- ok Transformer/Metastability/AngularEnergy.lean (5): `inner_circlePoint`, `hasDerivAt_pairFst`, `hasDerivAt_pairSnd`, `hasDerivAt_angularEβ`, `Eβ_circlePoint` — angular gradient, helper
- ok Transformer/Metastability/BakryEmery.lean (2): `bakry_emery`, `not_forall_bakry_emery` — lem: bakry-emery with gradient/flow tied to E; refutation checked in §1
- issue Transformer/Metastability/CapVariance.lean (1): `variance_inequality` — variance_inequality weakened to 2n e^{-(1-α)β}; the paper's n is correct: outside term a_ij⟨x_j, w - η x_i⟩ ≥ -a_ij √(1-η²) ≥ -a_ij. Restore n
- issue Transformer/Metastability/CapVelocity.lean (3): `inner_proj_softmax_eq`, `cap_variance_bound`, `inner_proj_softmax_ge` — inner_proj_softmax_ge carries 2n; -1 per outside token suffices (see CapVariance)
- ok Transformer/Metastability/CollapseODE.lean (1): `exists_collapse_time` — lem: eminem constants exact
- ok Transformer/Metastability/ExponentialFlow.lean (2): `expFlow_spec`, `not_quantitative_inequality_two_mul` — witness + refutation checked in §1
- ok Transformer/Metastability/InitialUniform.lean (1): `uniform_separated` — coro: cm from concentration carried; technical.cond corrected, stronger; prop and claim faithful
- issue Transformer/Metastability/MainTheorem.lean (3): `rho_diff_ineq`, `not_rho_diff_ineq_of_free`, `eminem` — rho_diff_ineq weakened to 4n: paper's 2n holds (<x_k, x_j - rho x_i> >= -sqrt(1-rho^2) >= -1 per term); metastability: k,w free (not the isSeparated witnesses), lambda bound eq: lambda.3 dropped, T1/T2 bounds only in a comment (MetastabilityTimes unused), USA half omitted; eminem, propagation faithful
- issue Transformer/Metastability/MeanField.lean (2): `not_forall_cap_exit`, `not_forall_variance_small` — metastability_mf: T1 < (eps/k)e^{beta(1-alpha-8eps)} < T2 dropped; lam unbounded and fixed before T (paper: for all 0<lam<gamma, after T); centre weakened from argmin_{Phi^t(S_q(eps))}<x,w_q> to an existential point of the cap (docstring admits 'weaker'); refutations not_forall_cap_exit / not_forall_variance_small are strawmen (eta, V free), statements off the books
- issue Transformer/Metastability/OttoReznikoff.lean (2): `not_forall_otto_reznikoff`, `not_forall_claim_one` — [FIXED: not_otto_reznikoff] otto_reznikoff FALSE as written: gradE is not tied to E (NormedSpace, no HasGradientAt); M=R, E=x^2/2, N={0}, gradE u = -u satisfies H1/H2, flow u=e^t diverges. PL_borjan: kappa chosen after Theta (paper kappa(beta,n)) makes it nearly trivial; k, omega, lam free (not the hsep witnesses / eq: lambda.1-2). otto_attention: tau.small missing, k,omega free (k=0 gives N=univ where H2 fails), lam free -- likely false; not_forall_claim_one strawman
- issue Transformer/Metastability/PairVelocity.lean (3): `pair_sum_bound`, `inner_proj_softmax_pair`, `inner_proj_softmax_pair_sum` — lemmas true but loose: bracket <x_k,x_j - rho x_i> >= -sqrt(1-rho^2) >= -1, so leakage is n e^{-(1-alpha)beta} per half; tightening gives the paper's 2n in rho_diff_ineq
- ok Transformer/Metastability/QuantitativeInequality.lean (1): `quantitative_inequality` — c/2 correction (2c refuted in ExponentialFlow), sign convention of (H1), PL along the path and X(T)=v made explicit
- ok Transformer/Metastability/ReversePL.lean (2): `reverse_PL_acceleration`, `not_forall_reverse_PL_acceleration` — Groenwall with the chain rule carried; refutation shows hchain necessary; witness sharp
- issue Transformer/Metastability/Staircase.lean (1): `not_exact_time_scale` — modifiedUSA: uses the survey's angularUSA (-(1/n) sum e^{beta cos}) instead of this paper's eq: usa.angles (sum e^{beta(cos-1)}, no 1/n) -- time scale off by n e^beta, which the reparam tau_beta depends on; T_star not the infimum (any later time passes); no initial condition theta*(T*)=theta(T*); card=2 hypothesis absent; merged pair counted twice in the post-merge sum (paper: n-1 particles). isWellPrepared faithful. not_exact_time_scale: strawman noted in section 1
- issue Transformer/Metastability/StaircaseProfile.lean (2): `are`, `staircase_profile_vacuous_at_zero` — plateau indexing shifted: paper's max over i in 1..k covers (T_1,T_2),...,(T_k,+inf); Lean covers (T_0,T_1),...,(T_{k-1},T_k), dropping the final infinite plateau (energy reaching its maximum); reparam built from theta not the modified dynamics; inherits modifiedUSA issues

### Transformer.Modes — arXiv:2412.09080

- ok Transformer/Modes/Growth.lean (6): `tendsto_natSucc_atTop`, `isLittleO_rpow_rpow_atTop`, `isLittleO_rpow_rpow_nat`, `isLittleO_rpow_sq_div_log_atTop`, `isLittleO_rpow_sq_div_log_nat`, `tendsto_rpow_natSucc_atTop` — statements match thm:main-result / thm:mammen / sec: sketch; Mammen's added 0∈[a,b], a<b documented
- ok Transformer/Modes/Section1_Belt.lean (4): `sq_mem_Icc_iff`, `setOf_sq_mem_Icc`, `belt_eq_union`, `sqrt_sub_sqrt_isTheta` — statements match thm:main-result / thm:mammen / sec: sketch; Mammen's added 0∈[a,b], a<b documented
- ok Transformer/Modes/Section1_KDE.lean (10): `contDiff_kde`, `kde_nonneg`, `kde_pos`, `map_gaussianSample_eval`, `modeSet_mono`, `modeCount_mono`, `expectedModes_mono`, `expectedModesReal_nonneg`, `isLocalMax_kde_one`, `one_le_modeCount_kde_one` — statements match thm:main-result / thm:mammen / sec: sketch; Mammen's added 0∈[a,b], a<b documented
- ok Transformer/Modes/Section1_Main.lean (1): `isRegime_succ` — statements match thm:main-result / thm:mammen / sec: sketch; Mammen's added 0∈[a,b], a<b documented
- ok Transformer/Modes/Section1_Mammen.lean (1): `isLittleO_mammen_mid_sqrt` — statements match thm:main-result / thm:mammen / sec: sketch; Mammen's added 0∈[a,b], a<b documented
- ok Transformer/Modes/Section1_Sketch.lean (3): `isSlowGrowth_sqrt_log_log`, `tendsto_exp_neg_omega`, `isLittleO_tail_sqrt` — statements match thm:main-result / thm:mammen / sec: sketch; Mammen's added 0∈[a,b], a<b documented
- ok Transformer/Modes/Section2_Degenerate.lean (5): `twoPoint_sum_le`, `isLocalMax_kde_twoPoint`, `deriv_deriv_kde_twoPoint`, `not_forall_modeSet_subset_upcrossingSet`, `expectedUpcrossings_le_expectedModes` — matches §2 (eq:Fn, eq: Gt, thm:kac-rice items 1-4, eq:main-eq-form, lem:phi-t, lem:moments-p); corrections (mu_t2 +1/beta, n<=beta^{5/2}, pathwise mode=upcrossing refuted) documented
- ok Transformer/Modes/Section2_Field.lean (9): `hasDerivAt_fun_sum`, `hasDerivAt_bump`, `hasDerivAt_bump_deriv`, `hasDerivAt_kde`, `hasDerivAt_deriv_kde`, `fieldF_eq`, `isUpcrossing_fieldF_iff`, `upcrossingSet_subset_modeSet`, `upcrossingCount_le_modeCount` — matches §2 (eq:Fn, eq: Gt, thm:kac-rice items 1-4, eq:main-eq-form, lem:phi-t, lem:moments-p); corrections (mu_t2 +1/beta, n<=beta^{5/2}, pathwise mode=upcrossing refuted) documented
- ok Transformer/Modes/Section2_GaussianInt.lean (1): `integral_pow_mul_exp_neg_mul_sq` — matches §2 (eq:Fn, eq: Gt, thm:kac-rice items 1-4, eq:main-eq-form, lem:phi-t, lem:moments-p); corrections (mu_t2 +1/beta, n<=beta^{5/2}, pathwise mode=upcrossing refuted) documented
- ok Transformer/Modes/Section2_Gt.lean (5): `hasDerivAt_bigG`, `fieldF_eq_sum_bigG`, `hasDerivAt_fieldF`, `fieldF_pair_eq`, `beta_mul_bigG'_eq` — matches §2 (eq:Fn, eq: Gt, thm:kac-rice items 1-4, eq:main-eq-form, lem:phi-t, lem:moments-p); corrections (mu_t2 +1/beta, n<=beta^{5/2}, pathwise mode=upcrossing refuted) documented
- ok Transformer/Modes/Section2_KacRice.lean (1): `modulusOfContinuity_const` — matches §2 (eq:Fn, eq: Gt, thm:kac-rice items 1-4, eq:main-eq-form, lem:phi-t, lem:moments-p); corrections (mu_t2 +1/beta, n<=beta^{5/2}, pathwise mode=upcrossing refuted) documented
- ok Transformer/Modes/Section2_MainForm.lean (6): `isLittleO_sqrt_window`, `expectedUpcrossingsReal_eq`, `main_eq_form_T`, `main_eq_form_T'_isLittleO`, `main_eq_form_T'`, `main_eq_form_tail` — matches §2 (eq:Fn, eq: Gt, thm:kac-rice items 1-4, eq:main-eq-form, lem:phi-t, lem:moments-p); corrections (mu_t2 +1/beta, n<=beta^{5/2}, pathwise mode=upcrossing refuted) documented
- ok Transformer/Modes/Section2_MainIntPhi.lean (2): `sq_le_phiRate`, `integral_exp_phiRate_T'` — matches §2 (eq:Fn, eq: Gt, thm:kac-rice items 1-4, eq:main-eq-form, lem:phi-t, lem:moments-p); corrections (mu_t2 +1/beta, n<=beta^{5/2}, pathwise mode=upcrossing refuted) documented
- ok Transformer/Modes/Section2_MomentsP.lean (1): `tendsto_sq_div_of_mem_intervalT` — matches §2 (eq:Fn, eq: Gt, thm:kac-rice items 1-4, eq:main-eq-form, lem:phi-t, lem:moments-p); corrections (mu_t2 +1/beta, n<=beta^{5/2}, pathwise mode=upcrossing refuted) documented
- ok Transformer/Modes/Section2_PhiT.lean (2): `quadForm_complete_square`, `krQuad_zero_eq` — matches §2 (eq:Fn, eq: Gt, thm:kac-rice items 1-4, eq:main-eq-form, lem:phi-t, lem:moments-p); corrections (mu_t2 +1/beta, n<=beta^{5/2}, pathwise mode=upcrossing refuted) documented
- ok Transformer/Modes/Section2_RandomLine.lean (8): `deriv_randomLine`, `randomLine_zero`, `map_fst_gaussianPair`, `map_snd_gaussianPair`, `memLp_fst_gaussianPair`, `memLp_snd_gaussianPair`, `gaussianPair_eq_withDensity`, `continuous_gaussianPDFReal_std` — matches §2 (eq:Fn, eq: Gt, thm:kac-rice items 1-4, eq:main-eq-form, lem:phi-t, lem:moments-p); corrections (mu_t2 +1/beta, n<=beta^{5/2}, pathwise mode=upcrossing refuted) documented
- ok Transformer/Modes/Section3_BR.lean (2): `norm_charFun_stdGauss2`, `hasIntegrableCharFun_stdGauss2`
- ok Transformer/Modes/Section3_Cumulants.lean (4): `integral_sq_stdGaussian`, `isStandardized_stdGauss2`, `hasExpMoments_stdGauss2`, `isDensityOf_stdGauss2`
- ok Transformer/Modes/Section3_Edgeworth.lean (1): `measurable_singleY`
- ok Transformer/Modes/Section3_ErrorHigher.lean (5): `rate_base_eq`, `rate_eq`, `sq_le_of_mem_intervalT`, `rate_T`, `rate_T'` — lem:error-higher is stated uniformly on T (the paper fixes t); documented
- ok Transformer/Modes/Section3_ErrorKR.lean (2): `rpow_neg_half_nonneg`, `gThreeKR_Ioi_le`
- ok Transformer/Modes/Section3_ErrorThird.lean (3): `not_integrableOn_tildeY`, `eucl_whiten_bounds`, `abs_hermite_whiten_le`
- ok Transformer/Modes/Section3_Hermite.lean (3): `not_exists_hermite_le_cube`, `abs_hermite_le`, `eucl_whiten_sq`
- ok Transformer/Modes/Section4_ScaleSpace.lean (6): `kde_neg`, `isLocalMax_comp_neg_iff`, `modeCount_kde_Iio`, `modeCount_kde_Iio_le`, `eq_of_isLocalMax_kde_one`, `scale_space_one`
- ok Transformer/Modes/Section4_Tail.lean (7): `gaussianReal_Ici_le`, `gaussianReal_Iic_le`, `countIn_eq_sum`, `measurable_countIn`, `lintegral_countIn`, `two_mul_exp_eq`, `expectedModes_compl_le` — lem:scale-space carried as hypothesis hss
- ok Transformer/Modes/Section5_PtBdd.lean (8): `bigG_eq_gPt`, `hasDerivAt_gaussFactor`, `hasDerivAt_gPt`, `hasDerivAt_gPt1`, `hasDerivAt_gPt2`, `det_psi`, `det_psi_neg`, `phase_nondegenerate`
- ok Transformer/Modes/Section5_PtBddFourier.lean (5): `tendsto_pow_mul_gaussFactor`, `tendsto_bigG`, `tendsto_fourierNu`, `not_uniform_decay`, `lintegral_pow_lt_top_of_decay`
- ok Transformer/Modes/Section5_PtBddOne.lean (2): `volume_range_eq_zero`, `not_isDensityOf_one`

### Transformer.Normalization — arXiv:2510.22026

- issue Transformer/Normalization/ClusteringLine.lean (9): `proj_one_eq_zero`, `na_const_one`, `not_synchronizes_const`, `not_clusters_from_uniform_one`, `ne_zero_of_pos_neg`, `two_le_exp_one`, `pre_line_counter`, `not_clusters_or_stalls_from_gaussian_one`, `not_unconditional_synchronization_one` — refutations ok; but Clustering.lean's thm:convergence statements require SchemeDynamics for all t∈ℝ (paper t≥0): for Pre/Mix/Peri a backward-global solution may not exist (r hits 0), which can make them vacuous; β unconstrained (paper implicitly β>0) — FIXED: SchemeDynamics now holds for t ≥ 0 with one-sided derivatives (Ici 0); β left unconstrained: for β ≤ 0 the weights stay positive and attraction persists, no evidence the claim needs β > 0
- ok Transformer/Normalization/Convergence.lean (3): `hasDerivAt_energy`, `proj_smul`, `na_velocity_eq_energyGrad`
- ok Transformer/Normalization/Line.lean (10): `stdGaussian_one_singleton`, `stdGaussian_one_pos`, `sph0_coord`, `sq_coord_of_norm_one`, `sph0_cases`, `sph0_ne`, `uniform_sph0_pos`, `uniformSph0_invariant`, `uniformTuple_sph0`, `not_ae_gaussian_pair`
- ok Transformer/Normalization/Lojasiewicz.lean (1): `na_time_change`
- ok Transformer/Normalization/Radial.lean (3): `inner_attentionVec_self`, `inner_attentionVec_self_lower_bound`, `radialDerivative_pre_lower_bound`
- issue Transformer/Normalization/Rates.lean (1): `not_forall_initial_velocity_small` — refutation ok, initial_velocity_small faithful; clustering_rate likely false as written: Θ-bounds claimed for every t>0 with constants independent of r(0) (paper: asymptotic; at t→0 Var/t→∞ while Pre-LN derivative ≈ -Var/r(0)); no ‖θ_j(0)‖=1; dynamics on all ℝ — FIXED: moved to Normalization/ClusterSpeed.lean; `not_clustering_rate` refutes the source's hypotheses (β=1/100 admits δ=5/2, antipodal Post-LN pair is stationary); corrected statement adds δ<1/(100n²), ‖θ_j(0)‖=1, r_j(0)>0, α>0, Θ from some T₀ on, nGPT scale α⁻¹ (source typo); dynamics now on t ≥ 0
- issue Transformer/Normalization/Symmetric.lean (6): `partition_symmetricInit`, `inner_attentionVec_symmetricInit`, `hasDerivAt_similarity_symmetricInit`, `thm_symmetric_post`, `thm_symmetric_pre`, `returns` — proved rows faithful; thm:symmetric only t→0 for Post/Pre: Mix/Peri/nGPT/CoD rows, whole t→∞ column and 'γ constant across pairs' are off the books
- ok Transformer/Normalization/UnstableProduct.lean (1): `unstable_mul_of_posDef`
- issue Transformer/Normalization/Velocities.lean (2): `norm_attentionVec_le_one`, `radialDerivative_pre_ge_of_localCone` — proved lemmas ok; thm:preln-slow (i) r_k(t)≥(1-δ)t (Pre and Peri, all t) is off the books — only the pointwise Pre velocity bound; stale docstrings say (ii) 'not formalized' though Rates states it

### Transformer.Perceptron — arXiv:2601.21366

- ok Transformer/Perceptron/Analytic.lean (6): `norm_greatCircle`, `inner_greatCircle`, `analyticOnNhd_greatCircle`, `hasDerivAt_reluSq`, `analyticOnNhd_potential_greatCircle`, `not_isAnalyticOnSphere_relu`
- ok Transformer/Perceptron/Atomicity.lean (4): `isFinitelyAtomic_diracProb`, `norm_secondAxis`, `norm_basePoint_one`, `inner_basePoint_secondAxis`
- ok Transformer/Perceptron/Atoms.lean (11): `coe_atomicProb`, `isFinitelyAtomic_atomicProb`, `coe_circlePoint`, `inner_basePoint_circlePoint`, `hasDerivAt_circlePoint`, `hasDerivAt_circleVel`, `inner_circlePoint`, `inner_circleVel`, `hasDerivAt_inner_circlePoint`, `hasDerivAt_inner_circleVel`, `isAtomicOnCircle_atomicProb`
- ok Transformer/Perceptron/Basic.lean (5): `drift_zero`, `hasGradientAt_potential`, `proj_gradient_potential`, `isStationary_diracProb_of_radial`, `isStationary_diracProb`
- ok Transformer/Perceptron/Bias.lean (7): `biasedPotential_zero`, `biasedDrift_zero`, `biasedEnergyGrad_zero`, `isBiasedStationary_zero_iff`, `isBiasedAnalyticOnSphere_zero_iff`, `hasGradientAt_biasedPotential`, `proj_gradient_biasedPotential`
- ok Transformer/Perceptron/BiasedAtomicity.lean (3): `hasTransverseHyperplane_zero`, `basePoint_ne_zero`, `single_neg_basePoint_ne_zero`
- ok Transformer/Perceptron/CircleDeriv.lean (3): `hasDerivAt_potential_circlePoint`, `deriv_potential_circlePoint`, `secondDeriv_potential_circlePoint`
- ok Transformer/Perceptron/Dirac.lean (8): `interactionEnergy_of_dirac`, `energy_of_dirac`, `potential_pin`, `drift_pin`, `isStationary_pin`, `isStationary_relu_pin`, `isStationary_linear_pin`, `isStrictSOPD_pin`
- ok Transformer/Perceptron/GeneralAtomicity.lean (1): `isSymmetric_bijective_id`
- ok Transformer/Perceptron/GeneralAttention.lean (11): `potential_smul`, `proj_smul_right`, `drift_smul`, `interactionEnergyMap_smul_id`, `energyMap_smul_id`, `energyGradMap_smul_id`, `isStationaryMap_smul_id_iff`, `energyMap_id_eq`, `isStationaryMap_id_iff`, `isStrictSOPDMap_id_iff`, `isSOPDMap_id_iff`
- ok Transformer/Perceptron/Geodesic.lean (6): `sphereExp_zero`, `norm_sphereExp`, `inner_sphereExp_smul`, `isGradientField_zero`, `inner_isGradientField`, `IsSOPD_of_IsStrictSOPD`
- ok Transformer/Perceptron/HigherDim.lean (1): `mutuallySingular_of_measure_support_eq_zero`
- ok Transformer/Perceptron/Hyperplane.lean (3): `exists_unit_inner_eq_zero`, `smul_eq_of_inner_eq_norm`, `not_subsingleton_sphereHyperplane_iff`
- issue Transformer/Perceptron/Kernel.lean (13): `hasDerivAt_kernelK`, `deriv_kernelK`, `hasDerivAt_deriv_kernelK`, `deriv2_kernelK`, `cosArg_mem_Ioo`, `cos_thetaC`, `thetaC_pos`, `thetaC_lt_pi_div_two`, `thetaC_le_pi`, `quadratic_cos_thetaC`, `kernelK2_neg`, `strictConcaveOn_kernelK`, `tendsto_thetaC_nhdsWithin_zero` — proved lemmas ok; Bound.lean (sorried): bound_cluster_mass and bound_atom_count choose C in O(e^{-β}) before the weights ϑ, but the paper's proof gives a constant that depends on C_ϑ = 2Σ|ω_j|‖a_j‖², so the statement is stronger than the paper and possibly false; fix: ∀ ω a, ∃ C β₀. FIXED: both now quantify `∀ ω a, ∃ C β₀`, with the dependence documented
- ok Transformer/Perceptron/KernelSup.lean (4): `two_mul_le_exp`, `abs_kernelK2_le`, `abs_kernelK2_zero`, `isGreatest_abs_kernelK2`
- ok Transformer/Perceptron/MinMax.lean (6): `continuous_of_hasDerivAt`, `continuous_potential`, `integrable_potential`, `energy_diracProb`, `isMaxEnergy_diracProb_of_isMaxOn`, `exists_isMaxOn_potential_of_isMaxEnergy`
- ok Transformer/Perceptron/Normalized.lean (8): `energyGrad_eq_attentionGrad_add_drift`, `attentionWeight_pos`, `proj_integral_expInner_smul`, `proj_gradient_attentionWeight`, `proj_inv_smul_gradient_attentionWeight`, `attentionGrad_diracProb`, `isNormalizedStationary_diracProb_of_radial`, `isNormalizedStationary_relu_pin`
- ok Transformer/Perceptron/NormalizedMap.lean (7): `energyGradMap_eq_attentionGradMap_add_drift`, `attentionWeightMap_smul_id`, `attentionGradMap_smul_id`, `normalizedField_smul_id`, `isNormalizedStationaryMap_smul_id_iff`, `isNormalizedStationaryMap_id_iff`, `isNormalizedStationaryMap_relu_pin`
- ok Transformer/Perceptron/Piecewise.lean (2): `isPiecewisePolynomial_relu`, `lipschitzWith_relu`
- ok Transformer/Perceptron/SignedGram.lean (4): `sum_single_neuron`, `signedGram_single`, `isNonDegenerate_single`, `isNonDegenerate_pin`
- ok Transformer/Perceptron/StrictSOPD.lean (1): `secondDeriv_ge_iff_simplified`
- ok Transformer/Perceptron/Transform.lean (9): `attentionTransformMap_smul_id`, `coe_antipodeMap`, `measurable_antipodeMap`, `coe_antipode`, `attentionTransformMap_antipode`, `attentionTransform_antipode`, `even_attentionTransformMap_iff`, `even_attentionTransform_iff`, `isPolyOfDegreeLE_const`

### Transformer.Perspective — arXiv:2312.10794

- ok Transformer/Perspective/AppendixA_Beta0.lean (8): `hasDerivAt_E0`, `inner_proj_self`, `ne_neg_self_of_norm_eq_one`, `proj_smul`, `beta0Dynamics_smul`, `hasDerivAt_E0_ascent`, `taylor_eq`, `antipodalPair_critical_nonTrivial` — eq:taylor as in Step 1 (nontrivial = not all equal); e:gradfl written out
- ok Transformer/Perspective/AppendixA_Hessian.lean (1): `hessian_at_critical` — e:helpcl; criticality dropped (unused), recorded in docstring; ODE form of e^{tB}
- ok Transformer/Perspective/AppendixA_Rotation.lean (4): `hasDerivAt_expSkew`, `expSkew_zero`, `norm_expSkew`, `exists_perturbationBy` — existence of e^{tB} curve
- ok Transformer/Perspective/AppendixA_Saddle.lean (2): `with`, `yury_lemma` — yury_lemma: strict saddle as positive 2nd derivative along a rotation; d>=2 added and recorded (false at d=1)
- ok Transformer/Perspective/AppendixB_BetaInterval.lean (6): `sq_sqrt_disc`, `cosTauStar_pos`, `cosTauStar_le_one`, `cosTauStar_quadratic`, `τ_β_star_spec`, `τ_β_star_unique` — checked vs App. B; hessian hypothesis along block rotations is implied by the paper's at a critical point
- ok Transformer/Perspective/AppendixB_ClaimYury.lean (1): `claim_yury` — checked vs App. B; hessian hypothesis along block rotations is implied by the paper's at a critical point
- ok Transformer/Perspective/AppendixB_EBeta.lean (2): `selfEnergy_one`, `singleToken_isSkew_critical_hessianNonPos` — checked vs App. B; hessian hypothesis along block rotations is implied by the paper's at a critical point
- ok Transformer/Perspective/AppendixB_Expansion.lean (1): `selfEnergy_expansion_aux` — App. B computations; beta!=0 added, criticality dropped, both recorded
- ok Transformer/Perspective/AppendixB_HessBeta.lean (1): `secondDeriv_selfEnergy` — App. B computations; beta!=0 added, criticality dropped, both recorded
- ok Transformer/Perspective/AppendixB_HighD.lean (1): `dr1_skew_inequality` — App. B computations; beta!=0 added, criticality dropped, both recorded
- ok Transformer/Perspective/AppendixB_Intrinsic.lean (2): `isLittleO_selfEnergy_secondOrder`, `hessian_at_critical_intrinsic` — App. B; O(beta) with explicit uniform constant; skew dropped in metric.hess (unused), recorded
- ok Transformer/Perspective/AppendixB_MetricGrad.lean (3): `hasDerivAt_selfEnergy`, `symmetrized_double_sum`, `metric_grad_comparison` — App. B; O(beta) with explicit uniform constant; skew dropped in metric.hess (unused), recorded
- ok Transformer/Perspective/AppendixB_MetricHess.lean (1): `metric_hess_comparison` — App. B; O(beta) with explicit uniform constant; skew dropped in metric.hess (unused), recorded
- ok Transformer/Perspective/AppendixB_Taylor.lean (6): `hasDerivAt_expCosLine`, `hasDerivAt_expCosLine_deriv`, `g_β_2d_symm`, `secondDeriv_torusEnergy_block`, `taylor2_inequality`, `taylor3_inequality` — App. B; O(beta) with explicit uniform constant; skew dropped in metric.hess (unused), recorded
- ok Transformer/Perspective/AppendixC_BetaTiny.lean (3): `Etilde_eq_Etilde0_add`, `tendsto_Etilde_zero`, `eigvalBetaConst_pos` — honest names; eigval.beta bound itself (a proof step) off the books, noted in module doc
- issue Transformer/Perspective/AppendixD_Alpha.lean (2): `alpha_at_one_over_n`, `not_alpha_at_one_over_n_of_free` — alpha_at_one_over_n: x* chosen as x_0(1/n), not the paper's limit x*; faithful form: x* with eta x* in hull of X(1/n)
- issue Transformer/Perspective/AppendixD_AlphaDeriv.lean (2): `diff_ineq_alpha`, `not_forall_diff_ineq_alpha` — diff_ineq_alpha: hhull near-vacuous (hull vs cone), see not_forall_diff_ineq_alpha
- ok Transformer/Perspective/AppendixD_Assembly.lean (2): `it`, `ineq_second_part` — deduction with hcp/hyb/hγle as hypotheses; constant 4 instead of 1, justified; valid for any x_star
- ok Transformer/Perspective/AppendixD_PhaseTransition.lean (3): `ineq_first_part`, `exists_le_div_log`, `d_star_definition` — ineqfirstpart with c(β)=e^{10max(1,β)} as paper, β≥0 added; d.large stronger (hyp dropped)
- issue Transformer/Perspective/AppendixD_Product.lean (3): `isMinInner_const_consensus`, `product_close_to_one`, `not_forall_product_close_to_one` — product_close_to_one ok as a Gronwall deduction, but not_forall_product_close_to_one is a strawman claiming to refute the survey
- ok Transformer/Perspective/AppendixD_Stability.lean (3): `one_le_cBeta`, `stability_orthogonal`, `shortdist_bound` — stability.4ortho with paper's c(β); β≥0 added and justified
- issue Transformer/Perspective/AppendixD_Ybeta.lean (2): `ybeta_close_to_1`, `not_forall_ybeta_close_to_1` — hinv ([0,1]-invariance) and hhalf (γ(ne^β/2)≥1/2) are consequences of eq:ybeta with γ(0)=0 — the paper derives hhalf in the proof; carried as hypotheses they weaken e:ybetacloseto1; prove them
- issue Transformer/Perspective/AppendixD_YbetaUSA.lean (2): `usa_analogue`, `not_forall_usa_analogue` — hinv is a consequence of eq:ybetaUSA with γ(0)=0; carried as hypothesis it weakens rem:usa.d; prove it
- ok Transformer/Perspective/Beta0Field.lean (4): `beta0Dynamics_iff`, `norm_meanTuple_le`, `meanTuple_sub`, `lipschitzOnWith_beta0Field` — approxsphere with uniform C hoisted (stronger); helpers
- ok Transformer/Perspective/Beta0Gronwall.lean (3): `proj_sub`, `norm_SA_drift_sub_beta0Field_le`, `solutions_close_at_small_beta` — approxsphere with uniform C hoisted (stronger); helpers
- ok Transformer/Perspective/DoubleSum.lean (4): `hasDerivAt_double_sum`, `const_mul_double_sum`, `double_sum_sub`, `abs_double_sum_le` — approxsphere with uniform C hoisted (stronger); helpers
- ok Transformer/Perspective/Gronwall.lean (1): `decay_of_deriv_ge` — approxsphere with uniform C hoisted (stronger); helpers
- ok Transformer/Perspective/InnerAsymptotics.lean (10): `isBigO_inner`, `isLittleO_inner`, `isBigO_inner_const_left`, `isBigO_inner_const_right`, `isLittleO_inner_const_left`, `isLittleO_inner_const_right`, `isLittleO_cube_sq`, `isLittleO_sq_id`, `isBigO_mul_const`, `isBigO_smul_const` — analytic helpers; MinCurve one-sided form justified
- ok Transformer/Perspective/MinCurve.lean (1): `le_min_curve_of_deriv_nonneg` — analytic helpers; MinCurve one-sided form justified
- ok Transformer/Perspective/PartitionGradient.lean (8): `hasFDerivAt_expInner`, `abs_inner_le_of_dist_le`, `integrable_expInner_ambient`, `integrable_expInner_smul`, `hasGradientAt_partitionMu`, `partitionMu_pos`, `gradient_partitionMu`, `gradient_log_partitionMu` — analytic helpers; MinCurve one-sided form justified
- ok Transformer/Perspective/PeanoTaylor.lean (2): `isLittleO_secondOrder`, `eq_zero_of_isLittleO_pow` — analytic helpers; MinCurve one-sided form justified
- ok Transformer/Perspective/RussianPairs.lean (2): `sum_sq_cross`, `russian_trick_pairs` — pair family, no hypothesis
- issue Transformer/Perspective/RussianTrick.lean (6): `skewPair_apply`, `skewPair_isSkew`, `skewPair_sq`, `sum_skewPair_sq`, `russian_trick`, `not_russian_trick_one` — russian_trick ok (d>=2 instead of Odd d, stronger); module doc and not_russian_trick_one overclaim a refutation resting on 0⁻¹=0 (see §1)
- ok Transformer/Perspective/SAField.lean (3): `SA_iff`, `norm_saAvg_le`, `saField_of_one`
- ok Transformer/Perspective/SALipschitz.lean (1): `lipschitzOnWith_saField` — lip.1-3 with β≥0 added and justified
- ok Transformer/Perspective/Section1_IPS.lean (3): `SA_const_consensus`, `partitionSA_bounds`, `SA_permutation_equivariant` — defs match §2-3; maximiser half as paper
- ok Transformer/Perspective/Section2_EnergyKernel.lean (10): `inner_sphere_le_one`, `eq_of_inner_sphere_eq_one`, `exp_inner_le`, `continuous_expInner_right`, `continuous_expInner_left`, `integrable_expInner`, `partitionMu_nonneg`, `partitionMu_le`, `continuous_partitionMu`, `integrable_partitionMu` — defs match §2-3; maximiser half as paper
- ok Transformer/Perspective/Section2_EnergyMax.lean (6): `interactionEnergy_eq_partition`, `interactionEnergy_le`, `interactionEnergy_diracProb`, `isMaxEnergy_diracProb`, `eq_dirac_of_ae_eq`, `exists_eq_dirac_of_isMaxEnergy` — defs match §2-3; maximiser half as paper
- issue Transformer/Perspective/Section2_FlowMap.lean (3): `continuityEquation_eq_auxCE`, `vectorField_diracProb_self`, `continuityEquation_const_diracProb` — dissipation_softmax lacks β>0: at β=0 E≡0 but RHS≠0; min half faithful (σ_d via O(d)-invariance)
- issue Transformer/Perspective/Section2_GradientFlow.lean (6): `vectorField_eq_grad_log`, `usaVectorField_eq_grad_first_variation`, `usaVectorField_diracProb_self`, `usaContinuityEquation_const_diracProb`, `sa_is_gradient_flow`, `transformerODE_const_one` — usa_dissipation lacks β>0 (same as dissipation_softmax); logder, XmuE, sa_is_gradient_flow faithful
- ok Transformer/Perspective/Section2_ParticleFlow.lean (2): `hasDerivAt_fun_sum`, `usa_isGradientFlow` — e:dynonX, β≠0 included
- issue Transformer/Perspective/Section3_Gronwall.lean (3): `const_of_SA_one`, `const_of_beta0Dynamics_one`, `distance_bound_at_time_m` — distance_bound_at_time_m carries hclose, already proved as solutions_close_at_small_beta (imported): drop C/hC/hclose
- issue Transformer/Perspective/Section3_SmallBeta.lean (3): `antipodalPair_not_mem_consensusSet0`, `SA_const_antipodalPair`, `antipodalPair_not_mem_clusteringSet` — statements faithful; omission: the USA half of thm:beta.tiny ('resp. USA') is off the books
- issue Transformer/Perspective/Section5_ConeCollapse.lean (1): `hemisphere_clustering` — hemisphere_clustering keeps the lemma's name but carries steps 1-2 (the whole content) as hstep and drops the hemisphere hypothesis; the actual lemma is off the books
- ok Transformer/Perspective/Section5_Exceptional.lean (1): `antipodalPair_not_exponential` — antipodal pair: no exponential rate, proved
- issue Transformer/Perspective/Section5_Hemisphere.lean (5): `step1_deriv_nonneg`, `hemisphere_step1_monotone`, `exists_inner_le_of_mem_convexHull`, `step2_decomposition`, `not_step2_decomposition` — step2_decomposition: hhull too strong (hull instead of cone); exists_inner_le_of_mem_convexHull inherits it — check e:mineqalpha usage
- issue Transformer/Perspective/Section5_HighD.lean (2): `step1_rhs`, `not_forall_step2_alpha_diff_ineq` — hemisphere_step1_qual_conv lacks β>0 (paper lem: hemisphere.clustering has β>0); USA/QKV halves of thm: boumal, thm: d.infty off the books; boumal, d.infty, wendel faithful
- issue Transformer/Perspective/Section5_HighDCurve.lean (4): `ybetaODE_SA_one_zero`, `hasDerivAt_tanh`, `ybetaODE_SA_two_zero`, `ybetaODE_USA_two_zero` — FIXED: phase_transition_curve now reads the probability over UniformTuple, C,λ before d. Was FALSE AS WRITTEN: phase_transition_curve quantifies ∀ X₀, the probability ≥ 1-2n²d^{-1/64} is only a comment; X₀=(p,p) constant consensus gives |1-γ(0)|=1 > 2√(log d/d). Also C,λ must not depend on d. orthogonal_initial faithful
- ok Transformer/Perspective/Section5_InvariantMeasure.lean (1): `no_smooth_invariant_measure` — boumal and existence carried as hypotheses (true, not exhibitable)
- ok Transformer/Perspective/Section5_Vanishing.lean (1): `ez_lemma` — lem: ez.lemma faithful
- ok Transformer/Perspective/Section6_Circle.lean (4): `hasDerivAt_torusEnergy`, `angularUSA_is_gradient_flow`, `h_β_neg`, `h_β_pos` — eq:onangles gradient flow, defs faithful
- ok Transformer/Perspective/Section8_CohnKumar.lean (7): `inner_antipode`, `ne_antipode`, `discreteEnergy_pair`, `discreteEnergy_antipodal_min`, `not_sharpConfiguration_antipodal`, `exists_orthogonal_pair`, `not_cohn_kumar_dichotomy` — refutation checked in §1; defs as printed
- issue Transformer/Perspective/Section8_General.lean (5): `exp_inner_eq_exp_sqDist`, `squaredDistEnergy_eq_interactionEnergy`, `discreteEnergy_eq_sqDist`, `exp_sqDist_strictAnti`, `discreteEnergy_singleton` — rescaledEquation (e:Rres) drops e^{tV}: weights are ⟨Q e^{tV} z_i, K e^{tV} z_j⟩ in the paper; the def is a copy of preLimitODE (unused, no theorem affected). Other defs, kernel rewriting faithful
- ok Transformer/Perspective/Softmax.lean (4): `softmaxPartition_pos`, `softmaxWeight_nonneg`, `sum_softmaxWeight`, `sum_abs_softmaxWeight_sub_le` — faithful / helper
- ok Transformer/Perspective/SphereInvariant.lean (9): `exists_sphereMap_apply_eq`, `continuous_sphereMap`, `measurable_sphereMap`, `injective_sphereMap`, `measure_singleton_eq_of_invariant`, `norm_spherePt`, `inner_spherePt`, `infinite_sSphere`, `measure_singleton_eq_zero_of_invariant` — faithful / helper
- ok Transformer/Perspective/StrictSaddle.lean (1): `not_torusHessianNonPos_of_strictSaddle` — faithful / helper
- ok Transformer/Perspective/UniformAtomless.lean (3): `map_pair_pi`, `measure_coords_eq_eq_zero`, `measure_singleton_tuple_eq_zero` — faithful / helper

### Transformer.Precision — arXiv:2410.01104

- ok Transformer/Precision/Accumulate.lean (6): `HasSignificand.dvd`, `HasSignificand.gap`, `hasSignificand_one_pow_two`, `accum_stall`, `accum_le`, `pairwise_exact`
- ok Transformer/Precision/Basic.lean (6): `softmax_denom_pos`, `softmax_nonneg`, `softmax_le`, `le_softmax`, `quantize_eq_zero`, `quantize_pos`
- ok Transformer/Precision/Blind.lean (2): `IsNearest.exists_const`, `qAttn_blind`
- ok Transformer/Precision/BlockScale.lean (5): `IsNearest.le_of_eq_zero`, `blockScale_eq_zero`, `blockScale_dropped_le`, `nvfp4_eq_zero`, `nvfp4_dropped_le`
- ok Transformer/Precision/ContextLength.lean (7): `qAttn_eq_zero`, `qAttn_update_injective`, `quantize_bits_eq_zero`, `qAttn_bits_eq_zero`, `qAttn_bits_injective`, `exists_length_qAttn_eq_zero`, `exists_scores_qAttn_injective`
- ok Transformer/Precision/Float.lean (6): `ieee_eq_zero`, `exists_isNearest_grid`, `qAttn_ieee_eq_zero`, `qAttn_f16_eq_zero`, `qAttn_e4m3_eq_zero`, `qAttn_e5m2_eq_zero`
- ok Transformer/Precision/FloatSum.lean (6): `accum_softmax_le`, `accum_softmax_f32_le`, `accum_softmax_f16_le`, `accum_softmax_bf16_le`, `accum_softmax_e4m3_le`, `accum_softmax_e5m2_le`
- ok Transformer/Precision/IEEE.lean (14): `minSub_pos`, `minSub_le_abs`, `minSub_f16`, `minSub_bf16`, `minSub_f32`, `minSub_e4m3`, `minSub_e5m2`, `minSub_e2m1`, `f16_one`, `f16_max`, `zero_mem_grid`, `grid_finite`, `ieee_abs`, `grid_hasSignificand`
- ok Transformer/Precision/Nearest.lean (4): `IsNearest.eq_zero`, `IsNearest.eq_self`, `exists_isNearest`, `IsNearest.add_eq`
- ok Transformer/Precision/Tail.lean (4): `tail_eq`, `exp_neg_eight_lt`, `tail_e4m3`, `tail_e4m3_131072`

### Transformer.Quartet — arXiv:2601.22813

- ok Transformer/Quartet/AppendixA_Concentration.lean (3): `meanSqErr_of_seedMean_eq`, `meanSqErr_eq`, `tendsto_meanSqErr` — exact bias-variance identity; the paper's ~1/B and plateau
- ok Transformer/Quartet/Fp8Grid.lean (2): `mem_fp8`, `le_rtn_fp8` — normal-range hypothesis 2^-6 ≤ v added, deviation documented with the subnormal counterexample
- issue Transformer/Quartet/Hadamard.lean (7): `hadamard_eq`, `hadamard_symm`, `sum_hadamard_mul`, `sum_hadamard_sum_hadamard`, `rhtInv_rht`, `sum_hadamard_mul_sum_hadamard_mul`, `sum_rht_mul_rht` — sorried Section3_Eden.mean_rhtInv_msEden (Corollary) is faithful to the paper but likely false at finite d: EDEN Thm 2.1 is exact for Haar rotations, not for RHT; the docstring's 'only as d→∞' is imprecise. Candidate for a refutation. FIXED: false as stated over `s ≠ 0` — at s = 21 the corrected E4M3 scale saturates at 448 and the mean is x/2 (`not_mean_rhtInv_msEden`, Section3_EdenFalse); restated for 6·16/17 ≤ s ≤ 6·16/17/0.93. 'd→∞' is §3.2's own reading of EDEN for RHT; finite d stays open
- ok Transformer/Quartet/Section3_Grids.lean (6): `floorOn_mem_le_and_le_ceilOn_mem`, `floorOn_eq`, `ceilOn_eq`, `rtn_mem`, `sr_mem`, `integral_sr` — fp4/fp8 grids, RTN ties-down, SR as in §3.1. FIXED: `floorOn`/`ceilOn` returned `sSup ∅ = 0` beyond the grid, so `rtn fp4 7 = 0` where §3.3's clipping `Q_RTN` gives `6`; they now saturate (`rtn_of_forall_le`), and `integral_sr` moved to Section3_Unbiased
- ok Transformer/Quartet/Section3_NVFP4.lean (4): `groupAbsMax_le_absMax`, `abs_le_groupAbsMax`, `groupAbsMax_nonneg`, `groupScaleRTN_le` — scale displays match §3.1 (l.240) and §3.3 (l.331)
- ok Transformer/Quartet/Section3_NonClipping.lean (3): `normal_groupScale_arg`, `abs_div_groupScaleSR_le`, `abs_div_groupScaleRTN_le` — hypothesis max|x| ≤ 2^14·max_g|x| added (the paper omits it), documented
- ok Transformer/Quartet/Section3_Unbiased.lean (2): `groupScaleSR_pos`, `integral_qSRAt` — §3.1 unbiasedness, with the same documented hypothesis
- ok Transformer/Quartet/Section4_Bias.lean (10): `measurableSet_coinsUp`, `measurableSet_coinsDown`, `volume_coinsUp`, `volume_coinsDown`, `coinCube_eq_union`, `disjoint_coinsUp_coinsDown`, `eqOn_coinsUp`, `eqOn_coinsDown`, `meanGroup_q46At_biasWitness`, `exists_mean_q46At_ne` — refutation of unbiasedness for SR+4/6 on a model: the selection reads the realized SR errors, c enters both scales (effective block scale gmax/c)
- ok Transformer/Quartet/Section4_FourOverSix.lean (1): `integral_qSRAt_four_and_six` — model of Cook et al. with SR, per §4.2 wording
- ok Transformer/Quartet/Section4_Rounding.lean (5): `qSRAt_biasWitness_ne`, `qSRAt_biasWitness_four`, `qSRAt_biasWitness_six`, `groupErr_biasWitness`, `q46At_biasWitness` — computations on the witness
- ok Transformer/Quartet/Section4_Witness.lean (11): `floorOn_fp4_zero`, `ceilOn_fp4_zero`, `floorOn_fp4_64_17`, `ceilOn_fp4_64_17`, `floorOn_fp4_96_17`, `ceilOn_fp4_96_17`, `rtn_fp8_448`, `absMax_biasWitness`, `groupAbsMax_biasWitness`, `tensorScaleSR_biasWitness`, `groupScaleSR_biasWitness` — computations on the witness
- ok Transformer/Quartet/SeedSums.lean (4): `sum_pi_succ`, `card_pi`, `sum_pi_sum_eq_zero`, `sum_pi_sum_sq` — combinatorial support
- ok Transformer/Quartet/Walsh.lean (5): `walsh_comm`, `walsh_congr_right`, `walsh_succ`, `sum_walsh_mul_range`, `sum_walsh_mul_fin` — Walsh orthogonality

### Transformer.RASP — arXiv:2106.06981

- ok Transformer/RASP/Basic.lean (12): `mem_selected`, `selected_selectAll`, `card_selected_selectAll`, `selected_selectZero`, `selectorWidth_selectAll`, `aggregate_selectAll`, `aggregate_of_selected_eq_singleton`, `aggregateOne_of_selected_eq_singleton`, `aggregate_of_selected_eq_empty`, `sum_light0`, `length_eq_one_div_aggregate`, `selectorWidth_sameToken`
- ok Transformer/RASP/Compilation.lean (5): `layers_lt_agg_value`, `layers_lt_agg_select`, `layers_le_heads`, `eval_eq_of_heads_eq_zero`, `not_reverse_of_heads_eq_zero`
- ok Transformer/RASP/Programs.lean (5): `selected_flip`, `reverse_apply`, `reverse_reverse`, `fracVal_eq`, `fracVal_mem_Icc`
- ok Transformer/RASP/SelectorWidth.lean (8): `selected_or0`, `selected_and0`, `card_selected_or0`, `sum_light0_or0`, `or0Width_eq`, `and0Width_eq`, `noBosRes_eq_selectorWidth`, `bosRes_eq`
- ok Transformer/RASP/Sort.lean (12): `rankOf_lt_rankOf`, `rankOf_lt`, `rank_lt_iff`, `rank_injective`, `rank_rankEquiv_symm`, `strictMono_key_rankEquiv_symm`, `lexKey_injective`, `mem_selected_smaller`, `selectorWidth_smaller`, `selected_selNew`, `sortProg_apply`, `sortProg_keys_monotone`

### Transformer.RASPL — arXiv:2310.16028

- ok Transformer/RASPL/Attention.lean (4): `sum_oneHot_mul`, `Constructable.add`, `constructable_value`, `argmax_shift`
- ok Transformer/RASPL/Conjecture.lean (2): `Realizable.of_simple`, `not_forall_raspGeneralizationConjecture`
- ok Transformer/RASPL/Defs.lean (5): `mem_selected_select`, `le_of_mem_selected_select`, `self_mem_selected_select`, `selWidth_eq`, `aggrMax_of_mem`
- ok Transformer/RASPL/Degree.lean (4): `dependsOn_iff`, `levelWeight_split`, `levelWeight_restrict`, `degP_restrict_lt`
- ok Transformer/RASPL/Fourier.lean (15): `bitSign_false`, `bitSign_true`, `bitSign_mul_self`, `one_add_bitSign_mul`, `chi_empty`, `chi_mul_self`, `chi_mul_chi`, `sum_chi`, `sum_chi_mul_chi`, `sum_coeff_mul_chi`, `coeff_chi`, `coeff_add`, `coeff_smul`, `coeff_sum`, `eq_of_coeff_eq`
- ok Transformer/RASPL/Gotu.lean (7): `andProg_apply`, `andProg_last`, `minDeg_ne_andAll`, `filter_card_zero`, `coeff_const_one`, `levelWeight_const_one_zero`, `levelWeight_const_one_of_ne`
- ok Transformer/RASPL/MinDegree.lean (5): `minDeg_not_dependsOn`, `apply_update_of_not_dependsOn`, `eq_of_forall_not_dependsOn`, `minDeg_eq_of_constant`, `not_degPLt_zero`
- ok Transformer/RASPL/Restrict.lean (6): `chi_update_of_notMem`, `chi_insert`, `sum_subsets_split`, `restrict_eq`, `coeff_restrict_of_mem`, `coeff_restrict_of_notMem`

### Transformer.Wasserstein — —

- ok Transformer/Wasserstein/Basic.lean (9): `transportCosts_nonneg`, `bddBelow_transportCosts`, `W2_nonneg`, `W2_le_of_coupling`, `transportCosts_nonempty`, `ae_fst_mem`, `ae_snd_mem`, `W2_self`, `norm_integral_sub_le_W2` — W2 as inf over couplings; Lipschitz bound checked
- ok Transformer/Wasserstein/LowerBound.lean (1): `measureReal_mul_sq_le_W2_sq`
- ok Transformer/Wasserstein/Collapse.lean (1): `W2_sq_le_collapse`

### Transformer.XSA — arXiv:2603.09078

- ok Transformer/XSA.lean (2): `xsa_output_orthogonal_to_value`, `xsa_equals_spherical_SA_when_V_is_identity` — eq:sa and eq:xsa match 2603.09078 §2-3
