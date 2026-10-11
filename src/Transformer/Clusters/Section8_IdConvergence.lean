/-
# Identity-value clustering toward the limiting polytope

ArXiv:2305.05465v6, §8.1, `t:Idcase11` and `e:Ax15`.
Every token converges to a candidate of the future-hull intersection.
The hull of these limits is exactly that intersection, so every extreme
point is one of the token limits and is a candidate as well.

The proof uses the actual rescaled ODE throughout. Integrable softmax
regrets, bounded speed, and finiteness of the cluster set supply token
convergence; no quantitative candidate-distance claim is assumed.
The full paper theorem retains its original hypotheses and conclusion.
-/

import Transformer.Clusters.Section8_ClusterLimits

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d n : ℕ}

/-- The hull of the limiting token tuple is the intersection of all future
hulls. Source: arXiv:2305.05465v6, §8.1, `t:Idcase11` and `p:noninc`.
Closedness passes nesting to the limit; convex combinations of the tokens
at integer times approximate any point of the intersection. -/
theorem IdRescaledDynamics.tokenHull_limit_eq_future (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z)
    (Y : Idx n → EucSpace d) (hlim : Tendsto Z atTop (nhds Y)) :
    tokenHull Y = {x | ∀ t : ℝ, 0 ≤ t → x ∈ tokenHull (Z t)} := by
  have hsub : ∀ t : ℝ, tokenHull Y ⊆ tokenHull (Z t) := by
    intro t
    apply tokenHull_subset_of_tendsto Z Y (tokenHull (Z t))
      (isCompact_tokenHull _).isClosed (convex_convexHull ℝ _) hlim
    filter_upwards [eventually_ge_atTop t] with s hs
    intro i
    exact tokenHull_antitone A Z hZ hs (mem_tokenHull _ i)
  have hlimNat : Tendsto (fun k : ℕ => Z (k : ℝ)) atTop (nhds Y) :=
    hlim.comp tendsto_natCast_atTop_atTop
  have hnorm : Tendsto (fun k : ℕ => ‖Z (k : ℝ) - Y‖) atTop (nhds 0) := by
    simpa only [sub_self, norm_zero] using
      (hlimNat.sub (tendsto_const_nhds (x := Y))).norm
  ext x
  constructor
  · exact fun hx t _ => hsub t hx
  · intro hx
    have hnear (k : ℕ) : ∃ y ∈ tokenHull Y, ‖x - y‖ ≤ ‖Z (k : ℝ) - Y‖ :=
      exists_tokenHull_norm_sub_le _ _ _
        (fun i => norm_le_pi_norm (Z (k : ℝ) - Y) i) (hx k (Nat.cast_nonneg k))
    choose y hy hxy using hnear
    have hy_lim : Tendsto y atTop (nhds x) := by
      apply Metric.tendsto_nhds.mpr
      intro ε hε
      filter_upwards [hnorm.eventually_lt_const hε] with k hk
      rw [dist_eq_norm, norm_sub_rev]
      exact (hxy k).trans_lt hk
    exact (isCompact_tokenHull Y).isClosed.mem_of_tendsto hy_lim
      (Eventually.of_forall hy)

/-- Both the ODE and the tuple convergence hold for two stationary tokens.
Source: arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
example (z : EucSpace 1) : IdRescaledDynamics (1 : ParamMatrix 1)
    (fun _ (_ : Idx 2) => z) ∧
    Tendsto (fun _ : ℝ => fun _ : Idx 2 => z) atTop (nhds (fun _ : Idx 2 => z)) :=
  ⟨idRescaledDynamics_const 1 z, tendsto_const_nhds⟩

/-- The complete polytope conclusion holds for any injective `A`.
Source: arXiv:2305.05465v6, §8.1, `t:Idcase11` and `e:Ax15`.
Injectivity is the only part of the positive-root assumption needed here.
For positive token count the polytope is the actual future intersection;
the empty configuration is handled separately. -/
theorem IdRescaledDynamics.exists_polytope_limits (A : ParamMatrix d)
    (hA : Function.Injective A) (Z : ℝ → Idx n → EucSpace d)
    (hZ : IdRescaledDynamics A Z) :
    ∃ (P : Set (EucSpace d)) (S : Finset (EucSpace d)),
      P = convexHull ℝ (S : Set (EucSpace d)) ∧
      Tendsto (fun t => Metric.hausdorffDist (tokenHull (Z t)) P) atTop (nhds 0) ∧
      (limitCandidates A P).Finite ∧
      P.extremePoints ℝ ⊆ limitCandidates A P ∧
      limitCandidates A P ⊆ frontier P ∪ {0} ∧
      ∀ i : Idx n, ∃ p ∈ limitCandidates A P, Tendsto (fun t => Z t i) atTop (nhds p) := by
  classical
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · have hcand : limitCandidates A (∅ : Set (EucSpace d)) = ∅ := by
      ext x
      simp [limitCandidates]
    refine ⟨∅, ∅, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp
    · simp only [Metric.hausdorffDist_empty]
      exact tendsto_const_nhds
    · rw [hcand]
      exact Set.finite_empty
    · rw [extremePoints_empty, hcand]
    · rw [hcand]
      exact Set.empty_subset _
    · intro i
      exact Fin.elim0 i
  · obtain ⟨S, -, -, hHull, hHaus⟩ := hZ.exists_tokenHull_polytope_limit A Z hn
    choose Y hY hlim using fun i : Idx n => hZ.exists_limit_in_candidates A hA Z i
    have htuple : Tendsto Z atTop (nhds Y) := tendsto_pi_nhds.mpr hlim
    have hHullY : tokenHull Y = convexHull ℝ (S : Set (EucSpace d)) :=
      (hZ.tokenHull_limit_eq_future A Z Y htuple).trans hHull.symm
    have hYP (i : Idx n) : Y i ∈ limitCandidates A (convexHull ℝ (S : Set (EucSpace d))) := by
      rw [hHull]
      exact hY i
    refine ⟨convexHull ℝ (S : Set (EucSpace d)), S, rfl, hHaus,
      finite_limitCandidates_convexHull A hA S, ?_,
      limitCandidates_convexHull_subset_frontier_union_zero A hA S, ?_⟩
    · intro v hv
      rw [← hHullY] at hv
      obtain ⟨i, rfl⟩ := extremePoints_convexHull_subset hv
      exact hYP i
    · exact fun i => ⟨Y i, hYP i, hlim i⟩

/-- Injectivity and the complete dynamical input are simultaneously realized
by the identity and two stationary unit tokens; arXiv:2305.05465v6, §8.1. -/
example : Function.Injective (ContinuousLinearMap.id ℝ (EucSpace 1)) ∧
    IdRescaledDynamics (1 : ParamMatrix 1)
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨Function.injective_id, idRescaledDynamics_const 1 _⟩

/-- **Theorem (t:Idcase11).** Let `V = I_d` and `QᵀK ≻ 0`. For any initial
datum the hull of the solution of `e:dynIdzi` converges to a convex polytope
`𝒦`. Its vertex-based candidate set `𝒮` is finite, contains every vertex,
lies in `∂𝒦 ∪ {0}`, and contains the limit of each token.

The hypotheses and conclusion retain the source statement. The proof uses
vanishing attention gains instead of the quantitative auxiliary estimate.
Source: arXiv:2305.05465v6, §8.1, `t:Idcase11`, `e:Ax15`, and `eq: A`. -/
theorem idCase_tendsto_limitCandidates (Q K A : ParamMatrix d) (hA : IsAttentionRoot Q K A)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) :
    ∃ (P : Set (EucSpace d)) (S : Finset (EucSpace d)),
      P = convexHull ℝ (S : Set (EucSpace d)) ∧
      Tendsto (fun t => Metric.hausdorffDist (tokenHull (Z t)) P) atTop (nhds 0) ∧
      (limitCandidates A P).Finite ∧
      P.extremePoints ℝ ⊆ limitCandidates A P ∧
      limitCandidates A P ⊆ frontier P ∪ {0} ∧
      ∀ i : Idx n, ∃ p ∈ limitCandidates A P, Tendsto (fun t => Z t i) atTop (nhds p) :=
  hZ.exists_polytope_limits A hA.injective Z

/-- The full paper hypotheses are realized by stationary tokens and
`A = Q = K = I_d`. Source: arXiv:2305.05465v6, §8.1, `e:dynIdzi`, `eq: A`. -/
example (z : EucSpace d) :
    IsAttentionRoot (d := d) (ContinuousLinearMap.id ℝ (EucSpace d))
        (ContinuousLinearMap.id ℝ (EucSpace d)) (ContinuousLinearMap.id ℝ (EucSpace d)) ∧
      IdRescaledDynamics (n := n) (ContinuousLinearMap.id ℝ (EucSpace d)) (fun _ _ => z) :=
  ⟨isAttentionRoot_id d, idRescaledDynamics_const _ z⟩

/-- **Theorem (t:Idcase11int).** Let `V = I_d` and `QᵀK ≻ 0`. For every
initial token configuration, each rescaled token converges to zero or to
the boundary of one convex polytope. Source: arXiv:2305.05465v6, §3,
`t:Idcase11int`, proved by the complete statement in §8.1, `t:Idcase11`.
The original positivity and rescaled-ODE hypotheses are unchanged. -/
theorem idCase_tendsto_zero_or_frontier (Q K : ParamMatrix d) (hQK : IsPosDefQK Q K)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K 1 Z) :
    ∃ S : Finset (EucSpace d), ∀ i : Idx n,
      Filter.Tendsto (fun t => Z t i) Filter.atTop (nhds 0) ∨
        ∃ p ∈ frontier (convexHull ℝ (S : Set (EucSpace d))),
          Filter.Tendsto (fun t => Z t i) Filter.atTop (nhds p) := by
  obtain ⟨A, hA, hAZ⟩ := hZ.exists_identity_root Q K hQK Z
  obtain ⟨P, S, hP, -, -, -, hboundary, hlim⟩ :=
    idCase_tendsto_limitCandidates Q K A hA Z hAZ
  refine ⟨S, ?_⟩
  intro i
  obtain ⟨p, hp, hpi⟩ := hlim i
  rcases hboundary hp with hfrontier | hzero
  · rw [hP] at hfrontier
    exact Or.inr ⟨p, hfrontier, hpi⟩
  · have hp0 : p = 0 := Set.mem_singleton_iff.mp hzero
    subst p
    exact Or.inl hpi

/-- The original §3 hypotheses are satisfied by identity query and key
maps and stationary tokens; arXiv:2305.05465v6, §3, `e:Rres`. -/
example (z : EucSpace d) :
    IsPosDefQK (d := d) (ContinuousLinearMap.id ℝ (EucSpace d))
        (ContinuousLinearMap.id ℝ (EucSpace d)) ∧
      RescaledDynamics (n := n) (ContinuousLinearMap.id ℝ (EucSpace d))
        (ContinuousLinearMap.id ℝ (EucSpace d)) 1 (fun _ _ => z) :=
  ⟨isPosDefQK_of_isAttentionRoot (isAttentionRoot_id d), rescaledDynamics_one_const _ _ z⟩

end Transformer.Clusters
