module

public import Transformer.Clusters.Section10_ProjectedClusterLimits
public import Transformer.Clusters.Section10_RelativeCandidates

/-!
# Clustering toward the relative polytope boundary times the complement

ArXiv:2305.05465v6, §5, `t:multiplicity`, with its proof in §10.
The actual leading hull converges to a nonempty finite polytope whenever
there is a token. Every leading token converges to a candidate of this
same polytope. The positive attention root makes the corrected candidate
set finite and places its nonzero points on the relative boundary in `F`.

The complementary token component may be unbounded. Adding it to a
limiting leading point constructs a point of `(∂K ∪ {0}) × G`, and the
distance to that point is exactly the leading displacement. This closes
the original theorem without any token-limit input or a stronger hypothesis.
The source's faulty `π_F A` auxiliary estimates are replaced by the proved
score and energy estimates in the correct order `A π_F`.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d n : ℕ}

/-- **The actual projected hull converges to a polytope with finite
leading candidates on its relative boundary or at zero.**
Source: arXiv:2305.05465v6, §10, the opening paragraph and the adaptation
of `cl:propofS`. The source uses `π_F A` and states `S ⊂ ∂K`; here the
original scores give `A π_F`, and `{0}` is restored as in §8 and §5.
Positive token count is explicit, and the nonempty polytope is identified
with the future-hull intersection rather than only by real Hausdorff distance. -/
theorem exists_polytope_tendsto_image_tokenHull (Q K V A proj : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hQKV : IsGoodTripleMulti Q K V F G lam) (hA : IsAttentionRoot Q K A)
    (hproj : IsProjOnto proj F G) (Z : ℝ → Idx n → EucSpace d)
    (hZ : RescaledDynamics Q K V Z) (hn : 0 < n) :
    ∃ S : Finset F, S.Nonempty ∧ S.card ≤ n ∧
      polytopeIn F S = {x | ∀ t : ℝ, 0 ≤ t → x ∈ proj '' tokenHull (Z t)} ∧
      Tendsto (fun t => Metric.hausdorffDist (proj '' tokenHull (Z t)) (polytopeIn F S))
        atTop (nhds 0) ∧ (leadingLimitCandidates A F S).Finite ∧
      leadingLimitCandidates A F S ⊆ frontier (convexHull ℝ (S : Set F)) ∪ {0} := by
  obtain ⟨S, hS, hcard, hHull, hHaus⟩ :=
    hZ.exists_projected_hull_polytope_limit Q K V proj F G lam hQKV.2 hproj Z hn
  exact ⟨S, hS, hcard, hHull, hHaus, finite_leadingLimitCandidates A hA.injective F S,
    leadingLimitCandidates_subset_frontier_union_zero A hA.injective F S⟩

/-- **The actual leading hull and every projected token have compatible
polytope limits.** Source: arXiv:2305.05465v6, §10, `t:multiplicity`.
The same polytope is the future-hull intersection, has at most `n`
generators, and contains each token limit in its finite corrected
candidate set. No convergence or candidate hypothesis is assumed. -/
theorem RescaledDynamics.exists_leading_polytope_limits
    (Q K V proj A : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (hA : IsAttentionRoot Q K A) (Z : ℝ → Idx n → EucSpace d)
    (hZ : RescaledDynamics Q K V Z) (hn : 0 < n) :
    ∃ S : Finset F, S.Nonempty ∧ S.card ≤ n ∧
      polytopeIn F S = {x | ∀ t : ℝ, 0 ≤ t → x ∈ tokenHull (fun j => proj (Z t j))} ∧
      Tendsto (fun t => Metric.hausdorffDist (proj '' tokenHull (Z t)) (polytopeIn F S))
        atTop (nhds 0) ∧ (leadingLimitCandidates A F S).Finite ∧
      ∀ i : Idx n, ∃ p ∈ leadingLimitCandidates A F S,
        Tendsto (fun t => proj (Z t i)) atTop (nhds (p : EucSpace d)) := by
  obtain ⟨S, hS, hcard, hHull, hHaus⟩ :=
    hZ.exists_projected_hull_polytope_limit Q K V proj F G lam hV hproj Z hn
  have hHull' : polytopeIn F S =
      {x | ∀ t : ℝ, 0 ≤ t → x ∈ tokenHull (fun j => proj (Z t j))} := by
    simpa only [image_tokenHull_eq_tokenHull] using hHull
  refine ⟨S, hS, hcard, hHull', hHaus, finite_leadingLimitCandidates A hA.injective F S, ?_⟩
  intro i
  obtain ⟨p, hp, hlim⟩ :=
    hZ.exists_projected_limit_in_candidates Q K V proj A F G lam hV hproj hA Z i
  have hpPoly : p ∈ polytopeIn F S := by rw [hHull']; exact hp.1
  obtain ⟨q, _, hqp⟩ := hpPoly
  refine ⟨q, ?_, ?_⟩
  · change (q : EucSpace d) ∈ limitCandidates A (polytopeIn F S)
    rw [hHull', hqp]
    exact hp
  · simpa only [hqp] using hlim

/-- All root, splitting, projection, actual-ODE and nonempty-token inputs
hold together. Source: arXiv:2305.05465v6, §3, `e:Rres`, and §10. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧ IsAttentionRoot (1 : ParamMatrix 1) 1 1 ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) ∧ 0 < 2 :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, isAttentionRoot_id 1,
    rescaledDynamics_one_const 1 1 _, by norm_num⟩

/-- The full spatial distance to the boundary-times-complement set is
at most the displacement of the leading projection from any of its
boundary or zero points. Source: arXiv:2305.05465v6, §5, `t:multiplicity`,
and the conclusion of §10. The comparison point is constructed in the set. -/
theorem infDist_polytopeTimesSubspace_le_norm_projected_sub
    (proj : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (S : Finset F)
    (hFG : IsCompl F G) (hproj : IsProjOnto proj F G) (p : F)
    (hp : p ∈ frontier (convexHull ℝ (S : Set F)) ∪ {0}) (x : EucSpace d) :
    Metric.infDist x (polytopeTimesSubspace F G S) ≤ ‖proj x - (p : EucSpace d)‖ := by
  have hy : (p : EucSpace d) + (1 - proj) x ∈ polytopeTimesSubspace F G S := by
    refine ⟨p, hp, ?_⟩
    rw [add_sub_cancel_left]
    simpa only [sub_apply, one_apply_eq_self] using hproj.mem_complement proj F G hFG x
  calc
    _ ≤ dist x ((p : EucSpace d) + (1 - proj) x) := Metric.infDist_le_dist_of_mem hy
    _ = _ := by
      rw [dist_eq_norm]
      congr 1
      simp only [sub_apply, one_apply_eq_self]
      abel

/-- The complement, projection, and boundary-or-zero hypotheses all
hold with a genuine singleton polytope and its origin.
Source: arXiv:2305.05465v6, §5, `t:multiplicity`. -/
example : IsCompl (⊤ : Submodule ℝ (EucSpace 1)) ⊥ ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧
    (0 : (⊤ : Submodule ℝ (EucSpace 1))) ∈
      frontier (convexHull ℝ ({0} : Set (⊤ : Submodule ℝ (EucSpace 1)))) ∪ {0} :=
  ⟨isCompl_top_bot, isProjOnto_one 1, Or.inr (Set.mem_singleton _)⟩

/-- **Theorem (t:multiplicity), clustering for `λ₁` with multiplicity.**
For every good triple with multiplicity and actual rescaled solution,
there is a bounded convex polytope in `F` such that all tokens approach
`(∂K ∪ {0}) × G`. The frontier is relative to `F`.

Source: arXiv:2305.05465v6, §5, `t:multiplicity`, proof in §10. The proof
uses `A π_F` for the leading form and retains the leading speed `λ`,
correcting the source's auxiliary projection-order and derivative errors
while preserving the theorem's original hypotheses and conclusion. -/
theorem multiplicity_dist_tendsto_zero (Q K V : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hQKV : IsGoodTripleMulti Q K V F G lam) (Z : ℝ → Idx n → EucSpace d)
    (hZ : RescaledDynamics Q K V Z) :
    ∃ T : Finset F, ∀ i : Idx n,
      Tendsto (fun t => Metric.infDist (Z t i) (polytopeTimesSubspace F G T))
        atTop (nhds 0) := by
  rcases isEmpty_or_nonempty (Idx n) with hn | hn
  · exact ⟨∅, fun i => isEmptyElim i⟩
  obtain ⟨proj, hproj⟩ := exists_isProjOnto F G hQKV.2.2.2.1
  obtain ⟨A, hA⟩ := hQKV.1.exists_attentionRoot
  obtain ⟨S, _, _, _, _, _, hlimits⟩ := hZ.exists_leading_polytope_limits
    Q K V proj A F G lam hQKV.2 hproj hA Z (Fin.pos_iff_nonempty.mpr hn)
  refine ⟨S, fun i => ?_⟩
  obtain ⟨p, hp, hlim⟩ := hlimits i
  have hpBoundary := leadingLimitCandidates_subset_frontier_union_zero A hA.injective F S hp
  have hconst : Tendsto (fun _ : ℝ => (p : EucSpace d)) atTop (nhds (p : EucSpace d)) :=
    tendsto_const_nhds
  have hnorm : Tendsto (fun t => ‖proj (Z t i) - (p : EucSpace d)‖) atTop (nhds 0) := by
    simpa only [sub_self, norm_zero] using (hlim.sub hconst).norm
  exact squeeze_zero' (Eventually.of_forall fun _ => Metric.infDist_nonneg)
    (Eventually.of_forall fun t => infDist_polytopeTimesSubspace_le_norm_projected_sub
      proj F G S hQKV.2.2.2.1 hproj p hpBoundary (Z t i)) hnorm

/-- The original hypotheses are satisfiable: the identity triple with
`F` the whole space and `G = 0`, together with stationary coincident
tokens. Source: arXiv:2305.05465v6, §5, `t:multiplicity`. -/
example (d : ℕ) (z : EucSpace d) :
    IsGoodTripleMulti (1 : ParamMatrix d) 1 1 ⊤ ⊥ 1 ∧
      RescaledDynamics (n := n) (1 : ParamMatrix d) 1 1 (fun _ _ => z) :=
  ⟨isGoodTripleMulti_one d, rescaledDynamics_one_const _ _ z⟩

/-- Positive scalar speed need not be one: the complete hypotheses also
hold at speed two for arbitrary positive attention forms and coincident
tokens. Source: arXiv:2305.05465v6, §5, `d:goodmulti` and `t:multiplicity`. -/
example (Q K : ParamMatrix d) (hQK : IsPosDefQK Q K) (z : EucSpace d) :
    IsGoodTripleMulti Q K ((2 : ℝ) • 1) ⊤ ⊥ 2 ∧
      RescaledDynamics (n := n) Q K ((2 : ℝ) • 1) (fun _ _ => z) :=
  ⟨isGoodTripleMulti_smul_one Q K hQK 2 (by norm_num),
    rescaledDynamics_smul_one_const Q K 2 z⟩

end Transformer.Clusters
