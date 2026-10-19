module

public import Transformer.Clusters.Section10_ProjectedHullLimit
public import Transformer.Clusters.Section8_CandidateFinite

/-!
# Correct leading candidates lie on the relative boundary or at zero

ArXiv:2305.05465v6, §10, the candidate geometry in `t:multiplicity`.
The manuscript defines candidates with `π_F A`; the original attention
scores instead give `A π_F`, as proved in `Section10_ScoreError`.
On the leading subspace this is the restriction of the positive root `A`.
Its injectivity makes the corrected candidate set finite.

The boundary is taken inside `F`, as required by §5, `t:multiplicity`.
An interior candidate maximizes a linear functional locally on `F`.
That functional's derivative vanishes, while its value at the candidate
is `||A p||²`; injectivity forces `p = 0`. Thus the correct conclusion
retains the origin, which the source's auxiliary boundary claim omits.
-/

@[expose] public section

open Real Filter Topology

namespace Transformer.Clusters

variable {d : ℕ}

/-- The true leading candidates use the restriction of `A` to `F`.
Source: arXiv:2305.05465v6, §10, the candidate equality; the source uses
`π_F A`, corrected here to the order given by its original attention scores. -/
def leadingLimitCandidates (A : ParamMatrix d) (F : Submodule ℝ (EucSpace d))
    (S : Finset F) : Set F :=
  {p | (p : EucSpace d) ∈ limitCandidates A (polytopeIn F S)}

/-- Membership in the ambient image of the polytope is exactly
membership in its hull inside `F`. Source: arXiv:2305.05465v6, §5,
`t:multiplicity`, the polytope contained in the leading space. -/
theorem mem_polytopeIn_iff (F : Submodule ℝ (EucSpace d)) (S : Finset F) (p : F) :
    (p : EucSpace d) ∈ polytopeIn F S ↔ p ∈ convexHull ℝ (S : Set F) := by
  constructor
  · rintro ⟨q, hq, hqp⟩
    have hqp' : q = p := Subtype.val_injective hqp
    simpa only [hqp'] using hq
  · intro hp
    exact ⟨p, hp, rfl⟩

/-- A finite leading polytope is compact also in its ambient image.
Source: arXiv:2305.05465v6, §10, the limiting projected hull. -/
theorem isCompact_polytopeIn (F : Submodule ℝ (EucSpace d)) (S : Finset F) :
    IsCompact (polytopeIn F S) :=
  (S.finite_toSet.isCompact_convexHull ℝ).image F.subtypeL.continuous

/-- Its ambient image remains convex under the subspace inclusion.
Source: arXiv:2305.05465v6, §10, the limiting projected hull. -/
theorem convex_polytopeIn (F : Submodule ℝ (EucSpace d)) (S : Finset F) :
    Convex ℝ (polytopeIn F S) :=
  (convex_convexHull ℝ (S : Set F)).linear_image F.subtype

/-- Every corrected candidate is in the actual relative polytope.
Source: arXiv:2305.05465v6, §10, the candidate equality on `K ⊂ F`. -/
theorem leadingLimitCandidates_mem_hull (A : ParamMatrix d)
    (F : Submodule ℝ (EucSpace d)) (S : Finset F) (p : F)
    (hp : p ∈ leadingLimitCandidates A F S) : p ∈ convexHull ℝ (S : Set F) :=
  (mem_polytopeIn_iff F S p).mp hp.1

/-- The candidate-membership hypothesis holds at every singleton
vertex, including a nonzero one. Source: arXiv:2305.05465v6, §10. -/
example (A : ParamMatrix d) (F : Submodule ℝ (EucSpace d)) (p : F) :
    p ∈ leadingLimitCandidates A F {p} := by
  change (p : EucSpace d) ∈ limitCandidates A (polytopeIn F {p})
  simp only [polytopeIn, Finset.coe_singleton, convexHull_singleton, Set.image_singleton]
  exact ⟨Set.mem_singleton _, isAttentionMax_singleton A _⟩

/-- Injectivity of the root makes the corrected leading candidate set
finite. Source: arXiv:2305.05465v6, §10, the finiteness adaptation of
Claim `cl:propofS`, using the true leading form rather than `π_F A`. -/
theorem finite_leadingLimitCandidates (A : ParamMatrix d) (hA : Function.Injective A)
    (F : Submodule ℝ (EucSpace d)) (S : Finset F) :
    (leadingLimitCandidates A F S).Finite := by
  classical
  have hfinite : (limitCandidates A (polytopeIn F S)).Finite := by
    rw [polytopeIn_eq_convexHull_image]
    simpa only [Finset.coe_image] using
      finite_limitCandidates_convexHull A hA (S.image fun p : F => (p : EucSpace d))
  exact Set.Finite.preimage (fun _ _ _ _ h => Subtype.val_injective h) hfinite

/-- The full injectivity input holds for the identity root.
Source: arXiv:2305.05465v6, §10, `A = (QᵀK)^{1/2}`. -/
example : Function.Injective (1 : ParamMatrix d) := Function.injective_id

/-- A candidate in a compact convex subset of `F` is on its relative
boundary or is zero. Source: arXiv:2305.05465v6, §10, the boundary
adaptation of Claim `cl:propofS`, with the origin retained as in §5.
The local maximum and its derivative are taken on `F` itself. -/
theorem candidate_mem_relative_frontier_union_zero (A : ParamMatrix d)
    (hA : Function.Injective A) (F : Submodule ℝ (EucSpace d)) (P : Set F)
    (hcompact : IsCompact P) (hconvex : Convex ℝ P) (p : F)
    (hp : (p : EucSpace d) ∈ limitCandidates A (((↑) : F → EucSpace d) '' P)) :
    p ∈ frontier P ∪ {0} := by
  rcases eq_or_ne p 0 with hzero | hnonzero
  · exact Or.inr (Set.mem_singleton_iff.mpr hzero)
  · have hpP : p ∈ P := by
      obtain ⟨q, hq, hqp⟩ := hp.1
      simpa only [Subtype.val_injective hqp] using hq
    apply Or.inl
    apply (mem_frontier_iff_notMem_interior hpP).mpr
    intro hinterior
    let f : F →L[ℝ] ℝ := ((innerSL ℝ (A (p : EucSpace d))).comp A).comp F.subtypeL
    have hlocal : IsLocalMax f p := by
      filter_upwards [mem_interior_iff_mem_nhds.mp hinterior] with q hq
      have hle := hp.2.inner_le A _ (hcompact.image F.subtypeL.continuous)
        (hconvex.linear_image F.subtype) (p : EucSpace d) ⟨q, hq, rfl⟩
      simpa only [f, ContinuousLinearMap.comp_apply, Submodule.subtypeL_apply,
        innerSL_apply_apply, real_inner_self_eq_norm_sq] using hle
    have hfzero : f = 0 := hlocal.hasFDerivAt_eq_zero f.hasFDerivAt
    have hsq : ‖A (p : EucSpace d)‖ ^ 2 = 0 := by
      simpa only [f, ContinuousLinearMap.comp_apply, Submodule.subtypeL_apply,
        innerSL_apply_apply, real_inner_self_eq_norm_sq, zero_apply] using
        congrArg (fun g : F →L[ℝ] ℝ => g p) hfzero
    have hAp : A (p : EucSpace d) = 0 :=
      norm_eq_zero.mp (by nlinarith [norm_nonneg (A (p : EucSpace d))])
    have hpzero : (p : EucSpace d) = 0 := hA (by simpa only [map_zero] using hAp)
    exact hnonzero (Subtype.val_injective hpzero)

/-- Injectivity, compactness, convexity, and the actual candidate input
coexist at any singleton vertex of a subspace.
Source: arXiv:2305.05465v6, §10, corrected Claim `cl:propofS`. -/
example (F : Submodule ℝ (EucSpace d)) (p : F) :
    Function.Injective (1 : ParamMatrix d) ∧ IsCompact ({p} : Set F) ∧
    Convex ℝ ({p} : Set F) ∧ (p : EucSpace d) ∈
      limitCandidates (1 : ParamMatrix d) (((↑) : F → EucSpace d) '' ({p} : Set F)) := by
  refine ⟨Function.injective_id, isCompact_singleton, convex_singleton _, ?_⟩
  rw [Set.image_singleton]
  exact ⟨Set.mem_singleton _, isAttentionMax_singleton 1 _⟩

/-- **Correct leading candidates lie on the relative polytope boundary
or at zero.** Source: arXiv:2305.05465v6, §5, `t:multiplicity`, and §10,
the candidate geometry, corrected to the true operator order. -/
theorem leadingLimitCandidates_subset_frontier_union_zero
    (A : ParamMatrix d) (hA : Function.Injective A) (F : Submodule ℝ (EucSpace d))
    (S : Finset F) : leadingLimitCandidates A F S ⊆
      frontier (convexHull ℝ (S : Set F)) ∪ {0} :=
  fun p hp => candidate_mem_relative_frontier_union_zero A hA F _
    (S.finite_toSet.isCompact_convexHull ℝ) (convex_convexHull ℝ _) p hp

/-- All boundary-set hypotheses hold with a genuine nonempty polytope.
Source: arXiv:2305.05465v6, §10, corrected Claim `cl:propofS`. -/
example (F : Submodule ℝ (EucSpace d)) (p : F) :
    Function.Injective (1 : ParamMatrix d) ∧ ({p} : Finset F).Nonempty :=
  ⟨Function.injective_id, Finset.singleton_nonempty p⟩

end Transformer.Clusters
