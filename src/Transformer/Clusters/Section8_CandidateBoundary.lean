module

public import Transformer.Clusters.Section8_Polytope
public import Mathlib.Analysis.Convex.KreinMilman
public import Mathlib.Analysis.Calculus.LocalExtr.Basic

/-!
# Supporting hyperplanes of candidate limits

The geometric part of Claim `cl:propofS` in arXiv:2305.05465v6, §8.1:
`S ⊆ ∂K ∪ {0}`. The condition defining `S` bounds a linear functional
on the extreme points. Krein--Milman extends this bound to the whole
compact convex set. At an interior candidate the functional has a local
maximum, hence vanishes; injectivity of `A` then forces the candidate to
be zero.

These conclusions apply to arbitrary compact convex sets. They do not
require token convergence or the assertion that all vertices are candidates.
That latter assertion uses the dynamics of the limiting hull in the paper.
-/

@[expose] public section

open Real Filter Topology

namespace Transformer.Clusters

variable {d : ℕ}

/-- The vertex maximum in `e:Ax15` bounds the same functional at every
point of the compact convex hull. Source: arXiv:2305.05465v6, §8.1,
proof of Claim `cl:propofS`, the convex-combination argument. -/
theorem IsAttentionMax.inner_le (A : ParamMatrix d) (P : Set (EucSpace d))
    (hcompact : IsCompact P) (hconvex : Convex ℝ P) (x : EucSpace d)
    (hx : IsAttentionMax A P x) {y : EucSpace d} (hy : y ∈ P) :
    inner (𝕜 := ℝ) (A x) (A y) ≤ ‖A x‖ ^ 2 := by
  let f : EucSpace d →L[ℝ] ℝ := (innerSL ℝ (A x)).comp A
  have hclosed : IsClosed {y | f y ≤ ‖A x‖ ^ 2} :=
    isClosed_le f.continuous continuous_const
  have hconv : Convex ℝ {y | f y ≤ ‖A x‖ ^ 2} := by
    intro y hy z hz a b ha hb hab
    change f (a • y + b • z) ≤ ‖A x‖ ^ 2
    simp only [map_add, map_smul, smul_eq_mul]
    calc
      a * f y + b * f z ≤ a * ‖A x‖ ^ 2 + b * ‖A x‖ ^ 2 :=
        add_le_add (mul_le_mul_of_nonneg_left hy ha) (mul_le_mul_of_nonneg_left hz hb)
      _ = ‖A x‖ ^ 2 := by rw [← add_mul, hab, one_mul]
  have hvertices : P.extremePoints ℝ ⊆ {y | f y ≤ ‖A x‖ ^ 2} := by
    intro v hv
    exact hx.2 ⟨v, hv, rfl⟩
  have hhull := closure_minimal (convexHull_min hvertices hconv) hclosed
  rw [closure_convexHull_extremePoints hcompact hconvex] at hhull
  exact hhull hy

/-- Singleton candidates provide genuine instances of the maximum in
`e:Ax15`, including the case of a nonzero candidate. -/
theorem isAttentionMax_singleton (A : ParamMatrix d) (x : EucSpace d) :
    IsAttentionMax A {x} x := by
  simp only [IsAttentionMax, extremePoints_singleton, Set.image_singleton,
    real_inner_self_eq_norm_sq]
  exact ⟨Set.mem_singleton _, fun _ hy => by simpa only [Set.mem_singleton_iff] using hy.le⟩

/-- Every hypothesis of the supporting bound is realized by a singleton
polytope and its unique vertex. Source: arXiv:2305.05465v6, §8.1. -/
example (A : ParamMatrix d) (x : EucSpace d) :
    IsCompact ({x} : Set (EucSpace d)) ∧ Convex ℝ ({x} : Set (EucSpace d)) ∧
      IsAttentionMax A {x} x ∧ x ∈ ({x} : Set (EucSpace d)) :=
  ⟨isCompact_singleton, convex_singleton _, isAttentionMax_singleton A x,
    Set.mem_singleton _⟩

/-- The supporting estimate in the displacement form used in the proof
of Claim `cl:propofS`, arXiv:2305.05465v6, §8.1. At a candidate, the
attention inner product with every displacement inside the hull is nonpositive. -/
theorem IsAttentionMax.inner_sub_nonpos (A : ParamMatrix d) (P : Set (EucSpace d))
    (hcompact : IsCompact P) (hconvex : Convex ℝ P) (x : EucSpace d)
    (hx : IsAttentionMax A P x) {y : EucSpace d} (hy : y ∈ P) :
    inner (𝕜 := ℝ) (A x) (A (y - x)) ≤ 0 := by
  rw [map_sub, inner_sub_right, real_inner_self_eq_norm_sq]
  exact sub_nonpos.mpr (hx.inner_le A P hcompact hconvex x hy)

/-- The displacement estimate has an instance at every singleton vertex;
arXiv:2305.05465v6, §8.1, Claim `cl:propofS`. -/
example (A : ParamMatrix d) (x : EucSpace d) :
    IsCompact ({x} : Set (EucSpace d)) ∧ Convex ℝ ({x} : Set (EucSpace d)) ∧
      IsAttentionMax A {x} x ∧ x ∈ ({x} : Set (EucSpace d)) :=
  ⟨isCompact_singleton, convex_singleton _, isAttentionMax_singleton A x,
    Set.mem_singleton _⟩

/-- Zero belongs to the candidate set precisely when it belongs to the
compact convex set. Source: arXiv:2305.05465v6, §8.1, Claim `cl:propofS`.
The nonempty extreme-point set supplies an attained maximum at zero. -/
theorem zero_mem_limitCandidates_iff (A : ParamMatrix d) (P : Set (EucSpace d))
    (hcompact : IsCompact P) : 0 ∈ limitCandidates A P ↔ 0 ∈ P := by
  constructor
  · exact fun hx => hx.1
  · intro h0
    obtain ⟨v, hv⟩ := hcompact.extremePoints_nonempty ⟨0, h0⟩
    refine ⟨h0, ?_, ?_⟩
    · exact ⟨v, hv, by simp⟩
    · rintro r ⟨w, -, rfl⟩
      simp

/-- The compactness hypothesis is satisfied by the origin polytope;
arXiv:2305.05465v6, §8.1, Claim `cl:propofS`. -/
example : IsCompact ({0} : Set (EucSpace 1)) := isCompact_singleton

/-- **Boundary part of Claim `cl:propofS`.** Nonzero candidates are on
the boundary. Source: arXiv:2305.05465v6, §8.1. Injectivity of `A` suffices
for the source's positive-definiteness condition. No dynamical hypothesis
is needed for this part of the claim. -/
theorem limitCandidates_subset_frontier_union_zero
    (A : ParamMatrix d) (hA : Function.Injective A) (P : Set (EucSpace d))
    (hcompact : IsCompact P) (hconvex : Convex ℝ P) :
    limitCandidates A P ⊆ frontier P ∪ {0} := by
  intro x hx
  rcases eq_or_ne x 0 with hzero | hnonzero
  · exact Or.inr (Set.mem_singleton_iff.mpr hzero)
  · apply Or.inl
    apply (mem_frontier_iff_notMem_interior hx.1).mpr
    intro hinterior
    let f : EucSpace d →L[ℝ] ℝ := (innerSL ℝ (A x)).comp A
    have hlocal : IsLocalMax f x := by
      filter_upwards [mem_interior_iff_mem_nhds.mp hinterior] with y hy
      have hle := hx.2.inner_le A P hcompact hconvex x hy
      simpa only [f, ContinuousLinearMap.comp_apply, innerSL_apply_apply,
        real_inner_self_eq_norm_sq] using hle
    have hfzero : f = 0 := hlocal.hasFDerivAt_eq_zero f.hasFDerivAt
    have hsq : ‖A x‖ ^ 2 = 0 := by
      simpa only [f, ContinuousLinearMap.comp_apply, innerSL_apply_apply,
        real_inner_self_eq_norm_sq, zero_apply] using congrArg (fun g => g x) hfzero
    have hAx : A x = 0 := norm_eq_zero.mp (by nlinarith [norm_nonneg (A x)])
    exact hnonzero (hA (by simpa only [map_zero] using hAx))

/-- All boundary-claim hypotheses hold for the identity operator and a
singleton polytope, with an actual candidate at its vertex. -/
example (x : EucSpace 1) :
    Function.Injective (ContinuousLinearMap.id ℝ (EucSpace 1)) ∧
      IsCompact ({x} : Set (EucSpace 1)) ∧ Convex ℝ ({x} : Set (EucSpace 1)) ∧
      x ∈ limitCandidates (ContinuousLinearMap.id ℝ (EucSpace 1)) {x} :=
  ⟨Function.injective_id, isCompact_singleton, convex_singleton _,
    Set.mem_singleton _, isAttentionMax_singleton _ x⟩

/-- The boundary conclusion specialized to the finite polytope in
`t:Idcase11`, arXiv:2305.05465v6, §8.1. The generating set may be empty;
the candidate set is then empty as well. -/
theorem limitCandidates_convexHull_subset_frontier_union_zero
    (A : ParamMatrix d) (hA : Function.Injective A) (S : Finset (EucSpace d)) :
    limitCandidates A (convexHull ℝ (S : Set (EucSpace d))) ⊆
      frontier (convexHull ℝ (S : Set (EucSpace d))) ∪ {0} :=
  limitCandidates_subset_frontier_union_zero A hA _
    (S.finite_toSet.isCompact_convexHull ℝ) (convex_convexHull ℝ _)

/-- Injectivity is realized by the identity, with a nonempty generating
set for a unit-token polytope. Source: arXiv:2305.05465v6, §8.1. -/
example : Function.Injective (ContinuousLinearMap.id ℝ (EucSpace 1)) ∧
    ({(basePoint 0 : EucSpace 1)} : Finset (EucSpace 1)).Nonempty :=
  ⟨Function.injective_id, Finset.singleton_nonempty _⟩

end Transformer.Clusters
