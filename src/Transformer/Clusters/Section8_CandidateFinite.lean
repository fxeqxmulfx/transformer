module

public import Transformer.Clusters.Section8_CandidateBoundary
public import Mathlib.Analysis.Convex.Combination
public import Mathlib.Data.Finset.Powerset
public import Mathlib.Data.Set.Card

/-!
# Finitely many candidate limits

The finiteness part of Claim `cl:propofS`, arXiv:2305.05465v6, §8.1.
A convex combination representing a candidate has zero weight at every
generator where its supporting functional is strictly below its maximum.
Thus the candidate lies in the hull of its active generators. For a fixed
active set the supporting equalities determine at most one candidate, so
the candidate set injects into the finite powerset of the generators.

The source describes the candidate as a projection onto the linear span
of active vertices. The equalities instead give orthogonality to their
displacements from the candidate, hence to the direction of their affine
hull. The proof below uses these equalities directly, without the source's
linear-span assertion. It includes the zero candidate and allows arbitrary
finite generators, including redundant generators and the empty set.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Clusters

variable {d : ℕ}

/-- The generators attaining the supporting value of `e:Ax15` at `x`.
Source: arXiv:2305.05465v6, §8.1, the index set `I` in `cl:propofS`. -/
noncomputable def attentionActiveGenerators (A : ParamMatrix d)
    (S : Finset (EucSpace d)) (x : EucSpace d) : Finset (EucSpace d) := by
  classical
  exact S.filter fun v => inner (𝕜 := ℝ) (A x) (A v) = ‖A x‖ ^ 2

/-- Every candidate is a convex combination of its active generators.
Source: arXiv:2305.05465v6, §8.1, proof of Claim `cl:propofS`.
The nonnegative weighted supporting gaps sum to zero, so inactive weights
vanish. No positive-definiteness assumption is needed for this step. -/
theorem mem_convexHull_attentionActiveGenerators (A : ParamMatrix d)
    (S : Finset (EucSpace d)) (x : EucSpace d)
    (hx : x ∈ limitCandidates A (convexHull ℝ (S : Set (EucSpace d)))) :
    x ∈ convexHull ℝ (attentionActiveGenerators A S x : Set (EucSpace d)) := by
  classical
  obtain ⟨w, hw, hsum, hrepr⟩ := Finset.mem_convexHull'.mp hx.1
  let f : EucSpace d →L[ℝ] ℝ := (innerSL ℝ (A x)).comp A
  have hle (v : EucSpace d) (hv : v ∈ S) : f v ≤ ‖A x‖ ^ 2 :=
    hx.2.inner_le A _ (S.finite_toSet.isCompact_convexHull ℝ)
      (convex_convexHull ℝ _) x (subset_convexHull ℝ _ hv)
  have hzero : ∑ v ∈ S, w v * (‖A x‖ ^ 2 - f v) = 0 := by
    calc
      _ = (∑ v ∈ S, w v) * ‖A x‖ ^ 2 - f (∑ v ∈ S, w v • v) := by
        simp only [mul_sub, Finset.sum_sub_distrib, Finset.sum_mul, map_sum,
          map_smul, smul_eq_mul]
      _ = 0 := by rw [hsum, hrepr]; simp [f]
  have hterms := (Finset.sum_eq_zero_iff_of_nonneg
    (fun v hv => mul_nonneg (hw v hv) (sub_nonneg.mpr (hle v hv)))).mp hzero
  have hactive (v : EucSpace d) (hv : v ∈ S) (hweight : w v ≠ 0) :
      inner (𝕜 := ℝ) (A x) (A v) = ‖A x‖ ^ 2 :=
    (sub_eq_zero.mp ((mul_eq_zero.mp (hterms v hv)).resolve_left hweight)).symm
  apply Finset.mem_convexHull'.mpr
  refine ⟨w, fun v hv => hw v (Finset.mem_filter.mp hv).1, ?_, ?_⟩
  · exact (Finset.sum_filter_of_ne hactive).trans hsum
  · apply (Finset.sum_filter_of_ne ?_).trans hrepr
    intro v hv hterm
    apply hactive v hv
    intro hweight
    exact hterm (by rw [hweight, zero_smul])

/-- The candidate-membership hypothesis holds at every singleton vertex;
arXiv:2305.05465v6, §8.1, `e:Ax15`. -/
example (A : ParamMatrix d) (x : EucSpace d) :
    x ∈ limitCandidates A (convexHull ℝ ({x} : Set (EucSpace d))) := by
  rw [convexHull_singleton]
  exact ⟨Set.mem_singleton _, isAttentionMax_singleton A x⟩

/-- A supporting equality on generators extends to their convex hull.
Source: arXiv:2305.05465v6, §8.1, Claim `cl:propofS`, applied to active
generators. This is linearity of a finite convex combination. -/
theorem inner_eq_on_convexHull_of_generators (A : ParamMatrix d)
    (x : EucSpace d) (S : Finset (EucSpace d)) (c : ℝ)
    (hS : ∀ v ∈ S, inner (𝕜 := ℝ) (A x) (A v) = c)
    {y : EucSpace d} (hy : y ∈ convexHull ℝ (S : Set (EucSpace d))) :
    inner (𝕜 := ℝ) (A x) (A y) = c := by
  obtain ⟨w, -, hsum, hrepr⟩ := Finset.mem_convexHull'.mp hy
  let f : EucSpace d →L[ℝ] ℝ := (innerSL ℝ (A x)).comp A
  change f y = c
  calc
    f y = ∑ v ∈ S, w v * f v := by rw [← hrepr]; simp only [map_sum, map_smul, smul_eq_mul]
    _ = ∑ v ∈ S, w v * c := Finset.sum_congr rfl fun v hv => by rw [show f v = c from hS v hv]
    _ = c := by rw [← Finset.sum_mul, hsum, one_mul]

/-- Both hypotheses hold on the origin singleton with the identity map;
arXiv:2305.05465v6, §8.1, the active-hull equality in `cl:propofS`. -/
example : (∀ v ∈ ({0} : Finset (EucSpace 1)),
    inner (𝕜 := ℝ) (ContinuousLinearMap.id ℝ (EucSpace 1) 0)
      (ContinuousLinearMap.id ℝ (EucSpace 1) v) = (0 : ℝ)) ∧
    (0 : EucSpace 1) ∈ convexHull ℝ ({0} : Set (EucSpace 1)) := by simp

/-- A fixed active set determines at most one candidate.
Source: arXiv:2305.05465v6, §8.1, Claim `cl:propofS`.
Supporting equalities at both candidates force the squared norm of their
image difference to vanish; injectivity of `A` gives equality of the points. -/
theorem eq_of_attentionActiveGenerators_eq (A : ParamMatrix d)
    (hA : Function.Injective A) (S : Finset (EucSpace d)) (x y : EucSpace d)
    (hx : x ∈ limitCandidates A (convexHull ℝ (S : Set (EucSpace d))))
    (hy : y ∈ limitCandidates A (convexHull ℝ (S : Set (EucSpace d))))
    (hactive : attentionActiveGenerators A S x = attentionActiveGenerators A S y) :
    x = y := by
  classical
  have hxy : inner (𝕜 := ℝ) (A x) (A y) = ‖A x‖ ^ 2 := by
    apply inner_eq_on_convexHull_of_generators A x (attentionActiveGenerators A S x) _
      (fun _ hv => (Finset.mem_filter.mp hv).2)
    rw [hactive]
    exact mem_convexHull_attentionActiveGenerators A S y hy
  have hyx : inner (𝕜 := ℝ) (A y) (A x) = ‖A y‖ ^ 2 := by
    apply inner_eq_on_convexHull_of_generators A y (attentionActiveGenerators A S y) _
      (fun _ hv => (Finset.mem_filter.mp hv).2)
    rw [← hactive]
    exact mem_convexHull_attentionActiveGenerators A S x hx
  have hsq : ‖A x - A y‖ ^ 2 = 0 := by
    rw [norm_sub_sq_real, hxy]
    rw [real_inner_comm, hxy] at hyx
    linarith
  apply hA
  exact sub_eq_zero.mp (norm_eq_zero.mp (by nlinarith [norm_nonneg (A x - A y)]))

/-- Every uniqueness hypothesis is realized by the same origin candidate
at both positions, with the identity attention operator. -/
example (x : EucSpace 1) : Function.Injective (ContinuousLinearMap.id ℝ (EucSpace 1)) ∧
    x ∈ limitCandidates (1 : ParamMatrix 1) (convexHull ℝ ({x} : Set (EucSpace 1))) ∧
    x ∈ limitCandidates (1 : ParamMatrix 1) (convexHull ℝ ({x} : Set (EucSpace 1))) ∧
    attentionActiveGenerators (1 : ParamMatrix 1) {x} x =
      attentionActiveGenerators (1 : ParamMatrix 1) {x} x := by
  have hx : x ∈ limitCandidates (1 : ParamMatrix 1) {x} :=
    ⟨Set.mem_singleton _, isAttentionMax_singleton _ _⟩
  have hx' : x ∈ limitCandidates (1 : ParamMatrix 1)
      (convexHull ℝ ({x} : Set (EucSpace 1))) := by
    rw [convexHull_singleton]
    exact hx
  exact ⟨Function.injective_id, hx', hx', rfl⟩

/-- **Finiteness part of Claim `cl:propofS`.** The candidates of a finite
polytope form a finite set. Source: arXiv:2305.05465v6, §8.1.
Only injectivity of `A` is needed from the paper's positive definiteness. -/
theorem finite_limitCandidates_convexHull (A : ParamMatrix d)
    (hA : Function.Injective A) (S : Finset (EucSpace d)) :
    (limitCandidates A (convexHull ℝ (S : Set (EucSpace d)))).Finite := by
  classical
  apply Set.Finite.of_injOn (f := attentionActiveGenerators A S)
    (t := (S.powerset : Set (Finset (EucSpace d))))
  · exact fun _ _ => Finset.mem_powerset.mpr (Finset.filter_subset _ _)
  · exact fun x hx y hy hxy => eq_of_attentionActiveGenerators_eq A hA S x y hx hy hxy
  · exact S.powerset.finite_toSet

/-- The injectivity hypothesis holds for the identity operator, with a
nonempty unit-token polytope. Source: arXiv:2305.05465v6, §8.1. -/
example : Function.Injective (ContinuousLinearMap.id ℝ (EucSpace 1)) ∧
    ({(basePoint 0 : EucSpace 1)} : Finset (EucSpace 1)).Nonempty :=
  ⟨Function.injective_id, Finset.singleton_nonempty _⟩

/-- The powerset argument also bounds the number of candidates by
`2 ^ |S|`. Source: arXiv:2305.05465v6, §8.1, the subset-counting argument
in the proof of Claim `cl:propofS`; the numerical bound makes it explicit. -/
theorem ncard_limitCandidates_convexHull_le (A : ParamMatrix d)
    (hA : Function.Injective A) (S : Finset (EucSpace d)) :
    (limitCandidates A (convexHull ℝ (S : Set (EucSpace d)))).ncard ≤ 2 ^ S.card := by
  classical
  have hbound := Set.ncard_le_ncard_of_injOn (attentionActiveGenerators A S)
    (t := (S.powerset : Set (Finset (EucSpace d))))
    (fun _ _ => Finset.mem_powerset.mpr (Finset.filter_subset _ _))
    (fun x hx y hy hxy => eq_of_attentionActiveGenerators_eq A hA S x y hx hy hxy)
    S.powerset.finite_toSet
  simpa only [Set.ncard_coe_finset, Finset.card_powerset] using hbound

/-- The counting bound has an injective identity-operator instance on a
nonempty singleton polytope. Source: arXiv:2305.05465v6, §8.1. -/
example : Function.Injective (ContinuousLinearMap.id ℝ (EucSpace 1)) ∧
    ({(basePoint 0 : EucSpace 1)} : Finset (EucSpace 1)).Nonempty :=
  ⟨Function.injective_id, Finset.singleton_nonempty _⟩

end Transformer.Clusters
