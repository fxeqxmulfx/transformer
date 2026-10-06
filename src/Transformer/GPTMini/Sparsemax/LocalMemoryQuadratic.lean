import Transformer.GPTMini.Sparsemax.LocalMemoryParameters

/-!
# A strictly convex criterion on every compact attention parameter

Derived additional selection criterion for arXiv:1602.02068v2, Eq. (1)
compact memory. Sum the squared differences from a supplied reference over
all 3P-1 stored edge and query/key-norm coordinates. The criterion is zero
exactly at that reference and is strictly convex. An exact affine gap gives
convexity and rules out two distinct parameter minima on a convex domain.

The reference is additional information, possibly computed from observations;
it need not lie in the feasible domain. The criterion does not follow from
output targets, supply target attention labels, or identify attention from
the unchanged output-only chart. It makes the extra choice explicit and
does not select the model's task loss or an FFN. Compact-domain projection
and the actual joint objective are treated in subsequent modules.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Squared distance on all compact learned coordinates, with no omitted embedding family.
Source: an additional derived selection criterion for arXiv:1602.02068v2, Eq. (1). -/
def localMemoryQuadratic {N : ℕ} (reference p : LocalMemoryParameters N) : ℝ :=
  ∑ i, (localMemoryParameterCoordinates p i - localMemoryParameterCoordinates reference i) ^ 2

/-- Every stored coordinate respects arbitrary linear combinations of the learned parameters.
Source: the compact coordinate representation for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryParameterCoordinates_linear {N : ℕ} (p q : LocalMemoryParameters N)
    (a b : ℝ) (i : LocalMemoryParameterIndex N) :
    localMemoryParameterCoordinates (a • p + b • q) i =
      a * localMemoryParameterCoordinates p i + b * localMemoryParameterCoordinates q i := by
  rcases i with e | x <;> rfl

/-- The criterion is nonnegative for arbitrary references and learned parameter points.
Source: the additional squared-coordinate criterion for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryQuadratic_nonneg {N : ℕ} (reference p : LocalMemoryParameters N) :
    0 ≤ localMemoryQuadratic reference p :=
  Finset.sum_nonneg (fun i _ => sq_nonneg
    (localMemoryParameterCoordinates p i - localMemoryParameterCoordinates reference i))

/-- Zero criterion means equality of all path weights and both Q/K-norm tables.
Source: the complete compact coordinate criterion for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryQuadratic_eq_zero {N : ℕ} (reference p : LocalMemoryParameters N) :
    localMemoryQuadratic reference p = 0 ↔ p = reference := by
  constructor
  · intro h
    have he := (Finset.sum_eq_zero_iff_of_nonneg (fun i hi => sq_nonneg
      (localMemoryParameterCoordinates p i - localMemoryParameterCoordinates reference i))).1 h
    apply localMemoryParameterCoordinates_injective N
    funext i
    exact sub_eq_zero.1 (sq_eq_zero_iff.1 (he i (Finset.mem_univ i)))
  · intro h
    rw [h]
    unfold localMemoryQuadratic
    simp only [sub_self, zero_pow (by decide : (2 : ℕ) ≠ 0), Finset.sum_const_zero]

/-- Every parameter point distinct from the reference has a strictly positive criterion.
Source: the separating compact-coordinate criterion for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryQuadratic_pos {N : ℕ} (reference p : LocalMemoryParameters N)
    (hp : p ≠ reference) : 0 < localMemoryQuadratic reference p := by
  have hn := localMemoryQuadratic_nonneg reference p
  by_contra h
  have hz : localMemoryQuadratic reference p = 0 := by linarith
  exact hp ((localMemoryQuadratic_eq_zero reference p).1 hz)

/-- A changed edge and both changed norms inhabit the positive-distance premise. -/
example : 0 < localMemoryQuadratic (0 : LocalMemoryParameters 1) localMemoryExampleParameters := by
  apply localMemoryQuadratic_pos
  intro h
  have hc := congrArg (fun p : LocalMemoryParameters 1 => p.1 0) h
  norm_num [localMemoryExampleParameters] at hc

/-- The exact affine gap is the product of interpolation weights and squared parameter distance.
Source: the additional compact-coordinate criterion for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryQuadratic_affine_gap {N : ℕ} (reference p q : LocalMemoryParameters N)
    (a b : ℝ) (hab : a + b = 1) :
    localMemoryQuadratic reference (a • p + b • q) =
      a * localMemoryQuadratic reference p + b * localMemoryQuadratic reference q -
        a * b * localMemoryQuadratic q p := by
  unfold localMemoryQuadratic
  rw [Finset.mul_sum, Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib,
    ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i hi
  rw [localMemoryParameterCoordinates_linear]
  have ha : a = 1 - b := by linarith
  rw [ha]
  ring

/-- Distinct parameter endpoints inhabit the exact gap formula. -/
example : localMemoryQuadratic localMemoryExampleParameters
    ((1 / 2 : ℝ) • (0 : LocalMemoryParameters 1) + (1 / 2 : ℝ) • localMemoryExampleParameters) =
    (1 / 2 : ℝ) * localMemoryQuadratic localMemoryExampleParameters (0 : LocalMemoryParameters 1) +
      (1 / 2 : ℝ) * localMemoryQuadratic localMemoryExampleParameters localMemoryExampleParameters -
        (1 / 2 : ℝ) * (1 / 2 : ℝ) * localMemoryQuadratic localMemoryExampleParameters
          (0 : LocalMemoryParameters 1) :=
  localMemoryQuadratic_affine_gap _ _ _ _ _ (by norm_num)

/-- The extra criterion is convex without any structural constraint on the reference.
Source: the derived complete-coordinate selection for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryQuadratic_convex {N : ℕ} (reference : LocalMemoryParameters N) :
    ConvexOn ℝ Set.univ (localMemoryQuadratic reference) := by
  refine ⟨convex_univ, ?_⟩
  intro p hp q hq a b ha hb hab
  change localMemoryQuadratic reference (a • p + b • q) ≤
    a * localMemoryQuadratic reference p + b * localMemoryQuadratic reference q
  have hg := localMemoryQuadratic_affine_gap reference p q a b hab
  have hn := mul_nonneg (mul_nonneg ha hb) (localMemoryQuadratic_nonneg q p)
  linarith

/-- The extra criterion is strictly convex in every compact path and norm coordinate.
Source: the derived compact selection criterion for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryQuadratic_strictConvex {N : ℕ} (reference : LocalMemoryParameters N) :
    StrictConvexOn ℝ Set.univ (localMemoryQuadratic reference) := by
  refine ⟨convex_univ, ?_⟩
  intro p hp q hq hpq a b ha hb hab
  change localMemoryQuadratic reference (a • p + b • q) <
    a * localMemoryQuadratic reference p + b * localMemoryQuadratic reference q
  have hg := localMemoryQuadratic_affine_gap reference p q a b hab
  have hn := mul_pos (mul_pos ha hb) (localMemoryQuadratic_pos q p hpq)
  linarith

/-- Every reference is the exact unrestricted minimizer of the additional criterion.
Source: the complete squared-coordinate selection for arXiv:1602.02068v2, Eq. (1).
No claim that this reference is determined by the output loss is made. -/
theorem localMemoryQuadratic_reference_min {N : ℕ} (reference : LocalMemoryParameters N) :
    IsMinOn (localMemoryQuadratic reference) Set.univ reference := by
  intro p hp
  rw [(localMemoryQuadratic_eq_zero reference reference).2 rfl]
  exact localMemoryQuadratic_nonneg reference p

/-- Distinct parameter minima cannot exist on any convex structural domain.
Source: the strictly convex additional criterion for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryQuadratic_unique_min {N : ℕ} (reference : LocalMemoryParameters N)
    (s : Set (LocalMemoryParameters N)) (hs : Convex ℝ s) (p q : LocalMemoryParameters N)
    (hp : p ∈ s) (hq : q ∈ s) (hpm : IsMinOn (localMemoryQuadratic reference) s p)
    (hqm : IsMinOn (localMemoryQuadratic reference) s q) : p = q := by
  have ht : StrictConvexOn ℝ s (localMemoryQuadratic reference) := ⟨hs, by
    intro x hx y hy hxy a b ha hb hab
    exact (localMemoryQuadratic_strictConvex reference).2
      (Set.mem_univ x) (Set.mem_univ y) hxy ha hb hab⟩
  exact ht.eq_of_isMinOn hpm hqm hp hq

/-- A convex domain with changed feasible norms and edges inhabits all uniqueness premises. -/
example : localMemoryExampleParameters = localMemoryExampleParameters := by
  have hm : IsMinOn (localMemoryQuadratic localMemoryExampleParameters)
      (localMemoryParameterDomain 1 4 (3 / 4)) localMemoryExampleParameters := by
    intro p hp
    rw [(localMemoryQuadratic_eq_zero _ _).2 rfl]
    exact localMemoryQuadratic_nonneg _ p
  exact localMemoryQuadratic_unique_min _ _ (localMemoryParameterDomain_convex _ _ _)
    _ _ localMemoryExampleParameters_mem localMemoryExampleParameters_mem hm hm

end Transformer.GPTMini.Sparsemax
