import Transformer.GPTMini.Sparsemax.CubeContentBits

/-!
# Sparse content diffusion with a convex learned budget

New learned memory following arXiv:1602.02068v2, Eq. (1). Nonnegative
weights select independently generated bit-flip destinations. The total
outgoing mass is bounded by one minus the prescribed self-weight floor.
This is a linear-size routing state and a convex domain, for exponentially
many virtual bit states. Each row has at most one plus the bit count
possible destinations, including changes from zero to positive weights.

The generated matrix below will be realized by actual Q/K dot products
and the original variational sparsemax. It is not defined as that
attention operator. Its feature action is proved for arbitrary functions,
so the later compact value inverse has no stored virtual-state table.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Convex nonnegative outgoing budgets for every content coordinate.
Source: new structural attention restriction before sparsemax Eq. (1). -/
def cubeContentRouteDomain (ι : Type*) [Fintype ι] (floor : ℝ) : Set (ι → ℝ) :=
  {t | (∀ i, 0 ≤ t i) ∧ ∑ i, t i ≤ 1 - floor}

/-- The unallocated probability mass stays at the actual query state.
Source: the generated sparse content matrix before sparsemax Eq. (1). -/
def cubeContentSelf (t : ι → ℝ) : ℝ := 1 - ∑ i, t i

/-- Generated identity and coordinate-flip probability atoms.
Source: the new local virtual-state memory before sparsemax Eq. (1). -/
def cubeContentRouting (t : ι → ℝ) (x y : CubeContentState ι) : ℝ :=
  (if x = y then cubeContentSelf t else 0) +
    ∑ i, if cubeContentFlip i x = y then t i else 0

omit [DecidableEq ι] in
/-- All selected content coordinates can learn over one convex domain.
Source: the linear nonnegative simplex budget before sparsemax Eq. (1). -/
theorem cubeContentRouteDomain_convex (floor : ℝ) :
    Convex ℝ (cubeContentRouteDomain ι floor) := by
  intro t ht s hs a b ha hb hab
  constructor
  · intro i
    exact add_nonneg (mul_nonneg ha (ht.1 i)) (mul_nonneg hb (hs.1 i))
  · change (∑ i, (a * t i + b * s i)) ≤ 1 - floor
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
    have h := add_le_add (mul_le_mul_of_nonneg_left ht.2 ha) (mul_le_mul_of_nonneg_left hs.2 hb)
    rw [← add_mul, hab, one_mul] at h
    exact h

omit [DecidableEq ι] in
/-- Zero outgoing mass is feasible for every compatible floor.
Source: a full-domain witness before sparsemax Eq. (1). -/
theorem zero_mem_cubeContentRouteDomain (floor : ℝ) (hf : floor ≤ 1) :
    (0 : ι → ℝ) ∈ cubeContentRouteDomain ι floor := by
  constructor
  · intro i
    exact le_refl 0
  · simp only [Pi.zero_apply, Finset.sum_const_zero]
    linarith

/-- A nontrivial bit space inhabits the zero-routing feasibility premise. -/
example : (0 : Fin 2 → ℝ) ∈ cubeContentRouteDomain (Fin 2) (3 / 4) :=
  zero_mem_cubeContentRouteDomain _ (by norm_num)

/-- Two simultaneous positive flips inhabit the joint route budget. -/
example : (fun _ : Fin 2 => (1 / 8 : ℝ)) ∈ cubeContentRouteDomain (Fin 2) (3 / 4) := by
  constructor
  · intro i
    norm_num
  · norm_num [Fin.sum_univ_two]

/-- Every generated routing row has exact total mass one.
Source: identity and local permutation atoms before sparsemax Eq. (1). -/
theorem cubeContentRouting_rowSum (t : ι → ℝ) (x : CubeContentState ι) :
    (∑ y, cubeContentRouting t x y) = 1 := by
  classical
  unfold cubeContentRouting
  rw [Finset.sum_add_distrib, Finset.sum_comm]
  simp only [Fintype.sum_ite_eq, cubeContentSelf]
  ring

/-- The genuine generated diagonal is precisely the unallocated self mass.
Source: flip destinations are distinct from the query before sparsemax Eq. (1). -/
theorem cubeContentRouting_self (t : ι → ℝ) (x : CubeContentState ι) :
    cubeContentRouting t x x = cubeContentSelf t := by
  unfold cubeContentRouting
  simp only [ite_true, cubeContentFlip_ne, ite_false, Finset.sum_const_zero, add_zero]

/-- Each actual neighbor receives exactly its own learned flip weight.
Source: distinct local bit-flip destinations before sparsemax Eq. (1). -/
theorem cubeContentRouting_flip (t : ι → ℝ) (x : CubeContentState ι) (i : ι) :
    cubeContentRouting t x (cubeContentFlip i x) = t i := by
  unfold cubeContentRouting
  have hn : x ≠ cubeContentFlip i x := (cubeContentFlip_ne i x).symm
  simp only [hn, ite_false, zero_add, (cubeContentFlip_index_injective x).eq_iff]
  exact Fintype.sum_ite_eq' _ _

/-- Outside the query and generated neighbors, the routing matrix is zero.
Source: the explicit sparse structural mask before sparsemax Eq. (1). -/
theorem cubeContentRouting_zero (t : ι → ℝ) (x y : CubeContentState ι)
    (hself : x ≠ y) (hflip : ∀ i, cubeContentFlip i x ≠ y) :
    cubeContentRouting t x y = 0 := by
  unfold cubeContentRouting
  simp only [hself, hflip, ite_false, Finset.sum_const_zero, add_zero]

/-- With two bits, the state differing in both is a genuine excluded destination. -/
example : cubeContentRouting (fun _ : Fin 2 => (1 / 8 : ℝ))
    (fun _ => false) (fun _ => true) = 0 := by
  apply cubeContentRouting_zero
  · intro h
    have hb := congrFun h 0
    contradiction
  · intro i h
    fin_cases i
    · have hb := congrFun h 1
      norm_num [cubeContentFlip] at hb
    · have hb := congrFun h 0
      norm_num [cubeContentFlip] at hb

/-- Feasible routing entries are actual nonnegative probabilities.
Source: the convex outgoing budget before sparsemax Eq. (1). -/
theorem cubeContentRouting_nonneg (floor : ℝ) (t : ι → ℝ) (hf : 0 ≤ floor)
    (ht : t ∈ cubeContentRouteDomain ι floor) (x y : CubeContentState ι) :
    0 ≤ cubeContentRouting t x y := by
  have hself : 0 ≤ cubeContentSelf t := by unfold cubeContentSelf; linarith [ht.2]
  apply add_nonneg
  · split_ifs <;> positivity
  · exact Finset.sum_nonneg (fun i hi => by split_ifs; exact ht.1 i; exact le_refl 0)

/-- Applying generated local memory to a feature requires only the query and its neighbors.
Source: finite identity/flip action before sparsemax Eq. (1). -/
theorem cubeContentRouting_action (t : ι → ℝ) (f : CubeContentState ι → ℝ) (x : CubeContentState ι) :
    (∑ y, cubeContentRouting t x y * f y) =
      cubeContentSelf t * f x + ∑ i, t i * f (cubeContentFlip i x) := by
  classical
  simp only [cubeContentRouting, add_mul, Finset.sum_add_distrib, Finset.sum_mul, ite_mul, zero_mul]
  rw [Finset.sum_comm]
  simp only [Fintype.sum_ite_eq]

/-- Positive simultaneous flip weights satisfy all nonnegative-entry premises. -/
example : 0 ≤ cubeContentRouting (fun _ : Fin 2 => (1 / 8 : ℝ)) (fun _ => false) (fun _ => true) :=
  cubeContentRouting_nonneg (3 / 4) _ (by norm_num)
    (by constructor; intro i; norm_num; norm_num [Fin.sum_univ_two]) _ _

end Transformer.GPTMini.Sparsemax
