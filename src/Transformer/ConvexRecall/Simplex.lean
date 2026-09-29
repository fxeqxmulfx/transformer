/-
# Convex simplex routing and its unique sparse winner

Extension of the simplex relaxation in arXiv:2211.11052v1, §3.1. Instead
of training a positional simplex matrix shared across samples, each query
solves a convex quadratic program with learned content costs. A cost gap
of one half yields an exact basis-vector solution, including a null slot.
Recall context: arXiv:2312.04927v1, Appendix `prop: app-attention`.
-/

import Transformer.ConvexRecall.Basic

open scoped BigOperators

noncomputable section

namespace Transformer.ConvexRecall

variable {ι : Type*} [Fintype ι]

/-- The probability simplex with forbidden positions constrained to zero. -/
def simplexOn (allowed : Set ι) : Set (ι → ℝ) :=
  {a | (∀ j, 0 ≤ a j) ∧ ∑ j, a j = 1 ∧ ∀ j, j ∉ allowed → a j = 0}

/-- A sparse attention vector selecting one slot. -/
def basis (winner : ι) : ι → ℝ := by
  classical
  exact fun j => if j = winner then 1 else 0

/-- Content cost with a strictly convex quadratic simplex penalty. -/
def routingObjective (cost : ι → ℝ) (a : ι → ℝ) : ℝ :=
  (∑ j, (a j) ^ 2) / 4 + ∑ j, a j * cost j

/-- An allowed slot supplies a feasible basis vector. Extension of
arXiv:2211.11052v1, §3.1, with a causal mask for the task in
arXiv:2312.04927v1, §3. -/
theorem basis_mem_simplex (allowed : Set ι) (winner : ι)
    (hw : winner ∈ allowed) : basis winner ∈ simplexOn allowed := by
  classical
  refine ⟨fun j => ?_, by simp [basis], fun j hj => ?_⟩
  · simp only [basis]
    split_ifs <;> norm_num
  · have h : j ≠ winner := fun he => hj (he ▸ hw)
    simp [basis, h]

/-- The masked simplex is convex. Extension of arXiv:2211.11052v1, §3.1,
adding the causal exclusion needed by arXiv:2312.04927v1, §3. -/
theorem simplexOn_convex (allowed : Set ι) : Convex ℝ (simplexOn allowed) := by
  intro x hx y hy a t ha ht hat
  refine ⟨fun j => ?_, ?_, fun j hj => ?_⟩
  · exact add_nonneg (mul_nonneg ha (hx.1 j)) (mul_nonneg ht (hy.1 j))
  · change (∑ j, (a * x j + t * y j)) = 1
    simp [Finset.sum_add_distrib, ← Finset.mul_sum, hx.2.1, hy.2.1, hat]
  · change a * x j + t * y j = 0
    simp [hx.2.2 j hj, hy.2.2 j hj]

/-- Inference is a convex optimization problem for every learned cost
vector; learning query-key matrices jointly is not asserted convex here.
Extension of arXiv:2211.11052v1, §3.1. -/
theorem routingObjective_convex (allowed : Set ι) (cost : ι → ℝ) :
    ConvexOn ℝ (simplexOn allowed) (routingObjective cost) := by
  have hsq : ConvexOn ℝ Set.univ (fun x : ℝ => x ^ 2) :=
    (by decide : Even (2 : ℕ)).convexOn_pow
  refine ⟨simplexOn_convex allowed, ?_⟩
  intro x _ y _ a t ha ht hat
  have hs : (∑ j, (a * x j + t * y j) ^ 2) ≤
      a * (∑ j, (x j) ^ 2) + t * (∑ j, (y j) ^ 2) := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_le_sum fun j _ =>
      hsq.2 (Set.mem_univ (x j)) (Set.mem_univ (y j)) ha ht hat
  have hl : (∑ j, (a * x j + t * y j) * cost j) =
      a * (∑ j, x j * cost j) + t * (∑ j, y j * cost j) := by
    simp [add_mul, mul_assoc, Finset.sum_add_distrib, Finset.mul_sum]
  simp only [routingObjective, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  rw [hl]
  nlinarith

/-- The squared distance to a basis vector has this exact expansion.
Extension of arXiv:2211.11052v1, §3.1, quadratic simplex routing. -/
theorem basis_distance (a : ι → ℝ) (winner : ι) :
    (∑ j, (a j - basis winner j) ^ 2) =
      (∑ j, (a j) ^ 2) - 2 * a winner + 1 := by
  classical
  simp [sub_sq, basis, Finset.sum_add_distrib, Finset.sum_sub_distrib]

/-- A half-unit cost gap gives a quantitative global optimality
certificate, respecting forbidden slots. Extension of arXiv:2211.11052v1,
§3.1; the mask addresses arXiv:2312.04927v1, §3. -/
theorem routing_error_bound (allowed : Set ι) (cost : ι → ℝ) (winner : ι)
    (hgap : ∀ j, j ∈ allowed → j ≠ winner → cost winner + 1 / 2 ≤ cost j)
    (a : ι → ℝ) (ha : a ∈ simplexOn allowed) :
    routingObjective cost (basis winner) +
        (∑ j, (a j - basis winner j) ^ 2) / 4 ≤ routingObjective cost a := by
  classical
  have hp : ∀ j, a j * cost winner + a j / 2 -
      (if j = winner then a j / 2 else 0) ≤ a j * cost j := by
    intro j
    by_cases he : j = winner
    · subst j
      simp
    · by_cases hj : j ∈ allowed
      · have h := mul_le_mul_of_nonneg_left (hgap j hj he) (ha.1 j)
        simp only [he, ite_false, sub_zero]
        nlinarith
      · simp [ha.2.2 j hj, he]
  have hs := Finset.sum_le_sum (fun j (_ : j ∈ Finset.univ) => hp j)
  have hl : cost winner + (1 - a winner) / 2 ≤ ∑ j, a j * cost j := by
    simp only [Finset.sum_sub_distrib, Finset.sum_add_distrib,
      ← Finset.sum_mul, ← Finset.sum_div] at hs
    simp [ha.2.1] at hs
    linarith
  have hb : routingObjective cost (basis winner) = 1 / 4 + cost winner := by
    simp [routingObjective, basis]
  rw [hb, basis_distance]
  unfold routingObjective
  linarith

/-- An allowed winner separated by a half-unit gap is a global optimum.
This solves the inference program explicitly, without an existence
assumption. Extension of arXiv:2211.11052v1, §3.1. -/
theorem basis_minimizes_routing (allowed : Set ι) (cost : ι → ℝ) (winner : ι)
    (hw : winner ∈ allowed)
    (hgap : ∀ j, j ∈ allowed → j ≠ winner → cost winner + 1 / 2 ≤ cost j) :
    basis winner ∈ simplexOn allowed ∧
      ∀ a ∈ simplexOn allowed,
        routingObjective cost (basis winner) ≤ routingObjective cost a := by
  refine ⟨basis_mem_simplex allowed winner hw, ?_⟩
  intro a ha
  have h := routing_error_bound allowed cost winner hgap a ha
  have hn : 0 ≤ ∑ j, (a j - basis winner j) ^ 2 :=
    Finset.sum_nonneg fun _ _ => sq_nonneg _
  linarith

/-- Every optimum equals the winning basis vector. Consequently retrieval
does not depend on which optimization algorithm finds a global optimum.
Extension of arXiv:2211.11052v1, §3.1. -/
theorem routing_minimizer_unique (allowed : Set ι) (cost : ι → ℝ) (winner : ι)
    (hgap : ∀ j, j ∈ allowed → j ≠ winner → cost winner + 1 / 2 ≤ cost j)
    (a : ι → ℝ) (ha : a ∈ simplexOn allowed)
    (hmin : routingObjective cost a ≤ routingObjective cost (basis winner)) :
    a = basis winner := by
  have h := routing_error_bound allowed cost winner hgap a ha
  have hn : ∀ j, 0 ≤ (a j - basis winner j) ^ 2 := fun _ => sq_nonneg _
  have hz : (∑ j, (a j - basis winner j) ^ 2) = 0 := by
    have hs := Finset.sum_nonneg (fun j (_ : j ∈ Finset.univ) => hn j)
    linarith
  have hall := (Finset.sum_eq_zero_iff_of_nonneg
    (fun j (_ : j ∈ Finset.univ) => hn j)).mp hz
  funext j
  have hj := hall j (Finset.mem_univ j)
  nlinarith

/-- An allowed winner and a strict-enough gap occur in a two-slot example.
Source context: arXiv:2211.11052v1, §3.1, simplex constraints. -/
example : ∃ cost : Fin 2 → ℝ, ∃ winner : Fin 2,
    winner ∈ (Set.univ : Set (Fin 2)) ∧
    (∀ j, j ∈ (Set.univ : Set (Fin 2)) → j ≠ winner →
      cost winner + 1 / 2 ≤ cost j) ∧
    basis winner ∈ simplexOn (Set.univ : Set (Fin 2)) := by
  refine ⟨fun j => if j = 0 then 0 else 1, 0, Set.mem_univ _, ?_,
    basis_mem_simplex _ _ (Set.mem_univ _)⟩
  intro j _ hj
  simp [hj]
  norm_num

end Transformer.ConvexRecall
