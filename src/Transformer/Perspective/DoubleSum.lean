/-
# Finite double sums over `Idx n × Idx n`

The energies of the survey are double sums over the token indices:
`𝖤_0 = n⁻¹ Σ_i Σ_j ⟨x_i, x_j⟩` and `𝖤_β = (2β)⁻¹ Σ_i Σ_j e^{β ⟨x_i, x_j⟩}`.
Differentiating them along a curve, subtracting two of them and bounding the
result are the same three manipulations every time, and they are collected
here rather than repeated: term-by-term differentiation, pulling a constant
factor or a difference through both sums, and the triangle inequality against
a uniform bound.

Nothing here is specific to the survey; it is the arithmetic of `Finset.univ`.
-/

import Transformer.Basic
import Mathlib.Analysis.Calculus.Deriv.Add

open scoped BigOperators

namespace Transformer
namespace Perspective

variable (n : ℕ)

/-- A finite double sum is differentiated term by term. -/
theorem hasDerivAt_double_sum (u : Idx n → Idx n → ℝ → ℝ) (c : Idx n → Idx n → ℝ) (t : ℝ)
    (h : ∀ i j : Idx n, HasDerivAt (u i j) (c i j) t) :
    HasDerivAt (fun s => ∑ i : Idx n, ∑ j : Idx n, u i j s)
      (∑ i : Idx n, ∑ j : Idx n, c i j) t :=
  HasDerivAt.fun_sum fun i _ => HasDerivAt.fun_sum fun j _ => h i j

/-- The hypothesis of `hasDerivAt_double_sum` is satisfiable: constants have
derivative `0`. -/
example (t : ℝ) : ∀ _ _ : Idx n, HasDerivAt (fun _ : ℝ => (0 : ℝ)) 0 t :=
  fun _ _ => hasDerivAt_const _ _

/-- A constant factor passes through both sums. -/
theorem const_mul_double_sum (c : ℝ) (u : Idx n → Idx n → ℝ) :
    c * ∑ i : Idx n, ∑ j : Idx n, u i j = ∑ i : Idx n, ∑ j : Idx n, c * u i j := by
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun i _ => Finset.mul_sum _ _ _

/-- A difference of double sums is the double sum of the differences. -/
theorem double_sum_sub (u v : Idx n → Idx n → ℝ) :
    (∑ i : Idx n, ∑ j : Idx n, u i j) - ∑ i : Idx n, ∑ j : Idx n, v i j
      = ∑ i : Idx n, ∑ j : Idx n, (u i j - v i j) := by
  rw [← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl fun i _ => (Finset.sum_sub_distrib _ _).symm

/-- A uniform bound on the terms bounds the double sum by `n²` times it. -/
theorem abs_double_sum_le (u : Idx n → Idx n → ℝ) (M : ℝ)
    (h : ∀ i j : Idx n, |u i j| ≤ M) :
    |∑ i : Idx n, ∑ j : Idx n, u i j| ≤ (n : ℝ) ^ 2 * M := by
  have hrow : ∀ i : Idx n, |∑ j : Idx n, u i j| ≤ (n : ℝ) * M := by
    intro i
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    calc ∑ j : Idx n, |u i j| ≤ ∑ _j : Idx n, M := Finset.sum_le_sum fun j _ => h i j
      _ = (n : ℝ) * M := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  calc ∑ i : Idx n, |∑ j : Idx n, u i j| ≤ ∑ _i : Idx n, (n : ℝ) * M :=
        Finset.sum_le_sum fun i _ => hrow i
    _ = (n : ℝ) ^ 2 * M := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        ring

/-- The hypothesis of `abs_double_sum_le` is satisfiable: the zero family is
bounded by `0`. -/
example : ∀ _ _ : Idx n, |(0 : ℝ)| ≤ 0 := fun _ _ => by simp

end Perspective
end Transformer
