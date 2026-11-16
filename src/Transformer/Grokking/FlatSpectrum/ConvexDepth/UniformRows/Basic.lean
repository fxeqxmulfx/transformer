module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedRows.Projection

/-!
# Exact distance from a class-uniform row

Source: `review/flat_spectrum.md`, CM69. At a fixed actual row mass, subtracting
its mean splits the distance to any class-uniform proposal into a constant-row
term and a nonnegative variance. The class type is explicitly nonempty when
mass is divided by its cardinality to recover a row with that mass.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.UniformRows

open scoped BigOperators

variable {C : Type*} [Fintype C] [Nonempty C]

/-- CM69's nonempty raw-class type has a positive actual cardinality. -/
theorem card_pos : (0 : ℝ) < Fintype.card C := by
  exact_mod_cast (Fintype.card_pos_iff.mpr (inferInstance : Nonempty C))

/-- A constant row at mass divided by class count has that mass, CM69. -/
theorem sum_mean (m : ℝ) : (∑ _c : C, m / Fintype.card C) = m := by
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  exact mul_div_cancel₀ m (ne_of_gt (card_pos (C := C)))

/-- Centering an arbitrary row at its actual mean removes its sum, CM69. -/
theorem sum_centered (x : C → ℝ) (m : ℝ) (hm : ∑ c, x c = m) :
    (∑ c, (x c - m / Fintype.card C)) = 0 := by
  rw [Finset.sum_sub_distrib, hm, sum_mean]
  ring

/-- Exact squared distance to any class-uniform proposal, CM69. All class
coordinates remain in the variance, even when the mass or proposal is zero. -/
theorem square_decomposition (x : C → ℝ) (m u : ℝ) (hm : ∑ c, x c = m) :
    (∑ c, (x c - u) ^ 2) =
      (∑ c, (x c - m / Fintype.card C) ^ 2) +
        Fintype.card C * (m / Fintype.card C - u) ^ 2 := by
  calc
    _ = ∑ c, ((x c - m / Fintype.card C) ^ 2 +
        (2 * (m / Fintype.card C - u)) * (x c - m / Fintype.card C) +
          (m / Fintype.card C - u) ^ 2) := by
      apply Finset.sum_congr rfl
      intro c hc
      ring
    _ = _ := by
      rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum,
        sum_centered x m hm]
      simp

omit [Nonempty C] in
/-- The complete centered-row remainder is nonnegative, CM69. -/
theorem square_remainder_nonneg (x : C → ℝ) (m : ℝ) :
    0 ≤ ∑ c, (x c - m / Fintype.card C) ^ 2 := by
  exact Finset.sum_nonneg fun c hc => sq_nonneg _

omit [Nonempty C] in
/-- The remainder vanishes exactly when every coordinate is its stated
mean, CM69; a nonuniform row cannot silently take the uniform branch. -/
theorem square_remainder_zero_iff (x : C → ℝ) (m : ℝ) :
    (∑ c, (x c - m / Fintype.card C) ^ 2) = 0 ↔
      ∀ c, x c = m / Fintype.card C := by
  rw [Finset.sum_eq_zero_iff_of_nonneg fun c hc => sq_nonneg _]
  simp only [Finset.mem_univ, forall_const, sq_eq_zero_iff, sub_eq_zero]

/-- Among all rows with the same mass, their common mean is closest to a
class-uniform proposal, CM69. Nonnegativity is not needed for this identity. -/
theorem uniform_row_closest (x : C → ℝ) (m u : ℝ) (hm : ∑ c, x c = m) :
    Fintype.card C * (m / Fintype.card C - u) ^ 2 ≤ ∑ c, (x c - u) ^ 2 := by
  rw [square_decomposition x m u hm]
  exact le_add_of_nonneg_left (square_remainder_nonneg x m)

/-- Every minimizing row equals its mass divided by actual class count,
CM69. This is uniqueness at fixed mass, not a choice of an unseen raw label. -/
theorem uniform_row_unique (x : C → ℝ) (m u : ℝ) (hm : ∑ c, x c = m)
    (hmin : (∑ c, (x c - u) ^ 2) ≤ Fintype.card C * (m / Fintype.card C - u) ^ 2) :
    ∀ c, x c = m / Fintype.card C := by
  rw [square_decomposition x m u hm] at hmin
  apply (square_remainder_zero_iff x m).mp
  have hn := square_remainder_nonneg x m
  linarith

end Transformer.Grokking.FlatSpectrum.ConvexDepth.UniformRows
