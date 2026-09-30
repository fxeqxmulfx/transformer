/-
# AdaFisher: min-max scaling does not preserve inverse approximation error

arXiv:2405.16397v3, Proposition 3.2, Appendix A.2, Part 2.
The printed relative error O(epsilon + lambda) is false even when the
original factors are exactly diagonal. Min-max normalization maps their
positive minimum to zero, so the damped inverse grows like 1/lambda.
-/

import Transformer.AdaFisher.Section3_Spectrum
import Mathlib.Analysis.Matrix.Normed

open scoped BigOperators Matrix Kronecker Matrix.Norms.Elementwise

noncomputable section

namespace Transformer.AdaFisher

/-- Exact diagonal original Kronecker block in the inverse-error
counterexample to Proposition 3.2: factors `(1,2)` and `(1,2)`. -/
def inverseErrorOriginal : Matrix (Fin 2 × Fin 2) (Fin 2 × Fin 2) ℝ :=
  Matrix.diagonal (fun p => (1 + (p.1 : ℝ)) * (1 + (p.2 : ℝ)))

/-- The normalized, damped block for those same factors, Appendix A.2. -/
def inverseErrorApprox (δ : ℝ) : Matrix (Fin 2 × Fin 2) (Fin 2 × Fin 2) ℝ :=
  fisherMatrix δ (fun i => (i : ℝ)) (fun i => (i : ℝ))

/-- The original block is exactly the Kronecker product of diagonal
positive factors `(1,2)`, Appendix A.2, Part 2. Its diagonal approximation
error epsilon is therefore zero in the refuted relative-error claim. -/
theorem inverseErrorOriginal_exactFactors :
    inverseErrorOriginal = Matrix.diagonal (fun i : Fin 2 => 1 + (i : ℝ)) ⊗ₖ
      Matrix.diagonal (fun i : Fin 2 => 1 + (i : ℝ)) := by
  rw [Matrix.diagonal_kronecker_diagonal]
  rfl

/-- The nonconstant factor `(1,2)` normalizes to `(0,1)` exactly,
Proposition 3.2. Hence no degenerate-range extension enters this example. -/
theorem inverseError_normalized_factor :
    minMaxDiagonal (fun i : Fin 2 => 1 + (i : ℝ)) = (fun i : Fin 2 => (i : ℝ)) := by
  funext i
  fin_cases i <;>
    norm_num [minMaxDiagonal, diagonalMin, diagonalMax, Finset.univ_fin2]

/-- The inverse of the original block has max-entry norm one,
Appendix A.2, the norm in its relative inverse-error claim. -/
theorem inverseErrorOriginal_inverse_norm : ‖inverseErrorOriginal⁻¹‖ = 1 := by
  rw [inverseErrorOriginal, diagonal_inverse _ (by intro p; positivity)]
  apply le_antisymm
  · apply (pi_norm_le_iff_of_nonneg (by norm_num : (0 : ℝ) ≤ 1)).mpr
    intro p
    apply (pi_norm_le_iff_of_nonneg (by norm_num : (0 : ℝ) ≤ 1)).mpr
    intro q
    by_cases hpq : p = q
    · subst q
      rcases p with ⟨i, j⟩
      fin_cases i <;> fin_cases j <;> norm_num
    · simp [hpq]
  · calc
      (1 : ℝ) = ‖Matrix.diagonal (fun p : Fin 2 × Fin 2 =>
        ((1 + (p.1 : ℝ)) * (1 + (p.2 : ℝ)))⁻¹) (0, 0) (0, 0)‖ := by norm_num
      _ ≤ _ := Matrix.norm_entry_le_entrywise_sup_norm _

/-- The inverse-error norm is at least `|1/δ-1|`, Proposition 3.2,
Appendix A.2. The original diagonal approximation error is exactly zero. -/
theorem inverseError_lower (δ : ℝ) (hδ : 0 < δ) :
    |δ⁻¹ - 1| ≤ ‖(inverseErrorApprox δ)⁻¹ - inverseErrorOriginal⁻¹‖ := by
  have hnonzero : ∀ p : Fin 2 × Fin 2, fisherDiagonal δ
      (fun i : Fin 2 => (i : ℝ)) (fun i : Fin 2 => (i : ℝ)) p ≠ 0 := by
    intro p
    unfold fisherDiagonal
    positivity
  calc
    _ = ‖((inverseErrorApprox δ)⁻¹ - inverseErrorOriginal⁻¹) (0, 0) (0, 0)‖ := by
      rw [inverseErrorApprox, fisherMatrix, diagonal_inverse _ hnonzero,
        inverseErrorOriginal, diagonal_inverse _ (by intro p; positivity)]
      simp [fisherDiagonal, Real.norm_eq_abs]
    _ ≤ _ := Matrix.norm_entry_le_entrywise_sup_norm _

example : (0 : ℝ) < 1 / 1000 := by norm_num

/-- Refutation of the relative-error O(epsilon + lambda) claim,
Proposition 3.2, Appendix A.2. For epsilon=0 and every candidate constant C,
some positive damping below one violates the proposed C*lambda bound.
The norm is the max-entry matrix norm, for which the original inverse
has norm one; the same exploding entry also obstructs operator norms. -/
theorem relative_inverse_error_counterexample (C : ℝ) (hC : 0 ≤ C) :
    ∃ δ : ℝ, 0 < δ ∧ δ < 1 ∧
      C * δ < ‖(inverseErrorApprox δ)⁻¹ - inverseErrorOriginal⁻¹‖ /
        ‖inverseErrorOriginal⁻¹‖ := by
  let δ : ℝ := 1 / (2 * (C + 1))
  have hδ : 0 < δ := by dsimp [δ]; positivity
  have hδ1 : δ < 1 := by
    dsimp [δ]
    apply (div_lt_one (by positivity)).mpr
    linarith
  have hinv : δ⁻¹ = 2 * (C + 1) := by simp [δ]
  have hlower := (le_abs_self (δ⁻¹ - 1)).trans (inverseError_lower δ hδ)
  rw [hinv] at hlower
  refine ⟨δ, hδ, hδ1, ?_⟩
  rw [inverseErrorOriginal_inverse_norm, div_one]
  nlinarith [mul_le_mul_of_nonneg_left hδ1.le hC]

example : (0 : ℝ) ≤ 10 := by norm_num

end Transformer.AdaFisher
