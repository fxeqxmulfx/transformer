/-
# AdaFisher: efficient damped diagonal Fisher

arXiv:2405.16397v3, §3.2, Proposition 3.2, `eq:FIMDiag`;
Appendix A.2, eigenvalue bounds and the inverse preconditioner.
-/

import Transformer.AdaFisher.Section3_MinMax
import Mathlib.LinearAlgebra.Matrix.Kronecker
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Algebra.Order.Star.Real

open scoped BigOperators Matrix Kronecker

noncomputable section

namespace Transformer.AdaFisher

variable {a b : ℕ}

/-- Damped diagonal block-Kronecker preconditioner, `eq:FIMDiag`, §3.2.
`h` and `s` are the normalized factor diagonals. -/
def fisherDiagonal (δ : ℝ) (h : Fin a → ℝ) (s : Fin b → ℝ) :
    Fin a × Fin b → ℝ := fun p => h p.1 * s p.2 + δ

/-- The matrix represented by the efficient diagonal, Proposition 3.2. -/
def fisherMatrix (δ : ℝ) (h : Fin a → ℝ) (s : Fin b → ℝ) :
    Matrix (Fin a × Fin b) (Fin a × Fin b) ℝ := Matrix.diagonal (fisherDiagonal δ h s)

/-- The entrywise diagonal implementation equals the paper's Kronecker
formula with Tikhonov damping, §3.2, `eq:FIMDiag`. -/
theorem fisherMatrix_eq_kronecker (δ : ℝ) (h : Fin a → ℝ) (s : Fin b → ℝ) :
    fisherMatrix δ h s = Matrix.diagonal h ⊗ₖ Matrix.diagonal s + δ • 1 := by
  rw [Matrix.diagonal_kronecker_diagonal]
  ext p q
  by_cases hpq : p = q
  · subst q
    simp [fisherMatrix, fisherDiagonal]
  · simp [fisherMatrix, hpq]

/-- Entry/eigenvalue bounds in Proposition 3.2, Appendix A.2, Part 2. -/
theorem fisherDiagonal_bounds (δ : ℝ) (h : Fin a → ℝ) (s : Fin b → ℝ)
    (hh : ∀ i, 0 ≤ h i ∧ h i ≤ 1) (hs : ∀ j, 0 ≤ s j ∧ s j ≤ 1)
    (p : Fin a × Fin b) : δ ≤ fisherDiagonal δ h s p ∧ fisherDiagonal δ h s p ≤ 1 + δ := by
  have hprod0 := mul_nonneg (hh p.1).1 (hs p.2).1
  have hprod1 : h p.1 * s p.2 ≤ 1 := by
    nlinarith [(hh p.1).1, (hh p.1).2, (hs p.2).1, (hs p.2).2]
  unfold fisherDiagonal
  constructor <;> linarith

example : (∀ i : Fin 2, 0 ≤ (fun _ => (1 / 2 : ℝ)) i ∧
    (fun _ => (1 / 2 : ℝ)) i ≤ 1) ∧
    (∀ j : Fin 3, 0 ≤ (fun _ => (1 / 3 : ℝ)) j ∧
      (fun _ => (1 / 3 : ℝ)) j ≤ 1) := by norm_num

/-- Positive damping makes the efficient FIM positive definite,
Proposition 3.2, Appendix A.2. -/
theorem fisherMatrix_posDef (δ : ℝ) (h : Fin a → ℝ) (s : Fin b → ℝ)
    (hδ : 0 < δ) (hh : ∀ i, 0 ≤ h i ∧ h i ≤ 1)
    (hs : ∀ j, 0 ≤ s j ∧ s j ≤ 1) : (fisherMatrix δ h s).PosDef := by
  apply Matrix.posDef_diagonal_iff.mpr
  intro p
  exact lt_of_lt_of_le hδ (fisherDiagonal_bounds δ h s hh hs p).1

example : (0 : ℝ) < 1 / 1000 ∧
    (∀ i : Fin 2, 0 ≤ (fun _ => (0 : ℝ)) i ∧ (fun _ => (0 : ℝ)) i ≤ 1) ∧
    (∀ j : Fin 3, 0 ≤ (fun _ => (1 : ℝ)) j ∧ (fun _ => (1 : ℝ)) j ≤ 1) := by
  norm_num

/-- Closed-form preconditioning in Proposition 3.2 and Algorithm 1. -/
def precondition {d : ℕ} (f g : Fin d → ℝ) : Fin d → ℝ := fun i => g i / f i

/-- The diagonal inverse solves the linear system, Proposition 3.2.
Nonzero entries, supplied by positive damping, are essential. -/
theorem diagonal_precondition_solves {d : ℕ} (f g : Fin d → ℝ)
    (hf : ∀ i, f i ≠ 0) : (Matrix.diagonal f).mulVec (precondition f g) = g := by
  funext i
  simp only [Matrix.mulVec_diagonal, precondition]
  field_simp [hf i]

example : ∀ i : Fin 2, (fun _ => (1 / 1000 : ℝ)) i ≠ 0 := by norm_num

/-- Two diagonal solves coincide when damping prevents zero entries,
Proposition 3.2, the closed-form preconditioned gradient. -/
theorem diagonal_precondition_unique {d : ℕ} (f g v : Fin d → ℝ)
    (hf : ∀ i, f i ≠ 0) (hv : (Matrix.diagonal f).mulVec v = g) :
    v = precondition f g := by
  funext i
  apply (eq_div_iff (hf i)).mpr
  have hi := congrFun hv i
  simpa [Matrix.mulVec_diagonal, mul_comm] using hi

example : (∀ i : Fin 1, (fun _ => (1 : ℝ)) i ≠ 0) ∧
    (Matrix.diagonal (fun _ : Fin 1 => (1 : ℝ))).mulVec (fun _ => 2) = (fun _ => 2) := by
  simp

end Transformer.AdaFisher
