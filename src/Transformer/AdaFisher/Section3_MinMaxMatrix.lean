/-
# AdaFisher: the min-max matrix formula and Loewner bounds

arXiv:2405.16397v3, Proposition 3.2, Appendix A.2, Part 2.
The sandwich formula requires a strictly positive diagonal range.
Loewner bounds are expressed by positive semidefiniteness of M' and I-M'.
-/

import Transformer.AdaFisher.Section3_Spectrum
import Mathlib.Analysis.Real.Sqrt

noncomputable section

namespace Transformer.AdaFisher

variable {n : ℕ} [NeZero n]

/-- Matrix form of the normalized diagonal in Proposition 3.2,
Appendix A.2, Part 2. -/
def minMaxMatrix (x : Fin n → ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.diagonal (minMaxDiagonal x)

/-- The normalized matrix is positive semidefinite, Appendix A.2,
the lower Loewner bound 0≼M'. -/
theorem minMaxMatrix_posSemidef (x : Fin n → ℝ) : (minMaxMatrix x).PosSemidef := by
  exact Matrix.posSemidef_diagonal_iff.mpr (fun i => (minMaxDiagonal_bounds x i).1)

/-- The normalized matrix is at most the identity in Loewner order,
Appendix A.2: I-M' is positive semidefinite. -/
theorem minMaxMatrix_le_identity (x : Fin n → ℝ) :
    (1 - minMaxMatrix x).PosSemidef := by
  have heq : 1 - minMaxMatrix x = Matrix.diagonal (fun i => 1 - minMaxDiagonal x i) := by
    rw [← Matrix.diagonal_one, minMaxMatrix, Matrix.diagonal_sub]
  rw [heq]
  exact Matrix.posSemidef_diagonal_iff.mpr
    (fun i => sub_nonneg.mpr (minMaxDiagonal_bounds x i).2)

/-- The exact sandwich formula from Appendix A.2, Part 2.
The source defines D=diag(sqrt(max-min)); its inverse exists only when
max>min. This hypothesis is stated explicitly and the formula is proved
for the ordinary matrix inverse. -/
theorem minMaxMatrix_sandwich (x : Fin n → ℝ)
    (hrange : diagonalMin x < diagonalMax x) :
    let D := Matrix.diagonal (fun _ : Fin n =>
      Real.sqrt (diagonalMax x - diagonalMin x))
    minMaxMatrix x = D⁻¹ *
      (Matrix.diagonal x - diagonalMin x • (1 : Matrix (Fin n) (Fin n) ℝ)) * D⁻¹ := by
  dsimp only
  have hr : 0 < diagonalMax x - diagonalMin x := sub_pos.mpr hrange
  have hs : Real.sqrt (diagonalMax x - diagonalMin x) ≠ 0 := Real.sqrt_ne_zero'.mpr hr
  rw [diagonal_inverse _ (fun _ => hs)]
  have hscalar : diagonalMin x • (1 : Matrix (Fin n) (Fin n) ℝ) =
      Matrix.diagonal (fun _ : Fin n => diagonalMin x) := by
    ext i j
    by_cases hij : i = j <;> simp [hij]
  rw [hscalar, Matrix.diagonal_sub, Matrix.diagonal_mul_diagonal,
    Matrix.diagonal_mul_diagonal]
  unfold minMaxMatrix
  apply congrArg Matrix.diagonal
  funext i
  rw [minMaxDiagonal_eq x hrange i]
  have hsq := Real.sq_sqrt hr.le
  field_simp
  rw [hsq]

example : diagonalMin (fun i : Fin 2 => 1 + (i : ℝ)) <
    diagonalMax (fun i : Fin 2 => 1 + (i : ℝ)) := by
  have h0 := diagonalMin_le (fun i : Fin 2 => 1 + (i : ℝ)) 0
  have h1 := le_diagonalMax (fun i : Fin 2 => 1 + (i : ℝ)) 1
  norm_num at h0 h1
  linarith

end Transformer.AdaFisher
