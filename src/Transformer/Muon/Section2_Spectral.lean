/-
# Muon — singular coordinates

arXiv:2502.16982, §2.1 and Appendix A. A thin singular factorization uses
orthonormal columns in both rectangular factors. These are genuine
algebraic conditions, not assumptions of the results to be proved.
-/

import Transformer.Muon.Section2_Models
import Mathlib.LinearAlgebra.Matrix.Trace

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.Muon

variable {a b r : ℕ}

/-- Orthonormal columns, arXiv:2502.16982, §2.1 and Appendix A. -/
def OrthonormalColumns (U : Matrix (Fin a) (Fin r) ℝ) : Prop := U.transpose * U = 1

/-- A matrix represented in its thin singular coordinates, arXiv:2502.16982,
§2.1: `U diag(s) Vᵀ`. -/
def singularMatrix (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) : Matrix (Fin a) (Fin b) ℝ :=
  U * Matrix.diagonal s * V.transpose

/-- An exact polar update `U Vᵀ`, arXiv:2502.16982, §2.1 and Appendix A. -/
def polarUpdate (U : Matrix (Fin a) (Fin r) ℝ) (V : Matrix (Fin b) (Fin r) ℝ) :
    Matrix (Fin a) (Fin b) ℝ := U * V.transpose

/-- The spectral inverse-root construction, which is the Moore–Penrose
inverse square root for nonnegative singular values and orthonormal columns,
arXiv:2502.16982, §2.1. The source writes an ordinary
inverse, which is undefined for a tall matrix or a deficient rank. The
spectral pseudoinverse is the correction: zero singular values map to zero. -/
def inverseRootGram (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ) :
    Matrix (Fin a) (Fin a) ℝ :=
  U * Matrix.diagonal (fun k => (s k)⁻¹) * U.transpose

/-- The explicit left Gram matrix, arXiv:2502.16982, §2.1.
Only the right factor's column orthonormality is needed here. -/
theorem singularMatrix_gram (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) (hV : OrthonormalColumns V) :
    singularMatrix U s V * (singularMatrix U s V).transpose =
      U * Matrix.diagonal (fun k => s k ^ 2) * U.transpose := by
  simp only [singularMatrix, Matrix.transpose_mul, Matrix.transpose_transpose]
  have hd : (Matrix.diagonal s).transpose = Matrix.diagonal s := by
    ext i j
    by_cases hij : i = j
    · subst j; simp
    · simp [hij, Ne.symm hij]
  rw [hd]
  simp only [Matrix.mul_assoc]
  rw [← Matrix.mul_assoc V.transpose V, hV, Matrix.one_mul]
  rw [← Matrix.mul_assoc (Matrix.diagonal s) (Matrix.diagonal s), Matrix.diagonal_mul_diagonal]
  simp only [pow_two]

/-- Orthonormality is satisfiable: the one-dimensional identity, as used for
the singular-coordinate examples in arXiv:2502.16982, §2.1. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [OrthonormalColumns]

/-- Applying the inverse root to a matrix with nonzero singular values gives
`U Vᵀ`, arXiv:2502.16982, §2.1, after `eq:Ot`. This proves the stated identity
with the pseudoinverse convention appropriate for rectangular matrices. -/
theorem inverseRootGram_mul (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) (hU : OrthonormalColumns U)
    (hs : ∀ k, s k ≠ 0) :
    inverseRootGram U s * singularMatrix U s V = polarUpdate U V := by
  simp only [inverseRootGram, singularMatrix, polarUpdate, Matrix.mul_assoc]
  rw [← Matrix.mul_assoc U.transpose U, hU, Matrix.one_mul]
  rw [← Matrix.mul_assoc (Matrix.diagonal (fun k => (s k)⁻¹)) (Matrix.diagonal s),
    Matrix.diagonal_mul_diagonal]
  have hd : Matrix.diagonal (fun k => (s k)⁻¹ * s k) = (1 : Matrix (Fin r) (Fin r) ℝ) := by
    simp [hs]
  rw [hd, Matrix.one_mul]

/-- The inverse-root hypotheses are satisfiable, arXiv:2502.16982, §2.1:
the identity factors and positive singular value `2`. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    ∀ k : Fin 1, (fun _ : Fin 1 => (2 : ℝ)) k ≠ 0 := by
  simp [OrthonormalColumns]

/-- Partial isometry identity for the exact polar update, arXiv:2502.16982,
Appendix A. -/
theorem polarUpdate_gram (U : Matrix (Fin a) (Fin r) ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) (hU : OrthonormalColumns U) :
    (polarUpdate U V).transpose * polarUpdate U V = V * V.transpose := by
  simp only [polarUpdate, Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc]
  rw [← Matrix.mul_assoc U.transpose U, hU, Matrix.one_mul]

/-- The partial-isometry hypothesis is satisfiable, arXiv:2502.16982, Appendix A. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [OrthonormalColumns]

end Transformer.Muon
