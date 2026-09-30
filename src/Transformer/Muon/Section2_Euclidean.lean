/-
# Muon — the Euclidean operator constraint

arXiv:2502.16982, §2.1, “Steepest Descent Under Norm Constraints”. The
spectral unit ball is expressed by its defining action on every Euclidean
column vector. Squaring its two nonnegative norms is equivalent to the
usual induced-operator-norm condition.
-/

import Transformer.Muon.AppendixA_RMS
import Mathlib.Data.Matrix.Basis

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.Muon

variable {a b r : ℕ}

/-- Frobenius pairing for the first-order objective in arXiv:2502.16982, §2.1. -/
def frobeniusPair (X Y : Matrix (Fin a) (Fin b) ℝ) : ℝ :=
  ∑ i, ∑ j, X i j * Y i j

/-- The induced Euclidean operator unit ball, arXiv:2502.16982, §2.1,
“Steepest Descent Under Norm Constraints”. -/
def EuclideanContraction (D : Matrix (Fin a) (Fin b) ℝ) : Prop :=
  ∀ X : Matrix (Fin b) (Fin 1) ℝ, squaredFrobenius (D * X) ≤ squaredFrobenius X

/-- Trace form of the first-order objective, arXiv:2502.16982, §2.1. -/
theorem frobeniusPair_eq_trace (X Y : Matrix (Fin a) (Fin b) ℝ) :
    frobeniusPair X Y = (X.transpose * Y).trace := by
  simp only [frobeniusPair, Matrix.trace, Matrix.diag, Matrix.mul_apply, Matrix.transpose_apply]
  exact Finset.sum_comm

/-- Euclidean squared distance expansion, used for the operator constraint in
arXiv:2502.16982, §2.1. -/
theorem squaredFrobenius_sub (X Y : Matrix (Fin a) (Fin b) ℝ) :
    squaredFrobenius (X - Y) =
      squaredFrobenius X + squaredFrobenius Y - 2 * frobeniusPair X Y := by
  unfold squaredFrobenius frobeniusPair
  simp only [Matrix.sub_apply]
  calc
    _ = ∑ i, ∑ j, (X i j ^ 2 + Y i j ^ 2 - 2 * (X i j * Y i j)) := by
      apply Finset.sum_congr rfl
      intro i hi
      apply Finset.sum_congr rfl
      intro j hj
      ring
    _ = _ := by simp [Finset.sum_sub_distrib, Finset.sum_add_distrib, Finset.mul_sum]

/-- Orthonormal columns preserve Euclidean energy,
arXiv:2502.16982, §2.1 and Appendix A. -/
theorem squaredFrobenius_frame_mul (U : Matrix (Fin a) (Fin r) ℝ)
    (hU : OrthonormalColumns U) (X : Matrix (Fin r) (Fin b) ℝ) :
    squaredFrobenius (U * X) = squaredFrobenius X := by
  rw [squaredFrobenius_eq_trace, Matrix.transpose_mul]
  simp only [Matrix.mul_assoc]
  rw [← Matrix.mul_assoc U.transpose U, hU, Matrix.one_mul, ← squaredFrobenius_eq_trace]

/-- An energy-preserving frame exists, arXiv:2502.16982, §2.1. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [OrthonormalColumns]

/-- The cross term in projection energy is the energy of the adjoint image,
arXiv:2502.16982, §2.1, the Euclidean operator constraint. -/
theorem projection_cross_term (U : Matrix (Fin a) (Fin r) ℝ)
    (X : Matrix (Fin a) (Fin b) ℝ) :
    frobeniusPair X (U * (U.transpose * X)) = squaredFrobenius (U.transpose * X) := by
  rw [frobeniusPair_eq_trace, squaredFrobenius_eq_trace, Matrix.transpose_mul,
    Matrix.transpose_transpose]
  simp only [Matrix.mul_assoc]

/-- The adjoint of an orthonormal frame is a contraction, arXiv:2502.16982,
§2.1, “Steepest Descent Under Norm Constraints”. -/
theorem frame_transpose_contraction (U : Matrix (Fin a) (Fin r) ℝ)
    (hU : OrthonormalColumns U) : EuclideanContraction U.transpose := by
  intro X
  have h := squaredFrobenius_nonneg (X - U * (U.transpose * X))
  rw [squaredFrobenius_sub, squaredFrobenius_frame_mul U hU,
    projection_cross_term] at h
  linarith

/-- The transpose-contraction hypothesis is satisfiable, arXiv:2502.16982, §2.1. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [OrthonormalColumns]

/-- The exact polar update belongs to the spectral unit ball,
arXiv:2502.16982, §2.1, “Steepest Descent Under Norm Constraints”. -/
theorem polarUpdate_contraction (U : Matrix (Fin a) (Fin r) ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) (hU : OrthonormalColumns U)
    (hV : OrthonormalColumns V) : EuclideanContraction (polarUpdate U V) := by
  intro X
  rw [polarUpdate, Matrix.mul_assoc, squaredFrobenius_frame_mul U hU]
  exact frame_transpose_contraction V hV X

/-- Exact polar updates with orthonormal frames exist, arXiv:2502.16982, §2.1. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [OrthonormalColumns]

/-- An entry's square is bounded by the matrix's Euclidean energy,
arXiv:2502.16982, §2.1, the operator constraint. -/
theorem entry_sq_le_squaredFrobenius (X : Matrix (Fin a) (Fin b) ℝ) (i : Fin a) (j : Fin b) :
    X i j ^ 2 ≤ squaredFrobenius X := by
  apply le_trans (Finset.single_le_sum (fun k _ => sq_nonneg (X i k)) (Finset.mem_univ j))
  exact Finset.single_le_sum
    (fun k _ => Finset.sum_nonneg fun l _ => sq_nonneg (X k l)) (Finset.mem_univ i)

/-- A coordinate unit column has energy one, arXiv:2502.16982, §2.1. -/
theorem unitColumn_energy (k : Fin r) :
    squaredFrobenius (Matrix.single k (0 : Fin 1) (1 : ℝ)) = 1 := by
  unfold squaredFrobenius
  simp only [Fin.sum_univ_one]
  simp [Matrix.single_apply]

end Transformer.Muon
