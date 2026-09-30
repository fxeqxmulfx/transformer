/-
# DASH — shifted ReLU and rank of the spectral inverse

arXiv:2602.02016v2, Appendix B, “Shifted-ReLU heuristic”.
Inactive eigenvalues are set to zero before taking inverse powers.
-/

import Transformer.DASH.Section3_Spectral
import Mathlib.LinearAlgebra.Matrix.Rank

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- Shifted ReLU of the corrected spectrum,
arXiv:2602.02016v2, Appendix B, `ReLU(λ-ε)`. -/
def shiftedReLU (ε value : ℝ) : ℝ := max (value - ε) 0

/-- Spectral power of active eigenvalues only. This uses an explicit zero
branch instead of applying a negative power to a filtered zero eigenvalue.
Source: arXiv:2602.02016v2, Appendix B, shifted-ReLU inverse construction. -/
def thresholdPower (ε exponent value : ℝ) : ℝ :=
  if ε < value then (value - ε) ^ exponent else 0

/-- The shifted-ReLU spectrum is nonnegative,
arXiv:2602.02016v2, Appendix B. -/
theorem shiftedReLU_nonneg (ε value : ℝ) : 0 ≤ shiftedReLU ε value := le_max_right _ _

/-- The active eigenvalues are exactly those exceeding the threshold,
arXiv:2602.02016v2, Appendix B, the rank `r_ε`. -/
theorem shiftedReLU_pos_iff (ε value : ℝ) : 0 < shiftedReLU ε value ↔ ε < value := by
  simp [shiftedReLU, lt_max_iff, sub_pos]

/-- Taking a real power preserves nonzeroness of every active eigenvalue.
Source: arXiv:2602.02016v2, Appendix B, `λ_i^(-1/p)` on positive entries. -/
theorem thresholdPower_ne_zero_iff (ε exponent value : ℝ) :
    thresholdPower ε exponent value ≠ 0 ↔ ε < value := by
  by_cases h : ε < value
  · simp only [thresholdPower, ite_eq_left h]
    exact iff_of_true (Real.rpow_pos_of_pos (sub_pos.mpr h) exponent).ne' h
  · simp [thresholdPower, h]

/-- Orthogonal spectral reconstruction preserves diagonal rank, even when
some eigenvalues are zero, arXiv:2602.02016v2, Appendix B, the low-rank claim. -/
theorem spectralMatrix_rank (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ)
    (hQ : Orthogonal Q) :
    (spectralMatrix Q s).rank = Fintype.card {i : Fin n // s i ≠ 0} := by
  classical
  have hcompress : Q.transpose * spectralMatrix Q s * Q = Matrix.diagonal s := by
    simp only [spectralMatrix, Muon.singularMatrix, Matrix.mul_assoc]
    rw [← Matrix.mul_assoc Q.transpose Q, hQ.1, Matrix.one_mul, Matrix.mul_one]
  have hlower := Matrix.rank_mul_le_left (Q.transpose * spectralMatrix Q s) Q
  rw [hcompress] at hlower
  have hupper : (spectralMatrix Q s).rank ≤ (Matrix.diagonal s).rank :=
    (Matrix.rank_mul_le_left (Q * Matrix.diagonal s) Q.transpose).trans
      (Matrix.rank_mul_le_right Q (Matrix.diagonal s))
  exact (le_antisymm hupper
    (hlower.trans (Matrix.rank_mul_le_right Q.transpose _))).trans (Matrix.rank_diagonal s)

/-- Rank assumptions are satisfiable, arXiv:2602.02016v2, Appendix B. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- The thresholded inverse has exactly the claimed active rank `r_ε`.
This is matrix rank, not a definition that merely labels a count “rank”.
Source: arXiv:2602.02016v2, Appendix B, “Shifted-ReLU heuristic”. -/
theorem threshold_inverse_rank (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ)
    (ε p : ℝ) (hQ : Orthogonal Q) :
    (spectralMatrix Q (fun i => thresholdPower ε (-(1 / p)) (s i))).rank =
      Fintype.card {i : Fin n // ε < s i} := by
  classical
  rw [spectralMatrix_rank Q _ hQ]
  exact Fintype.card_congr (Equiv.subtypeEquivRight
    (fun i => thresholdPower_ne_zero_iff ε (-(1 / p)) (s i)))

/-- Threshold-rank assumptions are satisfiable, arXiv:2602.02016v2, Appendix B. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

end Transformer.DASH
