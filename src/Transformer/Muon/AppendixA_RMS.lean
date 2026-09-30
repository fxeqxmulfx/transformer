/-
# Muon — the update RMS lemma

arXiv:2502.16982, Lemma 1 (`lemma:updaterms`) and Appendix A.
The first displayed line of the paper's proof drops the cross terms before
justifying their cancellation. The trace argument below cancels them using
both sets of orthonormal columns and works in either rectangular orientation.
-/

import Transformer.Muon.Section2_Spectral

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.Muon

variable {a b r : ℕ}

/-- Nonnegativity of the squared Frobenius norm, arXiv:2502.16982, Appendix A. -/
theorem squaredFrobenius_nonneg (X : Matrix (Fin a) (Fin b) ℝ) :
    0 ≤ squaredFrobenius X := by
  exact Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => sq_nonneg (X i j)

/-- The squared Frobenius norm is the trace of the right Gram matrix,
arXiv:2502.16982, Appendix A. -/
theorem squaredFrobenius_eq_trace (X : Matrix (Fin a) (Fin b) ℝ) :
    squaredFrobenius X = (X.transpose * X).trace := by
  simp only [squaredFrobenius, Matrix.trace, Matrix.diag, Matrix.mul_apply,
    Matrix.transpose_apply, pow_two]
  exact Finset.sum_comm

/-- The exact rank-`r` polar factor has squared Frobenius norm `r`,
arXiv:2502.16982, Appendix A. This supplies the missing cross-term
cancellation in the source proof. -/
theorem squaredFrobenius_polarUpdate (U : Matrix (Fin a) (Fin r) ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) (hU : OrthonormalColumns U)
    (hV : OrthonormalColumns V) :
    squaredFrobenius (polarUpdate U V) = r := by
  rw [squaredFrobenius_eq_trace, polarUpdate_gram U V hU,
    Matrix.trace_mul_comm, hV, Matrix.trace_one]
  simp

/-- Both orthonormal-column hypotheses hold at the one-dimensional identity,
arXiv:2502.16982, Appendix A. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [OrthonormalColumns]

/-- The general rank formula proved in arXiv:2502.16982, Appendix A:
`RMS(U Vᵀ) = √(r/(a b))`. -/
theorem matrixRMS_polarUpdate (U : Matrix (Fin a) (Fin r) ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) (hU : OrthonormalColumns U)
    (hV : OrthonormalColumns V) :
    matrixRMS (polarUpdate U V) = Real.sqrt ((r : ℝ) / ((a : ℝ) * b)) := by
  rw [matrixRMS, squaredFrobenius_polarUpdate U V hU hV]

/-- A witness for the rank formula's hypotheses, arXiv:2502.16982, Appendix A. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [OrthonormalColumns]

/-- The rectangular dimension identity behind Lemma 1,
arXiv:2502.16982, §2.2 and Appendix A. Positive dimensions are required. -/
theorem min_div_product (ha : 0 < a) (hb : 0 < b) :
    ((min a b : ℕ) : ℝ) / ((a : ℝ) * b) = 1 / max (a : ℝ) b := by
  have ha' : (a : ℝ) ≠ 0 := by positivity
  have hb' : (b : ℝ) ≠ 0 := by positivity
  rcases le_total a b with h | h
  · rw [min_eq_left h, max_eq_right (by exact_mod_cast h)]
    field_simp
  · rw [min_eq_right h, max_eq_left (by exact_mod_cast h)]
    field_simp

/-- Positive dimensions exist, arXiv:2502.16982, Lemma 1. -/
example : 0 < (1 : ℕ) ∧ 0 < (1 : ℕ) := by norm_num

/-- **Lemma 1 (`lemma:updaterms`), corrected scope.** The exact, full-rank
polar update of an `a × b` momentum matrix has RMS `√(1/max(a,b))`.

The source says “a full-rank matrix parameter”. The needed rank is that of
the momentum being orthogonalized, not of the weights: full-rank weights
can have zero gradient. Also the identity is for exact orthogonalization,
not for the five-step approximation with the printed polynomial.
The thin SVD's two frames have `min a b` orthonormal columns.

Source: arXiv:2502.16982, §2.2, `lemma:updaterms`; Appendix A. -/
theorem fullRank_update_rms (U : Matrix (Fin a) (Fin (min a b)) ℝ)
    (V : Matrix (Fin b) (Fin (min a b)) ℝ) (ha : 0 < a) (hb : 0 < b)
    (hU : OrthonormalColumns U) (hV : OrthonormalColumns V) :
    matrixRMS (polarUpdate U V) = Real.sqrt (1 / max (a : ℝ) b) := by
  rw [matrixRMS_polarUpdate U V hU hV, min_div_product ha hb]

/-- The corrected lemma has satisfiable hypotheses, arXiv:2502.16982, Lemma 1. -/
example : 0 < (1 : ℕ) ∧ 0 < (1 : ℕ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  norm_num [OrthonormalColumns]

/-- RMS scales by the absolute scalar, arXiv:2502.16982, §2.2 and §3.1. -/
theorem matrixRMS_smul (c : ℝ) (X : Matrix (Fin a) (Fin b) ℝ) :
    matrixRMS (c • X) = |c| * matrixRMS X := by
  have henergy : squaredFrobenius (c • X) = c ^ 2 * squaredFrobenius X := by
    simp only [squaredFrobenius, Matrix.smul_apply, smul_eq_mul, mul_pow,
      Finset.mul_sum]
  rw [matrixRMS, henergy, mul_div_assoc, Real.sqrt_mul (sq_nonneg c),
    Real.sqrt_sq_eq_abs, matrixRMS]

/-- Zero momentum is a counterexample to interpreting Lemma 1 as a statement
about the rank of the weights alone: the one-dimensional identity weight is
invertible, but its Muon update is zero for a zero momentum input.
Source: arXiv:2502.16982, §2.1–2.2, `lemma:updaterms`. -/
theorem fullRank_weight_zero_update :
    (1 : Matrix (Fin 1) (Fin 1) ℝ).det ≠ 0 ∧
      matrixRMS (approximatePolar (0 : Matrix (Fin 1) (Fin 1) ℝ)) = 0 ∧
      Real.sqrt (1 / max (1 : ℝ) 1) = 1 := by
  norm_num [approximatePolar, schulzIterate_zero, matrixRMS, squaredFrobenius]

end Transformer.Muon
