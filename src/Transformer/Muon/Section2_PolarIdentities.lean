/-
# Muon — square roots and exact normalization

arXiv:2502.16982, §2.1 and Appendix A. The constructions are verified
algebraically, including positivity of the spectral square root. This
separates an exact polar update from its finite polynomial approximation.
-/

import Transformer.Muon.Section2_SteepestDescent
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Algebra.Order.Star.Real

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.Muon

variable {a b r : ℕ}

/-- The left-Gram square root in nonnegative singular coordinates,
arXiv:2502.16982, §2.1, after `eq:Ot`. -/
def rootGram (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ) :
    Matrix (Fin a) (Fin a) ℝ := singularMatrix U s U

/-- The spectral root squares to the actual Gram matrix,
arXiv:2502.16982, §2.1, after `eq:Ot`. -/
theorem rootGram_sq (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) (hU : OrthonormalColumns U)
    (hV : OrthonormalColumns V) :
    rootGram U s * rootGram U s =
      singularMatrix U s V * (singularMatrix U s V).transpose := by
  rw [rootGram, singularMatrix_mul U s U s U hU, singularMatrix_gram U s V hV]
  simp only [singularMatrix, pow_two]

/-- The Gram-root hypotheses are satisfiable, arXiv:2502.16982, §2.1. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [OrthonormalColumns]

/-- The spectral root is positive semidefinite when its coefficients are
nonnegative singular values, arXiv:2502.16982, §2.1. -/
theorem rootGram_posSemidef (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ)
    (hs : ∀ k, 0 ≤ s k) : (rootGram U s).PosSemidef := by
  have hd := (Matrix.posSemidef_diagonal_iff.mpr hs).mul_mul_conjTranspose_same U
  simpa only [rootGram, singularMatrix, Matrix.conjTranspose_eq_transpose_of_trivial] using hd

/-- Nonnegative singular values exist, arXiv:2502.16982, §2.1. -/
example : ∀ k : Fin 1, (0 : ℝ) ≤ (fun _ : Fin 1 => (2 : ℝ)) k := by norm_num

/-- The inverse root times the root is the projector onto the active singular
subspace. In a full row-rank factorization this projector is the identity.
Source: arXiv:2502.16982, §2.1, corrected inverse in `eq:Ot`. -/
theorem inverseRootGram_mul_root (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ)
    (hU : OrthonormalColumns U) (hs : ∀ k, s k ≠ 0) :
    inverseRootGram U s * rootGram U s = U * U.transpose := by
  exact inverseRootGram_mul U s U hU hs

/-- An active singular subspace exists, arXiv:2502.16982, §2.1. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    ∀ k : Fin 1, (fun _ : Fin 1 => (2 : ℝ)) k ≠ 0 := by
  simp [OrthonormalColumns]

/-- The full-rank exact inverse-root update has the RMS stated in Lemma 1.
This includes the momentum matrix, its nonzero singular coefficients, and
the corrected spectral inverse explicitly.
Source: arXiv:2502.16982, §2.1–2.2, `lemma:updaterms`; Appendix A. -/
theorem inverseRootGram_update_rms (U : Matrix (Fin a) (Fin (min a b)) ℝ)
    (s : Fin (min a b) → ℝ) (V : Matrix (Fin b) (Fin (min a b)) ℝ)
    (ha : 0 < a) (hb : 0 < b) (hU : OrthonormalColumns U)
    (hV : OrthonormalColumns V) (hs : ∀ k, s k ≠ 0) :
    matrixRMS (inverseRootGram U s * singularMatrix U s V) =
      Real.sqrt (1 / max (a : ℝ) b) := by
  rw [inverseRootGram_mul U s V hU hs, fullRank_update_rms U V ha hb hU hV]

/-- All hypotheses of the exact update RMS hold at identity frames and
singular value two, arXiv:2502.16982, §2.1–2.2. -/
example : 0 < (1 : ℕ) ∧ 0 < (1 : ℕ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    ∀ k : Fin 1, (fun _ : Fin 1 => (2 : ℝ)) k ≠ 0 := by
  norm_num [OrthonormalColumns]

/-- Frobenius energy is the sum of the squared singular values,
arXiv:2502.16982, §2.1 and §3.4, “Dynamics of Singular Spectrum”. -/
theorem squaredFrobenius_singularMatrix (U : Matrix (Fin a) (Fin r) ℝ)
    (s : Fin r → ℝ) (V : Matrix (Fin b) (Fin r) ℝ)
    (hU : OrthonormalColumns U) (hV : OrthonormalColumns V) :
    squaredFrobenius (singularMatrix U s V) = ∑ k, s k ^ 2 := by
  rw [squaredFrobenius_eq_trace, singularMatrix_transpose,
    singularMatrix_mul V s U s V hU]
  unfold singularMatrix
  rw [Matrix.trace_mul_cycle, hV, Matrix.one_mul, Matrix.trace_diagonal]
  simp only [pow_two]

/-- Orthonormal singular frames exist, arXiv:2502.16982, §2.1. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [OrthonormalColumns]

/-- Squared energy scales quadratically, arXiv:2502.16982, §2.1 and Appendix A. -/
theorem squaredFrobenius_smul (c : ℝ) (X : Matrix (Fin a) (Fin b) ℝ) :
    squaredFrobenius (c • X) = c ^ 2 * squaredFrobenius X := by
  simp only [squaredFrobenius, Matrix.smul_apply, smul_eq_mul, mul_pow, Finset.mul_sum]

/-- Frobenius normalization gives energy one for nonzero-energy input. The
source's normalization requires this condition to normalize to a unit size.
Source: arXiv:2502.16982, §2.1, before `eq:iteration`. -/
theorem schulzInitial_energy (M : Matrix (Fin a) (Fin b) ℝ)
    (hM : 0 < squaredFrobenius M) : squaredFrobenius (schulzInitial M) = 1 := by
  rw [schulzInitial, squaredFrobenius_smul, inv_pow, Real.sq_sqrt hM.le]
  exact inv_mul_cancel₀ hM.ne'

/-- A nonzero-energy momentum exists, arXiv:2502.16982, §2.1. -/
example : 0 < squaredFrobenius (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  norm_num [squaredFrobenius]

end Transformer.Muon
