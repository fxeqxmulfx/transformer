/-
# Muon — rank and the matrix-level RMS statement

arXiv:2502.16982, §2.1–2.2 and Appendix A. Exact polar orthogonalization is
specified by the active thin SVD, including positive singular values.
The rank formula connects the singular-coordinate result with the
paper's full-rank condition on the momentum matrix itself.
-/

import Transformer.Muon.Section2_PolarIdentities
import Mathlib.LinearAlgebra.Matrix.Rank

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.Muon

variable {a b r : ℕ}

/-- A nonzero thin singular factorization has rank exactly the number of its
active singular values, arXiv:2502.16982, §2.1 and Appendix A. -/
theorem singularMatrix_rank (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) (hU : OrthonormalColumns U)
    (hV : OrthonormalColumns V) (hs : ∀ k, s k ≠ 0) :
    (singularMatrix U s V).rank = r := by
  classical
  have hdiag : (Matrix.diagonal s).rank = r := by
    rw [Matrix.rank_diagonal]
    simp [hs]
  have hcompress : U.transpose * singularMatrix U s V * V = Matrix.diagonal s := by
    simp only [singularMatrix, Matrix.mul_assoc]
    rw [← Matrix.mul_assoc U.transpose U, hU, Matrix.one_mul, hV, Matrix.mul_one]
  have hlower := Matrix.rank_mul_le_left (U.transpose * singularMatrix U s V) V
  rw [hcompress, hdiag] at hlower
  have hupper : (singularMatrix U s V).rank ≤ r := by
    exact (Matrix.rank_mul_le_left (U * Matrix.diagonal s) V.transpose).trans
      ((Matrix.rank_mul_le_right U (Matrix.diagonal s)).trans_eq hdiag)
  exact le_antisymm hupper (hlower.trans (Matrix.rank_mul_le_right U.transpose _))

/-- The rank hypotheses are satisfiable, arXiv:2502.16982, §2.1. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    ∀ k : Fin 1, (fun _ : Fin 1 => (2 : ℝ)) k ≠ 0 := by
  norm_num [OrthonormalColumns]

/-- Exact polar factors have the same active rank as their momentum inputs,
arXiv:2502.16982, §2.1 and Appendix A. -/
theorem polarUpdate_rank (U : Matrix (Fin a) (Fin r) ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) (hU : OrthonormalColumns U)
    (hV : OrthonormalColumns V) : (polarUpdate U V).rank = r := by
  have h := singularMatrix_rank U (fun _ => 1) V hU hV (fun _ => one_ne_zero)
  simpa only [singularMatrix, Matrix.diagonal_one, Matrix.mul_one, polarUpdate] using h

/-- Exact polar factors exist, arXiv:2502.16982, §2.1. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [OrthonormalColumns]

/-- The exact polar-orthogonalization relation from the active thin SVD.
This is a genuine predicate of the momentum and its update, not an
assumption of an RMS or convergence theorem.
Source: arXiv:2502.16982, §2.1, after `eq:Ot`; Appendix A. -/
def IsExactPolar (M O : Matrix (Fin a) (Fin b) ℝ) : Prop :=
  ∃ (r : ℕ) (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ) (V : Matrix (Fin b) (Fin r) ℝ),
    OrthonormalColumns U ∧ OrthonormalColumns V ∧ (∀ k, 0 < s k) ∧
      M = singularMatrix U s V ∧ O = polarUpdate U V

/-- **Lemma 1 (`lemma:updaterms`), the matrix-level statement.** A full-rank
momentum matrix with an exact polar update has theoretical RMS
`√(1/max(a,b))` in positive dimensions.

The source attributes full rank to the matrix parameter. It is the
momentum rank that matters. Exact polar orthogonalization is required;
the printed finite polynomial iteration is an approximation.

Source: arXiv:2502.16982, §2.2, `lemma:updaterms`; Appendix A. -/
theorem muon_update_rms (M O : Matrix (Fin a) (Fin b) ℝ)
    (hfull : M.rank = min a b) (hpolar : IsExactPolar M O) (ha : 0 < a) (hb : 0 < b) :
    matrixRMS O = Real.sqrt (1 / max (a : ℝ) b) := by
  obtain ⟨r, U, s, V, hU, hV, hs, rfl, rfl⟩ := hpolar
  rw [singularMatrix_rank U s V hU hV (fun k => (hs k).ne')] at hfull
  rw [matrixRMS_polarUpdate U V hU hV, hfull, min_div_product ha hb]

/-- The matrix-level lemma is satisfiable: one-dimensional identity momentum
and its exact identity update, arXiv:2502.16982, Lemma 1. -/
example : (1 : Matrix (Fin 1) (Fin 1) ℝ).rank = min 1 1 ∧
    IsExactPolar (1 : Matrix (Fin 1) (Fin 1) ℝ) 1 ∧ 0 < (1 : ℕ) ∧ 0 < (1 : ℕ) := by
  refine ⟨by simp, ?_, by norm_num, by norm_num⟩
  refine ⟨1, 1, fun _ => 1, 1, ?_, ?_, ?_, ?_, ?_⟩
  · simp [OrthonormalColumns]
  · simp [OrthonormalColumns]
  · intro k; norm_num
  · simp [singularMatrix]
  · simp [polarUpdate]

/-- The corrected inverse-root construction is an exact polar update for
positive singular values, arXiv:2502.16982, §2.1, after `eq:Ot`. -/
theorem inverseRootGram_isExactPolar (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) (hU : OrthonormalColumns U)
    (hV : OrthonormalColumns V) (hs : ∀ k, 0 < s k) :
    IsExactPolar (singularMatrix U s V) (inverseRootGram U s * singularMatrix U s V) := by
  exact ⟨r, U, s, V, hU, hV, hs, rfl, inverseRootGram_mul U s V hU (fun k => (hs k).ne')⟩

/-- Positive singular values and orthonormal frames coexist,
arXiv:2502.16982, §2.1. -/
example : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    ∀ k : Fin 1, (0 : ℝ) < (fun _ : Fin 1 => (2 : ℝ)) k := by
  norm_num [OrthonormalColumns]

/-- The tall matrix `[1;0]` has nonzero energy but a singular left Gram
matrix. Thus the source's ordinary inverse square root cannot be used for
all rectangular matrices; the spectral pseudoinverse is necessary.
Source: arXiv:2502.16982, §2.1, after `eq:Ot`. -/
theorem tall_gram_singular :
    let M : Matrix (Fin 2) (Fin 1) ℝ := Matrix.of (fun i _ => if i = 0 then 1 else 0)
    (M * M.transpose).det = 0 ∧ squaredFrobenius M = 1 := by
  let M : Matrix (Fin 2) (Fin 1) ℝ := Matrix.of (fun i _ => if i = 0 then 1 else 0)
  change (M * M.transpose).det = 0 ∧ squaredFrobenius M = 1
  have hgram : M * M.transpose = Matrix.of (fun i j => if i = 0 ∧ j = 0 then 1 else 0) := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      norm_num [Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_one, M]
  constructor
  · rw [hgram, Matrix.det_fin_two]
    norm_num
  · norm_num [squaredFrobenius, Fin.sum_univ_two, Fin.sum_univ_one, M]

end Transformer.Muon
