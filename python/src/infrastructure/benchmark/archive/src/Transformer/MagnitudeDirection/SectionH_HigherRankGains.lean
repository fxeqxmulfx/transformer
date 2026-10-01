/-
# Higher-rank multiplicative gains

arXiv:2606.25971v2, Appendix H, equation `eq:higherrank-gain`.
The offset is the all-ones matrix, not the identity matrix. Rank `k`
describes the update `ABᵀ`; the offset gain can have larger rank.
-/

import Transformer.MagnitudeDirection.Section3_Factorization
import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.MagnitudeDirection

variable {m n k : ℕ}

/-- The rank-at-most-one row/column gain matrix, arXiv:2606.25971v2,
§4.1.3 and Appendix H. “Rank-1” requires nonzero factors; in general
the mathematically correct bound is rank at most one. -/
def outerGain (row : Fin m → ℝ) (col : Fin n → ℝ) : Matrix (Fin m) (Fin n) ℝ :=
  Matrix.vecMulVec row col

/-- The row/column factorization is elementwise multiplication by the outer
product, arXiv:2606.25971v2, Appendix H, “From rank-1 to rank-k”. -/
theorem fuse_outerGain (row : Fin m → ℝ) (col : Fin n → ℝ)
    (D : Matrix (Fin m) (Fin n) ℝ) (i : Fin m) (j : Fin n) :
    fuse row col D i j = outerGain row col i j * D i j := by
  dsimp [fuse, outerGain, Matrix.vecMulVec]
  ring

/-- Actual linear-algebraic rank bound for the combined gain,
arXiv:2606.25971v2, §4.1.3 and Appendix H. -/
theorem outerGain_rank_le_one (row : Fin m → ℝ) (col : Fin n → ℝ) :
    (outerGain row col).rank ≤ 1 := Matrix.rank_vecMulVec_le row col

/-- The all-ones offset plus a low-rank product, arXiv:2606.25971v2,
Appendix H, `eq:higherrank-gain`. -/
def higherGain (A : Matrix (Fin m) (Fin k) ℝ) (B : Matrix (Fin n) (Fin k) ℝ) :
    Matrix (Fin m) (Fin n) ℝ := fun i j => 1 + ∑ l, A i l * B j l

/-- The entry definition is the printed `ones + ABᵀ` formula,
arXiv:2606.25971v2, Appendix H, `eq:higherrank-gain`. -/
theorem higherGain_eq_matrix (A : Matrix (Fin m) (Fin k) ℝ)
    (B : Matrix (Fin n) (Fin k) ℝ) :
    higherGain A B = Matrix.of (fun _ _ => (1 : ℝ)) + A * B.transpose := by
  ext i j
  rfl

/-- Rank `k` is a bound on the *increment* `ABᵀ`,
arXiv:2606.25971v2, Appendix H, `eq:higherrank-gain`. -/
theorem higherGain_increment_rank_le (A : Matrix (Fin m) (Fin k) ℝ)
    (B : Matrix (Fin n) (Fin k) ℝ) : (A * B.transpose).rank ≤ k := by
  exact (Matrix.rank_mul_le_left A B.transpose).trans (by simpa using Matrix.rank_le_card_width A)

/-- Multiplicative higher-rank fused weights, arXiv:2606.25971v2,
Appendix H. This differs from LoRA's additive update. -/
def higherFuse (A : Matrix (Fin m) (Fin k) ℝ) (B : Matrix (Fin n) (Fin k) ℝ)
    (D : Matrix (Fin m) (Fin n) ℝ) : Matrix (Fin m) (Fin n) ℝ :=
  fun i j => higherGain A B i j * D i j

/-- Initializing `B=0` makes the effective gain exactly one and leaves the
initial model unchanged, arXiv:2606.25971v2, Appendix H. -/
theorem higherFuse_initialization (A : Matrix (Fin m) (Fin k) ℝ)
    (D : Matrix (Fin m) (Fin n) ℝ) : higherFuse A 0 D = D := by
  ext i j
  simp [higherFuse, higherGain]

/-- Two factors suffice to represent any row/column gain despite the offset:
`ABᵀ = row colᵀ - ones`. Source: arXiv:2606.25971v2, Appendix H,
“A rank-k gain with k ≥ 2 can represent the rank-1 row-and-column gain exactly”. -/
theorem two_factor_representation (row : Fin m → ℝ) (col : Fin n → ℝ) :
    ∃ (A : Matrix (Fin m) (Fin 2) ℝ) (B : Matrix (Fin n) (Fin 2) ℝ),
      higherGain A B = outerGain row col := by
  refine ⟨fun i => ![row i, 1], fun j => ![col j, -1], ?_⟩
  ext i j
  simp [higherGain, outerGain, Matrix.vecMulVec, Fin.sum_univ_two]

/-- Every factor width at least two represents every row/column gain,
arXiv:2606.25971v2, Appendix H. Extra factor columns are zero. -/
theorem higher_factor_representation (row : Fin m → ℝ) (col : Fin n → ℝ)
    (hk : 2 ≤ k) : ∃ (A : Matrix (Fin m) (Fin k) ℝ) (B : Matrix (Fin n) (Fin k) ℝ),
      higherGain A B = outerGain row col := by
  let z : Fin k := ⟨0, by omega⟩
  let o : Fin k := ⟨1, by omega⟩
  have hzo : z ≠ o := by simp [z, o, Fin.ext_iff]
  refine ⟨fun i l => if l = z then row i else if l = o then 1 else 0,
    fun j l => if l = z then col j else if l = o then -1 else 0, ?_⟩
  ext i j
  have he (l : Fin k) :
      (if l = z then row i else if l = o then 1 else 0) *
        (if l = z then col j else if l = o then -1 else 0) =
      (if l = z then row i * col j else 0) + (if l = o then -1 else 0) := by
    by_cases hz : l = z
    · subst l
      simp [hzo]
    · by_cases ho : l = o <;> simp [hz, ho, Ne.symm hzo]
  simp only [higherGain, he, Finset.sum_add_distrib]
  simp [outerGain, Matrix.vecMulVec]

/-- The factor-width premise holds for the tested rank four,
arXiv:2606.25971v2, Appendix H. -/
example : 2 ≤ (4 : ℕ) := by norm_num

/-- Higher-rank gains are strictly more expressive already in two dimensions:
the gain `[[2,1],[1,2]]` has a two-factor representation but is not any outer
product. Source: arXiv:2606.25971v2, Appendix H, expressivity claim. -/
theorem higherGain_strictly_more_expressive :
    let A : Matrix (Fin 2) (Fin 2) ℝ := 1
    ¬ ∃ (row col : Fin 2 → ℝ), higherGain A A = outerGain row col := by
  dsimp
  rintro ⟨row, col, h⟩
  have h00 := congrArg (fun M : Matrix (Fin 2) (Fin 2) ℝ => M 0 0) h
  have h01 := congrArg (fun M : Matrix (Fin 2) (Fin 2) ℝ => M 0 1) h
  have h10 := congrArg (fun M : Matrix (Fin 2) (Fin 2) ℝ => M 1 0) h
  have h11 := congrArg (fun M : Matrix (Fin 2) (Fin 2) ℝ => M 1 1) h
  norm_num [higherGain, outerGain, Matrix.vecMulVec, Fin.sum_univ_two] at h00 h01 h10 h11
  nlinarith [mul_mul_mul_comm (row 0) (col 0) (row 1) (col 1)]

/-- The offset gain can exceed the factor rank: two columns give a nonsingular
3-by-3 gain. Thus Appendix H's “rank-k gain matrix” must refer to its increment,
not an unconditional rank bound on `Gamma`.
Source: arXiv:2606.25971v2, Appendix H, `eq:higherrank-gain`. -/
theorem offset_gain_rank_counterexample :
    let A : Matrix (Fin 3) (Fin 2) ℝ := !![1, 0; 0, 1; 0, 0]
    (higherGain A A).det = 1 := by
  norm_num [higherGain, Matrix.det_fin_three, Fin.sum_univ_two]

end Transformer.MagnitudeDirection
