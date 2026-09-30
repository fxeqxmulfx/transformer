/-
# DASH — instantaneous Gram rank versus accumulated preconditioner rank

arXiv:2602.02016v2, §5, “Block Size”. A single `B×E` gradient
has Gram rank at most `E`. Its temporal EMA need not keep that rank.
-/

import Transformer.DASH.Section2_Models
import Mathlib.LinearAlgebra.Matrix.Rank

open scoped Matrix

noncomputable section

namespace Transformer.DASH

/-- An instantaneous left Gram matrix has rank at most the gradient width.
This is the valid part of the source's `B>E` rank discussion.
Source: arXiv:2602.02016v2, §5, “Block Size”. -/
theorem leftGram_rank_le_width {m n : ℕ} (G : Matrix (Fin m) (Fin n) ℝ) :
    (G * G.transpose).rank ≤ n := by
  exact (Matrix.rank_mul_le_left G G.transpose).trans
    (by simpa using Matrix.rank_le_card_width G)

/-- An instantaneous right Gram has rank at most the gradient height,
arXiv:2602.02016v2, §5, “Block Size”. -/
theorem rightGram_rank_le_height {m n : ℕ} (G : Matrix (Fin m) (Fin n) ℝ) :
    (G.transpose * G).rank ≤ m := by
  exact (Matrix.rank_mul_le_left G.transpose G).trans
    (by simpa using Matrix.rank_le_card_width G.transpose)

/-- Two orthogonal `2×1` gradients produce an EMA of rank two although
`E=1`. This refutes the source's unconditional rank bound for preconditioner
blocks: the bound applies to each instantaneous Gram, not their EMA.
Source: arXiv:2602.02016v2, §5, “blocks ... guaranteed to have rank at most E”. -/
theorem ema_rank_counterexample :
    let G₁ : Matrix (Fin 2) (Fin 1) ℝ := fun i _ => if i = 0 then 1 else 0
    let G₂ : Matrix (Fin 2) (Fin 1) ℝ := fun i _ => if i = 0 then 0 else 1
    (leftEma (1 / 2) (leftEma (1 / 2) 0 G₁) G₂).rank = 2 ∧ (1 : ℕ) < 2 := by
  let G₁ : Matrix (Fin 2) (Fin 1) ℝ := fun i _ => if i = 0 then 1 else 0
  let G₂ : Matrix (Fin 2) (Fin 1) ℝ := fun i _ => if i = 0 then 0 else 1
  change (leftEma (1 / 2) (leftEma (1 / 2) 0 G₁) G₂).rank = 2 ∧ (1 : ℕ) < 2
  have hgram₁ : ∀ i j, (G₁ * G₁.transpose) i j = G₁ i 0 * G₁ j 0 := by
    intro i j
    change (∑ k : Fin 1, G₁ i k * G₁ j k) = G₁ i 0 * G₁ j 0
    rw [Fin.sum_univ_one]
  have hgram₂ : ∀ i j, (G₂ * G₂.transpose) i j = G₂ i 0 * G₂ j 0 := by
    intro i j
    change (∑ k : Fin 1, G₂ i k * G₂ j k) = G₂ i 0 * G₂ j 0
    rw [Fin.sum_univ_one]
  have hmatrix : leftEma (1 / 2) (leftEma (1 / 2) 0 G₁) G₂ =
      Matrix.diagonal (![1 / 4, 1 / 2] : Fin 2 → ℝ) := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp only [leftEma, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul] <;>
      norm_num [hgram₁, hgram₂, Matrix.diagonal, G₁, G₂]
  have hs : ∀ i : Fin 2, (![1 / 4, 1 / 2] : Fin 2 → ℝ) i ≠ 0 := by
    intro i
    fin_cases i <;> norm_num
  rw [hmatrix, Matrix.rank_diagonal]
  constructor
  · exact (Fintype.card_congr (Equiv.subtypeUnivEquiv hs)).trans (by simp)
  · norm_num

end Transformer.DASH
