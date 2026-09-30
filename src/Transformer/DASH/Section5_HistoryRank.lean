/-
# DASH — corrected rank bounds for temporal preconditioners

arXiv:2602.02016v2, §5, “Block Size”, and §2, Algorithm 1.
The manuscript's unconditional bound by gradient width applies to
one instantaneous Gram matrix. An EMA over `t` gradients instead
has rank at most `min(block height, t * gradient width)`.
-/

import Transformer.DASH.Section5_Rank
import Transformer.DASH.Section2_History
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas

open scoped Matrix

noncomputable section

namespace Transformer.DASH

variable {m n : ℕ}

/-- Subadditivity of matrix rank supplies the corrected temporal bound.
Source: arXiv:2602.02016v2, §2, the sum in the EMA recurrence, and §5,
“Block Size”, whose instantaneous bound need not survive that sum. -/
theorem matrix_rank_add_le (A B : Matrix (Fin m) (Fin n) ℝ) :
    (A + B).rank ≤ A.rank + B.rank := by
  rw [Matrix.rank, Matrix.mulVecLin_add]
  have hdim := Submodule.finrank_sup_add_finrank_inf_eq
    (LinearMap.range A.mulVecLin) (LinearMap.range B.mulVecLin)
  exact (Submodule.finrank_mono (LinearMap.range_add_le A.mulVecLin B.mulVecLin)).trans
    (by simpa only [Matrix.rank] using Nat.le.intro hdim)

/-- Scalar weighting cannot increase a Gram or history rank.
Source: arXiv:2602.02016v2, §2, the coefficients `β` and `1-β` in Algorithm 1. -/
theorem matrix_rank_smul_le (c : ℝ) (A : Matrix (Fin m) (Fin n) ℝ) :
    (c • A).rank ≤ A.rank := by
  have heq : c • A = (c • (1 : Matrix (Fin m) (Fin m) ℝ)) * A := by
    simp
  rw [heq]
  exact Matrix.rank_mul_le_right _ _

/-- The left EMA over `t` gradients has rank at most `t*n`, with no
restriction on the scalar EMA coefficient. This corrects the source's
claim of a universal bound by `n`, refuted in `ema_rank_counterexample`.
Source: arXiv:2602.02016v2, §2, Algorithm 1, and §5, “Block Size”. -/
theorem leftHistory_rank_le_samples (β : ℝ) (G : ℕ → Matrix (Fin m) (Fin n) ℝ)
    (t : ℕ) : (leftHistory β G t).rank ≤ t * n := by
  induction t with
  | zero => simp [leftHistory]
  | succ t ih =>
    rw [leftHistory, leftEma]
    calc
      _ ≤ (β • leftHistory β G t).rank + ((1 - β) • (G t * (G t).transpose)).rank :=
        matrix_rank_add_le _ _
      _ ≤ (leftHistory β G t).rank + (G t * (G t).transpose).rank :=
        Nat.add_le_add (matrix_rank_smul_le _ _) (matrix_rank_smul_le _ _)
      _ ≤ t * n + n := Nat.add_le_add ih (leftGram_rank_le_width (G t))
      _ = (t + 1) * n := by ring

/-- The right EMA over `t` gradients has rank at most `t*m`.
Source: arXiv:2602.02016v2, §2, Algorithm 1, and the corrected §5 rank discussion. -/
theorem rightHistory_rank_le_samples (β : ℝ) (G : ℕ → Matrix (Fin m) (Fin n) ℝ)
    (t : ℕ) : (rightHistory β G t).rank ≤ t * m := by
  induction t with
  | zero => simp [rightHistory]
  | succ t ih =>
    rw [rightHistory, rightEma]
    calc
      _ ≤ (β • rightHistory β G t).rank + ((1 - β) • ((G t).transpose * G t)).rank :=
        matrix_rank_add_le _ _
      _ ≤ (rightHistory β G t).rank + ((G t).transpose * G t).rank :=
        Nat.add_le_add (matrix_rank_smul_le _ _) (matrix_rank_smul_le _ _)
      _ ≤ t * m + m := Nat.add_le_add ih (rightGram_rank_le_height (G t))
      _ = (t + 1) * m := by ring

/-- Both accumulated preconditioners obey the time-dependent rank bound
and their ambient dimensions. For a `B×E` gradient block, the left bound
is `min(B,t*E)`, not the source's unconditional `E`.
Source: arXiv:2602.02016v2, §5, “Block Size”, corrected using Algorithm 1. -/
theorem history_rank_le (β : ℝ) (G : ℕ → Matrix (Fin m) (Fin n) ℝ) (t : ℕ) :
    (leftHistory β G t).rank ≤ min m (t * n) ∧
      (rightHistory β G t).rank ≤ min n (t * m) := by
  constructor
  · exact le_min (by simpa using Matrix.rank_le_card_height (leftHistory β G t))
      (leftHistory_rank_le_samples β G t)
  · exact le_min (by simpa using Matrix.rank_le_card_height (rightHistory β G t))
      (rightHistory_rank_le_samples β G t)

end Transformer.DASH
