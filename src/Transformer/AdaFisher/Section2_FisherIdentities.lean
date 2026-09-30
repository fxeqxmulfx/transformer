/-
# AdaFisher: symmetry, positivity, and sum-of-score Fisher identities

arXiv:2405.16397v3, §2, `eq:fishermatrix`.
The Fisher of a sum equals the sum of Fishers only when cross-score
moments vanish, as for independent centered scores. This condition is
explicit rather than silently dropping off-diagonal sample terms.
-/

import Transformer.AdaFisher.Section2_Fisher
import Mathlib.LinearAlgebra.Matrix.PosDef

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.AdaFisher

variable {a b n m : ℕ}

/-- Every empirical Fisher block is symmetric, §2, `eq:fishermatrix`. -/
theorem empiricalFisher_symmetric (w : Fin n → ℝ)
    (h : Fin n → Fin a → ℝ) (s : Fin n → Fin b → ℝ) :
    (empiricalFisher w h s).transpose = empiricalFisher w h s := by
  ext p q
  simp only [Matrix.transpose_apply, empiricalFisher]
  apply Finset.sum_congr rfl
  intro k hk
  ring

/-- The empirical Fisher is genuinely positive semidefinite, §2,
`eq:fishermatrix`, for nonnegative sampling weights. -/
theorem empiricalFisher_posSemidef (w : Fin n → ℝ)
    (h : Fin n → Fin a → ℝ) (s : Fin n → Fin b → ℝ) (hw : ∀ k, 0 ≤ w k) :
    (empiricalFisher w h s).PosSemidef := by
  apply Matrix.posSemidef_iff_dotProduct_mulVec.mpr
  constructor
  · change (empiricalFisher w h s).conjTranspose = empiricalFisher w h s
    ext p q
    simpa using congrArg (fun M => M p q) (empiricalFisher_symmetric w h s)
  · intro v
    have hn := empiricalFisher_quadratic_nonneg w h s hw v
    simpa [dotProduct, Matrix.mulVec, Finset.mul_sum, mul_assoc] using hn

example : ∀ k : Fin 1, 0 ≤ (fun _ => (1 : ℝ)) k := by norm_num

/-- Two sample scores' mixed moment, §2, `eq:fishermatrix`. -/
def scoreCrossMoment (w : Fin n → ℝ) (g : Fin m → Fin n → Fin a → ℝ)
    (k l : Fin m) (i j : Fin a) : ℝ := ∑ ω, w ω * g k ω i * g l ω j

/-- Exact Fisher of a sum, including cross-score terms, §2. -/
theorem scoreSum_fisher_expansion (w : Fin n → ℝ) (g : Fin m → Fin n → Fin a → ℝ)
    (i j : Fin a) :
    secondMoment w (fun ω q => ∑ k, g k ω q) i j =
      ∑ k, ∑ l, scoreCrossMoment w g k l i j := by
  simp only [secondMoment, scoreCrossMoment, Finset.mul_sum, Finset.sum_mul]
  calc
    _ = ∑ ω, ∑ k, ∑ l, w ω * g k ω i * g l ω j := by
      apply Finset.sum_congr rfl
      intro ω hω
      rw [Finset.sum_comm]
    _ = ∑ k, ∑ ω, ∑ l, w ω * g k ω i * g l ω j := Finset.sum_comm
    _ = _ := by
      apply Finset.sum_congr rfl
      intro k hk
      rw [Finset.sum_comm]

/-- Correct sum-of-Fishers identity from `eq:fishermatrix`, §2.
The source suppresses cross-observation moments; their vanishing is an
essential hypothesis, normally supplied by independent centered scores. -/
theorem scoreSum_fisher_of_cross_zero (w : Fin n → ℝ)
    (g : Fin m → Fin n → Fin a → ℝ)
    (hcross : ∀ k l, k ≠ l → ∀ i j, scoreCrossMoment w g k l i j = 0)
    (i j : Fin a) :
    secondMoment w (fun ω q => ∑ k, g k ω q) i j =
      ∑ k, secondMoment w (g k) i j := by
  rw [scoreSum_fisher_expansion]
  apply Finset.sum_congr rfl
  intro k hk
  rw [Finset.sum_eq_single k]
  · rfl
  · intro l hl hlk
    exact hcross k l (Ne.symm hlk) i j
  · intro hknot
    exact False.elim (hknot (Finset.mem_univ k))

example :
    ∀ k l : Fin 2, k ≠ l → ∀ i j : Fin 1,
      scoreCrossMoment (fun _ : Fin 1 => (1 : ℝ))
        (fun (_ : Fin 2) (_ : Fin 1) (_ : Fin 1) => (0 : ℝ)) k l i j = 0 := by
  simp [scoreCrossMoment]

end Transformer.AdaFisher
