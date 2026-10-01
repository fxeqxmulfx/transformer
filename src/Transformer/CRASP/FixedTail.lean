/-
# Fixed-point rounding near zero and at the saturation endpoints

arXiv:2506.16055v3, Appendix B.1, `def:fixed_precision`, and Appendix F,
`sec:alibi_pes`. Downward rounding of a small negative numerator gives
minus one mantissa unit, even when a small positive weight rounds to zero.
-/

import Transformer.CRASP.FixedFinite

namespace Transformer.CRASP.Fx

variable {p s : ℕ}

/-- Minus one mantissa unit, present at every precision (B.1/F). -/
def negUnit (p s : ℕ) : Fx p s where
  m := -1
  lo := by
    have h : (0 : ℤ) < 2 ^ (p - 1) := by positivity
    omega
  hi := by
    have h : (0 : ℤ) < 2 ^ (p - 1) := by positivity
    omega

/-- The upper saturation endpoint (Appendix B.1). -/
def top (p s : ℕ) : Fx p s where
  m := 2 ^ (p - 1) - 1
  lo := by
    have h : (0 : ℤ) < 2 ^ (p - 1) := by positivity
    omega
  hi := by omega

/-- The lower saturation endpoint (Appendix B.1). -/
def bottom (p s : ℕ) : Fx p s where
  m := -2 ^ (p - 1)
  lo := le_rfl
  hi := by
    have h : (0 : ℤ) < 2 ^ (p - 1) := by positivity
    omega

/-- A real smaller than one mantissa unit rounds according to its sign (F). -/
theorem round_small (x : ℝ) (hx : |x * 2 ^ s| < 1) :
    round p s x = if x < 0 then negUnit p s else 0 := by
  have hs : (0 : ℝ) < 2 ^ s := by positivity
  have hp : (0 : ℤ) < 2 ^ (p - 1) := by positivity
  have hb := abs_lt.mp hx
  by_cases hn : x < 0
  · rw [ite_eq_left hn]
    apply Fx.ext
    have hf : ⌊x * 2 ^ s⌋ = (-1 : ℤ) := by
      apply Int.floor_eq_iff.mpr
      simp only [Int.cast_neg, Int.cast_one, neg_add_cancel]
      exact ⟨hb.1.le, mul_neg_of_neg_of_pos hn hs⟩
    simp only [m_round, hf, negUnit, clamp]
    omega
  · rw [ite_eq_right hn]
    apply Fx.ext
    have hf : ⌊x * 2 ^ s⌋ = (0 : ℤ) := by
      apply Int.floor_eq_iff.mpr
      simp only [Int.cast_zero, zero_add]
      exact ⟨mul_nonneg (le_of_not_gt hn) hs.le, hb.2⟩
    simp only [m_round, hf, m_zero, clamp]
    omega

/-- A sufficiently large nonnegative value rounds to the upper endpoint (B.1). -/
theorem round_top (x : ℝ) (hx : ((2 ^ (p - 1) - 1 : ℤ) : ℝ) ≤ x * 2 ^ s) :
    round p s x = top p s := by
  apply Fx.ext
  have hf : (2 ^ (p - 1) - 1 : ℤ) ≤ ⌊x * 2 ^ s⌋ := Int.le_floor.mpr hx
  have hp : (0 : ℤ) < 2 ^ (p - 1) := by positivity
  simp only [m_round, top, clamp]
  omega

/-- A sufficiently negative value rounds to the lower endpoint (Appendix B.1). -/
theorem round_bottom (x : ℝ) (hx : x * 2 ^ s ≤ ((-2 ^ (p - 1) : ℤ) : ℝ)) :
    round p s x = bottom p s := by
  apply Fx.ext
  have hf : ⌊x * 2 ^ s⌋ ≤ (-2 ^ (p - 1) : ℤ) := by
    exact_mod_cast (Int.floor_le (x * 2 ^ s)).trans hx
  have hp : (0 : ℤ) < 2 ^ (p - 1) := by positivity
  simp only [m_round, bottom, clamp]
  omega

/-- The small-value and saturation hypotheses all have integer witnesses (B.1/F). -/
example : |(0 : ℝ) * 2 ^ (0 : ℕ)| < 1 ∧
    (((2 ^ (2 - 1) - 1 : ℤ) : ℝ) ≤ 1 * 2 ^ (0 : ℕ)) ∧
    ((-2 : ℝ) * 2 ^ (0 : ℕ) ≤ ((-2 ^ (2 - 1) : ℤ) : ℝ)) := by
  norm_num

end Transformer.CRASP.Fx
