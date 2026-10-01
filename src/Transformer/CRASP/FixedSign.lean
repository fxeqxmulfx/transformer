/-
# Boolean storage and the sign of downward rounding

arXiv:2506.16055v3, Appendix B.2, `thm:TLCl_to_rtfr`:
uniform attention followed by downward rounding implements an integer
comparison, while Boolean features are stored as zero and one.
-/

import Transformer.CRASP.FixedFinite

namespace Transformer.CRASP.Fx

/-- Boolean storage at a precision with at least two total bits (B.2). -/
def ofBool (n : ℕ) (b : Bool) : Fx (n + 2) 0 where
  m := if b then 1 else 0
  lo := by
    rw [show n + 2 - 1 = n + 1 by omega]
    have hp : (0 : ℤ) < 2 ^ (n + 1) := by positivity
    cases b <;> simp only [Bool.false_eq_true, ite_false, ite_true] <;> omega
  hi := by
    rw [show n + 2 - 1 = n + 1 by omega]
    have hp : (2 : ℤ) ≤ 2 ^ (n + 1) := by
      rw [pow_succ]
      have hpos : (0 : ℤ) < 2 ^ n := by positivity
      linarith
    cases b <;> simp only [Bool.false_eq_true, ite_false, ite_true] <;> omega

/-- Stored Booleans have mantissa zero or one (Appendix B.2). -/
@[simp] theorem m_ofBool (n : ℕ) (b : Bool) : (ofBool n b).m = if b then 1 else 0 := rfl

/-- Stored Booleans have real value zero or one (Appendix B.2). -/
@[simp] theorem val_ofBool (n : ℕ) (b : Bool) : (ofBool n b).val = if b then 1 else 0 := by
  cases b <;> simp [val, ofBool]

/-- Reading the stored Boolean recovers its truth value (Appendix B.2). -/
@[simp] theorem read_ofBool (n : ℕ) (b : Bool) : decide ((ofBool n b).m = 1) = b := by
  cases b <;> simp

/-- Saturation preserves the sign of an integer mantissa (Appendix B.2). -/
theorem clamp_neg_iff (p : ℕ) (z : ℤ) : clamp p z < 0 ↔ z < 0 := by
  have hp : (0 : ℤ) < 2 ^ (p - 1) := by positivity
  unfold clamp
  omega

/-- Downward rounding preserves the negative/nonnegative dichotomy (B.2). -/
theorem val_round_neg_iff (p : ℕ) (x : ℝ) : (round p 0 x).val < 0 ↔ x < 0 := by
  simp only [val, m_round, pow_zero, div_one, mul_one, Int.cast_lt_zero,
    clamp_neg_iff, Int.floor_lt_zero]

/-- Zero rounds to itself (Appendix B.1). -/
@[simp] theorem round_zero (p s : ℕ) : round p s 0 = 0 := by
  simpa using round_val p s (0 : Fx p s)

/-- A zero attention contribution leaves a residual feature unchanged (B.2). -/
@[simp] theorem zero_add {p s : ℕ} (x : Fx p s) : add 0 x = x := by
  simp [add, round_val]

/-- A zero scratch feature leaves the attention contribution unchanged (B.2). -/
@[simp] theorem add_zero {p s : ℕ} (x : Fx p s) : add x 0 = x := by
  simp [add, round_val]

/-- A bounded integer has an exact representation at the chosen precision (B.2). -/
theorem m_round_int (n : ℕ) (z : ℤ) (hz : |z| ≤ n) :
    (round (n + 2) 0 (z : ℝ)).m = z := by
  have hpow : (n : ℤ) < 2 ^ (n + 1) := by
    exact_mod_cast lt_trans (Nat.lt_succ_self n) (Nat.lt_two_pow_self (n := n + 1))
  have habs := abs_le.mp hz
  simp only [m_round, pow_zero, mul_one, Int.floor_intCast]
  apply clamp_eq_self <;> rw [show n + 2 - 1 = n + 1 by omega] <;> omega

/-- The bounded-integer hypothesis is satisfiable (Appendix B.2). -/
example : |(-2 : ℤ)| ≤ (3 : ℕ) := by decide

end Transformer.CRASP.Fx
