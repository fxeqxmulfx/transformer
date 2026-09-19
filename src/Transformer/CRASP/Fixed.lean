/-
# Fixed-precision numbers

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix B.1, `def:fixed_precision`.

"A *fixed-precision number* with `p` total bits and `s` fractional bits is a
rational number of the form `m · 2^{-s}` where `m` is an integer and
`-2^{p-1} ≤ m < 2^{p-1}`."  So a fixed-precision number *is* its mantissa,
which is what `Fx` records; `Fx.val` reads it back as a real number.

`Fx.bit` is the paper's two's-complement bit extraction `[x]_b`, and `Fx.round`
is `round_𝔽`, the greatest element of `𝔽` below a given real.  The paper does
not say what `round` does outside the representable range; `Fx.round` saturates
there, and `val_round_le` is the defining inequality under the hypothesis that
no saturation occurs.
-/

import Transformer.CRASP.Defs

namespace Transformer
namespace CRASP

/-- A fixed-precision number with `p` total bits and `s` fractional bits,
recorded by its mantissa `m` (Definition `def:fixed_precision`). -/
structure Fx (p s : ℕ) where
  /-- The mantissa: the number represented is `m · 2^{-s}`. -/
  m : ℤ
  /-- Two's complement lower bound. -/
  lo : -2 ^ (p - 1) ≤ m
  /-- Two's complement upper bound. -/
  hi : m < 2 ^ (p - 1)

namespace Fx

variable {p s : ℕ}

/-- The real number a fixed-precision number denotes, `m · 2^{-s}`. -/
noncomputable def val (x : Fx p s) : ℝ := (x.m : ℝ) / 2 ^ s

@[ext] theorem ext {x y : Fx p s} (h : x.m = y.m) : x = y := by
  cases x; cases y; simp_all

/-- Zero is representable at every precision. -/
instance : Zero (Fx p s) where
  zero := ⟨0, neg_nonpos.2 (by positivity), by positivity⟩

@[simp] theorem m_zero : (0 : Fx p s).m = 0 := rfl

@[simp] theorem val_zero : (0 : Fx p s).val = 0 := by simp [val]

/-- The `b`-th bit of a fixed-precision number, read in two's complement:
`[x]_b = 1` when `⌊x / 2^{b-s-1}⌋` is odd (Definition `def:fixed_precision`). -/
noncomputable def bit (x : Fx p s) (b : ℕ) : Bool :=
  decide (Odd ⌊x.val * (2 : ℝ) ^ ((s : ℤ) + 1 - b)⌋)

/-- Mantissas are clamped into the representable range.  Inside the range this
is the identity; outside it saturates, which is the one point where `Fx.round`
goes beyond `def:fixed_precision`. -/
def clamp (p : ℕ) (m : ℤ) : ℤ := max (-2 ^ (p - 1)) (min (2 ^ (p - 1) - 1) m)

theorem le_clamp (p : ℕ) (m : ℤ) : -2 ^ (p - 1) ≤ clamp p m := le_max_left _ _

theorem clamp_lt (p : ℕ) (m : ℤ) : clamp p m < 2 ^ (p - 1) := by
  have hA : (0 : ℤ) < 2 ^ (p - 1) := by positivity
  exact max_lt (by omega) (lt_of_le_of_lt (min_le_left _ _) (by omega))

theorem clamp_eq_self {p : ℕ} {m : ℤ} (h₁ : -2 ^ (p - 1) ≤ m) (h₂ : m < 2 ^ (p - 1)) :
    clamp p m = m := by
  have h₂' : m ≤ 2 ^ (p - 1) - 1 := by omega
  rw [clamp, min_eq_right h₂', max_eq_right h₁]

/-- `round_𝔽(x)`, the greatest fixed-precision number at most `x`, saturating
outside the representable range (Definition `def:fixed_precision`). -/
noncomputable def round (p s : ℕ) (x : ℝ) : Fx p s :=
  ⟨clamp p ⌊x * 2 ^ s⌋, le_clamp _ _, clamp_lt _ _⟩

@[simp] theorem m_round (p s : ℕ) (x : ℝ) : (round p s x).m = clamp p ⌊x * 2 ^ s⌋ := rfl

theorem val_round (p s : ℕ) (x : ℝ) :
    (round p s x).val = ((clamp p ⌊x * 2 ^ s⌋ : ℤ) : ℝ) / 2 ^ s := rfl

/-- **Rounding never overshoots.**  As long as the value is in range, the
rounded number is at most the number rounded. -/
theorem val_round_le (p s : ℕ) (x : ℝ) (h₁ : -2 ^ (p - 1) ≤ ⌊x * 2 ^ s⌋)
    (h₂ : ⌊x * 2 ^ s⌋ < 2 ^ (p - 1)) : (round p s x).val ≤ x := by
  have hpos : (0 : ℝ) < 2 ^ s := by positivity
  rw [val_round, clamp_eq_self h₁ h₂, div_le_iff₀ hpos]
  exact Int.floor_le _

/-- **And it is the greatest such.**  Any fixed-precision number below `x` is
below its rounding. -/
theorem le_val_round (p s : ℕ) (x : ℝ) (y : Fx p s) (hy : y.val ≤ x)
    (h₁ : -2 ^ (p - 1) ≤ ⌊x * 2 ^ s⌋) (h₂ : ⌊x * 2 ^ s⌋ < 2 ^ (p - 1)) :
    y.val ≤ (round p s x).val := by
  have hpos : (0 : ℝ) < 2 ^ s := by positivity
  have hy' : (y.m : ℝ) ≤ x * 2 ^ s := by rwa [val, div_le_iff₀ hpos] at hy
  have key : (y.m : ℝ) ≤ ((⌊x * 2 ^ s⌋ : ℤ) : ℝ) := Int.cast_le.2 (Int.le_floor.2 hy')
  rw [val_round, clamp_eq_self h₁ h₂, val]
  gcongr

/-- Rounding a representable number returns it. -/
theorem round_val (p s : ℕ) (y : Fx p s) : round p s y.val = y := by
  have hpos : (0 : ℝ) < 2 ^ s := by positivity
  have : ⌊y.val * 2 ^ s⌋ = y.m := by
    rw [val, div_mul_cancel₀ _ (ne_of_gt hpos), Int.floor_intCast]
  ext
  rw [m_round, this, clamp_eq_self y.lo y.hi]

/-- **One grid step separates distinct fixed-precision numbers.**  Two numbers
whose values differ by less than `2^{-s}` are equal: both are integer
multiples of `2^{-s}`, so their mantissas differ by less than one. -/
theorem eq_of_abs_val_sub_lt {x y : Fx p s} (h : |x.val - y.val| < 2⁻¹ ^ s) :
    x = y := by
  have hpos : (0 : ℝ) < 2 ^ s := by positivity
  have hval : x.val - y.val = ((x.m - y.m : ℤ) : ℝ) / 2 ^ s := by
    rw [val, val, div_sub_div_same]
    push_cast
    ring
  rw [hval, inv_pow, abs_div, abs_of_pos hpos, div_lt_iff₀ hpos,
    inv_mul_cancel₀ (ne_of_gt hpos), ← Int.cast_abs] at h
  have hm : |x.m - y.m| < 1 := by exact_mod_cast h
  have := abs_lt.mp hm
  exact ext (by omega)

/-- Rounded addition, which is exact whenever the sum is representable: the
paper writes `c + h` for the residual connection without saying what happens on
overflow (`def:transformer`). -/
noncomputable def add (x y : Fx p s) : Fx p s := round p s (x.val + y.val)

/-- The hypotheses of `val_round_le` are satisfiable, and rounding is not the
identity: at `p = 8`, `s = 0`, the number `1/2` rounds down to `0`. -/
example : (round 8 0 (1 / 2 : ℝ)).val = 0 ∧
    (-2 ^ (8 - 1) : ℤ) ≤ ⌊(1 / 2 : ℝ) * 2 ^ 0⌋ ∧ ⌊(1 / 2 : ℝ) * 2 ^ 0⌋ < 2 ^ (8 - 1) := by
  have h : ⌊(1 / 2 : ℝ) * 2 ^ 0⌋ = 0 := by norm_num
  refine ⟨?_, by rw [h]; norm_num, by rw [h]; norm_num⟩
  rw [val_round, h, clamp_eq_self (by norm_num) (by norm_num)]
  norm_num

end Fx

end CRASP
end Transformer
