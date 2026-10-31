module

public import Transformer.CRASP.Fixed

/-!
# Error bounds for the specified fixed-point rounding

Source: arXiv:2506.16055v3, Appendix B.1, `def:fixed_precision`, and
`review/flat_spectrum.md`, *Quantitative RoPE approximation*, A6. `Fx.round`
floors and saturates; it is not IEEE round-to-nearest. The quantum error
bound is valid only when its explicit no-saturation condition holds.
-/

@[expose] public section
noncomputable section

namespace Transformer.CRASP.Fx

/-- The unrounded mantissa is inside the representable two's-complement
range, so clamping does not change the floor. -/
def NoSaturation (p s : ℕ) (x : ℝ) : Prop :=
  -2 ^ (p - 1) ≤ ⌊x * 2 ^ s⌋ ∧ ⌊x * 2 ^ s⌋ < 2 ^ (p - 1)

/-- A6: one floor-rounding error is strictly smaller than the fixed-point
quantum, provided saturation has been ruled out for the number rounded. -/
theorem round_error_lt (p s : ℕ) (x : ℝ) (h : NoSaturation p s x) :
    |(round p s x).val - x| < 1 / (2 : ℝ) ^ s := by
  have hs : 0 < (2 : ℝ) ^ s := by positivity
  have hle := val_round_le p s x h.1 h.2
  rw [abs_sub_comm, abs_of_nonneg (sub_nonneg.mpr hle)]
  rw [val_round, clamp_eq_self h.1 h.2]
  apply (mul_lt_mul_iff_left₀ hs).mp
  rw [sub_mul, div_mul_cancel₀ _ hs.ne', div_mul_cancel₀ _ hs.ne']
  nlinarith [Int.lt_floor_add_one (x * 2 ^ s)]

/-- A6: a positive lower bound exceeding one quantum prevents a rounded
coefficient from underflowing to zero. Saturation remains a separate check. -/
theorem round_pos (p s : ℕ) (x lower : ℝ) (h : NoSaturation p s x)
    (hx : lower ≤ x) (hlower : 1 / (2 : ℝ) ^ s < lower) :
    0 < (round p s x).val := by
  have he := (abs_lt.mp (round_error_lt p s x h)).1
  linarith

/-- A6: flooring also gives a useful one-sided lower estimate for sums
of positive softmax coefficients. -/
theorem round_lower (p s : ℕ) (x : ℝ) (h : NoSaturation p s x) :
    x - 1 / (2 : ℝ) ^ s ≤ (round p s x).val := by
  have he := (abs_lt.mp (round_error_lt p s x h)).1
  linarith

/-- A6: no fixed finite absolute error bound holds globally for a
saturating format. Increasing real inputs eventually exceeds every bound. -/
theorem no_uniform_round_error (p s : ℕ) (E : ℝ) :
    ∃ x : ℝ, E < |(round p s x).val - x| := by
  let upper := (2 : ℝ) ^ (p - 1) / (2 : ℝ) ^ s
  let x := upper + |E| + 1
  have hs : 0 < (2 : ℝ) ^ s := by positivity
  have hm : ((round p s x).m : ℝ) < (2 : ℝ) ^ (p - 1) := by
    exact_mod_cast (round p s x).hi
  have hv : (round p s x).val < upper := (div_lt_div_iff_of_pos_right hs).mpr hm
  refine ⟨x, ?_⟩
  rw [abs_sub_comm, abs_of_pos (by dsimp [x]; linarith [abs_nonneg E])]
  dsimp [x]
  linarith [le_abs_self E]

example := round_error_lt 8 4 (1 / 3) (by norm_num [NoSaturation])
example := round_pos 8 4 (1 / 3) (1 / 4) (by norm_num [NoSaturation])
  (by norm_num) (by norm_num)
example := round_lower 8 4 (-(1 / 3)) (by norm_num [NoSaturation])

/-- Saturation is a real obstruction, not just a technical proof premise. -/
example : 1 < |(round 8 0 (129 : ℝ)).val - 129| := by
  norm_num [val_round, clamp]

end Transformer.CRASP.Fx
