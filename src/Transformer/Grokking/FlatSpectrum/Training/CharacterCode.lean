module

public import Transformer.Grokking.FlatSpectrum.RealKernels
public import Transformer.Grokking.FlatSpectrum.Training.MirroredCode

/-!
# The answer code `ψ` of `ℤ/p` is mirrored

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE11.
The answer code of the note is `ψ(c) = exp (2πi c/p)` on `ℤ/p`. Two codewords are at the
squared distance `|ψ(t) − ψ(c)|² = 2 − 2 cos (2π (t − c)/p)` (`norm_stdAddChar_sub_sq`), which
is least at `c = t ± 1` (`cos_two_pi_mul_div_le`), and
`ψ(t + 1) + ψ(t − 1) = 2 cos (2π/p) ψ(t)`. So for `p ≥ 2` the code is mirrored at the gap
`g = 1 − cos (2π/p)` (`isMirroredCode_stdAddChar`). The gap is positive
(`one_sub_cos_two_pi_div_pos`), and `g < 2` for `p ≥ 3` (`one_sub_cos_two_pi_div_lt_two`); for
`p = 2` the code is `±1` and `g = 2`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Real

variable {p : ℕ}

/-- `cos (2π k/p) ≤ cos (2π/p)` for `0 < k < p`: of the `p`-th roots of unity other than `1`,
the nearest to `1` are the two next to it. -/
lemma cos_two_pi_mul_div_le {k : ℕ} (hk : 0 < k) (hkp : k < p) :
    cos (2 * π * k / p) ≤ cos (2 * π / p) := by
  have hp : (0 : ℝ) < p := by exact_mod_cast hk.trans hkp
  have hk1 : (1 : ℝ) ≤ k := by exact_mod_cast hk
  have hkp' : (k : ℝ) + 1 ≤ p := by exact_mod_cast hkp
  rcases le_total (2 * π * k / p) π with h | h
  · exact cos_le_cos_of_nonneg_of_le_pi (by positivity) h
      (div_le_div_of_nonneg_right (le_mul_of_one_le_right (by positivity) hk1) hp.le)
  · rw [← cos_two_pi_sub (2 * π * k / p)]
    refine cos_le_cos_of_nonneg_of_le_pi (by positivity) (by linarith) ?_
    rw [le_sub_iff_add_le, ← add_div, div_le_iff₀ hp]
    nlinarith [mul_le_mul_of_nonneg_left hkp' (by positivity : (0 : ℝ) ≤ 2 * π)]

/-- The gap `1 − cos (2π/p)` of the code `ψ` is positive for `p ≥ 2`. -/
lemma one_sub_cos_two_pi_div_pos (hp : 2 ≤ p) : 0 < 1 - cos (2 * π / p) := by
  have hp' : (2 : ℝ) ≤ p := by exact_mod_cast hp
  have h0 : 0 < 2 * π / p := div_pos (by positivity) (by linarith)
  have h1 : 2 * π / p < 2 * π := by
    rw [div_lt_iff₀ (by linarith)]
    nlinarith [pi_pos]
  have h2 : cos (2 * π / p) ≠ 1 := by
    rw [Ne, cos_eq_one_iff_of_lt_of_lt (by linarith) h1]
    exact h0.ne'
  linarith [lt_of_le_of_ne (cos_le_one _) h2]

/-- The gap `1 − cos (2π/p)` of the code `ψ` is less than `2` for `p ≥ 3`. -/
lemma one_sub_cos_two_pi_div_lt_two (hp : 3 ≤ p) : 1 - cos (2 * π / p) < 2 := by
  have hp' : (3 : ℝ) ≤ p := by exact_mod_cast hp
  have h1 : 2 * π / p < π := by
    rw [div_lt_iff₀ (by linarith)]
    nlinarith [pi_pos]
  have h2 := cos_lt_cos_of_nonneg_of_le_pi (by positivity : (0 : ℝ) ≤ 2 * π / p) le_rfl h1
  rw [cos_pi] at h2
  linarith

variable [NeZero p]

/-- `|ψ(t) − ψ(c)|² = 2 − 2 cos (2π (t − c).val / p)`. -/
lemma norm_stdAddChar_sub_sq (t c : ZMod p) :
    ‖(ZMod.stdAddChar t : ℂ) - ZMod.stdAddChar c‖ ^ 2
      = 2 - 2 * cos (2 * π * (t - c).val / p) := by
  have h : (ZMod.stdAddChar t : ℂ) = ZMod.stdAddChar (t - c) * ZMod.stdAddChar c := by
    rw [← AddChar.map_add_eq_mul, sub_add_cancel]
  rw [h, show (ZMod.stdAddChar (t - c) : ℂ) * ZMod.stdAddChar c - ZMod.stdAddChar c =
      (ZMod.stdAddChar (t - c) - 1) * ZMod.stdAddChar c by ring, norm_mul,
    show ‖(ZMod.stdAddChar c : ℂ)‖ = 1 by simp, mul_one, stdAddChar_eq_exp, Complex.sq_norm,
    Complex.normSq_apply]
  simp only [Complex.sub_re, Complex.sub_im, Complex.exp_ofReal_mul_I_re,
    Complex.exp_ofReal_mul_I_im, Complex.one_re, Complex.one_im, sub_zero]
  linear_combination sin_sq_add_cos_sq (2 * π * (t - c).val / p)

/-- **The code `ψ` of `ℤ/p` is mirrored at the gap `1 − cos (2π/p)`**
(`review/flat_spectrum.md`, item XE11), for `p ≥ 2`: the neighbours of `ψ(t)` are
`ψ(t + 1)` and `ψ(t − 1)`. -/
theorem isMirroredCode_stdAddChar (hp : 2 ≤ p) :
    IsMirroredCode (ZMod.stdAddChar : ZMod p → ℂ) (1 - cos (2 * π / p)) := by
  have : Fact (1 < p) := ⟨hp⟩
  refine ⟨fun c => AddChar.norm_apply _ c, fun t c hc => ?_,
    fun t => ⟨t + 1, t - 1, ?_, ?_, ?_⟩⟩
  · rw [norm_stdAddChar_sub_sq]
    have hv : (t - c).val ≠ 0 := (ZMod.val_eq_zero _).not.mpr (sub_ne_zero.mpr hc.symm)
    have := cos_two_pi_mul_div_le (Nat.pos_of_ne_zero hv) (ZMod.val_lt (t - c))
    linarith
  · exact fun h' => one_ne_zero (by linear_combination h' : (1 : ZMod p) = 0)
  · exact fun h' => one_ne_zero (by linear_combination -h' : (1 : ZMod p) = 0)
  · have h1 := ofReal_cos_eq (p := p) 1
    rw [ZMod.val_one, Nat.cast_one, mul_one] at h1
    rw [show 2 * (1 - (1 - cos (2 * π / p))) = 2 * cos (2 * π / p) by ring, sub_eq_add_neg,
      AddChar.map_add_eq_mul, AddChar.map_add_eq_mul, Complex.ofReal_mul, Complex.ofReal_ofNat,
      h1]
    ring

example := cos_two_pi_mul_div_le (p := 5) (k := 2) (by norm_num) (by norm_num)

example := one_sub_cos_two_pi_div_pos (p := 3) (by norm_num)

example := one_sub_cos_two_pi_div_lt_two (p := 3) (by norm_num)

example := norm_stdAddChar_sub_sq (p := 5) 1 2

/-- The code `ψ` of `ℤ/3` is mirrored at the gap `1 − cos (2π/3)`. -/
example := isMirroredCode_stdAddChar (p := 3) (by norm_num)

end Transformer.Grokking.FlatSpectrum.Training
