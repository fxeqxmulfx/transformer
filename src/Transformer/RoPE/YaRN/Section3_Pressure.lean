module

public import Transformer.RoPE.YaRN.Section2_Background

/-!
# YaRN §§3.1–3.2: where PI and the NTK-aware base put the interpolation pressure

Peng et al., arXiv:2309.00071v3, §3.1, §3.2 and its footnotes † and ‡.

* `pi_frequency_le`: "as we strech all dimensions equally by a factor `s`, it removes the high
  frequency components of RoPE": no PI frequency exceeds `1/s`, and the first pair, of frequency
  `1`, drops to `1/s`;
* `ntk_factor_monotone`, `ntk_factor_bounds`: the NTK-aware base spreads "the interpolation
  pressure across multiple dimensions by scaling high frequencies less and low frequencies
  more": it divides the pair `d` by `s^{2d/(|D|−2)}` (`coordinate_frequency_adjusted`), a factor
  growing with `d` from `1` at the first pair to `s` at the last;
* `period_frequency_strictMono`, `llama_period_gt_iff`: "there are some dimensions `d` where the
  wavelength is longer than the maximum context length seen during pretraining (`λ > L`)": for
  LLaMA, `b = 10000` (§2.1), `L = 2k = 2048` (footnote §) and the head dimension `|D| = 128`, which
  the manuscript does not state, they are the pairs `d ≥ 41`, the `23` slowest of `64`;
* `eq_of_rotation_eq`: footnote †, "Since the dimension never rotates fully at least once during
  pre-training, if we pick the first token as the anchor, every other token during pre-training
  has an unique distance to it": for `L ≤ λ` the positions `0, …, L − 1` have distinct rotations;
* footnote ‡, "a harmonic mean over `θ_d/s` and `θ_d` converted from a linear interpolation on
  wavelengths", is `harmonicFrequency` in `YaRN.Section3_Bands`.

The bullets of §3.2 hold by Eqs. (11)–(13): `ramp_low`, `ramp_high` and
`blended_frequency_endpoints` leave the pairs with `r > β` untouched and interpolate those with
`r < α` exactly like PI, whose phase over the extended window `sL` is the pretrained one
(`interpolated_phase`). "Only relative positional information is accessible" to the fast pairs is
an interpretation and is not formalized.
-/

@[expose] public section

namespace Transformer.RoPE.YaRN

open RoFormer

/-- §3.1: PI divides every frequency by `s`, so none exceeds `1/s`; the first pair, of frequency
`1`, drops to `1/s`. -/
theorem pi_frequency_le {base s : ℝ} (hbase : 1 ≤ base) (hs : 0 < s) (D d : ℕ) :
    frequency base D d / s ≤ 1 / s ∧ frequency base D 0 / s = 1 / s := by
  have h0 : frequency base D 0 = 1 := by simp [frequency]
  refine ⟨div_le_div_of_nonneg_right ?_ hs.le, by rw [h0]⟩
  exact h0 ▸ frequency_antitone hbase D (Nat.zero_le d)

example : frequency 10000 128 5 / 8 ≤ 1 / 8 ∧ frequency 10000 128 0 / 8 = 1 / 8 :=
  pi_frequency_le (by norm_num) (by norm_num) 128 5

/-- §3.1: the NTK-aware divisor `s^{2d/(|D|−2)}` of the pair `d` grows with `d`: high frequencies
are scaled less, low frequencies more. -/
theorem ntk_factor_monotone {s : ℝ} (hs : 1 ≤ s) {D : ℕ} (hD : 2 < D) :
    Monotone fun d : ℕ => s ^ (2 * (d : ℝ) / ((D : ℝ) - 2)) := by
  intro i j hij
  have hD' : (2 : ℝ) < D := by exact_mod_cast hD
  have hij' : (i : ℝ) ≤ j := by exact_mod_cast hij
  apply Real.rpow_le_rpow_of_exponent_le hs
  apply div_le_div_of_nonneg_right _ (by linarith)
  linarith

example : Monotone fun d : ℕ => (8 : ℝ) ^ (2 * (d : ℝ) / (((128 : ℕ) : ℝ) - 2)) :=
  ntk_factor_monotone (by norm_num) (by norm_num)

/-- §3.1 and Appendix A.2: in a head of `p + 1` pairs the NTK-aware divisors run from `1` at the
first pair to `s` at the last. -/
theorem ntk_factor_bounds {s : ℝ} (hs : 1 ≤ s) {p d : ℕ} (hp : 0 < p) (hd : d ≤ p) :
    1 ≤ s ^ (2 * (d : ℝ) / (((2 * (p + 1) : ℕ) : ℝ) - 2)) ∧
      s ^ (2 * (d : ℝ) / (((2 * (p + 1) : ℕ) : ℝ) - 2)) ≤ s := by
  have hp' : (0 : ℝ) < p := by exact_mod_cast hp
  have hd' : (d : ℝ) ≤ p := by exact_mod_cast hd
  have he : 2 * (d : ℝ) / (((2 * (p + 1) : ℕ) : ℝ) - 2) = d / p := by
    rw [show (((2 * (p + 1) : ℕ) : ℝ) - 2) = 2 * p by push_cast; ring,
      mul_div_mul_left _ _ two_ne_zero]
  rw [he]
  refine ⟨Real.one_le_rpow hs (by positivity), ?_⟩
  calc s ^ ((d : ℝ) / p) ≤ s ^ (1 : ℝ) :=
        Real.rpow_le_rpow_of_exponent_le hs ((div_le_one hp').mpr hd')
    _ = s := Real.rpow_one s

example : 1 ≤ (8 : ℝ) ^ (2 * ((5 : ℕ) : ℝ) / (((2 * (63 + 1) : ℕ) : ℝ) - 2)) ∧
    (8 : ℝ) ^ (2 * ((5 : ℕ) : ℝ) / (((2 * (63 + 1) : ℕ) : ℝ) - 2)) ≤ 8 :=
  ntk_factor_bounds (by norm_num) (by norm_num) (by norm_num)

/-- Eq. (8): for `b > 1` the wavelength grows with the pair index. -/
theorem period_frequency_strictMono {base : ℝ} (hbase : 1 < base) {D : ℕ} (hD : 0 < D) :
    StrictMono fun d : ℕ => period (frequency base D d) := by
  intro i j hij
  have hD' : (0 : ℝ) < D := by exact_mod_cast hD
  have hij' : (i : ℝ) < j := by exact_mod_cast hij
  simp only [period_frequency (by linarith : (0 : ℝ) < base)]
  apply mul_lt_mul_of_pos_left _ Real.two_pi_pos
  exact Real.rpow_lt_rpow_of_exponent_lt hbase (div_lt_div_of_pos_right (by linarith) hD')

example : StrictMono fun d : ℕ => period (frequency 10000 128 d) :=
  period_frequency_strictMono (by norm_num) (by norm_num)

/-- `x^{a/b} < y` from `xᵃ < yᵇ`, for the numerical wavelengths below. -/
theorem rpow_div_lt {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) {a b : ℕ} (hb : 0 < b)
    (h : x ^ a < y ^ b) : x ^ ((a : ℝ) / b) < y := by
  rw [div_eq_mul_inv, Real.rpow_mul hx, Real.rpow_natCast,
    Real.rpow_inv_lt_iff_of_pos (by positivity) hy (by exact_mod_cast hb), Real.rpow_natCast]
  exact h

example : (10000 : ℝ) ^ (((5 : ℕ) : ℝ) / ((8 : ℕ) : ℝ)) < 317 :=
  rpow_div_lt (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `y < x^{a/b}` from `yᵇ < xᵃ`. -/
theorem lt_rpow_div {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) {a b : ℕ} (hb : 0 < b)
    (h : y ^ b < x ^ a) : y < x ^ ((a : ℝ) / b) := by
  rw [div_eq_mul_inv, Real.rpow_mul hx, Real.rpow_natCast,
    Real.lt_rpow_inv_iff_of_pos hy (by positivity) (by exact_mod_cast hb), Real.rpow_natCast]
  exact h

example : (327 : ℝ) < (10000 : ℝ) ^ (((41 : ℕ) : ℝ) / ((64 : ℕ) : ℝ)) :=
  lt_rpow_div (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- §3.2: for LLaMA, `b = 10000`, `L = 2048` and the head dimension `|D| = 128`, the wavelength
`λ_d` exceeds the pretrained context exactly at the pairs `d ≥ 41`: `λ_40 ≈ 1987` and
`λ_41 ≈ 2294`. -/
theorem llama_period_gt_iff (d : ℕ) : 2048 < period (frequency 10000 128 d) ↔ 41 ≤ d := by
  have hmono := period_frequency_strictMono (by norm_num : (1 : ℝ) < 10000) (by norm_num : 0 < 128)
  have h40 : period (frequency 10000 128 40) < 2048 := by
    rw [period_frequency (by norm_num)]
    have hx : (10000 : ℝ) ^ (2 * ((40 : ℕ) : ℝ) / ((128 : ℕ) : ℝ)) < 317 := by
      rw [show 2 * ((40 : ℕ) : ℝ) / ((128 : ℕ) : ℝ) = ((5 : ℕ) : ℝ) / ((8 : ℕ) : ℝ) by norm_num]
      exact rpow_div_lt (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    have hx0 := Real.rpow_nonneg (by norm_num : (0 : ℝ) ≤ 10000)
      (2 * ((40 : ℕ) : ℝ) / ((128 : ℕ) : ℝ))
    nlinarith [Real.pi_lt_d2, Real.pi_pos]
  have h41 : 2048 < period (frequency 10000 128 41) := by
    rw [period_frequency (by norm_num)]
    have hx : (327 : ℝ) < (10000 : ℝ) ^ (2 * ((41 : ℕ) : ℝ) / ((128 : ℕ) : ℝ)) := by
      rw [show 2 * ((41 : ℕ) : ℝ) / ((128 : ℕ) : ℝ) = ((41 : ℕ) : ℝ) / ((64 : ℕ) : ℝ) by norm_num]
      exact lt_rpow_div (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    nlinarith [Real.pi_gt_d2]
  constructor
  · intro h
    by_contra hd
    have := hmono.monotone (by omega : d ≤ 40)
    linarith
  · intro hd
    exact h41.trans_le (hmono.monotone hd)

/-- Footnote †: a pair whose wavelength is at least the pretrained context `L` never turns fully
over the positions `0, …, L − 1`, so their rotations differ: with the first token as the anchor,
every other token has a distance of its own. -/
theorem eq_of_rotation_eq {ω : ℝ} (hω : 0 < ω) {L m n : ℕ} (hL : (L : ℝ) ≤ period ω)
    (hm : m < L) (hn : n < L) (h : rotation (m * ω) = rotation (n * ω)) : m = n := by
  have hc : Real.cos (m * ω) = Real.cos (n * ω) := by
    simpa [rotation] using congr_fun (congr_fun h 0) 0
  have hs : Real.sin (m * ω) = Real.sin (n * ω) := by
    simpa [rotation] using congr_fun (congr_fun h 1) 0
  obtain ⟨k, hk⟩ := Real.Angle.angle_eq_iff_two_pi_dvd_sub.mp (Real.Angle.cos_sin_inj hc hs)
  have hLω : (L : ℝ) * ω ≤ 2 * Real.pi := by rwa [period, le_div_iff₀ hω] at hL
  have hmL : (m : ℝ) < L := by exact_mod_cast hm
  have hnL : (n : ℝ) < L := by exact_mod_cast hn
  have h1 : ((m : ℝ) - n) * ω < L * ω :=
    mul_lt_mul_of_pos_right (by linarith [n.cast_nonneg' (α := ℝ)]) hω
  have h2 : ((n : ℝ) - m) * ω < L * ω :=
    mul_lt_mul_of_pos_right (by linarith [m.cast_nonneg' (α := ℝ)]) hω
  have hk1 : (k : ℝ) < 1 := by nlinarith [Real.pi_pos]
  have hk2 : -1 < (k : ℝ) := by nlinarith [Real.pi_pos]
  have hk0 : k = 0 := by
    have : k < 1 := by exact_mod_cast hk1
    have : -1 < k := by exact_mod_cast hk2
    omega
  rw [hk0, Int.cast_zero, mul_zero, sub_eq_zero] at hk
  exact_mod_cast mul_right_cancel₀ hω.ne' hk

example : (3 : ℕ) = 3 :=
  eq_of_rotation_eq (ω := 1) (L := 6) (by norm_num) (by rw [period]; nlinarith [Real.pi_gt_d2])
    (by norm_num) (by norm_num) rfl

end Transformer.RoPE.YaRN
