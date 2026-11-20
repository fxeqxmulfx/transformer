module

public import Transformer.RoPE.Basic

/-!
# YaRN Appendix A.2: NTK-aware base scaling

Peng et al., *YaRN*, arXiv:2309.00071v3, Appendix A.2, Definition 3 and Eqs. (17)–(22): keep the
positions, `g(m) = m`, and change the base to `b' = b·s^{|D|/(|D|−2)}` (`adjustedBase`), so that
"the lowest frequency needs to be scaled as much as linear positional scaling and the highest
frequency to stay constant".

* `slow_frequency_adjusted`, `slow_phase_preserved`: Eqs. (21)–(22), the slowest frequency is
  divided by exactly `s`, so its phase over `sT` is the pretrained phase over `T`;
* `coordinate_frequency_adjusted`, `highest_frequency_preserved`: the pair `d` is divided by
  `s^{2d/(|D|−2)}`, and the highest frequency stays constant;
* `ntk_phase_eq`, `ntk_phase_out_of_bound`: "given it is not just an interpolation scheme, some
  dimensions are slightly extrapolated to "out-of-bound" values": over the extended window `sT`
  the pair `d` reaches `s^{1−2d/(|D|−2)}` times its pretrained phase over `T`, beyond it for every
  pair but the last;
* `ntk_phase_le_iff`, `ntk_scale_gt`: "the theoretical scale factor `s` does not accurately
  describe the true context extension scale. In practice, the scale value `s` has to be set
  higher than the expected scale for a given context length extension": the pair `d` keeps its
  pretrained phase over `tT` iff `t ≤ s^{2d/(|D|−2)}`, so it is extended by `s^{2d/(|D|−2)}`
  rather than `s`, and keeping a pair below the last within its phase over `tT`, `t > 1`, needs
  `s > t`;
* `adjustedBase_eq_iff`: Eqs. (19)–(22) solved for the scale, `s = (b'/b)^{(|D|−2)/|D|}`.

"The last dimension `d ∈ D` is `|D| − 2`" counts real coordinates, whose frequency is
`b^{−d/|D|}`; in the pair index of Eqs. (6) and (18) the last pair is `d = |D|/2 − 1`
(`slow_frequency_eq_last_pair`). Eq. (21)'s exponent `(|D|−2)/|D|` is that pair's in both
readings. That fine-tuning with this base "yields inferior results to PI" is a measurement, and
no loss or retrieval guarantee is inferred here.
-/

@[expose] public section

namespace Transformer.RoPE.YaRN

/-- YaRN Eq. (19), with a real rotary dimension for the algebraic identities. -/
noncomputable def adjustedBase (base scale dimension : ℝ) : ℝ :=
  base * scale ^ (dimension / (dimension - 2))

/-- Eq. (22): changing the base interpolates the slowest frequency by exactly
the nominal factor. The dimension must exceed two. -/
theorem slow_frequency_adjusted {base scale dimension : ℝ}
    (hbase : 0 < base) (hscale : 0 < scale) (hdimension : 2 < dimension) :
    slowFrequency (adjustedBase base scale dimension) dimension =
      slowFrequency base dimension / scale := by
  have hd : dimension ≠ 0 := ne_of_gt (by linarith)
  have hd2 : dimension - 2 ≠ 0 := ne_of_gt (by linarith)
  unfold slowFrequency adjustedBase
  rw [Real.mul_rpow hbase.le (Real.rpow_nonneg hscale.le _), ← Real.rpow_mul hscale.le]
  rw [show dimension / (dimension - 2) * (-(dimension - 2) / dimension) = -1 by
    field_simp]
  rw [Real.rpow_neg_one]
  simp only [div_eq_mul_inv]

example : slowFrequency (adjustedBase 10000 4 128) 128 =
    slowFrequency 10000 128 / 4 :=
  slow_frequency_adjusted (by norm_num) (by norm_num) (by norm_num)

/-- Appendix A.2: every coordinate has its own scaling factor. In particular,
this is not uniform position interpolation across the frequency grid. -/
theorem coordinate_frequency_adjusted {base scale : ℝ} {dimension : ℕ}
    (hbase : 0 < base) (hscale : 0 < scale) (hdimension : 2 < dimension) (index : ℕ) :
    frequency (adjustedBase base scale dimension) dimension index =
      frequency base dimension index / scale ^ (2 * (index : ℝ) / ((dimension : ℝ) - 2)) := by
  have hd : (dimension : ℝ) ≠ 0 := by exact_mod_cast (by omega : dimension ≠ 0)
  have hd2 : (dimension : ℝ) - 2 ≠ 0 := by
    have : (2 : ℝ) < dimension := by exact_mod_cast hdimension
    linarith
  unfold frequency adjustedBase
  rw [Real.mul_rpow hbase.le (Real.rpow_nonneg hscale.le _), ← Real.rpow_mul hscale.le]
  rw [show (dimension : ℝ) / ((dimension : ℝ) - 2) *
      (-(2 * (index : ℝ)) / (dimension : ℝ)) =
      -(2 * (index : ℝ) / ((dimension : ℝ) - 2)) by field_simp]
  rw [Real.rpow_neg hscale.le]
  simp only [div_eq_mul_inv]

example : frequency (adjustedBase 10000 4 128) 128 7 =
    frequency 10000 128 7 / (4 : ℝ) ^ (2 * (7 : ℝ) / (128 - 2)) := by
  simpa using coordinate_frequency_adjusted (by norm_num : (0 : ℝ) < 10000)
    (by norm_num : (0 : ℝ) < 4) (by omega : 2 < 128) 7

/-- Appendix A.2: the highest frequency is fixed even when the base changes. -/
theorem highest_frequency_preserved (base scale : ℝ) (dimension : ℕ) :
    frequency (adjustedBase base scale dimension) dimension 0 =
      frequency base dimension 0 := by
  simp [frequency]

/-- Eq. (21): matching the slowest phase range follows from the frequency
identity, without asserting that a trained model uses that coordinate. -/
theorem slow_phase_preserved {base scale dimension : ℝ}
    (hbase : 0 < base) (hscale : 0 < scale) (hdimension : 2 < dimension) (distance : ℝ) :
    slowFrequency (adjustedBase base scale dimension) dimension * (scale * distance) =
      slowFrequency base dimension * distance := by
  rw [slow_frequency_adjusted hbase hscale hdimension, interpolated_phase hscale.ne']

example : slowFrequency (adjustedBase 10000 4 128) 128 * (4 * 2047) =
    slowFrequency 10000 128 * 2047 :=
  slow_phase_preserved (by norm_num) (by norm_num) (by norm_num) 2047

/-- Appendix A.2: over the extended window `sT` the NTK-aware pair `d` reaches `s^{1−2d/(|D|−2)}`
times its phase over `T`. -/
theorem ntk_phase_eq {base s : ℝ} (hbase : 0 < base) (hs : 0 < s) {D : ℕ} (hD : 2 < D) (d : ℕ)
    (T : ℝ) :
    frequency (adjustedBase base s D) D d * (s * T) =
      s ^ (1 - 2 * (d : ℝ) / ((D : ℝ) - 2)) * (frequency base D d * T) := by
  rw [coordinate_frequency_adjusted hbase hs hD, Real.rpow_sub hs, Real.rpow_one]
  ring

example : frequency (adjustedBase 10000 8 ((128 : ℕ) : ℝ)) 128 5 * (8 * 2048) =
    8 ^ (1 - 2 * ((5 : ℕ) : ℝ) / (((128 : ℕ) : ℝ) - 2)) * (frequency 10000 128 5 * 2048) :=
  ntk_phase_eq (by norm_num) (by norm_num) (by norm_num) 5 2048

/-- Appendix A.2: for `s > 1` every NTK-aware pair below the last, `2d + 2 < |D|`, goes beyond its
pretrained phase over the extended window. -/
theorem ntk_phase_out_of_bound {base s T : ℝ} (hbase : 0 < base) (hs : 1 < s) (hT : 0 < T)
    {D d : ℕ} (hD : 2 < D) (hd : 2 * d + 2 < D) :
    frequency base D d * T < frequency (adjustedBase base s D) D d * (s * T) := by
  rw [ntk_phase_eq hbase (by linarith) hD]
  have hd' : (2 * d + 2 : ℝ) < D := by exact_mod_cast hd
  have he : 0 < 1 - 2 * (d : ℝ) / ((D : ℝ) - 2) := by
    rw [sub_pos, div_lt_one (by linarith)]
    linarith
  have h1 := Real.one_lt_rpow hs he
  have h0 : 0 < frequency base D d * T := mul_pos (Real.rpow_pos_of_pos hbase _) hT
  nlinarith

example : frequency 10000 128 5 * 2048 <
    frequency (adjustedBase 10000 8 ((128 : ℕ) : ℝ)) 128 5 * (8 * 2048) :=
  ntk_phase_out_of_bound (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Appendix A.2: the NTK-aware pair `d` keeps its pretrained phase over `tT` iff
`t ≤ s^{2d/(|D|−2)}`. -/
theorem ntk_phase_le_iff {base s t T : ℝ} (hbase : 0 < base) (hs : 0 < s) (hT : 0 < T)
    {D : ℕ} (hD : 2 < D) (d : ℕ) :
    frequency (adjustedBase base s D) D d * (t * T) ≤ frequency base D d * T ↔
      t ≤ s ^ (2 * (d : ℝ) / ((D : ℝ) - 2)) := by
  rw [coordinate_frequency_adjusted hbase hs hD]
  have hθT : 0 < frequency base D d * T := mul_pos (Real.rpow_pos_of_pos hbase _) hT
  have hp : 0 < s ^ (2 * (d : ℝ) / ((D : ℝ) - 2)) := Real.rpow_pos_of_pos hs _
  rw [div_mul_eq_mul_div, div_le_iff₀ hp]
  constructor <;> intro h <;> nlinarith

example : frequency (adjustedBase 10000 4 ((128 : ℕ) : ℝ)) 128 32 * (2 * 1) ≤
      frequency 10000 128 32 * 1 ↔
    2 ≤ (4 : ℝ) ^ (2 * ((32 : ℕ) : ℝ) / (((128 : ℕ) : ℝ) - 2)) :=
  ntk_phase_le_iff (by norm_num) (by norm_num) (by norm_num) (by norm_num) 32

/-- Appendix A.2: keeping a pair below the last within its pretrained phase over `tT`, `t > 1`,
needs an NTK-aware scale `s > t`. -/
theorem ntk_scale_gt {base s t T : ℝ} (hbase : 0 < base) (hs : 0 < s) (ht : 1 < t) (hT : 0 < T)
    {D d : ℕ} (hD : 2 < D) (hd : 2 * d + 2 < D)
    (h : frequency (adjustedBase base s D) D d * (t * T) ≤ frequency base D d * T) : t < s := by
  rw [ntk_phase_le_iff hbase hs hT hD] at h
  have hd' : (2 * d + 2 : ℝ) < D := by exact_mod_cast hd
  have he0 : 0 ≤ 2 * (d : ℝ) / ((D : ℝ) - 2) := div_nonneg (by positivity) (by linarith)
  have he1 : 2 * (d : ℝ) / ((D : ℝ) - 2) < 1 := by
    rw [div_lt_one (by linarith)]
    linarith
  rcases le_or_gt s 1 with hs1 | hs1
  · linarith [Real.rpow_le_one hs.le hs1 he0]
  · linarith [Real.rpow_lt_self_of_one_lt hs1 he1]

example : (4 : ℝ) ^ (2 * ((32 : ℕ) : ℝ) / (((128 : ℕ) : ℝ) - 2)) < 4 :=
  ntk_scale_gt (base := 10000) (T := 1) (D := 128) (d := 32) (by norm_num) (by norm_num)
    (Real.one_lt_rpow (by norm_num) (by norm_num)) (by norm_num) (by norm_num) (by norm_num)
    ((ntk_phase_le_iff (by norm_num) (by norm_num) (by norm_num) (by norm_num) 32).mpr le_rfl)

/-- Eqs. (19)–(22) solved for the scale: `b' = b·s^{|D|/(|D|−2)}` iff `s = (b'/b)^{(|D|−2)/|D|}`. -/
theorem adjustedBase_eq_iff {base base' s D : ℝ} (hbase : 0 < base) (hbase' : 0 < base')
    (hs : 0 < s) (hD : 2 < D) :
    adjustedBase base s D = base' ↔ s = (base' / base) ^ ((D - 2) / D) := by
  have he : D / (D - 2) ≠ 0 := (div_pos (by linarith) (by linarith)).ne'
  rw [← inv_div D (D - 2), adjustedBase]
  constructor
  · intro h
    rw [← h, mul_div_cancel_left₀ _ hbase.ne', Real.rpow_rpow_inv hs.le he]
  · intro h
    rw [h, Real.rpow_inv_rpow (div_nonneg hbase'.le hbase.le) he]
    field_simp

example : adjustedBase 10000 8 128 = 1000000 ↔
    (8 : ℝ) = (1000000 / 10000) ^ (((128 : ℝ) - 2) / 128) :=
  adjustedBase_eq_iff (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Transformer.RoPE.YaRN
