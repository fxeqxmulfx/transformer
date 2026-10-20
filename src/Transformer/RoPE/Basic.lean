module

public import Transformer.GPTMini.RoPE
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Mathlib.MeasureTheory.Integral.Bochner.Basic

/-!
# RoPE frequencies, fields and positional contrast

Shared notation for Liu et al., arXiv:2310.05209v2, §3; Peng et al.,
arXiv:2309.00071v3, §2; and Wu, Liu and Jadbabaie, arXiv:2607.07678v1, §3.
`dimension` is the rotary dimension, not the model width. The scalar contrast
objective is kept separate from the content-dependent attention score.
-/

@[expose] public section

open MeasureTheory Set

namespace Transformer.RoPE

/-- The geometric frequency grid, Liu et al. §3.2 and YaRN Eq. (7). -/
noncomputable def frequency (base : ℝ) (dimension index : ℕ) : ℝ :=
  base ^ (-(2 * (index : ℝ)) / (dimension : ℝ))

/-- The last pair's frequency, the slowest for `base >= 1` in an even dimension `dimension > 2`.
The real-valued dimension permits the same algebra without integer division. -/
noncomputable def slowFrequency (base dimension : ℝ) : ℝ :=
  base ^ (-(dimension - 2) / dimension)

/-- One full rotation, Liu et al. §3 and LongRoPE2 Eq. (6). -/
noncomputable def period (omega : ℝ) : ℝ := 2 * Real.pi / omega

/-- The monotone half-period field, Wu et al. §3.2. -/
noncomputable def field (omega : ℝ) : ℝ := Real.pi / omega

/-- Positional contrast against distance zero, Wu et al. §3.2. -/
noncomputable def contrast (omega distance : ℝ) : ℝ :=
  1 - Real.cos (omega * distance)

/-- Wu et al. Definition 3: positive frequency and half-period coverage. -/
def Admissible (width omega : ℝ) : Prop :=
  0 < omega ∧ omega * width ≤ Real.pi

/-- Wu et al. §3.3: average contrast under the dependency distribution. -/
noncomputable def utility (mu : Measure ℝ) (omega : ℝ) : ℝ :=
  ∫ distance, contrast omega distance ∂mu

/-- The new frequency notation is exactly the existing executable-model
formalization of `inv_freq`; no alternative frequency grid is introduced. -/
theorem frequency_eq_invFreq (base : ℝ) (dimension : ℕ)
    (index : Fin (dimension / 2)) :
    frequency base dimension index = GPTMini.invFreq dimension base index := by
  simp [frequency, GPTMini.invFreq, Nat.cast_mul]

/-- Liu et al. §3.2: `slowFrequency` is the last actual coordinate of every
even rotary head, including the two-dimensional limiting case. -/
theorem slow_frequency_eq_last_pair (base : ℝ) (pairs : ℕ) :
    slowFrequency base (2 * ((pairs : ℝ) + 1)) =
      frequency base (2 * (pairs + 1)) pairs := by
  unfold slowFrequency frequency
  congr 1
  push_cast
  ring

/-- Liu et al. §3.2: frequencies decrease with coordinate index for base at
least one. The proof also covers a zero dimension, where real division is zero. -/
theorem frequency_antitone {base : ℝ} (hbase : 1 ≤ base) (dimension : ℕ) :
    Antitone (frequency base dimension) := by
  intro i j hij
  apply Real.rpow_le_rpow_of_exponent_le hbase
  apply div_le_div_of_nonneg_right _ (Nat.cast_nonneg dimension)
  have hcast : (i : ℝ) ≤ (j : ℝ) := by exact_mod_cast hij
  linarith

example : Antitone (frequency 10000 128) := frequency_antitone (by norm_num) 128

/-- The slowest frequency really bounds every coordinate in the finite head;
the highest coordinate has frequency one. Liu et al. §3.2. -/
theorem frequency_extremes {base : ℝ} (hbase : 1 ≤ base) (pairs : ℕ)
    (index : Fin (pairs + 1)) :
    slowFrequency base (2 * ((pairs : ℝ) + 1)) ≤
      frequency base (2 * (pairs + 1)) index ∧
      frequency base (2 * (pairs + 1)) index ≤ 1 := by
  rw [slow_frequency_eq_last_pair]
  constructor
  · exact frequency_antitone hbase _ (by have := index.isLt; omega)
  · have h := frequency_antitone hbase (2 * (pairs + 1)) (Nat.zero_le index.val)
    simpa only [frequency, Nat.cast_zero, mul_zero, neg_zero, zero_div, Real.rpow_zero] using h

example : slowFrequency 10000 (2 * ((63 : ℝ) + 1)) ≤ frequency 10000 (2 * (63 + 1)) 7 ∧
    frequency 10000 (2 * (63 + 1)) 7 ≤ 1 :=
  frequency_extremes (by norm_num) 63 ⟨7, by omega⟩

/-- Wu et al. §3.2: contrast is bounded between zero and two. -/
theorem contrast_bounds (omega distance : ℝ) :
    0 ≤ contrast omega distance ∧ contrast omega distance ≤ 2 := by
  constructor <;> unfold contrast
  · linarith [Real.cos_le_one (omega * distance)]
  · linarith [Real.neg_one_le_cos (omega * distance)]

/-- Wu et al. Definition 3, equivalence with half-period coverage. -/
theorem admissible_iff_field {width omega : ℝ} (homega : 0 < omega) :
    Admissible width omega ↔ width ≤ field omega := by
  rw [Admissible, and_iff_right homega, field, le_div_iff₀ homega, mul_comm]

example : Admissible 1 Real.pi ↔ (1 : ℝ) ≤ field Real.pi :=
  admissible_iff_field Real.pi_pos

/-- Wu et al. §3.3: bounded contrast is integrable for every finite dependency
measure, including atomic and long-tailed distributions. -/
theorem contrast_integrable (mu : Measure ℝ) [IsFiniteMeasure mu] (omega : ℝ) :
    Integrable (contrast omega) mu := by
  apply (integrable_const (2 : ℝ)).mono'
  · exact (continuous_const.sub (Real.continuous_cos.comp
      (continuous_const.mul continuous_id))).aestronglyMeasurable
  · exact Filter.Eventually.of_forall fun distance => by
      rw [Real.norm_eq_abs, abs_of_nonneg (contrast_bounds omega distance).1]
      exact (contrast_bounds omega distance).2

example : Integrable (contrast 1) (Measure.dirac 1) :=
  contrast_integrable (Measure.dirac 1) 1

/-- Wu et al. §4.1: interpolation preserves a phase after matching dilation. -/
theorem interpolated_phase {scale : ℝ} (hscale : scale ≠ 0)
    (omega distance : ℝ) :
    (omega / scale) * (scale * distance) = omega * distance := by
  field_simp

example : ((3 : ℝ) / 2) * (2 * 5) = 3 * 5 := interpolated_phase (by norm_num) 3 5

/-- Wu et al. §4.1: both the field and the full period expand under PI. -/
theorem interpolated_field_period (scale omega : ℝ) :
    field (omega / scale) = scale * field omega ∧
      period (omega / scale) = scale * period omega := by
  constructor <;> simp only [field, period, div_eq_mul_inv, mul_inv_rev, inv_inv] <;> ring

example : field ((3 : ℝ) / 2) = 2 * field 3 ∧
    period ((3 : ℝ) / 2) = 2 * period 3 := interpolated_field_period 2 3

end Transformer.RoPE
