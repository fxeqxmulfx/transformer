module

public import Transformer.RoPE.Basic

/-!
# Coordinatewise context rescaling

Shang et al., *LongRoPE2*, arXiv:2502.20082v1, §2.2, Eqs. (8)–(10),
and §3.3. Only the exact phase-range constraint and mixed-window rule are
asserted. The search's benchmark performance is an empirical result.
-/

@[expose] public section

namespace Transformer.RoPE.LongRoPE2

/-- Eq. (8): each coordinate has its own frequency divisor. -/
noncomputable def scaledFrequency (omega factor : ℝ) : ℝ := omega / factor

/-- Eqs. (9)–(10): the rescaled endpoint stays in the training phase range
exactly when the coordinate's scale factor covers the extension ratio. -/
theorem phase_range_iff {omega factor train target : ℝ}
    (homega : 0 < omega) (hfactor : 0 < factor) (htrain : 0 < train) :
    target * scaledFrequency omega factor ≤ train * omega ↔ target / train ≤ factor := by
  unfold scaledFrequency
  rw [← mul_div_assoc, div_le_iff₀ hfactor, div_le_iff₀ htrain]
  constructor <;> intro h
  · nlinarith
  · nlinarith

example : (8192 : ℝ) * scaledFrequency 1 4 ≤ 2048 * 1 ↔
    (8192 : ℝ) / 2048 ≤ 4 := phase_range_iff (by norm_num) (by norm_num) (by norm_num)

/-- Eq. (9): endpoint coverage implies coverage at every nonnegative distance
in the target context. The frequency is still the actual rescaled frequency. -/
theorem all_phases_in_range {omega factor train target : ℝ}
    (homega : 0 < omega) (hfactor : 0 < factor) (htrain : 0 < train)
    (hscale : target / train ≤ factor) :
    ∀ distance ∈ Set.Icc 0 target,
      0 ≤ distance * scaledFrequency omega factor ∧
        distance * scaledFrequency omega factor ≤ train * omega := by
  intro distance hdistance
  have hpositive : 0 < scaledFrequency omega factor := div_pos homega hfactor
  exact ⟨mul_nonneg hdistance.1 hpositive.le,
    (mul_le_mul_of_nonneg_right hdistance.2 hpositive.le).trans
      ((phase_range_iff homega hfactor htrain).2 hscale)⟩

example : ∀ distance ∈ Set.Icc 0 (8192 : ℝ),
    0 ≤ distance * scaledFrequency 1 4 ∧ distance * scaledFrequency 1 4 ≤ 2048 * 1 :=
  all_phases_in_range (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Rescaling multiplies the genuine trigonometric period by the coordinate's
factor. This does not assume that training has covered that period. -/
theorem scaled_period (omega factor : ℝ) :
    period (scaledFrequency omega factor) = factor * period omega :=
  (interpolated_field_period factor omega).2

/-- §3.3: select the pretrained frequency inside the original context and the
rescaled frequency for the extended context. -/
noncomputable def mixedFrequency (train length omega factor : ℝ) : ℝ :=
  if length ≤ train then omega else scaledFrequency omega factor

/-- §3.3: the original short-context positional encoding is exactly preserved. -/
theorem mixed_frequency_short {train length : ℝ} (hshort : length ≤ train)
    (omega factor distance : ℝ) :
    mixedFrequency train length omega factor * distance = omega * distance := by
  rw [mixedFrequency, ite_eq_left hshort]

example : mixedFrequency 2048 1024 1 4 * 1023 = (1 : ℝ) * 1023 :=
  mixed_frequency_short (by norm_num) 1 4 1023

/-- §3.3: the extended-context branch obeys exactly the same phase bound. -/
theorem mixed_frequency_long {train length omega factor : ℝ}
    (hlong : train < length) (homega : 0 < omega) (hfactor : 0 < factor)
    (htrain : 0 < train) (hscale : length / train ≤ factor) :
    length * mixedFrequency train length omega factor ≤ train * omega := by
  rw [mixedFrequency, ite_eq_right (not_le.mpr hlong)]
  exact (phase_range_iff homega hfactor htrain).2 hscale

example : (8192 : ℝ) * mixedFrequency 2048 8192 1 4 ≤ 2048 * 1 :=
  mixed_frequency_long (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Transformer.RoPE.LongRoPE2
