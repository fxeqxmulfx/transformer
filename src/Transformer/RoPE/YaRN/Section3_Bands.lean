module

public import Transformer.RoPE.YaRN.SectionA2_NTK

/-!
# Frequency-dependent interpolation

Peng et al., arXiv:2309.00071v3, §3.2, NTK-by-parts interpolation. The blend
weight is the paper's ramp value in [0,1]. Its bounds apply to every ramp
shape with that range and preserve the distinction between PI and no scaling.
-/

@[expose] public section

namespace Transformer.RoPE.YaRN

/-- YaRN §3.2: interpolate low frequencies and retain high frequencies. -/
noncomputable def blendedFrequency (omega scale weight : ℝ) : ℝ :=
  (1 - weight) * (omega / scale) + weight * omega

/-- §3.2: the bandwise frequency always lies between PI and the original
frequency. This follows for the actual convex blend, not an assumed bound. -/
theorem blended_frequency_bounds {omega scale weight : ℝ}
    (homega : 0 < omega) (hscale : 1 ≤ scale) (hweight : weight ∈ Set.Icc 0 1) :
    omega / scale ≤ blendedFrequency omega scale weight ∧
      blendedFrequency omega scale weight ≤ omega := by
  have hs : 0 < scale := lt_of_lt_of_le (by norm_num) hscale
  have hle : omega / scale ≤ omega := (div_le_iff₀ hs).2 (by nlinarith)
  unfold blendedFrequency
  constructor
  · nlinarith [mul_nonneg hweight.1 (sub_nonneg.mpr hle)]
  · nlinarith [mul_nonneg (sub_nonneg.mpr hweight.2) (sub_nonneg.mpr hle)]

example : (1 : ℝ) / 4 ≤ blendedFrequency 1 4 (1 / 2) ∧
    blendedFrequency 1 4 (1 / 2) ≤ 1 :=
  blended_frequency_bounds (by norm_num) (by norm_num) ⟨by norm_num, by norm_num⟩

/-- The two regimes of §3.2 are recovered exactly at the ramp endpoints. -/
theorem blended_frequency_endpoints (omega scale : ℝ) :
    blendedFrequency omega scale 0 = omega / scale ∧
      blendedFrequency omega scale 1 = omega := by
  simp [blendedFrequency]

/-- Nonuniform interpolation preserves a positive frequency, hence still
defines a genuine rotation with finite positive period. YaRN §3.2. -/
theorem blended_frequency_positive {omega scale weight : ℝ}
    (homega : 0 < omega) (hscale : 1 ≤ scale) (hweight : weight ∈ Set.Icc 0 1) :
    0 < blendedFrequency omega scale weight := by
  exact lt_of_lt_of_le (div_pos homega (by linarith))
    (blended_frequency_bounds homega hscale hweight).1

example : 0 < blendedFrequency 1 4 (1 / 2) :=
  blended_frequency_positive (by norm_num) (by norm_num) ⟨by norm_num, by norm_num⟩

end Transformer.RoPE.YaRN
