module

public import Transformer.RoPE.YaRN.SectionA2_NTK

/-!
# Frequency-dependent interpolation

Peng et al., arXiv:2309.00071v3, §3.2, NTK-by-parts interpolation. The blend
weight is the paper's ramp value in [0,1]. Its bounds apply to every ramp
shape with that range and preserve the distinction between PI and no scaling.

Footnote ‡ names a variant, "a harmonic mean over `θ_d/s` and `θ_d` converted from a linear
interpolation on wavelengths" (`harmonicFrequency`): the two descriptions agree
(`harmonicFrequency_eq`), its ends are PI and the original frequency
(`harmonicFrequency_endpoints`), and it lies between PI and Eq. (13)'s band
(`harmonic_le_blended`).
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

/-- Footnote ‡: the band whose wavelength interpolates linearly between PI's `sλ` and the
original `λ`. -/
noncomputable def harmonicFrequency (ω s γ : ℝ) : ℝ := ω / ((1 - γ) * s + γ)

/-- Footnote ‡: its wavelength is `(1 − γ)sλ + γλ`, and it is the harmonic mean of `θ/s` and `θ`
with the weights `1 − γ` and `γ`. -/
theorem harmonicFrequency_eq (ω s γ : ℝ) :
    period (harmonicFrequency ω s γ) = (1 - γ) * (s * period ω) + γ * period ω ∧
      harmonicFrequency ω s γ = 1 / ((1 - γ) / (ω / s) + γ / ω) := by
  refine ⟨?_, ?_⟩
  · rw [period, period, harmonicFrequency, div_div_eq_mul_div]
    ring
  · rw [harmonicFrequency, div_div_eq_mul_div, ← add_div, one_div_div]

/-- Footnote ‡: like Eq. (13)'s band, the harmonic one is PI at `γ = 0` and the original
frequency at `γ = 1`. -/
theorem harmonicFrequency_endpoints (ω s : ℝ) :
    harmonicFrequency ω s 0 = ω / s ∧ harmonicFrequency ω s 1 = ω := by
  simp [harmonicFrequency]

/-- Footnote ‡: the harmonic band lies between PI and Eq. (13)'s band,
`θ/s ≤ θ/((1 − γ)s + γ) ≤ (1 − γ)θ/s + γθ`. -/
theorem harmonic_le_blended {ω s γ : ℝ} (hω : 0 < ω) (hs : 1 ≤ s) (hγ : γ ∈ Set.Icc 0 1) :
    ω / s ≤ harmonicFrequency ω s γ ∧ harmonicFrequency ω s γ ≤ blendedFrequency ω s γ := by
  obtain ⟨h0, h1⟩ := hγ
  have hs0 : 0 < s := by linarith
  have hX : 0 < (1 - γ) * s + γ := by nlinarith
  unfold harmonicFrequency blendedFrequency
  refine ⟨div_le_div_of_nonneg_left hω.le hX (by nlinarith), ?_⟩
  rw [div_le_iff₀ hX]
  have key : ((1 - γ) * (ω / s) + γ * ω) * ((1 - γ) * s + γ) - ω =
      ω * γ * (1 - γ) * (s - 1) ^ 2 / s := by
    field_simp
    ring
  have : 0 ≤ ω * γ * (1 - γ) * (s - 1) ^ 2 / s :=
    div_nonneg (mul_nonneg (mul_nonneg (mul_nonneg hω.le h0) (sub_nonneg.mpr h1))
      (sq_nonneg _)) hs0.le
  linarith

example : (1 : ℝ) / 4 ≤ harmonicFrequency 1 4 (1 / 2) ∧
    harmonicFrequency 1 4 (1 / 2) ≤ blendedFrequency 1 4 (1 / 2) :=
  harmonic_le_blended (by norm_num) (by norm_num) ⟨by norm_num, by norm_num⟩

end Transformer.RoPE.YaRN
