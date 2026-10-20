module

public import Transformer.RoPE.YaRN.Section3_Bands

/-!
# The actual YaRN ramp and context-derived scale

Peng et al., arXiv:2309.00071v3, §3.2, Eqs. (10)–(13), and §3.4.
The defaults `alpha = 1`, `beta = 32` are the paper's empirical LLaMA
recommendation, not universal optima. Clipping implements Eq. (11) when
`alpha < beta`, including both continuous boundary values.
-/

@[expose] public section

namespace Transformer.RoPE.YaRN

/-- Eq. (10): rotations over the original training context, not the target. -/
noncomputable def rotations (original omega : ℝ) : ℝ :=
  original * omega / (2 * Real.pi)

/-- Eq. (11), written as a clipped affine ramp. -/
noncomputable def ramp (alpha beta count : ℝ) : ℝ :=
  max 0 (min 1 ((count - alpha) / (beta - alpha)))

/-- The actual clipped ramp supplies the hypothesis of the blend bounds. -/
theorem ramp_bounds (alpha beta count : ℝ) : ramp alpha beta count ∈ Set.Icc 0 1 := by
  constructor
  · exact le_max_left _ _
  · exact max_le (by norm_num) (min_le_left _ _)

/-- Eq. (11): at or below the lower rotation-count boundary the weight is zero. -/
theorem ramp_low {alpha beta count : ℝ} (hab : alpha < beta) (hc : count ≤ alpha) :
    ramp alpha beta count = 0 := by
  have h : (count - alpha) / (beta - alpha) ≤ 0 :=
    div_nonpos_of_nonpos_of_nonneg (by linarith) (by linarith)
  unfold ramp
  exact max_eq_left ((min_le_right _ _).trans h)

example : ramp 1 32 (1 / 2) = 0 := ramp_low (by norm_num) (by norm_num)

/-- Eq. (11): high-frequency bands are left unscaled. -/
theorem ramp_high {alpha beta count : ℝ} (hab : alpha < beta) (hc : beta ≤ count) :
    ramp alpha beta count = 1 := by
  have h : 1 ≤ (count - alpha) / (beta - alpha) :=
    (le_div_iff₀ (by linarith : 0 < beta - alpha)).2 (by linarith)
  simp [ramp, min_eq_left h]

example : ramp 1 32 64 = 1 := ramp_high (by norm_num) (by norm_num)

/-- Eq. (11): the intermediate regime is linear in rotation count. -/
theorem ramp_middle {alpha beta count : ℝ} (hab : alpha < beta)
    (hc : count ∈ Set.Icc alpha beta) :
    ramp alpha beta count = (count - alpha) / (beta - alpha) := by
  have h0 : 0 ≤ (count - alpha) / (beta - alpha) :=
    div_nonneg (by linarith [hc.1]) (by linarith)
  have h1 : (count - alpha) / (beta - alpha) ≤ 1 :=
    (div_le_iff₀ (by linarith : 0 < beta - alpha)).2 (by linarith [hc.2])
  simp [ramp, min_eq_right h1, max_eq_right h0]

example : ramp 1 32 16 = (16 - 1 : ℝ) / (32 - 1) :=
  ramp_middle (by norm_num) ⟨by norm_num, by norm_num⟩

/-- §3.4 fixed-context scaling, clamped to one for an unextended context.
Using the current sequence length instead gives the source's dynamic rule. -/
noncomputable def contextScale (original target : ℝ) : ℝ := max 1 (target / original)

/-- No frequency compression or temperature sharpening below scale one. -/
theorem context_scale_ge_one (original target : ℝ) : 1 ≤ contextScale original target :=
  le_max_left _ _

/-- The extension factor is computed, rather than supplied independently. -/
theorem context_scale_extension {original target : ℝ} (ho : 0 < original)
    (ht : original ≤ target) : contextScale original target = target / original := by
  apply max_eq_right
  exact (le_div_iff₀ ho).2 (by simpa using ht)

example : contextScale 4096 16384 = 16384 / (4096 : ℝ) :=
  context_scale_extension (by norm_num) (by norm_num)

/-- A configured context no longer than training leaves the scale at one. -/
theorem context_scale_unextended {original target : ℝ} (ho : 0 < original)
    (ht : target ≤ original) : contextScale original target = 1 := by
  apply max_eq_left
  exact (div_le_iff₀ ho).2 (by simpa using ht)

example : contextScale 4096 2048 = 1 :=
  context_scale_unextended (by norm_num) (by norm_num)

/-- Eqs. (10)–(13) with the paper's LLaMA defaults `alpha=1`, `beta=32`. -/
noncomputable def yarnFrequency (omega scale original : ℝ) : ℝ :=
  blendedFrequency omega scale (ramp 1 32 (rotations original omega))

/-- Every actual YaRN band lies between PI and the native positive frequency. -/
theorem yarn_frequency_bounds {omega scale : ℝ} (homega : 0 < omega)
    (hs : 1 ≤ scale) (original : ℝ) :
    omega / scale ≤ yarnFrequency omega scale original ∧
      yarnFrequency omega scale original ≤ omega :=
  blended_frequency_bounds homega hs (ramp_bounds _ _ _)

example : (1 : ℝ) / 4 ≤ yarnFrequency 1 4 4096 ∧ yarnFrequency 1 4 4096 ≤ 1 :=
  yarn_frequency_bounds (by norm_num) (by norm_num) _

/-- YaRN never creates zero or negative frequencies from a positive band. -/
theorem yarn_frequency_positive {omega scale : ℝ} (homega : 0 < omega)
    (hs : 1 ≤ scale) (original : ℝ) : 0 < yarnFrequency omega scale original :=
  blended_frequency_positive homega hs (ramp_bounds _ _ _)

example : 0 < yarnFrequency 1 4 4096 :=
  yarn_frequency_positive (by norm_num) (by norm_num) _

/-- At scale one every band is exactly native, regardless of its ramp. -/
theorem yarn_frequency_one (omega original : ℝ) : yarnFrequency omega 1 original = omega := by
  unfold yarnFrequency blendedFrequency
  ring

end Transformer.RoPE.YaRN
