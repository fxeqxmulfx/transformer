module

public import Transformer.RoPE.LeRoPE.Section7_Extrapolation
public import Transformer.RoPE.YaRN.Section3_Ramp

/-!
# LeRoPE, Appendix D.2: NTK-by-parts interpolates the dominant band fully

Karypis et al., arXiv:2607.10134v1, Appendix D.2, and §7 (dominant-band interpolation).
NTK-by-parts multiplies the frequency `φ_m` of band `m` by `M_m = s + w_m (1 − s)`, with the
per-document factor `s = min(1, L_train / L_doc)`, the rotation count `r_m = L_train φ_m / 2π`
and `w_m = clip((r_m − r_lo)/(r_hi − r_lo), 0, 1)`, `r_lo = 1`, `r_hi = 32`; in LeRoPE
`φ_m = θ̂_m`.

* `ntkMultiplier`, `ntkMultiplier_mul_eq_yarnFrequency`: this is YaRN's NTK-by-parts,
  `Transformer.RoPE.YaRN.yarnFrequency` (arXiv:2309.00071v3, Eqs. (10)–(13)), at the scale
  `max(1, L_doc / L_train)`;
* `ntkMultiplier_of_rotations_le_one`: "LeRoPE's dominant band, slowed to ∼0.5 rotations per
  window, is always fully interpolated": a band of at most one rotation gets `M_m = s` at every
  document length. A band `≤ 0` throughout its window, not zero, turns at most half a rotation
  (`rotations_le_half_of_nonpos`, from §7's `mul_le_pi_of_nonpos`);
* `score_interpolated_nonpos`, `score_ntkMultiplier_nonpos`: so interpolated, the band scores at
  the distance `d ≤ L_doc` its trained value at `s d ≤ L_train`. A band `≤ 0` throughout its
  window stays `≤ 0` at every distance of the document, where unmodified it turns positive
  (`exists_pos_score_beyond_window`): the effect §7 measures as dominant-band interpolation
  (Table 11). Every distance `d ≠ 0` moves, though, the window's included.
-/

@[expose] public section

namespace Transformer.RoPE.LeRoPE

open Set Transformer.RoPE.YaRN

/-- Appendix D.2: the multiplier `M_m = s + w_m (1 − s)` of NTK-by-parts for a band of frequency
`φ` trained on `L` tokens and run on a document of `L_doc`, with `s = min(1, L / L_doc)`,
`r_lo = 1` and `r_hi = 32`. -/
noncomputable def ntkMultiplier (L Ldoc φ : ℝ) : ℝ :=
  min 1 (L / Ldoc) + ramp 1 32 (rotations L φ) * (1 - min 1 (L / Ldoc))

/-- Appendix D.2: **LeRoPE's NTK-by-parts is YaRN's** at the scale `max(1, L_doc / L)`, since
`min(1, L / L_doc) = 1 / max(1, L_doc / L)`. -/
theorem ntkMultiplier_mul_eq_yarnFrequency {L Ldoc : ℝ} (hL : 0 < L) (hLdoc : 0 < Ldoc)
    (φ : ℝ) : ntkMultiplier L Ldoc φ * φ = yarnFrequency φ (contextScale L Ldoc) L := by
  have hs : min 1 (L / Ldoc) = 1 / contextScale L Ldoc := by
    unfold contextScale
    rcases le_total Ldoc L with h | h
    · rw [min_eq_left ((one_le_div hLdoc).2 h), max_eq_left ((div_le_one hL).2 h), div_one]
    · rw [min_eq_right ((div_le_one hLdoc).2 h), max_eq_right ((one_le_div hL).2 h),
        one_div_div]
  rw [ntkMultiplier, yarnFrequency, blendedFrequency, hs]
  ring

/-- The hypotheses of `ntkMultiplier_mul_eq_yarnFrequency` hold for `L = 2048`, a document of
`8192` tokens and Table 10's `θ̂ = 0.001507`. -/
example := ntkMultiplier_mul_eq_yarnFrequency (L := 2048) (Ldoc := 8192) (by norm_num)
  (by norm_num) 0.001507

/-- Appendix D.2: **a band of at most one rotation is always fully interpolated**: `M = s` at
every document length. -/
theorem ntkMultiplier_of_rotations_le_one {L φ : ℝ} (h : rotations L φ ≤ 1) (Ldoc : ℝ) :
    ntkMultiplier L Ldoc φ = min 1 (L / Ldoc) := by
  rw [ntkMultiplier, ramp_low (by norm_num) h, zero_mul, add_zero]

/-- Table 10's 2.52B band turns `0.001507 · 2048 / 2π < 1` rotation over its window. -/
example := ntkMultiplier_of_rotations_le_one (L := 2048) (φ := 0.001507)
  (by
    rw [rotations, div_le_one (by positivity)]
    nlinarith [Real.pi_gt_d2])
  8192

/-- Appendix D.2 with §7: **a band `≤ 0` throughout its window, not zero, turns at most half a
rotation over it** ("slowed to ∼0.5 rotations per window"), since `θ L ≤ π`. -/
theorem rotations_le_half_of_nonpos {a b θ L : ℝ} (hθ : 0 < θ) (hab : a ≠ 0 ∨ b ≠ 0)
    (hneg : ∀ d ∈ Icc 0 L, score a b θ d ≤ 0) : rotations L θ ≤ 1 / 2 := by
  have h := mul_le_pi_of_nonpos hθ hab hneg
  rw [rotations, div_le_iff₀ (by positivity)]
  linarith [mul_comm L θ]

/-- The hypotheses of `rotations_le_half_of_nonpos` hold for the band of the examples. -/
example := rotations_le_half_of_nonpos (a := 0) (b := -1) (θ := 0.001507) (L := 2048)
  (by norm_num) (Or.inr (by norm_num)) score_table10_nonpos

/-- §7, dominant-band interpolation: **the fully interpolated band keeps its trained values**.
At the document length `L_doc > 0` and the distance `0 ≤ d ≤ L_doc` the band of frequency
`s θ`, `s = min(1, L / L_doc)`, scores its value at `s d ∈ [0, L]`; a band `≤ 0` throughout its
window stays `≤ 0`. -/
theorem score_interpolated_nonpos {a b θ L Ldoc : ℝ} (hL : 0 < L) (hLdoc : 0 < Ldoc)
    (hneg : ∀ d ∈ Icc 0 L, score a b θ d ≤ 0) {d : ℝ} (hd : d ∈ Icc 0 Ldoc) :
    score a b (min 1 (L / Ldoc) * θ) d ≤ 0 := by
  have hs0 : 0 ≤ min 1 (L / Ldoc) := le_min zero_le_one (by positivity)
  have hsd : min 1 (L / Ldoc) * d ≤ L := by
    rcases le_total Ldoc L with hle | hle
    · calc min 1 (L / Ldoc) * d ≤ 1 * d := mul_le_mul_of_nonneg_right (min_le_left _ _) hd.1
        _ ≤ L := by linarith [hd.2]
    · calc min 1 (L / Ldoc) * d ≤ L / Ldoc * d :=
            mul_le_mul_of_nonneg_right (min_le_right _ _) hd.1
        _ ≤ L / Ldoc * Ldoc := mul_le_mul_of_nonneg_left hd.2 (by positivity)
        _ = L := div_mul_cancel₀ L hLdoc.ne'
  have h := hneg _ ⟨mul_nonneg hs0 hd.1, hsd⟩
  have he : min 1 (L / Ldoc) * θ * d = θ * (min 1 (L / Ldoc) * d) := by ring
  simpa only [score, he] using h

/-- The hypotheses of `score_interpolated_nonpos` hold for the band of the examples on a
document of `8192` tokens. -/
example := score_interpolated_nonpos (a := 0) (b := -1) (θ := 0.001507) (L := 2048)
  (Ldoc := 8192) (by norm_num) (by norm_num) score_table10_nonpos (d := 8000)
  ⟨by norm_num, by norm_num⟩

/-- Appendix D.2: **NTK-by-parts keeps the dominant band `≤ 0`** at every document length: a band
`≤ 0` throughout its window, not zero, is fully interpolated and stays `≤ 0` at every distance
of the document. -/
theorem score_ntkMultiplier_nonpos {a b θ L Ldoc : ℝ} (hθ : 0 < θ) (hL : 0 < L)
    (hLdoc : 0 < Ldoc) (hab : a ≠ 0 ∨ b ≠ 0) (hneg : ∀ d ∈ Icc 0 L, score a b θ d ≤ 0) {d : ℝ}
    (hd : d ∈ Icc 0 Ldoc) : score a b (ntkMultiplier L Ldoc θ * θ) d ≤ 0 := by
  rw [ntkMultiplier_of_rotations_le_one
    ((rotations_le_half_of_nonpos hθ hab hneg).trans (by norm_num)) Ldoc]
  exact score_interpolated_nonpos hL hLdoc hneg hd

/-- The hypotheses of `score_ntkMultiplier_nonpos` hold for the band of the examples on a
document of `8192` tokens. -/
example := score_ntkMultiplier_nonpos (a := 0) (b := -1) (θ := 0.001507) (L := 2048)
  (Ldoc := 8192) (by norm_num) (by norm_num) (by norm_num) (Or.inr (by norm_num))
  score_table10_nonpos (d := 8000) ⟨by norm_num, by norm_num⟩

end Transformer.RoPE.LeRoPE
