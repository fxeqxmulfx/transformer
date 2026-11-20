module

public import Transformer.RoPE.LeRoPE.Section3_LearnedFrequencies
public import Mathlib.Analysis.Real.Pi.Bounds

/-!
# LeRoPE, §7: the dominant band turns positive beyond the window

Karypis et al., arXiv:2607.10134v1, §7 (Naive Extrapolation), with §6 and Appendix D.1. The
dominant band (§6) is the one band whose contribution to the attention logits is large and nearly
purely positional; at the distance `d` it contributes `score a b θ̂ d = a cos(θ̂ d) + b sin(θ̂ d)`.
§7: "The dominant frequency band's logit contribution follows a negative half-cycle of a sinusoid
with period ≈ 2.2 L_train, so within the training window it remains negative or near zero. On
relative distances beyond those seen during training, the contribution becomes positive."

That the contribution is `≤ 0` throughout the window `[0, L]` is the observation; the theorems
take it as a hypothesis, of a band that is not zero, and derive the rest:

* `score_add_pi_div`: half a period on, every value of a band comes back negated;
* `eq_zero_of_score_eq_zero`: a band that vanishes on an interval is zero;
* `mul_le_pi_of_nonpos`: a band `≤ 0` throughout the window has `θ̂ L ≤ π`, a period of at least
  `2 L` and at most half a rotation per window: the window fits in one negative half-cycle only if
  the half-cycle is as long. LeRoPE's learned periods sit just above this bound;
* `exists_pos_score_beyond_window`: such a band is positive at some distance in `(L, L + π/θ̂]`,
  within half a period past the window: a sign it never had in training.

Appendix D.1, Table 10, lists six runs with `θ̂ L` from `2.714` to `3.086` and `λ / L` from
`2.036` to `2.315`, mean `2.138`: §6's "≈ 2.205 L_train" is not their mean. Only `λ ≥ 2 L` enters
here. The examples take the band `−sin(θ̂ d)` of the 2.52B run, `θ̂ = 0.001507` at `L = 2048`.
-/

@[expose] public section

namespace Transformer.RoPE.LeRoPE

open Set

/-- §7: **half a period on, a band comes back negated**: `θ (d + π/θ) = θ d + π`. -/
theorem score_add_pi_div (a b : ℝ) {θ : ℝ} (hθ : θ ≠ 0) (d : ℝ) :
    score a b θ (d + Real.pi / θ) = -score a b θ d := by
  have h : θ * (d + Real.pi / θ) = θ * d + Real.pi := by field_simp
  simp only [score, h, Real.cos_add_pi, Real.sin_add_pi]
  ring

/-- The hypothesis of `score_add_pi_div` holds for the band of Table 10's 2.52B run. -/
example := score_add_pi_div 0 (-1) (θ := 0.001507) (by norm_num) 1000

/-- **A band that vanishes on `[0, ε]`, `ε > 0`, is zero**: `a` is its value at `0`, and `b`
times a positive sine is its value at `min(ε, π/(2θ))`. -/
theorem eq_zero_of_score_eq_zero {a b θ ε : ℝ} (hθ : 0 < θ) (hε : 0 < ε)
    (h : ∀ d ∈ Icc 0 ε, score a b θ d = 0) : a = 0 ∧ b = 0 := by
  have ha : a = 0 := by simpa [score] using h 0 ⟨le_rfl, hε.le⟩
  refine ⟨ha, ?_⟩
  have hd0 : 0 < min ε (Real.pi / (2 * θ)) := lt_min hε (by positivity)
  have hθd : θ * min ε (Real.pi / (2 * θ)) < Real.pi := by
    calc θ * min ε (Real.pi / (2 * θ)) ≤ θ * (Real.pi / (2 * θ)) :=
          mul_le_mul_of_nonneg_left (min_le_right _ _) hθ.le
      _ = Real.pi / 2 := by field_simp
      _ < Real.pi := by linarith [Real.pi_pos]
  have hs := Real.sin_pos_of_pos_of_lt_pi (mul_pos hθ hd0) hθd
  have h0 := h _ ⟨hd0.le, min_le_left _ _⟩
  simp only [score, ha, zero_mul, zero_add] at h0
  exact (mul_eq_zero.mp h0).resolve_right hs.ne'

/-- The hypotheses of `eq_zero_of_score_eq_zero` hold for the zero band. -/
example := eq_zero_of_score_eq_zero (a := 0) (b := 0) (θ := 1) (ε := 1) one_pos one_pos
  fun _ _ => by simp [score]

/-- The band of the examples: `−sin(θ̂ d)` with Table 10's `θ̂ = 0.001507` (2.52B, band 17) is
`≤ 0` on the window `[0, 2048]`, since `θ̂ L = 3.086 < π`. -/
theorem score_table10_nonpos : ∀ d ∈ Icc (0 : ℝ) 2048, score 0 (-1) 0.001507 d ≤ 0 := by
  intro d hd
  have := Real.sin_nonneg_of_nonneg_of_le_pi (x := 0.001507 * d) (by nlinarith [hd.1])
    (by nlinarith [hd.2, Real.pi_gt_d2])
  simp only [score]
  linarith

/-- §7 and Appendix D.1: **a band `≤ 0` throughout the window has a period of at least twice
the window**, `θ L ≤ π`. Were `θ L > π`, the band would be `≤ 0` both on `[0, L − π/θ]` and half
a period on, where it is negated, so it would vanish on that interval and be zero. -/
theorem mul_le_pi_of_nonpos {a b θ L : ℝ} (hθ : 0 < θ) (hab : a ≠ 0 ∨ b ≠ 0)
    (hneg : ∀ d ∈ Icc 0 L, score a b θ d ≤ 0) : θ * L ≤ Real.pi := by
  by_contra hlt
  push Not at hlt
  have hπθ : 0 < Real.pi / θ := by positivity
  have hε : 0 < L - Real.pi / θ := by
    rw [sub_pos, div_lt_iff₀ hθ]
    linarith [mul_comm θ L]
  have h0 : ∀ d ∈ Icc 0 (L - Real.pi / θ), score a b θ d = 0 := by
    intro d hd
    have h1 := hneg d ⟨hd.1, by linarith [hd.2]⟩
    have h2 := hneg (d + Real.pi / θ) ⟨by linarith [hd.1], by linarith [hd.2]⟩
    rw [score_add_pi_div a b hθ.ne'] at h2
    linarith
  obtain ⟨ha, hb⟩ := eq_zero_of_score_eq_zero hθ hε h0
  exact hab.elim (· ha) (· hb)

/-- The hypotheses of `mul_le_pi_of_nonpos` hold for the band of the examples. -/
example := mul_le_pi_of_nonpos (a := 0) (b := -1) (θ := 0.001507) (L := 2048) (by norm_num)
  (Or.inr (by norm_num)) score_table10_nonpos

/-- §7: **beyond the window the band turns positive**. A band `≤ 0` throughout the window
`[0, L]`, `L > 0`, and not zero is negative somewhere in it, and half a period on, at a distance in
`(L, L + π/θ]`, positive. -/
theorem exists_pos_score_beyond_window {a b θ L : ℝ} (hθ : 0 < θ) (hL : 0 < L)
    (hab : a ≠ 0 ∨ b ≠ 0) (hneg : ∀ d ∈ Icc 0 L, score a b θ d ≤ 0) :
    ∃ d ∈ Ioc L (L + Real.pi / θ), 0 < score a b θ d := by
  obtain ⟨d, hd, hlt⟩ : ∃ d ∈ Icc 0 L, score a b θ d < 0 := by
    by_contra h
    push Not at h
    obtain ⟨ha, hb⟩ :=
      eq_zero_of_score_eq_zero hθ hL fun d hd => le_antisymm (hneg d hd) (h d hd)
    exact hab.elim (· ha) (· hb)
  have hpos : 0 < score a b θ (d + Real.pi / θ) := by
    rw [score_add_pi_div a b hθ.ne']
    linarith
  have hπθ : 0 < Real.pi / θ := by positivity
  refine ⟨d + Real.pi / θ, ⟨?_, by linarith [hd.2]⟩, hpos⟩
  by_contra hle
  push Not at hle
  linarith [hneg _ ⟨by linarith [hd.1], hle⟩]

/-- The hypotheses of `exists_pos_score_beyond_window` hold for the band of the examples. -/
example := exists_pos_score_beyond_window (a := 0) (b := -1) (θ := 0.001507) (L := 2048)
  (by norm_num) (by norm_num) (Or.inr (by norm_num)) score_table10_nonpos

end Transformer.RoPE.LeRoPE
