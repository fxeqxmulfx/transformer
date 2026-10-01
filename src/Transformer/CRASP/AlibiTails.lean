/-
# Rounded ALiBi coefficients stabilize beyond a finite distance

arXiv:2506.16055v3, Appendix F, `lem:alibi_window` and
`thm:rtfr_to_TLCly`. For a positive slope the weight becomes zero, but
negative numerators become minus one mantissa unit under downward rounding.
For a negative slope, each nonzero contribution saturates instead.
-/

import Transformer.CRASP.FixedTail
import Transformer.CRASP.FixedSign

namespace Transformer.CRASP

/-- A positive linear bias makes every scaled exponential smaller than one
mantissa unit beyond a finite distance. Source: Appendix F, ALiBi window. -/
theorem alibi_small_tail (s : ℕ) (x v a : ℝ) (ha : 0 < a) :
    ∃ Δ : ℕ, ∀ δ : ℕ, Δ ≤ δ → |Real.exp (x - a * δ) * v * 2 ^ s| < 1 := by
  let B : ℝ := (|v| + 1) * 2 ^ s
  have hB : 0 < B := by dsimp [B]; positivity
  obtain ⟨Δ, hΔ⟩ := exists_nat_gt ((x - Real.log B⁻¹) / a)
  refine ⟨Δ, fun δ hδ => ?_⟩
  have hd : (Δ : ℝ) ≤ δ := by exact_mod_cast hδ
  have hb : x - Real.log B⁻¹ < a * δ := by
    have h := (div_lt_iff₀ ha).mp (hΔ.trans_le hd)
    nlinarith
  have he : Real.exp (x - a * δ) < B⁻¹ := by
    rw [← Real.lt_log_iff_exp_lt (inv_pos.mpr hB)]
    linarith
  have hh : Real.exp (x - a * δ) * B < 1 := by
    have h := mul_lt_mul_of_pos_right he hB
    simpa only [inv_mul_cancel₀ (ne_of_gt hB)] using h
  have hs : (0 : ℝ) < 2 ^ s := by positivity
  rw [abs_mul, abs_mul, abs_of_pos (Real.exp_pos _), abs_of_pos hs]
  apply lt_of_le_of_lt _ hh
  dsimp [B]
  calc
    Real.exp (x - a * δ) * |v| * 2 ^ s ≤
        Real.exp (x - a * δ) * (|v| + 1) * 2 ^ s := by
      gcongr
      linarith
    _ = Real.exp (x - a * δ) * ((|v| + 1) * 2 ^ s) := by ring

/-- A negative slope eventually makes every nonzero magnitude saturate.
Source: Appendix F, Equation `eq:alibi`, with the slope extended to negatives. -/
theorem alibi_large_tail (p s : ℕ) (x v a : ℝ) (ha : a < 0) (hv : v ≠ 0) :
    ∃ Δ : ℕ, ∀ δ : ℕ, Δ ≤ δ → (2 : ℝ) ^ (p - 1) ≤
      Real.exp (x - a * δ) * |v| * 2 ^ s := by
  let B : ℝ := (2 : ℝ) ^ (p - 1) / (|v| * 2 ^ s)
  have hvs : (0 : ℝ) < |v| * 2 ^ s := by
    exact mul_pos (abs_pos.mpr hv) (by positivity)
  obtain ⟨Δ, hΔ⟩ := exists_nat_gt ((B - x) / (-a))
  refine ⟨Δ, fun δ hδ => ?_⟩
  have hd : (Δ : ℝ) ≤ δ := by exact_mod_cast hδ
  have hx : B < x - a * δ := by
    have h := (div_lt_iff₀ (neg_pos.mpr ha)).mp (hΔ.trans_le hd)
    linarith
  have he : B ≤ Real.exp (x - a * δ) :=
    hx.le.trans (le_trans (by linarith) (Real.add_one_le_exp _))
  have hh := mul_le_mul_of_nonneg_right he hvs.le
  dsimp [B] at hh
  rw [div_mul_cancel₀ _ (ne_of_gt hvs)] at hh
  simpa only [mul_assoc] using hh

/-- Every rounded numerator or denominator coefficient of ALiBi eventually
becomes constant, including zero and negative slopes.

The proof preserves rounding-before-summing from Appendix B.1. In particular
it does not discard distant negative numerator contributions when a positive
slope rounds their corresponding denominator weights to zero.

Source: arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`. -/
theorem alibi_coefficient_stabilizes (p s : ℕ) (x v a : ℝ) :
    ∃ (Δ : ℕ) (y : Fx p s), ∀ δ : ℕ, Δ ≤ δ →
      Fx.round p s (Real.exp (x - a * δ) * v) = y := by
  rcases lt_trichotomy a 0 with ha | rfl | ha
  · by_cases hv : v = 0
    · subst v
      exact ⟨0, 0, fun _ _ => by simp⟩
    obtain ⟨Δ, hΔ⟩ := alibi_large_tail p s x v a ha hv
    refine ⟨Δ, if 0 < v then Fx.top p s else Fx.bottom p s, fun δ hδ => ?_⟩
    have hh := hΔ δ hδ
    by_cases hpv : 0 < v
    · rw [ite_eq_left hpv]
      apply Fx.round_top
      rw [abs_of_pos hpv] at hh
      have hcast : (((2 ^ (p - 1) - 1 : ℤ) : ℝ)) = (2 : ℝ) ^ (p - 1) - 1 := by
        push_cast; rfl
      rw [hcast]
      linarith
    · rw [ite_eq_right hpv]
      apply Fx.round_bottom
      have hnv : v < 0 := lt_of_le_of_ne (le_of_not_gt hpv) hv
      rw [abs_of_neg hnv] at hh
      have hcast : (((-2 ^ (p - 1) : ℤ) : ℝ)) = -(2 : ℝ) ^ (p - 1) := by
        push_cast; rfl
      rw [hcast]
      nlinarith
  · exact ⟨0, Fx.round p s (Real.exp x * v), fun _ _ => by simp⟩
  · obtain ⟨Δ, hΔ⟩ := alibi_small_tail s x v a ha
    refine ⟨Δ, if v < 0 then Fx.negUnit p s else 0, fun δ hδ => ?_⟩
    rw [Fx.round_small _ (hΔ δ hδ)]
    have hneg : Real.exp (x - a * δ) * v < 0 ↔ v < 0 := by
      constructor
      · intro h
        nlinarith [Real.exp_pos (x - a * δ)]
      · intro h
        exact mul_neg_of_pos_of_neg (Real.exp_pos _) h
    simp only [hneg]

/-- The slope and nonzero-value hypotheses have positive and negative witnesses.
Source: arXiv:2506.16055v3, Appendix F, Equation `eq:alibi`. -/
example : (0 : ℝ) < 1 ∧ (-1 : ℝ) < 0 ∧ (1 : ℝ) ≠ 0 := by norm_num

end Transformer.CRASP
