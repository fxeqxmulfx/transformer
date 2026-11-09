module

public import Transformer.Grokking.GradientEvidence.ForecastGram

/-!
# Signed class contrasts before final normalization

Source: `review/flat_spectrum.md`, *Forecast tightening*, FT3. Final RMS has
a common positive denominator. Preserve signed numerator intervals, bound
that denominator, and intersect with the independent final-feature interval.
These statements do not use the unknown future feature to choose a bound.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

/-- FT3: lower endpoint of a signed numerator divided by a positive interval. -/
def signedQuotientLower (a l u : ℝ) : ℝ := if 0 ≤ a then a / u else a / l

/-- FT3: upper endpoint; a negative numerator reverses the denominator choice. -/
def signedQuotientUpper (b l u : ℝ) : ℝ := if 0 ≤ b then b / l else b / u

/-- FT3: a sign-aware positive-denominator interval encloses every quotient. -/
theorem signed_quotient_interval (a b z l u d : ℝ) (hl : 0 < l)
    (hd : l ≤ d) (hu : d ≤ u) (ha : a ≤ z) (hb : z ≤ b) :
    signedQuotientLower a l u ≤ z / d ∧ z / d ≤ signedQuotientUpper b l u := by
  have hd0 : 0 < d := hl.trans_le hd
  have hu0 : 0 < u := hd0.trans_le hu
  constructor
  · apply (le_div_iff₀ hd0).mpr
    unfold signedQuotientLower
    split_ifs with h
    · have hq := div_nonneg h hu0.le
      have he := div_mul_cancel₀ a hu0.ne'
      nlinarith
    · have hq := div_nonpos_of_nonpos_of_nonneg (le_of_not_ge h) hl.le
      have he := div_mul_cancel₀ a hl.ne'
      nlinarith
  · apply (div_le_iff₀ hd0).mpr
    unfold signedQuotientUpper
    split_ifs with h
    · have hq := div_nonneg h hl.le
      have he := div_mul_cancel₀ b hl.ne'
      nlinarith
    · have hq := div_nonpos_of_nonpos_of_nonneg (le_of_not_ge h) hu0.le
      have he := div_mul_cancel₀ b hu0.ne'
      nlinarith

example := signed_quotient_interval (-3) (-1) (-2) 1 3 2
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

variable {H : Type*} [NormedAddCommGroup H]

/-- FT3: reverse triangle inequality gives a denominator interval using
only the nominal vector and its error radius. -/
theorem softNorm_ball_interval (eps : ℝ) (x y : H) (r : ℝ)
    (hy : ‖y - x‖ ≤ r) :
    Real.sqrt ((max 0 (‖x‖ - r)) ^ 2 + eps) ≤ softNorm eps y ∧
      softNorm eps y ≤ Real.sqrt ((‖x‖ + r) ^ 2 + eps) := by
  have hr := (norm_nonneg (y - x)).trans hy
  obtain ⟨hl, hu⟩ := abs_le.mp ((abs_norm_sub_norm_le y x).trans hy)
  have hm : max 0 (‖x‖ - r) ≤ ‖y‖ := max_le (norm_nonneg _) (by linarith)
  have hm0 : 0 ≤ max 0 (‖x‖ - r) := le_max_left _ _
  unfold softNorm
  constructor <;> apply Real.sqrt_le_sqrt <;> nlinarith [norm_nonneg x, norm_nonneg y]

example := softNorm_ball_interval 1 (2 : ℝ) 3 1 (by norm_num)

/-- FT3: positivity of the lower denominator survives a radius containing
the origin, because regularization is not discarded. -/
theorem softNorm_ball_lower_pos (eps : ℝ) (heps : 0 < eps) (x : H) (r : ℝ) :
    0 < Real.sqrt ((max 0 (‖x‖ - r)) ^ 2 + eps) := by
  apply Real.sqrt_pos.mpr
  positivity

example := softNorm_ball_lower_pos 1 (by norm_num) (2 : ℝ) 3

variable [InnerProductSpace ℝ H]

/-- FT3: each class-contrast direction gets its own numerator interval.
The native source multiplies this normalized score by sqrt(width). -/
theorem directional_normalized_interval (eps : ℝ) (heps : 0 < eps)
    (w x y : H) (r : ℝ) (hy : ‖y - x‖ ≤ r) :
    signedQuotientLower (inner ℝ w x - ‖w‖ * r)
        (Real.sqrt ((max 0 (‖x‖ - r)) ^ 2 + eps))
        (Real.sqrt ((‖x‖ + r) ^ 2 + eps)) ≤ inner ℝ w (softNormalize eps y) ∧
    inner ℝ w (softNormalize eps y) ≤
      signedQuotientUpper (inner ℝ w x + ‖w‖ * r)
        (Real.sqrt ((max 0 (‖x‖ - r)) ^ 2 + eps))
        (Real.sqrt ((‖x‖ + r) ^ 2 + eps)) := by
  have hn := (abs_real_inner_le_norm w (y - x)).trans
    (mul_le_mul_of_nonneg_left hy (norm_nonneg w))
  rw [inner_sub_right] at hn
  obtain ⟨ha, hb⟩ := abs_le.mp hn
  obtain ⟨hl, hu⟩ := softNorm_ball_interval eps x y r hy
  have h := signed_quotient_interval (inner ℝ w x - ‖w‖ * r)
    (inner ℝ w x + ‖w‖ * r) (inner ℝ w y) _ _ (softNorm eps y)
    (softNorm_ball_lower_pos eps heps x r) hl hu (by linarith) (by linarith)
  simpa only [softNormalize, inner_smul_right, div_eq_mul_inv, mul_comm] using h

example := directional_normalized_interval 1 (by norm_num) (3 : ℝ) 2 3 1 (by norm_num)

/-- FT3: the fixed source sqrt-width multiplier and any independently
bounded execution error preserve the signed endpoints. -/
theorem scaled_interval_with_error (a b z scale e δ : ℝ) (hs : 0 ≤ scale)
    (ha : a ≤ z) (hb : z ≤ b) (he : |e| ≤ δ) :
    scale * a - δ ≤ scale * z + e ∧ scale * z + e ≤ scale * b + δ := by
  obtain ⟨hl, hu⟩ := abs_le.mp he
  constructor <;> nlinarith

example := scaled_interval_with_error (-2) 3 1 2 (-1) 1
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- FT3: intersect two valid intervals without selecting a bound based
on the future outcome. Both independent obligations remain explicit. -/
theorem forecast_interval_intersection (a b c d z : ℝ)
    (ha : a ≤ z) (hb : z ≤ b) (hc : c ≤ z) (hd : z ≤ d) :
    max a c ≤ z ∧ z ≤ min b d := by
  exact ⟨max_le ha hc, le_min hb hd⟩

example := forecast_interval_intersection (-2) 3 0 4 1
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- FT3: a strictly negative upper competitor gap certifies the actual
class ordering; zero or inconclusive intervals never become certificates. -/
theorem directional_order (target competitor upper : ℝ)
    (h : competitor - target ≤ upper) (hu : upper < 0) : competitor < target := by
  linarith

example := directional_order 3 1 (-1) (by norm_num) (by norm_num)

end Transformer.Grokking.GradientEvidence
