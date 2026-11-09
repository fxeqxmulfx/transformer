module

public import Transformer.Grokking.GradientEvidence.ForecastOutcomes

/-!
# Local secant bounds for forecast normalization

Source: `review/flat_spectrum.md`, *Forecast tightening*, FT1, and the frozen
`experiments/forecast_tightening/PROTOCOL.md`. Keep the same nominal tangent
and nonlinear defect. Only the propagation coefficient changes. The identity
below retains the radial cancellation discarded by the earlier triangle bound.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]

/-- FT1: exact regularized secant identity; the radial correction is nonnegative.
The factor `eps` is retained, including near-zero inputs. -/
theorem softNormalize_secant_identity (eps : ℝ) (heps : 0 < eps) (x y : H) :
    ‖softNormalize eps y - softNormalize eps x‖ ^ 2 *
        (softNorm eps x * softNorm eps y) ^ 2 +
      (softNorm eps x - softNorm eps y) ^ 2 *
        (softNorm eps x * softNorm eps y + eps) =
      ‖y - x‖ ^ 2 * (softNorm eps x * softNorm eps y) := by
  have hx := softNorm_pos eps heps x
  have hy := softNorm_pos eps heps y
  have hnx : ‖x‖ ^ 2 = softNorm eps x ^ 2 - eps := by
    have h : softNorm eps x ^ 2 = ‖x‖ ^ 2 + eps := Real.sq_sqrt (by positivity)
    linarith
  have hny : ‖y‖ ^ 2 = softNorm eps y ^ 2 - eps := by
    have h : softNorm eps y ^ 2 = ‖y‖ ^ 2 + eps := Real.sq_sqrt (by positivity)
    linarith
  have hv : (softNorm eps x * softNorm eps y) •
      (softNormalize eps y - softNormalize eps x) =
      softNorm eps x • y - softNorm eps y • x := by
    simp [softNormalize, smul_sub, smul_smul, hx.ne', hy.ne', mul_comm]
  have hn := congrArg (fun z : H => ‖z‖ ^ 2) hv
  simp only [norm_smul, Real.norm_eq_abs, abs_of_pos hx, abs_of_pos hy,
    abs_of_pos (mul_pos hx hy), norm_sub_sq_real, inner_smul_left,
    inner_smul_right, starRingEnd_apply, star_trivial, mul_pow] at hn
  rw [hnx, hny] at hn
  simp only [norm_sub_sq_real, hnx, hny]
  nlinarith [hn]

example := softNormalize_secant_identity 1 (by norm_num) (2 : ℝ) (-1)

/-- FT1: the normalized squared distance is controlled by the product of
the two regularized lengths, rather than a factor-two triangle estimate. -/
theorem softNormalize_secant_sq (eps : ℝ) (heps : 0 < eps) (x y : H) :
    ‖softNormalize eps y - softNormalize eps x‖ ^ 2 *
        (softNorm eps x * softNorm eps y) ≤ ‖y - x‖ ^ 2 := by
  have hx := softNorm_pos eps heps x
  have hy := softNorm_pos eps heps y
  have h := softNormalize_secant_identity eps heps x y
  have hp := mul_pos hx hy
  apply (mul_le_mul_iff_right₀ hp).mp
  have hc := mul_nonneg (sq_nonneg (softNorm eps x - softNorm eps y))
    (show 0 ≤ softNorm eps x * softNorm eps y + eps by positivity)
  nlinarith

example := softNormalize_secant_sq 1 (by norm_num) (2 : ℝ) (-1)

/-- FT1: a ball smaller than the known nominal regularized length gives
a computable local secant coefficient. No future denominator is required. -/
theorem softNormalize_local_error (eps : ℝ) (heps : 0 < eps) (x y : H) (r : ℝ)
    (hy : ‖y - x‖ ≤ r) (hr : r < softNorm eps x) :
    ‖softNormalize eps y - softNormalize eps x‖ ≤
      r / Real.sqrt (softNorm eps x * (softNorm eps x - r)) := by
  have hx := softNorm_pos eps heps x
  have hr0 := (norm_nonneg (y - x)).trans hy
  have hp : 0 < softNorm eps x * (softNorm eps x - r) := mul_pos hx (sub_pos.mpr hr)
  have ht := Real.sqrt_pos.mpr hp
  have hs := Real.sq_sqrt hp.le
  have hd := (abs_le.mp (softNorm_error eps heps y x)).1
  have hl : softNorm eps x * (softNorm eps x - r) ≤
      softNorm eps x * softNorm eps y := by nlinarith
  have hb := softNormalize_secant_sq eps heps x y
  have hm := mul_le_mul_of_nonneg_left hl
    (sq_nonneg ‖softNormalize eps y - softNormalize eps x‖)
  have hdist : ‖y - x‖ ^ 2 ≤ r ^ 2 := by nlinarith [norm_nonneg (y - x)]
  apply (le_div_iff₀ ht).mpr
  nlinarith [sq_nonneg
    (‖softNormalize eps y - softNormalize eps x‖ *
      Real.sqrt (softNorm eps x * (softNorm eps x - r)) + r),
    norm_nonneg (softNormalize eps y - softNormalize eps x)]

example := softNormalize_local_error 1 (by norm_num) (2 : ℝ) 3 1
  (by norm_num) (by norm_num [softNorm, Real.lt_sqrt (by norm_num : (0 : ℝ) ≤ 1)])

/-- FT1: source RMS scaling preserves the local bound. The native code
intersects it with the old global bound and falls back when `r ≥ s`. -/
theorem scaled_softNormalize_local_error (scale eps : ℝ) (hs : 0 ≤ scale)
    (heps : 0 < eps) (x y : H) (r : ℝ) (hy : ‖y - x‖ ≤ r)
    (hr : r < softNorm eps x) :
    ‖scale • softNormalize eps y - scale • softNormalize eps x‖ ≤
      scale * r / Real.sqrt (softNorm eps x * (softNorm eps x - r)) := by
  rw [← smul_sub, norm_smul, Real.norm_eq_abs, abs_of_nonneg hs]
  have h := mul_le_mul_of_nonneg_left (softNormalize_local_error eps heps x y r hy hr) hs
  simpa only [mul_div_assoc] using h

example := scaled_softNormalize_local_error 2 1 (by norm_num) (by norm_num)
  (2 : ℝ) 3 1 (by norm_num) (by norm_num [softNorm, Real.lt_sqrt (by norm_num : (0 : ℝ) ≤ 1)])

/-- FT1: transport the unchanged tangent center with a sharper local
normalization term and the same nominal nonlinear defect. -/
theorem softNormalize_local_forecast (eps : ℝ) (heps : 0 < eps)
    (a v y u : H) (r : ℝ) (hy : ‖y - (a + v)‖ ≤ r)
    (hr : r < softNorm eps (a + v)) :
    ‖softNormalize eps y - (softNormalize eps a + u)‖ ≤
      r / Real.sqrt (softNorm eps (a + v) * (softNorm eps (a + v) - r)) +
        ‖softNormalize eps (a + v) - (softNormalize eps a + u)‖ := by
  exact (norm_sub_le_norm_sub_add_norm_sub _ _ _).trans
    (add_le_add (softNormalize_local_error eps heps (a + v) y r hy hr) le_rfl)

example := softNormalize_local_forecast 1 (by norm_num) (2 : ℝ) 1 4 1 1
  (by norm_num) (by norm_num [softNorm, Real.lt_sqrt (by norm_num : (0 : ℝ) ≤ 1)])

end Transformer.Grokking.GradientEvidence
