module

public import Transformer.Grokking.GradientEvidence.ForecastTransport

/-!
# Anchored regularized normalization bounds

Source: `review/flat_spectrum.md`, *Feature forecasts*, FF2. The native RMS
map is `sqrt(width) * softNormalize (width*epsilon)`. The bound is anchored
at the known nominal input, avoiding a global inverse-sqrt-epsilon factor.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- FF2: regularized Euclidean length; also the source radial gate input. -/
def softNorm (eps : ℝ) (x : E) : ℝ := Real.sqrt (‖x‖ ^ 2 + eps)

/-- FF2: regularized normalization, before the fixed sqrt-width scale. -/
def softNormalize (eps : ℝ) (x : E) : E := (softNorm eps x)⁻¹ • x

omit [NormedSpace ℝ E] in
/-- Positive epsilon makes every nominal denominator positive, FF2. -/
theorem softNorm_pos (eps : ℝ) (heps : 0 < eps) (x : E) : 0 < softNorm eps x := by
  unfold softNorm
  exact Real.sqrt_pos.mpr (by positivity)

example := softNorm_pos 1 (by norm_num) (2 : ℝ)

omit [NormedSpace ℝ E] in
/-- FF2: regularization dominates the unregularized length. -/
theorem norm_le_softNorm (eps : ℝ) (heps : 0 ≤ eps) (x : E) : ‖x‖ ≤ softNorm eps x := by
  unfold softNorm
  exact (Real.le_sqrt (norm_nonneg _) (by positivity)).mpr (by linarith)

example := norm_le_softNorm 1 (by norm_num) (2 : ℝ)

omit [NormedSpace ℝ E] in
/-- FF2: regularized length is 1-Lipschitz, including near zero inputs. -/
theorem softNorm_error (eps : ℝ) (heps : 0 < eps) (x y : E) :
    |softNorm eps x - softNorm eps y| ≤ ‖x - y‖ := by
  have hx := softNorm_pos eps heps x
  have hy := softNorm_pos eps heps y
  have hnx := norm_le_softNorm eps heps.le x
  have hny := norm_le_softNorm eps heps.le y
  have hsx : (softNorm eps x) ^ 2 = ‖x‖ ^ 2 + eps := Real.sq_sqrt (by positivity)
  have hsy : (softNorm eps y) ^ 2 = ‖y‖ ^ 2 + eps := Real.sq_sqrt (by positivity)
  have he : (softNorm eps x - softNorm eps y) * (softNorm eps x + softNorm eps y) =
      (‖x‖ - ‖y‖) * (‖x‖ + ‖y‖) := by nlinarith
  have ha := congrArg abs he
  rw [abs_mul, abs_mul, abs_of_pos (by linarith : 0 < softNorm eps x + softNorm eps y),
    abs_of_nonneg (by positivity : 0 ≤ ‖x‖ + ‖y‖)] at ha
  have hb := mul_le_mul_of_nonneg_right (abs_norm_sub_norm_le x y)
    (show 0 ≤ ‖x‖ + ‖y‖ by positivity)
  have hc := mul_le_mul_of_nonneg_left (show ‖x‖ + ‖y‖ ≤ softNorm eps x + softNorm eps y by linarith)
    (norm_nonneg (x - y))
  nlinarith

example := softNorm_error 1 (by norm_num) (2 : ℝ) 3

omit [NormedSpace ℝ E] in
/-- FF2: the radial gate transports its incoming ball with unit sensitivity
and the known nominal directional defect. -/
theorem softNorm_forecast (eps : ℝ) (heps : 0 < eps)
    (a v y : E) (u r : ℝ) (hy : ‖y - (a + v)‖ ≤ r) :
    |softNorm eps y - (softNorm eps a + u)| ≤
      r + |softNorm eps (a + v) - (softNorm eps a + u)| := by
  have h := forecast_transport (softNorm eps) a v y u 1 r (by norm_num) hy
    (by intro z; simpa using softNorm_error eps heps z (a + v))
  simpa only [Real.norm_eq_abs, one_mul] using h

example := softNorm_forecast 1 (by norm_num) (2 : ℝ) (-1) 2 (-1) 1 (by norm_num)

/-- FF2: normalization sensitivity anchored at the known point x.
Unlike the global epsilon bound, its denominator uses x's actual length. -/
theorem softNormalize_error (eps : ℝ) (heps : 0 < eps) (x y : E) :
    ‖softNormalize eps y - softNormalize eps x‖ ≤ 2 / softNorm eps x * ‖y - x‖ := by
  have hx := softNorm_pos eps heps x
  have hy := softNorm_pos eps heps y
  have hn := norm_le_softNorm eps heps.le y
  have hd := softNorm_error eps heps y x
  have hi : |(softNorm eps y)⁻¹ - (softNorm eps x)⁻¹| =
      |softNorm eps y - softNorm eps x| / (softNorm eps x * softNorm eps y) := by
    rw [inv_sub_inv hy.ne' hx.ne']
    rw [abs_div, abs_of_pos (mul_pos hy hx), abs_sub_comm]
    congr 1
    ring
  have he : softNormalize eps y - softNormalize eps x =
      (softNorm eps x)⁻¹ • (y - x) +
        ((softNorm eps y)⁻¹ - (softNorm eps x)⁻¹) • y := by
    unfold softNormalize
    simp only [smul_sub, sub_smul]
    abel
  have hb : |(softNorm eps y)⁻¹ - (softNorm eps x)⁻¹| * ‖y‖ ≤
      ‖y - x‖ / softNorm eps x := by
    rw [hi, div_mul_eq_mul_div, div_le_div_iff₀ (mul_pos hx hy) hx]
    have h := mul_le_mul hd hn (norm_nonneg _) (norm_nonneg _)
    nlinarith [mul_le_mul_of_nonneg_left h hx.le]
  rw [he]
  calc
    _ ≤ ‖(softNorm eps x)⁻¹ • (y - x)‖ +
      ‖((softNorm eps y)⁻¹ - (softNorm eps x)⁻¹) • y‖ := norm_add_le _ _
    _ = ‖y - x‖ / softNorm eps x +
      |(softNorm eps y)⁻¹ - (softNorm eps x)⁻¹| * ‖y‖ := by
        rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs,
          abs_of_pos (inv_pos.mpr hx)]
        ring
    _ ≤ ‖y - x‖ / softNorm eps x + ‖y - x‖ / softNorm eps x := add_le_add le_rfl hb
    _ = _ := by ring

example := softNormalize_error 1 (by norm_num) (2 : ℝ) 3

/-- FF2: the fixed sqrt-width scale gives the source RMS sensitivity.
With eps=width*epsilon and scale=sqrt(width), this is exactly 2/s(x). -/
theorem scaled_softNormalize_error (scale eps : ℝ) (hs : 0 ≤ scale) (heps : 0 < eps)
    (x y : E) : ‖scale • softNormalize eps y - scale • softNormalize eps x‖ ≤
      (2 * scale / softNorm eps x) * ‖y - x‖ := by
  rw [← smul_sub, norm_smul, Real.norm_eq_abs, abs_of_nonneg hs]
  have h := mul_le_mul_of_nonneg_left (softNormalize_error eps heps x y) hs
  convert h using 1
  ring

example := scaled_softNormalize_error 2 1 (by norm_num) (by norm_num) (2 : ℝ) 3

/-- FF2: a computable nominal defect transports an incoming normalization
ball. The tangent output u can be the source RMS directional derivative. -/
theorem softNormalize_forecast (eps : ℝ) (heps : 0 < eps)
    (a v y u : E) (r : ℝ) (hy : ‖y - (a + v)‖ ≤ r) :
    ‖softNormalize eps y - (softNormalize eps a + u)‖ ≤
      2 / softNorm eps (a + v) * r +
        ‖softNormalize eps (a + v) - (softNormalize eps a + u)‖ := by
  exact forecast_transport (softNormalize eps) a v y u _ r
    (by have := softNorm_pos eps heps (a + v); positivity) hy
    (fun z => softNormalize_error eps heps (a + v) z)

example := softNormalize_forecast 1 (by norm_num) (2 : ℝ) (-1) 2 (-1) 1 (by norm_num)

end Transformer.Grokking.GradientEvidence
