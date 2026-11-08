module

public import Transformer.Grokking.GradientEvidence.AdaptiveMargin
public import Mathlib.Analysis.Calculus.MeanValue
public import Mathlib.Analysis.SpecialFunctions.Sigmoid

/-!
# Propagating a forecast without observing future features

Source: `review/flat_spectrum.md`, *Feature forecasts*, FF1–FF2, and the
frozen `experiments/single_update/FORECAST_PROTOCOL.md`. A nominal nonlinear
defect is evaluated only at the old value plus its known tangent. Error from
the previous layer is transported by an anchored Lipschitz bound.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

omit [NormedSpace ℝ E] [NormedSpace ℝ F] in
/-- FF1: any known nominal output and anchored sensitivity propagate a
ball. The defect contains no evaluation at the unknown true input y. -/
theorem forecast_transport (f : E → F) (x v y : E) (u : F) (L r : ℝ)
    (hL : 0 ≤ L) (hy : ‖y - (x + v)‖ ≤ r)
    (hf : ∀ z, ‖f z - f (x + v)‖ ≤ L * ‖z - (x + v)‖) :
    ‖f y - (f x + u)‖ ≤ L * r + ‖f (x + v) - (f x + u)‖ := by
  calc
    _ ≤ ‖f y - f (x + v)‖ + ‖f (x + v) - (f x + u)‖ := norm_sub_le_norm_sub_add_norm_sub _ _ _
    _ ≤ L * r + ‖f (x + v) - (f x + u)‖ := by
      exact add_le_add ((hf y).trans (mul_le_mul_of_nonneg_left hy hL)) le_rfl

example := forecast_transport (fun z : ℝ => z) 2 (-1) 2 (-1) 1 1
  (by norm_num) (by norm_num) (by intro z; simp)

/-- FF1: the actual proposed linear layer transports the previous ball;
its tangent-tangent mixed term is known before any future forward pass. -/
theorem linear_forecast (A D : E →L[ℝ] F) (x v y : E) (r : ℝ)
    (hy : ‖y - (x + v)‖ ≤ r) :
    ‖(A + D) y - (A x + (A v + D x))‖ ≤ ‖A + D‖ * r + ‖D v‖ := by
  have he : (A + D) y - (A x + (A v + D x)) =
      (A + D) (y - (x + v)) + D v := by
    simp only [add_apply, map_sub, map_add]
    abel
  rw [he]
  exact (norm_add_le _ _).trans (add_le_add
    (((A + D).le_opNorm _).trans (mul_le_mul_of_nonneg_left hy (norm_nonneg _))) le_rfl)

example := linear_forecast (ContinuousLinearMap.id ℝ ℝ)
  (ContinuousLinearMap.id ℝ ℝ) 2 (-1) 2 1 (by norm_num)

omit [NormedSpace ℝ E] in
/-- FF1: residual additions add the two independent error radii. -/
theorem residual_forecast (a b x y : E) (r s : ℝ)
    (ha : ‖x - a‖ ≤ r) (hb : ‖y - b‖ ≤ s) :
    ‖(x + y) - (a + b)‖ ≤ r + s := by
  have he : (x + y) - (a + b) = (x - a) + (y - b) := by abel
  rw [he]
  exact (norm_add_le _ _).trans (add_le_add ha hb)

example := residual_forecast (1 : ℝ) 2 2 4 1 2 (by norm_num) (by norm_num)

/-- FF2: the source sigmoid has global Lipschitz constant 1/4.
The proof bounds its actual derivative, not a surrogate gate. -/
theorem sigmoid_quarter_lipschitz : LipschitzWith (1 / 4) Real.sigmoid := by
  apply lipschitzWith_of_nnnorm_deriv_le differentiable_sigmoid
  intro t
  change ‖deriv Real.sigmoid t‖ ≤ (1 / 4 : ℝ)
  rw [Real.deriv_sigmoid, Real.norm_eq_abs,
    abs_of_nonneg (mul_nonneg (Real.sigmoid_nonneg t)
      (sub_nonneg.mpr (Real.sigmoid_le_one t)))]
  nlinarith [sq_nonneg (Real.sigmoid t - 1 / 2)]

/-- FF2: compute the sigmoid defect at a known nominal point; only the
previous layer's error receives the global sensitivity factor. -/
theorem sigmoid_forecast (a v y r : ℝ) (hy : |y - (a + v)| ≤ r) :
    |Real.sigmoid y - (Real.sigmoid a + Real.sigmoid a * (1 - Real.sigmoid a) * v)| ≤
      r / 4 + |Real.sigmoid (a + v) -
        (Real.sigmoid a + Real.sigmoid a * (1 - Real.sigmoid a) * v)| := by
  have h := forecast_transport Real.sigmoid a v y
    (Real.sigmoid a * (1 - Real.sigmoid a) * v) (1 / 4) r (by norm_num) hy
    (by intro z; simpa using sigmoid_quarter_lipschitz.norm_sub_le z (a + v))
  simpa only [Real.norm_eq_abs, one_div_mul_eq_div] using h

example := sigmoid_forecast 0 1 2 1 (by norm_num)

/-- FF1: scalar bilinear defect, retaining the known tangent-tangent term.
Coordinate and complex block products use this algebra before norm bounds. -/
theorem product_forecast_identity (a b u v x y : ℝ) :
    x * y - (a * b + u * b + a * v) = u * v +
      (a + u) * (y - (b + v)) + (b + v) * (x - (a + u)) +
      (x - (a + u)) * (y - (b + v)) := by ring

/-- FF1: input error balls give an explicit product forecast radius,
including their interaction and the known tangent-tangent term. -/
theorem product_forecast_bound (a b u v x y r s : ℝ)
    (hx : |x - (a + u)| ≤ r) (hy : |y - (b + v)| ≤ s) :
    |x * y - (a * b + u * b + a * v)| ≤
      |u * v| + |a + u| * s + |b + v| * r + r * s := by
  rw [product_forecast_identity]
  have h1 := abs_add_le (u * v) ((a + u) * (y - (b + v)))
  have h2 := abs_add_le (u * v + (a + u) * (y - (b + v))) ((b + v) * (x - (a + u)))
  have h3 := abs_add_le (u * v + (a + u) * (y - (b + v)) + (b + v) * (x - (a + u)))
    ((x - (a + u)) * (y - (b + v)))
  have h4 := mul_le_mul_of_nonneg_left hy (abs_nonneg (a + u))
  have h5 := mul_le_mul_of_nonneg_left hx (abs_nonneg (b + v))
  have h6 := mul_le_mul hx hy (abs_nonneg _) ((abs_nonneg _).trans hx)
  simp only [abs_mul] at *
  linarith

example := product_forecast_bound 1 1 1 (-1) 3 1 1 1 (by norm_num) (by norm_num)

/-- FF1: a nonzero product tangent can cancel, while its known mixed term
still moves the output; the forecast radius must retain that term. -/
theorem product_forecast_mixed_witness :
    (1 : ℝ) * (-1) + 1 * 1 = 0 ∧
      (1 + 1) * (1 + (-1)) - 1 * 1 = -1 := by norm_num

end Transformer.Grokking.GradientEvidence
