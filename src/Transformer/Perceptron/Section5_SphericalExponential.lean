module

public import Transformer.Perceptron.Section5_CircleRotation
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Sinc

/-!
# Actual spherical exponential curves in arbitrary dimension

ArXiv:2601.21366v2, §2.2, the exponential-map geodesic, and §5.3,
the localized directions in `thm: any.d` (i). We rewrite the original
`sphereExp(x,tv)` using `sinc` to retain continuity at zero speed.
Its actual time velocity and radial acceleration are then computed
for every ambient dimension, including zero directions.
At zero speed the original curve is constant; at nonzero speed its
sinc coefficient is exactly the sine divided by the initial speed.

The original exponential map is unchanged. Norm one follows from its
proved geometric identity, and the norm of its actual velocity is the
initial speed. These facts precede construction of the actual pushforward
geodesics and the second differentiation of the original energy.
-/

@[expose] public section

open Real

namespace Transformer.Perceptron

/-- The even sinc coefficient removes the absolute value in the
original spherical exponential. Source: arXiv:2601.21366v2, §2.2, `exp_x(tv)`. -/
theorem sinc_abs_eq (r : ℝ) : Real.sinc |r| = Real.sinc r := by
  rcases le_total 0 r with hr | hr
  · rw [abs_of_nonneg hr]
  · rw [abs_of_nonpos hr, Real.sinc_neg]

/-- The original exponential formula has the continuous sinc form,
including the original value at zero direction.
Source: arXiv:2601.21366v2, §2.2, the spherical exponential-map footnote. -/
theorem sphereExp_eq_cos_sinc {d : ℕ} (x v : EucSpace d) :
    sphereExp x v = Real.cos ‖v‖ • x + Real.sinc ‖v‖ • v := by
  by_cases hv : v = 0
  · simp [hv, sphereExp]
  · rw [sphereExp, Real.sinc_of_ne_zero (norm_ne_zero_iff.mpr hv)]

/-- The original exponential curve, with its signed real time exposed.
Source: arXiv:2601.21366v2, §2.2, `T(t)(x)=exp_x(tξ(x))`. -/
theorem sphereExp_smul_eq {d : ℕ} (x v : EucSpace d) (t : ℝ) :
    sphereExp x (t • v) = Real.cos (t * ‖v‖) • x + (t * Real.sinc (t * ‖v‖)) • v := by
  have hn : ‖t • v‖ = |t * ‖v‖| := by
    rw [norm_smul, Real.norm_eq_abs, abs_mul, abs_of_nonneg (norm_nonneg v)]
  rw [sphereExp_eq_cos_sinc, hn, Real.cos_abs, sinc_abs_eq, smul_smul]
  congr 1
  congr 1
  ring

/-- The actual original spherical exponential curve in continuous coordinates.
Source: arXiv:2601.21366v2, §2.2 and §5.3, the Monge geodesic. -/
noncomputable def spherePosition {d : ℕ} (x v : EucSpace d) (t : ℝ) : EucSpace d :=
  Real.cos (t * ‖v‖) • x + (t * Real.sinc (t * ‖v‖)) • v

/-- The actual time velocity of the original exponential curve.
Source: arXiv:2601.21366v2, §2.2 and §5.3, the geodesic first derivative. -/
noncomputable def sphereVelocity {d : ℕ} (x v : EucSpace d) (t : ℝ) : EucSpace d :=
  -(‖v‖ * Real.sin (t * ‖v‖)) • x + Real.cos (t * ‖v‖) • v

/-- For nonzero fixed speed, the original sinc coefficient is the
usual sine divided by that speed, at every time including zero.
Source: arXiv:2601.21366v2, §2.2, the spherical exponential formula. -/
theorem time_sinc_eq_sin_div {r : ℝ} (hr : r ≠ 0) (t : ℝ) :
    t * Real.sinc (t * r) = Real.sin (t * r) / r := by
  by_cases ht : t = 0
  · simp [ht]
  · rw [Real.sinc_of_ne_zero (mul_ne_zero ht hr)]
    field_simp

example := time_sinc_eq_sin_div one_ne_zero 0

/-- The original exponential curve starts at the original point.
Source: arXiv:2601.21366v2, §2.2, the initial value of the Monge map. -/
@[simp] theorem spherePosition_zero {d : ℕ} (x v : EucSpace d) : spherePosition x v 0 = x := by
  simp [spherePosition]

/-- The actual first derivative of the original exponential curve.
Source: arXiv:2601.21366v2, §2.2 and §5.3, the geodesic velocity in the Hessian. -/
theorem hasDerivAt_spherePosition {d : ℕ} (x v : EucSpace d) (t : ℝ) :
    HasDerivAt (spherePosition x v) (sphereVelocity x v t) t := by
  by_cases hv : v = 0
  · subst v
    have heq : spherePosition x (0 : EucSpace d) = fun _ => x := by
      funext r
      simp [spherePosition]
    rw [heq]
    simpa [spherePosition, sphereVelocity] using hasDerivAt_const t x
  · have hr : ‖v‖ ≠ 0 := norm_ne_zero_iff.mpr hv
    have heq : spherePosition x v = fun r =>
        Real.cos (r * ‖v‖) • x + (Real.sin (r * ‖v‖) / ‖v‖) • v := by
      funext r
      rw [spherePosition, time_sinc_eq_sin_div hr]
    rw [heq]
    have ht := (hasDerivAt_id' (x := t)).mul_const ‖v‖
    have h := (ht.cos.smul_const x).fun_add ((ht.sin.div_const ‖v‖).smul_const v)
    apply h.congr_deriv
    simp only [sphereVelocity, one_mul, mul_div_cancel_right₀ _ hr]
    congr 1
    congr 1
    ring

/-- The actual velocity has the original radial acceleration, with
squared initial speed. Source: arXiv:2601.21366v2, §2.2 and §5.3, the spherical Hessian. -/
theorem hasDerivAt_sphereVelocity {d : ℕ} (x v : EucSpace d) (t : ℝ) :
    HasDerivAt (sphereVelocity x v) (-(‖v‖ ^ 2) • spherePosition x v t) t := by
  by_cases hv : v = 0
  · subst v
    have heq : sphereVelocity x (0 : EucSpace d) = fun _ => 0 := by
      funext r
      simp [sphereVelocity]
    rw [heq]
    simpa [sphereVelocity] using hasDerivAt_const t (0 : EucSpace d)
  · have hr : ‖v‖ ≠ 0 := norm_ne_zero_iff.mpr hv
    have ht := (hasDerivAt_id' (x := t)).mul_const ‖v‖
    have h := ((ht.sin.const_mul ‖v‖).neg.smul_const x).fun_add (ht.cos.smul_const v)
    apply h.congr_deriv
    rw [spherePosition, time_sinc_eq_sin_div hr, smul_add, smul_smul, smul_smul]
    simp only [one_mul]
    congr 1 <;> congr 1 <;> field_simp

/-- The actual exponential curve stays on the original sphere.
Source: arXiv:2601.21366v2, §2.2, the original exponential-map geodesic. -/
theorem norm_spherePosition {d : ℕ} {x v : EucSpace d} (hx : ‖x‖ = 1)
    (hv : inner (𝕜 := ℝ) x v = 0) (t : ℝ) : ‖spherePosition x v t‖ = 1 := by
  rw [show spherePosition x v t = sphereExp x (t • v) from (sphereExp_smul_eq x v t).symm]
  exact norm_sphereExp hx (by rw [real_inner_smul_right, hv, mul_zero])

/-- The actual velocity norm is the original constant initial speed.
Source: arXiv:2601.21366v2, §2.2 and §5.3, geodesic speed in the Hessian localization. -/
theorem norm_sphereVelocity {d : ℕ} {x v : EucSpace d} (hx : ‖x‖ = 1)
    (hv : inner (𝕜 := ℝ) x v = 0) (t : ℝ) : ‖sphereVelocity x v t‖ = ‖v‖ := by
  have hs : ‖sphereVelocity x v t‖ ^ 2 = ‖v‖ ^ 2 := by
    rw [sphereVelocity, norm_add_sq_real, real_inner_smul_left, real_inner_smul_right, hv,
      mul_zero, mul_zero, norm_smul, norm_smul, hx, Real.norm_eq_abs, Real.norm_eq_abs,
      mul_one, mul_pow, sq_abs, sq_abs]
    nlinarith [Real.sin_sq_add_cos_sq (t * ‖v‖)]
  nlinarith [norm_nonneg (sphereVelocity x v t), norm_nonneg v]

/-- The original circle point and its nonzero unit tangent realize all
unit-position and tangent hypotheses in this arbitrary-dimensional calculus.
Source: arXiv:2601.21366v2, §2.2 and §5.2, `x(θ)` and `x'(θ)`. -/
example : ‖(circlePoint 0 : EucSpace 2)‖ = 1 ∧
    inner (𝕜 := ℝ) (circlePoint 0 : EucSpace 2) (circleTangent (circlePoint 0 : EucSpace 2)) = 0 ∧
    ‖circleTangent (circlePoint 0 : EucSpace 2)‖ = 1 :=
  ⟨norm_eq_of_mem_sphere _, inner_circleTangent _,
    (norm_circleTangent _).trans (norm_eq_of_mem_sphere _)⟩

end Transformer.Perceptron
