/-
# The error function in signed-gradient drift

arXiv:2506.12543v1, Section 4.3, Theorem 1 and its takeaway.
The source says that erf is linear on a large interval around zero.
It is only asymptotically linear: the strict inequality below refutes
literal linearity on any positive interval.
-/

import Mathlib
import Transformer.Basic

open MeasureTheory Filter
open scoped Topology

noncomputable section

namespace Transformer.BatchSize

/-- The integral definition in Section 4.3, Theorem 1. -/
def errorFunction (x : ℝ) : ℝ :=
  (2 / Real.sqrt Real.pi) * ∫ t in 0..x, Real.exp (-(t ^ 2))

/-- Continuity of the integrand in Section 4.3's error function. -/
theorem continuous_error_integrand : Continuous (fun t : ℝ => Real.exp (-(t ^ 2))) :=
  Real.continuous_exp.comp (continuous_id.pow 2).neg

/-- Exact derivative of the error function, Section 4.3. -/
theorem errorFunction_hasDerivAt (x : ℝ) :
    HasDerivAt errorFunction
      ((2 / Real.sqrt Real.pi) * Real.exp (-(x ^ 2))) x := by
  exact (intervalIntegral.integral_hasDerivAt_right
    (continuous_error_integrand.intervalIntegrable 0 x)
    continuous_error_integrand.stronglyMeasurable.stronglyMeasurableAtFilter
    continuous_error_integrand.continuousAt).const_mul _

/-- Smoothness of the actual error function, Section 4.3, equation (3).
This supplies smooth state coefficients for the signed diffusion. -/
theorem errorFunction_contDiff : ContDiff ℝ (⊤ : ℕ∞) errorFunction := by
  apply contDiff_infty_iff_deriv.mpr
  refine ⟨fun x => (errorFunction_hasDerivAt x).differentiableAt, ?_⟩
  have hd : deriv errorFunction =
      fun x => (2 / Real.sqrt Real.pi) * Real.exp (-(x ^ 2)) :=
    funext fun x => (errorFunction_hasDerivAt x).deriv
  rw [hd]
  fun_prop

/-- The exact drift response is globally Lipschitz with its maximal
derivative at zero, Section 4.3, equation (3). -/
theorem errorFunction_lipschitz :
    LipschitzWith (Real.toNNReal (2 / Real.sqrt Real.pi)) errorFunction := by
  apply lipschitzWith_of_nnnorm_deriv_le
    (fun x => (errorFunction_hasDerivAt x).differentiableAt)
  intro x
  rw [← NNReal.coe_le_coe, coe_nnnorm, (errorFunction_hasDerivAt x).deriv]
  have hc : 0 ≤ 2 / Real.sqrt Real.pi := by positivity
  rw [Real.coe_toNNReal _ hc, Real.norm_eq_abs,
    abs_of_nonneg (mul_nonneg hc (Real.exp_pos _).le)]
  exact mul_le_of_le_one_right hc (Real.exp_le_one_iff.mpr (neg_nonpos.mpr (sq_nonneg x)))

/-- Zero drift at zero signal, Section 4.3, Theorem 1. -/
theorem errorFunction_zero : errorFunction 0 = 0 := by
  simp [errorFunction]

/-- Oddness supplies the negative-gradient case in Section 4.3's proof sketch. -/
theorem errorFunction_neg (x : ℝ) : errorFunction (-x) = -errorFunction x := by
  have h : (∫ t in 0..x, Real.exp (-(t ^ 2))) =
      ∫ t in -x..0, Real.exp (-(t ^ 2)) := by
    have heven : (fun t : ℝ => Real.exp (-((-t) ^ 2))) =
        (fun t : ℝ => Real.exp (-(t ^ 2))) := by funext t; simp
    have hi := intervalIntegral.integral_comp_neg
      (f := fun t : ℝ => Real.exp (-(t ^ 2))) (a := 0) (b := x)
    rw [heven, neg_zero] at hi
    exact hi
  dsimp [errorFunction]
  rw [intervalIntegral.integral_symm (-x) 0, ← h]
  ring

/-- The drift response is strictly increasing, Section 4.3's takeaway. -/
theorem errorFunction_strictMono : StrictMono errorFunction := by
  apply strictMono_of_deriv_pos
  intro x
  rw [(errorFunction_hasDerivAt x).deriv]
  exact mul_pos (div_pos (by norm_num) (Real.sqrt_pos.2 Real.pi_pos)) (Real.exp_pos _)

/-- Absolute signal controls absolute signed response, Section 4.3,
equation (3). This is used to bound diffusion coefficients uniformly. -/
theorem errorFunction_abs (x : ℝ) : |errorFunction x| = errorFunction |x| := by
  rcases le_total 0 x with hx | hx
  · have h : 0 ≤ errorFunction x := by
      simpa [errorFunction_zero] using errorFunction_strictMono.monotone hx
    rw [abs_of_nonneg hx, abs_of_nonneg h]
  · have h : errorFunction x ≤ 0 := by
      simpa [errorFunction_zero] using errorFunction_strictMono.monotone hx
    rw [abs_of_nonpos hx, abs_of_nonpos h, errorFunction_neg]

/-- The source's literal linearity claim is false: for every positive
signal erf(x) lies strictly below its tangent at zero; Section 4.3's takeaway. -/
theorem errorFunction_lt_linear (x : ℝ) (hx : 0 < x) :
    errorFunction x < (2 / Real.sqrt Real.pi) * x := by
  have h := intervalIntegral.integral_lt_integral_of_continuousOn_of_le_of_exists_lt
    hx continuous_error_integrand.continuousOn continuous_const.continuousOn
    (fun t _ => Real.exp_le_one_iff.2 (neg_nonpos.2 (sq_nonneg t)))
    ⟨x, ⟨hx.le, le_rfl⟩, Real.exp_lt_one_iff.2 (neg_neg_of_pos (sq_pos_of_pos hx))⟩
  simpa [errorFunction] using mul_lt_mul_of_pos_left h
    (div_pos (by norm_num) (Real.sqrt_pos.2 Real.pi_pos))

/-- Nonvacuity of the positive-signal counterexample, Section 4.3. -/
example : (0 : ℝ) < 1 / 1000 := by norm_num

/-- Corrected local linearity: erf(x)/x tends to 2/sqrt(pi), rather
than being exactly linear on an interval; Section 4.3's takeaway. -/
theorem errorFunction_linearization :
    Tendsto (fun x => errorFunction x / x) (𝓝[≠] 0)
      (𝓝 (2 / Real.sqrt Real.pi)) := by
  simpa [errorFunction_zero, div_eq_mul_inv, mul_comm] using
    (errorFunction_hasDerivAt 0).tendsto_slope_zero

/-- Saturation is a limit at infinite signal, Section 4.3's takeaway. -/
theorem errorFunction_tendsto_atTop : Tendsto errorFunction atTop (𝓝 1) := by
  have hi : Integrable (fun t : ℝ => Real.exp (-(t ^ 2))) := by
    simpa using integrable_exp_neg_mul_sq (b := 1) (by norm_num)
  have h := (intervalIntegral_tendsto_integral_Ioi 0 hi.integrableOn tendsto_id).const_mul
    (2 / Real.sqrt Real.pi)
  have hval : (2 / Real.sqrt Real.pi) *
      (∫ t in Set.Ioi 0, Real.exp (-(t ^ 2))) = 1 := by
    have hg := integral_gaussian_Ioi 1
    simp only [neg_mul, one_mul, div_one] at hg
    rw [hg]
    field_simp
  change Tendsto (fun x => (2 / Real.sqrt Real.pi) *
    ∫ t in 0..x, Real.exp (-(t ^ 2))) atTop (𝓝 1)
  simpa only [hval, id_eq] using h

/-- At finite signal the response has magnitude less than one, so
the diffusion covariance in Section 4.3, Theorem 1 is positive. -/
theorem errorFunction_abs_lt_one (x : ℝ) : |errorFunction x| < 1 := by
  have hle : ∀ y : ℝ, errorFunction y ≤ 1 :=
    errorFunction_strictMono.monotone.ge_of_tendsto errorFunction_tendsto_atTop
  have hlt (y : ℝ) : errorFunction y < 1 :=
    (errorFunction_strictMono (lt_add_one y)).trans_le (hle (y + 1))
  apply abs_lt.mpr
  constructor
  · have hn := hlt (-x)
    rw [errorFunction_neg] at hn
    linarith
  · exact hlt x

end Transformer.BatchSize
