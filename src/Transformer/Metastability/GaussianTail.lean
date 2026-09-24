/-
# An exponential-radius bound for one Gaussian component

The corrected radial Gaussian density has an exponential moment. Markov's
inequality then gives a dimension-dependent radius tail bound. This is a
partial concentration estimate for `prop: mixture.of.gaussians`, not the
sharper Gaussian concentration inequality used in the manuscript.

Source: arXiv:2410.06833v1, §4, proof of `prop: mixture.of.gaussians`.
-/

import Transformer.Metastability.GaussianIntegral
import Mathlib.MeasureTheory.Integral.Bochner.ContinuousLinearMap

open Real MeasureTheory

namespace Transformer
namespace Metastability

/-- One normalized radial Gaussian density defines a probability measure.
Source: arXiv:2410.06833v1, §4, `eq: gaussian.mixture`. -/
theorem radial_gaussian_probability (d : ℕ) (σ : ℝ) (hσ : 0 < σ)
    (μ : EucSpace d) :
    IsProbabilityMeasure (volume.withDensity fun x : EucSpace d =>
      ENNReal.ofReal ((2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) *
        Real.exp (-‖x - μ‖ ^ 2 / (2 * σ ^ 2)))) := by
  constructor
  rw [withDensity_apply _ MeasurableSet.univ]
  simp only [Measure.restrict_univ]
  rw [← ofReal_integral_eq_lintegral_ofReal
    (radial_gaussian_integrable d σ hσ μ)
    (Filter.Eventually.of_forall (by intro x; positivity))]
  rw [radial_gaussian_integral d σ hσ μ]
  simp

/-- The quadratic exponential moment under one normalized radial Gaussian
law equals `2^(d/2)`.
Source: arXiv:2410.06833v1, §4, proof of `prop: mixture.of.gaussians`. -/
theorem radial_gaussian_exp_norm_sq_withDensity_integral (d : ℕ) (σ : ℝ)
    (hσ : 0 < σ) (μ : EucSpace d) :
    ∫ x : EucSpace d, Real.exp (‖x - μ‖ ^ 2 / (4 * σ ^ 2))
      ∂(volume.withDensity fun x : EucSpace d =>
        ENNReal.ofReal ((2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) *
          Real.exp (-‖x - μ‖ ^ 2 / (2 * σ ^ 2)))) =
      (2 : ℝ) ^ ((d : ℝ) / 2) := by
  have hmeas : Measurable (fun x : EucSpace d =>
      ENNReal.ofReal ((2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) *
        Real.exp (-‖x - μ‖ ^ 2 / (2 * σ ^ 2)))) := by fun_prop
  have hfinite : ∀ᵐ x ∂(volume : Measure (EucSpace d)),
      ENNReal.ofReal ((2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) *
        Real.exp (-‖x - μ‖ ^ 2 / (2 * σ ^ 2))) < ⊤ := by
    filter_upwards [] with x
    simp
  rw [integral_withDensity_eq_integral_toReal_smul hmeas hfinite]
  have hpoint (x : EucSpace d) :
      (ENNReal.ofReal ((2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) *
        Real.exp (-‖x - μ‖ ^ 2 / (2 * σ ^ 2)))).toReal •
        Real.exp (‖x - μ‖ ^ 2 / (4 * σ ^ 2)) =
      Real.exp (‖x - μ‖ ^ 2 / (4 * σ ^ 2)) *
        ((2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) *
          Real.exp (-‖x - μ‖ ^ 2 / (2 * σ ^ 2))) := by
    rw [ENNReal.toReal_ofReal (by positivity)]
    simp [smul_eq_mul, mul_comm]
  simp_rw [hpoint]
  exact radial_gaussian_exp_norm_sq_integral d σ hσ μ

/-- Exponential Markov bound for the squared radius of one Gaussian
component. It provides a rigorous, though weaker, substitute for the
Gaussian concentration bound invoked in the manuscript.
Source: arXiv:2410.06833v1, §4, proof of `prop: mixture.of.gaussians`. -/
theorem radial_gaussian_exp_norm_sq_tail (d : ℕ) (σ : ℝ) (hσ : 0 < σ)
    (μ : EucSpace d) (R : ℝ) :
    Real.exp R *
      (volume.withDensity fun x : EucSpace d =>
        ENNReal.ofReal ((2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) *
          Real.exp (-‖x - μ‖ ^ 2 / (2 * σ ^ 2)))).real
        {x | R ≤ ‖x - μ‖ ^ 2 / (4 * σ ^ 2)} ≤
      (2 : ℝ) ^ ((d : ℝ) / 2) := by
  let m : Measure (EucSpace d) := volume.withDensity fun x =>
    ENNReal.ofReal ((2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) *
      Real.exp (-‖x - μ‖ ^ 2 / (2 * σ ^ 2)))
  let f (x : EucSpace d) : ℝ := Real.exp (‖x - μ‖ ^ 2 / (4 * σ ^ 2))
  have hmoment : ∫ x, f x ∂m = (2 : ℝ) ^ ((d : ℝ) / 2) :=
    radial_gaussian_exp_norm_sq_withDensity_integral d σ hσ μ
  have hint : Integrable f m := Integrable.of_integral_ne_zero (by
    rw [hmoment]
    positivity)
  have hnonneg : 0 ≤ᵐ[m] f := Filter.Eventually.of_forall fun x =>
    (Real.exp_pos _).le
  have hmarkov := mul_meas_ge_le_integral_of_nonneg hnonneg hint (Real.exp R)
  have hset : {x : EucSpace d | Real.exp R ≤ f x} =
      {x | R ≤ ‖x - μ‖ ^ 2 / (4 * σ ^ 2)} := by
    ext x
    simp only [Set.mem_ofPred_eq, f, Real.exp_le_exp]
  simpa only [hset, hmoment, m] using hmarkov

/-- Unit scale gives a nondegenerate instance of the tail bound. -/
example : (0 : ℝ) < 1 := one_pos

end Metastability
end Transformer
