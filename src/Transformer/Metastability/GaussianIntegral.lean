/-
# Gaussian normalization in Euclidean space

The corrected `d`-dimensional normalizer of `eq: gaussian.mixture` is checked
by identifying the radial Gaussian density with a product of one-dimensional
Gaussian densities. The product integrates to one by Fubini.

Source: arXiv:2410.06833v1, §4, `eq: gaussian.mixture`.
-/

import Transformer.Basic
import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace
import Mathlib.MeasureTheory.Integral.Pi
import Mathlib.Probability.Distributions.Gaussian.Real

open scoped BigOperators
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Metastability

/-- The coordinatewise Gaussian density integrates to one on `ℝ^d`.
Source: arXiv:2410.06833v1, §4, `eq: gaussian.mixture`. -/
theorem coordinate_gaussian_integral (d : ℕ) (σ : ℝ) (hσ : 0 < σ)
    (μ : EucSpace d) :
    ∫ x : EucSpace d,
      ∏ i : Fin d, gaussianPDFReal (μ i) ⟨σ ^ 2, sq_nonneg σ⟩ (x i) = 1 := by
  let v : NNReal := ⟨σ ^ 2, sq_nonneg σ⟩
  have hv : v ≠ 0 := by
    apply ne_of_gt
    exact_mod_cast (pow_pos hσ 2)
  have hprod : (∫ y : Fin d → ℝ, ∏ i : Fin d, gaussianPDFReal (μ i) v (y i)) = 1 := by
    rw [integral_fintype_prod_volume_eq_prod]
    simp [integral_gaussianPDFReal_eq_one _ hv]
  have hmp := EuclideanSpace.volume_preserving_symm_measurableEquiv_toLp (Fin d)
  have hchange := hmp.integral_comp' (fun y : Fin d → ℝ =>
    ∏ i : Fin d, gaussianPDFReal (μ i) v (y i))
  simpa [v] using hchange.trans hprod

/-- The coordinate product is exactly the corrected radial density in
`eq: gaussian.mixture`. Source: arXiv:2410.06833v1, §4. -/
theorem coordinate_gaussian_eq_radial (d : ℕ) (σ : ℝ) (hσ : 0 < σ)
    (v : NNReal) (hv : (v : ℝ) = σ ^ 2) (μ x : EucSpace d) :
    (∏ i : Fin d, gaussianPDFReal (μ i) v (x i)) =
      (2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) *
        Real.exp (-‖x - μ‖ ^ 2 / (2 * σ ^ 2)) := by
  let a : ℝ := 2 * Real.pi * σ ^ 2
  have ha : 0 < a := by dsimp [a]; positivity
  have hsum : (∑ i : Fin d, (x i - μ i) ^ 2) = ‖x - μ‖ ^ 2 := by
    rw [EuclideanSpace.norm_sq_eq]
    congr 1
    funext i
    simp only [PiLp.sub_apply, Real.norm_eq_abs, sq_abs]
  simp_rw [gaussianPDFReal_def]
  rw [hv]
  rw [Finset.prod_mul_distrib, Finset.prod_const, Finset.card_fin]
  have hexp : (∏ i : Fin d, Real.exp (-(x i - μ i) ^ 2 / (2 * σ ^ 2))) =
      Real.exp (-‖x - μ‖ ^ 2 / (2 * σ ^ 2)) := by
    rw [← Real.exp_sum]
    congr 1
    rw [← Finset.sum_div, Finset.sum_neg_distrib, hsum]
  rw [hexp]
  congr 1
  rw [Real.sqrt_eq_rpow, ← Real.rpow_neg ha.le,
    ← Real.rpow_natCast (a ^ (-(1 / 2 : ℝ))) d, ← Real.rpow_mul ha.le]
  congr 1
  ring

/-- Coordinatewise Gaussian densities are integrable on `ℝ^d`.
Source: arXiv:2410.06833v1, §4, `eq: gaussian.mixture`. -/
theorem coordinate_gaussian_integrable (d : ℕ) (v : NNReal) (μ : EucSpace d) :
    Integrable (fun x : EucSpace d =>
      ∏ i : Fin d, gaussianPDFReal (μ i) v (x i)) volume := by
  have hprod : Integrable (fun y : Fin d → ℝ =>
      ∏ i : Fin d, gaussianPDFReal (μ i) v (y i)) volume := by
    rw [volume_pi]
    exact Integrable.fintype_prod (fun i => integrable_gaussianPDFReal (μ i) v)
  have hmp : MeasurePreserving
      ((MeasurableEquiv.toLp 2 (Fin d → ℝ)).symm : EucSpace d → Fin d → ℝ)
      (volume : Measure (EucSpace d)) (volume : Measure (Fin d → ℝ)) :=
    EuclideanSpace.volume_preserving_symm_measurableEquiv_toLp (Fin d)
  simpa only [Function.comp_def, MeasurableEquiv.toLp_symm_apply] using
    hmp.integrable_comp_of_integrable hprod

/-- The corrected radial Gaussian density is integrable.
Source: arXiv:2410.06833v1, §4, `eq: gaussian.mixture`. -/
theorem radial_gaussian_integrable (d : ℕ) (σ : ℝ) (hσ : 0 < σ)
    (μ : EucSpace d) :
    Integrable (fun x : EucSpace d =>
      (2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) *
        Real.exp (-‖x - μ‖ ^ 2 / (2 * σ ^ 2))) volume := by
  let v : NNReal := ⟨σ ^ 2, sq_nonneg σ⟩
  have hv : (v : ℝ) = σ ^ 2 := rfl
  have hfun : (fun x : EucSpace d =>
      ∏ i : Fin d, gaussianPDFReal (μ i) v (x i)) =
      (fun x : EucSpace d =>
        (2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) *
          Real.exp (-‖x - μ‖ ^ 2 / (2 * σ ^ 2))) := by
    funext x
    exact coordinate_gaussian_eq_radial d σ hσ v hv μ x
  rw [← hfun]
  exact coordinate_gaussian_integrable d v μ

/-- The corrected radial Gaussian density has total mass one.
Source: arXiv:2410.06833v1, §4, `eq: gaussian.mixture`. -/
theorem radial_gaussian_integral (d : ℕ) (σ : ℝ) (hσ : 0 < σ)
    (μ : EucSpace d) :
    ∫ x : EucSpace d,
      (2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) *
        Real.exp (-‖x - μ‖ ^ 2 / (2 * σ ^ 2)) = 1 := by
  let v : NNReal := ⟨σ ^ 2, sq_nonneg σ⟩
  have hv : (v : ℝ) = σ ^ 2 := rfl
  simp_rw [← coordinate_gaussian_eq_radial d σ hσ v hv μ]
  exact coordinate_gaussian_integral d σ hσ μ

/-- The quadratic exponential moment of a radial Gaussian:
`E exp(‖X-μ‖²/(4σ²)) = 2^(d/2)`.  This is the normalization formula at the
larger scale `√2 σ`, and is the starting point for a Gaussian-radius tail
bound in `prop: mixture.of.gaussians`.
Source: arXiv:2410.06833v1, §4, proof of `prop: mixture.of.gaussians`. -/
theorem radial_gaussian_exp_norm_sq_integral (d : ℕ) (σ : ℝ) (hσ : 0 < σ)
    (μ : EucSpace d) :
    ∫ x : EucSpace d, Real.exp (‖x - μ‖ ^ 2 / (4 * σ ^ 2)) *
      ((2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) *
        Real.exp (-‖x - μ‖ ^ 2 / (2 * σ ^ 2))) =
      (2 : ℝ) ^ ((d : ℝ) / 2) := by
  have hsqrt : (Real.sqrt 2) ^ 2 = (2 : ℝ) := Real.sq_sqrt (by norm_num)
  have hσ' : 0 < σ * Real.sqrt 2 := mul_pos hσ (Real.sqrt_pos.2 (by norm_num))
  have ha : 0 < 2 * Real.pi * σ ^ 2 := by positivity
  have hscale : 2 * Real.pi * (σ * Real.sqrt 2) ^ 2 =
      2 * (2 * Real.pi * σ ^ 2) := by nlinarith [hsqrt]
  have hden : 2 * (σ * Real.sqrt 2) ^ 2 = 4 * σ ^ 2 := by
    nlinarith [hsqrt]
  have h2pow : (2 : ℝ) ^ ((d : ℝ) / 2) * (2 : ℝ) ^ (-(d : ℝ) / 2) = 1 := by
    rw [show -(d : ℝ) / 2 = -((d : ℝ) / 2) by ring]
    rw [Real.rpow_neg (by norm_num : (0 : ℝ) ≤ 2)]
    exact mul_inv_cancel₀ (by positivity)
  have hfac : (2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) =
      (2 : ℝ) ^ ((d : ℝ) / 2) *
        (2 * Real.pi * (σ * Real.sqrt 2) ^ 2) ^ (-(d : ℝ) / 2) := by
    rw [hscale, Real.mul_rpow (by norm_num : (0 : ℝ) ≤ 2) ha.le]
    rw [← mul_assoc, h2pow, one_mul]
  have hpoint (x : EucSpace d) :
      Real.exp (‖x - μ‖ ^ 2 / (4 * σ ^ 2)) *
          ((2 * Real.pi * σ ^ 2) ^ (-(d : ℝ) / 2) *
            Real.exp (-‖x - μ‖ ^ 2 / (2 * σ ^ 2))) =
        (2 : ℝ) ^ ((d : ℝ) / 2) *
          ((2 * Real.pi * (σ * Real.sqrt 2) ^ 2) ^ (-(d : ℝ) / 2) *
            Real.exp (-‖x - μ‖ ^ 2 / (2 * (σ * Real.sqrt 2) ^ 2))) := by
    rw [hfac, hden]
    have hexp : Real.exp (‖x - μ‖ ^ 2 / (4 * σ ^ 2)) *
        Real.exp (-‖x - μ‖ ^ 2 / (2 * σ ^ 2)) =
          Real.exp (-‖x - μ‖ ^ 2 / (4 * σ ^ 2)) := by
      rw [← Real.exp_add]
      congr 1
      ring
    calc
      _ = (2 : ℝ) ^ ((d : ℝ) / 2) *
          (2 * Real.pi * (σ * Real.sqrt 2) ^ 2) ^ (-(d : ℝ) / 2) *
            (Real.exp (‖x - μ‖ ^ 2 / (4 * σ ^ 2)) *
              Real.exp (-‖x - μ‖ ^ 2 / (2 * σ ^ 2))) := by ring
      _ = _ := by rw [hexp]; ring
  simp_rw [hpoint]
  rw [integral_const_mul, radial_gaussian_integral d (σ * Real.sqrt 2) hσ' μ]
  ring

/-- The hypotheses of the Gaussian normalization lemmas are satisfiable at
`σ = 1`, with variance `1`. -/
example : (0 : ℝ) < 1 ∧ ((1 : NNReal) : ℝ) = (1 : ℝ) ^ 2 := by norm_num

end Metastability
end Transformer
