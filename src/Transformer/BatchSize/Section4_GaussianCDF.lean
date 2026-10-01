/-
# Gaussian probabilities behind the signed-gradient drift

arXiv:2506.12543v1, Section 4.3, proof sketch following Theorem 1.
The CDF formula is valid at every threshold, including below the mean.
-/

import Transformer.BatchSize.Section4_ErrorFunction

open MeasureTheory ProbabilityTheory

noncomputable section

namespace Transformer.BatchSize

/-- The rescaled centered Gaussian has density exp(-x^2)/sqrt(pi);
Section 4.3's error-function proof sketch. -/
theorem gaussian_half_density (x : ℝ) :
    gaussianPDFReal 0 (1 / 2) x = (Real.sqrt Real.pi)⁻¹ * Real.exp (-(x ^ 2)) := by
  unfold gaussianPDFReal
  have hv : ((1 / 2 : NNReal) : ℝ) = 1 / 2 := by norm_num
  rw [hv]
  have hπ : 2 * Real.pi * (1 / 2) = Real.pi := by ring
  rw [hπ]
  norm_num

/-- The centered Gaussian with variance 1/2 assigns half its mass to
each side of zero; Section 4.3's proof sketch. -/
theorem gaussian_half_integral_Iic_zero :
    (∫ x in Set.Iic (0 : ℝ), gaussianPDFReal 0 (1 / 2) x) = 1 / 2 := by
  have hi := integrable_gaussianPDFReal 0 (1 / 2)
  have hsum := intervalIntegral.integral_Iic_add_Ioi hi.integrableOn hi.integrableOn
    (b := 0)
  rw [integral_gaussianPDFReal_eq_one 0 (by norm_num)] at hsum
  have hright : (∫ x in Set.Ioi (0 : ℝ), gaussianPDFReal 0 (1 / 2) x) = 1 / 2 := by
    simp_rw [gaussian_half_density]
    rw [integral_const_mul]
    have hg := integral_gaussian_Ioi 1
    simp only [neg_mul, one_mul, div_one] at hg
    rw [hg]
    field_simp
  rw [hright] at hsum
  linarith

/-- The Gaussian CDF really equals the integral-defined error function;
Section 4.3's proof sketch. -/
theorem gaussian_half_cdf (x : ℝ) :
    (gaussianReal 0 (1 / 2)).real (Set.Iic x) = (1 + errorFunction x) / 2 := by
  have hi := integrable_gaussianPDFReal 0 (1 / 2)
  have hdiff := intervalIntegral.integral_Iic_sub_Iic
    (a := 0) (b := x) hi.integrableOn hi.integrableOn
  rw [gaussian_half_integral_Iic_zero] at hdiff
  simp_rw [gaussian_half_density] at hdiff
  rw [intervalIntegral.integral_const_mul] at hdiff
  rw [Measure.real, gaussianReal_apply_eq_integral 0 (by norm_num),
    ENNReal.toReal_ofReal (integral_nonneg (fun t => gaussianPDFReal_nonneg _ _ t))]
  simp_rw [gaussian_half_density]
  dsimp [errorFunction]
  simp only [div_eq_mul_inv] at *
  nlinarith [hdiff]

/-- The affine centered normal law is N(mu,s^2/2); Section 4.3.
Here s is sqrt(2*variance), the denominator in the paper's CDF formula. -/
theorem gaussian_affine_law (μ s : ℝ) :
    (gaussianReal 0 (1 / 2)).map (fun z => μ + s * z) =
      gaussianReal μ (.mk (s ^ 2 / 2) (by positivity)) := by
  have hf : (fun z : ℝ => μ + s * z) = (fun z => μ + z) ∘ (fun z => s * z) := rfl
  rw [hf, ← Measure.map_map (g := fun z : ℝ => μ + z) (f := fun z => s * z)
    (measurable_const.add measurable_id)
    (measurable_const.mul measurable_id), gaussianReal_map_const_mul,
    gaussianReal_map_const_add]
  congr 1
  · simp
  · apply NNReal.coe_injective
    simp
    ring

/-- Correct CDF identity for every threshold, Section 4.3's proof sketch.
The source restricts to ell>mu, but subsequently uses ell=0 below a
positive mean. No such restriction is needed. -/
theorem gaussian_cdf (μ s ℓ : ℝ) (hs : 0 < s) :
    (gaussianReal μ (.mk (s ^ 2 / 2) (by positivity))).real (Set.Iic ℓ) =
      (1 + errorFunction ((ℓ - μ) / s)) / 2 := by
  rw [← gaussian_affine_law, Measure.real,
    Measure.map_apply (f := fun z : ℝ => μ + s * z)
      (measurable_const.add (measurable_const.mul measurable_id))
      measurableSet_Iic]
  have hset : (fun z : ℝ => μ + s * z) ⁻¹' Set.Iic ℓ = Set.Iic ((ℓ - μ) / s) := by
    ext z
    simp only [Set.mem_preimage, Set.mem_Iic, le_div_iff₀ hs]
    constructor <;> intro hz <;> nlinarith
  rw [hset]
  exact gaussian_half_cdf _

/-- Nonvacuity of the nondegenerate CDF hypothesis, Section 4.3. -/
example : (0 : ℝ) < Real.sqrt 2 := by positivity

end Transformer.BatchSize
