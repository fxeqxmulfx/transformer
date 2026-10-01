/-
# Gaussian integration by parts for the diffusion generator

arXiv:2506.12543v1, Section 4.3, equations (2)--(3).
This analytic auxiliary result identifies the Brownian generator; it is
not a proof of existence of the state-dependent SDE in Theorem 1.

The density-transfer and integration-by-parts argument is adapted from
MathFin/Foundations/FeynmanKacHeatEquation.lean, formal-applied-math/formal-mathfin.
Copyright (c) 2026 Raphael Coelho. Released under Apache 2.0 license as
described in https://www.apache.org/licenses/LICENSE-2.0.
-/

import Transformer.BatchSize.Section4_GaussianCDF

open MeasureTheory ProbabilityTheory

noncomputable section

namespace Transformer.BatchSize

/-- Density representation of the Gaussian expectations used in
Section 4.3, equations (2)--(3). The variance here is one. -/
theorem standardGaussian_integral_density (g : ℝ → ℝ) :
    (∫ z, g z ∂gaussianReal 0 1) = ∫ z, g z * gaussianPDFReal 0 1 z := by
  rw [integral_gaussianReal_eq_integral_smul (by norm_num : (1 : NNReal) ≠ 0)]
  simp only [smul_eq_mul, mul_comm]

/-- Transfer integrability to the Gaussian density, Section 4.3,
equations (2)--(3); needed for improper integration by parts. -/
theorem standardGaussian_integrable_density {g : ℝ → ℝ}
    (hg : Integrable g (gaussianReal 0 1)) :
    Integrable (fun z => g z * gaussianPDFReal 0 1 z) volume := by
  rw [gaussianReal_of_var_ne_zero 0 (by norm_num : (1 : NNReal) ≠ 0),
    integrable_withDensity_iff_integrable_smul' (by fun_prop)
      (ae_of_all _ fun z => gaussianPDF_lt_top)] at hg
  refine hg.congr (ae_of_all _ fun z => ?_)
  simp only [gaussianPDF, ENNReal.toReal_ofReal (gaussianPDFReal_nonneg _ _ _),
    smul_eq_mul, mul_comm]

/-- A nonzero bounded observable satisfies the integrability hypothesis
in the density transfer for Section 4.3. -/
example : Integrable (fun _ : ℝ => (1 : ℝ)) (gaussianReal 0 1) := integrable_const 1

/-- The logarithmic derivative of the standard normal density is -z,
Section 4.3's Gaussian model. -/
theorem standardGaussian_density_hasDerivAt (z : ℝ) :
    HasDerivAt (gaussianPDFReal 0 1) (-z * gaussianPDFReal 0 1 z) z := by
  have h := (((hasDerivAt_pow 2 z).neg).div_const 2).exp.const_mul
    ((Real.sqrt (2 * Real.pi))⁻¹)
  convert h using 1 <;> simp [gaussianPDFReal_def]
  ring

/-- Gaussian integration by parts: E[Z g(Z)] = E[g'(Z)], Section 4.3,
equations (2)--(3). Both g and g' are actual bounded functions; the
improper-integral theorem discharges the boundary term through integrability. -/
theorem standardGaussian_integration_by_parts {g g' : ℝ → ℝ}
    (hg : ∀ z, HasDerivAt g (g' z) z) (hg' : Continuous g')
    {C D : ℝ} (hC : ∀ z, |g z| ≤ C) (hD : ∀ z, |g' z| ≤ D) :
    (∫ z, z * g z ∂gaussianReal 0 1) = ∫ z, g' z ∂gaussianReal 0 1 := by
  have hgc : Continuous g := continuous_iff_continuousAt.mpr fun z => (hg z).continuousAt
  have hd := integrable_gaussianPDFReal (0 : ℝ) 1
  have hzd := standardGaussian_integrable_density
    ((memLp_id_gaussianReal (μ := 0) (v := 1) 1).integrable (by norm_num))
  have hprod : Integrable (fun z => g z * gaussianPDFReal 0 1 z) volume :=
    hd.bdd_mul hgc.aestronglyMeasurable
      (ae_of_all _ fun z => by simpa only [Real.norm_eq_abs] using hC z)
  have hprod' : Integrable (fun z => g' z * gaussianPDFReal 0 1 z) volume :=
    hd.bdd_mul hg'.aestronglyMeasurable
      (ae_of_all _ fun z => by simpa only [Real.norm_eq_abs] using hD z)
  have hprodderiv : Integrable (fun z => g z * (-z * gaussianPDFReal 0 1 z)) volume := by
    simpa only [Pi.neg_apply, neg_mul, id_eq] using
      (hzd.neg.bdd_mul hgc.aestronglyMeasurable
        (ae_of_all _ fun z => by simpa only [Real.norm_eq_abs] using hC z))
  have hibp := integral_mul_deriv_eq_deriv_mul_of_integrable
    (u := g) (u' := g') (v := gaussianPDFReal 0 1)
    (v' := fun z => -z * gaussianPDFReal 0 1 z)
    (fun z _ => hg z) (fun z _ => standardGaussian_density_hasDerivAt z)
    hprodderiv hprod' hprod
  simp only [standardGaussian_integral_density]
  have heq : (fun z => g z * (-z * gaussianPDFReal 0 1 z)) =
      -(fun z => (z * g z) * gaussianPDFReal 0 1 z) := by funext z; simp; ring
  rw [heq] at hibp
  change (∫ z, -((z * g z) * gaussianPDFReal 0 1 z)) =
    -(∫ z, g' z * gaussianPDFReal 0 1 z) at hibp
  rw [integral_neg] at hibp
  linarith

/-- Joint nonvacuity of the integration-by-parts hypotheses, Section 4.3:
the nonconstant observable sin has derivative cos, both bounded by one. -/
example : (∀ z : ℝ, HasDerivAt Real.sin (Real.cos z) z) ∧ Continuous Real.cos ∧
    (∀ z : ℝ, |Real.sin z| ≤ 1) ∧ (∀ z : ℝ, |Real.cos z| ≤ 1) :=
  ⟨Real.hasDerivAt_sin, Real.continuous_cos, Real.abs_sin_le_one, Real.abs_cos_le_one⟩

end Transformer.BatchSize
