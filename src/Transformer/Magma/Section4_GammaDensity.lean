/-
# Exponential moments of the paper's Gamma benchmark

Audit of arXiv:2602.15322v1, Appendix B, Heavy-Tailed Gradient Noise
Benchmark Setup. Mathlib uses shape and RATE; the paper uses shape and
SCALE. Thus shape 0.1 and scale 10 mean shape=rate=0.1 below.
The Gamma variable has positive exponential moments. Calling it
heavy-tailed in the strict exponential-moment sense is incorrect.
-/

import Mathlib.Probability.Distributions.Gamma
import Mathlib.Tactic

open MeasureTheory ProbabilityTheory

noncomputable section

namespace Transformer.Magma

/-- The real Gamma density is integrable for positive shape and rate.
Source: arXiv:2602.15322v1, Appendix B, Gamma covariate construction. -/
theorem gamma_density_integrable (a r : ℝ) (ha : 0 < a) (hr : 0 < r) :
    Integrable (gammaPDFReal a r) volume := by
  apply (lintegral_ofReal_ne_top_iff_integrable
    (stronglyMeasurable_gammaPDFReal a r).aestronglyMeasurable
    (Filter.Eventually.of_forall (gammaPDFReal_nonneg ha hr))).mp
  have hpdf : (fun x => ENNReal.ofReal (gammaPDFReal a r x)) = gammaPDF a r := by
    funext x
    exact (gammaPDF_eq a r x).symm
  rw [hpdf, lintegral_gammaPDF_eq_one ha hr]
  norm_num

/-- Positive shape and rate have a joint witness. Source:
arXiv:2602.15322v1, Appendix B, Gamma construction. -/
example : (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 10 := by norm_num

/-- Exponential tilting lowers the rate while preserving the Gamma
density up to a finite constant. Source: arXiv:2602.15322v1, Appendix B,
correction to the description of Gamma variables as heavy-tailed. -/
theorem gamma_exponential_integrable (a r t : ℝ) (ha : 0 < a) (hr : 0 < r) (ht : t < r) :
    Integrable (fun x => Real.exp (t * |x|)) (gammaMeasure a r) := by
  have hrate : 0 < r - t := sub_pos.mpr ht
  have hden : (r - t) ^ a ≠ 0 := (Real.rpow_pos_of_pos hrate a).ne'
  have htilt (x : ℝ) : Real.exp (t * |x|) * gammaPDFReal a r x =
      (r ^ a / (r - t) ^ a) * gammaPDFReal a (r - t) x := by
    by_cases hx : 0 ≤ x
    · simp only [gammaPDFReal, ite_eq_left hx, abs_of_nonneg hx]
      have he : Real.exp (t * x) * Real.exp (-(r * x)) = Real.exp (-((r - t) * x)) := by
        rw [← Real.exp_add]
        congr 1
        ring
      calc
        _ = (r ^ a / Real.Gamma a * x ^ (a - 1)) *
            (Real.exp (t * x) * Real.exp (-(r * x))) := by ring
        _ = _ := by
          rw [he]
          field_simp [hden]
    · simp [gammaPDFReal, hx]
  have hweighted : Integrable (fun x => Real.exp (t * |x|) * gammaPDFReal a r x) volume := by
    convert (gamma_density_integrable a (r - t) ha hrate).const_mul
      (r ^ a / (r - t) ^ a) using 1
    funext x
    exact htilt x
  have hpdf : gammaPDF a r = fun x => ENNReal.ofReal (gammaPDFReal a r x) := by
    funext x
    exact gammaPDF_eq a r x
  have hm : Measurable (gammaPDF a r) := by
    rw [hpdf]
    exact (measurable_gammaPDFReal a r).ennreal_ofReal
  have hfinite : ∀ᵐ x ∂volume, gammaPDF a r x < ⊤ := by
    rw [hpdf]
    exact Filter.Eventually.of_forall (fun _ => ENNReal.ofReal_lt_top)
  change Integrable _ (volume.withDensity (gammaPDF a r))
  apply (integrable_withDensity_iff hm hfinite).mpr
  simpa only [hpdf, ENNReal.toReal_ofReal (gammaPDFReal_nonneg ha hr _)] using hweighted

/-- All tilting hypotheses are jointly satisfied by the paper's shape
and rate and a positive exponent 0.05. Source:
arXiv:2602.15322v1, Appendix B, Gamma benchmark correction. -/
example : (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 10 ∧ (1 / 20 : ℝ) < 1 / 10 := by
  norm_num

end Transformer.Magma
