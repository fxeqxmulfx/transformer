/-
# The square-root Gamma covariates are not heavy-tailed in the strict sense

Correction of arXiv:2602.15322v1, Appendix B, Heavy-Tailed Gradient Noise
Benchmark Setup. The stated radius sqrt(Gamma(0.1, scale 10)) has a finite
positive exponential moment. This refutes the strict heavy-tail label for
the radius; it does not refute the reported optimization observations or
claim that arbitrary nonlinear gradient functions must be light-tailed.
-/

import Transformer.Magma.Section4_GammaDensity

open MeasureTheory ProbabilityTheory

noncomputable section

namespace Transformer.Magma

/-- The conventional exponential-moment definition of a heavy-tailed
absolute observable. This is a property of the supplied law and observable,
not an assumption about optimization. Source: clarification of terminology
in arXiv:2602.15322v1, Appendix B, Gamma covariate construction. -/
def HeavyTailedExponential (law : Measure ℝ) (observable : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, 0 < t → ¬ Integrable (fun x => Real.exp (t * |observable x|)) law

/-- The paper's Gamma law has a positive exponential moment 0.05.
Source: arXiv:2602.15322v1, Appendix B, shape 0.1 and scale 10
(equivalently rate 0.1), correction to the heavy-tail label. -/
theorem benchmark_gamma_exponential_moment :
    Integrable (fun x => Real.exp ((1 / 20 : ℝ) * |x|))
      (gammaMeasure (1 / 10) (1 / 10)) :=
  gamma_exponential_integrable _ _ _ (by norm_num) (by norm_num) (by norm_num)

/-- An explicit quadratic upper bound on sqrt(x) controls the radius's
exponential moment. Source: arXiv:2602.15322v1, Appendix B, square-root
Gamma radius, correction to the heavy-tail claim. -/
theorem benchmark_sqrt_exponential_bound (x : ℝ) :
    Real.sqrt x ≤ 5 + (1 / 20 : ℝ) * |x| := by
  by_cases hx : 0 ≤ x
  · rw [abs_of_nonneg hx]
    have hs := Real.sq_sqrt hx
    nlinarith [sq_nonneg (Real.sqrt x - 10)]
  · rw [Real.sqrt_eq_zero_of_nonpos (le_of_not_ge hx)]
    nlinarith [abs_nonneg x]

/-- The actual square-root Gamma radius has a finite exponential moment
at positive exponent 1. Source: arXiv:2602.15322v1, Appendix B,
correction to calling sqrt(Gamma(0.1,10)) a heavy-tailed random variable. -/
theorem benchmark_sqrt_gamma_exponential_moment :
    Integrable (fun x => Real.exp |Real.sqrt x|) (gammaMeasure (1 / 10) (1 / 10)) := by
  have hm : AEStronglyMeasurable (fun x => Real.exp |Real.sqrt x|)
      (gammaMeasure (1 / 10) (1 / 10)) :=
    (Real.continuous_exp.comp Real.continuous_sqrt.abs).measurable.aestronglyMeasurable
  apply (benchmark_gamma_exponential_moment.const_mul (Real.exp 5)).mono' hm
  exact Filter.Eventually.of_forall (fun x => by
    rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _), abs_of_nonneg (Real.sqrt_nonneg _)]
    have h := Real.exp_le_exp.mpr (benchmark_sqrt_exponential_bound x)
    simpa only [Real.exp_add] using h)

/-- The square-root Gamma radius in the paper is not heavy-tailed under
the strict exponential-moment definition. Large finite kurtosis and
empirical gradient instability are distinct questions. Source:
arXiv:2602.15322v1, Appendix B, Heavy-Tailed Gradient Noise Benchmark Setup. -/
theorem benchmark_radius_not_heavy_tailed :
    ¬ HeavyTailedExponential (gammaMeasure (1 / 10) (1 / 10)) Real.sqrt := by
  intro h
  have hnon := h 1 (by norm_num)
  exact hnon (by simpa using benchmark_sqrt_gamma_exponential_moment)

end Transformer.Magma
