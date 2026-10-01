/-
# Higher moments of the constructed Brownian increments

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The fourth-moment identity supplies the Kolmogorov continuity estimate
for limits of finite Brownian Euler approximations.
-/

import Transformer.BatchSize.Section4_BrownianConditional

open MeasureTheory ProbabilityTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- Brownian increments have finite fourth moment, Section 4.3 (2)--(3). -/
theorem brownianIncrement_memLp_four {d : ℕ} (k : Fin d) (s t : ℝ≥0) :
    MemLp (brownianIncrement k s t) 4 (brownianNoiseLaw d) :=
  (coordinateBrownian_isBrownian k).isGaussianProcess.hasGaussianLaw_sub.memLp (by norm_num)

/-- The third moment of a Brownian increment vanishes, Section 4.3
(2)--(3). Reversing the endpoints preserves its centered Gaussian law. -/
theorem brownianIncrement_thirdMoment {d : ℕ} (k : Fin d) (s t : ℝ≥0) :
    (∫ ω, brownianIncrement k s t ω ^ 3 ∂brownianNoiseLaw d) = 0 := by
  have hf := (brownianIncrement_hasLaw k s t).integral_comp (continuous_pow 3).aestronglyMeasurable
  have hr := (brownianIncrement_hasLaw k t s).integral_comp (continuous_pow 3).aestronglyMeasurable
  rw [nndist_comm (s : ℝ) (t : ℝ)] at hr
  have heq (ω : BrownianSample d) : brownianIncrement k t s ω ^ 3 =
      -(brownianIncrement k s t ω ^ 3) := by
    dsimp [brownianIncrement]
    ring
  simp only [Function.comp_def] at hf hr
  simp_rw [heq] at hr
  rw [integral_neg] at hr
  linarith

/-- The centered Gaussian fourth moment is three times variance
squared, used for the driver of Section 4.3 (2)--(3). -/
theorem gaussianReal_fourthMoment (v : ℝ≥0) :
    (∫ z : ℝ, z ^ 4 ∂gaussianReal 0 v) = 3 * (v : ℝ) ^ 2 := by
  have h := centralMoment_fun_two_mul_gaussianReal 0 (NNReal.sqrt v) 2
  rw [NNReal.sq_sqrt] at h
  simp only [centralMoment, integral_id_gaussianReal, Pi.pow_apply, Pi.sub_apply, sub_zero] at h
  norm_num [Nat.doubleFactorial] at h
  rw [show Real.sqrt (v : ℝ) ^ 4 = (Real.sqrt (v : ℝ) ^ 2) ^ 2 by ring,
    Real.sq_sqrt v.coe_nonneg] at h
  simpa only [mul_comm] using h

/-- Exact fourth moment of a forward Brownian increment,
Section 4.3 (2)--(3). -/
theorem brownianIncrement_fourthMoment {d : ℕ} (k : Fin d) (s t : ℝ≥0) (hst : s ≤ t) :
    (∫ ω, brownianIncrement k s t ω ^ 4 ∂brownianNoiseLaw d) = 3 * ((t : ℝ) - s) ^ 2 := by
  have h := (brownianIncrement_hasLaw k s t).integral_comp (continuous_pow 4).aestronglyMeasurable
  rw [gaussianReal_fourthMoment] at h
  simpa only [Function.comp_def, coe_nndist, Real.dist_eq,
    abs_of_nonneg (sub_nonneg.mpr (NNReal.coe_le_coe.mpr hst))] using h

/-- The third moment kernel vanishes even for an arbitrary integrable
coefficient measurable in the joint past, Section 4.3 (2)--(3). -/
theorem brownianIncrement_adapted_thirdMoment {d : ℕ} (k : Fin d) (s t : ℝ≥0) (hst : s ≤ t)
    (H : BrownianSample d → ℝ) (hH : StronglyMeasurable[brownianFiltration d s] H)
    (hHint : Integrable H (brownianNoiseLaw d)) :
    (∫ ω, H ω * brownianIncrement k s t ω ^ 3 ∂brownianNoiseLaw d) = 0 := by
  have hi := (brownianIncrement_independent_adapted k s t hst hH).comp
    (φ := id) (ψ := fun z : ℝ => z ^ 3) measurable_id (continuous_pow 3).measurable
  simp only [Function.comp_def, id_eq] at hi
  rw [hi.integral_fun_mul_eq_mul_integral hHint.aestronglyMeasurable
    ((continuous_pow 3).comp_aestronglyMeasurable (brownianIncrement_memLp k s t).aestronglyMeasurable),
    brownianIncrement_thirdMoment, mul_zero]

/-- Exact adapted fourth-moment kernel, Section 4.3 (2)--(3),
retaining dependence of the coefficient on the entire joint past. -/
theorem brownianIncrement_adapted_fourthMoment {d : ℕ} (k : Fin d) (s t : ℝ≥0) (hst : s ≤ t)
    (H : BrownianSample d → ℝ) (hH : StronglyMeasurable[brownianFiltration d s] H)
    (hHint : Integrable H (brownianNoiseLaw d)) :
    (∫ ω, H ω * brownianIncrement k s t ω ^ 4 ∂brownianNoiseLaw d) =
      (∫ ω, H ω ∂brownianNoiseLaw d) * (3 * ((t : ℝ) - s) ^ 2) := by
  have hi := (brownianIncrement_independent_adapted k s t hst hH).comp
    (φ := id) (ψ := fun z : ℝ => z ^ 4) measurable_id (continuous_pow 4).measurable
  simp only [Function.comp_def, id_eq] at hi
  rw [hi.integral_fun_mul_eq_mul_integral hHint.aestronglyMeasurable
    ((continuous_pow 4).comp_aestronglyMeasurable (brownianIncrement_memLp k s t).aestronglyMeasurable),
    brownianIncrement_fourthMoment k s t hst]

/-- Joint nonvacuity of all higher-moment kernel hypotheses,
Section 4.3: the coefficient is a past Brownian coordinate. -/
example : (1 : ℝ≥0) ≤ 2 ∧
    StronglyMeasurable[brownianFiltration 2 1] (coordinateBrownian (1 : Fin 2) 1) ∧
    Integrable (coordinateBrownian (1 : Fin 2) 1) (brownianNoiseLaw 2) :=
  ⟨by norm_num, (coordinateBrownian_filtered (1 : Fin 2)).stronglyAdapted 1,
    (coordinateBrownian_isBrownian (1 : Fin 2)).toIsPreBrownianReal.integrable_eval 1⟩

end Transformer.BatchSize
