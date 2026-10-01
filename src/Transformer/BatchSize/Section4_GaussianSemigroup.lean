/-
# Gaussian transition semigroup

arXiv:2506.12543v1, Section 4.3, equations (2)--(3).
The continuous comparison for constant Brownian coefficients is a genuine
Gaussian semigroup: sequential independent transitions add their variances.
No semigroup identity is included as an unproved premise.
-/

import Transformer.BatchSize.Section4_GaussianSmoothness

open MeasureTheory ProbabilityTheory

noncomputable section

namespace Transformer.BatchSize

/-- Initial transition is the identity, Section 4.3's continuous process. -/
theorem gaussianTransition_zero (f : ℝ → ℝ) (x : ℝ) : gaussianTransition f 0 x = f x := by
  simp [gaussianTransition, gaussianFlow_zero]

/-- Transition expectation against the actual Gaussian law, Section 4.3,
equations (2)--(3), including zero variance. -/
theorem gaussianTransition_eq_integral {f : ℝ → ℝ} (hf : Continuous f)
    (t x : ℝ) (ht : 0 ≤ t) :
    gaussianTransition f t x = ∫ z, f (x + z) ∂gaussianReal 0 t.toNNReal :=
  gaussianFlow_eq_integral (hf.comp (continuous_const.add continuous_id)) t ht

/-- Joint nonvacuity of the Gaussian transition-law hypotheses, Section 4.3. -/
example : Continuous Real.sin ∧ (0 : ℝ) ≤ 1 := ⟨Real.continuous_sin, by norm_num⟩

/-- Gaussian transitions compose by addition of elapsed times, Section 4.3,
equations (2)--(3). Fubini and the convolution law of independent Gaussians
prove this for actual bounded continuous observables. -/
theorem gaussianTransition_add {f : ℝ → ℝ} (hf : Continuous f)
    {C : ℝ} (hC : ∀ y, |f y| ≤ C) {s t : ℝ} (hs : 0 ≤ s) (ht : 0 ≤ t) (x : ℝ) :
    gaussianTransition f (s + t) x =
      gaussianTransition (gaussianTransition f t) s x := by
  have hsum : (s + t).toNNReal = s.toNNReal + t.toNNReal := by
    apply NNReal.coe_injective
    simp only [NNReal.coe_add, Real.coe_toNNReal _ hs, Real.coe_toNNReal _ ht,
      Real.coe_toNNReal _ (add_nonneg hs ht)]
  have hconv : (gaussianReal 0 s.toNNReal).conv (gaussianReal 0 t.toNNReal) =
      gaussianReal 0 (s + t).toNNReal := by
    rw [gaussianReal_conv_gaussianReal, zero_add, hsum]
  have hg : Integrable (fun y => f (x + y))
      ((gaussianReal 0 s.toNNReal).conv (gaussianReal 0 t.toNNReal)) := by
    rw [hconv]
    exact (integrable_const C).mono'
      (hf.comp (continuous_const.add continuous_id)).aestronglyMeasurable
      (ae_of_all _ fun y => by simpa only [Real.norm_eq_abs] using hC (x + y))
  rw [gaussianTransition_eq_integral hf (s + t) x (add_nonneg hs ht), ← hconv,
    integral_conv hg,
    gaussianTransition_eq_integral (continuous_gaussianTransition hf hC t) s x hs]
  apply integral_congr_ae (ae_of_all _ fun y => ?_)
  rw [gaussianTransition_eq_integral hf t (x + y) ht]
  simp only [add_assoc]

/-- Joint nonvacuity of all semigroup hypotheses, Section 4.3. -/
example : Continuous Real.sin ∧ (∀ y : ℝ, |Real.sin y| ≤ 1) ∧
    (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 2 :=
  ⟨Real.continuous_sin, Real.abs_sin_le_one, by norm_num, by norm_num⟩

end Transformer.BatchSize
