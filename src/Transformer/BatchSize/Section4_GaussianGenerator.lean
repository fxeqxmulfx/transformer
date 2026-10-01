/-
# Integrated Gaussian generator identity

arXiv:2506.12543v1, Section 4.3, equations (2)--(3).
This proves the expectation form of the scalar Brownian generator identity
from actual Gaussian probability laws, including the zero-time boundary.
It does not assume a stochastic Taylor formula or an SDE existence theorem.
-/

import Transformer.BatchSize.Section4_GaussianFlow

open MeasureTheory ProbabilityTheory

noncomputable section

namespace Transformer.BatchSize

/-- The rescaled standard normal has variance t at nonnegative time,
Section 4.3's Brownian driving noise in equations (2)--(3). -/
theorem gaussianFlow_hasLaw (t : ℝ) (ht : 0 ≤ t) :
    HasLaw (fun z => Real.sqrt t * z) (gaussianReal 0 t.toNNReal) (gaussianReal 0 1) := by
  refine ⟨(measurable_const.mul measurable_id).aemeasurable, ?_⟩
  rw [gaussianReal_map_const_mul]
  congr 1
  · simp
  · apply NNReal.coe_injective
    simp only [NNReal.coe_mk, mul_one]
    rw [Real.sq_sqrt ht, Real.coe_toNNReal t ht]

/-- Nonvacuity of the nonnegative-time rescaling, Section 4.3. -/
example : (0 : ℝ) ≤ 1 := by norm_num

/-- The flow is the actual Gaussian expectation, Section 4.3,
equations (2)--(3); it is not a separately postulated semigroup. -/
theorem gaussianFlow_eq_integral {f : ℝ → ℝ} (hf : Continuous f)
    (t : ℝ) (ht : 0 ≤ t) :
    gaussianFlow f t = ∫ z, f z ∂gaussianReal 0 t.toNNReal := by
  exact (gaussianFlow_hasLaw t ht).integral_comp hf.aestronglyMeasurable

/-- Joint nonvacuity of the flow-law identification, Section 4.3. -/
example : Continuous Real.sin ∧ (0 : ℝ) ≤ 1 := ⟨Real.continuous_sin, by norm_num⟩

/-- The integrated Brownian generator identity for bounded C2 observables,
Section 4.3, equations (2)--(3). The second derivative is the genuine
derivative of the first derivative, and all expectations use Gaussian laws. -/
theorem gaussianFlow_generator_identity {f f' f'' : ℝ → ℝ}
    (hf : ∀ x, HasDerivAt f (f' x) x) (hf' : ∀ x, HasDerivAt f' (f'' x) x)
    (hf'' : Continuous f'') {C D E : ℝ}
    (hC : ∀ x, |f x| ≤ C) (hD : ∀ x, |f' x| ≤ D) (hE : ∀ x, |f'' x| ≤ E)
    {t : ℝ} (ht : 0 ≤ t) :
    gaussianFlow f t = f 0 + ∫ s in (0 : ℝ)..t, (1 / 2) * gaussianFlow f'' s := by
  have hfc : Continuous f := continuous_iff_continuousAt.mpr fun x => (hf x).continuousAt
  have hderiv : ∀ s ∈ Set.Ioo (0 : ℝ) t,
      HasDerivWithinAt (gaussianFlow f) ((1 / 2) * gaussianFlow f'' s) (Set.Ioi s) s :=
    fun s hs => (gaussianFlow_hasDerivAt hf hf' hf'' hC hD hE hs.1).hasDerivWithinAt
  have hint : IntervalIntegrable (fun s => (1 / 2) * gaussianFlow f'' s) volume 0 t :=
    ((continuous_gaussianFlow hf'' hE).const_mul (1 / 2)).intervalIntegrable 0 t
  have hFTC := intervalIntegral.integral_eq_sub_of_hasDeriv_right_of_le ht
    (continuous_gaussianFlow hfc hC).continuousOn hderiv hint
  rw [gaussianFlow_zero] at hFTC
  linarith

/-- Joint nonvacuity of the integrated-generator hypotheses, Section 4.3. -/
example : (∀ x : ℝ, HasDerivAt Real.sin (Real.cos x) x) ∧
    (∀ x : ℝ, HasDerivAt Real.cos (-Real.sin x) x) ∧ Continuous (fun x => -Real.sin x) ∧
    (∀ x : ℝ, |Real.sin x| ≤ 1) ∧ (∀ x : ℝ, |Real.cos x| ≤ 1) ∧
    (∀ x : ℝ, |-Real.sin x| ≤ 1) ∧ (0 : ℝ) ≤ 1 :=
  ⟨Real.hasDerivAt_sin, Real.hasDerivAt_cos, Real.continuous_sin.neg,
    Real.abs_sin_le_one, Real.abs_cos_le_one,
    fun x => by simpa only [abs_neg] using Real.abs_sin_le_one x, by norm_num⟩

/-- Linear-in-time expectation change from the continuous generator,
Section 4.3, equations (2)--(3). -/
theorem gaussianFlow_sub_initial_le {f f' f'' : ℝ → ℝ}
    (hf : ∀ x, HasDerivAt f (f' x) x) (hf' : ∀ x, HasDerivAt f' (f'' x) x)
    (hf'' : Continuous f'') {C D E : ℝ}
    (hC : ∀ x, |f x| ≤ C) (hD : ∀ x, |f' x| ≤ D) (hE : ∀ x, |f'' x| ≤ E)
    {t : ℝ} (ht : 0 ≤ t) : |gaussianFlow f t - f 0| ≤ E * t / 2 := by
  rw [gaussianFlow_generator_identity hf hf' hf'' hC hD hE ht, add_sub_cancel_left]
  have hb : ∀ s ∈ Set.uIoc (0 : ℝ) t,
      ‖(1 / 2) * gaussianFlow f'' s‖ ≤ E / 2 := by
    intro s _
    rw [Real.norm_eq_abs, abs_mul]
    norm_num only [abs_of_pos (by norm_num : (0 : ℝ) < 1 / 2)]
    have h := gaussianFlow_abs_le hE s
    linarith
  have hi := intervalIntegral.norm_integral_le_of_norm_le_const hb
  simpa only [Real.norm_eq_abs, sub_zero, abs_of_nonneg ht, div_mul_eq_mul_div] using hi

/-- Joint nonvacuity of the continuous expectation bound, Section 4.3. -/
example : (∀ x : ℝ, HasDerivAt Real.sin (Real.cos x) x) ∧
    (∀ x : ℝ, HasDerivAt Real.cos (-Real.sin x) x) ∧ Continuous (fun x => -Real.sin x) ∧
    (∀ x : ℝ, |Real.sin x| ≤ 1) ∧ (∀ x : ℝ, |Real.cos x| ≤ 1) ∧
    (∀ x : ℝ, |-Real.sin x| ≤ 1) ∧ (0 : ℝ) ≤ 1 :=
  ⟨Real.hasDerivAt_sin, Real.hasDerivAt_cos, Real.continuous_sin.neg,
    Real.abs_sin_le_one, Real.abs_cos_le_one,
    fun x => by simpa only [abs_neg] using Real.abs_sin_le_one x, by norm_num⟩

end Transformer.BatchSize
