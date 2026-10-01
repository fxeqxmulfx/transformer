/-
# Local weak expansion of the Gaussian flow

arXiv:2506.12543v1, Section 4.3, equations (2)--(3).
The continuous Gaussian expectation has a second-order local defect after
subtracting its generator. This is the constant-coefficient Brownian case;
the state-dependent diffusion law of Theorem 1 is a separate proof obligation.
-/

import Transformer.BatchSize.Section4_GaussianGenerator

open MeasureTheory

noncomputable section

namespace Transformer.BatchSize

/-- A proved continuous local weak expansion, Section 4.3, equations
(2)--(3). The four derivatives are linked by HasDerivAt; none are arbitrary
Taylor coefficients. The bound is intentionally nonsharp. -/
theorem gaussianFlow_local_generator_error {f₀ f₁ f₂ f₃ f₄ : ℝ → ℝ}
    (h₀ : ∀ x, HasDerivAt f₀ (f₁ x) x) (h₁ : ∀ x, HasDerivAt f₁ (f₂ x) x)
    (h₂ : ∀ x, HasDerivAt f₂ (f₃ x) x) (h₃ : ∀ x, HasDerivAt f₃ (f₄ x) x)
    (h₄ : Continuous f₄) {C₀ C₁ C₂ C₃ C₄ : ℝ}
    (hC₀ : ∀ x, |f₀ x| ≤ C₀) (hC₁ : ∀ x, |f₁ x| ≤ C₁)
    (hC₂ : ∀ x, |f₂ x| ≤ C₂) (hC₃ : ∀ x, |f₃ x| ≤ C₃)
    (hC₄ : ∀ x, |f₄ x| ≤ C₄) {t : ℝ} (ht : 0 ≤ t) :
    |gaussianFlow f₀ t - f₀ 0 - t / 2 * f₂ 0| ≤ C₄ * t ^ 2 / 4 := by
  have h₂c : Continuous f₂ := continuous_iff_continuousAt.mpr fun x => (h₂ x).continuousAt
  have hC₄0 : 0 ≤ C₄ := (abs_nonneg (f₄ 0)).trans (hC₄ 0)
  have hflowc := continuous_gaussianFlow h₂c hC₂
  have hint : IntervalIntegrable (fun s => (1 / 2) * gaussianFlow f₂ s) volume 0 t :=
    (hflowc.const_mul (1 / 2)).intervalIntegrable 0 t
  have heq : gaussianFlow f₀ t - f₀ 0 - t / 2 * f₂ 0 =
      ∫ s in (0 : ℝ)..t, (1 / 2) * (gaussianFlow f₂ s - f₂ 0) := by
    rw [gaussianFlow_generator_identity h₀ h₁ h₂c hC₀ hC₁ hC₂ ht]
    have hi := intervalIntegral.integral_sub hint
      (intervalIntegrable_const : IntervalIntegrable (fun _ : ℝ => (1 / 2) * f₂ 0) volume 0 t)
    simp only [intervalIntegral.integral_const, sub_zero, smul_eq_mul] at hi
    simp_rw [mul_sub] at ⊢
    rw [hi]
    ring
  rw [heq]
  have hb : ∀ s ∈ Set.uIoc (0 : ℝ) t,
      ‖(1 / 2) * (gaussianFlow f₂ s - f₂ 0)‖ ≤ C₄ * t / 4 := by
    intro s hs
    rw [Set.uIoc_of_le ht] at hs
    have hd := gaussianFlow_sub_initial_le h₂ h₃ h₄ hC₂ hC₃ hC₄ hs.1.le
    have hst : C₄ * s ≤ C₄ * t := mul_le_mul_of_nonneg_left hs.2 hC₄0
    rw [Real.norm_eq_abs, abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 1 / 2)]
    linarith
  have hi := intervalIntegral.norm_integral_le_of_norm_le_const hb
  rw [Real.norm_eq_abs, sub_zero, abs_of_nonneg ht] at hi
  convert hi using 1
  ring

/-- Joint nonvacuity of the four derivative links and all bounds in the
continuous expansion, Section 4.3: the derivatives of sin cycle. -/
example : (∀ x : ℝ, HasDerivAt Real.sin (Real.cos x) x) ∧
    (∀ x : ℝ, HasDerivAt Real.cos (-Real.sin x) x) ∧
    (∀ x : ℝ, HasDerivAt (fun y => -Real.sin y) (-Real.cos x) x) ∧
    (∀ x : ℝ, HasDerivAt (fun y => -Real.cos y) (Real.sin x) x) ∧
    Continuous Real.sin ∧ (∀ x : ℝ, |Real.sin x| ≤ 1) ∧
    (∀ x : ℝ, |Real.cos x| ≤ 1) ∧ (∀ x : ℝ, |-Real.sin x| ≤ 1) ∧
    (∀ x : ℝ, |-Real.cos x| ≤ 1) ∧ (∀ x : ℝ, |Real.sin x| ≤ 1) ∧ (0 : ℝ) ≤ 1 := by
  exact ⟨Real.hasDerivAt_sin, Real.hasDerivAt_cos,
    fun x => (Real.hasDerivAt_sin x).neg,
    fun x => by
      convert (Real.hasDerivAt_cos x).neg using 1
      simp only [neg_neg],
    Real.continuous_sin, Real.abs_sin_le_one, Real.abs_cos_le_one,
    fun x => by simpa only [abs_neg] using Real.abs_sin_le_one x,
    fun x => by simpa only [abs_neg] using Real.abs_cos_le_one x,
    Real.abs_sin_le_one, by norm_num⟩

end Transformer.BatchSize
