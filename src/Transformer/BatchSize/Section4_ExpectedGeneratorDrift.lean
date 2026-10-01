/-
# Expected generator variation within one optimizer interval

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The covariance correction and actual path increments give a uniform
O(eta) difference from the first-order generator at the interval's start.
-/

import Transformer.BatchSize.Section4_C2DriftObservable
import Transformer.BatchSize.Section4_C2GeneratorBounds
import Transformer.BatchSize.Section4_OptimizerSmallIncrements
import Transformer.BatchSize.Section4_PathExpectationBounds

open MeasureTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- Over any interval of length at most eta, the actual optimizer's
expected generator differs by C*R*eta from its initial expected
drift observable, Section 4.3 (2)--(3), Theorem 1. The model constant
precedes eta, the test, initial state and observation times. -/
theorem optimizerEulerPath_expected_generator_drift_error {d : ℕ}
    (method : UpdateKind) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hB : 0 < B) (hmodel : RegularGaussianModel f σ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ (η : ℝ) (hη : 0 < η) (R : NNReal), 0 < R →
      ∀ φ : EucSpace d → ℝ, ContDiff ℝ 2 φ →
      (∀ j ≤ 2, ∀ x, ‖iteratedFDeriv ℝ j φ x‖ ≤ R) →
      ∀ (x₀ : EucSpace d) (s t : NNReal), s ≤ t → (t : ℝ) - s ≤ η →
      |(∫ ω, diffusionGenerator method η B f σ φ
        (optimizerEulerPath method η B f σ hη.le hB hmodel x₀ ω t) ∂brownianNoiseLaw d) -
        (∫ ω, driftTest (diffusionDrift method B f σ) φ
          (optimizerEulerPath method η B f σ hη.le hB hmodel x₀ ω s) ∂brownianNoiseLaw d)| ≤
        C * R * η := by
  obtain ⟨U, hU, hcorr⟩ := diffusionGenerator_C2_correction_bound method B f σ hB hmodel
  obtain ⟨K, hK⟩ := diffusionDrift_lipschitz method B f σ hmodel
  obtain ⟨M, hM⟩ := diffusionDrift_uniform_bound method B f σ hmodel
  obtain ⟨D, hD, hinc⟩ := optimizerEulerPath_small_increment_bound method B f σ hB hmodel
  refine ⟨U + ((M : ℝ) + K) * D, by positivity, ?_⟩
  intro η hη R hR φ hφ hbound x₀ s t hst hlen
  let b := diffusionDrift method B f σ
  let Z := optimizerEulerPath method η B f σ hη.le hB hmodel x₀
  have hZ := optimizerEulerPath_measurable method η B f σ hη.le hB hmodel x₀
  have hRreal : 0 < (R : ℝ) := by exact_mod_cast hR
  have hd (x : EucSpace d) : ‖fderiv ℝ φ x‖ ≤ R := by
    rw [← norm_iteratedFDeriv_one φ]; exact hbound 1 (by norm_num) x
  have hq := driftTest_lipschitz b φ R M K hφ hK hM hd (hbound 2 le_rfl)
  have hqb := driftTest_abs_le b φ R M hd hM
  obtain ⟨hgc, G, hG, hgb⟩ := diffusionGenerator_scaled_continuous_bounded method η B f σ
    hη.le hB hmodel φ R hRreal hφ hbound
  have hgi := continuousPath_observable_integrable (brownianNoiseLaw d) Z hZ
    (diffusionGenerator method η B f σ φ) hgc G hgb t
  have hqi := continuousPath_observable_integrable (brownianNoiseLaw d) Z hZ
    (driftTest b φ) hq.continuous ((R : ℝ) * M) hqb s
  obtain ⟨hiInc, hbInc⟩ := hinc η hη x₀ s t hst hlen
  have hiUp : Integrable (fun ω => U * R * η +
      (R : ℝ) * ((M : ℝ) + K) * ‖Z ω t - Z ω s‖) (brownianNoiseLaw d) :=
    (integrable_const _).add (hiInc.const_mul _)
  have hpoint (ω : BrownianSample d) :
      |diffusionGenerator method η B f σ φ (Z ω t) - driftTest b φ (Z ω s)| ≤
        U * R * η + (R : ℝ) * ((M : ℝ) + K) * ‖Z ω t - Z ω s‖ := by
    refine (abs_sub_le _ (driftTest b φ (Z ω t)) _).trans (add_le_add
      (hcorr η hη.le R R.coe_nonneg φ (Z ω t) (hbound 2 le_rfl _)) ?_)
    simpa only [Real.norm_eq_abs, NNReal.coe_mul, NNReal.coe_add] using hq.norm_sub_le (Z ω t) (Z ω s)
  rw [← integral_sub hgi hqi]
  have h := norm_integral_le_of_norm_le (f := fun ω =>
    diffusionGenerator method η B f σ φ (Z ω t) - driftTest b φ (Z ω s)) hiUp
    (ae_of_all _ (fun ω => by simpa only [Real.norm_eq_abs] using hpoint ω))
  rw [integral_add (integrable_const _) (hiInc.const_mul _)] at h
  simp only [integral_const_mul, integral_const, probReal_univ, one_smul, Real.norm_eq_abs] at h
  refine h.trans ?_
  calc
    _ ≤ U * R * η + (R : ℝ) * ((M : ℝ) + K) * (D * η) :=
      add_le_add le_rfl (mul_le_mul_of_nonneg_left hbInc (by positivity))
    _ = _ := by ring

/-- Joint nonvacuity of the expected-generator hypotheses,
Section 4.3: flat regular model, unit noise, nonzero C2 test,
positive step and an interval of that length. -/
example : 0 < (1 : ℕ) ∧ RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) ∧ (0 : ℝ) < 1 / 1000 ∧
    (0 : NNReal) < 2 ∧ ContDiff ℝ 2 (fun _ : EucSpace 1 => (1 : ℝ)) ∧
    (∀ j ≤ 2, ∀ x : EucSpace 1, ‖iteratedFDeriv ℝ j (fun _ => (1 : ℝ)) x‖ ≤ (2 : NNReal)) ∧
    (0 : NNReal) ≤ 1 / 1000 ∧ ((1 / 1000 : NNReal) : ℝ) - 0 ≤ 1 / 1000 := by
  refine ⟨by norm_num, regularGaussianModel_flat 1, by norm_num, by norm_num,
    contDiff_const, ?_, by norm_num, by norm_num⟩
  intro j hj x
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
