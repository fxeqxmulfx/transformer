/-
# Local diffusion error against the actual deterministic Euler step

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The constructed diffusion and deterministic mean Euler step have the
same first-order drift. Their proved weak Taylor defects give O(eta^2).
-/

import Transformer.BatchSize.Section4_OptimizerLocalWeakDrift
import Transformer.BatchSize.Section4_C2DiscreteError

open MeasureTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The actual optimizer diffusion differs from a deterministic
mean Euler step applied to its initial random state by C*R*eta^2,
Section 4.3 (2)--(3), Theorem 1. This holds at every interval start
and for every bounded C2 observable, without a Markov-property premise. -/
theorem optimizerEulerPath_deterministic_local_error {d : ℕ} (method : UpdateKind) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hB : 0 < B) (hmodel : RegularGaussianModel f σ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ (η : NNReal) (hη : 0 < (η : ℝ)) (R : NNReal), 0 < R →
      ∀ φ : EucSpace d → ℝ, ContDiff ℝ 2 φ →
      (∀ j ≤ 2, ∀ x, ‖iteratedFDeriv ℝ j φ x‖ ≤ R) → ∀ (x₀ : EucSpace d) (s : NNReal),
      |(∫ ω, φ (optimizerEulerPath method η B f σ hη.le hB hmodel x₀ ω (s + η)) ∂brownianNoiseLaw d) -
        (∫ ω, φ (deterministicEulerMap (diffusionDrift method B f σ) η
          (optimizerEulerPath method η B f σ hη.le hB hmodel x₀ ω s)) ∂brownianNoiseLaw d)| ≤
        C * R * (η : ℝ) ^ 2 := by
  obtain ⟨C, hC, hlocal⟩ := optimizerEulerPath_local_weak_drift method B f σ hB hmodel
  obtain ⟨M, hM⟩ := diffusionDrift_uniform_bound method B f σ hmodel
  refine ⟨C + (M : ℝ) ^ 2, by positivity, ?_⟩
  intro η hη R hR φ hφ hbound x₀ s
  let b := diffusionDrift method B f σ
  let Z := optimizerEulerPath method η B f σ hη.le hB hmodel x₀
  have hZ := optimizerEulerPath_measurable method η B f σ hη.le hB hmodel x₀
  have hb : ContDiff ℝ 2 b :=
    (regularGaussianModel_smooth_coefficients method 0 B f σ hmodel).1.of_le (by norm_num)
  have hval (x : EucSpace d) : |φ x| ≤ R := by
    simpa only [norm_iteratedFDeriv_zero, Real.norm_eq_abs] using hbound 0 (by norm_num) x
  have hd (x : EucSpace d) : ‖fderiv ℝ φ x‖ ≤ R := by
    rw [← norm_iteratedFDeriv_one φ]; exact hbound 1 (by norm_num) x
  obtain ⟨K, hK⟩ := diffusionDrift_lipschitz method B f σ hmodel
  have hq := driftTest_lipschitz b φ R M K hφ hK hM hd (hbound 2 le_rfl)
  have hiq := continuousPath_observable_integrable (brownianNoiseLaw d) Z hZ
    (driftTest b φ) hq.continuous ((R : ℝ) * M) (driftTest_abs_le b φ R M hd hM) s
  have hiφ := continuousPath_observable_integrable (brownianNoiseLaw d) Z hZ φ hφ.continuous R hval s
  have hiDet := continuousPath_observable_integrable (brownianNoiseLaw d) Z hZ
    (fun x => φ (deterministicEulerMap b η x))
    (hφ.continuous.comp (deterministicEulerMap_contDiff b η hb).continuous) R
    (fun x => hval (deterministicEulerMap b η x)) s
  have hiDiff : Integrable (fun ω => φ (deterministicEulerMap b η (Z ω s)) - φ (Z ω s))
      (brownianNoiseLaw d) := hiDet.sub hiφ
  let Et := ∫ ω, φ (Z ω (s + η)) ∂brownianNoiseLaw d
  let Es := ∫ ω, φ (Z ω s) ∂brownianNoiseLaw d
  let Ed := ∫ ω, φ (deterministicEulerMap b η (Z ω s)) ∂brownianNoiseLaw d
  let Q := ∫ ω, driftTest b φ (Z ω s) ∂brownianNoiseLaw d
  have hdet : |Ed - Es - (η : ℝ) * Q| ≤ (R : ℝ) * (M : ℝ) ^ 2 * (η : ℝ) ^ 2 := by
    have hpoint (ω : BrownianSample d) :
        |φ (deterministicEulerMap b η (Z ω s)) - φ (Z ω s) -
          (η : ℝ) * driftTest b φ (Z ω s)| ≤ (R : ℝ) * (M : ℝ) ^ 2 * (η : ℝ) ^ 2 :=
      deterministicEulerMap_taylor_bound b η R M R.coe_nonneg hM φ hφ (hbound 2 le_rfl) _
    have h := norm_integral_le_of_norm_le_const (f := fun ω =>
      φ (deterministicEulerMap b η (Z ω s)) - φ (Z ω s) - (η : ℝ) * driftTest b φ (Z ω s))
      (ae_of_all (brownianNoiseLaw d) (fun ω => by simpa only [Real.norm_eq_abs] using hpoint ω))
    rw [integral_sub hiDiff (hiq.const_mul _), integral_sub hiDet hiφ, integral_const_mul] at h
    simpa only [Real.norm_eq_abs, probReal_univ, mul_one] using h
  have ht : |Et - Es - (η : ℝ) * Q| ≤ C * R * (η : ℝ) ^ 2 :=
    hlocal η hη R hR φ hφ hbound x₀ s
  calc
    |Et - Ed| = |(Et - Es - (η : ℝ) * Q) - (Ed - Es - (η : ℝ) * Q)| := by congr 1; ring
    _ ≤ |Et - Es - (η : ℝ) * Q| + |Ed - Es - (η : ℝ) * Q| := abs_sub _ _
    _ ≤ C * R * (η : ℝ) ^ 2 + (R : ℝ) * (M : ℝ) ^ 2 * (η : ℝ) ^ 2 := add_le_add ht hdet
    _ = _ := by ring

/-- Joint nonvacuity of all diffusion/Euler local hypotheses,
Section 4.3: flat regular model, positive rate and batch, nonzero C2 test. -/
example : 0 < (1 : ℕ) ∧ RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) ∧
    (0 : ℝ) < ((1 / 1000 : NNReal) : ℝ) ∧ (0 : NNReal) < 2 ∧
    ContDiff ℝ 2 (fun _ : EucSpace 1 => (1 : ℝ)) ∧
    (∀ j ≤ 2, ∀ x : EucSpace 1, ‖iteratedFDeriv ℝ j (fun _ => (1 : ℝ)) x‖ ≤ (2 : NNReal)) := by
  refine ⟨by norm_num, regularGaussianModel_flat 1, by norm_num, by norm_num, contDiff_const, ?_⟩
  intro j hj x
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
