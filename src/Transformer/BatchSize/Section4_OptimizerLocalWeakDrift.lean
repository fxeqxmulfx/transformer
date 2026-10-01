/-
# Local weak drift error for the constructed optimizer diffusion

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Integration of the genuine generator identity and the proved expected
generator variation yields an O(eta^2) local weak drift defect.
-/

import Transformer.BatchSize.Section4_ExpectedGeneratorDrift

open MeasureTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The true continuous Brownian Euler limit has a uniformly
second-order weak defect from its initial mean drift on every bounded
C2 test, Section 4.3 (2)--(3), Theorem 1. The constant is chosen before
the rate, observable, initial state and interval start. -/
theorem optimizerEulerPath_local_weak_drift {d : ℕ} (method : UpdateKind) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hB : 0 < B) (hmodel : RegularGaussianModel f σ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ (η : NNReal) (hη : 0 < (η : ℝ)) (R : NNReal), 0 < R →
      ∀ φ : EucSpace d → ℝ, ContDiff ℝ 2 φ →
      (∀ j ≤ 2, ∀ x, ‖iteratedFDeriv ℝ j φ x‖ ≤ R) → ∀ (x₀ : EucSpace d) (s : NNReal),
      |(∫ ω, φ (optimizerEulerPath method η B f σ hη.le hB hmodel x₀ ω (s + η)) ∂brownianNoiseLaw d) -
        (∫ ω, φ (optimizerEulerPath method η B f σ hη.le hB hmodel x₀ ω s) ∂brownianNoiseLaw d) -
        (η : ℝ) * (∫ ω, driftTest (diffusionDrift method B f σ) φ
          (optimizerEulerPath method η B f σ hη.le hB hmodel x₀ ω s) ∂brownianNoiseLaw d)| ≤
        C * R * (η : ℝ) ^ 2 := by
  obtain ⟨C, hC, hvar⟩ := optimizerEulerPath_expected_generator_drift_error method B f σ hB hmodel
  refine ⟨C, hC, ?_⟩
  intro η hη R hR φ hφ hbound x₀ s
  let Z := optimizerEulerPath method η B f σ hη.le hB hmodel x₀
  let t : NNReal := s + η
  let G := fun u : ℝ => ∫ ω, diffusionGenerator method η B f σ φ (Z ω u) ∂brownianNoiseLaw d
  let Q := ∫ ω, driftTest (diffusionDrift method B f σ) φ (Z ω s) ∂brownianNoiseLaw d
  have hst : s ≤ t := le_add_of_nonneg_right zero_le
  have hstR : (s : ℝ) ≤ t := NNReal.coe_le_coe.mpr hst
  have hlen : (t : ℝ) - s = η := by simp only [t, NNReal.coe_add]; ring
  have hRreal : 0 < (R : ℝ) := by exact_mod_cast hR
  obtain ⟨hg, L, hL, hb⟩ := diffusionGenerator_scaled_continuous_bounded method η B f σ
    hη.le hB hmodel φ R hRreal hφ hbound
  have hiG : IntervalIntegrable G volume s t :=
    continuousPath_expectation_intervalIntegrable (brownianNoiseLaw d) Z
      (optimizerEulerPath_measurable method η B f σ hη.le hB hmodel x₀)
      (diffusionGenerator method η B f σ φ) hg L hb s t
  have hid := optimizerEulerPath_scaled_generator_identity method η B f σ hη.le hB hmodel
    x₀ s t hst φ R hRreal hφ hbound
  change (∫ ω, φ (Z ω t) ∂brownianNoiseLaw d) -
    (∫ ω, φ (Z ω s) ∂brownianNoiseLaw d) = ∫ u in (s : ℝ)..(t : ℝ), G u at hid
  have hpoint (u : ℝ) (hu : u ∈ Set.uIoc (s : ℝ) (t : ℝ)) :
      ‖G u - Q‖ ≤ C * R * η := by
    rw [Set.uIoc_of_le hstR] at hu
    let v : NNReal := ⟨u, s.coe_nonneg.trans hu.1.le⟩
    have hsv : s ≤ v := by exact_mod_cast hu.1.le
    have hδ : (v : ℝ) - s ≤ η := by
      change u - s ≤ η
      linarith [hu.2]
    have h := hvar η hη R hR φ hφ hbound x₀ s v hsv hδ
    rw [Real.norm_eq_abs]
    exact h
  have h := intervalIntegral.norm_integral_le_of_norm_le_const
    (f := fun u => G u - Q) hpoint
  rw [intervalIntegral.integral_sub hiG intervalIntegrable_const,
    intervalIntegral.integral_const, smul_eq_mul, hlen, abs_of_nonneg η.coe_nonneg, ← hid] at h
  simp only [Real.norm_eq_abs] at h
  calc
    _ ≤ C * R * η * η := h
    _ = _ := by ring

/-- Joint nonvacuity of the diffusion local drift assumptions,
Section 4.3: positive rate and batch, flat regular model and nonzero C2 test. -/
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
