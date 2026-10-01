/-
# The actual optimizer generator identity for tests of any bounded size

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The proved identity for normalized tests extends by linearity to every
bounded C2 test. This includes the genuine deterministic backward tests.
-/

import Transformer.BatchSize.Section4_OptimizerMartingale
import Transformer.BatchSize.Section4_ScaledWeakConsistency

open MeasureTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- Normalizing a genuine bounded Cn observable preserves all
specified derivatives, Section 4.3, Theorem 1. -/
theorem boundedSmoothTest_rescale {d : ℕ} (n : ℕ) (φ : EucSpace d → ℝ)
    (R : ℝ) (hR : 0 < R) (hφ : ContDiff ℝ n φ)
    (hbound : ∀ j ≤ n, ∀ x, ‖iteratedFDeriv ℝ j φ x‖ ≤ R) :
    BoundedSmoothTest n (fun y => R⁻¹ • φ y) := by
  refine ⟨hφ.const_smul _, ?_⟩
  intro j hj y
  have hjφ : ContDiffAt ℝ j φ y := (hφ.of_le (by exact_mod_cast hj)).contDiffAt
  rw [iteratedFDeriv_const_smul_apply' hjφ, norm_smul, Real.norm_eq_abs,
    abs_of_pos (inv_pos.mpr hR)]
  exact (mul_le_mul_of_nonneg_left (hbound j hj y) (inv_nonneg.mpr hR.le)).trans_eq
    (inv_mul_cancel₀ hR.ne')

/-- Nonvacuity of the rescaling assumptions, Section 4.3:
a nonzero constant smooth observable with a positive bound. -/
example : (0 : ℝ) < 2 ∧ ContDiff ℝ 2 (fun _ : EucSpace 1 => (1 : ℝ)) ∧
    (∀ j ≤ 2, ∀ x : EucSpace 1, ‖iteratedFDeriv ℝ j (fun _ => (1 : ℝ)) x‖ ≤ 2) := by
  refine ⟨by norm_num, contDiff_const, ?_⟩
  intro j hj x
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

/-- The actual continuous optimizer satisfies the integrated
generator identity for any bounded C2 observable, Section 4.3 (2)--(3).
The path is the proved Brownian Euler limit; no diffusion premise is added. -/
theorem optimizerEulerPath_scaled_generator_identity {d : ℕ} (method : UpdateKind)
    (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 ≤ η) (hB : 0 < B) (hmodel : RegularGaussianModel f σ) (x₀ : EucSpace d)
    (s t : NNReal) (hst : s ≤ t) (φ : EucSpace d → ℝ) (R : ℝ) (hR : 0 < R)
    (hφ : ContDiff ℝ 2 φ) (hbound : ∀ j ≤ 2, ∀ x, ‖iteratedFDeriv ℝ j φ x‖ ≤ R) :
    (∫ ω, φ (optimizerEulerPath method η B f σ hη hB hmodel x₀ ω t) ∂brownianNoiseLaw d) -
      (∫ ω, φ (optimizerEulerPath method η B f σ hη hB hmodel x₀ ω s) ∂brownianNoiseLaw d) =
      ∫ u in (s : ℝ)..(t : ℝ), ∫ ω, diffusionGenerator method η B f σ φ
        (optimizerEulerPath method η B f σ hη hB hmodel x₀ ω u) ∂brownianNoiseLaw d := by
  let ψ := fun y => R⁻¹ • φ y
  have hψ : BoundedSmoothTest 2 ψ := boundedSmoothTest_rescale 2 φ R hR hφ hbound
  have h := optimizerEulerPath_weighted_generator_identity method η B f σ hη hB hmodel x₀ s t hst
    (fun _ => 1) stronglyMeasurable_const (by intro ω; norm_num) ψ hψ
  have hg (x : EucSpace d) : diffusionGenerator method η B f σ ψ x =
      R⁻¹ * diffusionGenerator method η B f σ φ x :=
    diffusionGenerator_const_smul method η B f σ R⁻¹ φ x hφ
  simp_rw [hg] at h
  simp only [ψ, smul_eq_mul, one_mul, integral_const_mul, intervalIntegral.integral_const_mul] at h
  rw [← mul_sub] at h
  exact mul_left_cancel₀ (inv_ne_zero hR.ne') h

/-- Joint nonvacuity of all scaled optimizer identity hypotheses,
Section 4.3: flat regular loss, unit noise, positive interval and nonzero test. -/
example : (0 : ℝ) ≤ 1 / 1000 ∧ 0 < (1 : ℕ) ∧
    RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
      (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) ∧ (1 : NNReal) ≤ 2 ∧
    (0 : ℝ) < 2 ∧ ContDiff ℝ 2 (fun _ : EucSpace 1 => (1 : ℝ)) ∧
    (∀ j ≤ 2, ∀ x : EucSpace 1, ‖iteratedFDeriv ℝ j (fun _ => (1 : ℝ)) x‖ ≤ 2) := by
  refine ⟨by norm_num, by norm_num, regularGaussianModel_flat 1, by norm_num,
    by norm_num, contDiff_const, ?_⟩
  intro j hj x
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
