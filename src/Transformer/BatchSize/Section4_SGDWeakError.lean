/-
# Local weak error of SGD

arXiv:2506.12543v1, Section 4.3, equation (2).
The actual sampled Gaussian vector, including its unbounded tails, is
handled by its L2 moment rather than by a bounded-update assumption.
-/

import Transformer.BatchSize.Section4_WeakTaylorIntegral
import Transformer.BatchSize.Section4_GradientMoments

open MeasureTheory
open scoped BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- The one-step weak drift error is controlled by the squared full
gradient and the trace of sigma^2/B, Section 4.3, equation (2).
This gives an O(eta^2) bound without bounding Gaussian samples. -/
theorem sgd_step_weak_drift {d : ℕ} (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d) (x : EucSpace d)
    (φ : EucSpace d → ℝ) (M : ℝ) (hφ : ContDiff ℝ 2 φ)
    (hH : ∀ y, ‖iteratedFDeriv ℝ 2 φ y‖ ≤ M) :
    |(∫ z, φ (stochasticStep .gradient η B f σ x z)
        ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) - φ x -
      η * fderiv ℝ φ x (diffusionDrift .gradient B f σ x)| ≤
        M * (‖gradient f x‖ ^ 2 + ∑ k, (σ x k) ^ 2 / B) * η ^ 2 := by
  let ν := diagonalNoiseLaw 1 (fun _ : Fin d => 0) (fun _ => 1)
  let V : (Fin d → ℝ) → EucSpace d := fun z => -η • sampledGradient B f σ x z
  have hV : Measurable V := by
    simpa only [V, Pi.smul_def] using (sampledGradient_measurable B f σ x).const_smul (-η)
  have hV2 : MemLp V 2 ν := by
    simpa only [V, Pi.smul_def] using (sampledGradient_memLp B f σ x).const_smul (-η)
  have ht := weak_taylor_integral ν V hV hV2 φ M hφ hH x
  have hmean : (∫ z, V z ∂ν) = η • diffusionDrift .gradient B f σ x := by
    change (∫ z, -η • sampledGradient B f σ x z ∂ν) = _
    rw [integral_smul, sampledGradient_vector_mean]
    simp [diffusionDrift]
  have hm : (∫ z, ‖V z‖ ^ 2 ∂ν) =
      η ^ 2 * (‖gradient f x‖ ^ 2 + ∑ k, (σ x k) ^ 2 / B) := by
    simp only [V, norm_smul, mul_pow, Real.norm_eq_abs, abs_neg, sq_abs]
    rw [integral_const_mul, sampledGradient_norm_second_moment]
  rw [hmean, hm] at ht
  simpa [ν, V, stochasticStep, diffusionDrift, sub_eq_add_neg, mul_assoc, mul_comm,
    mul_left_comm] using ht

/-- Nonvacuity of the SGD local weak assumptions, Section 4.3. -/
example : ContDiff ℝ 2 (fun _ : EucSpace 1 => (0 : ℝ)) ∧
    (∀ y : EucSpace 1, ‖iteratedFDeriv ℝ 2 (fun _ : EucSpace 1 => (0 : ℝ)) y‖ ≤ 1) := by
  refine ⟨contDiff_const, ?_⟩
  intro y
  rw [iteratedFDeriv_const_of_ne (by norm_num)]
  simp

end Transformer.BatchSize
