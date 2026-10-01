/-
# Local weak error of SignSGD

arXiv:2506.12543v1, Section 4.3, Theorem 1.
The drift approximation has an explicit O(eta^2) expectation error.
This is a proved step toward the global weak-approximation claim;
it does not assume existence of a continuous diffusion law.
-/

import Transformer.BatchSize.Section3_WeakTaylor
import Transformer.BatchSize.Section4_StepMoments

open MeasureTheory

noncomputable section

namespace Transformer.BatchSize

/-- One signed step has weak drift error at most M*d*eta^2 when the
test function's Hessian norm is bounded by M; Section 4.3, Theorem 1.
The left-hand side uses the actual Gaussian stochastic optimizer. -/
theorem sign_step_weak_drift {d : ℕ} (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d) (x : EucSpace d)
    (φ : EucSpace d → ℝ) (M : ℝ) (hB : 0 < B) (hσ : ∀ k, 0 < σ x k)
    (hφ : ContDiff ℝ 2 φ) (hH : ∀ y, ‖iteratedFDeriv ℝ 2 φ y‖ ≤ M) :
    |(∫ z, φ (stochasticStep .sign η B f σ x z)
        ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) - φ x -
      η * fderiv ℝ φ x (diffusionDrift .sign B f σ x)| ≤ M * d * η ^ 2 := by
  let ν := diagonalNoiseLaw 1 (fun _ : Fin d => 0) (fun _ => 1)
  let U : (Fin d → ℝ) → EucSpace d := fun z => -η • signedGradient B f σ x z
  let R : (Fin d → ℝ) → ℝ := fun z => φ (x + U z) - φ x - fderiv ℝ φ x (U z)
  have hM : 0 ≤ M := (norm_nonneg _).trans (hH x)
  have hpoint (z : Fin d → ℝ) : |R z| ≤ M * d * η ^ 2 := by
    have ht := uniform_first_order_taylor φ M hφ hH x (U z)
    have hn : ‖U z‖ ^ 2 = η ^ 2 * ‖signedGradient B f σ x z‖ ^ 2 := by
      simp only [U, norm_smul, mul_pow, Real.norm_eq_abs, abs_neg, sq_abs]
    calc
      |R z| ≤ M * ‖U z‖ ^ 2 := ht
      _ = M * (η ^ 2 * ‖signedGradient B f σ x z‖ ^ 2) := by rw [hn]
      _ ≤ M * (η ^ 2 * d) :=
        mul_le_mul_of_nonneg_left
          (mul_le_mul_of_nonneg_left (signedGradient_norm_sq B f σ x z) (sq_nonneg η)) hM
      _ = M * d * η ^ 2 := by ring
  have hU : Measurable U := by
    simpa only [U, Pi.smul_def] using
      (signedGradient_measurable B f σ x).const_smul (-η)
  have hxU : Measurable (fun z => x + U z) := by
    simpa [Pi.add_def] using (measurable_const (a := x)).add hU
  have hR : Measurable R := by
    exact (hφ.continuous.measurable.comp hxU).sub measurable_const |>.sub
      ((fderiv ℝ φ x).continuous.measurable.comp hU)
  have hiU : Integrable U ν := by
    simpa only [U, Pi.smul_def] using Integrable.smul (-η) (signedGradient_integrable B f σ x)
  have hiL := (fderiv ℝ φ x).integrable_comp hiU
  have hiR : Integrable R ν := by
    apply (integrable_const (M * d * η ^ 2)).mono' hR.aestronglyMeasurable
    exact ae_of_all _ (fun z => by simpa [Real.norm_eq_abs] using hpoint z)
  have hiφ : Integrable (fun z => φ (x + U z)) ν := by
    convert (hiR.add (integrable_const (φ x))).add hiL using 1
    funext z
    dsimp [R]
    ring
  have hmean : (∫ z, U z ∂ν) = η • diffusionDrift .sign B f σ x := by
    change (∫ z, -η • signedGradient B f σ x z ∂ν) = _
    rw [integral_smul, signedGradient_mean B f σ x hB hσ]
    simp
  have hr : (∫ z, R z ∂ν) = (∫ z, φ (x + U z) ∂ν) - φ x -
      η * fderiv ℝ φ x (diffusionDrift .sign B f σ x) := by
    change (∫ z, φ (x + U z) - φ x - fderiv ℝ φ x (U z) ∂ν) = _
    rw [integral_sub (f := fun z => φ (x + U z) - φ x)
        (g := fun z => fderiv ℝ φ x (U z)) (hiφ.sub (integrable_const _)) hiL,
      integral_sub (f := fun z => φ (x + U z)) (g := fun _ => φ x)
        hiφ (integrable_const _),
      (fderiv ℝ φ x).integral_comp_comm hiU, hmean]
    simp
  have hb := norm_integral_le_of_norm_le_const (f := R) (C := M * d * η ^ 2)
    (ae_of_all ν (fun z => by simpa [Real.norm_eq_abs] using hpoint z))
  rw [hr] at hb
  simpa [Real.norm_eq_abs, U, ν, stochasticStep, signedGradient,
    sub_eq_add_neg] using hb

/-- Nonvacuity of the local weak estimate: unit noise and a constant
smooth test with Hessian norm bounded by one; Section 4.3. -/
example : 0 < (1 : ℕ) ∧
    (∀ k : Fin 1, (0 : ℝ) < (EuclideanSpace.single (0 : Fin 1) 1) k) ∧
    ContDiff ℝ 2 (fun _ : EucSpace 1 => (0 : ℝ)) ∧
    (∀ y : EucSpace 1, ‖iteratedFDeriv ℝ 2 (fun _ : EucSpace 1 => (0 : ℝ)) y‖ ≤ 1) := by
  refine ⟨by norm_num, ?_, contDiff_const, ?_⟩
  · intro k
    fin_cases k
    simp [EuclideanSpace.single]
  · intro y
    rw [iteratedFDeriv_const_of_ne (by norm_num)]
    simp

end Transformer.BatchSize
