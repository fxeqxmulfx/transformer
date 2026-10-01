/-
# Gaussian generator contractions for actual bounded C2 tests

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The derivative field is the actual Frechet derivative of the test;
its derivative is the actual Hessian, with the stated uniform bounds.
-/

import Transformer.BatchSize.Section4_GaussianDirectionalIntegration
import Transformer.BatchSize.Section4_Regularity

open MeasureTheory

noncomputable section

namespace Transformer.BatchSize

/-- A diagonal Gaussian innovation contracts a genuine bounded C2
test's first derivative into exactly its diagonal Hessian entry,
Section 4.3 (2)--(3). These are the derivatives in the specified
diffusion generator, not separately supplied formal coefficients. -/
theorem gaussianAffineState_test_integration_by_parts {n : ℕ}
    (x : EucSpace (n + 1)) (A : Fin (n + 1) → ℝ) (k : Fin (n + 1))
    (φ : EucSpace (n + 1) → ℝ) (hφ : BoundedSmoothTest 2 φ) :
    (∫ z, z k * fderiv ℝ φ (gaussianAffineState x A z) (EuclideanSpace.single k 1)
      ∂standardGaussianVectorLaw (n + 1)) =
      A k * ∫ z, iteratedFDeriv ℝ 2 φ (gaussianAffineState x A z)
        (fun _ => EuclideanSpace.single k 1) ∂standardGaussianVectorLaw (n + 1) := by
  have hdf : ContDiff ℝ 1 (fderiv ℝ φ) := (contDiff_succ_iff_fderiv (n := 1)).mp hφ.1 |>.2.2
  have hg (y : EucSpace (n + 1)) :
      HasFDerivAt (fderiv ℝ φ) (fderiv ℝ (fderiv ℝ φ) y) y :=
    (hdf.differentiable (by norm_num) y).hasFDerivAt
  have hG : Continuous (fderiv ℝ (fderiv ℝ φ)) := hdf.continuous_fderiv (by norm_num)
  have hC (y : EucSpace (n + 1)) : ‖fderiv ℝ φ y‖ ≤ 1 := by
    rw [← norm_iteratedFDeriv_one φ]
    exact hφ.2 1 (by norm_num) y
  have hD (y : EucSpace (n + 1)) : ‖fderiv ℝ (fderiv ℝ φ) y‖ ≤ 1 := by
    rw [← norm_iteratedFDeriv_one (fderiv ℝ φ), norm_iteratedFDeriv_fderiv]
    exact hφ.2 2 le_rfl y
  have h := gaussianAffineState_directional_integration_by_parts x A k (EuclideanSpace.single k 1)
    (fderiv ℝ φ) (fderiv ℝ (fderiv ℝ φ)) hg hG 1 1 hC hD
  simpa only [iteratedFDeriv_two_apply] using h

/-- Nonvacuity of the generator's bounded C2 test hypothesis,
Section 4.3: a nonzero constant observable under the actual probability law. -/
example : BoundedSmoothTest 2 (fun _ : EucSpace 2 => (1 : ℝ)) := by
  refine ⟨contDiff_const, ?_⟩
  intro j hj y
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j =>
    rw [iteratedFDeriv_const_of_ne (by omega)]
    simp

end Transformer.BatchSize
