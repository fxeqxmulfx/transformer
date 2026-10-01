/-
# The genuine frozen diagonal diffusion generator

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The coefficients are held at the left endpoint of an Euler interval;
the derivatives remain the actual derivatives of the observable.
-/

import Transformer.BatchSize.Section4_VectorGaussianDerivative
import Transformer.BatchSize.Section4_GeneratorBound

open MeasureTheory
open scoped BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- Coordinate expansion of the actual Gaussian noise vector,
Section 4.3 (2)--(3), in the Euclidean basis used by the generator. -/
theorem gaussianDiagonalMap_eq_sum {d : ℕ} (A : Fin d → ℝ) (z : Fin d → ℝ) :
    gaussianDiagonalMap A z =
      ∑ k, (A k * z k) • EuclideanSpace.single k 1 := by
  ext j
  change A j * z j = (∑ k, (A k * z k) • EuclideanSpace.single k 1) j
  have h := map_sum (PiLp.proj (𝕜 := ℝ) (p := 2) (β := fun _ : Fin d => ℝ) j)
    (fun k => (A k * z k) • EuclideanSpace.single k 1) Finset.univ
  change (∑ k, (A k * z k) • EuclideanSpace.single k 1) j =
    ∑ k, ((A k * z k) • EuclideanSpace.single k 1) j at h
  rw [h]
  simp [PiLp.smul_apply, EuclideanSpace.single]

/-- The actual generator with drift b and diagonal noise amplitudes A
held fixed, Section 4.3 (2)--(3). -/
def frozenGaussianGenerator {d : ℕ} (b : EucSpace d) (A : Fin d → ℝ)
    (φ : EucSpace d → ℝ) (y : EucSpace d) : ℝ :=
  fderiv ℝ φ y b +
    (∑ k, (A k) ^ 2 * iteratedFDeriv ℝ 2 φ y
      (fun _ => EuclideanSpace.single k 1)) / 2

/-- Bounded C2 tests give a genuine uniform generator bound,
Section 4.3 (2)--(3), sufficient for expectation and time integration. -/
theorem frozenGaussianGenerator_abs_le {d : ℕ} (b : EucSpace d) (A : Fin d → ℝ)
    (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ) (y : EucSpace d) :
    |frozenGaussianGenerator b A φ y| ≤ ‖b‖ + (∑ k, (A k) ^ 2) / 2 := by
  have hd : ‖fderiv ℝ φ y‖ ≤ 1 := by
    rw [← norm_iteratedFDeriv_one φ]
    exact hφ.2 1 (by norm_num) y
  have hb : |fderiv ℝ φ y b| ≤ ‖b‖ := by
    simpa only [Real.norm_eq_abs, one_mul] using
      ((fderiv ℝ φ y).le_opNorm b).trans
        (mul_le_mul_of_nonneg_right hd (norm_nonneg b))
  have hsum : |∑ k, (A k) ^ 2 * iteratedFDeriv ℝ 2 φ y
      (fun _ => EuclideanSpace.single k 1)| ≤ ∑ k, (A k) ^ 2 := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun k _ => ?_)
    rw [abs_mul, abs_of_nonneg (sq_nonneg _)]
    simpa only [mul_one] using mul_le_mul_of_nonneg_left
      (coordinate_hessian_bound φ y (hφ.2 2 le_rfl y) k) (sq_nonneg (A k))
  unfold frozenGaussianGenerator
  refine (abs_add_le _ _).trans (add_le_add hb ?_)
  simpa only [abs_div, abs_of_pos (by norm_num : (0 : ℝ) < 2)] using
    div_le_div_of_nonneg_right hsum (by norm_num : (0 : ℝ) ≤ 2)

/-- The frozen generator of every actual bounded C2 observable is
continuous, Section 4.3 (2)--(3). -/
theorem frozenGaussianGenerator_continuous {d : ℕ} (b : EucSpace d) (A : Fin d → ℝ)
    (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ) :
    Continuous (frozenGaussianGenerator b A φ) := by
  have hd := hφ.1.continuous_fderiv (by norm_num)
  have hH := hφ.1.continuous_iteratedFDeriv' (m := 2)
  exact (hd.clm_apply continuous_const).add
    ((continuous_finsetSum Finset.univ fun k _ => continuous_const.mul
      ((ContinuousMultilinearMap.apply ℝ (fun _ : Fin 2 => EucSpace d) ℝ
        (fun _ : Fin 2 => EuclideanSpace.single k (1 : ℝ))).continuous.comp hH)).div_const 2)

/-- Joint nonvacuity of the frozen generator's hypotheses, Section 4.3:
a normalized nonzero constant observable in a positive dimension. -/
example : BoundedSmoothTest 2 (fun _ : EucSpace 1 => (1 : ℝ)) := by
  refine ⟨contDiff_const, ?_⟩
  intro j hj y
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
