/-
# The vector Gaussian expectation solves its actual generator equation

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Coordinate Gaussian integration by parts removes the singular sqrt(t)
derivative and produces exactly half the diagonal Hessian contraction.
-/

import Transformer.BatchSize.Section4_FrozenGaussianGenerator

open MeasureTheory ProbabilityTheory Filter
open scoped NNReal ENNReal BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- The true coordinate integration-by-parts identity in every
dimension, Section 4.3 (2)--(3), including the empty-coordinate case. -/
theorem gaussianVectorState_test_integration_by_parts {d : ℕ}
    (x b : EucSpace d) (A : Fin d → ℝ) (t : ℝ) (k : Fin d)
    (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ) :
    (∫ z, z k * fderiv ℝ φ (gaussianVectorState x b A t z) (EuclideanSpace.single k 1)
      ∂standardGaussianVectorLaw d) =
      (Real.sqrt t * A k) * ∫ z, iteratedFDeriv ℝ 2 φ (gaussianVectorState x b A t z)
        (fun _ => EuclideanSpace.single k 1) ∂standardGaussianVectorLaw d := by
  cases d with
  | zero => exact Fin.elim0 k
  | succ n =>
    simpa only [gaussianVectorState_eq_affine] using
      gaussianAffineState_test_integration_by_parts (x + t • b)
        (fun j => Real.sqrt t * A j) k φ hφ

/-- For genuine bounded C2 observables, the positive-time derivative
of the actual frozen Gaussian expectation is its generator expectation,
Section 4.3 (2)--(3). -/
theorem gaussianVectorFlow_hasDerivAt_generator {d : ℕ}
    (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ)
    (x b : EucSpace d) (A : Fin d → ℝ) (t : ℝ) (ht : 0 < t) :
    HasDerivAt (gaussianVectorFlow φ x b A)
      (gaussianVectorFlow (frozenGaussianGenerator b A φ) x b A t) t := by
  let Y := gaussianVectorState x b A t
  let H (k : Fin d) (z : Fin d → ℝ) :=
    iteratedFDeriv ℝ 2 φ (Y z) (fun _ => EuclideanSpace.single k 1)
  let V (k : Fin d) (z : Fin d → ℝ) :=
    z k * fderiv ℝ φ (Y z) (EuclideanSpace.single k 1)
  let q := 1 / (2 * Real.sqrt t)
  have hY := gaussianVectorState_continuous_noise x b A t
  have hdc := hφ.1.continuous_fderiv (by norm_num)
  have hdb (y : EucSpace d) : ‖fderiv ℝ φ y‖ ≤ 1 := by
    rw [← norm_iteratedFDeriv_one φ]
    exact hφ.2 1 (by norm_num) y
  have hHi (k : Fin d) : Integrable (H k) (standardGaussianVectorLaw d) := by
    apply (integrable_const (1 : ℝ)).mono'
    · exact ((ContinuousMultilinearMap.apply ℝ (fun _ : Fin 2 => EucSpace d) ℝ
        (fun _ => EuclideanSpace.single k (1 : ℝ))).continuous.comp
          (hφ.1.continuous_iteratedFDeriv' (m := 2) |>.comp hY)).aestronglyMeasurable
    · exact Eventually.of_forall fun z => by
        simpa only [Real.norm_eq_abs] using coordinate_hessian_bound φ (Y z) (hφ.2 2 le_rfl _) k
  have hVi (k : Fin d) : Integrable (V k) (standardGaussianVectorLaw d) := by
    have hzi : Integrable (fun z : Fin d → ℝ => z k) (standardGaussianVectorLaw d) :=
      ((memLp_id_gaussianReal (μ := 0) (v := 1) 1).comp_measurePreserving
        (measurePreserving_eval (fun _ : Fin d => gaussianReal 0 1) k)).integrable (by norm_num)
    have hg := (hdc.comp hY).clm_apply (continuous_const (y := EuclideanSpace.single k (1 : ℝ)))
    apply hzi.mul_bdd hg.aestronglyMeasurable
    exact Eventually.of_forall fun z => by
      simpa [EuclideanSpace.single, PiLp.norm_single] using
        ((fderiv ℝ φ (Y z)).le_opNorm (EuclideanSpace.single k 1)).trans
          (mul_le_mul_of_nonneg_right (hdb _) (norm_nonneg _))
  have hbi : Integrable (fun z => fderiv ℝ φ (Y z) b) (standardGaussianVectorLaw d) :=
    (integrable_const ‖b‖).mono' ((hdc.comp hY).clm_apply continuous_const).aestronglyMeasurable
      (Eventually.of_forall fun z => by
        simpa only [one_mul] using ((fderiv ℝ φ (Y z)).le_opNorm b).trans
          (mul_le_mul_of_nonneg_right (hdb _) (norm_nonneg b)))
  have hsumV := integrable_finsetSum Finset.univ (fun k _ => (hVi k).const_mul (A k))
  have hsumH := integrable_finsetSum Finset.univ (fun k _ => (hHi k).const_mul ((A k) ^ 2))
  have hpoint (z : Fin d → ℝ) : fderiv ℝ φ (Y z) (b + q • gaussianDiagonalMap A z) =
      fderiv ℝ φ (Y z) b + q * ∑ k, A k * V k z := by
    rw [map_add, map_smul, gaussianDiagonalMap_eq_sum, map_sum]
    simp only [map_smul, smul_eq_mul, V]
    congr 1
    congr 1
    apply Finset.sum_congr rfl
    intro k hk
    ring
  have hstein (k : Fin d) : (∫ z, V k z ∂standardGaussianVectorLaw d) =
      (Real.sqrt t * A k) * ∫ z, H k z ∂standardGaussianVectorLaw d :=
    gaussianVectorState_test_integration_by_parts x b A t k φ hφ
  have heq : (∫ z, fderiv ℝ φ (Y z) (b + q • gaussianDiagonalMap A z) ∂standardGaussianVectorLaw d) =
      gaussianVectorFlow (frozenGaussianGenerator b A φ) x b A t := by
    simp_rw [hpoint]
    rw [integral_add hbi (hsumV.const_mul q), integral_const_mul,
      integral_finsetSum Finset.univ (fun k _ => (hVi k).const_mul (A k))]
    simp_rw [integral_const_mul, hstein]
    unfold gaussianVectorFlow frozenGaussianGenerator
    rw [integral_add hbi (hsumH.div_const 2), integral_div,
      integral_finsetSum Finset.univ (fun k _ => (hHi k).const_mul ((A k) ^ 2))]
    simp_rw [integral_const_mul]
    congr 1
    rw [Finset.mul_sum, Finset.sum_div]
    apply Finset.sum_congr rfl
    intro k hk
    dsimp only [q]
    field_simp
  rw [← heq]
  exact gaussianVectorFlow_hasDerivAt_rescaled φ hφ x b A t ht

/-- Joint nonvacuity of the generator-equation hypotheses,
Section 4.3: a normalized nonzero test at positive time. -/
example : BoundedSmoothTest 2 (fun _ : EucSpace 1 => (1 : ℝ)) ∧ (0 : ℝ) < 1 := by
  refine ⟨⟨contDiff_const, ?_⟩, by norm_num⟩
  intro j hj y
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
