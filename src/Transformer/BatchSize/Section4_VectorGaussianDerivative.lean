/-
# Differentiating the actual vector Gaussian expectation in time

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The fixed product measure permits differentiation under the integral.
The derivative has a genuine integrable domination bound on s > t/2.
-/

import Transformer.BatchSize.Section4_VectorGaussianFlow

open MeasureTheory ProbabilityTheory Filter
open scoped NNReal Topology

noncomputable section

namespace Transformer.BatchSize

/-- The time derivative of the actual frozen vector Gaussian
expectation is the expectation of the genuine samplewise derivative,
Section 4.3 (2)--(3). Domination uses the proved finite Gaussian noise
moment, and applies to every bounded C2 test in the martingale problem. -/
theorem gaussianVectorFlow_hasDerivAt_rescaled {d : ℕ} (φ : EucSpace d → ℝ)
    (hφ : BoundedSmoothTest 2 φ) (x b : EucSpace d) (A : Fin d → ℝ)
    (t : ℝ) (ht : 0 < t) :
    HasDerivAt (gaussianVectorFlow φ x b A)
      (∫ z, fderiv ℝ φ (gaussianVectorState x b A t z)
        (b + (1 / (2 * Real.sqrt t)) • gaussianDiagonalMap A z) ∂standardGaussianVectorLaw d) t := by
  have hφc := hφ.1.continuous
  have hdc : Continuous (fderiv ℝ φ) := hφ.1.continuous_fderiv (by norm_num)
  have hφbound (y : EucSpace d) : ‖φ y‖ ≤ 1 := by
    rw [← norm_iteratedFDeriv_zero (𝕜 := ℝ)]
    exact hφ.2 0 (by norm_num) y
  have hdbound (y : EucSpace d) : ‖fderiv ℝ φ y‖ ≤ 1 := by
    rw [← norm_iteratedFDeriv_one φ]
    exact hφ.2 1 (by norm_num) y
  let F' (s : ℝ) (z : Fin d → ℝ) : ℝ :=
    fderiv ℝ φ (gaussianVectorState x b A s z)
      (b + (1 / (2 * Real.sqrt s)) • gaussianDiagonalMap A z)
  have hint : Integrable (fun z => φ (gaussianVectorState x b A t z)) (standardGaussianVectorLaw d) :=
    (integrable_const (1 : ℝ)).mono'
      (hφc.comp (gaussianVectorState_continuous_noise x b A t)).aestronglyMeasurable
      (Eventually.of_forall fun z => hφbound _)
  have hF'm : AEStronglyMeasurable (F' t) (standardGaussianVectorLaw d) :=
    ((hdc.comp (gaussianVectorState_continuous_noise x b A t)).clm_apply
      (continuous_const.add ((gaussianDiagonalMap A).continuous.const_smul (1 / (2 * Real.sqrt t))))).aestronglyMeasurable
  have hnoise := ((gaussianDiagonalMap_memLp A).integrable (by norm_num : 1 ≤ (2 : ENNReal))).norm
  refine (hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := standardGaussianVectorLaw d)
    (F := fun s z => φ (gaussianVectorState x b A s z)) (F' := F')
    (bound := fun z => ‖b‖ + (1 / (2 * Real.sqrt (t / 2))) * ‖gaussianDiagonalMap A z‖)
    (Ioi_mem_nhds (by linarith : t / 2 < t))
    (Eventually.of_forall fun s =>
      (hφc.comp (gaussianVectorState_continuous_noise x b A s)).aestronglyMeasurable)
    hint hF'm ?_ ?_ ?_).2
  · exact Eventually.of_forall fun z s hs => by
      have hspos : 0 < s := by have hs' : t / 2 < s := hs; linarith
      have hsq : Real.sqrt (t / 2) ≤ Real.sqrt s := Real.sqrt_le_sqrt hs.le
      have hc : 0 ≤ 1 / (2 * Real.sqrt s) := by positivity
      dsimp only [F']
      calc
        _ ≤ ‖fderiv ℝ φ (gaussianVectorState x b A s z)‖ *
            ‖b + (1 / (2 * Real.sqrt s)) • gaussianDiagonalMap A z‖ :=
          (fderiv ℝ φ _).le_opNorm _
        _ ≤ 1 * ‖b + (1 / (2 * Real.sqrt s)) • gaussianDiagonalMap A z‖ :=
          mul_le_mul_of_nonneg_right (hdbound _) (norm_nonneg _)
        _ = ‖b + (1 / (2 * Real.sqrt s)) • gaussianDiagonalMap A z‖ := one_mul _
        _ ≤ ‖b‖ + ‖(1 / (2 * Real.sqrt s)) • gaussianDiagonalMap A z‖ := norm_add_le _ _
        _ ≤ _ := by
          rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hc]
          gcongr
  · exact (integrable_const ‖b‖).add (hnoise.const_mul (1 / (2 * Real.sqrt (t / 2))))
  · exact Eventually.of_forall fun z s hs => by
      have hspos : 0 < s := by have hs' : t / 2 < s := hs; linarith
      have h := (hφ.1.differentiable (by norm_num) (gaussianVectorState x b A s z)).hasFDerivAt.comp_hasDerivAt s
        (gaussianVectorState_hasDerivAt x b A z s hspos)
      simpa only [Function.comp_def, F'] using h

/-- Joint nonvacuity of normalized C2 and positive-time hypotheses,
Section 4.3: the nonzero constant observable on a positive interval. -/
example : BoundedSmoothTest 2 (fun _ : EucSpace 1 => (1 : ℝ)) ∧ (0 : ℝ) < 1 := by
  refine ⟨⟨contDiff_const, ?_⟩, by norm_num⟩
  intro j hj y
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j =>
    rw [iteratedFDeriv_const_of_ne (by omega)]
    simp

end Transformer.BatchSize
