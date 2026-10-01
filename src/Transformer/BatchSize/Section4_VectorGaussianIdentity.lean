/-
# Integrated generator identity for the actual vector Gaussian law

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The fundamental theorem of calculus includes the zero-time endpoint;
no stochastic Taylor or martingale identity is assumed.
-/

import Transformer.BatchSize.Section4_VectorGaussianGenerator
import Transformer.BatchSize.Section4_GaussianGenerator

open MeasureTheory

noncomputable section

namespace Transformer.BatchSize

/-- The integrated generator equation for the genuine frozen
Gaussian transition, Section 4.3 (2)--(3). The identity holds for every
bounded C2 test, with no third-derivative hypothesis. -/
theorem gaussianVectorFlow_generator_identity {d : ℕ}
    (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ)
    (x b : EucSpace d) (A : Fin d → ℝ) (t : ℝ) (ht : 0 ≤ t) :
    gaussianVectorFlow φ x b A t = φ x + ∫ s in (0 : ℝ)..t,
      gaussianVectorFlow (frozenGaussianGenerator b A φ) x b A s := by
  have hφbound (y : EucSpace d) : |φ y| ≤ 1 := by
    simpa only [norm_iteratedFDeriv_zero, Real.norm_eq_abs] using hφ.2 0 (by norm_num) y
  have hgc := frozenGaussianGenerator_continuous b A φ hφ
  have hgb := frozenGaussianGenerator_abs_le b A φ hφ
  have hcont := gaussianVectorFlow_continuous φ hφ.1.continuous x b A 1 hφbound
  have hgcont := gaussianVectorFlow_continuous _ hgc x b A
    (‖b‖ + (∑ k, (A k) ^ 2) / 2) hgb
  have hFTC := intervalIntegral.integral_eq_sub_of_hasDeriv_right_of_le ht
    hcont.continuousOn
    (fun s hs => (gaussianVectorFlow_hasDerivAt_generator φ hφ x b A s hs.1).hasDerivWithinAt)
    (hgcont.intervalIntegrable 0 t)
  rw [gaussianVectorFlow_zero] at hFTC
  linarith

/-- Joint nonvacuity of the actual integrated generator hypotheses,
Section 4.3: a normalized nonzero observable and a positive horizon. -/
example : BoundedSmoothTest 2 (fun _ : EucSpace 2 => (1 : ℝ)) ∧ (0 : ℝ) ≤ 1 := by
  refine ⟨⟨contDiff_const, ?_⟩, by norm_num⟩
  intro j hj y
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
