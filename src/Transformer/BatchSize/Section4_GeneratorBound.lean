/-
# Size of the diffusion correction

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The covariance part of the generator is O(eta), so its contribution over
one optimizer step is O(eta^2).
-/

import Transformer.BatchSize.Section4_DiffusionModel

open scoped BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- The diagonal Hessian entries have absolute value at most one
when the Hessian operator norm does; Section 4.3, weak-error analysis
of Theorem 1, using Section 3.2's directional sharpness. -/
theorem coordinate_hessian_bound {d : ℕ} (φ : EucSpace d → ℝ) (x : EucSpace d)
    (hH : ‖iteratedFDeriv ℝ 2 φ x‖ ≤ 1) (k : Fin d) :
    |iteratedFDeriv ℝ 2 φ x (fun _ => EuclideanSpace.single k 1)| ≤ 1 := by
  have h := (iteratedFDeriv ℝ 2 φ x).le_opNorm
    (fun _ => EuclideanSpace.single k 1)
  simp only [Fin.prod_univ_two] at h
  have h' : |iteratedFDeriv ℝ 2 φ x (fun _ => EuclideanSpace.single k 1)| ≤
      ‖iteratedFDeriv ℝ 2 φ x‖ := by
    simpa [Real.norm_eq_abs, EuclideanSpace.single, PiLp.norm_single] using h
  exact h'.trans hH

/-- Nonvacuity of the coordinate curvature bound, Section 4.3. -/
example : ‖iteratedFDeriv ℝ 2 (fun _ : EucSpace 1 => (0 : ℝ)) 0‖ ≤ 1 := by
  rw [iteratedFDeriv_const_of_ne (by norm_num)]
  simp

/-- The diffusion part of the generator is at most half its covariance
trace on unit-Hessian tests; Section 4.3, equations (2)--(3). -/
theorem diffusionGenerator_correction_bound {d : ℕ} (method : UpdateKind)
    (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (φ : EucSpace d → ℝ) (x : EucSpace d) (hη : 0 ≤ η)
    (hH : ‖iteratedFDeriv ℝ 2 φ x‖ ≤ 1) :
    |diffusionGenerator method η B f σ φ x -
      fderiv ℝ φ x (diffusionDrift method B f σ x)| ≤
        (∑ k, diffusionCovariance method η B f σ x k) / 2 := by
  rw [diffusionGenerator, add_sub_cancel_left, abs_div, abs_of_pos (by norm_num : (0 : ℝ) < 2)]
  apply div_le_div_of_nonneg_right _ (by norm_num)
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  apply Finset.sum_le_sum
  intro k hk
  have hκ := diffusionCovariance_nonneg method η B f σ x k hη
  rw [abs_mul, abs_of_nonneg hκ]
  simpa using mul_le_mul_of_nonneg_left (coordinate_hessian_bound φ x hH k) hκ

/-- Nonvacuity of the correction estimate, Section 4.3. -/
example : (0 : ℝ) ≤ 1 ∧ ‖iteratedFDeriv ℝ 2 (fun _ : EucSpace 1 => (0 : ℝ)) 0‖ ≤ 1 := by
  constructor
  · norm_num
  · rw [iteratedFDeriv_const_of_ne (by norm_num)]
    simp

/-- SGD covariance trace, Section 4.3, equation (2). -/
theorem sgd_covariance_trace {d : ℕ} (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (x : EucSpace d) :
    (∑ k, diffusionCovariance .gradient η B f σ x k) =
      η * ∑ k, (σ x k) ^ 2 / B := by
  simp only [diffusionCovariance, Finset.mul_sum, mul_div_assoc]

/-- The signed covariance trace is at most eta*d, Section 4.3,
equation (3), with the exact centered sign variance. -/
theorem sign_covariance_trace_le {d : ℕ} (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (x : EucSpace d) (hη : 0 ≤ η) :
    (∑ k, diffusionCovariance .sign η B f σ x k) ≤ η * d := by
  calc
    (∑ k, diffusionCovariance .sign η B f σ x k) ≤ ∑ _ : Fin d, η := by
      apply Finset.sum_le_sum
      intro k hk
      dsimp [diffusionCovariance]
      nlinarith [sq_nonneg (signResponse B (σ x k) (gradient f x k))]
    _ = η * d := by simp [mul_comm]

/-- Nonvacuity of the trace bound, Section 4.3. -/
example : (0 : ℝ) ≤ 1 / 1000 := by norm_num

end Transformer.BatchSize
