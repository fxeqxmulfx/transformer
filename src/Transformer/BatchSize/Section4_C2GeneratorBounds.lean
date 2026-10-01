/-
# Scaled C2 bounds for the true optimizer generator

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The covariance part is bounded by a model constant times eta and
the Hessian bound. The complete generator is continuous and bounded.
-/

import Transformer.BatchSize.Section4_ScaledGeneratorIdentity

open MeasureTheory
open scoped BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- Actual diagonal Hessian evaluations obey an arbitrary
operator-norm bound, Section 3.2 (1), Section 4.3, Theorem 1. -/
theorem coordinate_hessian_scaled_bound {d : ℕ} (φ : EucSpace d → ℝ)
    (x : EucSpace d) (R : ℝ) (hH : ‖iteratedFDeriv ℝ 2 φ x‖ ≤ R) (k : Fin d) :
    |iteratedFDeriv ℝ 2 φ x (fun _ => EuclideanSpace.single k 1)| ≤ R := by
  have h := (iteratedFDeriv ℝ 2 φ x).le_opNorm (fun _ => EuclideanSpace.single k 1)
  simp only [Fin.prod_univ_two] at h
  have hn : |iteratedFDeriv ℝ 2 φ x (fun _ => EuclideanSpace.single k 1)| ≤
      ‖iteratedFDeriv ℝ 2 φ x‖ := by
    simpa [Real.norm_eq_abs, EuclideanSpace.single, PiLp.norm_single] using h
  exact hn.trans hH

/-- The actual diffusion correction is at most U*R*eta for every
Hessian bounded by R, Section 4.3 (2)--(3), Theorem 1.
The model constant U is uniform in eta, the state and the observable. -/
theorem diffusionGenerator_C2_correction_bound {d : ℕ} (method : UpdateKind)
    (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hB : 0 < B) (hmodel : RegularGaussianModel f σ) :
    ∃ U : ℝ, 0 ≤ U ∧ ∀ η : ℝ, 0 ≤ η → ∀ R : ℝ, 0 ≤ R →
      ∀ φ : EucSpace d → ℝ, ∀ x, ‖iteratedFDeriv ℝ 2 φ x‖ ≤ R →
      |diffusionGenerator method η B f σ φ x -
        fderiv ℝ φ x (diffusionDrift method B f σ x)| ≤ U * R * η := by
  obtain ⟨c, A, hc, hA, hcov⟩ := diffusionCovariance_uniform_bounds method B f σ hB hmodel
  refine ⟨(d : ℝ) * A / 2, by positivity, ?_⟩
  intro η hη R hR φ x hH
  rw [diffusionGenerator, add_sub_cancel_left, abs_div, abs_of_pos (by norm_num : (0 : ℝ) < 2)]
  have hsum : |∑ k, diffusionCovariance method η B f σ x k *
      iteratedFDeriv ℝ 2 φ x (fun _ => EuclideanSpace.single k 1)| ≤ (d : ℝ) * (A * η * R) := by
    calc
      _ ≤ ∑ k, |diffusionCovariance method η B f σ x k *
          iteratedFDeriv ℝ 2 φ x (fun _ => EuclideanSpace.single k 1)| :=
        Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _ : Fin d, A * η * R := by
        apply Finset.sum_le_sum
        intro k hk
        rw [abs_mul, abs_of_nonneg (diffusionCovariance_nonneg method η B f σ x k hη)]
        exact mul_le_mul ((hcov η hη x k).2) (coordinate_hessian_scaled_bound φ x R hH k)
          (abs_nonneg _) (mul_nonneg hA hη)
      _ = _ := by simp
  refine (div_le_div_of_nonneg_right hsum (by norm_num : (0 : ℝ) ≤ 2)).trans_eq ?_
  ring

/-- Continuity and actual boundedness of the full generator on
arbitrarily sized bounded C2 observables, Section 4.3 (2)--(3). -/
theorem diffusionGenerator_scaled_continuous_bounded {d : ℕ} (method : UpdateKind)
    (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 ≤ η) (hB : 0 < B) (hmodel : RegularGaussianModel f σ)
    (φ : EucSpace d → ℝ) (R : ℝ) (hR : 0 < R) (hφ : ContDiff ℝ 2 φ)
    (hbound : ∀ j ≤ 2, ∀ x, ‖iteratedFDeriv ℝ j φ x‖ ≤ R) :
    Continuous (diffusionGenerator method η B f σ φ) ∧
      ∃ K : ℝ, 0 ≤ K ∧ ∀ x, |diffusionGenerator method η B f σ φ x| ≤ K := by
  let ψ := fun x => R⁻¹ • φ x
  have hψ := boundedSmoothTest_rescale 2 φ R hR hφ hbound
  obtain ⟨hc, K, hK, hk⟩ := diffusionGenerator_continuous_bounded method η B f σ hη hB hmodel ψ hψ
  have heq : diffusionGenerator method η B f σ φ =
      fun x => R * diffusionGenerator method η B f σ ψ x := by
    funext x
    rw [diffusionGenerator_const_smul method η B f σ R⁻¹ φ x hφ]
    rw [← mul_assoc, mul_inv_cancel₀ hR.ne', one_mul]
  rw [heq]
  refine ⟨hc.const_mul R, R * K, mul_nonneg hR.le hK, ?_⟩
  intro x
  rw [abs_mul, abs_of_pos hR]
  exact mul_le_mul_of_nonneg_left (hk x) hR.le

/-- Joint nonvacuity of the scaled Hessian and generator
hypotheses, Section 4.3: flat regular loss, unit noise and nonzero test. -/
example : 0 < (1 : ℕ) ∧ RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    (0 : ℝ) < 2 ∧ ContDiff ℝ 2 (fun _ : EucSpace 1 => (1 : ℝ)) ∧
    (∀ j ≤ 2, ∀ x : EucSpace 1, ‖iteratedFDeriv ℝ j (fun _ => (1 : ℝ)) x‖ ≤ 2) := by
  refine ⟨by norm_num, regularGaussianModel_flat 1, by norm_num, by norm_num, contDiff_const, ?_⟩
  intro j hj x
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
