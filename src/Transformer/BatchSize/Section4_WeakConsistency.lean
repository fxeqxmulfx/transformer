/-
# Uniform local weak consistency of both optimizers

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The discrete one-step expectations match the stated SDE generators
to order eta^2, uniformly in the state and normalized smooth tests.
This does not assert existence of the SDE or global weak approximation.
-/

import Transformer.BatchSize.Section4_Regularity
import Transformer.BatchSize.Section4_GeneratorBound
import Transformer.BatchSize.Section4_LocalWeakError
import Transformer.BatchSize.Section4_SGDWeakError

open MeasureTheory
open scoped BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- Proved local part of Section 4.3, Theorem 1: the actual Gaussian
SGD/SignSGD expectation has a uniform O(eta^2) defect against the generator
of equations (2)--(3). The constant is chosen before the state, test and step. -/
theorem sgd_sign_uniform_local_weak_consistency {d : ℕ} (method : UpdateKind)
    (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hB : 0 < B) (hmodel : RegularGaussianModel f σ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ η : ℝ, 0 ≤ η → ∀ φ, BoundedSmoothTest 6 φ →
      ∀ x, |(∫ z, φ (stochasticStep method η B f σ x z)
          ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) - φ x -
        η * diffusionGenerator method η B f σ φ x| ≤ C * η ^ 2 := by
  obtain ⟨L, hL, hbounds⟩ := regularGaussianModel_uniform_bounds f σ hmodel
  let C := 2 * ((d : ℝ) + 1) * (L ^ 2 + 1)
  refine ⟨C, by positivity, ?_⟩
  intro η hη φ hφ x
  have hφ2 : ContDiff ℝ 2 φ := hφ.1.of_le (by norm_num)
  have hH (y : EucSpace d) : ‖iteratedFDeriv ℝ 2 φ y‖ ≤ 1 := hφ.2 2 (by norm_num) y
  have hgen := diffusionGenerator_correction_bound method η B f σ φ x hη (hH x)
  let A := (∫ z, φ (stochasticStep method η B f σ x z)
    ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) - φ x
  let D := fderiv ℝ φ x (diffusionDrift method B f σ x)
  have hsplit : |A - η * diffusionGenerator method η B f σ φ x| ≤
      |A - η * D| + η * |diffusionGenerator method η B f σ φ x - D| := by
    have h := abs_sub_le A (η * D) (η * diffusionGenerator method η B f σ φ x)
    rw [← mul_sub, abs_mul, abs_of_nonneg hη, abs_sub_comm D] at h
    exact h
  cases method
  · let N := ∑ k, (σ x k) ^ 2 / B
    have hb : (0 : ℝ) < B := by exact_mod_cast hB
    have hb1 : (1 : ℝ) ≤ B := by exact_mod_cast hB
    have hN : N ≤ d * L ^ 2 := by
      calc
        N ≤ ∑ _ : Fin d, L ^ 2 := by
          apply Finset.sum_le_sum
          intro k hk
          have hs := (hbounds x).2 k |>.2
          have hs2 : (σ x k) ^ 2 ≤ L ^ 2 := by
            have h := abs_le.mp hs
            nlinarith [h.1, h.2]
          apply (div_le_iff₀ hb).2
          exact hs2.trans (by simpa using mul_le_mul_of_nonneg_left hb1 (sq_nonneg L))
        _ = d * L ^ 2 := by simp
    have hG : ‖gradient f x‖ ^ 2 ≤ L ^ 2 := by
      have h := (hbounds x).1
      nlinarith [norm_nonneg (gradient f x)]
    have htotal : ‖gradient f x‖ ^ 2 + N + N / 2 ≤ C := by
      dsimp [C]
      nlinarith [sq_nonneg L, mul_nonneg (Nat.cast_nonneg d) (sq_nonneg L)]
    have hstep := sgd_step_weak_drift η B f σ x φ 1 hφ2 hH
    rw [sgd_covariance_trace] at hgen
    calc
      |A - η * diffusionGenerator .gradient η B f σ φ x| ≤
          |A - η * D| + η * |diffusionGenerator .gradient η B f σ φ x - D| := hsplit
      _ ≤ (‖gradient f x‖ ^ 2 + N) * η ^ 2 + η * (η * N / 2) :=
        add_le_add (by simpa [A, D, N] using hstep)
          (mul_le_mul_of_nonneg_left hgen hη)
      _ = (‖gradient f x‖ ^ 2 + N + N / 2) * η ^ 2 := by ring
      _ ≤ C * η ^ 2 := mul_le_mul_of_nonneg_right htotal (sq_nonneg η)
  · have htotal : (d : ℝ) + d / 2 ≤ C := by
      dsimp [C]
      nlinarith [sq_nonneg L, mul_nonneg (Nat.cast_nonneg d) (sq_nonneg L)]
    have hstep := sign_step_weak_drift η B f σ x φ 1 hB
      (fun k => ((hbounds x).2 k).1) hφ2 hH
    have htrace := sign_covariance_trace_le η B f σ x hη
    have hgen' : |diffusionGenerator .sign η B f σ φ x - D| ≤ η * d / 2 :=
      hgen.trans (div_le_div_of_nonneg_right htrace (by norm_num))
    calc
      |A - η * diffusionGenerator .sign η B f σ φ x| ≤
          |A - η * D| + η * |diffusionGenerator .sign η B f σ φ x - D| := hsplit
      _ ≤ d * η ^ 2 + η * (η * d / 2) :=
        add_le_add (by simpa [A, D] using hstep)
          (mul_le_mul_of_nonneg_left hgen' hη)
      _ = ((d : ℝ) + d / 2) * η ^ 2 := by ring
      _ ≤ C * η ^ 2 := mul_le_mul_of_nonneg_right htotal (sq_nonneg η)

/-- Nonvacuity of the uniform local estimate's model hypotheses,
Section 4.3, Theorem 1: positive batch size and genuine unit Gaussian noise. -/
example : 0 < (1 : ℕ) ∧ RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) :=
  ⟨by norm_num, regularGaussianModel_flat 1⟩

end Transformer.BatchSize
