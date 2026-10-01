/-
# Actual discrete updates versus their deterministic mean Euler step

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Only two test derivatives are needed for this comparison. Gaussian
samples remain unbounded; the SGD estimate uses their proved second moment.
-/

import Transformer.BatchSize.Section4_SGDWeakError
import Transformer.BatchSize.Section4_LocalWeakError
import Transformer.BatchSize.Section4_BoundedCoefficients
import Transformer.BatchSize.Section4_DeterministicEulerMap

open MeasureTheory
open scoped BigOperators NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The actual deterministic Euler step has Taylor remainder
R*M^2*eta^2 under the stated drift and Hessian bounds,
Section 3.2 (1), Section 4.3, Theorem 1. -/
theorem deterministicEulerMap_taylor_bound {d : ℕ} (b : EucSpace d → EucSpace d)
    (η : NNReal) (R M : ℝ) (hR : 0 ≤ R) (hM : ∀ x, ‖b x‖ ≤ M)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 2 φ)
    (hH : ∀ x, ‖iteratedFDeriv ℝ 2 φ x‖ ≤ R) (x : EucSpace d) :
    |φ (deterministicEulerMap b η x) - φ x - (η : ℝ) * fderiv ℝ φ x (b x)| ≤
      R * M ^ 2 * (η : ℝ) ^ 2 := by
  have ht := uniform_first_order_taylor φ R hφ hH x ((η : ℝ) • b x)
  have hn : ‖b x‖ ^ 2 ≤ M ^ 2 := by
    have hm := hM x
    nlinarith [norm_nonneg (b x)]
  simp only [norm_smul, Real.norm_eq_abs, abs_of_nonneg η.coe_nonneg,
    mul_pow, map_smul, smul_eq_mul] at ht
  refine ht.trans ?_
  calc
    R * ((η : ℝ) ^ 2 * ‖b x‖ ^ 2) ≤ R * ((η : ℝ) ^ 2 * M ^ 2) :=
      mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hn (sq_nonneg (η : ℝ))) hR
    _ = _ := by ring

/-- Joint nonvacuity of the deterministic Taylor bounds,
Section 4.3: bounded zero drift and a nonzero constant C2 test. -/
example : (0 : ℝ) ≤ 1 ∧
    (∀ x : EucSpace 1, ‖(fun _ : EucSpace 1 => (0 : EucSpace 1)) x‖ ≤ (1 : ℝ)) ∧
    ContDiff ℝ 2 (fun _ : EucSpace 1 => (1 : ℝ)) ∧
    (∀ x : EucSpace 1, ‖iteratedFDeriv ℝ 2 (fun _ : EucSpace 1 => (1 : ℝ)) x‖ ≤ 1) := by
  refine ⟨by norm_num, by simp, contDiff_const, ?_⟩
  intro x
  rw [iteratedFDeriv_const_of_ne (by norm_num)]; simp

/-- The actual SGD and SignSGD expectation has a uniform second-order
defect against its first-order mean drift on arbitrary bounded-Hessian
C2 tests, Section 4.3 (2)--(3), Theorem 1. -/
theorem sgd_sign_uniform_C2_weak_drift {d : ℕ} (method : UpdateKind)
    (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hB : 0 < B) (hmodel : RegularGaussianModel f σ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ η R : ℝ, 0 ≤ R → ∀ φ : EucSpace d → ℝ,
      ContDiff ℝ 2 φ → (∀ y, ‖iteratedFDeriv ℝ 2 φ y‖ ≤ R) → ∀ x,
      |(∫ z, φ (stochasticStep method η B f σ x z)
        ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) - φ x -
          η * fderiv ℝ φ x (diffusionDrift method B f σ x)| ≤ C * R * η ^ 2 := by
  obtain ⟨L, hL, hbound⟩ := regularGaussianModel_uniform_bounds f σ hmodel
  let C : ℝ := ((d : ℝ) + 1) * (L ^ 2 + 1)
  refine ⟨C, by positivity, ?_⟩
  intro η R hR φ hφ hH x
  cases method with
  | gradient =>
    have hb : (0 : ℝ) < B := by exact_mod_cast hB
    have hb1 : (1 : ℝ) ≤ B := by exact_mod_cast hB
    have hN : (∑ k, (σ x k) ^ 2 / B) ≤ d * L ^ 2 := by
      calc
        _ ≤ ∑ _ : Fin d, L ^ 2 := by
          apply Finset.sum_le_sum
          intro k hk
          have hs := abs_le.mp ((hbound x).2 k).2
          have hs2 : (σ x k) ^ 2 ≤ L ^ 2 := by nlinarith
          apply (div_le_iff₀ hb).2
          exact hs2.trans (by simpa using mul_le_mul_of_nonneg_left hb1 (sq_nonneg L))
        _ = d * L ^ 2 := by simp
    have hG : ‖gradient f x‖ ^ 2 ≤ L ^ 2 := by
      have hg := (hbound x).1
      nlinarith [norm_nonneg (gradient f x)]
    have htotal : ‖gradient f x‖ ^ 2 + (∑ k, (σ x k) ^ 2 / B) ≤ C := by
      dsimp [C]
      nlinarith [Nat.cast_nonneg (α := ℝ) d]
    refine (sgd_step_weak_drift η B f σ x φ R hφ hH).trans ?_
    calc
      _ ≤ R * C * η ^ 2 := mul_le_mul_of_nonneg_right
        (mul_le_mul_of_nonneg_left htotal hR) (sq_nonneg η)
      _ = _ := by ring
  | sign =>
    have htotal : (d : ℝ) ≤ C := by
      dsimp [C]
      nlinarith [mul_nonneg (Nat.cast_nonneg (α := ℝ) d) (sq_nonneg L)]
    refine (sign_step_weak_drift η B f σ x φ R hB (fun k => ((hbound x).2 k).1) hφ hH).trans ?_
    calc
      _ ≤ R * C * η ^ 2 := mul_le_mul_of_nonneg_right
        (mul_le_mul_of_nonneg_left htotal hR) (sq_nonneg η)
      _ = _ := by ring

/-- The actual Gaussian optimizer and its deterministic mean Euler
step differ by at most C*R*eta^2 on every C2 test with Hessian bound R,
Section 4.3, Theorem 1. The model constant is independent of the
observable, state and step size. -/
theorem sgd_sign_C2_deterministic_local_error {d : ℕ} (method : UpdateKind)
    (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hB : 0 < B) (hmodel : RegularGaussianModel f σ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ η : NNReal, ∀ R : ℝ, 0 ≤ R → ∀ φ : EucSpace d → ℝ,
      ContDiff ℝ 2 φ → (∀ y, ‖iteratedFDeriv ℝ 2 φ y‖ ≤ R) → ∀ x,
      |(∫ z, φ (stochasticStep method η B f σ x z)
        ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) -
          φ (deterministicEulerMap (diffusionDrift method B f σ) η x)| ≤ C * R * (η : ℝ) ^ 2 := by
  obtain ⟨C, hC, hstep⟩ := sgd_sign_uniform_C2_weak_drift method B f σ hB hmodel
  obtain ⟨M, hM⟩ := diffusionDrift_uniform_bound method B f σ hmodel
  refine ⟨C + (M : ℝ) ^ 2, by positivity, ?_⟩
  intro η R hR φ hφ hH x
  let b := diffusionDrift method B f σ
  have hdet : |φ (deterministicEulerMap b η x) - φ x - (η : ℝ) * fderiv ℝ φ x (b x)| ≤
      R * (M : ℝ) ^ 2 * (η : ℝ) ^ 2 :=
    deterministicEulerMap_taylor_bound b η R M hR hM φ hφ hH x
  have hs := hstep η R hR φ hφ hH x
  let E := ∫ z, φ (stochasticStep method η B f σ x z)
    ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)
  calc
    |E - φ (deterministicEulerMap b η x)| =
        |(E - φ x - (η : ℝ) * fderiv ℝ φ x (b x)) -
          (φ (deterministicEulerMap b η x) - φ x - (η : ℝ) * fderiv ℝ φ x (b x))| := by
      congr 1; ring
    _ ≤ |E - φ x - (η : ℝ) * fderiv ℝ φ x (b x)| +
        |φ (deterministicEulerMap b η x) - φ x - (η : ℝ) * fderiv ℝ φ x (b x)| := abs_sub _ _
    _ ≤ C * R * (η : ℝ) ^ 2 + R * (M : ℝ) ^ 2 * (η : ℝ) ^ 2 := add_le_add hs hdet
    _ = _ := by ring

/-- Joint nonvacuity of both local C2 comparisons, Section 4.3:
flat regular loss, positive batch, positive rate and a nonzero constant test. -/
example : 0 < (1 : ℕ) ∧ RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) ∧ (0 : ℝ) ≤ 1 ∧
    ContDiff ℝ 2 (fun _ : EucSpace 1 => (1 : ℝ)) ∧
    (∀ y, ‖iteratedFDeriv ℝ 2 (fun _ : EucSpace 1 => (1 : ℝ)) y‖ ≤ 1) := by
  refine ⟨by norm_num, regularGaussianModel_flat 1, by norm_num, contDiff_const, ?_⟩
  intro y
  rw [iteratedFDeriv_const_of_ne (by norm_num)]; simp

end Transformer.BatchSize
