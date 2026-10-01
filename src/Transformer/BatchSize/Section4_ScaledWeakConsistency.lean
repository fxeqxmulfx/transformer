/-
# Local weak consistency for bounded smooth tests of any size

arXiv:2506.12543v1, Section 4.3, Theorem 1.
Propagation by a diffusion semigroup can increase derivative bounds.
Rescaling tests makes the local estimate usable with any fixed bound,
rather than assuming preservation of the unit smoothness bounds.
-/

import Transformer.BatchSize.Section4_WeakConsistency

open MeasureTheory

noncomputable section

namespace Transformer.BatchSize

/-- The SDE generator is linear under constant rescaling of a smooth
test, Section 4.3, equations (2)--(3). -/
theorem diffusionGenerator_const_smul {d : ℕ} (method : UpdateKind) (η : ℝ)
    (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (c : ℝ) (φ : EucSpace d → ℝ) (x : EucSpace d) (hφ : ContDiff ℝ 2 φ) :
    diffusionGenerator method η B f σ (fun y => c • φ y) x =
      c * diffusionGenerator method η B f σ φ x := by
  have hd := fderiv_fun_const_smul (hφ.differentiable (by norm_num) x) c
  have hH := iteratedFDeriv_const_smul_apply' (a := c) (x := x) hφ.contDiffAt
  unfold diffusionGenerator
  rw [hd, hH]
  change c * fderiv ℝ φ x (diffusionDrift method B f σ x) +
    (∑ k, diffusionCovariance method η B f σ x k *
      (c * iteratedFDeriv ℝ 2 φ x (fun _ => EuclideanSpace.single k 1))) / 2 = _
  have hs : (∑ k, diffusionCovariance method η B f σ x k *
      (c * iteratedFDeriv ℝ 2 φ x (fun _ => EuclideanSpace.single k 1))) =
      c * ∑ k, diffusionCovariance method η B f σ x k *
        iteratedFDeriv ℝ 2 φ x (fun _ => EuclideanSpace.single k 1) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro k hk
    ring
  rw [hs]
  ring

/-- Nonvacuity of the generator's smooth-test hypothesis,
Section 4.3, equations (2)--(3). -/
example : ContDiff ℝ 2 (fun _ : EucSpace 1 => (0 : ℝ)) := contDiff_const

/-- The uniform local consistency bound scales linearly with any
positive bound on the test and its first six derivatives; Section 4.3,
Theorem 1. The model constant is independent of that bound and of eta. -/
theorem sgd_sign_scaled_local_weak_consistency {d : ℕ} (method : UpdateKind)
    (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hB : 0 < B) (hmodel : RegularGaussianModel f σ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ η : ℝ, 0 ≤ η → ∀ R : ℝ, 0 < R →
      ∀ φ : EucSpace d → ℝ, ContDiff ℝ 6 φ →
        (∀ j ≤ 6, ∀ x, ‖iteratedFDeriv ℝ j φ x‖ ≤ R) → ∀ x,
          |(∫ z, φ (stochasticStep method η B f σ x z)
            ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) - φ x -
              η * diffusionGenerator method η B f σ φ x| ≤ C * R * η ^ 2 := by
  obtain ⟨C, hC, hstep⟩ := sgd_sign_uniform_local_weak_consistency method B f σ hB hmodel
  refine ⟨C, hC, ?_⟩
  intro η hη R hR φ hφ hbound x
  let ψ := fun y => R⁻¹ • φ y
  have hψ : BoundedSmoothTest 6 ψ := by
    refine ⟨ContDiff.const_smul R⁻¹ hφ, ?_⟩
    intro j hj y
    have hjφ : ContDiffAt ℝ j φ y := (hφ.of_le (by exact_mod_cast hj)).contDiffAt
    rw [iteratedFDeriv_const_smul_apply' hjφ, norm_smul, Real.norm_eq_abs,
      abs_of_pos (inv_pos.mpr hR)]
    calc
      R⁻¹ * ‖iteratedFDeriv ℝ j φ y‖ ≤ R⁻¹ * R :=
        mul_le_mul_of_nonneg_left (hbound j hj y) (inv_nonneg.mpr hR.le)
      _ = 1 := inv_mul_cancel₀ hR.ne'
  have hg := diffusionGenerator_const_smul method η B f σ R⁻¹ φ x
    (hφ.of_le (by norm_num))
  have he : (∫ z, ψ (stochasticStep method η B f σ x z)
      ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) - ψ x -
        η * diffusionGenerator method η B f σ ψ x =
      R⁻¹ * ((∫ z, φ (stochasticStep method η B f σ x z)
        ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) - φ x -
          η * diffusionGenerator method η B f σ φ x) := by
    change (∫ z, R⁻¹ • φ (stochasticStep method η B f σ x z)
      ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) - R⁻¹ • φ x -
        η * diffusionGenerator method η B f σ (fun y => R⁻¹ • φ y) x = _
    rw [integral_smul, hg]
    simp only [smul_eq_mul]
    ring
  have h := hstep η hη ψ hψ x
  rw [he, abs_mul, abs_of_pos (inv_pos.mpr hR)] at h
  have h' := mul_le_mul_of_nonneg_left h hR.le
  have hcancel : R * (R⁻¹ * |(∫ z, φ (stochasticStep method η B f σ x z)
      ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) - φ x -
        η * diffusionGenerator method η B f σ φ x|) =
      |(∫ z, φ (stochasticStep method η B f σ x z)
        ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) - φ x -
          η * diffusionGenerator method η B f σ φ x| := by
    rw [← mul_assoc, mul_inv_cancel₀ hR.ne', one_mul]
  rw [hcancel] at h'
  simpa [mul_comm, mul_left_comm, mul_assoc] using h'

/-- Joint nonvacuity of the model and scaled-test hypotheses,
Section 4.3: unit noise and a zero test bounded together with all derivatives. -/
example : 0 < (1 : ℕ) ∧ RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) ∧ (0 : ℝ) < 1 ∧
    ContDiff ℝ 6 (fun _ : EucSpace 1 => (0 : ℝ)) ∧
      (∀ j ≤ 6, ∀ x : EucSpace 1,
        ‖iteratedFDeriv ℝ j (fun _ : EucSpace 1 => (0 : ℝ)) x‖ ≤ 1) := by
  refine ⟨by norm_num, regularGaussianModel_flat 1, by norm_num, contDiff_const, ?_⟩
  intro j hj x
  simp

end Transformer.BatchSize
