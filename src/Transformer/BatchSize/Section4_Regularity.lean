/-
# Explicit regularity for weak approximation

arXiv:2506.12543v1, Section 4.3, Theorem 1.
The source's missing regularity conditions are stated separately from
the approximation claim, and their nonvacuity and moment bounds are proved.
-/

import Transformer.BatchSize.Section4_DiffusionModel

noncomputable section

namespace Transformer.BatchSize

/-- Smooth bounded test functions, with derivatives normalized to one.
This makes the constant in Section 4.3's weak-error bound uniform over tests. -/
def BoundedSmoothTest {d : ℕ} (n : ℕ) (φ : EucSpace d → ℝ) : Prop :=
  ContDiff ℝ n φ ∧ ∀ j ≤ n, ∀ x, ‖iteratedFDeriv ℝ j φ x‖ ≤ 1

/-- Explicit sufficient regularity for the SDE assertion in Section 4.3.
The source states only Gaussianity and diagonal covariance. We additionally
require bounded loss derivatives of orders one through eight, bounded noise
derivatives through order eight, and uniformly positive coordinate standard
deviations, rather than asserting approximation for an arbitrary nonsmooth
loss or a singular inverse covariance. -/
def RegularGaussianModel {d : ℕ}
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d) : Prop :=
  ContDiff ℝ 8 f ∧ ContDiff ℝ 8 σ ∧
    ∃ c L : ℝ, 0 < c ∧ 1 ≤ L ∧
      (∀ x k, c ≤ σ x k ∧ σ x k ≤ L) ∧
      (∀ j, 1 ≤ j → j ≤ 8 → ∀ x, ‖iteratedFDeriv ℝ j f x‖ ≤ L) ∧
      (∀ j, 1 ≤ j → j ≤ 8 → ∀ x, ‖iteratedFDeriv ℝ j σ x‖ ≤ L)

/-- Flat losses and unit Gaussian coordinate noise satisfy the explicit
regularity conditions of the corrected Section 4.3, Theorem 1. -/
theorem regularGaussianModel_flat (d : ℕ) :
    RegularGaussianModel (fun _ : EucSpace d => (0 : ℝ))
      (fun _ => WithLp.toLp 2 (fun _ : Fin d => (1 : ℝ))) := by
  refine ⟨contDiff_const, contDiff_const, 1, 1, by norm_num, le_rfl, ?_, ?_, ?_⟩
  · intro x k
    exact ⟨le_rfl, le_rfl⟩
  · intro j hj hupper x
    rw [iteratedFDeriv_const_of_ne (by omega)]
    simp
  · intro j hj hupper x
    rw [iteratedFDeriv_const_of_ne (by omega)]
    simp

/-- The regularity assumptions imply uniform gradient/noise bounds
needed for the local weak estimates; Section 4.3, Theorem 1. -/
theorem regularGaussianModel_uniform_bounds {d : ℕ} (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (hmodel : RegularGaussianModel f σ) :
    ∃ L : ℝ, 1 ≤ L ∧ ∀ x, ‖gradient f x‖ ≤ L ∧
      ∀ k, 0 < σ x k ∧ |σ x k| ≤ L := by
  obtain ⟨hf, hσ, c, L, hc, hL, hcoord, hfd, hσd⟩ := hmodel
  refine ⟨L, hL, fun x => ⟨?_, ?_⟩⟩
  · change ‖(InnerProductSpace.toDual ℝ (EucSpace d)).symm (fderiv ℝ f x)‖ ≤ L
    rw [LinearIsometryEquiv.norm_map, ← norm_iteratedFDeriv_one f]
    exact hfd 1 le_rfl (by norm_num) x
  · intro k
    have hpos := hc.trans_le (hcoord x k).1
    exact ⟨hpos, by rw [abs_of_pos hpos]; exact (hcoord x k).2⟩

/-- Nonvacuity of the uniform bounds, Section 4.3. -/
example : RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) := regularGaussianModel_flat 1

end Transformer.BatchSize
