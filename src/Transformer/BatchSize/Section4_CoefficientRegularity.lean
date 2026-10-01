/-
# Regularity of the Gaussian model's state coefficients

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The explicit model assumptions give global Lipschitz gradient and noise
scales. This verifies part of the regularity needed for SDE existence;
it does not invoke an unproved stochastic existence theorem.
-/

import Transformer.BatchSize.Section4_Regularity

noncomputable section

namespace Transformer.BatchSize

/-- The corrected regularity assumptions give globally Lipschitz
full gradients and Gaussian noise scales, with uniformly positive noise
coordinates; Section 4.3, Theorem 1. -/
theorem regularGaussianModel_lipschitz {d : ℕ} (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (hmodel : RegularGaussianModel f σ) :
    ∃ (c : ℝ) (L : NNReal), 0 < c ∧ (1 : ℝ) ≤ L ∧
      LipschitzWith L (gradient f) ∧ LipschitzWith L σ ∧
      ∀ x k, c ≤ σ x k ∧ σ x k ≤ L := by
  obtain ⟨hf, hσ, c, L, hc, hL, hcoord, hfd, hσd⟩ := hmodel
  let K : NNReal := ⟨L, by linarith⟩
  have hdf : Differentiable ℝ (fderiv ℝ f) :=
    (hf.fderiv_right (m := 1) (by norm_num)).differentiable (by norm_num)
  have hLipD : LipschitzWith K (fderiv ℝ f) := by
    apply lipschitzWith_of_nnnorm_fderiv_le hdf
    intro x
    change ‖fderiv ℝ (fderiv ℝ f) x‖ ≤ L
    rw [← norm_iteratedFDeriv_one (fderiv ℝ f), norm_iteratedFDeriv_fderiv]
    exact hfd 2 (by norm_num) (by norm_num) x
  refine ⟨c, K, hc, hL, ?_, ?_, hcoord⟩
  · apply lipschitzWith_iff_dist_le_mul.mpr
    intro x y
    rw [dist_eq_norm, dist_eq_norm]
    change ‖(InnerProductSpace.toDual ℝ (EucSpace d)).symm (fderiv ℝ f x) -
      (InnerProductSpace.toDual ℝ (EucSpace d)).symm (fderiv ℝ f y)‖ ≤ _
    rw [← map_sub, LinearIsometryEquiv.norm_map]
    exact hLipD.norm_sub_le x y
  · apply lipschitzWith_of_nnnorm_fderiv_le (hσ.differentiable (by norm_num))
    intro x
    change ‖fderiv ℝ σ x‖ ≤ L
    rw [← norm_iteratedFDeriv_one σ]
    exact hσd 1 le_rfl (by norm_num) x

/-- Nonvacuity of the coefficient-regularity hypotheses,
Section 4.3: a flat loss and positive unit noise. -/
example : RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) := regularGaussianModel_flat 1

/-- The SGD state drift in equation (2) is globally Lipschitz under
the explicit hypotheses of Section 4.3, Theorem 1. -/
theorem sgd_drift_lipschitz {d : ℕ} (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (hmodel : RegularGaussianModel f σ) :
    ∃ L : NNReal, LipschitzWith L (diffusionDrift .gradient B f σ) := by
  obtain ⟨c, L, hc, hL, hg, hσ, hcoord⟩ := regularGaussianModel_lipschitz f σ hmodel
  refine ⟨L, lipschitzWith_iff_dist_le_mul.mpr ?_⟩
  intro x y
  rw [dist_eq_norm, dist_eq_norm]
  change ‖-gradient f x - -gradient f y‖ ≤ _
  rw [neg_sub_neg, norm_sub_rev]
  exact hg.norm_sub_le x y

/-- Nonvacuity of the SGD drift hypotheses, Section 4.3, equation (2). -/
example : RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) := regularGaussianModel_flat 1

end Transformer.BatchSize
