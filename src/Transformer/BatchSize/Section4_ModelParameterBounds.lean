/-
# The actual Gaussian model's smooth bounded parameter map

arXiv:2506.12543v1, Section 4.3, Theorem 1's corrected regularity conditions.
The gradient and coordinate standard deviations form the actual arguments
of the signed response; their first two derivatives inherit the model bounds.
-/

import Transformer.BatchSize.Section4_CompactDerivativeBounds

open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The true signal and noise parameters of Section 4.3's signed
Gaussian response, evaluated at the current parameter state. -/
def gaussianModelParameters {d : ℕ} (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (x : EucSpace d) : EucSpace d × EucSpace d := (gradient f x, σ x)

/-- The corrected Gaussian model bounds the true signal/noise
parameter range and its first two derivatives uniformly,
Section 4.3, Theorem 1. The gradient uses loss derivatives through
order three; no boundedness of the loss itself is asserted. -/
theorem regularGaussianModel_parameter_bounds {d : ℕ} (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (hmodel : RegularGaussianModel f σ) :
    ∃ (c : ℝ) (L : NNReal), 0 < c ∧ (1 : ℝ) ≤ L ∧
      (∀ x, ‖gradient f x‖ ≤ L) ∧ (∀ x k, c ≤ σ x k ∧ σ x k ≤ L) ∧
      ContDiff ℝ 2 (gaussianModelParameters f σ) ∧
      (∀ i, 1 ≤ i → i ≤ 2 → ∀ x, ‖iteratedFDeriv ℝ i (gaussianModelParameters f σ) x‖ ≤ L) := by
  obtain ⟨hf, hσ, c, L, hc, hL, hcoord, hfd, hσd⟩ := hmodel
  let K : NNReal := ⟨L, by linarith⟩
  have hg : ContDiff ℝ 2 (gradient f) :=
    (InnerProductSpace.toDual ℝ (EucSpace d)).symm.contDiff.comp
      (hf.fderiv_right (m := 2) (by norm_num))
  have hs : ContDiff ℝ 2 σ := hσ.of_le (by norm_num)
  have hgnorm (i : ℕ) (hi : i ≤ 2) (x : EucSpace d) :
      ‖iteratedFDeriv ℝ i (gradient f) x‖ ≤ L := by
    change ‖iteratedFDeriv ℝ i
      ((InnerProductSpace.toDual ℝ (EucSpace d)).symm ∘ fderiv ℝ f) x‖ ≤ L
    rw [LinearIsometryEquiv.norm_iteratedFDeriv_comp_left, norm_iteratedFDeriv_fderiv]
    exact hfd (i + 1) (by omega) (by omega) x
  refine ⟨c, K, hc, hL, ?_, hcoord, hg.prodMk hs, ?_⟩
  · intro x
    change ‖(InnerProductSpace.toDual ℝ (EucSpace d)).symm (fderiv ℝ f x)‖ ≤ L
    rw [LinearIsometryEquiv.norm_map, ← norm_iteratedFDeriv_one f]
    exact hfd 1 le_rfl (by norm_num) x
  · intro i hi hj x
    exact iteratedFDeriv_prod_norm_le (gradient f) σ hg hs i hj x K (hgnorm i hj x)
      (hσd i hi (by omega) x)

/-- Joint nonvacuity of parameter-bound hypotheses,
Section 4.3: a flat loss with positive unit coordinate standard deviations. -/
example : RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) := regularGaussianModel_flat 1

end Transformer.BatchSize
