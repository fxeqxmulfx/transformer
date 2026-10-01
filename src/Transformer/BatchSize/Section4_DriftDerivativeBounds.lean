/-
# Uniform second state derivatives of the actual optimizer drift

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The signed response has compact signal/noise parameter range away from
zero denominators. Its true second derivatives therefore have a global bound.
-/

import Transformer.BatchSize.Section4_SignParameters

open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The corrected Gaussian model gives a genuine global second
derivative bound for the actual SGD and SignSGD drift,
Section 4.3 (2)--(3), Theorem 1. The bound is independent of eta,
state and test observable. -/
theorem diffusionDrift_second_derivative_bound {d : ℕ} (method : UpdateKind) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d) (hmodel : RegularGaussianModel f σ) :
    ∃ H : NNReal, ∀ x, ‖iteratedFDeriv ℝ 2 (diffusionDrift method B f σ) x‖ ≤ H := by
  cases method with
  | gradient =>
    obtain ⟨hf, hσ, c, L, hc, hL, hcoord, hfd, hσd⟩ := hmodel
    refine ⟨⟨L, by linarith⟩, fun x => ?_⟩
    change ‖iteratedFDeriv ℝ 2 (-(gradient f)) x‖ ≤ L
    rw [iteratedFDeriv_neg_apply, norm_neg]
    change ‖iteratedFDeriv ℝ 2
      ((InnerProductSpace.toDual ℝ (EucSpace d)).symm ∘ fderiv ℝ f) x‖ ≤ L
    rw [LinearIsometryEquiv.norm_iteratedFDeriv_comp_left, norm_iteratedFDeriv_fderiv]
    exact hfd 3 (by norm_num) (by norm_num) x
  | sign =>
    obtain ⟨c, L, hc, hL, hgL, hcoord, hparam, hder⟩ := regularGaussianModel_parameter_bounds f σ hmodel
    let K : Set (EucSpace d × EucSpace d) :=
      Metric.closedBall 0 (L : ℝ) ×ˢ noiseParameterBox d c L
    have hK : IsCompact K := (isCompact_closedBall (0 : EucSpace d) (L : ℝ)).prod
      (noiseParameterBox_isCompact d c L)
    have hKU : K ⊆ positiveNoiseParameters d := by
      intro p hp k
      obtain ⟨z, hz, heq⟩ := hp.2
      rw [← heq]
      exact hc.trans_le (hz k).1
    have hrange : Set.range (gaussianModelParameters f σ) ⊆ K := by
      rintro _ ⟨x, rfl⟩
      refine ⟨?_, WithLp.ofLp (σ x), (fun k => hcoord x k), ?_⟩
      · simpa only [Metric.mem_closedBall, dist_zero_right, gaussianModelParameters] using hgL x
      · exact WithLp.toLp_ofLp _ _
    have hD (i : ℕ) (hi : 1 ≤ i) (hj : i ≤ 2) (x : EucSpace d) :
        ‖iteratedFDeriv ℝ i (gaussianModelParameters f σ) x‖ ≤ (L : ℝ) ^ i := by
      refine (hder i hi hj x).trans ?_
      rcases (show i = 1 ∨ i = 2 by omega) with rfl | rfl
      · simp
      · nlinarith
    change ∃ H : NNReal, ∀ x,
      ‖iteratedFDeriv ℝ 2 (signedParameterDrift d B ∘ gaussianModelParameters f σ) x‖ ≤ H
    exact compact_range_second_derivative_bound (signedParameterDrift d B)
      (positiveNoiseParameters d) K (positiveNoiseParameters_isOpen d) hK hKU
      (signedParameterDrift_contDiffOn d B) (gaussianModelParameters f σ) hparam hrange L hD

/-- The true drift derivative is globally Lipschitz under the
corrected model, Section 4.3, Theorem 1. This controls second
derivatives of finite deterministic Euler comparison observables. -/
theorem diffusionDrift_fderiv_lipschitz {d : ℕ} (method : UpdateKind) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d) (hmodel : RegularGaussianModel f σ) :
    ∃ H : NNReal, LipschitzWith H (fderiv ℝ (diffusionDrift method B f σ)) := by
  obtain ⟨H, hH⟩ := diffusionDrift_second_derivative_bound method B f σ hmodel
  have hb : ContDiff ℝ 2 (diffusionDrift method B f σ) :=
    (regularGaussianModel_smooth_coefficients method 0 B f σ hmodel).1.of_le (by norm_num)
  refine ⟨H, lipschitzWith_of_nnnorm_fderiv_le
    ((hb.fderiv_right (m := 1) (by norm_num)).differentiable (by norm_num)) ?_⟩
  intro x
  change ‖fderiv ℝ (fderiv ℝ (diffusionDrift method B f σ)) x‖ ≤ H
  rw [← norm_iteratedFDeriv_one (fderiv ℝ (diffusionDrift method B f σ)), norm_iteratedFDeriv_fderiv]
  exact hH x

/-- Joint nonvacuity of both drift derivative-bound hypotheses,
Section 4.3: a flat loss and positive unit coordinate noise. -/
example : RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) := regularGaussianModel_flat 1

end Transformer.BatchSize
