/-
# Globally Lipschitz diffusion amplitudes

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The actual square root of the covariance is globally Lipschitz in the state.
Its Lipschitz constant factors as sqrt(eta) times a model-dependent constant.
-/

import Transformer.BatchSize.Section4_SignLipschitz
import Transformer.BatchSize.Section4_SmoothCoefficients

noncomputable section

namespace Transformer.BatchSize

/-- The normalized signed-noise response in Section 4.3 (3), before
the sqrt(eta) prefactor. It is strictly positive at every finite signal. -/
def signNoiseResponse (u : ℝ) : ℝ := Real.sqrt (1 - errorFunction u ^ 2)

/-- Smoothness of the actual signed-noise response, Section 4.3 (3).
The strict finite-signal bound on erf justifies the square root. -/
theorem signNoiseResponse_contDiff : ContDiff ℝ (⊤ : ℕ∞) signNoiseResponse := by
  apply (contDiff_const.sub (errorFunction_contDiff.pow 2)).sqrt
  intro u
  have hu := errorFunction_abs_lt_one u
  have hp := abs_nonneg (errorFunction u)
  have he : 0 < 1 - errorFunction u ^ 2 := by
    nlinarith [sq_abs (errorFunction u)]
  exact he.ne'

/-- A compact range of normalized signals bounds the derivative of
the signed-noise response, Section 4.3 (3). -/
theorem signNoiseResponse_lipschitzOn (R : ℝ) :
    ∃ K : NNReal, LipschitzOnWith K signNoiseResponse (Set.Icc (-R) R) := by
  have hC := signNoiseResponse_contDiff.continuous_deriv (by simp)
  obtain ⟨C, hCbound⟩ := isCompact_Icc.exists_bound_of_continuousOn
    (hC.continuousOn (s := Set.Icc (-R) R))
  refine ⟨C.toNNReal, (convex_Icc (-R) R).lipschitzOnWith_of_nnnorm_deriv_le
    (fun u _ => signNoiseResponse_contDiff.differentiable (by simp) u) ?_⟩
  intro u hu
  rw [← NNReal.coe_le_coe, coe_nnnorm, Real.coe_toNNReal']
  exact (hCbound u hu).trans (le_max_left _ _)

/-- Both Brownian noise amplitudes have a state Lipschitz constant
proportional to sqrt(eta), Section 4.3 (2)--(3). The signed coefficient
uses the bounded normalized signal, not a global Lipschitz claim for sqrt. -/
theorem diffusionNoiseScale_lipschitz {d : ℕ} (method : UpdateKind) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hmodel : RegularGaussianModel f σ) :
    ∃ K : NNReal, ∀ η : ℝ, 0 ≤ η → ∀ k,
      LipschitzWith (‖Real.sqrt η‖₊ * K)
        (fun x => diffusionNoiseScale method η B f σ x k) := by
  cases method
  · obtain ⟨c, L, hc, hL, hg, hs, hcoord⟩ := regularGaussianModel_lipschitz f σ hmodel
    refine ⟨‖Real.sqrt (1 / (B : ℝ))‖₊ * L, fun η hη k => ?_⟩
    have hsk : LipschitzWith L (fun x => σ x k) := by
      apply LipschitzWith.of_dist_le_mul
      intro x y
      have hb : dist (σ x k) (σ y k) ≤ dist (σ x) (σ y) := by
        simpa only [dist_eq_norm, PiLp.sub_apply] using PiLp.norm_apply_le (σ x - σ y) k
      exact hb.trans (hs.dist_le_mul x y)
    have hform (x : EucSpace d) : diffusionNoiseScale .gradient η B f σ x k =
        Real.sqrt η * (Real.sqrt (1 / (B : ℝ)) * σ x k) := by
      have hp : 0 < σ x k := hc.trans_le (hcoord x k).1
      dsimp [diffusionNoiseScale, diffusionCovariance]
      rw [show η * σ x k ^ 2 / B = η * ((1 / B) * σ x k ^ 2) by ring,
        Real.sqrt_mul hη, Real.sqrt_mul (by positivity), Real.sqrt_sq_eq_abs,
        abs_of_pos hp]
    rw [show (fun x => diffusionNoiseScale .gradient η B f σ x k) =
      (fun x => Real.sqrt η * (Real.sqrt (1 / (B : ℝ)) * σ x k)) from funext hform]
    simpa only [Function.comp_def, smul_eq_mul, mul_assoc] using
      (lipschitzWith_smul (Real.sqrt η)).comp
        ((lipschitzWith_smul (Real.sqrt (1 / (B : ℝ)))).comp hsk)
  · obtain ⟨R, hR, hbound⟩ := sign_signal_uniform_bound B f σ hmodel
    obtain ⟨L, hL⟩ := sign_signal_lipschitz B f σ hmodel
    obtain ⟨K, hK⟩ := signNoiseResponse_lipschitzOn R
    have hsignal (k : Fin d) : LipschitzWith (K * L)
        (fun x => signNoiseResponse (Real.sqrt ((B : ℝ) / 2) * gradient f x k / σ x k)) := by
      apply LipschitzWith.of_dist_le_mul
      intro x y
      have hx : Real.sqrt ((B : ℝ) / 2) * gradient f x k / σ x k ∈ Set.Icc (-R) R :=
        abs_le.mp (hbound x k)
      have hy : Real.sqrt ((B : ℝ) / 2) * gradient f y k / σ y k ∈ Set.Icc (-R) R :=
        abs_le.mp (hbound y k)
      exact (hK.dist_le_mul _ hx _ hy).trans (by
        simpa only [NNReal.coe_mul, mul_assoc] using
          mul_le_mul_of_nonneg_left ((hL k).dist_le_mul x y) K.coe_nonneg)
    refine ⟨K * L, fun η hη k => ?_⟩
    have hform (x : EucSpace d) : diffusionNoiseScale .sign η B f σ x k =
        Real.sqrt η * signNoiseResponse (Real.sqrt ((B : ℝ) / 2) * gradient f x k / σ x k) := by
      exact Real.sqrt_mul hη _
    rw [show (fun x => diffusionNoiseScale .sign η B f σ x k) =
      (fun x => Real.sqrt η * signNoiseResponse
        (Real.sqrt ((B : ℝ) / 2) * gradient f x k / σ x k)) from funext hform]
    simpa only [Function.comp_def, smul_eq_mul] using
      (lipschitzWith_smul (Real.sqrt η)).comp (hsignal k)

/-- Joint nonvacuity of model and rate hypotheses, Section 4.3 (2)--(3). -/
example : RegularGaussianModel (fun _ : EucSpace 2 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 2 => (1 : ℝ))) ∧ (0 : ℝ) ≤ 1 / 1000 :=
  ⟨regularGaussianModel_flat 2, by norm_num⟩

end Transformer.BatchSize
