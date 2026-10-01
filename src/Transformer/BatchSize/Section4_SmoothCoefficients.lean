/-
# Smooth drift, covariance and diffusion amplitude

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The corrected model hypotheses give seven continuous state derivatives of
both SDEs' actual coefficients. Strict ellipticity makes the square root of
each diagonal covariance smooth at every state for a positive learning rate.
These are coefficient results, not an assumed stochastic existence theorem.
-/

import Transformer.BatchSize.Section4_UniformCovariance

noncomputable section

namespace Transformer.BatchSize

/-- Under the corrected Section 4.3 hypotheses, both optimizer drifts
and diagonal covariance entries are C7 functions of the state.
The gradient loses one derivative from the C8 loss. -/
theorem regularGaussianModel_smooth_coefficients {d : ℕ} (method : UpdateKind)
    (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hmodel : RegularGaussianModel f σ) :
    ContDiff ℝ 7 (diffusionDrift method B f σ) ∧
      ∀ k, ContDiff ℝ 7 (fun x => diffusionCovariance method η B f σ x k) := by
  have hg : ContDiff ℝ 7 (gradient f) :=
    (InnerProductSpace.toDual ℝ (EucSpace d)).symm.contDiff.comp
      (hmodel.1.fderiv_right (m := 7) (by norm_num))
  have hσ : ContDiff ℝ 7 σ := hmodel.2.1.of_le (by norm_num)
  have hgk (k : Fin d) : ContDiff ℝ 7 (fun x => gradient f x k) :=
    (contDiff_piLp_apply (𝕜 := ℝ) (i := k) 2).comp hg
  have hσk (k : Fin d) : ContDiff ℝ 7 (fun x => σ x k) :=
    (contDiff_piLp_apply (𝕜 := ℝ) (i := k) 2).comp hσ
  obtain ⟨c, L, hc, hL, hcoord, hfd, hσd⟩ := hmodel.2.2
  have hσne (x : EucSpace d) (k : Fin d) : σ x k ≠ 0 :=
    (hc.trans_le (hcoord x k).1).ne'
  cases method with
  | gradient =>
    refine ⟨hg.neg, fun k => ?_⟩
    exact (contDiff_const.mul ((hσk k).pow 2)).div_const (B : ℝ)
  | sign =>
    have hsign (k : Fin d) : ContDiff ℝ 7 (fun x => signResponse B (σ x k) (gradient f x k)) := by
      exact (errorFunction_contDiff.of_le (by simp)).comp
        ((contDiff_const.mul (hgk k)).div (hσk k) (fun x => hσne x k))
    have hv : ContDiff ℝ 7 (fun x => WithLp.toLp 2
        (fun k : Fin d => signResponse B (σ x k) (gradient f x k))) :=
      (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.contDiff.comp
        (contDiff_pi.mpr hsign)
    exact ⟨hv.neg, fun k => contDiff_const.mul (contDiff_const.sub ((hsign k).pow 2))⟩

/-- Joint nonvacuity of the smooth-coefficient model, Section 4.3. -/
example : RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) := regularGaussianModel_flat 1

/-- Diagonal Brownian noise amplitude, Section 4.3, equations (2)--(3).
Its square is the stated covariance, rather than the covariance itself. -/
def diffusionNoiseScale {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (x : EucSpace d) (k : Fin d) : ℝ :=
  Real.sqrt (diffusionCovariance method η B f σ x k)

/-- The diffusion amplitude reproduces the actual diagonal covariance,
Section 4.3, equations (2)--(3), for every nonnegative learning rate. -/
theorem diffusionNoiseScale_sq {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (x : EucSpace d) (k : Fin d) (hη : 0 ≤ η) :
    diffusionNoiseScale method η B f σ x k ^ 2 = diffusionCovariance method η B f σ x k :=
  Real.sq_sqrt (diffusionCovariance_nonneg method η B f σ x k hη)

/-- Nonvacuity of the amplitude-square hypothesis, Section 4.3. -/
example : (0 : ℝ) ≤ 1 / 1000 := by norm_num

/-- Positive learning rate and the corrected regularity hypotheses give
smooth diffusion amplitudes throughout state space, Section 4.3, Theorem 1.
Uniform positivity was proved independently for the signed covariance. -/
theorem regularGaussianModel_smooth_noiseScale {d : ℕ} (method : UpdateKind)
    (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 < η) (hB : 0 < B) (hmodel : RegularGaussianModel f σ) (k : Fin d) :
    ContDiff ℝ 7 (fun x => diffusionNoiseScale method η B f σ x k) := by
  obtain ⟨a, A, ha, hA, hbound⟩ := diffusionCovariance_uniform_bounds method B f σ hB hmodel
  exact ((regularGaussianModel_smooth_coefficients method η B f σ hmodel).2 k).sqrt
    (fun x => ((mul_pos ha hη).trans_le (hbound η hη.le x k).1).ne')

/-- Joint nonvacuity of all smooth-amplitude hypotheses, Section 4.3. -/
example : (0 : ℝ) < 1 / 1000 ∧ 0 < (1 : ℕ) ∧
    RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
      (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) :=
  ⟨by norm_num, by norm_num, regularGaussianModel_flat 1⟩

end Transformer.BatchSize
