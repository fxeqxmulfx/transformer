/-
# Globally Lipschitz signed drift and bounded normalized signal

arXiv:2506.12543v1, Section 4.3, equation (3), Theorem 1.
The normalization by the positive state-dependent Gaussian noise and the
actual error function preserve global Lipschitz regularity.
-/

import Transformer.BatchSize.Section4_QuotientLipschitz

noncomputable section

namespace Transformer.BatchSize

/-- The normalized signed signal has a common global Lipschitz constant
in all coordinates under the corrected Section 4.3 model hypotheses.
The batch-size factor is the actual sqrt(B/2) from equation (3). -/
theorem sign_signal_lipschitz {d : ℕ} (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (hmodel : RegularGaussianModel f σ) :
    ∃ K : NNReal, ∀ k : Fin d, LipschitzWith K
      (fun x => Real.sqrt ((B : ℝ) / 2) * gradient f x k / σ x k) := by
  obtain ⟨M, hM, hbound⟩ := regularGaussianModel_uniform_bounds f σ hmodel
  obtain ⟨c, L, hc, hL, hg, hs, hcoord⟩ := regularGaussianModel_lipschitz f σ hmodel
  let Q : NNReal := Real.toNNReal ((L : ℝ) / c + M * L / c ^ 2)
  refine ⟨‖Real.sqrt ((B : ℝ) / 2)‖₊ * Q, fun k => ?_⟩
  have hgk : LipschitzWith L (fun x => gradient f x k) := by
    apply LipschitzWith.of_dist_le_mul
    intro x y
    have hb : dist (gradient f x k) (gradient f y k) ≤ dist (gradient f x) (gradient f y) := by
      simpa only [dist_eq_norm, PiLp.sub_apply] using
        PiLp.norm_apply_le (gradient f x - gradient f y) k
    exact hb.trans (hg.dist_le_mul x y)
  have hsk : LipschitzWith L (fun x => σ x k) := by
    apply LipschitzWith.of_dist_le_mul
    intro x y
    have hb : dist (σ x k) (σ y k) ≤ dist (σ x) (σ y) := by
      simpa only [dist_eq_norm, PiLp.sub_apply] using PiLp.norm_apply_le (σ x - σ y) k
    exact hb.trans (hs.dist_le_mul x y)
  have hgb (x : EucSpace d) : |gradient f x k| ≤ M := by
    have hb : |gradient f x k| ≤ ‖gradient f x‖ := by
      simpa only [Real.norm_eq_abs] using PiLp.norm_apply_le (gradient f x) k
    exact hb.trans (hbound x).1
  have hq := positive_quotient_lipschitz (fun x => gradient f x k) (fun x => σ x k)
    L L M c hgk hsk (by linarith) hc hgb (fun x => (hcoord x k).1)
  convert (lipschitzWith_smul (Real.sqrt ((B : ℝ) / 2))).comp hq using 1
  simp only [Function.comp_def, smul_eq_mul, mul_div_assoc]

/-- Joint nonvacuity of signed-signal regularity, Section 4.3 (3). -/
example : RegularGaussianModel (fun _ : EucSpace 2 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 2 => (1 : ℝ))) := regularGaussianModel_flat 2

/-- All normalized signed signals stay in one compact real interval,
Section 4.3 (3). This bound controls smooth nonlinear functions of them. -/
theorem sign_signal_uniform_bound {d : ℕ} (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (hmodel : RegularGaussianModel f σ) :
    ∃ R : ℝ, 0 ≤ R ∧ ∀ x k,
      |Real.sqrt ((B : ℝ) / 2) * gradient f x k / σ x k| ≤ R := by
  obtain ⟨M, hM, hbound⟩ := regularGaussianModel_uniform_bounds f σ hmodel
  obtain ⟨hf, hs, c, L, hc, hL, hcoord, hfd, hsd⟩ := hmodel
  refine ⟨Real.sqrt ((B : ℝ) / 2) * M / c, by positivity, fun x k => ?_⟩
  have hp : 0 < σ x k := hc.trans_le (hcoord x k).1
  have hg : |gradient f x k| ≤ M := by
    have hb : |gradient f x k| ≤ ‖gradient f x‖ := by
      simpa only [Real.norm_eq_abs] using PiLp.norm_apply_le (gradient f x) k
    exact hb.trans (hbound x).1
  rw [abs_div, abs_mul, abs_of_nonneg (Real.sqrt_nonneg _), abs_of_pos hp]
  exact (div_le_div_of_nonneg_right
    (mul_le_mul_of_nonneg_left hg (Real.sqrt_nonneg _)) hp.le).trans
      (div_le_div_of_nonneg_left (by positivity) hc (hcoord x k).1)

/-- Nonvacuity of the compact-signal bound, Section 4.3 (3). -/
example : RegularGaussianModel (fun _ : EucSpace 2 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 2 => (1 : ℝ))) := regularGaussianModel_flat 2

/-- The signed response itself is globally Lipschitz in the full state,
with one constant for all coordinates, Section 4.3 (3). -/
theorem sign_response_lipschitz {d : ℕ} (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (hmodel : RegularGaussianModel f σ) :
    ∃ K : NNReal, ∀ k, LipschitzWith K (fun x => signResponse B (σ x k) (gradient f x k)) := by
  obtain ⟨K, hK⟩ := sign_signal_lipschitz B f σ hmodel
  exact ⟨Real.toNNReal (2 / Real.sqrt Real.pi) * K, fun k =>
    errorFunction_lipschitz.comp (hK k)⟩

/-- Nonvacuity of the response hypotheses, Section 4.3 (3). -/
example : RegularGaussianModel (fun _ : EucSpace 2 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 2 => (1 : ℝ))) := regularGaussianModel_flat 2

/-- Both actual SDE drifts are globally Lipschitz under the corrected
model, Section 4.3 (2)--(3), including normalization in the signed case. -/
theorem diffusionDrift_lipschitz {d : ℕ} (method : UpdateKind) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hmodel : RegularGaussianModel f σ) :
    ∃ L : NNReal, LipschitzWith L (diffusionDrift method B f σ) := by
  cases method
  · exact sgd_drift_lipschitz B f σ hmodel
  · obtain ⟨K, hK⟩ := sign_response_lipschitz B f σ hmodel
    have hv := coordinate_lipschitz_to_euclidean
      (fun x k => signResponse B (σ x k) (gradient f x k)) K hK
    exact ⟨_, hv.neg⟩

/-- Nonvacuity of the drift hypotheses, Section 4.3 (2)--(3). -/
example : RegularGaussianModel (fun _ : EucSpace 2 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 2 => (1 : ℝ))) := regularGaussianModel_flat 2

end Transformer.BatchSize
