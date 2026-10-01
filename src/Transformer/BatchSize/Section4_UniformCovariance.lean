/-
# Uniform bounds on the SDE covariance

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The corrected bounded-gradient and positive-noise assumptions imply uniform
nondegeneracy for each fixed positive batch size and learning rate.
-/

import Transformer.BatchSize.Section4_Regularity

noncomputable section

namespace Transformer.BatchSize

/-- On the corrected model class, the SDE covariance has uniform
positive lower and finite upper bounds after factoring out eta;
Section 4.3, equations (2)--(3). In particular, finite-signal saturation
does not make the signed diffusion degenerate. -/
theorem diffusionCovariance_uniform_bounds {d : ℕ} (method : UpdateKind)
    (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hB : 0 < B) (hmodel : RegularGaussianModel f σ) :
    ∃ a A : ℝ, 0 < a ∧ 0 ≤ A ∧ ∀ η : ℝ, 0 ≤ η → ∀ x k,
      a * η ≤ diffusionCovariance method η B f σ x k ∧
        diffusionCovariance method η B f σ x k ≤ A * η := by
  have hb : (0 : ℝ) < B := by exact_mod_cast hB
  obtain ⟨M, hM, hbounds⟩ := regularGaussianModel_uniform_bounds f σ hmodel
  obtain ⟨hf, hσ, c, L, hc, hL, hcoord, hfd, hσd⟩ := hmodel
  cases method
  · refine ⟨c ^ 2 / B, L ^ 2 / B, div_pos (sq_pos_of_pos hc) hb, by positivity, ?_⟩
    intro η hη x k
    have hs := hcoord x k
    have hspos : 0 < σ x k := hc.trans_le hs.1
    have hlow : c ^ 2 ≤ (σ x k) ^ 2 := by nlinarith [hs.1]
    have hupp : (σ x k) ^ 2 ≤ L ^ 2 := by nlinarith [hs.2]
    constructor
    · simpa [diffusionCovariance, mul_div_assoc, mul_comm] using
        mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right hlow hb.le) hη
    · simpa [diffusionCovariance, mul_div_assoc, mul_comm] using
        mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right hupp hb.le) hη
  · let R := Real.sqrt ((B : ℝ) / 2) * M / c
    have hR : 0 ≤ R := by positivity
    have he : 0 ≤ errorFunction R := by
      simpa [errorFunction_zero] using errorFunction_strictMono.monotone hR
    have hε : 0 < 1 - errorFunction R ^ 2 := by
      have h := errorFunction_abs_lt_one R
      rw [abs_of_nonneg he] at h
      nlinarith
    refine ⟨1 - errorFunction R ^ 2, 1, hε, by norm_num, ?_⟩
    intro η hη x k
    have hg : |gradient f x k| ≤ M := by
      have hk : |gradient f x k| ≤ ‖gradient f x‖ := by
        simpa [Real.norm_eq_abs] using PiLp.norm_apply_le (gradient f x) k
      exact hk.trans (hbounds x).1
    have hspos : 0 < σ x k := hc.trans_le (hcoord x k).1
    have harg : |Real.sqrt ((B : ℝ) / 2) * gradient f x k / σ x k| ≤ R := by
      rw [abs_div, abs_mul, abs_of_nonneg (Real.sqrt_nonneg _), abs_of_pos hspos]
      calc
        Real.sqrt ((B : ℝ) / 2) * |gradient f x k| / σ x k ≤
            Real.sqrt ((B : ℝ) / 2) * |gradient f x k| / c :=
          div_le_div_of_nonneg_left (by positivity) hc (hcoord x k).1
        _ ≤ Real.sqrt ((B : ℝ) / 2) * M / c :=
          div_le_div_of_nonneg_right
            (mul_le_mul_of_nonneg_left hg (Real.sqrt_nonneg _)) hc.le
    have habs : |signResponse B (σ x k) (gradient f x k)| ≤ errorFunction R := by
      unfold signResponse
      rw [errorFunction_abs]
      exact errorFunction_strictMono.monotone harg
    have hsq : signResponse B (σ x k) (gradient f x k) ^ 2 ≤ errorFunction R ^ 2 := by
      have hp := abs_nonneg (signResponse B (σ x k) (gradient f x k))
      nlinarith [sq_abs (signResponse B (σ x k) (gradient f x k))]
    constructor
    · dsimp [diffusionCovariance]
      nlinarith [mul_nonneg hη (sub_nonneg.mpr hsq)]
    · dsimp [diffusionCovariance]
      nlinarith [mul_nonneg hη (sq_nonneg (signResponse B (σ x k) (gradient f x k)))]

/-- Joint nonvacuity of the uniform covariance assumptions,
Section 4.3: positive batch size, flat loss and positive unit noise. -/
example : 0 < (1 : ℕ) ∧ RegularGaussianModel (fun _ : EucSpace 1 => (0 : ℝ))
    (fun _ => WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) :=
  ⟨by norm_num, regularGaussianModel_flat 1⟩

end Transformer.BatchSize
