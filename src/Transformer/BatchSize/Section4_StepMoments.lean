/-
# Vector moments of a signed step

arXiv:2506.12543v1, Section 4.3, Theorem 1.
The exact drift is a Bochner expectation of the full vector update,
with integrability proved from its bounded coordinates.
-/

import Transformer.BatchSize.Section4_Sampling

open MeasureTheory ProbabilityTheory
open scoped BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- The signed stochastic gradient vector applied by SignSGD,
Section 4.3, Theorem 1. -/
def signedGradient {d : ℕ} (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (x : EucSpace d) (z : Fin d → ℝ) : EucSpace d :=
  WithLp.toLp 2 (fun k => Real.sign (sampledGradient B f σ x z k))

/-- Signed vector steps have squared norm at most d, Section 4.3. -/
theorem signedGradient_norm_sq {d : ℕ} (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (x : EucSpace d) (z : Fin d → ℝ) :
    ‖signedGradient B f σ x z‖ ^ 2 ≤ d := by
  rw [EuclideanSpace.real_norm_sq_eq]
  calc
    ∑ k, (signedGradient B f σ x z k) ^ 2 ≤ ∑ _ : Fin d, (1 : ℝ) := by
      apply Finset.sum_le_sum
      intro k hk
      have h := abs_le.mp (sign_abs_le_one (sampledGradient B f σ x z k))
      change Real.sign (sampledGradient B f σ x z k) ^ 2 ≤ 1
      nlinarith [h.1, h.2]
    _ = d := by simp

/-- The full signed vector is measurable, Section 4.3, Theorem 1. -/
theorem signedGradient_measurable {d : ℕ} (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (x : EucSpace d) :
    Measurable (signedGradient B f σ x) := by
  apply (WithLp.measurable_toLp 2 (Fin d → ℝ)).comp
  apply Measurable.of_eval
  intro k
  exact measurable_real_sign.comp
    (measurable_const.add ((measurable_const.div_const _).mul (measurable_pi_apply k)))

/-- Bounded signed updates are Bochner integrable, Section 4.3. -/
theorem signedGradient_integrable {d : ℕ} (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (x : EucSpace d) :
    Integrable (signedGradient B f σ x)
      (diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) := by
  apply Integrable.mono' (integrable_const ((d : ℝ) + 1))
    (signedGradient_measurable B f σ x).aestronglyMeasurable
  apply ae_of_all
  intro z
  have hn := signedGradient_norm_sq B f σ x z
  have hd : (0 : ℝ) ≤ d := Nat.cast_nonneg d
  nlinarith [norm_nonneg (signedGradient B f σ x z)]

/-- The exact vector expectation is minus the SDE drift, Section 4.3,
equation (3). This is an integral of the implemented signed update. -/
theorem signedGradient_mean {d : ℕ} (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (x : EucSpace d)
    (hB : 0 < B) (hσ : ∀ k, 0 < σ x k) :
    (∫ z, signedGradient B f σ x z
      ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) = -diffusionDrift .sign B f σ x := by
  ext k
  have hc := (PiLp.proj (𝕜 := ℝ) 2 (fun _ : Fin d => ℝ) k).integral_comp_comm
    (signedGradient_integrable B f σ x)
  simp only [PiLp.proj_apply] at hc
  rw [← hc]
  simpa [signedGradient, diffusionDrift] using sampledGradient_sign_mean B f σ x k hB (hσ k)

/-- Nonvacuity of the exact vector drift, Section 4.3. -/
example : 0 < (1 : ℕ) ∧
    ∀ k : Fin 1, (0 : ℝ) < (EuclideanSpace.single (0 : Fin 1) 1) k := by
  constructor
  · norm_num
  · intro k
    fin_cases k
    simp [EuclideanSpace.single]

end Transformer.BatchSize
