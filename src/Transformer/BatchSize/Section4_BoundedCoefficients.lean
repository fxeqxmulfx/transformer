/-
# Uniform state bounds for the actual optimizer SDE coefficients

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The bounded model gives bounded drift and a noise amplitude proportional
to sqrt(eta), independently of the state or Euler mesh.
-/

import Transformer.BatchSize.Section4_UniformCovariance
import Transformer.BatchSize.Section4_NoiseLipschitz

noncomputable section

namespace Transformer.BatchSize

/-- Uniform drift boundedness under the corrected Gaussian model,
Section 4.3 (2)--(3). The signed drift has bounded coordinates regardless
of the signal, while the SGD drift uses the bounded full gradient. -/
theorem diffusionDrift_uniform_bound {d : ℕ} (method : UpdateKind) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hmodel : RegularGaussianModel f σ) :
    ∃ M : NNReal, ∀ x, ‖diffusionDrift method B f σ x‖ ≤ M := by
  cases method
  · obtain ⟨L, hL, hbound⟩ := regularGaussianModel_uniform_bounds f σ hmodel
    refine ⟨⟨L, by linarith⟩, fun x => ?_⟩
    change ‖-gradient f x‖ ≤ L
    simpa only [diffusionDrift, norm_neg] using (hbound x).1
  · refine ⟨(d : NNReal) + 1, fun x => ?_⟩
    have hs : ‖diffusionDrift .sign B f σ x‖ ^ 2 ≤ d := by
      rw [EuclideanSpace.real_norm_sq_eq]
      calc
        _ ≤ ∑ _ : Fin d, (1 : ℝ) := by
          apply Finset.sum_le_sum
          intro k hk
          have he := errorFunction_abs_lt_one (Real.sqrt ((B : ℝ) / 2) * gradient f x k / σ x k)
          have hn := abs_nonneg (errorFunction (Real.sqrt ((B : ℝ) / 2) * gradient f x k / σ x k))
          change (-errorFunction (Real.sqrt ((B : ℝ) / 2) * gradient f x k / σ x k)) ^ 2 ≤ 1
          rw [neg_sq]
          nlinarith [sq_abs (errorFunction (Real.sqrt ((B : ℝ) / 2) * gradient f x k / σ x k))]
        _ = _ := by simp
    have hd : (0 : ℝ) ≤ d := by positivity
    have hn := norm_nonneg (diffusionDrift .sign B f σ x)
    simp only [NNReal.coe_add, NNReal.coe_natCast, NNReal.coe_one]
    nlinarith

/-- The uniform covariance bound yields an amplitude bound factoring
out sqrt(eta), Section 4.3 (2)--(3). No Euler or SDE limit is assumed. -/
theorem diffusionNoiseScale_uniform_bound {d : ℕ} (method : UpdateKind) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hB : 0 < B) (hmodel : RegularGaussianModel f σ) :
    ∃ A : NNReal, ∀ η : ℝ, 0 ≤ η → ∀ x k,
      |diffusionNoiseScale method η B f σ x k| ≤ Real.sqrt η * A := by
  obtain ⟨c, U, hc, hU, hbound⟩ := diffusionCovariance_uniform_bounds method B f σ hB hmodel
  refine ⟨⟨Real.sqrt U, Real.sqrt_nonneg _⟩, fun η hη x k => ?_⟩
  change |Real.sqrt (diffusionCovariance method η B f σ x k)| ≤ Real.sqrt η * Real.sqrt U
  rw [abs_of_nonneg (Real.sqrt_nonneg _)]
  calc
    _ ≤ Real.sqrt (U * η) := Real.sqrt_le_sqrt (hbound η hη x k).2
    _ = _ := by rw [Real.sqrt_mul hU]; ring

/-- Joint nonvacuity of bounded optimizer-coefficient hypotheses,
Section 4.3: positive batch size, flat loss and unit noise. -/
example : 0 < (1 : ℕ) ∧
    RegularGaussianModel (fun _ : EucSpace 2 => (0 : ℝ))
      (fun _ => WithLp.toLp 2 (fun _ : Fin 2 => (1 : ℝ))) ∧ (0 : ℝ) ≤ 1 / 1000 :=
  ⟨by norm_num, regularGaussianModel_flat 2, by norm_num⟩

end Transformer.BatchSize
