/-
# Bochner moments of an unsigned Gaussian gradient

arXiv:2506.12543v1, Section 4.3, equation (2).
The square-integrable full gradient has second moment equal to squared
mean norm plus the trace of the minibatch noise covariance.
-/

import Transformer.BatchSize.Section4_Sampling

open MeasureTheory ProbabilityTheory
open scoped BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- The sampled vector gradient is measurable, Section 4.3. -/
theorem sampledGradient_measurable {d : ℕ} (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (x : EucSpace d) :
    Measurable (sampledGradient B f σ x) := by
  apply (WithLp.measurable_toLp 2 (Fin d → ℝ)).comp
  exact Measurable.of_eval (fun k =>
    measurable_const.add ((measurable_const.div_const _).mul (measurable_pi_apply k)))

/-- Every sampled Gaussian coordinate is square integrable, Section 4.3. -/
theorem sampledGradient_coordinate_memLp {d : ℕ} (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (x : EucSpace d) (k : Fin d) :
    MemLp (fun z => sampledGradient B f σ x z k) 2
      (diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) := by
  apply (sampledGradient_coordinate_law B f σ x k).memLp
  dsimp [gradientNoiseLaw]
  exact memLp_id_gaussianReal 2

/-- The full Gaussian sampled gradient is L2, Section 4.3, equation (2). -/
theorem sampledGradient_memLp {d : ℕ} (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (x : EucSpace d) :
    MemLp (sampledGradient B f σ x) 2
      (diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) := by
  apply (memLp_two_iff_integrable_sq_norm
    (sampledGradient_measurable B f σ x).aestronglyMeasurable).mpr
  simp_rw [EuclideanSpace.real_norm_sq_eq]
  exact integrable_finsetSum _ (fun k _ =>
    (sampledGradient_coordinate_memLp B f σ x k).integrable_sq)

/-- The true vector mean is the full gradient, Section 4.3, equation (2). -/
theorem sampledGradient_vector_mean {d : ℕ} (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (x : EucSpace d) :
    (∫ z, sampledGradient B f σ x z
      ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) = gradient f x := by
  ext k
  have hc := (PiLp.proj (𝕜 := ℝ) 2 (fun _ : Fin d => ℝ) k).integral_comp_comm
    ((sampledGradient_memLp B f σ x).integrable (by norm_num))
  simp only [PiLp.proj_apply] at hc
  rw [← hc, sampledGradient_mean]

/-- Exact coordinate second moment, Section 4.3's Gaussian model. -/
theorem sampledGradient_second_moment {d : ℕ} (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (x : EucSpace d) (k : Fin d) :
    (∫ z, (sampledGradient B f σ x z k) ^ 2
      ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) =
        (gradient f x k) ^ 2 + (σ x k) ^ 2 / B := by
  have hv := (sampledGradient_coordinate_law B f σ x k).variance_eq
  change variance (fun z => sampledGradient B f σ x z k)
      (diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) =
    variance (fun z : ℝ => z) (gradientNoiseLaw B (gradient f x k) (σ x k)) at hv
  rw [sgd_gradient_variance] at hv
  have he := variance_eq_sub (sampledGradient_coordinate_memLp B f σ x k)
  rw [sampledGradient_mean, hv] at he
  simpa only [Pi.pow_apply, add_comm] using (eq_add_of_sub_eq he.symm)

/-- Second moment equals squared gradient norm plus covariance trace,
Section 4.3, equation (2). This controls SGD's local weak Taylor error. -/
theorem sampledGradient_norm_second_moment {d : ℕ} (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (x : EucSpace d) :
    (∫ z, ‖sampledGradient B f σ x z‖ ^ 2
      ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) =
        ‖gradient f x‖ ^ 2 + ∑ k, (σ x k) ^ 2 / B := by
  simp_rw [EuclideanSpace.real_norm_sq_eq]
  rw [integral_finsetSum _ (fun k _ =>
    (sampledGradient_coordinate_memLp B f σ x k).integrable_sq)]
  simp_rw [sampledGradient_second_moment]
  exact Finset.sum_add_distrib

end Transformer.BatchSize
