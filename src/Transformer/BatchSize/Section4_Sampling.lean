/-
# Gaussian innovations realize the stochastic-gradient law

arXiv:2506.12543v1, Section 4.3, Theorem 1 and its proof sketch.
This connects the finite-step sampler to the Gaussian mean/covariance
computations, rather than only defining matching coefficients.
-/

import Transformer.BatchSize.Section4_DiffusionModel

open MeasureTheory ProbabilityTheory

noncomputable section

namespace Transformer.BatchSize

/-- An affine standard Gaussian has mean mu and variance s^2;
Section 4.3's Gaussian stochastic-gradient model. -/
theorem gaussian_standard_affine_law (μ s : ℝ) :
    (gaussianReal 0 1).map (fun z => μ + s * z) =
      gaussianReal μ (.mk (s ^ 2) (by positivity)) := by
  rw [show (fun z => μ + s * z) = (fun z => μ + z) ∘ (fun z => s * z) from rfl,
    ← Measure.map_map (g := fun z : ℝ => μ + z) (f := fun z : ℝ => s * z)
      (measurable_const.add measurable_id)
      (measurable_const.mul measurable_id),
    gaussianReal_map_const_mul, gaussianReal_map_const_add]
  simp

/-- Each coordinate of the implemented sampler has exactly the
stipulated gradient distribution, Section 4.3, Theorem 1. -/
theorem sampledGradient_coordinate_law {d : ℕ} (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (x : EucSpace d) (k : Fin d) :
    HasLaw (fun z => sampledGradient B f σ x z k)
      (gradientNoiseLaw B (gradient f x k) (σ x k))
      (diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) := by
  have h₀ : HasLaw (fun z : Fin d → ℝ => z k) (gaussianReal 0 1)
      (diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) := by
    simpa [gradientNoiseLaw] using
      diagonalNoiseLaw_coordinate 1 (fun _ : Fin d => 0) (fun _ => 1) k
  have ha : HasLaw (fun z : ℝ => gradient f x k + σ x k / Real.sqrt B * z)
      (gaussianReal (gradient f x k) (.mk ((σ x k / Real.sqrt B) ^ 2) (by positivity)))
      (gaussianReal 0 1) :=
    ⟨(measurable_const.add (measurable_const.mul measurable_id)).aemeasurable,
      gaussian_standard_affine_law _ _⟩
  have hv : NNReal.mk ((σ x k / Real.sqrt B) ^ 2) (by positivity) =
      NNReal.mk ((σ x k) ^ 2 / B) (by positivity) := by
    apply NNReal.coe_injective
    simp only [NNReal.coe_mk, div_pow, Real.sq_sqrt (Nat.cast_nonneg B)]
  simpa [Function.comp_def, sampledGradient, hv, gradientNoiseLaw] using ha.comp h₀

/-- The implemented unsigned step samples an unbiased coordinate
gradient, independently of batch size; Section 4.3, equation (2). -/
theorem sampledGradient_mean {d : ℕ} (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (x : EucSpace d) (k : Fin d) :
    (∫ z, sampledGradient B f σ x z k
      ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) = gradient f x k := by
  rw [(sampledGradient_coordinate_law B f σ x k).integral_eq]
  exact sgd_mean_gradient _ _ _

/-- The implemented signed update has the drift in Theorem 1;
Section 4.3, equation (3), including arbitrary gradient sign. -/
theorem sampledGradient_sign_mean {d : ℕ} (B : ℕ) (f : EucSpace d → ℝ)
    (σ : EucSpace d → EucSpace d) (x : EucSpace d) (k : Fin d)
    (hB : 0 < B) (hσ : 0 < σ x k) :
    (∫ z, Real.sign (sampledGradient B f σ x z k)
      ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1)) =
        signResponse B (σ x k) (gradient f x k) := by
  have h := (sampledGradient_coordinate_law B f σ x k).integral_comp
    measurable_real_sign.aestronglyMeasurable
  exact h.trans (signResponse_eq_mean B _ _ hB hσ)

/-- Nonvacuity of the sampling conditions, Section 4.3. -/
example : 0 < (2 : ℕ) ∧ (0 : ℝ) <
    (fun _ : EucSpace 1 => EuclideanSpace.single (0 : Fin 1) 1) 0 0 := by
  norm_num [EuclideanSpace.single]

end Transformer.BatchSize
