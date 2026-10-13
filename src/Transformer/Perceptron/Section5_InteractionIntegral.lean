/-
# Differentiating the actual angular interaction integral

ArXiv:2601.21366v2, §5.2, `eq:Hess-bilinear`, and §5.5,
`rem:strictSOPD-perceptron`. The first-variation interaction
force and its curvature are continuous functions of the sphere
point, dominated by the constant `exp β` at positive temperature.
Since the measure is a probability measure, those functions and
their common bound are integrable.

This justifies the derivative under the integral sign and gives
the lower curvature bound `-exp β`. It holds for arbitrary Borel
probability measures, before any finite-support conclusion.
-/

import Transformer.Perceptron.Section5_InteractionCurvature
import Mathlib.Analysis.Calculus.ParametricIntegral

open Real MeasureTheory Filter Set Topology

namespace Transformer.Perceptron

/-- **The actual angular force is integrable against any sphere
probability measure.** Source: arXiv:2601.21366v2, §5.2,
the first variation underlying `eq:Hess-bilinear`. -/
theorem integrable_circleInteractionForce (β : ℝ) (hβ : 0 ≤ β)
    (μ : Perspective.ProbSphere 2) (θ : ℝ) :
    Integrable (circleInteractionForce β θ) (μ : Measure (SSphere 2)) := by
  apply (integrable_const (Real.exp β)).mono'
    (continuous_circleInteractionForce β θ).aestronglyMeasurable
  exact ae_of_all _ fun y => by
    simpa only [Real.norm_eq_abs] using abs_circleInteractionForce_le β hβ θ y

/-- **The actual angular curvature is integrable against any sphere
probability measure.** Source: arXiv:2601.21366v2, §5.2,
the Hessian convolution in `eq:Hess-bilinear`. -/
theorem integrable_circleInteractionCurvature (β : ℝ) (hβ : 0 < β)
    (μ : Perspective.ProbSphere 2) (θ : ℝ) :
    Integrable (circleInteractionCurvature β θ) (μ : Measure (SSphere 2)) := by
  apply (integrable_const (Real.exp β)).mono'
    (continuous_circleInteractionCurvature β θ).aestronglyMeasurable
  exact ae_of_all _ fun y => by
    simpa only [Real.norm_eq_abs] using abs_circleInteractionCurvature_le β hβ θ y

/-- The temperature hypotheses of both integrability statements are
realized at `β=1`. The measure can be the actual Dirac probability
on the nonempty circle, and the angle can be zero. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    Integrable (circleInteractionForce 1 0)
      (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)) ∧
    Integrable (circleInteractionCurvature 1 0)
      (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)) :=
  ⟨zero_le_one, one_pos, integrable_circleInteractionForce 1 zero_le_one _ _,
    integrable_circleInteractionCurvature 1 one_pos _ _⟩

/-- **The actual angular interaction force differentiates under the
integral sign.** Source: arXiv:2601.21366v2, §5.2,
`eq:Hess-bilinear`, and §5.5, `rem:strictSOPD-perceptron`.
The domination is global in the angular variable, so no spatial
density, atomicity, or regularity of the measure is assumed. -/
theorem hasDerivAt_integral_circleInteractionForce (β : ℝ) (hβ : 0 < β)
    (μ : Perspective.ProbSphere 2) (θ : ℝ) :
    HasDerivAt (fun t => ∫ y, circleInteractionForce β t y ∂(μ : Measure (SSphere 2)))
      (∫ y, circleInteractionCurvature β θ y ∂(μ : Measure (SSphere 2))) θ := by
  have h := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := (μ : Measure (SSphere 2))) (F := fun t y => circleInteractionForce β t y)
    (F' := fun t y => circleInteractionCurvature β t y) (bound := fun _ => Real.exp β)
    (s := univ) (x₀ := θ) (by simp)
    (Eventually.of_forall fun t => (continuous_circleInteractionForce β t).aestronglyMeasurable)
    (integrable_circleInteractionForce β hβ.le μ θ)
    (continuous_circleInteractionCurvature β θ).aestronglyMeasurable
    (ae_of_all _ fun y t _ => by
      simpa only [Real.norm_eq_abs] using abs_circleInteractionCurvature_le β hβ t y)
    (integrable_const (Real.exp β))
    (ae_of_all _ fun y t _ => hasDerivAt_circleInteractionForce β t y)
  exact h.2

/-- **The interaction part of the actual first-variation curvature is
bounded below by `-exp β`.** Source: arXiv:2601.21366v2, §5.5,
`rem:strictSOPD-perceptron` and `eq:strictSOPD-sufficient`.
The unit mass of the actual measure is used to integrate the bound. -/
theorem neg_exp_le_integral_circleInteractionCurvature (β : ℝ) (hβ : 0 < β)
    (μ : Perspective.ProbSphere 2) (θ : ℝ) :
    -Real.exp β ≤ ∫ y, circleInteractionCurvature β θ y ∂(μ : Measure (SSphere 2)) := by
  have h := integral_mono (integrable_const (-Real.exp β))
    (integrable_circleInteractionCurvature β hβ μ θ)
    (fun y => neg_exp_le_circleInteractionCurvature β hβ θ y)
  simpa using h

/-- **The integrated curvature also has absolute bound `exp β`.**
Source: arXiv:2601.21366v2, §5.2 and §5.5,
`eq:Hess-bilinear` and `rem:strictSOPD-perceptron`.
Both sides of the actual pointwise bound are integrated against
the probability measure, preserving the sharp normalization. -/
theorem abs_integral_circleInteractionCurvature_le (β : ℝ) (hβ : 0 < β)
    (μ : Perspective.ProbSphere 2) (θ : ℝ) :
    |∫ y, circleInteractionCurvature β θ y ∂(μ : Measure (SSphere 2))| ≤ Real.exp β := by
  apply abs_le.mpr
  refine ⟨neg_exp_le_integral_circleInteractionCurvature β hβ μ θ, ?_⟩
  have h := integral_mono (integrable_circleInteractionCurvature β hβ μ θ)
    (integrable_const (Real.exp β)) (fun y =>
      (le_abs_self _).trans (abs_circleInteractionCurvature_le β hβ θ y))
  simpa using h

/-- The derivative and lower-bound inputs are realized at positive
temperature on the actual Dirac probability on the circle. -/
example : (0 : ℝ) < 1 ∧
    HasDerivAt (fun t => ∫ y, circleInteractionForce 1 t y
      ∂(Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)))
      (∫ y, circleInteractionCurvature 1 0 y
        ∂(Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2))) 0 :=
  ⟨one_pos, hasDerivAt_integral_circleInteractionForce 1 one_pos _ _⟩

/-- **A Dirac mass at the interacting point attains the curvature
lower bound after integration.** Source: arXiv:2601.21366v2, §5.5,
the sharp kernel constant in `rem:strictSOPD-perceptron`.
This verifies the normalization of the measure and the factor `β⁻¹`
before adding the perceptron's curvature. -/
theorem integral_circleInteractionCurvature_dirac_self (β θ : ℝ) :
    (∫ y, circleInteractionCurvature β θ y
      ∂(Perspective.diracProb 2 (circlePoint θ) : Measure (SSphere 2))) = -Real.exp β := by
  change (∫ y, circleInteractionCurvature β θ y ∂Measure.dirac (circlePoint θ)) = _
  rw [integral_dirac, circleInteractionCurvature_self]

/-- **The actual force derivative at this Dirac mass is `-exp β`.**
Source: arXiv:2601.21366v2, §5.2 and §5.5,
`eq:Hess-bilinear` and `rem:strictSOPD-perceptron`.
The result follows from justified differentiation of the actual
integral and an evaluation of its actual probability measure. -/
theorem hasDerivAt_integral_circleInteractionForce_dirac_self (β : ℝ) (hβ : 0 < β)
    (θ : ℝ) : HasDerivAt
      (fun t => ∫ y, circleInteractionForce β t y
        ∂(Perspective.diracProb 2 (circlePoint θ) : Measure (SSphere 2))) (-Real.exp β) θ := by
  simpa only [integral_circleInteractionCurvature_dirac_self] using
    hasDerivAt_integral_circleInteractionForce β hβ (Perspective.diracProb 2 (circlePoint θ)) θ

/-- The temperature restriction in the attained-bound consequence is
satisfied at `β=1`, including at the actual point of angle zero. -/
example : (0 : ℝ) < 1 ∧
    (∫ y, circleInteractionCurvature 1 0 y
      ∂(Perspective.diracProb 2 (circlePoint 0) : Measure (SSphere 2))) = -Real.exp 1 :=
  ⟨one_pos, integral_circleInteractionCurvature_dirac_self 1 0⟩

/-- The same nonempty Dirac example attains the integrated absolute
curvature bound at a positive temperature. -/
example : (0 : ℝ) < 1 ∧
    |∫ y, circleInteractionCurvature 1 0 y
      ∂(Perspective.diracProb 2 (circlePoint 0) : Measure (SSphere 2))| = Real.exp 1 := by
  rw [integral_circleInteractionCurvature_dirac_self, abs_neg, abs_of_pos (Real.exp_pos 1)]
  exact ⟨one_pos, rfl⟩

end Transformer.Perceptron
