/-
# Curvature of the actual angular first variation

ArXiv:2601.21366v2, §5.5, `rem:strictSOPD-perceptron` and
`eq:strictSOPD-sufficient`. The first variation's angular second
derivative is the integrated interaction curvature plus one half
of the perceptron potential's actual second angular derivative.

The integrated interaction has lower bound `-exp β`. Therefore
the source's potential curvature bound `2 exp β+2κ` gives actual
first-variation curvature at least `κ`. This is established for
an arbitrary sphere probability measure before using any
atomicity theorem or any finite Dirac presentation.
-/

import Transformer.Perceptron.Section5_AngularFirstVariation
import Transformer.Perceptron.Section5_AngularSupport
import Transformer.Perceptron.Section5_StrictCurvatureWitness

open scoped BigOperators
open Real MeasureTheory Filter

namespace Transformer.Perceptron

/-- **The first angular derivative of the potential is differentiable.**
Source: arXiv:2601.21366v2, §5.5, the differentiations in
`rem:strictSOPD-perceptron`. Only the pointwise derivative
identities `φ'=2σ` and the actual derivative of `σ` are required;
continuity of `σ'` is not added. -/
theorem differentiableAt_angular_potential_derivative (φ σ σ' : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s)
    (hσ : ∀ s : ℝ, HasDerivAt σ (σ' s) s)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (θ : ℝ) :
    DifferentiableAt ℝ (deriv (fun t => potential φ ω a (circlePoint t : EucSpace 2))) θ := by
  rw [deriv_potential_circlePoint φ σ hφ]
  have hterm (j : Idx 2) : DifferentiableAt ℝ
      (fun t => ω j * (σ (inner (𝕜 := ℝ) (a j) (circlePoint t : EucSpace 2)) *
        inner (𝕜 := ℝ) (a j) (circleVel t))) θ := by
    have hcomp := (hσ _).differentiableAt.comp θ
      (hasDerivAt_inner_circlePoint (a j) θ).differentiableAt
    exact (hcomp.fun_mul (hasDerivAt_inner_circleVel (a j) θ).differentiableAt).const_mul (ω j)
  exact (DifferentiableAt.sum (fun j (_ : j ∈ Finset.univ) => hterm j)).const_mul (2 : ℝ)

/-- **The actual second derivative of the angular first variation.**
Source: arXiv:2601.21366v2, §5.5, `rem:strictSOPD-perceptron`.
The interaction integral has already been differentiated under
global domination; the potential has a genuine second derivative. -/
theorem hasDerivAt_deriv_angularFirstVariation (β : ℝ) (hβ : 0 < β)
    (φ σ σ' : ℝ → ℝ) (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s)
    (hσ : ∀ s : ℝ, HasDerivAt σ (σ' s) s)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2) (θ : ℝ) :
    HasDerivAt (deriv (angularFirstVariation β φ ω a μ))
      ((∫ y, circleInteractionCurvature β θ y ∂(μ : Measure (SSphere 2))) +
        (1 / 2 : ℝ) * deriv^[2] (fun t => potential φ ω a (circlePoint t : EucSpace 2)) θ) θ := by
  have hI := hasDerivAt_integral_circleInteractionForce β hβ μ θ
  have hV := (differentiableAt_angular_potential_derivative φ σ σ' hφ hσ ω a θ).hasDerivAt
  have h := hI.fun_add (hV.const_mul (1 / 2 : ℝ))
  exact h.congr_of_eventuallyEq (Eventually.of_forall fun t =>
    deriv_angularFirstVariation_eq_force β hβ.ne' φ σ hφ ω a μ t)

/-- **The second-derivative value is the stated interaction-potential sum.**
Source: arXiv:2601.21366v2, §5.5, the curvature decomposition
in `rem:strictSOPD-perceptron`. All derivatives are of the actual
energy and primitive, not values supplied by a Hessian predicate. -/
theorem secondDeriv_angularFirstVariation (β : ℝ) (hβ : 0 < β)
    (φ σ σ' : ℝ → ℝ) (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s)
    (hσ : ∀ s : ℝ, HasDerivAt σ (σ' s) s)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2) (θ : ℝ) :
    deriv^[2] (angularFirstVariation β φ ω a μ) θ =
      (∫ y, circleInteractionCurvature β θ y ∂(μ : Measure (SSphere 2))) +
        (1 / 2 : ℝ) * deriv^[2] (fun t => potential φ ω a (circlePoint t : EucSpace 2)) θ := by
  exact (hasDerivAt_deriv_angularFirstVariation β hβ φ σ σ' hφ hσ ω a μ θ).deriv

/-- The derivative and decomposition hypotheses are realized by
`β=1`, `φ(s)=s²`, `σ(s)=s`, and its actual derivative `σ'=1`.
The measure and neuron vectors may be any actual sphere measure
and actual vectors, so no curvature or atomicity input is hidden. -/
example : (0 : ℝ) < 1 ∧
    (∀ s : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * s) s) ∧
    (∀ s : ℝ, HasDerivAt (fun t : ℝ => t) ((fun _ : ℝ => (1 : ℝ)) s) s) := by
  refine ⟨one_pos, fun s => ?_, fun s => hasDerivAt_id' (𝕜 := ℝ) (x := s)⟩
  simpa using hasDerivAt_pow 2 s

/-- **The source's sufficient potential-curvature inequality gives
first-variation curvature at least `κ`.** Source: arXiv:2601.21366v2,
§5.5, `eq:strictSOPD-sufficient`. The result is pointwise and does
not need stationarity; the support condition enters when it is applied. -/
theorem secondDeriv_angularFirstVariation_ge (β : ℝ) (hβ : 0 < β)
    (φ σ σ' : ℝ → ℝ) (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s)
    (hσ : ∀ s : ℝ, HasDerivAt σ (σ' s) s)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2) (κ θ : ℝ)
    (hge : 2 * Real.exp β + 2 * κ ≤
      deriv^[2] (fun t => potential φ ω a (circlePoint t : EucSpace 2)) θ) :
    κ ≤ deriv^[2] (angularFirstVariation β φ ω a μ) θ := by
  rw [secondDeriv_angularFirstVariation β hβ φ σ σ' hφ hσ]
  have h := neg_exp_le_integral_circleInteractionCurvature β hβ μ θ
  linarith

/-- All inputs of the pointwise curvature bound are realized at
`β=1`, `κ=1`, the quadratic primitive, and the source's one-neuron
potential `-4cos²θ`, evaluated at the actual angle zero.
Its actual angular second derivative is eight, and `2 exp 1+2<8`. -/
example : (0 : ℝ) < 1 ∧
    (∀ s : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * s) s) ∧
    (∀ s : ℝ, HasDerivAt (fun t : ℝ => t) ((fun _ : ℝ => (1 : ℝ)) s) s) ∧
    2 * Real.exp 1 + 2 * 1 ≤ deriv^[2]
      (fun t : ℝ => potential (fun s : ℝ => s ^ 2) (Pi.single 0 (-4 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))) (circlePoint t : EucSpace 2)) 0 := by
  have hφ : ∀ s : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * (fun s : ℝ => s) s) s :=
    fun s => by simpa using hasDerivAt_pow 2 s
  have hσ : ∀ s : ℝ, HasDerivAt (fun t : ℝ => t) ((fun _ : ℝ => (1 : ℝ)) s) s :=
    fun s => hasDerivAt_id' (𝕜 := ℝ) (x := s)
  refine ⟨one_pos, hφ, hσ, ?_⟩
  rw [secondDeriv_potential_circlePoint (fun s : ℝ => s ^ 2) (fun s : ℝ => s)
    (fun _ : ℝ => (1 : ℝ)) hφ hσ]
  norm_num [Fin.sum_univ_two, circlePoint, greatCircle, circleVel,
    inner_neg_left, inner_neg_right]
  linarith [Real.exp_one_lt_three]

/-- **The source's strict angular curvature condition forces finite
atomicity of a stationary measure.** Source: arXiv:2601.21366v2,
§5.5, `rem:strictSOPD-perceptron` and `eq:strictSOPD-sufficient`.
Stationarity makes the actual first variation critical on support,
and the stated curvature condition makes its actual second angular
derivative positive there. The compactness and measure-theory
lemmas then give a genuine finite Dirac presentation.
No analytic activation or pre-existing atomicity result is used. -/
theorem isFinitelyAtomic_of_strict_angular_curvature (β : ℝ) (hβ : 0 < β)
    (φ σ σ' : ℝ → ℝ) (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s)
    (hσ : ∀ s : ℝ, HasDerivAt σ (σ' s) s)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hμ : IsStationary β σ ω a μ) (κ : ℝ) (hκ : 0 < κ)
    (hge : ∀ θ : ℝ, circlePoint θ ∈ (μ : Measure (SSphere 2)).support →
      2 * Real.exp β + 2 * κ ≤
        deriv^[2] (fun t => potential φ ω a (circlePoint t : EucSpace 2)) θ) :
    IsFinitelyAtomic μ := by
  apply isFinitelyAtomic_of_positive_angular_curvature μ (angularFirstVariation β φ ω a μ)
  · exact fun θ hθ => deriv_angularFirstVariation_zero_on_support β hβ.ne'
      φ σ hφ ω a μ hμ θ hθ
  · exact fun θ hθ => hκ.trans_le
      (secondDeriv_angularFirstVariation_ge β hβ φ σ σ' hφ hσ ω a μ κ θ (hge θ hθ))

/-- The complete inputs of the finite-atomicity consequence are
jointly realized by the actual quadratic one-neuron Dirac witness,
including the all-representatives support curvature hypothesis. -/
example : (0 : ℝ) < 1 ∧ (∀ s : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * s) s) ∧
    (∀ s : ℝ, HasDerivAt (fun t : ℝ => t) ((fun _ : ℝ => (1 : ℝ)) s) s) ∧
    IsStationary 1 (fun s : ℝ => s) (Pi.single 0 (-4 : ℝ))
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) ∧ (0 : ℝ) < 1 ∧
    ∀ θ : ℝ, circlePoint θ ∈
      (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)).support →
      2 * Real.exp 1 + 2 * 1 ≤ deriv^[2]
        (fun t : ℝ => potential (fun s : ℝ => s ^ 2) (Pi.single 0 (-4 : ℝ))
          (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
          (circlePoint t : EucSpace 2)) θ := strict_curvature_quadratic_dirac_witness

end Transformer.Perceptron
