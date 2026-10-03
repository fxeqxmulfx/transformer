/-
# AMSGrad with decoupled weight decay

User-requested deterministic extension of arXiv:1904.03590v4,
Algorithm 1 and §6, using the decoupled decay discussed in
arXiv:2606.25971v2, §2 and §4.1. The genuine full gradient alone enters
the moment buffers. Decay is applied directly to the old parameter.
This model has constant coefficients, no bias correction and no projection.
-/

import Transformer.AMSGrad.Section4_TrainingModels
import Transformer.Optimization.Quadratic
import Mathlib.Topology.MetricSpace.Contracting
import Mathlib.Topology.Order.MonotoneConvergence

open scoped Topology
open Filter

noncomputable section

namespace Transformer.AMSGradW

abbrev TrainingSpace := AMSGrad.TrainingSpace
abbrev TrainingState := AMSGrad.TrainingState

variable {d : ℕ}

/-- Coordinates carry the maximum norm for the contraction estimates;
the loss still acts on actual Euclidean parameters. Source:
arXiv:1904.03590v4, Algorithm 1, AMSGradW training extension. -/
def coordinateGradient (f : TrainingSpace d → ℝ) (x : Fin d → ℝ) : Fin d → ℝ :=
  WithLp.ofLp (gradient f (WithLp.toLp 2 x))

/-- An objective assumption on its genuine gradient in the maximum norm,
not a trajectory or convergence premise. Source: deterministic AMSGradW
extension of arXiv:1904.03590v4, Algorithm 1 and §4. -/
def LipschitzGradient (f : TrainingSpace d → ℝ) (L : ℝ) : Prop :=
  ∀ x y, ‖coordinateGradient f x - coordinateGradient f y‖ ≤ L * ‖x - y‖

/-- AMSGrad's actual three buffer updates, followed by the independent
parameter decay `-eta*lambda*x`; decay is absent from both moments.
Source: arXiv:1904.03590v4, Algorithm 1 and §6, extended with the
decoupled decay discussed in arXiv:2606.25971v2, §2 and §4.1. -/
def trainingStep (η ε wd β β₂ : ℝ) (f : TrainingSpace d → ℝ)
    (s : TrainingState d) : TrainingState d :=
  let next := AMSGrad.trainingStep η ε β β₂ f s
  { next with position := next.position - (η * wd) • s.position }

/-- Actual sequential AMSGradW, with zero initial moment histories.
Source: AMSGradW extension of arXiv:1904.03590v4, Algorithm 1 and §6. -/
def trainingRun (η ε wd β β₂ : ℝ) (f : TrainingSpace d → ℝ)
    (initial : TrainingSpace d) : ℕ → TrainingState d
  | 0 => ⟨initial, 0, 0, 0⟩
  | t + 1 => trainingStep η ε wd β β₂ f (trainingRun η ε wd β β₂ f initial t)

/-- Decay does not alter any buffer; the effective parameter update
uses the newly computed denominator and the old parameter.
Source: arXiv:1904.03590v4, Algorithm 1, AMSGradW extension. -/
theorem trainingStep_coordinates (η ε wd β β₂ : ℝ) (f : TrainingSpace d → ℝ)
    (s : TrainingState d) (i : Fin d) :
    (trainingStep η ε wd β β₂ f s).momentum i =
        β * s.momentum i + (1 - β) * gradient f s.position i ∧
      (trainingStep η ε wd β β₂ f s).position i =
        (1 - η * wd) * s.position i -
          η * (trainingStep η ε wd β β₂ f s).momentum i /
            AMSGrad.trainingDenominator ε (trainingStep η ε wd β β₂ f s) i := by
  constructor
  · rfl
  · simp only [trainingStep, AMSGrad.trainingStep, AMSGrad.trainingDenominator,
      PiLp.sub_apply, PiLp.smul_apply, smul_eq_mul]
    ring

/-- Setting decay to zero gives the existing AMSGrad update exactly.
Source: arXiv:1904.03590v4, Algorithm 1, AMSGradW extension. -/
theorem trainingStep_zero_decay (η ε β β₂ : ℝ) (f : TrainingSpace d → ℝ)
    (s : TrainingState d) : trainingStep η ε 0 β β₂ f s =
      AMSGrad.trainingStep η ε β β₂ f s := by
  simp [trainingStep]

/-- Every actual denominator is at least epsilon, independently of the
decay and momentum parameters. Source: arXiv:1904.03590v4, §6,
AMSGradW extension. -/
theorem denominator_lower (ε : ℝ) (s : TrainingState d) (i : Fin d) :
    ε ≤ AMSGrad.trainingDenominator ε s i := by
  exact le_add_of_nonneg_right (Real.sqrt_nonneg _)

/-- The quadratic provides a nonconstant Lipschitz-gradient objective.
Source: arXiv:1904.03590v4, §4, AMSGradW convergence witness. -/
theorem energy_lipschitz : LipschitzGradient
    (Optimization.energy : TrainingSpace d → ℝ) 1 := by
  intro x y
  simp only [coordinateGradient, Optimization.energy_gradient, id_eq, WithLp.ofLp_toLp,
    one_mul]
  exact le_rfl

end Transformer.AMSGradW
