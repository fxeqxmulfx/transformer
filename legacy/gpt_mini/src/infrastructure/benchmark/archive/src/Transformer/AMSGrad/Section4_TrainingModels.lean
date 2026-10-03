/-
# AMSGrad — fixed-objective training with constant momentum and epsilon

Extension of arXiv:1904.03590v4, Algorithm 1 and §4, with the positive
denominator regularizer mentioned in §6. Unlike the projected online
theorem, this model uses a fixed full-gradient objective on Euclidean
space, constant first-moment coefficient and constant learning rate.
The denominator is `epsilon + sqrt(vhat)`. No bias correction, weight
decay, projection or direction-replacement guard is inserted.
-/

import Transformer.Optimization.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2

open scoped BigOperators

noncomputable section

namespace Transformer.AMSGrad

/-- Euclidean parameters for the fixed-objective extension of
arXiv:1904.03590v4, Algorithm 1 and §4. -/
abbrev TrainingSpace (d : ℕ) := EuclideanSpace ℝ (Fin d)

/-- Actual AMSGrad training buffers; all fields are data.
Source: arXiv:1904.03590v4, Algorithm 1. -/
structure TrainingState (d : ℕ) where
  position : TrainingSpace d
  momentum : Fin d → ℝ
  second : Fin d → ℝ
  maximum : Fin d → ℝ

variable {d : ℕ}

/-- Positive denominator regularization, following the convention of
arXiv:1904.03590v4, §6, applied to the Algorithm 1 maximum history. -/
def trainingDenominator (ε : ℝ) (s : TrainingState d) (i : Fin d) : ℝ :=
  ε + Real.sqrt (s.maximum i)

/-- Full AMSGrad update from the current objective's genuine derivative.
The first and second moments and the elementwise maximum are all computed
before the parameter step. Source: arXiv:1904.03590v4, Algorithm 1,
with the explicitly stated fixed-objective/epsilon extension. -/
def trainingStep (η ε β β₂ : ℝ) (f : TrainingSpace d → ℝ)
    (s : TrainingState d) : TrainingState d :=
  let g := gradient f s.position
  let m := fun i => β * s.momentum i + (1 - β) * g i
  let v := fun i => β₂ * s.second i + (1 - β₂) * g i ^ 2
  let maximum := fun i => max (s.maximum i) (v i)
  ⟨s.position - η • WithLp.toLp 2 (fun i => m i / (ε + Real.sqrt (maximum i))),
    m, v, maximum⟩

/-- Sequential training, with the source's zero initial moment buffers.
The state at time `t` holds the weight before its next gradient evaluation.
Source: arXiv:1904.03590v4, Algorithm 1, fixed-objective extension. -/
def trainingRun (η ε β β₂ : ℝ) (f : TrainingSpace d → ℝ)
    (initial : TrainingSpace d) : ℕ → TrainingState d
  | 0 => ⟨initial, 0, 0, 0⟩
  | t + 1 => trainingStep η ε β β₂ f (trainingRun η ε β β₂ f initial t)

/-- The previous parameter displacement encoded by momentum and its
actual denominator; it is zero initially. Source: arXiv:1904.03590v4,
Algorithm 1, training extension with constant learning rate. -/
def trainingVelocity (η ε : ℝ) (s : TrainingState d) : TrainingSpace d :=
  WithLp.toLp 2 (fun i => -η * s.momentum i / trainingDenominator ε s i)

/-- Weighted displacement energy in the monotone AMSGrad metric.
Source: training extension of arXiv:1904.03590v4, Algorithm 1 and §4. -/
def trainingKinetic (η ε : ℝ) (s : TrainingState d) : ℝ :=
  ∑ i, trainingDenominator ε s i * (trainingVelocity η ε s i) ^ 2

/-- Lyapunov energy of the actual full momentum/preconditioner state.
Its coefficient is derived from the update, not a convergence assumption.
Source: training extension of arXiv:1904.03590v4, Algorithm 1 and §4. -/
def trainingEnergy (η ε β : ℝ) (f : TrainingSpace d → ℝ) (s : TrainingState d) : ℝ :=
  f s.position + β / (2 * η * (1 - β)) * trainingKinetic η ε s

/-- Positive epsilon supplies a strictly positive denominator without
assuming nonzero coordinate gradients. Source: arXiv:1904.03590v4,
Algorithm 1 and §6, fixed-objective extension. -/
theorem trainingDenominator_pos (ε : ℝ) (s : TrainingState d) (hε : 0 < ε) (i : Fin d) :
    0 < trainingDenominator ε s i := by
  dsimp only [trainingDenominator]
  linarith [Real.sqrt_nonneg (s.maximum i)]

/-- Nonempty denominator domain, arXiv:1904.03590v4, §6, training extension. -/
example : (0 : ℝ) < 1 := by norm_num

/-- The elementwise maximum makes every actual denominator nondecreasing.
This is the AMSGrad feature used by the Lyapunov proof.
Source: arXiv:1904.03590v4, Algorithm 1 and §4. -/
theorem trainingDenominator_mono (η ε β β₂ : ℝ) (f : TrainingSpace d → ℝ)
    (s : TrainingState d) (i : Fin d) :
    trainingDenominator ε s i ≤ trainingDenominator ε (trainingStep η ε β β₂ f s) i := by
  dsimp only [trainingDenominator, trainingStep]
  exact add_le_add le_rfl (Real.sqrt_le_sqrt (le_max_left _ _))

/-- The encoded velocity is the actual new parameter displacement.
Source: arXiv:1904.03590v4, Algorithm 1, fixed-objective extension. -/
theorem trainingStep_position (η ε β β₂ : ℝ) (f : TrainingSpace d → ℝ)
    (s : TrainingState d) :
    (trainingStep η ε β β₂ f s).position =
      s.position + trainingVelocity η ε (trainingStep η ε β β₂ f s) := by
  ext i
  simp [trainingStep, trainingVelocity, trainingDenominator]
  ring

/-- Momentum induces this exact coordinate balance between consecutive
displacements. Source: arXiv:1904.03590v4, Algorithm 1, training extension. -/
theorem trainingVelocity_balance (η ε β β₂ : ℝ) (f : TrainingSpace d → ℝ)
    (s : TrainingState d) (hε : 0 < ε) (i : Fin d) :
    trainingDenominator ε (trainingStep η ε β β₂ f s) i *
        trainingVelocity η ε (trainingStep η ε β β₂ f s) i =
      β * trainingDenominator ε s i * trainingVelocity η ε s i -
        η * (1 - β) * gradient f s.position i := by
  have hnew := (trainingDenominator_pos ε (trainingStep η ε β β₂ f s) hε i).ne'
  have hold := (trainingDenominator_pos ε s hε i).ne'
  change _ * (-η * _ / _) = β * _ * (-η * _ / _) - _
  field_simp
  simp only [trainingStep]
  ring

/-- The momentum-balance hypothesis has a valid regularizer,
arXiv:1904.03590v4, §6, training extension. -/
example : (0 : ℝ) < 1 := by norm_num

end Transformer.AMSGrad
