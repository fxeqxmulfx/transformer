/-
# AdaFisher — the full deterministic learning trajectory

Training extension of arXiv:2405.16397v3, §3.3–3.4, Algorithm 1.
The existing corrected state transition computes both KF EMAs, min-max
normalization, damping, raw momentum and positive-time bias correction.
This module supplies the actual current loss gradient at every step.
AdaFisher is selected with zero decoupled weight decay.
-/

import Transformer.AdaFisher.Section3_State
import Transformer.Optimization.Quadratic
import Mathlib.Analysis.InnerProductSpace.PiL2

open scoped BigOperators

noncomputable section

namespace Transformer.AdaFisher

/-- Euclidean layer-block parameters, arXiv:2405.16397v3, §2–3. -/
abbrev TrainingSpace (a b : ℕ) := EuclideanSpace ℝ (Fin (a * b))

/-- Fresh measured KF diagonals as functions of current weights and their
full gradient. Source: arXiv:2405.16397v3, Algorithm 1 and Appendix A.3.
The convergence theorem is uniform over such measurements; they are
normalized and damped by the actual state transition, not supplied as a
free bounded preconditioner sequence. -/
abbrev TrainingFactors (a b : ℕ) := TrainingSpace a b → TrainingSpace a b →
  (Fin a → ℝ) × (Fin b → ℝ)

variable {a b : ℕ}

/-- The state's actual weights in Euclidean geometry,
arXiv:2405.16397v3, Algorithm 1. -/
def trainingPosition (s : OptimizerState a b) : TrainingSpace a b :=
  WithLp.toLp 2 s.parameters

/-- The positive-time corrected first moment; the zero-initialized
moment is zero at time zero. Source: arXiv:2405.16397v3, §3.3, Algorithm 1,
with the pseudocode's time and raw/corrected-buffer corrections. -/
def trainingMoment (β : ℝ) (s : OptimizerState a b) : TrainingSpace a b :=
  WithLp.toLp 2 (fun i => s.rawMoment i / (1 - β ^ s.time))

/-- Actual current damped Fisher diagonal, including both KF EMAs and
min-max scaling. Source: arXiv:2405.16397v3, equation (4), Algorithm 1. -/
def trainingMetric [NeZero a] [NeZero b] (δ : ℝ) (s : OptimizerState a b) :
    Fin (a * b) → ℝ := normalizedFisher δ s.factorH s.factorS

/-- A complete source update driven by the genuine current loss gradient,
with bias correction retained and no direction-replacement guard.
Source: arXiv:2405.16397v3, Algorithm 1, deterministic specialization. -/
def trainingStep [NeZero a] [NeZero b] (η β γ δ : ℝ)
    (f : TrainingSpace a b → ℝ) (factors : TrainingFactors a b)
    (s : OptimizerState a b) : OptimizerState a b :=
  let x := trainingPosition s
  let g := gradient f x
  let fresh := factors x g
  optimizerStep η 0 β γ δ s fresh.1 fresh.2 g.ofLp

/-- The complete deterministic training recurrence. Source:
arXiv:2405.16397v3, Algorithm 1, fixed full-gradient specialization. -/
def trainingRun [NeZero a] [NeZero b] (η β γ δ : ℝ)
    (f : TrainingSpace a b → ℝ) (factors : TrainingFactors a b)
    (initial : TrainingSpace a b) : ℕ → OptimizerState a b
  | 0 => initializeOptimizer initial.ofLp
  | t + 1 => trainingStep η β γ δ f factors (trainingRun η β γ δ f factors initial t)

/-- Previous actual displacement encoded by the corrected moment and the
current computed metric. Source: arXiv:2405.16397v3, Algorithm 1. -/
def trainingVelocity [NeZero a] [NeZero b] (η β δ : ℝ)
    (s : OptimizerState a b) : TrainingSpace a b :=
  WithLp.toLp 2 (fun i => -η * trainingMoment β s i / trainingMetric δ s i)

/-- Error between the bias-corrected moment and the true gradient at
the current weights. Source: arXiv:2405.16397v3, §3.3–3.4, deterministic
training extension. This error is evolved and bounded, not assumed small. -/
def trainingError (β : ℝ) (f : TrainingSpace a b → ℝ)
    (s : OptimizerState a b) : TrainingSpace a b :=
  trainingMoment β s - gradient f (trainingPosition s)

/-- Lyapunov loss plus squared momentum error for the actual training
state. Source: arXiv:2405.16397v3, §3.4, deterministic constant-step
extension; its error term handles the nonmonotone Fisher diagonal. -/
def trainingEnergy (η β δ : ℝ) (f : TrainingSpace a b → ℝ)
    (s : OptimizerState a b) : ℝ :=
  f (trainingPosition s) + η / (δ * (1 - β)) * ‖trainingError β f s‖ ^ 2

/-- Actual Fisher bounds are derived from min-max normalization and
damping, without restrictions on the measured factors or their history.
Source: arXiv:2405.16397v3, Proposition 3.2 and Algorithm 1. -/
theorem trainingMetric_bounds [NeZero a] [NeZero b] (δ : ℝ)
    (s : OptimizerState a b) (i : Fin (a * b)) :
    δ ≤ trainingMetric δ s i ∧ trainingMetric δ s i ≤ 1 + δ :=
  normalizedFisher_bounds δ s.factorH s.factorS i

/-- The velocity at a new state equals the actual displacement, including
bias correction. Source: arXiv:2405.16397v3, Algorithm 1. -/
theorem trainingStep_position [NeZero a] [NeZero b] (η β γ δ : ℝ)
    (f : TrainingSpace a b → ℝ) (factors : TrainingFactors a b) (s : OptimizerState a b) :
    trainingPosition (trainingStep η β γ δ f factors s) =
      trainingPosition s + trainingVelocity η β δ (trainingStep η β γ δ f factors s) := by
  ext i
  simp only [trainingStep, optimizerStep, trainingPosition, trainingVelocity,
    trainingMoment, trainingMetric, parameterStep, PiLp.toLp_apply, PiLp.add_apply]
  ring

/-- The actual buffer counter equals the number of completed updates.
Source: arXiv:2405.16397v3, Algorithm 1, corrected time indexing. -/
theorem trainingRun_time [NeZero a] [NeZero b] (η β γ δ : ℝ)
    (f : TrainingSpace a b → ℝ) (factors : TrainingFactors a b)
    (initial : TrainingSpace a b) (t : ℕ) :
    (trainingRun η β γ δ f factors initial t).time = t := by
  induction t with
  | zero => rfl
  | succ t ih => exact congrArg Nat.succ ih

end Transformer.AdaFisher
