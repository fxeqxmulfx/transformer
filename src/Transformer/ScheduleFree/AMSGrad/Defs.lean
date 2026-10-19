module

public import Transformer.ScheduleFree.Section4_AdamW

/-!
# Schedule-Free AMSGrad: explicit recurrence

This is a new composition of arXiv:2405.15682v4, §4.4, Algorithm 1 and
arXiv:1904.03590v4, Algorithm 1 and §6. It replaces the raw variance in
Schedule-Free AdamW by its running maximum. The first coefficient controls
query interpolation; no first-moment EMA is added. The effective rate,
query-point decay and squared-rate averaging retain Algorithm 1's convention.

`stepWith` also supports the base-direction projection used by the new MD
composition. A zero effective rate leaves the base untouched, while second
moments, maxima and the clock still advance. No convergence theorem for this
composition is asserted by either source.
-/

@[expose] public section

noncomputable section

namespace Transformer.ScheduleFree.AMSGrad

variable {ι : Type*}

/-- Local AMSGrad extension of §4.4: two iterates, raw histories and total weight. -/
structure State (ι : Type*) where
  evaluation : ι → ℝ
  base : ι → ℝ
  second : ι → ℝ
  maximum : ι → ℝ
  total : ℝ

/-- Local extension of Algorithm 1: equal initial iterates and zero histories. -/
def initial (point : ι → ℝ) : State ι := ⟨point, point, fun _ => 0, fun _ => 0, 0⟩

/-- §4.4 interpolation, applied to this extension's stored iterates. -/
def query (p : AdamParameters) (s : State ι) : ι → ℝ :=
  fun i => (1 - p.betaOne) * s.base i + p.betaOne * s.evaluation i

/-- Algorithm 1's raw variance update, before AMSGrad's maximum is taken. -/
def nextSecond (p : AdamParameters) (s : State ι) (g : ι → ℝ) : ι → ℝ :=
  fun i => p.betaTwo * s.second i + (1 - p.betaTwo) * g i ^ 2

/-- AMSGrad's maximum is taken on raw variances, before rate bias correction. -/
def nextMaximum (p : AdamParameters) (s : State ι) (g : ι → ℝ) : ι → ℝ :=
  fun i => max (s.maximum i) (nextSecond p s g i)

/-- §4.4 squared-effective-rate averaging, with Lean's zero-division convention. -/
def coefficient (p : AdamParameters) (t : ℕ) (s : State ι) : ℝ :=
  adamLearningRate p t ^ 2 / (s.total + adamLearningRate p t ^ 2)

/-- The unprojected base step, with regularization at the old query. -/
def proposal (p : AdamParameters) (t : ℕ) (s : State ι) (g : ι → ℝ) : ι → ℝ :=
  fun i => s.base i - adamLearningRate p t * g i /
    (Real.sqrt (nextMaximum p s g i) + p.epsilon) -
    adamLearningRate p t * p.decay * query p s i

/-- Local projected extension: update histories and base, then average that base. -/
def stepWith (project : (ι → ℝ) → (ι → ℝ)) (p : AdamParameters) (t : ℕ)
    (s : State ι) (g : ι → ℝ) : State ι :=
  let base := if adamLearningRate p t = 0 then s.base else project (proposal p t s g)
  let c := coefficient p t s
  ⟨fun i => (1 - c) * s.evaluation i + c * base i, base,
    nextSecond p s g, nextMaximum p s g, s.total + adamLearningRate p t ^ 2⟩

/-- Schedule-Free AMSGradW: the base step has no additional projection. -/
def step (p : AdamParameters) (t : ℕ) (s : State ι) (g : ι → ℝ) : State ι :=
  stepWith id p t s g

/-- Initialize and query the actual current interpolated point before every step. -/
def run (p : AdamParameters) (oracle : ℕ → (ι → ℝ) → (ι → ℝ))
    (point : ι → ℝ) : ℕ → State ι
  | 0 => initial point
  | t + 1 =>
    let s := run p oracle point t
    step p t s (oracle t (query p s))

end Transformer.ScheduleFree.AMSGrad
