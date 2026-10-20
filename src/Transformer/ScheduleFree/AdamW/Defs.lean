module

public import Transformer.ScheduleFree.Section4_AdamW
public import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# The authors' revised Schedule-Free AdamW

Source: facebookresearch/schedule_free, Aaron Defazio's commit
`59ed9b662d5ec5799aded7390b2482758423653c` (2024-11-04), "Stability improvements",
`schedulefree/adamw_schedulefree.py`, `step`, `train`, and `eval`.
The same formulas remain in commit `70785b53e778d0e872c0bbb75ff4ee54ee10c291`
with the default `inner_momentum=0`. The optional inner EMA added later is
outside this model. The original manuscript, arXiv:2405.15682v4, §4.4,
Algorithm 1, remains separately available as `ScheduleFree.adamStep`.

Unlike Algorithm 1, the rate contains warmup alone. Bias correction divides
the second moment before the square root. Decay is evaluated at the old `y`,
and averaging uses the running maximum of that warmup-only rate, with the
code's general real exponents `r` and `weight_lr_power` (defaults 0 and 2).
Time `t=0` is the source's `k=0`, i.e. the manuscript's first update.

These are real-arithmetic semantics for initialized runs with `0≤beta2<1`,
`epsilon>0`, nonnegative rates, defined powers, and a gradient at every step.
Zero rates with negative weight powers, sparse/missing gradients, floating-point
roundoff, and tensor aliasing are outside this model. Total real division
implements the code's explicit zero-total averaging fallback. The optimized
two-buffer update and the explicit three-sequence update are defined independently.
The code-to-sequence correspondence is `p=y`, `state['z']=z`,
`exp_avg_sq=v`, `weight_sum=W`, and `lr_max=M`. In the upstream reference
representation the parameter buffer also contains `y`; `evaluation` below
denotes its separately stored `x`, rather than that parameter buffer.
-/

@[expose] public section

noncomputable section

namespace Transformer.ScheduleFree.AdamW

variable {ι : Type*}

/-- Upstream `__init__`: ordinary AdamW parameters and the two averaging powers. -/
structure Parameters where
  adam : AdamParameters
  averagingPower : ℝ := 0
  ratePower : ℝ := 2

/-- Upstream `step`: raw second moment, group weight sum, and `lr_max`. -/
structure History (ι : Type*) where
  second : ι → ℝ
  total : ℝ
  maximumRate : ℝ

/-- The reference representation stores evaluation `x` and base `z` explicitly. -/
structure State (ι : Type*) where
  evaluation : ι → ℝ
  base : ι → ℝ
  history : History ι

/-- The optimized representation stores training query `y` and base `z`. -/
structure TrainingState (ι : Type*) where
  point : ι → ℝ
  base : ι → ℝ
  history : History ι

/-- Upstream initialization: zero variance/weight and the actual `lr_max=-1`. -/
def initial (point : ι → ℝ) : State ι :=
  ⟨point, point, ⟨fun _ => 0, 0, -1⟩⟩

/-- Upstream `bias_correction2`, kept outside the learning rate. -/
def correction (β : ℝ) (t : ℕ) : ℝ := 1 - β ^ (t + 1)

/-- Version 1.3's `lr = group['lr'] * sched`, with no bias-correction factor. -/
def rate (p : Parameters) (t : ℕ) : ℝ :=
  p.adam.learningRate * warmupFactor p.adam.warmup t

/-- The old query, used both by the gradient oracle and by weight decay. -/
def query (p : Parameters) (s : State ι) : ι → ℝ :=
  fun i => (1 - p.adam.betaOne) * s.base i + p.adam.betaOne * s.evaluation i

/-- Both upstream branches update the raw second moment before normalizing. -/
def nextSecond (p : Parameters) (h : History ι) (g : ι → ℝ) : ι → ℝ :=
  fun i => p.adam.betaTwo * h.second i + (1 - p.adam.betaTwo) * g i ^ 2

/-- Upstream `lr_max = max(lr, lr_max)`, including the initial sentinel. -/
def nextMaximum (p : Parameters) (t : ℕ) (h : History ι) : ℝ :=
  max (rate p t) h.maximumRate

/-- Upstream `weight = (k+1)**r * lr_max**weight_lr_power`. -/
def weight (p : Parameters) (t : ℕ) (h : History ι) : ℝ :=
  ((t : ℝ) + 1) ^ p.averagingPower * nextMaximum p t h ^ p.ratePower

/-- Upstream `ckp1`; `0/0=0` matches its explicit `ZeroDivisionError` fallback. -/
def coefficient (p : Parameters) (t : ℕ) (h : History ι) : ℝ :=
  weight p t h / (h.total + weight p t h)

/-- Version 1.3 divides by bias correction inside the root, then adds epsilon. -/
def denominator (p : Parameters) (t : ℕ) (h : History ι) (g : ι → ℝ) (i : ι) : ℝ :=
  Real.sqrt (nextSecond p h g i / correction p.adam.betaTwo t) + p.adam.epsilon

/-- Upstream normalized gradient plus decay at the old query, before any lerp. -/
def direction (p : Parameters) (t : ℕ) (h : History ι) (y g : ι → ℝ) : ι → ℝ :=
  fun i => g i / denominator p t h g i + p.adam.decay * y i

/-- The raw variance, group averaging total and maximum rate advance together. -/
def nextHistory (p : Parameters) (t : ℕ) (h : History ι) (g : ι → ℝ) : History ι :=
  ⟨nextSecond p h g, h.total + weight p t h, nextMaximum p t h⟩

/-- Explicit reference update: step `z`, average into `x`, then query the new state. -/
def step (p : Parameters) (t : ℕ) (s : State ι) (g : ι → ℝ) : State ι :=
  let z := fun i => s.base i - rate p t * direction p t s.history (query p s) g i
  let c := coefficient p t s.history
  ⟨fun i => (1 - c) * s.evaluation i + c * z i, z, nextHistory p t s.history g⟩

/-- Upstream `train`: store `y` in the parameter buffer, retaining `z` and history. -/
def pack (p : Parameters) (s : State ι) : TrainingState ι :=
  ⟨query p s, s.base, s.history⟩

/-- Upstream `eval`: `p.lerp_(z, 1-1/beta1)`. Recovery requires nonzero beta1. -/
def evaluation (p : Parameters) (s : TrainingState ι) : ι → ℝ :=
  fun i => s.point i + (1 - 1 / p.adam.betaOne) * (s.base i - s.point i)

/-- Upstream scalar and foreach branches: lerp the old `y` toward the old `z`,
add `lr*(beta1*(1-c)-1)*direction`, then step `z` using the same direction. -/
def trainingStep (p : Parameters) (t : ℕ) (s : TrainingState ι)
    (g : ι → ℝ) : TrainingState ι :=
  let u := direction p t s.history s.point g
  let c := coefficient p t s.history
  ⟨fun i => s.point i + c * (s.base i - s.point i) +
      rate p t * (p.adam.betaOne * (1 - c) - 1) * u i,
    fun i => s.base i - rate p t * u i, nextHistory p t s.history g⟩

/-- A reference trajectory whose oracle is always called at the current `y`. -/
def run (p : Parameters) (oracle : ℕ → (ι → ℝ) → (ι → ℝ))
    (point : ι → ℝ) : ℕ → State ι
  | 0 => initial point
  | t + 1 =>
    let s := run p oracle point t
    step p t s (oracle t (query p s))

/-- An independently executed trajectory storing only the two iterate buffers. -/
def trainingRun (p : Parameters) (oracle : ℕ → (ι → ℝ) → (ι → ℝ))
    (point : ι → ℝ) : ℕ → TrainingState ι
  | 0 => ⟨point, point, ⟨fun _ => 0, 0, -1⟩⟩
  | t + 1 =>
    let s := trainingRun p oracle point t
    trainingStep p t s (oracle t s.point)

end Transformer.ScheduleFree.AdamW
