module

public import Transformer.ScheduleFree.AMSGrad.Basic
public import Transformer.MagnitudeDirection.Section3_LossChainRule
public import Transformer.MagnitudeDirection.Section3_Sphere

/-!
# Schedule-Free AMSGrad in magnitude/direction coordinates

New composition of arXiv:2405.15682v4, §4.4, arXiv:1904.03590v4,
Algorithm 1 and arXiv:2606.25971v2, §3.1 and Appendix A, Algorithm 2.
Direction, raw row gains and raw column gains each have an independent
Schedule-Free AMSGrad history. Chain-rule gradients are evaluated at their
joint interpolated query. Only the updated base direction is projected;
the average and query directions retain their actual averaged values.
The model sees the fused query in training and fused averages in evaluation.

All component rates use §4.4's warmup and variance correction and accumulate
their own squared-rate weights. Gains use their own rate and have no decay.
Auxiliary model parameters follow `step` at their auxiliary rate; this module
models the selected matrix and its two gain vectors explicitly.
-/

@[expose] public section

noncomputable section

namespace Transformer.ScheduleFree.AMSGrad

open MagnitudeDirection

variable {m n : ℕ}

/-- Local MD composition: independently configurable direction and gain rates. -/
structure MDParameters where
  learningRate : ℝ
  gainRate : ℝ
  warmup : ℕ
  betaOne : ℝ
  betaTwo : ℝ
  epsilon : ℝ

/-- §4.4's adaptive settings for the local MD direction, with zero decay. -/
def mdDirectionParameters (p : MDParameters) : AdamParameters :=
  ⟨p.learningRate, 0, p.warmup, p.betaOne, p.betaTwo, p.epsilon⟩

/-- The two raw gains have their own effective rates and averaging totals. -/
def mdGainParameters (p : MDParameters) : AdamParameters :=
  ⟨p.gainRate, 0, p.warmup, p.betaOne, p.betaTwo, p.epsilon⟩

/-- All three actual factor histories and the initial direction radius. -/
structure MDState (m n : ℕ) where
  direction : State (Fin m × Fin n)
  row : State (Fin m)
  column : State (Fin n)
  radius : ℝ

/-- A point in the direction/raw-gain coordinates optimized by the composition. -/
structure FactorPoint (m n : ℕ) where
  direction : Matrix (Fin m) (Fin n) ℝ
  rawRow : Fin m → ℝ
  rawColumn : Fin n → ℝ

/-- The three chain-rule gradients with respect to those optimization variables. -/
structure FactorGradient (m n : ℕ) where
  direction : Matrix (Fin m) (Fin n) ℝ
  row : Fin m → ℝ
  column : Fin n → ℝ

/-- Unit effective gains and zero adaptive histories, following MD §4.1.3. -/
def mdInitial (weight : Matrix (Fin m) (Fin n) ℝ) : MDState m n :=
  ⟨initial (fun ij => weight ij.1 ij.2),
    initial (fun _ => Real.log (Real.exp 1 - 1)),
    initial (fun _ => Real.log (Real.exp 1 - 1)), frobeniusNorm weight⟩

/-- The selected matrix's actual base direction, before fusion with gains. -/
def mdBaseDirection (s : MDState m n) : Matrix (Fin m) (Fin n) ℝ :=
  fun i j => s.direction.base (i, j)

/-- Interpolate each latent coordinate using the same Schedule-Free coefficient. -/
def mdQuery (p : MDParameters) (s : MDState m n) : FactorPoint m n :=
  ⟨fun i j => query (mdDirectionParameters p) s.direction (i, j),
    query (mdGainParameters p) s.row, query (mdGainParameters p) s.column⟩

/-- Evaluation retains the averages in latent coordinates, without projection. -/
def mdEvaluation (s : MDState m n) : FactorPoint m n :=
  ⟨fun i j => s.direction.evaluation (i, j), s.row.evaluation, s.column.evaluation⟩

/-- MD's smooth positive-gain fusion, applied to a latent query or average. -/
def factorWeight (q : FactorPoint m n) : Matrix (Fin m) (Fin n) ℝ :=
  fuse (fun i => softplus (q.rawRow i)) (fun j => softplus (q.rawColumn j)) q.direction

/-- The matrix used by the model's next training forward and gradient oracle. -/
def mdQueryWeight (p : MDParameters) (s : MDState m n) : Matrix (Fin m) (Fin n) ℝ :=
  factorWeight (mdQuery p s)

/-- The matrix installed for validation and inference. -/
def mdEvaluationWeight (s : MDState m n) : Matrix (Fin m) (Fin n) ℝ :=
  factorWeight (mdEvaluation s)

/-- Algorithm 2's three genuine chain-rule gradients, evaluated at the query factors. -/
def mdGradients (q : FactorPoint m n) (G : Matrix (Fin m) (Fin n) ℝ) : FactorGradient m n :=
  ⟨directionGradient (fun i => softplus (q.rawRow i)) (fun j => softplus (q.rawColumn j)) G,
    fun i => rowGradient (fun j => softplus (q.rawColumn j)) q.direction G i *
      softplusDerivative (q.rawRow i),
    fun j => colGradient (fun i => softplus (q.rawRow i)) q.direction G j *
      softplusDerivative (q.rawColumn j)⟩

/-- Flattened interface to Algorithm 2's genuine Frobenius sphere projection. -/
def projectDirection (radius : ℝ) (direction : Fin m × Fin n → ℝ) : Fin m × Fin n → ℝ :=
  fun ij => matrixProject radius (fun i j => direction (i, j)) ij.1 ij.2

/-- The actual unprojected candidate, useful for stating the sphere theorem. -/
def mdCandidate (p : MDParameters) (t : ℕ) (s : MDState m n)
    (G : Matrix (Fin m) (Fin n) ℝ) : Matrix (Fin m) (Fin n) ℝ :=
  fun i j => proposal (mdDirectionParameters p) t s.direction
    (fun ij => (mdGradients (mdQuery p s) G).direction ij.1 ij.2) (i, j)

/-- Full local composition: compute all old-query gradients, update each base,
project the direction before averaging, and retain all three histories. -/
def mdStep (p : MDParameters) (t : ℕ) (s : MDState m n)
    (G : Matrix (Fin m) (Fin n) ℝ) : MDState m n :=
  let g := mdGradients (mdQuery p s) G
  ⟨stepWith (projectDirection s.radius) (mdDirectionParameters p) t s.direction
      (fun ij => g.direction ij.1 ij.2),
    step (mdGainParameters p) t s.row g.row,
    step (mdGainParameters p) t s.column g.column, s.radius⟩

/-- Every oracle call sees the fused current factor query, including after averaging. -/
def mdRun (p : MDParameters)
    (oracle : ℕ → Matrix (Fin m) (Fin n) ℝ → Matrix (Fin m) (Fin n) ℝ)
    (weight : Matrix (Fin m) (Fin n) ℝ) : ℕ → MDState m n
  | 0 => mdInitial weight
  | t + 1 =>
    let s := mdRun p oracle weight t
    mdStep p t s (oracle t (mdQueryWeight p s))

end Transformer.ScheduleFree.AMSGrad
