/-
# DASH — sequential training with actual finite matrix solvers

arXiv:2602.02016v2, §2, Algorithm 1, §2.3 and §3.3–3.5. The gradient
is recomputed from the current weights. All EMA states are updated before
regularization, finite multi-start PI, certified NDB inverse fourth roots
and Adam grafting. The new training guard is an explicit algorithm change.
-/

import Transformer.DASH.Section3_GuardedNewton
import Transformer.DASH.Section2_History
import Transformer.Optimization.Basic
import Transformer.Optimization.Matrix

noncomputable section

namespace Transformer.DASH

open Optimization

variable {m n : ℕ}

/-- Actual auxiliary buffers for a single rectangular Shampoo block.
These fields contain data, not unproved claims about convergence.
Source: arXiv:2602.02016v2, §2, Algorithm 1 and “Grafting”. -/
structure TrainingState (m n : ℕ) where
  left : Matrix (Fin m) (Fin m) ℝ
  right : Matrix (Fin n) (Fin n) ℝ
  accumulator : Matrix (Fin m) (Fin n) ℝ
  momentum : Matrix (Fin m) (Fin n) ℝ

/-- Zero initial histories and grafting buffers,
arXiv:2602.02016v2, §2, Algorithm 1 and “Grafting”. -/
def initialTrainingState : TrainingState m n := ⟨0, 0, 0, 0⟩

/-- Update all four actual histories, regularize, estimate both scales
with finite multi-start PI, run finite certified NDB inverse fourth roots,
and graft the resulting preconditioned gradient. The entrywise squared
gradient accumulator uses the EMA described in the source's grafting
implementation comment; no bias correction is inserted. `μ=0` disables
gradient EMA, as permitted by the grafting paragraph.
Source: arXiv:2602.02016v2, §2.2–2.3 and §3.3–3.5. -/
def dashTrainingCandidate {κ : Type} [Fintype κ] [Nonempty κ]
    (β ν μ ε : ℝ) (startsL : κ → Fin m → ℝ) (startsR : κ → Fin n → ℝ)
    (piSteps rootSteps : ℕ) (state : TrainingState m n) (g : MatrixSpace m n) :
    TrainingState m n × MatrixSpace m n :=
  let G := toMatrix g
  let left := leftEma β state.left G
  let right := rightEma β state.right G
  let squared : Matrix (Fin m) (Fin n) ℝ := fun i j => G i j ^ 2
  let accumulator := ν • state.accumulator + (1 - ν) • squared
  let momentum := μ • state.momentum + (1 - μ) • G
  let L := left + ε • 1
  let R := right + ε • 1
  let P := guardedInverseFourth L (pooledRayleigh L startsL piSteps) rootSteps
  let Q := guardedInverseFourth R (pooledRayleigh R startsR piSteps) rootSteps
  let U := preconditionedGradient P G Q
  let reference := adamGraftingDirection ε accumulator momentum
  (⟨left, right, accumulator, momentum⟩, fromMatrix (graft U reference))

/-- Unguarded learning recurrence using the already corrected numerical
solver, zero initial states and the current objective's true gradient.
An arbitrary schedule remains as in Algorithm 1. Numerical inverse-root
convergence alone does not imply learning convergence.
Source: arXiv:2602.02016v2, §2–3, numerical correction. -/
def dashTrainingRun {κ : Type} [Fintype κ] [Nonempty κ]
    (β ν μ ε : ℝ) (startsL : κ → Fin m → ℝ) (startsR : κ → Fin n → ℝ)
    (piSteps rootSteps : ℕ) (η : ℕ → ℝ) (f : MatrixSpace m n → ℝ)
    (initial : MatrixSpace m n) : ℕ → TrainingState m n × MatrixSpace m n
  | 0 => (initialTrainingState, initial)
  | t + 1 =>
    let state := dashTrainingRun β ν μ ε startsL startsR piSteps rootSteps η f initial t
    let candidate := dashTrainingCandidate β ν μ ε startsL startsR piSteps rootSteps
      state.1 (gradient f state.2)
    (candidate.1, state.2 - η t • candidate.2)

/-- Corrected full learning algorithm with fixed finite solver budgets.
The guard checks the actually grafted numerical candidate, and keeps all
updated EMA states whether that direction is accepted or rejected.
It supplies the missing learning-rate and direction guarantees itself;
it takes no PI reliability or numerical-accuracy proof arguments.
Source: training correction to arXiv:2602.02016v2, §2–4. -/
def safeguardedDashRun {κ : Type} [Fintype κ] [Nonempty κ]
    (σ smooth β ν μ ε : ℝ) (startsL : κ → Fin m → ℝ) (startsR : κ → Fin n → ℝ)
    (piSteps rootSteps : ℕ) (f : MatrixSpace m n → ℝ) (initial : MatrixSpace m n) :
    ℕ → TrainingState m n × MatrixSpace m n :=
  safeguardedRun σ smooth f
    (fun state _ g => dashTrainingCandidate β ν μ ε startsL startsR piSteps rootSteps state g)
    initialTrainingState initial

end Transformer.DASH
