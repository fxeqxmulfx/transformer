/-
# Actual training with stateful base and gain optimizers

arXiv:2606.25971v2, §3.1 and Appendix A, Algorithm 2. The auxiliary
states evolve at every step, including when the training correction
rejects a proposed fused weight. Gradients are recomputed from the
fixed objective at the current fused weight, rather than supplied as
an arbitrary gradient history.
-/

import Transformer.MagnitudeDirection.SectionA_TrainingStorage
import Transformer.Optimization.Matrix
import Transformer.Optimization.Basic

noncomputable section

namespace Transformer.MagnitudeDirection

open Optimization

variable {m n : ℕ} {B R C S : Type*}

/-- Stateful matrix optimizer: old state, direction, its gradient and LR;
new state and new direction. Source: arXiv:2606.25971v2, Appendix A,
Algorithm 2, line 7; stateful training extension. -/
abbrev StatefulMatrixStep (B : Type*) (m n : ℕ) := B →
  Matrix (Fin m) (Fin n) ℝ → Matrix (Fin m) (Fin n) ℝ → ℝ →
  B × Matrix (Fin m) (Fin n) ℝ

/-- Stateful gain optimizer, receiving the actual raw-gain chain-rule
gradient. Source: arXiv:2606.25971v2, Appendix A, Algorithm 2, lines 5, 9. -/
abbrev StatefulGainStep (R : Type*) (m : ℕ) := R → (Fin m → ℝ) → (Fin m → ℝ) →
  ℝ → R × (Fin m → ℝ)

/-- A fused MD proposal with explicit optimizer-memory evolution.
Source: arXiv:2606.25971v2, Appendix A, Algorithm 2, training extension. -/
abbrev FullProposal (S : Type*) (m n : ℕ) := S → FullState m n →
  Matrix (Fin m) (Fin n) ℝ → S × FullState m n

/-- Compute all three genuine factor gradients at the old fused weight,
update the three memories, project the direction and reassemble exactly
as Algorithm 2. No optimizer reliability is an assumption of this definition.
Source: arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
def statefulFullProposal (opt : StatefulMatrixStep B m n)
    (rowOpt : StatefulGainStep R m) (colOpt : StatefulGainStep C n)
    (etaW etaG c : ℝ) : FullProposal (B × R × C) m n := fun memory s G =>
  let D := fullDirection softplus s
  let row := fun i => softplus (s.rawRow i)
  let col := fun j => softplus (s.rawCol j)
  let nextD := opt memory.1 D (directionGradient row col G) etaW
  let nextRow := rowOpt memory.2.1 s.rawRow
    (fun i => rowGradient col D G i * softplusDerivative (s.rawRow i)) etaG
  let nextCol := colOpt memory.2.2 s.rawCol
    (fun j => colGradient row D G j * softplusDerivative (s.rawCol j)) etaG
  ((nextD.1, nextRow.1, nextCol.1),
    ⟨fuse (fun i => softplus (nextRow.2 i)) (fun j => softplus (nextCol.2 j))
      (matrixProject c nextD.2), nextRow.2, nextCol.2⟩)

/-- The stateful proposal's fused storage is the literal Algorithm 2
wrapper with the current optimizer states fixed. This checks the bridge
to the original source implementation, not only its eventual limits.
Source: arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem statefulFullProposal_eq (opt : StatefulMatrixStep B m n)
    (rowOpt : StatefulGainStep R m) (colOpt : StatefulGainStep C n)
    (etaW etaG c : ℝ) (memory : B × R × C) (s : FullState m n)
    (G : Matrix (Fin m) (Fin n) ℝ) :
    (statefulFullProposal opt rowOpt colOpt etaW etaG c memory s G).2 =
      fullStep softplus softplusDerivative
        (fun D H eta => (opt memory.1 D H eta).2)
        (fun a h eta => (rowOpt memory.2.1 a h eta).2)
        (fun a h eta => (colOpt memory.2.2 a h eta).2) s G etaW etaG c := rfl

/-- A nonzero fused proposal from Algorithm 2 has the prescribed direction
norm. Thus the accepted-step preservation theorem needs no separate
optimizer-alignment or sphere-invariance assumption about a callback.
Source: arXiv:2606.25971v2, Appendix A, Algorithm 2, training extension. -/
theorem statefulFullProposal_norm (opt : StatefulMatrixStep B m n)
    (rowOpt : StatefulGainStep R m) (colOpt : StatefulGainStep C n)
    (etaW etaG c : ℝ) (memory : B × R × C) (s : FullState m n)
    (G : Matrix (Fin m) (Fin n) ℝ) (hc : 0 ≤ c)
    (hW : (statefulFullProposal opt rowOpt colOpt etaW etaG c memory s G).2.weight ≠ 0) :
    frobeniusNorm (fullDirection softplus
      (statefulFullProposal opt rowOpt colOpt etaW etaG c memory s G).2) = c := by
  rw [statefulFullProposal_eq] at hW ⊢
  apply softplus_fullStep_direction_norm _ _ _ s G etaW etaG c hc
  intro hC
  apply hW
  dsimp only [fullStep]
  rw [hC]
  ext i j
  simp [matrixProject, fuse]

/-- A true-gradient callback with zero current gradient gives a nonzero
valid proposal. Source: arXiv:2606.25971v2, Appendix A, training extension. -/
example : (0 : ℝ) ≤ 1 ∧
    (statefulFullProposal
      (fun _ : Unit => fun D G eta => ((), D - eta • G))
      (fun _ : Unit => fun a g eta => ((), a - eta • g))
      (fun _ : Unit => fun a g eta => ((), a - eta • g))
      0 0 1 ((), (), ()) (unitGainState (1 : Matrix (Fin 1) (Fin 1) ℝ)) 0).2.weight ≠ 0 := by
  refine ⟨by norm_num, ?_⟩
  norm_num [statefulFullProposal, fullDirection, unitGainState, unfuse,
    matrixProject, frobeniusNorm_eq_sqrt, softplus_unit_initialization, fuse_one]

/-- Actual original MD training with a fixed differentiable objective.
The factorization and optimizer memories, including momentum and moment
estimates supplied by the callbacks, are carried to the next iteration.
Source: arXiv:2606.25971v2, Appendix A, Algorithm 2, training extension. -/
def fullTrainingRun (f : MatrixSpace m n → ℝ) (propose : FullProposal S m n)
    (initialMemory : S) (initial : FullState m n) : ℕ → S × FullState m n
  | 0 => (initialMemory, initial)
  | t + 1 =>
    let s := fullTrainingRun f propose initialMemory initial t
    propose s.1 s.2 (toMatrix (gradient f (fromMatrix s.2.weight)))

/-- A normalized genuine-gradient base step, with relative pre-projection
size `eta` when the gradient is nonzero. This is an allowed optimizer
callback, not a claim that it is Adam or finite Newton--Schulz Muon.
Source: arXiv:2606.25971v2, §3.1, “Updating the direction”. -/
def normalizedGradientStep (D G : Matrix (Fin m) (Fin n) ℝ) (eta : ℝ) :
    Matrix (Fin m) (Fin n) ℝ :=
  D - (eta * frobeniusNorm D / frobeniusNorm G) • G

/-- A simple state-free raw-gain gradient step witnessing the stateful API.
Source: arXiv:2606.25971v2, Appendix A, Algorithm 2, training extension;
this chooses SGD instead of the experimental Adam gain callback. -/
def rawGainGradientStep (a g : Fin m → ℝ) (eta : ℝ) : Fin m → ℝ := a - eta • g

/-- Normalized direction updates with arbitrary stateful gain optimizers.
The positive-map constraint alone will suffice for the counterexample.
Source: arXiv:2606.25971v2, §3.1 and Appendix A, Algorithm 2. -/
def normalizedMDProposal (rowOpt : StatefulGainStep R m) (colOpt : StatefulGainStep C n)
    (etaW etaG c : ℝ) : FullProposal (Unit × R × C) m n :=
  statefulFullProposal (fun _ D G eta => ((), normalizedGradientStep D G eta))
    rowOpt colOpt etaW etaG c

end Transformer.MagnitudeDirection
