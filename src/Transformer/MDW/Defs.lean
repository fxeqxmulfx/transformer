module

public import Transformer.MagnitudeDirection.SectionA_AMSGradCallbacks
public import Transformer.Grokking.AdamW.ScalarRecurrence

/-!
# Physical weight decay for magnitude/direction optimizers

User-requested compositions of arXiv:2606.25971v2, Appendix A, Algorithm 2,
with AdamW's decoupled decay. The MD proposal receives the loss gradient
without regularization. Its fused weight is then reduced by `eta * decay`
times the old physical weight. Positive-radius storage is repaired without
changing that weight. A zero result keeps finite raw gains and zero direction.
These definitions add no convergence claim to the source's empirical method.
-/

@[expose] public section

noncomputable section

namespace Transformer.MDW

open MagnitudeDirection

variable {ι S : Type*} {m n : ℕ}

/-- AdamW-style physical correction of an MD proposal, followed by storage repair.
Source: local decay extension of arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
def correctWeight (radius eta decay : ℝ) (old : Matrix (Fin m) (Fin n) ℝ)
    (proposal : FullState m n) : FullState m n :=
  let corrected := { proposal with weight := proposal.weight - (eta * decay) • old }
  if corrected.weight = 0 then corrected else rebalanceStorage radius corrected

/-- Apply physical decay after the actual MD callback, retaining its optimizer memories.
Source: local AdamW extension of arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
def stepWith (propose : FullProposal S m n) (radius eta decay : ℝ) : FullProposal S m n :=
  fun memory state gradient =>
    let next := propose memory state gradient
    (next.1, correctWeight radius eta decay state.weight next.2)

/-- Bias-corrected Adam histories for every always-updated factor coordinate.
Source: native AdamW scalar recurrence, with zero decay inside factor callbacks. -/
structure AdamBuffers (ι : Type*) where
  momentum : ι → ℝ
  second : ι → ℝ
  clock : ℕ

/-- Zero first/second moments and the common completed-update clock. -/
def initialAdamBuffers (ι : Type*) : AdamBuffers ι := ⟨0, 0, 0⟩

/-- Native bias-corrected Adam on a factor, without any decay in its histories.
Source: `Grokking.AdamW.scalarNativeStep`; local MD callback substitution. -/
def adamVectorCallback (epsilon betaOne betaTwo : ℝ) (state : AdamBuffers ι)
    (point gradient : ι → ℝ) (eta : ℝ) : AdamBuffers ι × (ι → ℝ) :=
  let first := fun i => betaOne * state.momentum i + (1 - betaOne) * gradient i
  let second := fun i => betaTwo * state.second i + (1 - betaTwo) * gradient i ^ 2
  let clock := state.clock + 1
  (⟨first, second, clock⟩, fun i => point i -
    eta * (first i / (1 - betaOne ^ clock)) /
      (Real.sqrt (second i / (1 - betaTwo ^ clock)) + epsilon))

/-- Flattened matrix Adam callback for Algorithm 2's direction update. -/
def adamMatrixCallback (epsilon betaOne betaTwo : ℝ) :
    StatefulMatrixStep (AdamBuffers (Fin m × Fin n)) m n := fun state point gradient eta =>
  let next := adamVectorCallback epsilon betaOne betaTwo state
    (fun ij => point ij.1 ij.2) (fun ij => gradient ij.1 ij.2) eta
  (next.1, fun i j => next.2 (i, j))

/-- Distinct Adam histories for the direction and both raw gain vectors. -/
abbrev AdamMDMemory (m n : ℕ) := AdamBuffers (Fin m × Fin n) ×
  AdamBuffers (Fin m) × AdamBuffers (Fin n)

/-- Zero histories for the three native Adam factor callbacks. -/
def initialAdamMDMemory (m n : ℕ) : AdamMDMemory m n :=
  (initialAdamBuffers _, initialAdamBuffers _, initialAdamBuffers _)

/-- Algorithm 2 with native bias-corrected Adam on all three old-query gradients.
Source: arXiv:2606.25971v2, Appendix A; callback and epsilon convention explicit. -/
def adamMDProposal (epsilon betaOne betaTwo etaDirection etaGain radius : ℝ) :
    FullProposal (AdamMDMemory m n) m n :=
  statefulFullProposal (adamMatrixCallback epsilon betaOne betaTwo)
    (adamVectorCallback epsilon betaOne betaTwo)
    (adamVectorCallback epsilon betaOne betaTwo) etaDirection etaGain radius

/-- Raw-moment AMSGradMD followed by physical decay at the direction's outer rate.
Source: local combination of MD Appendix A and AMSGrad Algorithm 1. -/
def amsgradMDWProposal (epsilon betaOne betaTwo etaDirection etaGain radius decay : ℝ) :
    FullProposal (AMSGradMDMemory m n) m n :=
  stepWith (amsgradMDProposal epsilon betaOne betaTwo etaDirection etaGain radius)
    radius etaDirection decay

/-- Native AdamMD followed by the same decoupled physical-weight correction. -/
def adamMDWProposal (epsilon betaOne betaTwo etaDirection etaGain radius decay : ℝ) :
    FullProposal (AdamMDMemory m n) m n :=
  stepWith (adamMDProposal epsilon betaOne betaTwo etaDirection etaGain radius)
    radius etaDirection decay

/-- The oracle is evaluated on the current physical weight on every MDW update. -/
def run (propose : ℕ → FullProposal S m n)
    (oracle : ℕ → Matrix (Fin m) (Fin n) ℝ → Matrix (Fin m) (Fin n) ℝ)
    (memory : S) (initial : FullState m n) : ℕ → S × FullState m n
  | 0 => (memory, initial)
  | t + 1 =>
    let previous := run propose oracle memory initial t
    propose t previous.1 previous.2 (oracle t previous.2.weight)

end Transformer.MDW
