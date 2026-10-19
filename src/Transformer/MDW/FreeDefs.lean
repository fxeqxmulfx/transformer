module

public import Transformer.MDW.Basic
public import Transformer.ScheduleFree.AMSGrad

/-!
# Schedule-Free AdamMDW and AMSGradMDW

Local physical-decay extension of arXiv:2405.15682v4, §4.4, Algorithm 1,
and arXiv:2606.25971v2, Appendix A, Algorithm 2. Compute all factor gradients
at the old fused query. Advance each base with raw variance or its AMSGrad
maximum, project the direction, and apply query-point decay to the fused
base. Repair that base before averaging in direction/raw-gain coordinates.

`useMaximum=false` uses the raw second moment and models Schedule-Free
AdamMDW. Its redundant mathematical maximum field equals that moment; the
Python implementation stores no maximum buffer in the Adam case.
-/

@[expose] public section

noncomputable section

namespace Transformer.MDW

open MagnitudeDirection ScheduleFree

variable {ι : Type*} {m n : ℕ}

/-- Raw second moment, or the running maximum of raw second moments. -/
def freeVariance (useMaximum : Bool) (p : AdamParameters) (state : AMSGrad.State ι)
    (gradient : ι → ℝ) : ι → ℝ :=
  if useMaximum then AMSGrad.nextMaximum p state gradient
  else AMSGrad.nextSecond p state gradient

/-- Algorithm 1's base proposal with query-point decay; interpolation has no first EMA. -/
def freeProposal (useMaximum : Bool) (p : AdamParameters) (t : ℕ)
    (state : AMSGrad.State ι) (gradient : ι → ℝ) : ι → ℝ :=
  fun i => state.base i - adamLearningRate p t * gradient i /
    (Real.sqrt (freeVariance useMaximum p state gradient i) + p.epsilon) -
    adamLearningRate p t * p.decay * AMSGrad.query p state i

/-- Project the adaptive base before forming the squared-effective-rate average. -/
def freeStepWith (project : (ι → ℝ) → (ι → ℝ)) (useMaximum : Bool)
    (p : AdamParameters) (t : ℕ) (state : AMSGrad.State ι)
    (gradient : ι → ℝ) : AMSGrad.State ι :=
  let eta := adamLearningRate p t
  let base := if eta = 0 then state.base else project (freeProposal useMaximum p t state gradient)
  let c := AMSGrad.coefficient p t state
  ⟨fun i => (1 - c) * state.evaluation i + c * base i, base,
    AMSGrad.nextSecond p state gradient, freeVariance useMaximum p state gradient,
    state.total + eta ^ 2⟩

/-- Advance all independent factor histories on the actual old query gradients. -/
def freeMDProposal (useMaximum : Bool) (p : AMSGrad.MDParameters) (t : ℕ)
    (state : AMSGrad.MDState m n) (gradient : Matrix (Fin m) (Fin n) ℝ) :
    AMSGrad.MDState m n :=
  let g := AMSGrad.mdGradients (AMSGrad.mdQuery p state) gradient
  ⟨freeStepWith (AMSGrad.projectDirection state.radius) useMaximum
      (AMSGrad.mdDirectionParameters p) t state.direction
      (fun ij => g.direction ij.1 ij.2),
    freeStepWith id useMaximum (AMSGrad.mdGainParameters p) t state.row g.row,
    freeStepWith id useMaximum (AMSGrad.mdGainParameters p) t state.column g.column,
    state.radius⟩

/-- Physical fused base, before query/evaluation interpolation. -/
def freeBaseState (state : AMSGrad.MDState m n) : FullState m n :=
  ⟨AMSGrad.factorWeight ⟨fun i j => state.direction.base (i, j),
      state.row.base, state.column.base⟩, state.row.base, state.column.base⟩

/-- The fused base receives decay at the direction's effective rate and old query weight. -/
def freeCorrectedBase (useMaximum : Bool) (p : AMSGrad.MDParameters) (decay : ℝ)
    (t : ℕ) (state : AMSGrad.MDState m n) (gradient : Matrix (Fin m) (Fin n) ℝ) :
    FullState m n :=
  correctWeight state.radius (adamLearningRate (AMSGrad.mdDirectionParameters p) t)
    decay (AMSGrad.mdQueryWeight p state)
    (freeBaseState (freeMDProposal useMaximum p t state gradient))

/-- Install a corrected base, then average from the old evaluation and old total. -/
def reaverage (p : AdamParameters) (t : ℕ) (old next : AMSGrad.State ι)
    (base : ι → ℝ) : AMSGrad.State ι :=
  { next with base := base, evaluation := fun i =>
      (1 - AMSGrad.coefficient p t old) * old.evaluation i +
        AMSGrad.coefficient p t old * base i }

/-- Full Schedule-Free MDW step. Repair precedes averaging, including in zero-result cases. -/
def freeMDStep (useMaximum : Bool) (p : AMSGrad.MDParameters) (decay : ℝ)
    (t : ℕ) (state : AMSGrad.MDState m n) (gradient : Matrix (Fin m) (Fin n) ℝ) :
    AMSGrad.MDState m n :=
  let next := freeMDProposal useMaximum p t state gradient
  let corrected := freeCorrectedBase useMaximum p decay t state gradient
  let direction := fullDirection softplus corrected
  ⟨reaverage (AMSGrad.mdDirectionParameters p) t state.direction next.direction
      (fun ij => direction ij.1 ij.2),
    reaverage (AMSGrad.mdGainParameters p) t state.row next.row corrected.rawRow,
    reaverage (AMSGrad.mdGainParameters p) t state.column next.column corrected.rawCol,
    state.radius⟩

/-- Every new gradient sees the actual fused query of the corrected factor histories. -/
def freeMDRun (useMaximum : Bool) (p : AMSGrad.MDParameters) (decay : ℝ)
    (oracle : ℕ → Matrix (Fin m) (Fin n) ℝ → Matrix (Fin m) (Fin n) ℝ)
    (initial : Matrix (Fin m) (Fin n) ℝ) : ℕ → AMSGrad.MDState m n
  | 0 => AMSGrad.mdInitial initial
  | t + 1 =>
    let previous := freeMDRun useMaximum p decay oracle initial t
    freeMDStep useMaximum p decay t previous (oracle t (AMSGrad.mdQueryWeight p previous))

end Transformer.MDW
