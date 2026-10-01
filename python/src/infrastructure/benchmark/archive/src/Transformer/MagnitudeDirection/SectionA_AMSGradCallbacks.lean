/-
# Concrete AMSGrad callbacks for MD

arXiv:2606.25971v2, §3.1 and Appendix A, Algorithm 2, optimizer-agnostic
training extension. The direction and both raw-gain vectors use the actual
first moment, second moment and running coordinatewise maximum of AMSGrad,
arXiv:1904.03590v4, Algorithm 1, with the §6 epsilon regularizer. This is
an explicitly specified AMSGradMD variant; the MD experiments use Adam
for gains. No bias correction or weight decay is inserted here.
-/

import Transformer.MagnitudeDirection.SectionA_StatefulAlgorithm
import Transformer.AMSGrad.Section4_TrainingModels

noncomputable section

namespace Transformer.MagnitudeDirection

open Optimization

variable {ι : Type*} {m n : ℕ}

/-- Actual three AMSGrad history buffers, indexed by parameter entries.
Source: arXiv:1904.03590v4, Algorithm 1; arXiv:2606.25971v2,
Appendix A, AMSGradMD training extension. -/
structure AMSGradBuffers (ι : Type*) where
  momentum : ι → ℝ
  second : ι → ℝ
  maximum : ι → ℝ

/-- The source's zero initial moment histories.
Source: arXiv:1904.03590v4, Algorithm 1; AMSGradMD training extension
of arXiv:2606.25971v2, Appendix A. -/
def zeroAMSGradBuffers (ι : Type*) : AMSGradBuffers ι := ⟨0, 0, 0⟩

/-- Update all actual moment histories before the parameter step.
The denominator is `epsilon + sqrt(maximum)`, exactly as in the existing
full-gradient AMSGrad formalization. Source: arXiv:1904.03590v4,
Algorithm 1 and §6; arXiv:2606.25971v2, Appendix A, AMSGradMD extension. -/
def amsgradVectorCallback (ε β β₂ : ℝ) (s : AMSGradBuffers ι)
    (x g : ι → ℝ) (eta : ℝ) : AMSGradBuffers ι × (ι → ℝ) :=
  let first := fun i => β * s.momentum i + (1 - β) * g i
  let second := fun i => β₂ * s.second i + (1 - β₂) * g i ^ 2
  let maximum := fun i => max (s.maximum i) (second i)
  (⟨first, second, maximum⟩, fun i => x i - eta * first i / (ε + Real.sqrt (maximum i)))

/-- The raw-vector callback agrees with the complete existing AMSGrad
training step when fed its actual objective gradient: all three buffers
and the actual parameter update agree. Source: arXiv:1904.03590v4,
Algorithm 1 and §6; arXiv:2606.25971v2, Appendix A, AMSGradMD extension. -/
theorem amsgradVectorCallback_matches (eta ε β β₂ : ℝ)
    (f : AMSGrad.TrainingSpace m → ℝ) (s : AMSGrad.TrainingState m) :
    let q := amsgradVectorCallback ε β β₂ ⟨s.momentum, s.second, s.maximum⟩
      (fun i => s.position i) (fun i => gradient f s.position i) eta
    q.1.momentum = (AMSGrad.trainingStep eta ε β β₂ f s).momentum ∧
      q.1.second = (AMSGrad.trainingStep eta ε β β₂ f s).second ∧
      q.1.maximum = (AMSGrad.trainingStep eta ε β β₂ f s).maximum ∧
      WithLp.toLp 2 q.2 = (AMSGrad.trainingStep eta ε β β₂ f s).position := by
  refine ⟨rfl, rfl, rfl, ?_⟩
  ext i
  dsimp [amsgradVectorCallback, AMSGrad.trainingStep]
  ring

/-- The running maximum remains nondecreasing in the actual callback,
independently of the factorization or a rejected weight proposal.
Source: arXiv:1904.03590v4, Algorithm 1; arXiv:2606.25971v2,
Appendix A, AMSGradMD training extension. -/
theorem amsgradVectorCallback_maximum_mono (ε β β₂ eta : ℝ) (s : AMSGradBuffers ι)
    (x g : ι → ℝ) (i : ι) :
    s.maximum i ≤ (amsgradVectorCallback ε β β₂ s x g eta).1.maximum i := le_max_left _ _

/-- Positive epsilon makes the actual callback's denominator positive
without a nonzero-gradient premise. Source: arXiv:1904.03590v4, §6;
arXiv:2606.25971v2, Appendix A, AMSGradMD training extension. -/
theorem amsgradVectorCallback_denominator_pos (ε β β₂ eta : ℝ) (s : AMSGradBuffers ι)
    (x g : ι → ℝ) (hε : 0 < ε) (i : ι) :
    0 < ε + Real.sqrt ((amsgradVectorCallback ε β β₂ s x g eta).1.maximum i) := by
  linarith [Real.sqrt_nonneg ((amsgradVectorCallback ε β β₂ s x g eta).1.maximum i)]

/-- The denominator's explicit regularizer domain is nonempty,
arXiv:1904.03590v4, §6; arXiv:2606.25971v2, Appendix A, AMSGradMD extension. -/
example : (0 : ℝ) < 1 := by norm_num

/-- Matrix AMSGrad with distinct buffers for every direction entry.
Source: arXiv:1904.03590v4, Algorithm 1; arXiv:2606.25971v2,
Appendix A, Algorithm 2, AMSGradMD direction callback. -/
def amsgradMatrixCallback (ε β β₂ : ℝ) :
    StatefulMatrixStep (AMSGradBuffers (Fin m × Fin n)) m n := fun s D G eta =>
  let q := amsgradVectorCallback ε β β₂ s
    (fun p => D p.1 p.2) (fun p => G p.1 p.2) eta
  (q.1, fun i j => q.2 (i, j))

/-- Separate raw-gain AMSGrad histories receive the actual chain-rule
gradient computed by Algorithm 2. Source: arXiv:1904.03590v4,
Algorithm 1; arXiv:2606.25971v2, Appendix A, AMSGradMD gain callback. -/
def amsgradGainCallback (ε β β₂ : ℝ) : StatefulGainStep (AMSGradBuffers (Fin m)) m :=
  amsgradVectorCallback ε β β₂

/-- All direction and gain optimizer memories for AMSGradMD.
Source: arXiv:2606.25971v2, Appendix A, Algorithm 2, AMSGradMD extension. -/
abbrev AMSGradMDMemory (m n : ℕ) := AMSGradBuffers (Fin m × Fin n) ×
  AMSGradBuffers (Fin m) × AMSGradBuffers (Fin n)

/-- Zero histories for all three AMSGrad optimizers.
Source: arXiv:1904.03590v4, Algorithm 1; arXiv:2606.25971v2,
Appendix A, Algorithm 2, AMSGradMD extension. -/
def zeroAMSGradMDMemory (m n : ℕ) : AMSGradMDMemory m n :=
  (zeroAMSGradBuffers _, zeroAMSGradBuffers _, zeroAMSGradBuffers _)

/-- Concrete AMSGradMD: AMSGrad direction, AMSGrad raw row gains and
AMSGrad raw column gains, distinct LRs, sphere projection and fused
reassembly. Source: arXiv:2606.25971v2, §3.1 and Appendix A, Algorithm 2,
explicit AMSGradMD variant using arXiv:1904.03590v4, Algorithm 1 and §6. -/
def amsgradMDProposal (ε β β₂ etaW etaG c : ℝ) : FullProposal (AMSGradMDMemory m n) m n :=
  statefulFullProposal (amsgradMatrixCallback ε β β₂)
    (amsgradGainCallback ε β β₂) (amsgradGainCallback ε β β₂) etaW etaG c

end Transformer.MagnitudeDirection
