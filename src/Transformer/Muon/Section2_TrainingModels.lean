/-
# Muon — actual training recurrences and the new direction safeguard

arXiv:2502.16982, §2.1–2.2. The original recurrence retains the printed
five-step Newton–Schulz coefficients, Nesterov input, shape scaling and
weight decay. The corrected recurrence adds the explicitly documented
gradient check and uses `η=σ/L` on a fixed deterministic objective.
-/

import Transformer.Muon.Section2_Models
import Transformer.Optimization.Basic
import Transformer.Optimization.Matrix

noncomputable section

namespace Transformer.Muon

open Optimization

variable {a b : ℕ}

/-- Compute the complete paper candidate and update momentum using the
current parameter and actual gradient entries. The direction includes
the source's weight decay. Source: arXiv:2502.16982, §2.1, `eq:Ot` and
Nesterov footnote, and §2.2, final shape-adjusted update. -/
def muonTrainingCandidate (μ wd : ℝ) (M : Matrix (Fin a) (Fin b) ℝ)
    (W G : MatrixSpace a b) : Matrix (Fin a) (Fin b) ℝ × MatrixSpace a b :=
  (momentumStep μ M (toMatrix G),
    fromMatrix (adjustedUpdate (approximatePolar (nesterovInput μ M (toMatrix G))) +
      wd • toMatrix W))

/-- Taking the candidate without the new guard is exactly the complete
practical Muon parameter step, including all five printed matrix
iterations. Source: arXiv:2502.16982, §2.1–2.2. -/
theorem muonTrainingCandidate_step (μ wd η : ℝ) (M : Matrix (Fin a) (Fin b) ℝ)
    (W G : MatrixSpace a b) :
    toMatrix (W - η • (muonTrainingCandidate μ wd M W G).2) =
      (muonStep μ η wd M (toMatrix W) (toMatrix G)).2 := by
  ext i j
  rfl

/-- Original fixed-objective Muon training with zero initial momentum
and an arbitrary learning-rate schedule. Gradients are derivatives of
the actual objective; no free gradient history is supplied.
Source: arXiv:2502.16982, §2.1–2.2. -/
def muonTrainingRun (μ wd : ℝ) (η : ℕ → ℝ) (f : MatrixSpace a b → ℝ)
    (initial : MatrixSpace a b) : ℕ → Matrix (Fin a) (Fin b) ℝ × MatrixSpace a b
  | 0 => (0, initial)
  | t + 1 =>
    let state := muonTrainingRun μ wd η f initial t
    let step := muonStep μ (η t) wd state.1 (toMatrix state.2)
      (toMatrix (gradient f state.2))
    (step.1, fromMatrix step.2)

/-- Corrected learning algorithm: compute the unchanged paper candidate,
update its actual momentum, then check the complete direction against
the current full gradient before stepping with `η=σ/L`.
This is a training correction, not the manuscript's unmodified update.
Source: extension of arXiv:2502.16982, §2.1–2.2. -/
def safeguardedMuonRun (σ L μ wd : ℝ) (f : MatrixSpace a b → ℝ)
    (initial : MatrixSpace a b) : ℕ → Matrix (Fin a) (Fin b) ℝ × MatrixSpace a b :=
  safeguardedRun σ L f (muonTrainingCandidate μ wd) 0 initial

end Transformer.Muon
