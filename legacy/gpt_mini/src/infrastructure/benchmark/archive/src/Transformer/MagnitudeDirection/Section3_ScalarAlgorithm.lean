/-
# Scalar-gain optimizer wrapper

arXiv:2606.25971v2, §3.1, Algorithm 1. Optimizer callbacks describe the
current step of any base optimizer; momentum/state evolution belongs to
the callback. No guarantee of convergence is assumed for either callback.
-/

import Transformer.MagnitudeDirection.Section3_Sphere
import Transformer.MagnitudeDirection.Section3_Gradients

noncomputable section

namespace Transformer.MagnitudeDirection

/-- A current base-optimizer step, arXiv:2606.25971v2, Algorithms 1–2.
The arguments are current direction, its gradient, and direction LR. -/
abbrev MatrixStep (m n : ℕ) := Matrix (Fin m) (Fin n) ℝ →
  Matrix (Fin m) (Fin n) ℝ → ℝ → Matrix (Fin m) (Fin n) ℝ

/-- A vector-gain optimizer step, arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
abbrev GainStep (k : ℕ) := (Fin k → ℝ) → (Fin k → ℝ) → ℝ → (Fin k → ℝ)

/-- Fused storage and scalar gain, arXiv:2606.25971v2, §3.1, Algorithm 1. -/
structure ScalarState (m n : ℕ) where
  weight : Matrix (Fin m) (Fin n) ℝ
  gain : ℝ

variable {m n : ℕ}

/-- Pre-projection direction candidate in Algorithm 1, lines 1–4,
arXiv:2606.25971v2, §3.1. -/
def scalarCandidate (opt : MatrixStep m n) (s : ScalarState m n)
    (G : Matrix (Fin m) (Fin n) ℝ) (etaW : ℝ) : Matrix (Fin m) (Fin n) ℝ :=
  opt (unfuse (fun _ => s.gain) (fun _ => 1) s.weight) (s.gain • G) etaW

/-- Literal scalar-gain MD step, arXiv:2606.25971v2, §3.1, Algorithm 1.
The gain callback receives the old gain and old recovered direction's
gradient; reassembly uses the newly updated gain. -/
def scalarStep (opt : MatrixStep m n) (gainOpt : ℝ → ℝ → ℝ → ℝ)
    (s : ScalarState m n) (G : Matrix (Fin m) (Fin n) ℝ)
    (etaW etaG c : ℝ) : ScalarState m n :=
  let D := unfuse (fun _ => s.gain) (fun _ => 1) s.weight
  let nextGain := gainOpt s.gain (frobeniusPairing D G) etaG
  let nextDirection := matrixProject c (scalarCandidate opt s G etaW)
  ⟨fuse (fun _ => nextGain) (fun _ => 1) nextDirection, nextGain⟩

/-- For every optimizer, the recovered new scalar direction lies on the
fixed sphere if the candidate and new gain are nonzero. These conditions
are missing from the source's unqualified divisions.
Source: arXiv:2606.25971v2, §3.1, Algorithm 1, lines 5–7. -/
theorem scalarStep_direction_norm (opt : MatrixStep m n) (gainOpt : ℝ → ℝ → ℝ → ℝ)
    (s : ScalarState m n) (G : Matrix (Fin m) (Fin n) ℝ) (etaW etaG c : ℝ)
    (hc : 0 ≤ c) (hC : scalarCandidate opt s G etaW ≠ 0)
    (hg : (scalarStep opt gainOpt s G etaW etaG c).gain ≠ 0) :
    frobeniusNorm (unfuse (fun _ => (scalarStep opt gainOpt s G etaW etaG c).gain)
      (fun _ => 1) (scalarStep opt gainOpt s G etaW etaG c).weight) = c := by
  dsimp [scalarStep] at hg ⊢
  rw [unfuse_fuse _ _ _ (fun _ => hg) (fun _ => one_ne_zero)]
  exact matrixProject_norm c _ hc hC

/-- A zero-LR gradient step witnesses all scalar invariant hypotheses,
arXiv:2606.25971v2, Algorithm 1. -/
example : (0 : ℝ) ≤ 1 ∧
    scalarCandidate (fun D G eta => D - eta • G)
      (⟨1, 1⟩ : ScalarState 1 1) 0 0 ≠ 0 ∧
    (scalarStep (fun D G eta => D - eta • G) (fun a g eta => a - eta * g)
      (⟨1, 1⟩ : ScalarState 1 1) 0 0 0 1).gain ≠ 0 := by
  norm_num [scalarCandidate, scalarStep, unfuse]

/-- Direct gains may hit zero even while the projected direction is valid.
The next recovery then loses the direction. This is a singularity of the
literal direct-gain algorithm, not a claim of observed training failure.
Source: arXiv:2606.25971v2, §4.1.3 and Algorithm 1, line 1. -/
theorem zero_direct_gain_counterexample :
    let s := scalarStep (fun D G eta => D - eta • G) (fun a g eta => a - eta * g)
      (⟨1, 1⟩ : ScalarState 1 1) 1 0 1 1
    s.gain = 0 ∧ s.weight = 0 ∧
      unfuse (fun _ => s.gain) (fun _ => 1) s.weight = 0 := by
  dsimp [scalarStep, scalarCandidate]
  norm_num [frobeniusPairing, unfuse, fuse]

end Transformer.MagnitudeDirection
