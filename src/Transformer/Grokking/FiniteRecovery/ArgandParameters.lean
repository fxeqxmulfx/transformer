module

public import Mathlib.Tactic.DeriveFintype
public import Transformer.Grokking.FiniteRecovery.Argand
public import Transformer.AMSGradW.Kernite

/-! Independent real coordinates for the single-stream Argand parameter
layout in experiments/phasor_value_transport/src/phasor_v2/weights.rs.
The inert alignment coordinate in the GPU array is omitted. The initial
values take the sampled tensor entries as inputs, so every seed is covered. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2
open Transformer.AMSGradW

variable {A : Type*} {H d layers T : ℕ}

/-- A complex number in a learned tensor. The component of an input
projection selects query/key/value or gate/up. A residual gate slot
selects the attention/FFN sublayer and the skip/write vector. -/
inductive ComplexParameter (A : Type*) (H d layers : ℕ) where
  | embedding : A → Fin (H * d) → ComplexParameter A H d layers
  | attentionIn : Fin layers → Fin 3 → Fin (H * d) → Fin H → Fin d → ComplexParameter A H d layers
  | attentionOut : Fin layers → Fin H → Fin d → Fin (H * d) → ComplexParameter A H d layers
  | feedForwardIn : Fin layers → Fin 2 → Fin (H * d) → Fin (4 * (H * d)) → ComplexParameter A H d layers
  | feedForwardOut : Fin layers → Fin (4 * (H * d)) → Fin (H * d) → ComplexParameter A H d layers
  | score : Fin layers → ComplexParameter A H d layers
  | residualGate : Fin layers → Fin 2 → Fin 2 → Fin (H * d) → ComplexParameter A H d layers
  deriving Fintype

/-- False selects the real part and true the imaginary part. The other
summand holds the shared real LeRoPE log-frequency adjustments. -/
abbrev Parameter (A : Type*) (H d layers : ℕ) := (ComplexParameter A H d layers × Bool) ⊕ Fin d

abbrev Weights (A : Type*) (H d layers : ℕ) := Parameter A H d layers → ℝ

def complexEntry (w : Weights A H d layers) (p : ComplexParameter A H d layers) : ℂ :=
  ⟨w (.inl (p, false)), w (.inl (p, true))⟩

/-- The frozen initial state with ARGAND_SCORE_INIT=1. Gaussian draws
provide the complex embedding and projection entries. -/
def initialWeights (draw : ComplexParameter A H d layers → ℂ) : Weights A H d layers
  | .inr _ => 0
  | .inl (.score _, part) => if part then 0 else 1
  | .inl (.residualGate _ _ _ _, _) => 0
  | .inl (p, part) => if part then (draw p).im else (draw p).re

noncomputable def streamProjection {c : ℕ} {E : Type*} [AddCommMonoid E] [Module ℂ E]
    (columns : Fin c → E) : CSpace c →ₗ[ℂ] E :=
  ∑ i, (PiLp.projₗ 2 (𝕜 := ℂ) (fun _ : Fin c => ℂ) i).smulRight (columns i)

noncomputable def vectorProjection {ι : Type*} [Fintype ι] {E : Type*} [AddCommMonoid E] [Module ℂ E]
    (columns : ι → E) : (ι → ℂ) →ₗ[ℂ] E :=
  ∑ i, (LinearMap.proj (R := ℂ) (φ := fun _ => ℂ) i).smulRight (columns i)

noncomputable def headProjection {c : ℕ}
    (columns : Fin H → Fin d → CSpace c) : Heads H d →ₗ[ℂ] CSpace c :=
  ∑ h, (vectorProjection (columns h)).comp (LinearMap.proj (R := ℂ) (φ := fun _ => Fin d → ℂ) h)

/-- Assemble the complete skip/write single-stream forward pass from the
independent real coordinates. RMS offset is the runner's fixed 1e-6. -/
noncomputable def Model.fromWeights (w : Weights A H d layers) : Model A H d layers where
  embedding a := WithLp.toLp 2 fun i => complexEntry w (.embedding a i)
  alpha i := w (.inr i)
  eps := 1 / 1000000
  layer l :=
    let attentionIn (k : Fin 3) := streamProjection fun i h j => complexEntry w (.attentionIn l k i h j)
    let ffnIn (k : Fin 2) := streamProjection fun i j => complexEntry w (.feedForwardIn l k i j)
    let gate (s k : Fin 2) := streamProjection fun i => complexEntry w (.residualGate l s k i)
    { attentionProjections :=
        ⟨attentionIn 0, attentionIn 1, attentionIn 2,
          headProjection fun h j => WithLp.toLp 2 fun i => complexEntry w (.attentionOut l h j i)⟩
      feedForward :=
        ⟨ffnIn 0, ffnIn 1,
          vectorProjection fun j => WithLp.toLp 2 fun i => complexEntry w (.feedForwardOut l j i)⟩
      scoreMultiplier := complexEntry w (.score l)
      attentionGates := ⟨gate 0 0, gate 0 1⟩
      feedForwardGates := ⟨gate 1 0, gate 1 1⟩ }

/-- The actual initial gate vectors and frequency corrections are zero,
and every learned score multiplier starts at one. -/
theorem initialWeights_layer (draw : ComplexParameter A H d layers → ℂ) (l : Fin layers) :
    let model := Model.fromWeights (initialWeights draw)
    model.alpha = 0 ∧ (model.layer l).scoreMultiplier = 1 ∧
      (model.layer l).attentionGates = ⟨0, 0⟩ ∧ (model.layer l).feedForwardGates = ⟨0, 0⟩ := by
  have hs (s k : Fin 2) : (streamProjection fun i =>
      complexEntry (initialWeights draw) (.residualGate l s k i)) = 0 := by
    have hz (i : Fin (H * d)) : complexEntry (initialWeights draw) (.residualGate l s k i) = 0 := by
      exact Complex.ext rfl rfl
    simp only [streamProjection, hz, LinearMap.smulRight_zero, Finset.sum_const_zero]
  refine ⟨rfl, ?_, ?_, ?_⟩
  · exact Complex.ext rfl rfl
  · exact congrArg₂ Gates.mk (hs 0 0) (hs 0 1)
  · exact congrArg₂ Gates.mk (hs 1 0) (hs 1 1)

/-- A supervised position in a packed training minibatch. Prefix lengths
can differ between batches, and a list can have multiple labeled positions
in one word. Padding positions with no label are absent from the list. -/
abbrev LabeledPosition (A : Type*) := Σ T : ℕ, Word A (T + 1) × Fin (T + 1) × A

noncomputable def batchLoss [Fintype A] (repeats : ℕ) (batch : List (LabeledPosition A))
    (w : Weights A H d layers) : ℝ :=
  ((batch.map fun p => -Real.log
    ((Model.fromWeights w).probability repeats p.2.1 p.2.2.1 p.2.2.2)).sum) / batch.length

noncomputable def batchGradient [Fintype A] (repeats : ℕ) (batch : List (LabeledPosition A))
    (w : Weights A H d layers) : Weights A H d layers :=
  WithLp.ofLp (gradient
    (fun x : Kernite.Space (Parameter A H d layers) => batchLoss repeats batch (WithLp.ofLp x))
    (WithLp.toLp 2 w))

/-- Run the clipped bias-corrected update on all trainable Argand coordinates
from the native initialization. The batch stream and learning rates are inputs. -/
noncomputable def nativeTrainingRun [Fintype A] (repeats : ℕ)
    (batches : ℕ → List (LabeledPosition A)) (rate : ℕ → ℝ)
    (draw : ComplexParameter A H d layers → ℂ) : ℕ → Kernite.State (Parameter A H d layers) :=
  Kernite.run rate (fun t => batchGradient repeats (batches t)) (initialWeights draw)

example (draw : ComplexParameter (Fin 256) 4 16 2 → ℂ) := initialWeights_layer draw 0

end Transformer.Grokking.FiniteRecovery.Argand
