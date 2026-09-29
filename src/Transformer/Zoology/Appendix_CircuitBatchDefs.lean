/-
# Parallel Coyote stages for circuit levels

Arora et al., arXiv:2312.04927v1, Appendix Theorem `thm: gen-ac`,
decomposition into circuit levels. The same Coyote feature matrix can
update every gate slot on one level in parallel. A second layer completes
multiplication gates after their right operands have been copied to their
target slots.
-/

import Transformer.Zoology.Appendix_CircuitSimulation

open scoped BigOperators

namespace Transformer.Zoology

/-- Identify the circuit-node slot corresponding to a feature coordinate;
input slots have no node owner. Source: Appendix Theorem `thm: gen-ac`,
persistent wire layout. -/
def circuitSlotOwner {inputs nodes : ℕ}
    (q : Fin (inputs + nodes)) : Option (Fin nodes) :=
  if h : inputs ≤ q.val then
    some ⟨q.val - inputs, by omega⟩
  else none

/-- A node owns its assigned memory coordinate.
Source: Appendix Theorem `thm: gen-ac`, wire layout. -/
theorem circuitSlotOwner_node {inputs nodes : ℕ} (i : Fin nodes) :
    circuitSlotOwner (circuitNodeSlot (inputs := inputs) i) = some i := by
  simp [circuitSlotOwner, circuitNodeSlot]

/-- No input coordinate belongs to a circuit node.
Source: Appendix Theorem `thm: gen-ac`, wire layout. -/
theorem circuitSlotOwner_input {inputs nodes : ℕ} (a : Fin inputs) :
    circuitSlotOwner (circuitInputSlot (nodes := nodes) a) = none := by
  simp [circuitSlotOwner, circuitInputSlot, Nat.not_le_of_gt a.isLt]

/-- The node on level `ℓ` assigned to a feature coordinate, if any.
Source: Appendix Theorem `thm: gen-ac`, levelwise gate stages. -/
def activeCircuitNode {inputs nodes : ℕ}
    (level : Fin nodes → ℕ) (ℓ : ℕ)
    (q : Fin (inputs + nodes)) : Option (Fin nodes) :=
  (circuitSlotOwner q).filter fun i => decide (level i = ℓ)

/-- A node slot is active exactly on its designated level.
Source: Appendix Theorem `thm: gen-ac`, level decomposition. -/
theorem activeCircuitNode_node {inputs nodes : ℕ}
    (level : Fin nodes → ℕ) (ℓ : ℕ) (i : Fin nodes) :
    activeCircuitNode (inputs := inputs) level ℓ (circuitNodeSlot i) =
      if level i = ℓ then some i else none := by
  simp [activeCircuitNode, circuitSlotOwner_node, Option.filter]

/-- Input slots are inactive on every circuit level.
Source: Appendix Theorem `thm: gen-ac`, retained inputs. -/
theorem activeCircuitNode_input {inputs nodes : ℕ}
    (level : Fin nodes → ℕ) (ℓ : ℕ) (a : Fin inputs) :
    activeCircuitNode level ℓ (circuitInputSlot (nodes := nodes) a) = none := by
  simp [activeCircuitNode, circuitSlotOwner_input]

/-- The first parallel Coyote layer simultaneously performs all affine
gate stages at level `ℓ`, preserving every other feature.
Source: Appendix Theorem `thm: gen-ac`, addition and wire stages. -/
def circuitBatchLinearParameters {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) (ℓ : ℕ) :
    CoyoteParameters 1 (inputs + c.size) := {
  weight := fun k q =>
    match activeCircuitNode level ℓ q with
    | some i => circuitGateLinearCoeff (c.gate i) k
    | none => identityWeight (inputs + c.size) k q
  filter := fun _ _ => 0
  bias₁ := fun _ q =>
    match activeCircuitNode level ℓ q with
    | some i => circuitGateLinearBias (c.gate i)
    | none => 0
  bias₂ := fun _ _ => 1
}

/-- Functional result of the first parallel layer.
Source: Appendix Theorem `thm: gen-ac`, affine stages. -/
def circuitBatchLinearState {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) (ℓ : ℕ)
    (state : CircuitMemory inputs c.size) : CircuitMemory inputs c.size :=
  fun q =>
    match activeCircuitNode level ℓ q with
    | some i =>
        (∑ k, state k * circuitGateLinearCoeff (c.gate i) k) +
          circuitGateLinearBias (c.gate i)
    | none => state q

/-- One Coyote layer equals the simultaneous affine state update.
Source: Appendix Theorem `thm: gen-ac`, levelwise linear stage. -/
theorem circuitBatchLinearParameters_apply {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) (ℓ : ℕ)
    (state : CircuitMemory inputs c.size) :
    coyoteLayer (circuitBatchLinearParameters c level ℓ)
        (memoryAsSequence state) =
      memoryAsSequence (circuitBatchLinearState c level ℓ state) := by
  funext p q
  fin_cases p
  cases hactive : activeCircuitNode level ℓ q with
  | none =>
      simp [coyoteLayer, circuitBatchLinearParameters,
        circuitBatchLinearState, memoryAsSequence, linearProjection,
        causalConvolution, identityWeight, hactive]
  | some i =>
      simp [coyoteLayer, circuitBatchLinearParameters,
        circuitBatchLinearState, memoryAsSequence, linearProjection,
        causalConvolution, hactive]

/-- If an active node is a multiplication gate, identify the feature slot
holding its left operand. Otherwise the second stage preserves the feature.
Source: Appendix Theorem `thm: gen-ac`, multiplication stage. -/
def circuitBatchMultiplyLeft {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) (ℓ : ℕ)
    (q : Fin (inputs + c.size)) : Option (Fin (inputs + c.size)) :=
  match activeCircuitNode level ℓ q with
  | some i =>
      match c.gate i with
      | .multiply a _ => some (circuitNodeSlot a)
      | _ => none
  | none => none

/-- A second parallel Coyote layer multiplies all active multiplication
gates and preserves every other coordinate. Source: Appendix Theorem
`thm: gen-ac`, multiplication stage. -/
def circuitBatchMultiplyParameters {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) (ℓ : ℕ) :
    CoyoteParameters 1 (inputs + c.size) := {
  weight := fun k q =>
    match circuitBatchMultiplyLeft c level ℓ q with
    | some left => circuitCoordinateCoeff left k
    | none => identityWeight (inputs + c.size) k q
  filter := fun _ q =>
    if (circuitBatchMultiplyLeft c level ℓ q).isSome then 1 else 0
  bias₁ := fun _ _ => 0
  bias₂ := fun _ q =>
    if (circuitBatchMultiplyLeft c level ℓ q).isSome then 0 else 1
}

/-- Functional result of the second parallel layer.
Source: Appendix Theorem `thm: gen-ac`, multiplication stage. -/
def circuitBatchMultiplyState {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) (ℓ : ℕ)
    (state : CircuitMemory inputs c.size) : CircuitMemory inputs c.size :=
  fun q =>
    match circuitBatchMultiplyLeft c level ℓ q with
    | some left => state left * state q
    | none => state q

/-- The second Coyote layer performs the simultaneous multiplication
state update. Source: Appendix Theorem `thm: gen-ac`, levelwise
multiplication stage. -/
theorem circuitBatchMultiplyParameters_apply {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) (ℓ : ℕ)
    (state : CircuitMemory inputs c.size) :
    coyoteLayer (circuitBatchMultiplyParameters c level ℓ)
        (memoryAsSequence state) =
      memoryAsSequence (circuitBatchMultiplyState c level ℓ state) := by
  funext p q
  fin_cases p
  cases hleft : circuitBatchMultiplyLeft c level ℓ q with
  | none =>
      simp [coyoteLayer, circuitBatchMultiplyParameters,
        circuitBatchMultiplyState, memoryAsSequence, linearProjection,
        causalConvolution, identityWeight, hleft]
  | some left =>
      simp [coyoteLayer, circuitBatchMultiplyParameters,
        circuitBatchMultiplyState, memoryAsSequence, linearProjection,
        causalConvolution, circuitCoordinateCoeff_sum, hleft]

end Transformer.Zoology
