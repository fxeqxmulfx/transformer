/-
# Correctness of parallel circuit-level updates

Arora et al., arXiv:2312.04927v1, Appendix Theorem `thm: gen-ac`.
All gates at one topological level can read earlier node slots and update
distinct target slots simultaneously. The first Coyote layer performs the
linear stages; the second completes multiplication gates.
-/

import Transformer.Zoology.Appendix_CircuitBatchDefs

namespace Transformer.Zoology

/-- A level assignment places every predecessor strictly below its gate.
Source: Appendix `def: circuit-tuple` and Theorem `thm: gen-ac`, circuit
depth and level decomposition. -/
def CircuitLevelOrder {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) : Prop :=
  ∀ i,
    match c.gate i with
    | .input _ | .constant _ => True
    | .add a b | .multiply a b =>
        level a < level i ∧ level b < level i

/-- The identity circuit admits a valid level assignment. -/
example : CircuitLevelOrder identityCircuit (fun _ => 0) := by
  intro i
  fin_cases i
  simp [identityCircuit]

/-- A feature coordinate can be active at level `ℓ` only when its owner
has that level. Source: Appendix Theorem `thm: gen-ac`. -/
theorem activeCircuitNode_level {inputs nodes : ℕ}
    (level : Fin nodes → ℕ) (ℓ : ℕ)
    (q : Fin (inputs + nodes)) (i : Fin nodes)
    (h : activeCircuitNode level ℓ q = some i) : level i = ℓ := by
  unfold activeCircuitNode at h
  cases ho : circuitSlotOwner q with
  | none => simp [ho] at h
  | some j =>
      simp [ho] at h
      exact h.2

/-- Ideal simultaneous semantic update of all gates on one level;
inactive coordinates are preserved. Source: Appendix Theorem
`thm: gen-ac`, circuit-level construction. -/
def circuitLevelStep {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) (ℓ : ℕ)
    (state : CircuitMemory inputs c.size) : CircuitMemory inputs c.size :=
  fun q =>
    match activeCircuitNode level ℓ q with
    | some i => circuitGateMemoryValue c i state
    | none => state q

/-- The two parallel Coyote stages equal the ideal simultaneous update.
The level-order hypothesis ensures the multiplication layer reads an
unchanged earlier left operand. Source: Appendix Theorem `thm: gen-ac`,
levelwise gate simulation. -/
theorem circuitBatchStages_correct {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) (horder : CircuitLevelOrder c level)
    (ℓ : ℕ) (state : CircuitMemory inputs c.size) :
    circuitBatchMultiplyState c level ℓ
        (circuitBatchLinearState c level ℓ state) =
      circuitLevelStep c level ℓ state := by
  funext q
  cases ha : activeCircuitNode level ℓ q with
  | none =>
      simp [circuitBatchMultiplyState, circuitBatchMultiplyLeft,
        circuitBatchLinearState, circuitLevelStep, ha]
  | some i =>
      have hi : level i = ℓ := activeCircuitNode_level level ℓ q i ha
      cases hg : c.gate i with
      | input a =>
          simp [circuitBatchMultiplyState, circuitBatchMultiplyLeft,
            circuitBatchLinearState, circuitLevelStep,
            circuitGateMemoryValue, circuitGateLinear_value, ha, hg]
      | constant r =>
          simp [circuitBatchMultiplyState, circuitBatchMultiplyLeft,
            circuitBatchLinearState, circuitLevelStep,
            circuitGateMemoryValue, circuitGateLinear_value, ha, hg]
      | add a b =>
          simp [circuitBatchMultiplyState, circuitBatchMultiplyLeft,
            circuitBatchLinearState, circuitLevelStep,
            circuitGateMemoryValue, circuitGateLinear_value, ha, hg]
      | multiply a b =>
          have hpred : level a < level i ∧ level b < level i := by
            simpa [CircuitLevelOrder, hg] using horder i
          have hleft : activeCircuitNode level ℓ
              (circuitNodeSlot (inputs := inputs) a) = none := by
            rw [activeCircuitNode_node]
            simp [ne_of_lt (hi ▸ hpred.1)]
          simp [circuitBatchMultiplyState, circuitBatchMultiplyLeft,
            circuitBatchLinearState, circuitLevelStep,
            circuitGateMemoryValue, circuitGateLinear_value,
            ha, hg, hleft]

/-- Two Coyote layers implement all gates on one circuit level.
Source: Appendix Theorem `thm: gen-ac`, addition and multiplication stages. -/
theorem circuitLevelLayers_correct {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) (horder : CircuitLevelOrder c level)
    (ℓ : ℕ) (state : CircuitMemory inputs c.size) :
    ([circuitBatchLinearParameters c level ℓ,
        circuitBatchMultiplyParameters c level ℓ] :
        List (CoyoteParameters 1 (inputs + c.size))).foldl
        (fun u p => coyoteLayer p u) (memoryAsSequence state) =
      memoryAsSequence (circuitLevelStep c level ℓ state) := by
  simp only [List.foldl_cons, List.foldl_nil]
  rw [circuitBatchLinearParameters_apply,
    circuitBatchMultiplyParameters_apply,
    circuitBatchStages_correct c level horder ℓ state]

end Transformer.Zoology
