/-
# Correctness of each compiled arithmetic gate

Arora et al., arXiv:2312.04927v1, Appendix Theorem `thm: gen-ac`.
The gate compiler uses one Coyote layer for input, constant, and addition
gates, and two layers for multiplication. Well-formedness makes the target
slot fresh, so the multiplication layer can still read its left operand.
-/

import Transformer.Zoology.Appendix_CircuitGateLayers

namespace Transformer.Zoology

/-- One compiled gate performs exactly the semantic memory update.
Source: Appendix Theorem `thm: gen-ac`, gatewise simulation, with the
topological predecessor condition made explicit. -/
theorem circuitGateLayers_correct {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (hwell : c.WellFormed)
    (i : Fin c.size) (state : CircuitMemory inputs c.size) :
    (circuitGateLayers c i).foldl
        (fun u p => coyoteLayer p u) (memoryAsSequence state) =
      memoryAsSequence (circuitMemoryStep c i state) := by
  cases hg : c.gate i with
  | input a =>
      simp only [circuitGateLayers, hg, List.foldl_cons, List.foldl_nil]
      rw [writeLinearGateParameters_apply]
      simp [circuitGateLinear_value, circuitMemoryStep,
        circuitGateMemoryValue, hg]
  | constant r =>
      simp only [circuitGateLayers, hg, List.foldl_cons, List.foldl_nil]
      rw [writeLinearGateParameters_apply]
      simp [circuitGateLinear_value, circuitMemoryStep,
        circuitGateMemoryValue, hg]
  | add a b =>
      simp only [circuitGateLayers, hg, List.foldl_cons, List.foldl_nil]
      rw [writeLinearGateParameters_apply]
      simp [circuitGateLinear_value, circuitMemoryStep,
        circuitGateMemoryValue, hg]
  | multiply a b =>
      have hab : a < i ∧ b < i := by
        simpa [ArithmeticCircuit.WellFormed, hg] using hwell i
      have hleft : circuitNodeSlot (inputs := inputs) a ≠
          circuitNodeSlot (inputs := inputs) i := by
        intro h
        exact (ne_of_lt hab.1) (circuitNodeSlot_injective h)
      simp only [circuitGateLayers, hg, List.foldl_cons, List.foldl_nil]
      rw [writeLinearGateParameters_apply]
      rw [circuitGateLinear_value]
      rw [multiplyGateParameters_apply]
      funext p q
      fin_cases p
      by_cases hq : q = circuitNodeSlot (inputs := inputs) i
      · subst q
        simp [memoryAsSequence, circuitMemoryStep,
          circuitGateMemoryValue, hg, writeCircuitMemory, hleft]
      · simp [memoryAsSequence, circuitMemoryStep,
          writeCircuitMemory, hq]

/-- Each circuit node costs at most two ordinary Coyote layers in the
persistent-memory construction. Source: Appendix Theorem `thm: gen-ac`,
gate stages. -/
theorem circuitGateLayers_length_le_two {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (i : Fin c.size) :
    (circuitGateLayers c i).length ≤ 2 := by
  cases h : c.gate i <;> simp [circuitGateLayers, h]

end Transformer.Zoology
