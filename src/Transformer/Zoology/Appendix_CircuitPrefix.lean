/-
# Topological prefixes of an arithmetic DAG

Arora et al., arXiv:2312.04927v1, §4 Theorem `thm:equiv` and Appendix
Theorem `thm: gen-ac`. Every well-formed circuit is compiled in topological
order, retaining inputs and all earlier gate values in feature memory.
This file proves that every compiled prefix preserves inputs and computes
all earlier gate values correctly in feature memory.
-/

import Transformer.Zoology.Appendix_CircuitGateCorrectness

namespace Transformer.Zoology

/-- Execute the first `k` circuit gates on persistent feature memory.
The recursion becomes stationary after all gates have been processed.
Source: Appendix Theorem `thm: gen-ac`, topological gate stages. -/
def circuitMemoryPrefix {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (x : Fin inputs → ℝ) :
    ℕ → CircuitMemory inputs c.size
  | 0 => initialCircuitMemory x
  | k + 1 =>
      if h : k < c.size then
        circuitMemoryStep c ⟨k, h⟩ (circuitMemoryPrefix c x k)
      else circuitMemoryPrefix c x k

/-- Compile the first `k` gates into a list of ordinary Coyote layers.
Source: Appendix Theorem `thm: gen-ac`, layerwise construction. -/
def compiledCircuitPrefix {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) :
    ℕ → List (CoyoteParameters 1 (inputs + c.size))
  | 0 => []
  | k + 1 =>
      if h : k < c.size then
        compiledCircuitPrefix c k ++ circuitGateLayers c ⟨k, h⟩
      else compiledCircuitPrefix c k

/-- The compiled prefix executes exactly the corresponding sequence of
semantic circuit-memory updates. Source: Appendix Theorem `thm: gen-ac`,
gatewise simulation and stacking. -/
theorem compiledCircuitPrefix_correct {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (hwell : c.WellFormed)
    (x : Fin inputs → ℝ) (k : ℕ) :
    (compiledCircuitPrefix c k).foldl
        (fun u p => coyoteLayer p u)
        (memoryAsSequence (initialCircuitMemory x)) =
      memoryAsSequence (circuitMemoryPrefix c x k) := by
  induction k with
  | zero => rfl
  | succ k ih =>
      by_cases hk : k < c.size
      · simp only [compiledCircuitPrefix, circuitMemoryPrefix,
          dite_eq_left hk, List.foldl_append]
        rw [ih]
        exact circuitGateLayers_correct c hwell ⟨k, hk⟩
          (circuitMemoryPrefix c x k)
      · simpa [compiledCircuitPrefix, circuitMemoryPrefix, hk] using ih

/-- Gate updates never change the stored original inputs.
Source: Appendix Theorem `thm: gen-ac`, remembered input wires. -/
theorem circuitMemoryPrefix_input {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (x : Fin inputs → ℝ)
    (k : ℕ) (a : Fin inputs) :
    circuitMemoryPrefix c x k (circuitInputSlot a) = x a := by
  induction k with
  | zero => exact initialCircuitMemory_input x a
  | succ k ih =>
      by_cases hk : k < c.size
      · simp only [circuitMemoryPrefix, dite_eq_left hk]
        rw [circuitMemoryStep_other c ⟨k, hk⟩ _ _
          (Ne.symm (circuitNodeSlot_ne_inputSlot ⟨k, hk⟩ a))]
        exact ih
      · simpa [circuitMemoryPrefix, hk] using ih

/-- Gate values for all earlier indices are correct after `k` updates.
This is the topological invariant needed for arbitrary DAG sharing.
Source: Appendix Theorem `thm: gen-ac`, preservation of earlier wires. -/
theorem circuitMemoryPrefix_node {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (hwell : c.WellFormed)
    (x : Fin inputs → ℝ) (k : ℕ) (j : Fin c.size)
    (hj : j.val < k) :
    circuitMemoryPrefix c x k (circuitNodeSlot j) =
      c.evalNode hwell x j := by
  induction k generalizing j with
  | zero => omega
  | succ k ih =>
      by_cases hk : k < c.size
      · let current : Fin c.size := ⟨k, hk⟩
        by_cases hcur : j = current
        · subst j
          simp only [circuitMemoryPrefix, dite_eq_left hk]
          rw [circuitMemoryStep_current]
          rw [c.evalNode_equation hwell x current]
          cases hg : c.gate ⟨k, hk⟩ with
          | input a =>
              simp [circuitGateMemoryValue, ArithmeticGate.value,
                circuitMemoryPrefix_input, hg]
          | constant r =>
              simp [circuitGateMemoryValue, ArithmeticGate.value, hg]
          | add a b =>
              have hab : a < current ∧ b < current := by
                simpa [ArithmeticCircuit.WellFormed, current, hg] using hwell current
              have ha : a.val < k := by
                simpa [Fin.lt_def, current] using hab.1
              have hb : b.val < k := by
                simpa [Fin.lt_def, current] using hab.2
              simp only [circuitGateMemoryValue, ArithmeticGate.value, hg]
              rw [ih a ha, ih b hb]
          | multiply a b =>
              have hab : a < current ∧ b < current := by
                simpa [ArithmeticCircuit.WellFormed, current, hg] using hwell current
              have ha : a.val < k := by
                simpa [Fin.lt_def, current] using hab.1
              have hb : b.val < k := by
                simpa [Fin.lt_def, current] using hab.2
              simp only [circuitGateMemoryValue, ArithmeticGate.value, hg]
              rw [ih a ha, ih b hb]
        · have hjk : j.val < k := by
            have hne : j.val ≠ k := by
              intro heq
              exact hcur (Fin.ext heq)
            omega
          simp only [circuitMemoryPrefix, dite_eq_left hk]
          rw [circuitMemoryStep_other c current _ _ (by
            intro heq
            exact hcur (circuitNodeSlot_injective heq))]
          exact ih j hjk
      · have hjk : j.val < k := by omega
        simpa [circuitMemoryPrefix, hk] using ih j hjk

end Transformer.Zoology
