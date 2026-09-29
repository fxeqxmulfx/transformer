/-
# Induction over parallel circuit levels

Arora et al., arXiv:2312.04927v1, Appendix Theorem `thm: gen-ac`.
The output of every gate on earlier levels is retained. Two ordinary
Coyote layers per level compute all new gates simultaneously, using the
topological level assignment from the paper's depth convention.
-/

import Transformer.Zoology.Appendix_CircuitBatchCorrectness

namespace Transformer.Zoology

/-- Execute all circuit levels below `k` on the persistent feature memory.
Source: Appendix Theorem `thm: gen-ac`, level decomposition. -/
def circuitLevelMemoryPrefix {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) (x : Fin inputs → ℝ) :
    ℕ → CircuitMemory inputs c.size
  | 0 => initialCircuitMemory x
  | ℓ + 1 =>
      circuitLevelStep c level ℓ (circuitLevelMemoryPrefix c level x ℓ)

/-- The two Coyote layers for each level below `k`.
Source: Appendix Theorem `thm: gen-ac`, parallel gate construction. -/
def compiledCircuitLevelPrefix {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) :
    ℕ → List (CoyoteParameters 1 (inputs + c.size))
  | 0 => []
  | ℓ + 1 =>
      compiledCircuitLevelPrefix c level ℓ ++
        [circuitBatchLinearParameters c level ℓ,
          circuitBatchMultiplyParameters c level ℓ]

/-- The compiled level prefix executes exactly the ideal simultaneous
level updates. Source: Appendix Theorem `thm: gen-ac`, parallel simulation. -/
theorem compiledCircuitLevelPrefix_correct {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) (horder : CircuitLevelOrder c level)
    (x : Fin inputs → ℝ) (k : ℕ) :
    (compiledCircuitLevelPrefix c level k).foldl
        (fun u p => coyoteLayer p u)
        (memoryAsSequence (initialCircuitMemory x)) =
      memoryAsSequence (circuitLevelMemoryPrefix c level x k) := by
  induction k with
  | zero => rfl
  | succ k ih =>
      simp only [compiledCircuitLevelPrefix, circuitLevelMemoryPrefix,
        List.foldl_append]
      rw [ih]
      exact circuitLevelLayers_correct c level horder k
        (circuitLevelMemoryPrefix c level x k)

/-- The levelwise updates leave every original input feature unchanged.
Source: Appendix Theorem `thm: gen-ac`, remembered inputs. -/
theorem circuitLevelMemoryPrefix_input {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) (x : Fin inputs → ℝ)
    (k : ℕ) (a : Fin inputs) :
    circuitLevelMemoryPrefix c level x k (circuitInputSlot a) = x a := by
  induction k with
  | zero => exact initialCircuitMemory_input x a
  | succ k ih =>
      simp [circuitLevelMemoryPrefix, circuitLevelStep,
        activeCircuitNode_input, ih]

/-- After processing levels below `k`, every gate on an earlier level
has its recursively evaluated value in its own feature slot.
Source: Appendix Theorem `thm: gen-ac`, induction across levels. -/
theorem circuitLevelMemoryPrefix_node {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (hwell : c.WellFormed)
    (level : Fin c.size → ℕ) (horder : CircuitLevelOrder c level)
    (x : Fin inputs → ℝ) (k : ℕ) (j : Fin c.size)
    (hj : level j < k) :
    circuitLevelMemoryPrefix c level x k (circuitNodeSlot j) =
      c.evalNode hwell x j := by
  induction k generalizing j with
  | zero => omega
  | succ k ih =>
      by_cases hcurrent : level j = k
      · simp only [circuitLevelMemoryPrefix, circuitLevelStep]
        rw [activeCircuitNode_node, ite_eq_left hcurrent]
        rw [c.evalNode_equation hwell x j]
        cases hg : c.gate j with
        | input a =>
            simp [circuitGateMemoryValue, ArithmeticGate.value,
              circuitLevelMemoryPrefix_input, hg]
        | constant r =>
            simp [circuitGateMemoryValue, ArithmeticGate.value, hg]
        | add a b =>
            have hab : level a < level j ∧ level b < level j := by
              simpa [CircuitLevelOrder, hg] using horder j
            simp only [circuitGateMemoryValue, ArithmeticGate.value, hg]
            rw [ih a (by omega), ih b (by omega)]
        | multiply a b =>
            have hab : level a < level j ∧ level b < level j := by
              simpa [CircuitLevelOrder, hg] using horder j
            simp only [circuitGateMemoryValue, ArithmeticGate.value, hg]
            rw [ih a (by omega), ih b (by omega)]
      · have hprev : level j < k := by omega
        simp only [circuitLevelMemoryPrefix, circuitLevelStep]
        rw [activeCircuitNode_node, ite_eq_right hcurrent]
        exact ih j hprev

/-- Exactly two ordinary Coyote layers are emitted per circuit level.
Source: Appendix Theorem `thm: gen-ac`, parallel stage count. -/
theorem compiledCircuitLevelPrefix_length {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) (k : ℕ) :
    (compiledCircuitLevelPrefix c level k).length = 2 * k := by
  induction k with
  | zero => rfl
  | succ k ih =>
      simp [compiledCircuitLevelPrefix, ih]
      omega

end Transformer.Zoology
