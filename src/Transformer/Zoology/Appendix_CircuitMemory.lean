/-
# A persistent memory layout for arbitrary arithmetic DAGs

Arora et al., arXiv:2312.04927v1, Appendix Theorem `thm: gen-ac`.
The paper notes that later gates can read outputs from any earlier layer.
We use separate feature coordinates for all circuit inputs and all node
outputs, so each gate update preserves the values needed by later gates.
This layout is a functional basis for a general Coyote simulation; its
width is `inputs + size`, not the sharper width bound in the paper.
-/

import Transformer.Zoology.Appendix_CircuitEvaluation
import Transformer.Zoology.Appendix_Network

namespace Transformer.Zoology

/-- Feature memory holding both original inputs and computed node values.
Source: Appendix Theorem `thm: gen-ac`, preservation of earlier wires. -/
abbrev CircuitMemory (inputs nodes : ℕ) :=
  Fin (inputs + nodes) → ℝ

/-- Feature coordinate reserved for an original circuit input.
Source: Appendix Theorem `thm: gen-ac`, initial input layout. -/
def circuitInputSlot {inputs nodes : ℕ} (a : Fin inputs) :
    Fin (inputs + nodes) := ⟨a.val, by omega⟩

/-- Feature coordinate reserved for a circuit node.
Source: Appendix Theorem `thm: gen-ac`, remembered gate outputs. -/
def circuitNodeSlot {inputs nodes : ℕ} (i : Fin nodes) :
    Fin (inputs + nodes) := ⟨inputs + i.val, by omega⟩

/-- Initial memory contains the input vector and zeroes in all gate slots.
Source: Appendix Theorem `thm: gen-ac`, initial padded vector. -/
def initialCircuitMemory {inputs nodes : ℕ} (x : Fin inputs → ℝ) :
    CircuitMemory inputs nodes :=
  fun q => if h : q.val < inputs then x ⟨q.val, h⟩ else 0

/-- The input region of initial memory holds exactly the circuit input.
Source: Appendix Theorem `thm: gen-ac`, initial padded vector. -/
theorem initialCircuitMemory_input {inputs nodes : ℕ}
    (x : Fin inputs → ℝ) (a : Fin inputs) :
    initialCircuitMemory (nodes := nodes) x (circuitInputSlot a) = x a := by
  simp [initialCircuitMemory, circuitInputSlot]

/-- Gate slots start at zero. Source: Appendix Theorem `thm: gen-ac`,
initial padded vector. -/
theorem initialCircuitMemory_node {inputs nodes : ℕ}
    (x : Fin inputs → ℝ) (i : Fin nodes) :
    initialCircuitMemory x (circuitNodeSlot (inputs := inputs) i) = 0 := by
  simp [initialCircuitMemory, circuitNodeSlot]

/-- Write one feature while preserving all other coordinates.
Source: Appendix Theorem `thm: gen-ac`, retained wires. -/
def writeCircuitMemory {inputs nodes : ℕ}
    (state : CircuitMemory inputs nodes)
    (target : Fin (inputs + nodes)) (value : ℝ) :
    CircuitMemory inputs nodes :=
  fun q => if q = target then value else state q

/-- A write sets its target. Source: Appendix Theorem `thm: gen-ac`. -/
theorem writeCircuitMemory_target {inputs nodes : ℕ}
    (state : CircuitMemory inputs nodes)
    (target : Fin (inputs + nodes)) (value : ℝ) :
    writeCircuitMemory state target value target = value := by
  simp [writeCircuitMemory]

/-- A write preserves every other coordinate.
Source: Appendix Theorem `thm: gen-ac`. -/
theorem writeCircuitMemory_other {inputs nodes : ℕ}
    (state : CircuitMemory inputs nodes)
    (target q : Fin (inputs + nodes)) (value : ℝ)
    (h : q ≠ target) :
    writeCircuitMemory state target value q = state q := by
  simp [writeCircuitMemory, h]

/-- A node slot cannot coincide with any input slot.
Source: Appendix Theorem `thm: gen-ac`, disjoint storage regions. -/
theorem circuitNodeSlot_ne_inputSlot {inputs nodes : ℕ}
    (i : Fin nodes) (a : Fin inputs) :
    circuitNodeSlot i ≠ circuitInputSlot (nodes := nodes) a := by
  intro h
  have hv := congrArg Fin.val h
  dsimp [circuitNodeSlot, circuitInputSlot] at hv
  omega

/-- Different nodes have different storage coordinates.
Source: Appendix Theorem `thm: gen-ac`, node layout. -/
theorem circuitNodeSlot_injective {inputs nodes : ℕ} :
    Function.Injective (circuitNodeSlot (inputs := inputs) (nodes := nodes)) := by
  intro i j h
  apply Fin.ext
  have hv := congrArg Fin.val h
  dsimp [circuitNodeSlot] at hv
  omega

/-- Read the operands of one arithmetic gate from the persistent memory.
Source: Appendix Theorem `thm: gen-ac`, gate evaluation. -/
def circuitGateMemoryValue {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (i : Fin c.size)
    (state : CircuitMemory inputs c.size) : ℝ :=
  match c.gate i with
  | .input a => state (circuitInputSlot a)
  | .constant r => r
  | .add a b => state (circuitNodeSlot a) + state (circuitNodeSlot b)
  | .multiply a b => state (circuitNodeSlot a) * state (circuitNodeSlot b)

/-- One semantic circuit step writes its gate result to the assigned slot
and leaves every other feature intact. Source: Appendix Theorem
`thm: gen-ac`, layerwise construction. -/
def circuitMemoryStep {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (i : Fin c.size)
    (state : CircuitMemory inputs c.size) : CircuitMemory inputs c.size :=
  writeCircuitMemory state (circuitNodeSlot i)
    (circuitGateMemoryValue c i state)

/-- The semantic step stores the current gate value.
Source: Appendix Theorem `thm: gen-ac`. -/
theorem circuitMemoryStep_current {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (i : Fin c.size)
    (state : CircuitMemory inputs c.size) :
    circuitMemoryStep c i state (circuitNodeSlot i) =
      circuitGateMemoryValue c i state := by
  simp [circuitMemoryStep, writeCircuitMemory]

/-- The semantic step preserves all other coordinates.
Source: Appendix Theorem `thm: gen-ac`. -/
theorem circuitMemoryStep_other {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (i : Fin c.size)
    (state : CircuitMemory inputs c.size)
    (q : Fin (inputs + c.size)) (h : q ≠ circuitNodeSlot i) :
    circuitMemoryStep c i state q = state q := by
  simp [circuitMemoryStep, writeCircuitMemory, h]

end Transformer.Zoology
