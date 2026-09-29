/-
# Coyote layers for an arbitrary arithmetic gate with persistent memory

Arora et al., arXiv:2312.04927v1, Appendix Theorem `thm: gen-ac`,
addition and multiplication stages. A shared feature matrix writes an
input, constant, or sum into a fresh node slot. For multiplication, one
linear layer first copies the right operand into that slot, and a second
gated layer multiplies it by the left operand. Every other slot is retained.
-/

import Transformer.Zoology.Appendix_CircuitMemory

open scoped BigOperators

namespace Transformer.Zoology

/-- View a feature memory as a one-token sequence.
Source: Appendix Theorem `thm: gen-ac`, the `n=1` specialization. -/
def memoryAsSequence {inputs nodes : ℕ}
    (state : CircuitMemory inputs nodes) : RealSequence 1 (inputs + nodes) :=
  fun _ q => state q

/-- A Coyote layer that preserves every feature except `target`, whose
new value is an affine form of the old memory.
Source: Appendix Theorem `thm: gen-ac`, linear-gate stage. -/
def writeLinearGateParameters {d : ℕ} (target : Fin d)
    (coeff : Fin d → ℝ) (bias : ℝ) : CoyoteParameters 1 d := {
  weight := fun k q => if q = target then coeff k else if k = q then 1 else 0
  filter := fun _ _ => 0
  bias₁ := fun _ q => if q = target then bias else 0
  bias₂ := fun _ _ => 1
}

/-- A single Coyote layer performs an affine write and preserves all other
features. Source: Appendix Theorem `thm: gen-ac`, linear-gate stage. -/
theorem writeLinearGateParameters_apply {inputs nodes : ℕ}
    (state : CircuitMemory inputs nodes)
    (target : Fin (inputs + nodes))
    (coeff : Fin (inputs + nodes) → ℝ) (bias : ℝ) :
    coyoteLayer (writeLinearGateParameters target coeff bias)
        (memoryAsSequence state) =
      memoryAsSequence (writeCircuitMemory state target
        ((∑ k, state k * coeff k) + bias)) := by
  funext p q
  fin_cases p
  by_cases hq : q = target
  · subst q
    simp [coyoteLayer, writeLinearGateParameters, memoryAsSequence,
      writeCircuitMemory, linearProjection, causalConvolution]
  · simp [coyoteLayer, writeLinearGateParameters, memoryAsSequence,
      writeCircuitMemory, linearProjection, causalConvolution, hq]

/-- The indicator coefficient selecting one feature.
Source: Appendix Theorem `thm: gen-ac`, wire selection. -/
def circuitCoordinateCoeff {d : ℕ} (source : Fin d) : Fin d → ℝ :=
  fun k => if k = source then 1 else 0

/-- A one-hot coefficient vector reads the chosen feature exactly.
Source: Appendix Theorem `thm: gen-ac`, wire selection. -/
theorem circuitCoordinateCoeff_sum {d : ℕ}
    (state : Fin d → ℝ) (source : Fin d) :
    (∑ k, state k * circuitCoordinateCoeff source k) = state source := by
  simp [circuitCoordinateCoeff]

/-- A Coyote layer that multiplies the value read from `left` by the
value already stored at `target`, preserving all other feature coordinates.
Source: Appendix Theorem `thm: gen-ac`, multiplication stage. -/
def multiplyGateParameters {d : ℕ} (target left : Fin d) :
    CoyoteParameters 1 d := {
  weight := fun k q =>
    if q = target then circuitCoordinateCoeff left k
    else if k = q then 1 else 0
  filter := fun _ q => if q = target then 1 else 0
  bias₁ := fun _ _ => 0
  bias₂ := fun _ q => if q = target then 0 else 1
}

/-- The multiplicative Coyote layer performs exactly that one-coordinate
product. Source: Appendix Theorem `thm: gen-ac`, multiplication stage. -/
theorem multiplyGateParameters_apply {inputs nodes : ℕ}
    (state : CircuitMemory inputs nodes)
    (target left : Fin (inputs + nodes)) :
    coyoteLayer (multiplyGateParameters target left)
        (memoryAsSequence state) =
      memoryAsSequence (writeCircuitMemory state target
        (state left * state target)) := by
  funext p q
  fin_cases p
  by_cases hq : q = target
  · subst q
    simp [coyoteLayer, multiplyGateParameters, memoryAsSequence,
      writeCircuitMemory, linearProjection, causalConvolution,
      circuitCoordinateCoeff_sum]
  · simp [coyoteLayer, multiplyGateParameters, memoryAsSequence,
      writeCircuitMemory, linearProjection, causalConvolution, hq]

/-- The affine expression used for a gate's linear stage. At a
multiplication gate it copies the right operand into the fresh target slot.
Source: Appendix Theorem `thm: gen-ac`, gate stages. -/
def circuitGateLinearCoeff {inputs nodes : ℕ}
    (g : ArithmeticGate inputs nodes) : Fin (inputs + nodes) → ℝ :=
  match g with
  | .input a => circuitCoordinateCoeff (circuitInputSlot a)
  | .constant _ => fun _ => 0
  | .add a b => fun k =>
      circuitCoordinateCoeff (circuitNodeSlot a) k +
        circuitCoordinateCoeff (circuitNodeSlot b) k
  | .multiply _ b => circuitCoordinateCoeff (circuitNodeSlot b)

/-- Constant offset of a gate's affine stage.
Source: Appendix Theorem `thm: gen-ac`, constant gates. -/
def circuitGateLinearBias {inputs nodes : ℕ}
    (g : ArithmeticGate inputs nodes) : ℝ :=
  match g with
  | .constant r => r
  | _ => 0

/-- The affine stage computes the gate value for all nonmultiplicative
gates, and copies the right operand for a multiplication gate.
Source: Appendix Theorem `thm: gen-ac`, gate stages. -/
theorem circuitGateLinear_value {inputs nodes : ℕ}
    (g : ArithmeticGate inputs nodes)
    (state : CircuitMemory inputs nodes) :
    (∑ k, state k * circuitGateLinearCoeff g k) +
        circuitGateLinearBias g =
      match g with
      | .input a => state (circuitInputSlot a)
      | .constant r => r
      | .add a b => state (circuitNodeSlot a) +
          state (circuitNodeSlot b)
      | .multiply _ b => state (circuitNodeSlot b) := by
  cases g with
  | input a => simp [circuitGateLinearCoeff, circuitGateLinearBias,
      circuitCoordinateCoeff_sum]
  | constant r => simp [circuitGateLinearCoeff, circuitGateLinearBias]
  | add a b =>
      simp [circuitGateLinearCoeff, circuitGateLinearBias,
        mul_add, Finset.sum_add_distrib, circuitCoordinateCoeff_sum]
  | multiply a b => simp [circuitGateLinearCoeff, circuitGateLinearBias,
      circuitCoordinateCoeff_sum]

/-- One or two Coyote layers implementing one topologically valid circuit
gate while retaining all earlier feature values. Source: Appendix Theorem
`thm: gen-ac`, layerwise simulation. -/
def circuitGateLayers {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (i : Fin c.size) :
    List (CoyoteParameters 1 (inputs + c.size)) :=
  let target := circuitNodeSlot (inputs := inputs) i
  let first := writeLinearGateParameters target
    (circuitGateLinearCoeff (c.gate i))
    (circuitGateLinearBias (c.gate i))
  match c.gate i with
  | .multiply a _ => [first, multiplyGateParameters target (circuitNodeSlot a)]
  | _ => [first]

end Transformer.Zoology
