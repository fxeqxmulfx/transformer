/-
# Elementary arithmetic gates as Coyote layers

Arora et al., arXiv:2312.04927v1, §4 Theorem `thm: equiv`, and Appendix
Theorem `thm: gen-ac`. These are elementary base cases of circuit simulation:
one length-one Coyote layer computes an addition, a multiplication, or a
constant gate in its first output feature. They do not imply the paper's
uniform depth and parameter bounds for arbitrary circuits.
-/

import Transformer.Zoology.Appendix_CircuitDefs
import Transformer.Zoology.Appendix_Shift

namespace Transformer.Zoology

/-- Two scalar inputs stored as one row with two feature coordinates.
Source: Appendix Theorem `thm: gen-ac`, gate inputs. -/
def gateInput (a b : ℝ) : RealSequence 1 2 :=
  fun _ q => if q = 0 then a else b

/-- Parameters for an addition gate in the first feature coordinate.
Source: Appendix Theorem `thm: gen-ac`, linear-gate case. -/
def additionGateParameters : CoyoteParameters 1 2 := {
  weight := fun _ q => if q = 0 then 1 else 0
  filter := fun _ _ => 0
  bias₁ := fun _ _ => 0
  bias₂ := fun _ _ => 1
}

/-- The Coyote addition gate returns the sum of both scalar inputs.
Source: Appendix Theorem `thm: gen-ac`, linear-gate case. -/
theorem coyote_addition_gate (a b : ℝ) :
    coyoteLayer additionGateParameters (gateInput a b) 0 0 = a + b := by
  simp [coyoteLayer, additionGateParameters, gateInput,
    linearProjection, causalConvolution, Fin.sum_univ_two]

/-- Parameters for a multiplication gate: the linear branch reads the
second feature while the identity convolution reads the first.
Source: Appendix Theorem `thm: gen-ac`, multiplication-gate case. -/
def multiplicationGateParameters : CoyoteParameters 1 2 := {
  weight := fun k q => if k = 1 ∧ q = 0 then 1 else 0
  filter := impulse 0
  bias₁ := fun _ _ => 0
  bias₂ := fun _ _ => 0
}

/-- The Coyote multiplication gate returns the product of both scalar
inputs. Source: Appendix Theorem `thm: gen-ac`, multiplication-gate case. -/
theorem coyote_multiplication_gate (a b : ℝ) :
    coyoteLayer multiplicationGateParameters (gateInput a b) 0 0 = a * b := by
  simp [coyoteLayer, multiplicationGateParameters, gateInput,
    linearProjection, causalConvolution_impulse, shiftDown_zero]
  ring

/-- Parameters for a constant gate in the first feature coordinate.
Source: Appendix `def: circuit-tuple`, constant gates, and Theorem
`thm: gen-ac`. -/
def constantGateParameters (r : ℝ) : CoyoteParameters 1 2 := {
  weight := fun _ _ => 0
  filter := fun _ _ => 0
  bias₁ := fun _ q => if q = 0 then r else 0
  bias₂ := fun _ _ => 1
}

/-- A Coyote layer can emit any fixed scalar constant, independently of
its input. Source: Appendix Theorem `thm: gen-ac`, constant-gate case. -/
theorem coyote_constant_gate (a b r : ℝ) :
    coyoteLayer (constantGateParameters r) (gateInput a b) 0 0 = r := by
  simp [coyoteLayer, constantGateParameters, gateInput,
    linearProjection, causalConvolution]

end Transformer.Zoology
