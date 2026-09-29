/-
# Exact simulation of small arithmetic circuits by K-constrained Coyote

Arora et al., arXiv:2312.04927v1, Appendix Theorem `thm: gen-ac`.
These are complete two-input base cases: the arithmetic DAG and the
K-constrained Coyote stack compute the same result for addition and
multiplication. The general width-sensitive K-matrix resource bound
remains open.
-/

import Transformer.Zoology.Appendix_KNetwork
import Transformer.Zoology.Appendix_CircuitEvaluation

namespace Transformer.Zoology

/-- A two-input, one-output addition circuit with inputs at nodes zero and
one and the add gate at node two. Source: Appendix `def: circuit-tuple`. -/
abbrev additionCircuit : ArithmeticCircuit 2 1 := {
  size := 3
  gate := fun i =>
    if i = 0 then .input 0 else
    if i = 1 then .input 1 else .add 0 1
  output := fun _ => 2
}

/-- The addition circuit satisfies the appendix's DAG condition.
Source: Appendix `def: circuit-tuple`. -/
theorem additionCircuit_wellFormed : additionCircuit.WellFormed := by
  intro i
  fin_cases i <;> norm_num [additionCircuit, ArithmeticCircuit.WellFormed]
  decide

/-- Its relational semantics is exactly scalar addition.
Source: Appendix `def: circuit-tuple`, add-gate semantics. -/
theorem additionCircuit_computes_iff (a b : ℝ) (y : Fin 1 → ℝ) :
    additionCircuit.Computes (fun i => if i = 0 then a else b) y ↔
      y 0 = a + b := by
  constructor
  · rintro ⟨z, hz, hy⟩
    have h₀ : z 0 = a := by
      simpa [additionCircuit, ArithmeticGate.value] using hz 0
    have h₁ : z 1 = b := by
      simpa [additionCircuit, ArithmeticGate.value] using hz 1
    have h₂ : z 2 = z 0 + z 1 := by
      simpa [additionCircuit, ArithmeticGate.value] using hz 2
    calc
      y 0 = z 2 := hy 0
      _ = z 0 + z 1 := h₂
      _ = a + b := by rw [h₀, h₁]
  · intro h
    let z : Fin 3 → ℝ := fun i =>
      if i = 0 then a else if i = 1 then b else a + b
    refine ⟨z, ?_, ?_⟩
    · intro i
      fin_cases i <;>
        simp [z, additionCircuit, ArithmeticGate.value]
    · intro o
      fin_cases o
      simpa [z] using h

/-- A two-input, one-output multiplication circuit.
Source: Appendix `def: circuit-tuple`. -/
abbrev multiplicationCircuit : ArithmeticCircuit 2 1 := {
  size := 3
  gate := fun i =>
    if i = 0 then .input 0 else
    if i = 1 then .input 1 else .multiply 0 1
  output := fun _ => 2
}

/-- The multiplication circuit also satisfies the DAG condition.
Source: Appendix `def: circuit-tuple`. -/
theorem multiplicationCircuit_wellFormed : multiplicationCircuit.WellFormed := by
  intro i
  fin_cases i <;> norm_num [multiplicationCircuit, ArithmeticCircuit.WellFormed]
  decide

/-- Its relational semantics is exactly scalar multiplication.
Source: Appendix `def: circuit-tuple`, multiply-gate semantics. -/
theorem multiplicationCircuit_computes_iff (a b : ℝ) (y : Fin 1 → ℝ) :
    multiplicationCircuit.Computes (fun i => if i = 0 then a else b) y ↔
      y 0 = a * b := by
  constructor
  · rintro ⟨z, hz, hy⟩
    have h₀ : z 0 = a := by
      simpa [multiplicationCircuit, ArithmeticGate.value] using hz 0
    have h₁ : z 1 = b := by
      simpa [multiplicationCircuit, ArithmeticGate.value] using hz 1
    have h₂ : z 2 = z 0 * z 1 := by
      simpa [multiplicationCircuit, ArithmeticGate.value] using hz 2
    calc
      y 0 = z 2 := hy 0
      _ = z 0 * z 1 := h₂
      _ = a * b := by rw [h₀, h₁]
  · intro h
    let z : Fin 3 → ℝ := fun i =>
      if i = 0 then a else if i = 1 then b else a * b
    refine ⟨z, ?_, ?_⟩
    · intro i
      fin_cases i <;>
        simp [z, multiplicationCircuit, ArithmeticGate.value]
    · intro o
      fin_cases o
      simpa [z] using h

/-- One K-constrained Coyote layer implementing the addition circuit.
Source: Appendix Theorem `thm: gen-ac`, addition base case. -/
def additionKNetwork : KCoyoteNetwork 1 1 0 :=
  liftTwoFeatureNetwork {layers := [additionGateParameters]}

/-- One K-constrained Coyote layer implementing the multiplication circuit.
Source: Appendix Theorem `thm: gen-ac`, multiplication base case. -/
def multiplicationKNetwork : KCoyoteNetwork 1 1 0 :=
  liftTwoFeatureNetwork {layers := [multiplicationGateParameters]}

/-- The addition DAG and its K-constrained Coyote realization have equal
outputs on every pair of real inputs. Source: Appendix Theorem
`thm: gen-ac`, verified base case. -/
theorem additionCircuit_equiv_KCoyote (a b : ℝ) (y : Fin 1 → ℝ) :
    additionCircuit.Computes (fun i => if i = 0 then a else b) y ↔
      y 0 = additionKNetwork.run (gateInput a b) 0 ⟨0, by decide⟩ := by
  rw [additionCircuit_computes_iff]
  change y 0 = a + b ↔
    y 0 = coyoteLayer
      (twoFeatureKParameters additionGateParameters).toParameters
      (gateInput a b) 0 ⟨0, by decide⟩
  rw [kCoyote_addition_gate]

/-- The multiplication DAG and its K-constrained Coyote realization have
equal outputs on every pair of real inputs. Source: Appendix Theorem
`thm: gen-ac`, verified base case. -/
theorem multiplicationCircuit_equiv_KCoyote (a b : ℝ) (y : Fin 1 → ℝ) :
    multiplicationCircuit.Computes (fun i => if i = 0 then a else b) y ↔
      y 0 = multiplicationKNetwork.run (gateInput a b) 0 ⟨0, by decide⟩ := by
  rw [multiplicationCircuit_computes_iff]
  change y 0 = a * b ↔
    y 0 = coyoteLayer
      (twoFeatureKParameters multiplicationGateParameters).toParameters
      (gateInput a b) 0 ⟨0, by decide⟩
  rw [kCoyote_multiplication_gate]

end Transformer.Zoology
