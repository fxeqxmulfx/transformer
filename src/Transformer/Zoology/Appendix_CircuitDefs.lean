/-
# Arithmetic-circuit syntax and resource bounds

Arora et al., arXiv:2312.04927v1, Appendix `def: circuit-tuple`
and `sec: arithmetic`. Nodes are numbered in topological order; a gate's
predecessors must have smaller numbers. Size is the number of gates and
depth is witnessed by gate levels. The level-cardinality predicate below
records parallel gate count; the paper also describes width as wires
crossing a cut, which requires a separate liveness condition.
-/

import Mathlib

namespace Transformer.Zoology

/-- An arithmetic-circuit gate.  A predecessor is an index into the same
finite gate array; `ArithmeticCircuit.WellFormed` requires it to be earlier.
Source: Appendix `def: circuit-tuple`. -/
inductive ArithmeticGate (inputs nodes : ℕ) where
  | input (i : Fin inputs)
  | constant (r : ℝ)
  | add (left right : Fin nodes)
  | multiply (left right : Fin nodes)

/-- A directed arithmetic circuit with designated output gates.
Source: Appendix `def: circuit-tuple`. -/
structure ArithmeticCircuit (inputs outputs : ℕ) where
  size : ℕ
  gate : Fin size → ArithmeticGate inputs size
  output : Fin outputs → Fin size

/-- A gate's value for an input and an assignment to all circuit nodes.
Source: Appendix `def: circuit-tuple`. -/
def ArithmeticGate.value {inputs nodes : ℕ}
    (g : ArithmeticGate inputs nodes) (x : Fin inputs → ℝ)
    (z : Fin nodes → ℝ) : ℝ :=
  match g with
  | .input i => x i
  | .constant r => r
  | .add a b => z a + z b
  | .multiply a b => z a * z b

/-- The circuit computes an output when its gates can be evaluated
consistently in their topological order.  Source: Appendix `def: circuit-tuple`.
This relation is used only with `WellFormed` circuits in substantive results. -/
def ArithmeticCircuit.Computes {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (x : Fin inputs → ℝ) (y : Fin outputs → ℝ) : Prop :=
  ∃ z : Fin c.size → ℝ,
    (∀ i, z i = (c.gate i).value x z) ∧
      ∀ o, y o = z (c.output o)

/-- A circuit has only edges from earlier to later gates.
Source: Appendix `def: circuit-tuple`, directed acyclic graph condition. -/
def ArithmeticCircuit.WellFormed {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) : Prop :=
  ∀ i,
    match c.gate i with
    | .input _ | .constant _ => True
    | .add a b | .multiply a b => a < i ∧ b < i

/-- A linear arithmetic circuit permits multiplication only when at least
one operand is a gate holding a fixed scalar constant.  Source: Appendix
`sec: linear_circuit`, Definition `Linear Arithmetic Circuit`. -/
def ArithmeticCircuit.IsLinear {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) : Prop :=
  ∀ i,
    match c.gate i with
    | .multiply a b =>
        (∃ r, c.gate a = .constant r) ∨
        (∃ r, c.gate b = .constant r)
    | _ => True

/-- A common level assignment witnesses depth and bounds the number of
gates on each level. This formalizes the paper's “parallel operations”
reading of width; its “wires intersected by a cut” reading also counts
values retained for later levels and is stronger than this predicate.
Source: Appendix `def: circuit-tuple`. -/
def ArithmeticCircuit.HasDepthWidth {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (depth width : ℕ) : Prop :=
  ∃ level : Fin c.size → ℕ,
    (∀ i, level i < depth) ∧
    (∀ i,
      match c.gate i with
      | .input _ | .constant _ => True
      | .add a b | .multiply a b =>
          a < i ∧ b < i ∧ level a < level i ∧ level b < level i) ∧
    ∀ l, (Finset.univ.filter fun i => level i = l).card ≤ width

/-- Input count `n`, at most `s` gates, depth strictly below `Δ`, and at
most `w` gates on each level. This is the parallel-operation portion of
the appendix's `(n,s,Δ,w)` convention; it does not assert cut-wire width.
Source: Appendix `def: circuit-tuple`. -/
def ArithmeticCircuit.HasResources {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (size depth width : ℕ) : Prop :=
  c.size ≤ size ∧ c.HasDepthWidth depth width

/-- The one-input, one-output identity circuit is a concrete example of
the appendix's circuit model.  Source: Appendix `def: circuit-tuple`. -/
def identityCircuit : ArithmeticCircuit 1 1 := {
  size := 1
  gate := fun _ => .input 0
  output := fun _ => 0
}

/-- The identity circuit computes exactly the input value.
Source: Appendix `def: circuit-tuple`, circuit semantics. -/
theorem identityCircuit_computes (x y : Fin 1 → ℝ) :
    identityCircuit.Computes x y ↔ y 0 = x 0 := by
  change (∃ z : Fin 1 → ℝ,
    (∀ i, z i = x 0) ∧ ∀ o, y o = z 0) ↔ y 0 = x 0
  constructor
  · rintro ⟨z, hz, hy⟩
    calc y 0 = z 0 := hy 0
      _ = x 0 := hz 0
  · intro h
    refine ⟨x, ?_, ?_⟩
    · intro i
      fin_cases i
      rfl
    · intro o
      fin_cases o
      exact h

/-- The identity circuit satisfies the acyclicity, depth and width
requirements with one gate. -/
example : identityCircuit.WellFormed ∧ identityCircuit.HasResources 1 1 1 := by
  constructor
  · intro i
    fin_cases i
    simp [identityCircuit]
  · constructor
    · simp [identityCircuit]
    · refine ⟨fun _ => 0, ?_, ?_, ?_⟩
      · intro i
        norm_num
      · intro i
        fin_cases i
        simp [identityCircuit]
      · intro l
        change (Finset.univ.filter (fun _ : Fin 1 => 0 = l)).card ≤ 1
        have hsub : (Finset.univ.filter (fun _ : Fin 1 => 0 = l)).card ≤
            (Finset.univ : Finset (Fin 1)).card :=
          Finset.card_le_card (Finset.filter_subset _ _)
        simpa using hsub

/-- The identity circuit is a linear arithmetic circuit. -/
example : identityCircuit.IsLinear := by
  intro i
  fin_cases i
  simp [identityCircuit]

end Transformer.Zoology
