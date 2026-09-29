/-
# Total evaluation of well-formed arithmetic circuits

Arora et al., arXiv:2312.04927v1, Appendix `def: circuit-tuple` and
Theorem `thm: gen-ac`. A topologically ordered arithmetic circuit evaluates
every gate by well-founded recursion on its index. Together with
`ArithmeticCircuit.computes_unique`, this makes its semantics functional.
-/

import Transformer.Zoology.Appendix_CircuitSemantics

namespace Transformer.Zoology

/-- Evaluate a gate by recursion on its topological index.
Source: Appendix `def: circuit-tuple`, directed acyclic circuit semantics. -/
noncomputable def ArithmeticCircuit.evalNode {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (hwell : c.WellFormed)
    (x : Fin inputs → ℝ) : Fin c.size → ℝ :=
  (measure (fun i : Fin c.size => i.val)).wf.fix fun i rec =>
    match hgate : c.gate i with
    | .input a => x a
    | .constant r => r
    | .add a b =>
        have hab : a < i ∧ b < i := by
          simpa [ArithmeticCircuit.WellFormed, hgate] using hwell i
        rec a hab.1 + rec b hab.2
    | .multiply a b =>
        have hab : a < i ∧ b < i := by
          simpa [ArithmeticCircuit.WellFormed, hgate] using hwell i
        rec a hab.1 * rec b hab.2

/-- The recursive gate values satisfy every gate equation.
Source: Appendix `def: circuit-tuple`, evaluation semantics. -/
theorem ArithmeticCircuit.evalNode_equation {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (hwell : c.WellFormed)
    (x : Fin inputs → ℝ) (i : Fin c.size) :
    c.evalNode hwell x i = (c.gate i).value x (c.evalNode hwell x) := by
  unfold ArithmeticCircuit.evalNode
  rw [WellFounded.fix_eq]
  split <;> simp [ArithmeticGate.value, *]

/-- A well-formed circuit always computes its recursively evaluated output.
Source: Appendix `def: circuit-tuple`, total DAG semantics. -/
theorem ArithmeticCircuit.computes_eval {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (hwell : c.WellFormed)
    (x : Fin inputs → ℝ) :
    c.Computes x (fun o => c.evalNode hwell x (c.output o)) := by
  refine ⟨c.evalNode hwell x, ?_, ?_⟩
  · intro i
    exact c.evalNode_equation hwell x i
  · intro o
    rfl

/-- The circuit's relational and recursive semantics coincide exactly.
Source: Appendix `def: circuit-tuple`, topologically ordered evaluation. -/
theorem ArithmeticCircuit.computes_iff_eval {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (hwell : c.WellFormed)
    (x : Fin inputs → ℝ) (y : Fin outputs → ℝ) :
    c.Computes x y ↔
      y = fun o => c.evalNode hwell x (c.output o) := by
  constructor
  · intro h
    exact c.computes_unique hwell x y _ h (c.computes_eval hwell x)
  · intro h
    subst y
    exact c.computes_eval hwell x

end Transformer.Zoology
