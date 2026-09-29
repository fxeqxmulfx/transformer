/-
# Gate count and live-wire width are different

Arora et al., arXiv:2312.04927v1, Appendix `def: circuit-tuple` and
Theorem `thm: gen-ac`. The paper describes width as parallel operations
or wires crossing a horizontal cut, and later retains earlier outputs
needed by future levels. The example below has one gate per level but two
live gate values after its second level. It explains why a width-sensitive
simulation needs a liveness condition in addition to level cardinality.
-/

import Transformer.Zoology.Appendix_CircuitParallel

namespace Transformer.Zoology

/-- A later gate directly reads an earlier node. Source: Appendix
`def: circuit-tuple`, arithmetic-circuit edges. -/
def ArithmeticCircuit.UsesNode {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (source target : Fin c.size) : Bool :=
  match c.gate target with
  | .add a b | .multiply a b => decide (source = a) || decide (source = b)
  | .input _ | .constant _ => false

/-- A computed node remains live after level `cut` if a later gate reads
it or it is a designated output. Source: Appendix Theorem `thm: gen-ac`,
values preserved for later layers. -/
def ArithmeticCircuit.LiveAfter {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) (cut : ℕ) (source : Fin c.size) : Bool :=
  decide (level source ≤ cut) &&
    ((List.finRange c.size).any (fun target =>
        decide (cut < level target) && c.UsesNode source target) ||
      (List.finRange outputs).any (fun o => decide (c.output o = source)))

/-- Number of live node values immediately after a level.
Source: Appendix `def: circuit-tuple`, “wires intersected by a cut.” -/
def ArithmeticCircuit.liveWidthAfter {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) (cut : ℕ) : ℕ :=
  (Finset.univ.filter fun i => c.LiveAfter level cut i).card

/-- A four-gate DAG that must retain its first input through later
levels. Source: Appendix Theorem `thm: gen-ac`, remembering old wires. -/
def longRangeCircuit : ArithmeticCircuit 2 1 := {
  size := 4
  gate := fun i =>
    if i = 0 then .input 0 else
    if i = 1 then .input 1 else
    if i = 2 then .add 0 1 else .multiply 0 2
  output := fun _ => 3
}

/-- The circuit has one gate on each of four levels.
Source: Appendix `def: circuit-tuple`, parallel-operation count. -/
theorem longRangeCircuit_oneGatePerLevel :
    longRangeCircuit.HasDepthWidth 4 1 := by
  refine ⟨fun i => i.val, ?_, ?_, ?_⟩
  · intro i
    exact i.isLt
  · intro i
    fin_cases i <;> simp [longRangeCircuit]
  · intro ℓ
    apply Finset.card_le_one.mpr
    intro a ha b hb
    have hav : a.val = ℓ := (Finset.mem_filter.mp ha).2
    have hbv : b.val = ℓ := (Finset.mem_filter.mp hb).2
    exact Fin.ext (hav.trans hbv.symm)

/-- Despite one gate per level, two earlier values are simultaneously
live after level one: node zero is needed again by the final product,
while node one is needed by the intervening sum. This distinguishes
level cardinality from the paper's cut-wire reading of width.
Source: Appendix `def: circuit-tuple` and Theorem `thm: gen-ac`. -/
theorem longRangeCircuit_cutWidth_two :
    longRangeCircuit.liveWidthAfter (fun i => i.val) 1 = 2 := by
  decide

end Transformer.Zoology
