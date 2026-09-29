/-
# Simulation of an arbitrary arithmetic DAG by a Coyote network

Arora et al., arXiv:2312.04927v1, §4 Theorem `thm:equiv` and Appendix
Theorem `thm: gen-ac`. Compiling all gates in topological order gives exact
functional equivalence with at most two ordinary Coyote layers per gate
and feature width `inputs + size`. The sharper depth/width/K-matrix and
polylogarithmic resource bounds remain separate.
-/

import Transformer.Zoology.Appendix_CircuitPrefix

namespace Transformer.Zoology

/-- The complete compiled Coyote network for an arbitrary well-formed
circuit. Source: Appendix Theorem `thm: gen-ac`, ordinary Coyote
functional construction. -/
def compileCircuit {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) :
    CoyoteNetwork 1 (inputs + c.size) :=
  { layers := compiledCircuitPrefix c c.size }

/-- Every designated output of the compiled network equals the circuit's
own recursively evaluated output. Source: Appendix Theorem `thm: gen-ac`,
functional equivalence at the output gates. -/
theorem compileCircuit_output {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (hwell : c.WellFormed)
    (x : Fin inputs → ℝ) (o : Fin outputs) :
    (compileCircuit c).run
        (memoryAsSequence (initialCircuitMemory x))
        0 (circuitNodeSlot (c.output o)) =
      c.evalNode hwell x (c.output o) := by
  rw [CoyoteNetwork.run, compileCircuit, compiledCircuitPrefix_correct c hwell x]
  exact circuitMemoryPrefix_node c hwell x c.size (c.output o)
    (c.output o).isLt

/-- The compiled network and the arithmetic circuit compute exactly the
same output relation, for every input and output vector. Source: §4 Theorem
`thm:equiv` and Appendix Theorem `thm: gen-ac`, functional component with
the weaker explicit width and depth bounds stated in this module. -/
theorem compileCircuit_equiv {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (hwell : c.WellFormed)
    (x : Fin inputs → ℝ) (y : Fin outputs → ℝ) :
    c.Computes x y ↔
      ∀ o : Fin outputs,
        y o = (compileCircuit c).run
          (memoryAsSequence (initialCircuitMemory x))
          0 (circuitNodeSlot (c.output o)) := by
  rw [c.computes_iff_eval hwell x y]
  constructor
  · intro h o
    rw [h, compileCircuit_output c hwell x o]
  · intro h
    funext o
    rw [h o, compileCircuit_output c hwell x o]

/-- At most two Coyote layers are used for each circuit gate.
Source: Appendix Theorem `thm: gen-ac`, weaker sequential compilation
bound `2 * size`; the paper states a depth-sensitive polylogarithmic bound. -/
theorem compiledCircuitPrefix_layer_bound {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (k : ℕ) :
    (compiledCircuitPrefix c k).length ≤ 2 * k := by
  induction k with
  | zero => simp [compiledCircuitPrefix]
  | succ k ih =>
      by_cases hk : k < c.size
      · simp only [compiledCircuitPrefix, dite_eq_left hk, List.length_append]
        have hg := circuitGateLayers_length_le_two c ⟨k, hk⟩
        omega
      · simpa [compiledCircuitPrefix, hk] using
          (Nat.le_trans ih (by omega : 2 * k ≤ 2 * (k + 1)))

/-- The complete compiled network has at most twice as many layers as the
circuit has gates. Source: Appendix Theorem `thm: gen-ac`, explicit weaker
bound for the sequential gate construction. -/
theorem compileCircuit_layer_bound {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) :
    (compileCircuit c).layerCount ≤ 2 * c.size := by
  exact compiledCircuitPrefix_layer_bound c c.size

/-- Stored scalar count for an ordinary dense Coyote stack: each layer
stores a `d × d` feature matrix and three `n × d` arrays. Source: §4
equation `eq: coyote-recursion`, dense parameter convention. -/
def CoyoteNetwork.denseParameterCount {n d : ℕ}
    (net : CoyoteNetwork n d) : ℕ :=
  net.layerCount * (d * d + 3 * n * d)

/-- The direct gate-by-gate compiler has an explicit dense parameter
bound. This is weaker than the paper's K-matrix and width-sensitive bound;
it demonstrates the exact resource cost of the construction proved here.
Source: §4 Theorem `thm:equiv`, ordinary Coyote specialization. -/
theorem compileCircuit_denseParameter_bound {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) :
    (compileCircuit c).denseParameterCount ≤
      2 * c.size *
        ((inputs + c.size) * (inputs + c.size) +
          3 * (inputs + c.size)) := by
  unfold CoyoteNetwork.denseParameterCount
  simpa using Nat.mul_le_mul_right
    ((inputs + c.size) * (inputs + c.size) +
      3 * (inputs + c.size)) (compileCircuit_layer_bound c)

/-- A nontrivial well-formed circuit satisfies the general simulation
hypothesis and computes a genuine input-dependent output. -/
example : identityCircuit.WellFormed ∧
    (compileCircuit identityCircuit).run
      (memoryAsSequence (initialCircuitMemory (fun _ => 7)))
      0 (circuitNodeSlot (identityCircuit.output 0)) = 7 := by
  constructor
  · intro i
    fin_cases i
    simp [identityCircuit]
  · have hwell : identityCircuit.WellFormed := by
      intro i
      fin_cases i
      simp [identityCircuit]
    have hc : identityCircuit.Computes (fun _ => 7) (fun _ => 7) :=
      (identityCircuit_computes _ _).mpr rfl
    have h := (compileCircuit_equiv identityCircuit hwell _ _).mp hc 0
    simpa [identityCircuit] using h.symm

end Transformer.Zoology
