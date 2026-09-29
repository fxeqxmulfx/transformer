/-
# Depth-sensitive simulation of an arithmetic DAG by Coyote

Arora et al., arXiv:2312.04927v1, §4 Theorem `thm:equiv` and Appendix
Theorem `thm: gen-ac`. Parallel gate updates give exact functional
equivalence in two ordinary Coyote layers per circuit level, with circuit
inputs packed into one token's feature coordinates. The feature width is
`inputs + size`, and the dense parameter count is quadratic in that width
per layer. The paper's original sequence layout and sharper width-sensitive
K-matrix parameter bound are not established by this construction.
-/

import Transformer.Zoology.Appendix_CircuitParallelPrefix
import Transformer.Zoology.Appendix_SmallCircuits

namespace Transformer.Zoology

/-- A Coyote network that executes every gate on a given level assignment
below `depth`. Source: Appendix Theorem `thm: gen-ac`, parallel stages. -/
def compileCircuitByLevels {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) (depth : ℕ) :
    CoyoteNetwork 1 (inputs + c.size) :=
  { layers := compiledCircuitLevelPrefix c level depth }

/-- Every designated output agrees with the arithmetic circuit whenever
all gates lie below `depth` and the level order is topological.
Source: Appendix Theorem `thm: gen-ac`, functional equivalence. -/
theorem compileCircuitByLevels_output {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (hwell : c.WellFormed)
    (level : Fin c.size → ℕ) (horder : CircuitLevelOrder c level)
    (depth : ℕ) (hbound : ∀ i, level i < depth)
    (x : Fin inputs → ℝ) (o : Fin outputs) :
    (compileCircuitByLevels c level depth).run
        (memoryAsSequence (initialCircuitMemory x))
        0 (circuitNodeSlot (c.output o)) =
      c.evalNode hwell x (c.output o) := by
  rw [CoyoteNetwork.run, compileCircuitByLevels,
    compiledCircuitLevelPrefix_correct c level horder x depth]
  exact circuitLevelMemoryPrefix_node c hwell level horder x depth
    (c.output o) (hbound (c.output o))

/-- The parallel Coyote network computes exactly the same output relation
as the circuit for every input vector. Source: §4 Theorem `thm:equiv`,
functional and depth-sensitive component. -/
theorem compileCircuitByLevels_equiv {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (hwell : c.WellFormed)
    (level : Fin c.size → ℕ) (horder : CircuitLevelOrder c level)
    (depth : ℕ) (hbound : ∀ i, level i < depth)
    (x : Fin inputs → ℝ) (y : Fin outputs → ℝ) :
    c.Computes x y ↔
      ∀ o : Fin outputs,
        y o = (compileCircuitByLevels c level depth).run
          (memoryAsSequence (initialCircuitMemory x))
          0 (circuitNodeSlot (c.output o)) := by
  rw [c.computes_iff_eval hwell x y]
  constructor
  · intro h o
    rw [h, compileCircuitByLevels_output c hwell level horder depth hbound x o]
  · intro h
    funext o
    rw [h o, compileCircuitByLevels_output c hwell level horder depth hbound x o]

/-- The parallel compiler uses exactly two ordinary Coyote layers per
circuit level. Source: Appendix Theorem `thm: gen-ac`, level count. -/
theorem compileCircuitByLevels_layerCount {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) (depth : ℕ) :
    (compileCircuitByLevels c level depth).layerCount = 2 * depth := by
  exact compiledCircuitLevelPrefix_length c level depth

/-- Exact dense scalar storage used by the depth-sensitive construction.
Source: §4 equation `eq: coyote-recursion`; this direct dense count is
larger than the paper's claimed K-matrix asymptotic bound. -/
theorem compileCircuitByLevels_denseParameterCount {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs)
    (level : Fin c.size → ℕ) (depth : ℕ) :
    (compileCircuitByLevels c level depth).denseParameterCount =
      2 * depth *
        ((inputs + c.size) * (inputs + c.size) +
          3 * (inputs + c.size)) := by
  simp [CoyoteNetwork.denseParameterCount,
    compileCircuitByLevels_layerCount]

/-- The paper's depth-and-width condition implies ordinary circuit
well-formedness. Source: Appendix `def: circuit-tuple`. -/
theorem ArithmeticCircuit.HasDepthWidth.wellFormed {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (depth width : ℕ)
    (h : c.HasDepthWidth depth width) : c.WellFormed := by
  obtain ⟨level, hbound, hedges, hwidth⟩ := h
  intro i
  have hi := hedges i
  cases hg : c.gate i <;> simp_all

/-- The paper's depth-and-width condition supplies the topological level
order needed for parallel gate updates. Source: Appendix `def: circuit-tuple`. -/
theorem ArithmeticCircuit.HasDepthWidth.levelOrder {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (depth width : ℕ)
    (h : c.HasDepthWidth depth width) :
    ∃ level : Fin c.size → ℕ,
      (∀ i, level i < depth) ∧ CircuitLevelOrder c level := by
  obtain ⟨level, hbound, hedges, hwidth⟩ := h
  refine ⟨level, hbound, ?_⟩
  intro i
  have hi := hedges i
  cases hg : c.gate i <;> simp_all

/-- For any arithmetic circuit with a depth/width witness, one ordinary
Coyote network computes exactly the same input-output relation on a
one-token feature-memory encoding, using `2 * depth` layers and width
`inputs + size`. The original `N × d` input layout and stronger K-matrix
parameter bound remain separate.
Source: §4 Theorem `thm:equiv` and Appendix Theorem `thm: gen-ac`. -/
theorem exists_parallel_coyote_simulation {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (depth width : ℕ)
    (hres : c.HasDepthWidth depth width) :
    ∃ net : CoyoteNetwork 1 (inputs + c.size),
      net.layerCount = 2 * depth ∧
        ∀ x : Fin inputs → ℝ, ∀ y : Fin outputs → ℝ,
          c.Computes x y ↔
            ∀ o : Fin outputs,
              y o = net.run (memoryAsSequence (initialCircuitMemory x))
                0 (circuitNodeSlot (c.output o)) := by
  obtain ⟨level, hbound, horder⟩ := hres.levelOrder c depth width
  have hwell : c.WellFormed := hres.wellFormed c depth width
  refine ⟨compileCircuitByLevels c level depth,
    compileCircuitByLevels_layerCount c level depth, ?_⟩
  intro x y
  exact compileCircuitByLevels_equiv c hwell level horder depth hbound x y

/-- The depth/width hypothesis is satisfiable for a nonconstant circuit.
Source: Appendix `def: circuit-tuple`, identity-gate example. -/
example : identityCircuit.HasDepthWidth 1 1 := by
  refine ⟨fun _ => 0, ?_, ?_, ?_⟩
  · intro i
    norm_num
  · intro i
    fin_cases i
    simp [identityCircuit]
  · intro ℓ
    change (Finset.univ.filter (fun _ : Fin 1 => 0 = ℓ)).card ≤ 1
    have hsub : (Finset.univ.filter (fun _ : Fin 1 => 0 = ℓ)).card ≤
        (Finset.univ : Finset (Fin 1)).card :=
      Finset.card_le_card (Finset.filter_subset _ _)
    simpa using hsub

/-- A valid level assignment for the two-input multiplication DAG: both
input gates lie on level zero and the product gate lies on level one.
Source: Appendix Theorem `thm: gen-ac`, multiplication stage. -/
def multiplicationCircuitLevel (i : Fin 3) : ℕ :=
  if i = 2 then 1 else 0

/-- The depth-sensitive compiler handles a nontrivial multiplication gate
and returns the correct product for every pair of real inputs.
Source: Appendix Theorem `thm: gen-ac`, concrete multiplication instance. -/
theorem multiplicationCircuit_parallel_output (a b : ℝ) :
    (compileCircuitByLevels multiplicationCircuit
        multiplicationCircuitLevel 2).run
        (memoryAsSequence
          (initialCircuitMemory (fun i : Fin 2 => if i = 0 then a else b)))
        0 (circuitNodeSlot (multiplicationCircuit.output 0)) = a * b := by
  have horder : CircuitLevelOrder multiplicationCircuit
      multiplicationCircuitLevel := by
    intro i
    fin_cases i <;>
      norm_num [multiplicationCircuit, multiplicationCircuitLevel]
    all_goals decide
  have hbound : ∀ i : Fin multiplicationCircuit.size,
      multiplicationCircuitLevel i < 2 := by
    intro i
    fin_cases i <;> decide
  have hc : multiplicationCircuit.Computes
      (fun i : Fin 2 => if i = 0 then a else b)
      (fun _ : Fin 1 => a * b) :=
    (multiplicationCircuit_computes_iff a b _).mpr rfl
  have h := (compileCircuitByLevels_equiv multiplicationCircuit
    multiplicationCircuit_wellFormed multiplicationCircuitLevel
    horder 2 hbound _ _).mp hc 0
  simpa using h.symm

end Transformer.Zoology
