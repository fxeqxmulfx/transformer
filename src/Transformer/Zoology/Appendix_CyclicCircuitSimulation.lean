/-
# A shape-preserving cyclic Coyote simulation of an arithmetic circuit

Arora et al., arXiv:2312.04927v1, §4 Theorem `thm:equiv` and Appendix
Theorem `thm: gen-ac`. The external input and output both have shape
`n × d`. Two cyclic layers pack the input at row zero, two layers per
circuit level execute a persistent feature-memory compiler, and four
cyclic layers place the outputs back at their original positions.

This proves exact functional equivalence with `2*depth+6` layers and
inner width `n*d+size`. It does not prove the paper's `O(w)` inner length,
unchanged feature width, K-matrix representation, or `O(depth*log w)`
parameter bound. The causal-only variant cannot generally perform the
required backward communication between input positions.
-/

import Transformer.Zoology.Appendix_CyclicCircuitOutput

namespace Transformer.Zoology

/-- The two cyclic Coyote layers that pack the external sequence into
feature memory at row zero. Source: Appendix Theorem `thm: gen-ac`,
input-layout stage. -/
def circuitInputLayoutLayers {n d size : ℕ} [NeZero n] :
    List (CoyoteParameters n (n * d + size)) :=
  [circuitInputCopyParameters, circuitGatherParameters]

/-- The four cyclic Coyote layers that return circuit outputs to the
external sequence layout. Source: Appendix Theorem `thm: gen-ac`,
output-layout stage. -/
def circuitOutputLayoutLayers {n d : ℕ} [NeZero n]
    (c : ArithmeticCircuit (n * d) (n * d)) :
    List (CoyoteParameters n (n * d + c.size)) :=
  [circuitRowZeroMaskParameters, circuitOutputCopyParameters c,
    circuitOutputScatterParameters, circuitOutputProjectParameters]

/-- The cyclic Coyote network compiled from a levelled arithmetic
circuit. Source: Appendix Theorem `thm: gen-ac`, layer composition. -/
def compileCircuitCyclic {n d : ℕ} [NeZero n]
    (c : ArithmeticCircuit (n * d) (n * d))
    (level : Fin c.size → ℕ) (depth : ℕ) :
    CyclicCoyoteNetwork n (n * d + c.size) := {
  layers := circuitInputLayoutLayers ++
    (compiledCircuitLevelPrefix c level depth).map liftOneTokenCyclic ++
    circuitOutputLayoutLayers c
}

/-- The explicit input-layout state. Source: Appendix Theorem
`thm: gen-ac`, input rearrangement. -/
def circuitCyclicEncodedInput {n d size : ℕ} [NeZero n]
    (u : RealSequence n d) : RealSequence n (n * d + size) :=
  coyoteLayerCyclic circuitGatherParameters
    (coyoteLayerCyclic circuitInputCopyParameters (padSequence u))

/-- The state after the levelwise arithmetic-circuit compiler.
Source: Appendix Theorem `thm: gen-ac`, gate stages. -/
def circuitCyclicCompiledState {n d : ℕ} [NeZero n]
    (c : ArithmeticCircuit (n * d) (n * d))
    (level : Fin c.size → ℕ) (depth : ℕ)
    (u : RealSequence n d) : RealSequence n (n * d + c.size) :=
  ((compiledCircuitLevelPrefix c level depth).map liftOneTokenCyclic).foldl
    (fun state p => coyoteLayerCyclic p state)
    (circuitCyclicEncodedInput (size := c.size) u)

/-- The first row of the compiled cyclic state contains the exact value
of each circuit output gate. Source: Appendix Theorem `thm: gen-ac`,
functional simulation. -/
theorem circuitCyclicCompiledState_output {n d : ℕ} [NeZero n]
    (c : ArithmeticCircuit (n * d) (n * d)) (hwell : c.WellFormed)
    (level : Fin c.size → ℕ) (horder : CircuitLevelOrder c level)
    (depth : ℕ) (hbound : ∀ i, level i < depth)
    (u : RealSequence n d) (o : Fin (n * d)) :
    circuitCyclicCompiledState c level depth u 0
      (circuitNodeSlot (c.output o)) =
        c.evalNode hwell (circuitFlatInput u) (c.output o) := by
  unfold circuitCyclicCompiledState
  rw [liftOneTokenCyclic_foldl]
  have hinput : (fun _ : Fin 1 =>
      circuitCyclicEncodedInput (size := c.size) u 0) =
        memoryAsSequence (initialCircuitMemory (nodes := c.size)
          (circuitFlatInput u)) := by
    funext j q
    fin_cases j
    exact congrFun (circuitGather_rowZero (size := c.size) u) q
  rw [hinput, compiledCircuitLevelPrefix_correct c level horder]
  exact circuitLevelMemoryPrefix_node c hwell level horder
    (circuitFlatInput u) depth (c.output o) (hbound (c.output o))

/-- The output-layout stack reads the first row of the compiled state
and places the requested value at each original token and feature.
Source: Appendix Theorem `thm: gen-ac`, final wiring. -/
theorem circuitOutputLayoutLayers_correct {n d : ℕ} [NeZero n]
    (c : ArithmeticCircuit (n * d) (n * d))
    (state : RealSequence n (n * d + c.size))
    (i : Fin n) (q : Fin d) :
    (circuitOutputLayoutLayers c).foldl
      (fun u p => coyoteLayerCyclic p u) state i
      (circuitOriginalFeature (size := c.size) q) =
        state 0 (circuitNodeSlot (c.output (circuitFlatIndex i q))) := by
  simpa [circuitOutputLayoutLayers] using
    circuitOutputProject_correct c state i q

/-- The compiled cyclic Coyote network returns the circuit's value at
every original output coordinate. Source: Appendix Theorem `thm: gen-ac`,
exact functional component. -/
theorem compileCircuitCyclic_output {n d : ℕ} [NeZero n]
    (c : ArithmeticCircuit (n * d) (n * d)) (hwell : c.WellFormed)
    (level : Fin c.size → ℕ) (horder : CircuitLevelOrder c level)
    (depth : ℕ) (hbound : ∀ j, level j < depth)
    (u : RealSequence n d) (i : Fin n) (q : Fin d) :
    (compileCircuitCyclic c level depth).run (padSequence u) i
      (circuitOriginalFeature (size := c.size) q) =
        c.evalNode hwell (circuitFlatInput u)
          (c.output (circuitFlatIndex i q)) := by
  change ((circuitInputLayoutLayers ++
      (compiledCircuitLevelPrefix c level depth).map liftOneTokenCyclic ++
      circuitOutputLayoutLayers c).foldl
      (fun state p => coyoteLayerCyclic p state) (padSequence u)) i
      (circuitOriginalFeature (size := c.size) q) = _
  simp only [List.foldl_append]
  have henc : (circuitInputLayoutLayers (n := n) (d := d)
      (size := c.size)).foldl
        (fun state p => coyoteLayerCyclic p state) (padSequence u) =
        circuitCyclicEncodedInput (size := c.size) u := rfl
  rw [henc]
  change (circuitOutputLayoutLayers c).foldl
    (fun state p => coyoteLayerCyclic p state)
    (circuitCyclicCompiledState c level depth u) i
    (circuitOriginalFeature (size := c.size) q) = _
  rw [circuitOutputLayoutLayers_correct]
  exact circuitCyclicCompiledState_output c hwell level horder depth
    hbound u (circuitFlatIndex i q)

/-- Exact layer count for the shape-preserving cyclic construction.
Source: Appendix Theorem `thm: gen-ac`, circuit-depth component with
six explicit layout layers. -/
theorem compileCircuitCyclic_layerCount {n d : ℕ} [NeZero n]
    (c : ArithmeticCircuit (n * d) (n * d))
    (level : Fin c.size → ℕ) (depth : ℕ) :
    (compileCircuitCyclic c level depth).layerCount =
      2 * depth + 6 := by
  simp [CyclicCoyoteNetwork.layerCount, compileCircuitCyclic,
    circuitInputLayoutLayers, circuitOutputLayoutLayers,
    compiledCircuitLevelPrefix_length]

/-- Exact dense scalar storage of the external-shape construction.
Source: §4 equation `eq: coyote-recursion` and Appendix Theorem
`thm: gen-ac`; this is a larger bound than the paper's K-matrix claim. -/
theorem compileCircuitCyclic_denseParameterCount {n d : ℕ} [NeZero n]
    (c : ArithmeticCircuit (n * d) (n * d))
    (level : Fin c.size → ℕ) (depth : ℕ) :
    (compileCircuitCyclic c level depth).denseParameterCount =
      (2 * depth + 6) *
        ((n * d + c.size) * (n * d + c.size) +
          3 * n * (n * d + c.size)) := by
  simp [CyclicCoyoteNetwork.denseParameterCount,
    compileCircuitCyclic_layerCount]

end Transformer.Zoology
