/-
# External-shape equivalence for the cyclic circuit compiler

Arora et al., arXiv:2312.04927v1, §4 Theorem `thm:equiv` and Appendix
Theorem `thm: gen-ac`. This file closes the functional transition from an
arbitrary arithmetic circuit on a flattened `n × d` input to a cyclic
Coyote model with the same external input and output shape. Resource
bounds remain those of the explicit dense construction, not the paper's
K-matrix and cut-width bound.
-/

import Transformer.Zoology.Appendix_CyclicCircuitSimulation

namespace Transformer.Zoology

/-- A padded cyclic Coyote model built from a levelled arithmetic
circuit. Its external input and output both have shape `n × d`.
Source: Appendix Theorem `thm: gen-ac`, functional equivalence. -/
def compileCircuitCyclicModel {n d : ℕ} [NeZero n]
    (c : ArithmeticCircuit (n * d) (n * d))
    (level : Fin c.size → ℕ) (depth : ℕ) :
    PaddedCyclicCoyoteModel n d := {
  innerLength := n
  innerWidth := n * d + c.size
  lengthBound := le_refl n
  widthBound := circuitInput_widthBound
  network := compileCircuitCyclic c level depth
}

/-- The padded model agrees with the circuit at every external output
coordinate. Source: Appendix Theorem `thm: gen-ac`, exact functional
equivalence for cyclic convolution. -/
theorem compileCircuitCyclicModel_output {n d : ℕ} [NeZero n]
    (c : ArithmeticCircuit (n * d) (n * d)) (hwell : c.WellFormed)
    (level : Fin c.size → ℕ) (horder : CircuitLevelOrder c level)
    (depth : ℕ) (hbound : ∀ j, level j < depth)
    (u : RealSequence n d) (i : Fin n) (q : Fin d) :
    (compileCircuitCyclicModel c level depth).run u i q =
      c.evalNode hwell (circuitFlatInput u)
        (c.output (circuitFlatIndex i q)) := by
  change (compileCircuitCyclic c level depth).run (padSequence u) i
    (circuitOriginalFeature (size := c.size) q) = _
  exact compileCircuitCyclic_output c hwell level horder depth
    hbound u i q

/-- For every pair of external sequences, the arithmetic circuit
computes the flattened output exactly when it equals the compiled cyclic
Coyote model's output. Source: §4 Theorem `thm:equiv` and Appendix
Theorem `thm: gen-ac`, functional equivalence only. The paper's resource
claim `N'=O(w), d'=d` is not supplied by this dense compiler. -/
theorem compileCircuitCyclicModel_equiv {n d : ℕ} [NeZero n]
    (c : ArithmeticCircuit (n * d) (n * d)) (hwell : c.WellFormed)
    (level : Fin c.size → ℕ) (horder : CircuitLevelOrder c level)
    (depth : ℕ) (hbound : ∀ j, level j < depth)
    (u v : RealSequence n d) :
    c.Computes (circuitFlatInput u) (circuitFlatInput v) ↔
      (compileCircuitCyclicModel c level depth).run u = v := by
  constructor
  · intro h
    have heval := (c.computes_iff_eval hwell
      (circuitFlatInput u) (circuitFlatInput v)).mp h
    funext i q
    calc
      (compileCircuitCyclicModel c level depth).run u i q =
          c.evalNode hwell (circuitFlatInput u)
            (c.output (circuitFlatIndex i q)) :=
        compileCircuitCyclicModel_output c hwell level horder
          depth hbound u i q
      _ = circuitFlatInput v (circuitFlatIndex i q) := by
        exact congrFun heval (circuitFlatIndex i q) |>.symm
      _ = v i q := circuitFlatInput_index v i q
  · intro h
    apply (c.computes_iff_eval hwell
      (circuitFlatInput u) (circuitFlatInput v)).mpr
    funext o
    let p := finProdFinEquiv.symm o
    have hp : circuitFlatIndex p.1 p.2 = o :=
      Equiv.apply_symm_apply finProdFinEquiv o
    have hout := compileCircuitCyclicModel_output c hwell level
      horder depth hbound u p.1 p.2
    rw [h, hp] at hout
    simpa [circuitFlatInput, p] using hout

/-- A depth/width witness yields a shape-preserving cyclic Coyote model
with an explicit dense resource bound. This corrects only the functional
part of Appendix Theorem `thm: gen-ac`; it does not assert the paper's
width-sensitive K-matrix complexity. -/
theorem exists_cyclic_coyote_circuit_equivalence {n d : ℕ} [NeZero n]
    (c : ArithmeticCircuit (n * d) (n * d))
    (depth width : ℕ) (hres : c.HasDepthWidth depth width) :
    ∃ model : PaddedCyclicCoyoteModel n d,
      model.innerLength = n ∧
      model.innerWidth = n * d + c.size ∧
      model.network.layerCount = 2 * depth + 6 ∧
      ∀ u v : RealSequence n d,
        c.Computes (circuitFlatInput u) (circuitFlatInput v) ↔
          model.run u = v := by
  obtain ⟨level, hbound, horder⟩ := hres.levelOrder c depth width
  have hwell : c.WellFormed := hres.wellFormed c depth width
  refine ⟨compileCircuitCyclicModel c level depth, rfl, rfl,
    compileCircuitCyclic_layerCount c level depth, ?_⟩
  intro u v
  exact compileCircuitCyclicModel_equiv c hwell level horder depth
    hbound u v

/-- The external-shape result has a concrete nonconstant instance:
the one-input identity circuit at shape `1 × 1`.
Source: Appendix `def: circuit-tuple`, identity example. -/
example : ∃ model : PaddedCyclicCoyoteModel 1 1,
    ∀ u v : RealSequence 1 1,
      identityCircuit.Computes (circuitFlatInput u)
        (circuitFlatInput v) ↔ model.run u = v := by
  have hres : identityCircuit.HasDepthWidth 1 1 := by
    refine ⟨fun _ => 0, ?_, ?_, ?_⟩
    · intro i
      norm_num
    · intro i
      fin_cases i
      simp [identityCircuit]
    · intro l
      change (Finset.univ.filter (fun _ : Fin 1 => 0 = l)).card ≤ 1
      have hsub : (Finset.univ.filter
          (fun _ : Fin 1 => 0 = l)).card ≤
          (Finset.univ : Finset (Fin 1)).card :=
        Finset.card_le_card (Finset.filter_subset _ _)
      simpa using hsub
  obtain ⟨model, _, _, _, h⟩ :=
    exists_cyclic_coyote_circuit_equivalence (n := 1) (d := 1)
      identityCircuit 1 1 hres
  exact ⟨model, h⟩

end Transformer.Zoology
