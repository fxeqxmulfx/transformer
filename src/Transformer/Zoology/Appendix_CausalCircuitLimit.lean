/-
# Causal Coyote cannot return information from a later input position

Arora et al., arXiv:2312.04927v1, §4 equation `eq: coyote-recursion`
and Appendix Theorem `thm: gen-ac`. The theorem's general external-shape
simulation uses the cyclic-convolution option allowed in the appendix.
With strictly causal convolution, row zero cannot depend on later rows,
regardless of depth or inner dimensions.
-/

import Transformer.Zoology.Appendix_CyclicCircuitEquivalence

namespace Transformer.Zoology

/-- At the first row, a causal convolution reads only the first row.
Source: §4 equation `eq: coyote-recursion`, causal variant. -/
theorem causalConvolution_firstRow {n d : ℕ} [NeZero n]
    (u h : RealSequence n d) (q : Fin d) :
    causalConvolution u h 0 q = h 0 q * u 0 q := by
  classical
  unfold causalConvolution
  rw [Finset.sum_eq_single (0 : Fin n)]
  · simp
  · intro k _ hk
    have hnot : ¬ k ≤ (0 : Fin n) := by
      intro hle
      have hk0 : k = 0 := Fin.ext (Nat.eq_zero_of_le_zero
        (Fin.le_def.mp hle))
      exact hk hk0
    simp [hnot]
  · intro h
    exact (h (Finset.mem_univ (0 : Fin n))).elim

/-- One causal Coyote layer preserves agreement of two inputs at row
zero. Source: §4 equation `eq: coyote-recursion`. -/
theorem coyoteLayer_firstRow_congr {n d : ℕ} [NeZero n]
    (p : CoyoteParameters n d) (u v : RealSequence n d)
    (hrow : ∀ q, u 0 q = v 0 q) (q : Fin d) :
    coyoteLayer p u 0 q = coyoteLayer p v 0 q := by
  simp only [coyoteLayer, causalConvolution_firstRow]
  have hlin : linearProjection u p.weight 0 q =
      linearProjection v p.weight 0 q := by
    simp [linearProjection, hrow]
  rw [hlin, hrow q]

/-- Every finite causal Coyote stack preserves agreement at row zero,
even with arbitrary learned weights, biases, filters, and depth.
Source: Appendix Lemma `lem: stacking-layers`, causal variant. -/
theorem CoyoteNetwork.firstRow_congr {n d : ℕ} [NeZero n]
    (net : CoyoteNetwork n d) (u v : RealSequence n d)
    (hrow : ∀ q, u 0 q = v 0 q) (q : Fin d) :
    net.run u 0 q = net.run v 0 q := by
  change net.layers.foldl (fun state p => coyoteLayer p state) u 0 q =
    net.layers.foldl (fun state p => coyoteLayer p state) v 0 q
  induction net.layers generalizing u v with
  | nil => exact hrow q
  | cons p rest ih =>
      simp only [List.foldl_cons]
      apply ih
      exact coyoteLayer_firstRow_congr p u v hrow

/-- No padded causal Coyote model can output the second input token at
the first position for every input. The cyclic circuit construction
therefore uses a genuinely needed operation from the appendix.
Source: Appendix Theorem `thm: gen-ac`, general-circuit scope. -/
theorem no_causal_coyote_future_copy
    (model : PaddedCoyoteModel 2 1) :
    ¬ ∀ u : RealSequence 2 1, model.run u 0 0 = u 1 0 := by
  intro hcopy
  have hn : 0 < model.innerLength := by
    have h := model.lengthBound
    omega
  let zeroInput : RealSequence 2 1 := fun _ _ => 0
  let futureInput : RealSequence 2 1 :=
    fun i _ => if i = 1 then 1 else 0
  have hrow : ∀ q : Fin model.innerWidth,
      (padSequence zeroInput : RealSequence model.innerLength
        model.innerWidth) ⟨0, hn⟩ q =
      (padSequence futureInput : RealSequence model.innerLength
        model.innerWidth) ⟨0, hn⟩ q := by
    intro q
    simp [padSequence, zeroInput, futureInput]
  have heq := @CoyoteNetwork.firstRow_congr
    model.innerLength model.innerWidth ⟨Nat.ne_of_gt hn⟩ model.network
    (padSequence zeroInput) (padSequence futureInput) hrow
    (⟨0, by have h := model.widthBound; omega⟩ : Fin model.innerWidth)
  have hout : model.run zeroInput 0 0 =
      model.run futureInput 0 0 := by
    exact heq
  rw [hcopy zeroInput, hcopy futureInput] at hout
  norm_num [zeroInput, futureInput] at hout

/-- A one-gate circuit that copies the second input scalar to both
output positions. Source: Appendix `def: circuit-tuple`, input gates. -/
def futureReadCircuit : ArithmeticCircuit 2 2 := {
  size := 1
  gate := fun _ => .input 1
  output := fun _ => 0
}

/-- The future-reading circuit computes a constant output sequence equal
to its second input scalar. Source: Appendix `def: circuit-tuple`. -/
theorem futureReadCircuit_computes_iff (x y : Fin 2 → ℝ) :
    futureReadCircuit.Computes x y ↔ ∀ o, y o = x 1 := by
  change (∃ z : Fin 1 → ℝ,
    (∀ i, z i = x 1) ∧ ∀ o, y o = z 0) ↔ ∀ o, y o = x 1
  constructor
  · rintro ⟨z, hz, hy⟩ o
    calc y o = z 0 := hy o
      _ = x 1 := hz 0
  · intro h
    refine ⟨fun _ => x 1, ?_, ?_⟩
    · intro i
      fin_cases i
      rfl
    · intro o
      exact h o

/-- A nonconstant arithmetic circuit on a two-token scalar sequence
cannot be simulated by any padded causal Coyote network with the same
external input/output shape. The cyclic option in Appendix Theorem
`thm: gen-ac` avoids this obstruction. -/
theorem futureReadCircuit_no_causal_equivalence :
    ¬ ∃ model : PaddedCoyoteModel 2 1,
      ∀ u v : RealSequence 2 1,
        futureReadCircuit.Computes (circuitFlatInput u)
          (circuitFlatInput v) ↔ model.run u = v := by
  rintro ⟨model, hequiv⟩
  apply no_causal_coyote_future_copy model
  intro u
  let v : RealSequence 2 1 := fun _ _ => u 1 0
  have hidx : circuitFlatIndex (1 : Fin 2) (0 : Fin 1) = 1 := by
    decide
  have hcomp : futureReadCircuit.Computes (circuitFlatInput u)
      (circuitFlatInput v) := by
    apply (futureReadCircuit_computes_iff _ _).mpr
    intro o
    calc
      circuitFlatInput v o = u 1 0 := rfl
      _ = circuitFlatInput u 1 := by
        have hh := (circuitFlatInput_index u (1 : Fin 2)
          (0 : Fin 1)).symm
        simpa only [hidx] using hh
  have hout := (hequiv u v).mp hcomp
  have h0 := congrFun (congrFun hout 0) 0
  exact h0.trans rfl

/-- The same future-reading circuit has a shape-preserving cyclic
Coyote simulation. This satisfies the circuit resource hypotheses and
separates the two convolution semantics. Source: Appendix Theorem
`thm: gen-ac`, cyclic option. -/
theorem futureReadCircuit_cyclic_equivalence :
    ∃ model : PaddedCyclicCoyoteModel 2 1,
      ∀ u v : RealSequence 2 1,
        futureReadCircuit.Computes (circuitFlatInput u)
          (circuitFlatInput v) ↔ model.run u = v := by
  have hres : futureReadCircuit.HasDepthWidth 1 1 := by
    refine ⟨fun _ => 0, ?_, ?_, ?_⟩
    · intro i
      norm_num
    · intro i
      fin_cases i
      simp [futureReadCircuit]
    · intro l
      change (Finset.univ.filter (fun _ : Fin 1 => 0 = l)).card ≤ 1
      have hsub : (Finset.univ.filter
          (fun _ : Fin 1 => 0 = l)).card ≤
          (Finset.univ : Finset (Fin 1)).card :=
        Finset.card_le_card (Finset.filter_subset _ _)
      simpa using hsub
  obtain ⟨model, _, _, _, h⟩ :=
    exists_cyclic_coyote_circuit_equivalence (n := 2) (d := 1)
      futureReadCircuit 1 1 hres
  exact ⟨model, h⟩

end Transformer.Zoology
