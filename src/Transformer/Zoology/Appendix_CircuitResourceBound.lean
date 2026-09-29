/-
# A finite resource bound for the explicit circuit compiler

Arora et al., arXiv:2312.04927v1, Appendix `def: circuit-tuple` and
Theorem `thm: gen-ac`. Counting gates by level gives `size ≤ depth*width`
for the formal parallel-gate-width predicate. The shape-preserving cyclic
compiler consequently uses at most `nd + depth*width` feature columns.
This does not imply the paper's `O(width)` inner-dimension bound.
-/

import Transformer.Zoology.Appendix_CyclicCircuitEquivalence

namespace Transformer.Zoology

/-- A circuit with at most `width` gates on each of `depth` levels has at
most `depth*width` gates in total. Source: Appendix `def: circuit-tuple`,
parallel-operation reading of width. -/
theorem ArithmeticCircuit.size_le_depth_mul_width {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (depth width : ℕ)
    (h : c.HasDepthWidth depth width) :
    c.size ≤ depth * width := by
  obtain ⟨level, hbound, _, hwidth⟩ := h
  let levels : Finset ℕ := Finset.range depth
  have hmaps : Set.MapsTo level
      (↑(Finset.univ : Finset (Fin c.size))) (↑levels) := by
    intro i _
    exact Finset.mem_range.mpr (hbound i)
  have hcount := Finset.card_eq_sum_card_fiberwise hmaps
  have hsum : (∑ l ∈ levels,
      (Finset.univ.filter fun i : Fin c.size => level i = l).card) ≤
      ∑ _l ∈ levels, width := by
    apply Finset.sum_le_sum
    intro l _
    exact hwidth l
  have hsize : c.size ≤ ∑ _l ∈ levels, width := by
    simpa using hcount.le.trans hsum
  simpa [levels] using hsize

/-- The concrete cyclic Coyote compiler has a feature-width bound in
terms of circuit depth and parallel gate count. Source: Appendix Theorem
`thm: gen-ac`, dense construction; the stronger cut-width/K-matrix bound
is not asserted. -/
theorem compileCircuitCyclicModel_widthBound {n d : ℕ} [NeZero n]
    (c : ArithmeticCircuit (n * d) (n * d))
    (depth width : ℕ) (hres : c.HasDepthWidth depth width)
    (level : Fin c.size → ℕ) :
    (compileCircuitCyclicModel c level depth).innerWidth ≤
      n * d + depth * width := by
  change n * d + c.size ≤ n * d + depth * width
  exact Nat.add_le_add_left
    (c.size_le_depth_mul_width depth width hres) (n * d)

/-- The width assumption has a concrete nonconstant instance, so the
resource bound applies to the one-gate identity circuit.
Source: Appendix `def: circuit-tuple`. -/
example : identityCircuit.size ≤ 1 * 1 := by
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
  exact identityCircuit.size_le_depth_mul_width 1 1 hres

end Transformer.Zoology
