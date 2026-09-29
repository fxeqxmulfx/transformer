/-
# Two-feature gate projections belong to the K-matrix class

Arora et al., arXiv:2312.04927v1, Appendix `def: butterfly`,
`def: W-kmat`, and Theorem `thm: gen-ac`. Every real `2 × 2` matrix is a
single butterfly factor. This verifies that the two-feature addition and
multiplication gate projections used in the circuit base cases satisfy
the article's K-matrix restriction.
-/

import Transformer.Zoology.Appendix_KCoyotePrimitives
import Transformer.Zoology.Appendix_CircuitPrimitives

namespace Transformer.Zoology

/-- The depth-one butterfly tree with arbitrary `2 × 2` entries.
Source: Appendix `def: butterfly`, size-two base case. -/
def matrix2ButterflyTree (A : Fin 2 → Fin 2 → ℝ) : ButterflyTree 1 :=
  .node {
    upperLeft := fun _ => A 0 0
    upperRight := fun _ => A 0 1
    lowerLeft := fun _ => A 1 0
    lowerRight := fun _ => A 1 1
  } .leaf .leaf

/-- The butterfly tree acts by the supplied ordinary matrix.
Source: Appendix `def: butterfly`, size-two base case. -/
theorem matrix2ButterflyTree_apply (A : Fin 2 → Fin 2 → ℝ)
    (u : Fin 2 → ℝ) (i : Fin 2) :
    (matrix2ButterflyTree A).apply u i =
      A i 0 * u 0 + A i 1 * u 1 := by
  fin_cases i <;> rfl

/-- The matrix extracted from the tree is the supplied matrix.
Source: Appendix `def: butterfly`, size-two base case. -/
theorem matrix2ButterflyTree_matrix (A : Fin 2 → Fin 2 → ℝ)
    (i j : Fin 2) :
    (matrix2ButterflyTree A).matrix i j = A i j := by
  calc
    (matrix2ButterflyTree A).matrix i j =
        A i 0 * coordinateBasis j 0 + A i 1 * coordinateBasis j 1 :=
      matrix2ButterflyTree_apply A (coordinateBasis j) i
    _ = A i j := by fin_cases j <;> simp [coordinateBasis]

/-- The two-dimensional identity matrix. -/
def identityMatrix2 (i j : Fin 2) : ℝ := if i = j then 1 else 0

/-- A single `BB*` factor encoding an arbitrary `2 × 2` matrix; its right
butterfly is the identity. Source: Appendix `def: kaleidoscope`, `BB*`. -/
def matrix2BBStar (A : Fin 2 → Fin 2 → ℝ) : BBStar 1 := {
  left := matrix2ButterflyTree A
  right := matrix2ButterflyTree identityMatrix2
}

/-- The `BB*` factor acts by the supplied matrix. Source: Appendix
`def: kaleidoscope`, size-two case. -/
theorem matrix2BBStar_apply (A : Fin 2 → Fin 2 → ℝ)
    (u : Fin 2 → ℝ) (i : Fin 2) :
    (matrix2BBStar A).apply u i = ∑ j : Fin 2, A i j * u j := by
  have hright : (fun j : Fin 2 =>
      ∑ t : Fin 2, (matrix2ButterflyTree identityMatrix2).matrix t j * u t) = u := by
    funext j
    fin_cases j <;> simp [matrix2ButterflyTree_matrix, identityMatrix2]
  change (matrix2ButterflyTree A).apply
    (fun j : Fin 2 =>
      ∑ t : Fin 2, (matrix2ButterflyTree identityMatrix2).matrix t j * u t) i = _
  rw [hright]
  calc
    (matrix2ButterflyTree A).apply u i = A i 0 * u 0 + A i 1 * u 1 :=
      matrix2ButterflyTree_apply A u i
    _ = ∑ j : Fin 2, A i j * u j := by simp [Fin.sum_univ_two]

/-- An unexpanded, width-one `BB*` product encoding an arbitrary `2 × 2`
weight matrix. Source: Appendix `def: kaleidoscope`. -/
def matrix2Kaleidoscope (A : Fin 2 → Fin 2 → ℝ) :
    ExpandedKaleidoscope 1 0 := {
  inner := {factors := [matrix2BBStar A]}
}

/-- The expanded K operator acts by the original `2 × 2` matrix.
Source: Appendix `def: W-kmat`, two-feature case. -/
theorem matrix2Kaleidoscope_apply (A : Fin 2 → Fin 2 → ℝ)
    (u : Fin 2 → ℝ) (i : Fin 2) :
    (matrix2Kaleidoscope A).apply u i = ∑ j : Fin 2, A i j * u j := by
  have hpad : padButterfly (k := 1) (e := 0) u = u := by
    funext j
    simp [padButterfly]
  change cropButterfly (k := 1) (e := 0)
    ((matrix2BBStar A).apply (padButterfly (k := 1) (e := 0) u)) i = _
  rw [hpad]
  change (matrix2BBStar A).apply u i = _
  exact matrix2BBStar_apply A u i

/-- Therefore the K-restricted Coyote weight can represent any
`2 × 2` projection, including the gate weights. Source: Appendix
`def: W-kmat` and Theorem `thm: gen-ac`, gate base cases. -/
theorem matrix2_kaleidoscopeWeight (A : Fin 2 → Fin 2 → ℝ)
    (input output : Fin 2) :
    kaleidoscopeWeight (matrix2Kaleidoscope A) input output =
      A output input := by
  unfold kaleidoscopeWeight ExpandedKaleidoscope.matrix
  calc
    (matrix2Kaleidoscope A).apply (coordinateBasis input) output =
        ∑ j : Fin 2, A output j * coordinateBasis input j :=
      matrix2Kaleidoscope_apply A _ output
    _ = A output input := by
      fin_cases input <;> simp [coordinateBasis]

/-- Convert any two-feature Coyote layer into a K-constrained layer with
one `BB*` factor and no dimension expansion. Source: Appendix
`def: W-kmat`, size-two butterfly representation. -/
def twoFeatureKParameters {n : ℕ} (p : CoyoteParameters n 2) :
    KCoyoteParameters n 1 0 := {
  weight := matrix2Kaleidoscope
    (fun output input => p.weight input output)
  filter := p.filter
  bias₁ := p.bias₁
  bias₂ := p.bias₂
}

/-- The K-constrained conversion has exactly the original Coyote
semantics and parameters after decoding the compressed weight.
Source: Appendix `def: W-kmat`, two-feature case. -/
theorem twoFeatureKParameters_toParameters {n : ℕ}
    (p : CoyoteParameters n 2) :
    (twoFeatureKParameters p).toParameters = p := by
  cases p with
  | mk W h b₁ b₂ =>
      have hw : kaleidoscopeWeight
          (matrix2Kaleidoscope (fun output input => W input output)) = W := by
        funext input output
        exact matrix2_kaleidoscopeWeight _ input output
      simp only [twoFeatureKParameters, KCoyoteParameters.toParameters]
      rw [hw]
      rfl

/-- The addition gate respects the article's K-matrix weight restriction.
Source: Appendix Theorem `thm: gen-ac`, linear-gate base case. -/
theorem kCoyote_addition_gate (a b : ℝ) :
    coyoteLayer (twoFeatureKParameters additionGateParameters).toParameters
      (gateInput a b) 0 ⟨0, by decide⟩ = a + b := by
  rw [twoFeatureKParameters_toParameters]
  exact coyote_addition_gate a b

/-- The multiplication gate also respects the K-matrix restriction.
Source: Appendix Theorem `thm: gen-ac`, multiplication-gate base case. -/
theorem kCoyote_multiplication_gate (a b : ℝ) :
    coyoteLayer (twoFeatureKParameters multiplicationGateParameters).toParameters
      (gateInput a b) 0 ⟨0, by decide⟩ = a * b := by
  rw [twoFeatureKParameters_toParameters]
  exact coyote_multiplication_gate a b

end Transformer.Zoology
