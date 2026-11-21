module

public import Transformer.GainMatroid.Section2_DeltaYExamples
public import Transformer.GainMatroid.Section2_ExchangeForms
public import Transformer.GainMatroid.Section2_ExchangeMatroids

/-!
# The columns of `I(K₄)` on a triad

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. `twoOfThreeMatrix` is a binary representation
of `U_{2,3}`, on which the results of `Section2_ExchangeForms` are applied.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

/-- A binary representation of `U_{2,3}`. -/
def twoOfThreeMatrix : Matrix (Fin 2) (Fin 3) (ZMod 2) := !![1, 0, 1; 0, 1, 1]

set_option maxRecDepth 100000 in
theorem twoOfThreeMatrix_represents : IsMatrixRepresentation twoOfThree twoOfThreeMatrix := by
  refine ⟨rfl, fun I => ?_⟩
  obtain ⟨I, rfl⟩ := I.toFinite.exists_finset_coe
  refine (matroidOfFinset_indep I).trans (Iff.trans ?_ linearIndepOn_finset_iff.symm)
  revert I
  decide +kernel

/-- The three elements of `U_{2,3}` form a triangle. -/
theorem twoOfThree_triangle : twoOfThree.IsCircuit {1, 2, 0} :=
  (twoOfThree_isCircuit_iff _).2 triangle_isLabelledTriangle.isCycle

example := (ProjEquiv.refl twoOfThreeMatrix).linearIndepOn_iff {0}

example := twoOfThreeMatrix_represents.of_projEquiv (ProjEquiv.refl twoOfThreeMatrix)

example := exists_mem_rowSpace_eq_on (A := twoOfThreeMatrix) (s := {0, 1})
  ((twoOfThreeMatrix_represents.2 _).1 ((matroidOfFinset_indep _).2 (by decide))) 1

example := exists_col_eq_of_isCircuit twoOfThreeMatrix_represents (by decide) (by decide)
  (by decide) twoOfThree_triangle

example := exists_isTriangleForm twoOfThreeMatrix_represents (by decide) (by decide) (by decide)
  twoOfThree_triangle

end Transformer.GainMatroid
