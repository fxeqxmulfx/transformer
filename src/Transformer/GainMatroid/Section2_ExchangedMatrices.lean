module

public import Transformer.GainMatroid.Section2_CoconnectionDeletion
public import Transformer.GainMatroid.Section2_CoconnectionFlats
public import Transformer.GainMatroid.Section2_ConnectionMatroid
public import Transformer.GainMatroid.Section2_K4Matrix

/-!
# The matroids of the exchanged matrices

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, citing Brylawski: "Let `Δ_X A` denote the
matrix obtained from `A` by replacing the columns for `X` with first three columns of `I(K₄)`, where
possibly the fourth row is omitted or additional zero rows are added. Then `M(Δ_X A) = Δ_X M(A)`",
and "Let `∇_Y A` denote the matrix `A` obtained by replacing the columns of `Y` with those of the
last three columns of `I(K₄)`. Then `M(∇_Y A) = ∇_Y M(A)`."

`isDeltaMatroid_of_isTriangleForm` and `isNablaMatroid_of_isTriadForm` state these for matrices in
the forms `IsTriangleForm` and `IsTriadForm`. As in the source, `Y` is a triad. `IsTriadForm` also
asks that the row `v` vanish off `Y`, which the source omits: without it the second claim fails
(`nablaMatrix_fails`, in `Section2_NablaZeroRow`). Brylawski's paper is not among the local
sources. The first is proved here: the matroid of `connectionMatrix` is the
generalized parallel connection of `IsDeltaMatroid` (`IsTriangleForm.isFlat_connection_iff`), and
deleting `X` from it leaves `Δ_X A` (`deltaMatrix_eq_connectionMatrix`). The second follows by
duality, since `∇_Y M = (Δ_Y M*)*`: the dual of the matroid of `coconnectionMatrix` is the
generalized parallel connection of `M(A)*` and `M(K₄)` across `Y`
(`IsTriadForm.isFlat_dual_coconnection_iff`), and deleting `Y` from it leaves `M(∇_Y A)*`
(`IsTriadForm.dual_comap_coconnection`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

variable {R E F : Type*} [Field F]

/-- **§2.4** (Brylawski). `M(Δ_X A) = Δ_X M(A)` when the columns of `A` on the triangle
`X = {a, b, c}` are the last three columns of `I(K₄)`. -/
theorem isDeltaMatroid_of_isTriangleForm [DecidableEq E] [DecidableEq R] [Finite E]
    {M M' : Matroid E} {A : Matrix R E F} {a b c : E} {x y z : R}
    (hA : IsMatrixRepresentation M A) (hform : IsTriangleForm A a b c x y z)
    (hA' : IsMatrixRepresentation M' (deltaMatrix A a b c x y z)) :
    IsDeltaMatroid M a b c M' := by
  have hP := matrixMatroid_isMatrixRepresentation (connectionMatrix A x y z)
  refine ⟨hform.a_ne_b, hform.a_ne_c, hform.b_ne_c, hform.isCircuit hA,
    matrixMatroid (connectionMatrix A x y z), ?_, hform.isFlat_connection_iff hA,
    hA'.eq (hP.comap (deltaRelabel_injective a b c) deltaMatrix_eq_connectionMatrix)⟩
  rw [hP.1, hA.1, Set.image_univ, Set.range_inl_union_range_inr]

/-- **§2.4, corrected** (Brylawski). `M(∇_Y A) = ∇_Y M(A)` when the columns of `A` on the triad
`Y = {a, b, c}` are the first three columns of `I(K₄)`, with no row omitted, and its row `v`
vanishes off `Y` (`IsTriadForm`). The source asks only for the columns, and then the claim fails
(`nablaMatrix_fails`). `BiasedGraph.exists_isTriadForm` brings every independent triad to this
form. -/
theorem isNablaMatroid_of_isTriadForm [DecidableEq E] [DecidableEq R] [Finite E]
    {M M' : Matroid E} {A : Matrix R E F} {v : R} {a b c : E} {x y z : R}
    (hA : IsMatrixRepresentation M A) (hform : IsTriadForm A v a b c x y z)
    (ht : M✶.IsCircuit {a, b, c})
    (hA' : IsMatrixRepresentation M' (nablaMatrix A a b c x y z)) :
    IsNablaMatroid M a b c M' := by
  have hQ := matrixMatroid_isMatrixRepresentation (coconnectionMatrix A a b c x y z)
  refine ⟨_, ⟨hform.a_ne_b, hform.a_ne_c, hform.b_ne_c, ht,
    (matrixMatroid (coconnectionMatrix A a b c x y z))✶, ?_, hform.isFlat_dual_coconnection_iff hA,
    rfl⟩, ?_⟩
  · rw [Matroid.dual_ground, Matroid.dual_ground, hQ.1, hA.1, Set.image_univ,
      Set.range_inl_union_range_inr]
  · rw [hform.dual_comap_coconnection hA', Matroid.dual_dual]

/-- The triangle of `U_{2,3}` over `𝔽₂`, with the last three columns of `I(K₄)`. -/
def triangleMatrix : Matrix (Fin 3) (Fin 3) (ZMod 2) := !![1, 0, 1; 1, 1, 0; 0, 1, 1]

set_option maxRecDepth 100000 in
theorem triangleMatrix_represents : IsMatrixRepresentation twoOfThree triangleMatrix := by
  refine ⟨rfl, fun I => ?_⟩
  obtain ⟨I, rfl⟩ := I.toFinite.exists_finset_coe
  refine (matroidOfFinset_indep I).trans (Iff.trans ?_ linearIndepOn_finset_iff.symm)
  revert I
  decide +kernel

theorem triangleMatrix_isTriangleForm : IsTriangleForm triangleMatrix 1 2 0 0 1 2 := by
  refine ⟨by decide, by decide, by decide, ?_, ?_, ?_⟩ <;> decide +kernel

set_option maxRecDepth 100000 in
/-- The exchanged triangle is the free matroid on three elements. -/
theorem deltaMatrix_triangleMatrix_represents :
    IsMatrixRepresentation freeThree (deltaMatrix triangleMatrix 1 2 0 0 1 2) := by
  refine ⟨rfl, fun I => ?_⟩
  obtain ⟨I, rfl⟩ := I.toFinite.exists_finset_coe
  refine (matroidOfFinset_indep I).trans (Iff.trans ?_ linearIndepOn_finset_iff.symm)
  revert I
  decide +kernel

theorem clawMatrix_isTriadForm : IsTriadForm clawMatrix 0 0 1 2 1 2 3 := by
  refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, ?_, ?_, ?_, ?_⟩ <;>
    decide +kernel

set_option maxRecDepth 100000 in
/-- The exchanged star of `clawGraph` is the matroid of `clawNabla`. -/
theorem nablaMatrix_clawMatrix_represents :
    IsMatrixRepresentation nablaMatroid (nablaMatrix clawMatrix 0 1 2 1 2 3) := by
  refine ⟨rfl, fun I => ?_⟩
  obtain ⟨I, rfl⟩ := I.toFinite.exists_finset_coe
  refine (matroidOfFinset_indep I).trans (Iff.trans ?_ linearIndepOn_finset_iff.symm)
  revert I
  decide +kernel

example := isDeltaMatroid_of_isTriangleForm triangleMatrix_represents
  triangleMatrix_isTriangleForm deltaMatrix_triangleMatrix_represents

example := isNablaMatroid_of_isTriadForm clawMatrix_represents clawMatrix_isTriadForm
  clawMatroid_triad nablaMatrix_clawMatrix_represents

end Transformer.GainMatroid
