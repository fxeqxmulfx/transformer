module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes.Diagonal
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes.Identification

/-!
# Full-kernel witnesses for basis changes and aliases

Source: `review/flat_spectrum.md`, CM34. The diagonal witness retains both
independent directions under a genuine shear. A separate binary-field grid
has three observed raw labels and a forced alias of two full class vectors.
Its third class remains distinguishable, so aliasing is not a zero-code artifact.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes

/-- Every possible query/class decision agrees under the nontrivial shear,
CM34. The observed kernel itself, including its gauges, is unchanged. -/
theorem diagonal_all_matches_invariant : ∀ i j c,
    queryCode diagonalBasis i j = classCode diagonalBasis c ↔
      queryCode shearedBasis i j = classCode shearedBasis c := by
  intro i j c
  exact code_match_basis_iff _ _ i j c

/-- Every alias decision agrees for the same two complete bases, CM34. -/
theorem diagonal_all_aliases_invariant : ∀ a c,
    classCode diagonalBasis a = classCode diagonalBasis c ↔
      classCode shearedBasis a = classCode shearedBasis c := by
  intro a c
  exact class_alias_basis_iff _ _ a c

/-- The alternate basis also refuses the ambiguous off-diagonal query,
CM34. A coordinate change cannot make missing information appear. -/
theorem sheared_refuses :
    chooseClass shearedBasis (queryCode shearedBasis true false) [false, true] = none := by
  rw [← chooseClass_basis_eq diagonalBasis shearedBasis]
  exact diagonal_refuses

/-- Four distinct observed cells, CM34. No input receives conflicting raw labels. -/
def gridCells : Fin 4 → Bool × Bool :=
  ![(false, false), (true, false), (false, true), (true, true)]

/-- Three distinct raw class names in the original four observations, CM34. -/
def gridLabels : Fin 4 → Fin 3 := ![0, 1, 1, 2]

/-- The actual binary-field kernel of this observed grid, CM34. -/
def gridKernel : Submodule (ZMod 2) (Coordinates Bool Bool (Fin 3) → ZMod 2) :=
  observedKernel gridCells gridLabels false 0

/-- A complete basis of the observed grid, without choosing a desired alias, CM34. -/
noncomputable def gridBasis := Module.Free.chooseBasis (ZMod 2) gridKernel

/-- The grid's original equations force raw classes0and2 to have equal
full code vectors, CM34. This is compatible with deterministic raw observations. -/
theorem grid_forced_alias : classCode gridBasis 0 = classCode gridBasis 2 := by
  apply (class_alias_iff _ _ _).mpr
  intro w
  have h00 := w.property.1 (0 : Fin 4)
  have h10 := w.property.1 (1 : Fin 4)
  have h01 := w.property.1 (2 : Fin 4)
  have h11 := w.property.1 (3 : Fin 4)
  change w.val (.inl false) + w.val (.inr (.inl false)) - w.val (.inr (.inr 0)) = 0 at h00
  change w.val (.inl true) + w.val (.inr (.inl false)) - w.val (.inr (.inr 1)) = 0 at h10
  change w.val (.inl false) + w.val (.inr (.inl true)) - w.val (.inr (.inr 1)) = 0 at h01
  change w.val (.inl true) + w.val (.inr (.inl true)) - w.val (.inr (.inr 2)) = 0 at h11
  change w.val (.inr (.inr 0)) - w.val (.inr (.inr 2)) = 0
  have htwo : (2 : ZMod 2) = 0 := by decide
  linear_combination (norm := (ring_nf; simp only [htwo, mul_zero, sub_self]))
    h00 + h11 - h10 - h01

/-- A nonzero actual fit separating class1from class0, CM34. -/
def gridVector : Coordinates Bool Bool (Fin 3) → ZMod 2 :=
  pack (fun b => if b then 1 else 0) (fun b => if b then 1 else 0)
    (fun c => if c = 1 then 1 else 0)

/-- The nonzero fit obeys all four original rows and both gauges, CM34. -/
theorem gridVector_mem : gridVector ∈ gridKernel := by
  change (∀ t, queryAction (gridCells t).1 (gridCells t).2 (gridLabels t) gridVector = 0) ∧
    gridVector (.inl false) = 0 ∧ gridVector (.inr (.inr 0)) = 0
  decide

/-- The full code has a real alias and a distinct remaining class, CM34.
Thus code comparisons cover a noninjective but nonconstant class map. -/
theorem grid_nonconstant_alias :
    ¬ Function.Injective (classCode gridBasis) ∧ classCode gridBasis 0 ≠ classCode gridBasis 1 := by
  constructor
  · intro hinj
    have h := hinj grid_forced_alias
    norm_num at h
  · intro h
    have hz := (class_alias_iff gridBasis 0 1).mp h ⟨gridVector, gridVector_mem⟩
    norm_num [classAction, gridVector, pack] at hz

example := code_match_iff diagonalBasis true false true

example := code_match_basis_iff diagonalBasis shearedBasis true false true

example := class_alias_iff gridBasis 0 2

example := class_alias_basis_iff diagonalBasis shearedBasis false true

example : queryCode diagonalBasis true true = classCode diagonalBasis true :=
  observed_code_match (fun b : Bool => (b, b)) id false false diagonalBasis true

example : queryCode gridBasis false false = classCode gridBasis 2 :=
  matched_alias gridBasis false false 0 2
    (observed_code_match gridCells gridLabels false 0 gridBasis 0) grid_forced_alias

example : classCode diagonalBasis true = classCode diagonalBasis true :=
  chooseClass_match diagonalBasis _ [true, false]
    (chooseClass_first diagonalBasis true [false])

example := chooseClass_first diagonalBasis true [false]

example := chooseClass_basis_eq diagonalBasis shearedBasis true false [false, true]

/-- Each original raw class in the alias example really occurs in the
observation bank, CM34. The alias is not caused by an unused class coordinate. -/
theorem grid_all_labels_observed : ∀ c, ∃ t, gridLabels t = c := by decide

/-- Both diagonal directions are nonzero and have different row/column
actions, CM34. This also records concrete witnesses for the basis construction. -/
theorem diagonal_independent_actions :
    (diagonalBasis false).val (.inl true) = 1 ∧
    (diagonalBasis false).val (.inr (.inl true)) = 0 ∧
    (diagonalBasis true).val (.inl true) = 0 ∧
    (diagonalBasis true).val (.inr (.inl true)) = 1 := by
  norm_num [diagonalBasis, Module.Basis.map_apply, Pi.basisFun_apply,
    diagonalEquiv, diagonalVector, pack, Pi.single_apply]

/-- A deterministic target using all three original grid labels, CM34.
The two aliased raw classes occur at different inputs. -/
def gridTarget (i j : Bool) : Fin 3 :=
  if i then (if j then 2 else 1) else (if j then 1 else 0)

/-- The grid alias admits a nonconstant additive code representation,
CM34. Thus its equations have an actual task realization with raw aliases. -/
theorem grid_target_realizable :
    (∀ t, gridTarget (gridCells t).1 (gridCells t).2 = gridLabels t) ∧
    (∀ i j, (if gridTarget i j = 1 then (1 : ZMod 2) else 0) =
      (if i then 1 else 0) + (if j then 1 else 0)) ∧
    gridTarget false false ≠ gridTarget true true := by
  decide

example := diagonal_all_matches_invariant true false true

example := diagonal_all_aliases_invariant false true

end Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes
