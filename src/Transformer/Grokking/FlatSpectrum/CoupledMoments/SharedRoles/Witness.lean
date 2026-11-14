module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.SharedRoles.Identification
public import Mathlib.Tactic.LinearCombination

/-!
# Strict coverage from an actual shared-role kernel

Source: `review/flat_spectrum.md`, CM40. Two observed diagonal cells have
one observed class. The independent-role kernel permits a nonzero cross
query. Sharing forces that query to match when two is nonzero in a domain.
This is an algebraic condition on the actual matrix, not a pinned answer.
In characteristic two the complete shared kernel still refuses that query.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.SharedRoles

open KernelCodes

/-- The actual two diagonal observations, with the cross cells absent, CM40. -/
def diagonalCells (b : Bool) : Bool × Bool := (b, b)

/-- Original equations and two gauges for the diagonal witness, CM40. -/
abbrev diagonalKernel (R : Type*) [CommRing R] :=
  observedKernel (R := R) diagonalCells (fun _ => ()) false ()

/-- The same observations plus aligned input identities, CM40. -/
abbrev diagonalSharedKernel (R : Type*) [CommRing R] :=
  sharedKernel (R := R) diagonalCells (fun _ => ()) false () diagonalCells false

/-- A full restricted basis identifies the unobserved cross cell when
two is nonzero and there are no zero divisors, CM40. The characteristic
restriction is needed for this coverage claim, not for generic soundness. -/
theorem diagonal_match {R K : Type*} [CommRing R] [NoZeroDivisors R]
    (b : Module.Basis K R (diagonalSharedKernel R)) (h2 : (2 : R) ≠ 0) :
    queryCode b false true = classCode b () := by
  apply (code_match_iff b false true ()).mpr
  intro w
  have h0 := w.property.1.1 false
  have h1 := w.property.1.1 true
  have hs := w.property.2 true
  change w.val (.inl false) + w.val (.inr (.inl false)) -
    w.val (.inr (.inr ())) = 0 at h0
  change w.val (.inl true) + w.val (.inr (.inl true)) -
    w.val (.inr (.inr ())) = 0 at h1
  change (w.val (.inl true) - w.val (.inr (.inl true))) -
    (w.val (.inl false) - w.val (.inr (.inl false))) = 0 at hs
  have ht : (2 : R) * queryAction false true () w.val = 0 := by
    change 2 * (w.val (.inl false) + w.val (.inr (.inl true)) -
      w.val (.inr (.inr ()))) = 0
    linear_combination h1 - hs + h0
  exact (mul_eq_zero.mp ht).resolve_left h2

/-- A genuine unrestricted direction, with opposite changes in the two
input roles and no change in the observed class, CM40. -/
def oppositeVector (R : Type*) [CommRing R] : Coordinates Bool Bool Unit → R :=
  pack (fun b => if b then 1 else 0) (fun b => if b then -1 else 0) (fun _ => 0)

/-- The opposite direction satisfies every original equation and gauge,
CM40. Its existence is independent of a chosen basis. -/
theorem oppositeVector_mem (R : Type*) [CommRing R] :
    oppositeVector R ∈ diagonalKernel R := by
  refine ⟨?_, rfl, rfl⟩
  intro b
  cases b <;> simp [diagonalCells, oppositeVector, pack, queryAction]

/-- A complete basis of the unrestricted witness over the field of three
elements, CM40. No one-dimensional code was chosen in its place. -/
noncomputable def diagonalBasis := Module.Free.chooseBasis (ZMod 3) (diagonalKernel (ZMod 3))

/-- A complete basis of the actual restricted witness over the same field,
CM40; observations and gauges are identical to the unrestricted case. -/
noncomputable def diagonalSharedBasis :=
  Module.Free.chooseBasis (ZMod 3) (diagonalSharedKernel (ZMod 3))

/-- The original full code does not match the only observed class at the
missing cross cell, CM40. The opposite vector witnesses the obstruction. -/
theorem diagonal_unrestricted_no_match :
    queryCode diagonalBasis false true ≠ classCode diagonalBasis () := by
  intro h
  have hz := (code_match_iff diagonalBasis false true ()).mp h
    ⟨oppositeVector (ZMod 3), oppositeVector_mem (ZMod 3)⟩
  norm_num [oppositeVector, pack, queryAction] at hz

/-- The additional shared-role rows strictly extend full-code coverage,
CM40. This query has no original observation. -/
theorem diagonal_shared_match :
    queryCode diagonalSharedBasis false true = classCode diagonalSharedBasis () :=
  diagonal_match diagonalSharedBasis (by decide)

/-- Both observed cells differ from the newly matched query, CM40. -/
theorem diagonal_unobserved : ∀ b, diagonalCells b ≠ (false, true) := by
  intro b
  cases b <;> decide

/-- The unchanged class search changes from refusal to acceptance on the
same observed class bank, CM40. Its equality test is the full basis test. -/
theorem diagonal_search_extension :
    chooseClass diagonalBasis (queryCode diagonalBasis false true) [()] = none ∧
    chooseClass diagonalSharedBasis (queryCode diagonalSharedBasis false true) [()] = some () := by
  constructor
  · simp only [chooseClass, diagonal_unrestricted_no_match, ite_false]
  · rw [diagonal_shared_match]
    exact chooseClass_first _ _ _

/-- In characteristic two, the opposite direction survives every actual
observation, gauge and shared-role equation, CM40. -/
theorem oppositeVector_mem_shared_two :
    oppositeVector (ZMod 2) ∈ diagonalSharedKernel (ZMod 2) := by
  refine ⟨oppositeVector_mem (ZMod 2), ?_⟩
  intro b
  cases b <;> norm_num [roleAction, diagonalCells, oppositeVector, pack]
  decide

/-- The complete restricted basis in characteristic two, CM40. -/
noncomputable def diagonalSharedBasisTwo :=
  Module.Free.chooseBasis (ZMod 2) (diagonalSharedKernel (ZMod 2))

/-- Sharing cannot force this unseen-query match in characteristic two,
CM40. The nonzero-two hypothesis in the strict coverage witness is essential. -/
theorem diagonal_shared_two_no_match :
    queryCode diagonalSharedBasisTwo false true ≠ classCode diagonalSharedBasisTwo () := by
  intro h
  have hz := (code_match_iff diagonalSharedBasisTwo false true ()).mp h
    ⟨oppositeVector (ZMod 2), oppositeVector_mem_shared_two⟩
  norm_num [oppositeVector, pack, queryAction] at hz

/-- The actual class search retains refusal in the exceptional
characteristic, CM40; it does not silently guess the only observed class. -/
theorem diagonal_shared_two_refuses :
    chooseClass diagonalSharedBasisTwo (queryCode diagonalSharedBasisTwo false true) [()] = none := by
  simp only [chooseClass, diagonal_shared_two_no_match, ite_false]

example : queryCode diagonalSharedBasis false true = classCode diagonalSharedBasis () :=
  diagonal_match diagonalSharedBasis (by decide)

example : oppositeVector (ZMod 3) ∈ diagonalKernel (ZMod 3) := oppositeVector_mem (ZMod 3)

example : queryCode diagonalSharedBasis false false = classCode diagonalSharedBasis () :=
  observed_match diagonalSharedBasis false

example : queryCode diagonalSharedBasis true true = classCode diagonalSharedBasis () :=
  match_of_unrestricted diagonalBasis diagonalSharedBasis true true ()
    (observed_code_match diagonalCells (fun _ => ()) false () diagonalBasis true)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.SharedRoles
