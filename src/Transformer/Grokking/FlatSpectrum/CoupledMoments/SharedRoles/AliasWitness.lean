module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.SharedRoles.Witness

/-!
# A declared answer without an identified raw alias

Source: `review/flat_spectrum.md`, CM40. Two shared-role targets fit the
same diagonal observations and differ at an unseen raw alias. Projection
onto their code identifies the declared answer. A third, independent-role
target fits those observations but has a different answer, so the extra
architectural premise cannot be inferred from observed fit alone.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.SharedRoles

open KernelCodes

/-- Two raw targets differ only at the unobserved cross cells, CM40. -/
def diagonalTarget (which i j : Bool) : ZMod 3 × Bool :=
  if i = j then (0, false) else (0, which)

/-- Both raw worlds have the same actual two original labels, CM40. -/
theorem diagonal_target_observed (which : Bool) :
    ∀ t, diagonalTarget which (diagonalCells t).1 (diagonalCells t).2 = (0, false) := by
  intro t
  simp [diagonalTarget, diagonalCells]

/-- Their unknown code has a shared-role additive realization, CM40.
The raw alias is unconstrained at the unobserved cross cells. -/
theorem diagonal_target_realized (which : Bool) :
    ∀ i j, (diagonalTarget which i j).1 = (0 : ZMod 3) + 0 := by
  intro i j
  cases i <;> cases j <;> rfl

/-- Actual different raw answers remain compatible with the same decoded
answer and observations, CM40. This is a many-to-one readout. -/
theorem diagonal_raw_ambiguity :
    diagonalTarget false false true ≠ diagonalTarget true false true ∧
    (diagonalTarget false false true).1 = (diagonalTarget true false true).1 := by
  decide

/-- An independent-role target using the opposite fitting direction,
CM40. This is a concrete possible unknown target, not a learner input. -/
def independentTarget (i j : Bool) : ZMod 3 × Bool :=
  ((if i then 1 else 0) + (if j then -1 else 0), false)

/-- Independent roles also fit both original observations, CM40. -/
theorem independent_target_observed :
    ∀ t, independentTarget (diagonalCells t).1 (diagonalCells t).2 = (0, false) := by
  intro t
  cases t <;> norm_num [independentTarget, diagonalCells]

/-- The same observed data admit a different decoded cross answer in the
broader independent-role family, CM40. Shared-role realizability is a real
extra premise, not a consequence of observed class separation or flatness. -/
theorem independent_answer_ambiguity :
    (independentTarget false true).1 ≠ (diagonalTarget true false true).1 := by
  decide

/-- The direction causing the independent answer ambiguity violates the
new actual role equation, CM40. It was not removed by an answer oracle. -/
theorem oppositeVector_not_shared :
    oppositeVector (ZMod 3) ∉ roleKernel diagonalCells false := by
  intro h
  have hs := h true
  norm_num [roleAction, diagonalCells, oppositeVector, pack] at hs
  exact (by decide : (2 : ZMod 3) ≠ 0) hs

example : ∀ t, diagonalTarget true (diagonalCells t).1 (diagonalCells t).2 = (0, false) :=
  diagonal_target_observed true

example : ∀ i j, (diagonalTarget true i j).1 = (0 : ZMod 3) + 0 :=
  diagonal_target_realized true

example : (diagonalTarget true false true).1 = (0 : ZMod 3) :=
  matched_code_sound diagonalSharedBasis (fun _ => (0, false)) Prod.fst
    (fun _ => 0) (fun _ => 0) (diagonalTarget true) (diagonal_target_realized true)
    (diagonal_target_observed true) (fun _ => rfl) false true () diagonal_shared_match

example : (diagonalTarget true false true).1 = (0 : ZMod 3) :=
  matched_readout_sound diagonalSharedBasis (fun _ => (0, false)) Prod.fst Prod.fst
    (fun _ _ h => h) (fun _ => 0) (fun _ => 0) (diagonalTarget true)
    (diagonal_target_realized true) (diagonal_target_observed true) (fun _ => rfl)
    false true () diagonal_shared_match

example : (diagonalTarget true false true).1 = (0 : ZMod 3) :=
  named_readout_sound (rawLabels := fun _ => (0, false)) Prod.fst diagonalSharedBasis
    (fun _ => 0) (fun _ => rfl) Prod.fst (fun d => ⟨(d, false), rfl⟩)
    (fun _ _ h => h) rfl (fun _ => 0) (fun _ => 0) (diagonalTarget true)
    (diagonal_target_realized true) (diagonal_target_observed true) (fun _ => rfl)
    false true () diagonal_shared_match

example : (diagonalTarget true false true).1 = (0 : ZMod 3) :=
  choose_readout_sound (rawLabels := fun _ => (0, false)) Prod.fst diagonalSharedBasis
    (fun _ => 0) (fun _ => rfl) Prod.fst (fun d => ⟨(d, false), rfl⟩)
    (fun _ _ h => h) rfl (fun _ => 0) (fun _ => 0) (diagonalTarget true)
    (diagonal_target_realized true) (diagonal_target_observed true) (fun _ => rfl)
    false true [()] diagonal_search_extension.2

end Transformer.Grokking.FlatSpectrum.CoupledMoments.SharedRoles
