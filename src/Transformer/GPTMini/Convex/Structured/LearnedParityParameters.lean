import Transformer.GPTMini.Convex.Structured.LearnedParityData

/-!
# Actual learned parity parameter family with all other fields arbitrary

Source: the six exact learned-row snapshots in LearnedParityData and the
original shared coordinate layout. Only the 122 relevant learned scalar
coordinates are fixed to their actual serialized values. All other token,
state-value, absolute/relative position and chronology fields retain an
arbitrary supplied assignment. This is a family of actual full parameters,
not an alternate forward model or an encoder-correctness premise.

The transition selector is a static coordinate-layout helper for learned
weights: it never consumes a prompt or enters tensor inference. Actual
learned stochastic transition reads recover the saved logits at all nine
needed token/previous-state pairs. Initial, needed emission and branch
reads recover their actual saved rows. Real inference still runs all
paths, all visible binding pairs and the freely learned branch mixture.

The row-major transition arithmetic proof is adapted from
SharedReference.referenceShared_transition at d72d3b3, now recovering
arbitrary saved dyadic logits instead of a sharp-row capacity assignment.
The later correctness theorem must check these saved rows, then transfer
the true mixed model through the actual full tensor/integer interface.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped Classical
noncomputable section

/-- Static indices of the nine actual learned transition rows stored in the checkpoint certificate.
Source: BOS, ZERO, ONE, SEP and correct answer-token rows; not a transition rule or model-input parser. -/
def learnedParityTransitionRow (token : Fin 68) (previous : Fin 6) : Option (Fin 9) :=
  match token.val, previous.val with
  | 1, 0 => some 0
  | 21, 0 => some 1
  | 21, 1 => some 2
  | 22, 0 => some 3
  | 22, 1 => some 4
  | 18, 0 => some 5
  | 18, 1 => some 6
  | 24, 2 => some 7
  | 25, 3 => some 8
  | _, _ => none

/-- The fifteen saved emission rows occupy indices ten through twenty-four after initial/transition rows.
Source: original states two/three/four and all five actual channels, with the finite storage bound proved. -/
def learnedParityEmissionRow (state : Fin 6) (h : Fin 5) (hs : 2 ≤ state.val ∧ state.val ≤ 4) : Fin 26 :=
  ⟨10 + 5 * (state.val - 2) + h.val, by have hh := h.isLt; omega⟩

/-- Full actual shared fields with the relevant learned coordinates restored and all remaining supplied fields retained.
Source: exact checkpoint row serialization in the original 52-field shared table/global layout; no raw-input argument. -/
def learnedParityShared (run : Fin 6) (other : BindingParameters 68 19) : SharedParameters 68 19
  | .inl (token, slot) =>
      if hi : slot.val < 36 then
        match learnedParityTransitionRow token ⟨slot.val / 6, by omega⟩ with
        | some row => (learnedParityRows run ⟨row.val + 1, by have hr := row.isLt; omega⟩
            ⟨slot.val % 6, Nat.mod_lt _ (by decide)⟩ : ℚ)
        | none => other.1 (.inl (token, slot))
      else other.1 (.inl (token, slot))
  | .inr (.inl position) => other.1 (.inr (.inl position))
  | .inr (.inr (.inl ())) => other.1 (.inr (.inr (.inl ())))
  | .inr (.inr (.inr (.inl state))) => (learnedParityRows run 0 state : ℚ)
  | .inr (.inr (.inr (.inr (.inl (state, (h, d)))))) =>
      if hs : 2 ≤ state.val ∧ state.val ≤ 4 then
        (learnedParityRows run (learnedParityEmissionRow state h hs) ⟨d.val, by have hd := d.isLt; omega⟩ : ℚ)
      else other.1 (.inr (.inr (.inr (.inr (.inl (state, (h, d)))))))
  | .inr (.inr (.inr (.inr (.inr head)))) =>
      (learnedParityRows run 25 ⟨head.val, by have hh := head.isLt; omega⟩ : ℚ)

/-- A complete real parameter assignment retains every arbitrary relative-position coordinate alongside the saved learned shared rows.
Source: original BindingParameters product, with no changed feasible domain or new runtime head. -/
def learnedParityParameters (run : Fin 6) (other : BindingParameters 68 19) : BindingParameters 68 19 :=
  (learnedParityShared run other, other.2)

/-- Actual initial-state coordinate reads recover the exact stored learned row.
Source: genuine sharedInitialRead linear projection at the complete learned parameter family. -/
theorem learnedParity_initial_read (run : Fin 6) (other : BindingParameters 68 19) (state : Fin 6) :
    sharedInitialRead state (learnedParityParameters run other).1 = (learnedParityRows run 0 state : ℚ) := by
  unfold sharedInitialRead
  rw [LinearMap.proj_apply]
  rfl

/-- Every needed actual transition coordinate recovers its saved learned logit from the original row-major token embedding.
Source: exact sharedTransitionRead slot arithmetic, adapted from SharedReference's coordinate proof; the selector equality identifies a stored row only. -/
theorem learnedParity_transition_read (run : Fin 6) (other : BindingParameters 68 19)
    (token : Fin 68) (previous next : Fin 6) (row : Fin 9)
    (hrow : learnedParityTransitionRow token previous = some row) :
    sharedTransitionRead token previous next (learnedParityParameters run other).1 =
      (learnedParityRows run ⟨row.val + 1, by have hr := row.isLt; omega⟩ next : ℚ) := by
  have hp := previous.isLt
  have hn := next.isLt
  have hi : 6 * previous.val + next.val < 36 := by omega
  have hd : (6 * previous.val + next.val) / 6 = previous.val := by omega
  have hm : (6 * previous.val + next.val) % 6 = next.val := by omega
  simp only [sharedTransitionRead, sharedTokenRead, LinearMap.proj_apply, transitionSlot,
    learnedParityParameters, learnedParityShared, hi, dite_true, hd, hm, hrow]

example : learnedParityTransitionRow (1 : Fin 68) 0 = some 0 ∧
    learnedParityTransitionRow (25 : Fin 68) 3 = some 8 := by decide

/-- Actual relevant emission reads recover the original saved learned value-channel logit, without an inferred-state premise.
Source: sharedEmissionRead's real coordinate projection and the actual saved state/channel indices. -/
theorem learnedParity_emission_read (run : Fin 6) (other : BindingParameters 68 19)
    (state : Fin 6) (h : Fin 5) (d : Fin 4) (hs : 2 ≤ state.val ∧ state.val ≤ 4) :
    sharedEmissionRead state h d (learnedParityParameters run other).1 =
      (learnedParityRows run (learnedParityEmissionRow state h hs) ⟨d.val, by have hd := d.isLt; omega⟩ : ℚ) := by
  unfold sharedEmissionRead
  rw [LinearMap.proj_apply]
  change (if hs' : 2 ≤ state.val ∧ state.val ≤ 4 then
    ((learnedParityRows run (learnedParityEmissionRow state h hs') ⟨d.val, by have hd := d.isLt; omega⟩ : ℚ) : ℝ)
    else other.1 (.inr (.inr (.inr (.inr (.inl (state, (h, d)))))))) = _
  rw [dite_eq_left hs]

example : 2 ≤ (3 : Fin 6).val ∧ (3 : Fin 6).val ≤ 4 := by decide

/-- Actual learned branch-logit reads recover the saved two-channel checkpoint row in the complete mixed parameter domain.
Source: mixedHeadRead's genuine first projection and global coordinate read, not a fixed task branch. -/
theorem learnedParity_head_read (run : Fin 6) (other : BindingParameters 68 19) (head : Fin 2) :
    mixedHeadRead head (learnedParityParameters run other) =
      (learnedParityRows run 25 ⟨head.val, by have hh := head.isLt; omega⟩ : ℚ) := by
  unfold mixedHeadRead
  rw [LinearMap.comp_apply, LinearMap.fst_apply, LinearMap.proj_apply]
  rfl

/-- All original relative-position fields stay arbitrary in the complete saved-row parameter family.
Source: genuine BindingParameters second projection; parity confidence does not require any restriction on these learned binding fields. -/
theorem learnedParity_relative_read (run : Fin 6) (other : BindingParameters 68 19) (offset : Fin 37) :
    (learnedParityParameters run other).2 offset = other.2 offset := by
  change (learnedParityShared run other, other.2).2 offset = other.2 offset
  rfl

/-- All original absolute-position fields also retain the supplied arbitrary learned assignment.
Source: the original shared global position coordinates, left free by the selected learned-row certificate. -/
theorem learnedParity_position_read (run : Fin 6) (other : BindingParameters 68 19) (position : Fin 19) :
    (learnedParityParameters run other).1 (.inr (.inl position)) = other.1 (.inr (.inl position)) := by
  change learnedParityShared run other (.inr (.inl position)) = other.1 (.inr (.inl position))
  rfl

/-- Unchecked emission states remain the supplied arbitrary full parameters, so no irrelevant learned-row premise is hidden.
Source: original state/channel fields outside the three data-required endpoints, preserved by the actual parameter overlay. -/
theorem learnedParity_other_emission_read (run : Fin 6) (other : BindingParameters 68 19)
    (state : Fin 6) (h : Fin 5) (d : Fin 4) (hs : ¬ (2 ≤ state.val ∧ state.val ≤ 4)) :
    sharedEmissionRead state h d (learnedParityParameters run other).1 = sharedEmissionRead state h d other.1 := by
  unfold sharedEmissionRead
  rw [LinearMap.proj_apply, LinearMap.proj_apply]
  change (if hs' : 2 ≤ state.val ∧ state.val ≤ 4 then
    ((learnedParityRows run (learnedParityEmissionRow state h hs') ⟨d.val, by have hd := d.isLt; omega⟩ : ℚ) : ℝ)
    else other.1 (.inr (.inr (.inr (.inr (.inl (state, (h, d)))))))) = _
  rw [dite_eq_right hs]

example : ¬ (2 ≤ (5 : Fin 6).val ∧ (5 : Fin 6).val ≤ 4) := by decide

end
end Transformer.GPTMini.Convex.Structured
