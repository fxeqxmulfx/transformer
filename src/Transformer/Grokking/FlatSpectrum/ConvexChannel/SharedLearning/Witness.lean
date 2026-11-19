module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.SharedLearning.Identification

/-!
# Four observations identify a nonconstant shared rule on new inputs

Source: `review/convex/moments.md`, CM73. Three categorical slots and two classes
give twelve shared joint coordinates. Four of eight Boolean dictionaries are
observed, under only one of two visible queries. The input-only alignment swaps
the first two slots at the other query. The labels select and negate one slot;
the rule supplies a feasible proof witness, never a forward input.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.SharedLearning

open scoped BigOperators

/-- Literal categorical dictionaries; no target formula is an input feature. -/
def witnessData : Fin 4 → Fin 3 → Bool :=
  ![![false, false, false], ![false, true, true],
    ![true, false, true], ![true, true, false]]

/-- Original answer labels, used only in the empirical loss. -/
def witnessLabel : Fin 4 → Bool := ![true, true, false, false]

/-- Only the first query occurs in the four original records. -/
def witnessQuery : Fin 4 → Bool := ![false, false, false, false]

/-- A declared positional sharing prior, independent of the answer labels. -/
def witnessAlign (q : Bool) : Fin 3 ≃ Fin 3 :=
  if q then Equiv.swap 0 1 else Equiv.refl _

/-- A complete dictionary absent from the observed set. -/
def unseenData : Fin 3 → Bool := ![true, true, true]

/-- The actual training alignment is identity on each observed record, CM73. -/
theorem witness_aligned (s : Fin 4) :
    alignedData witnessAlign (witnessQuery s) (witnessData s) = witnessData s := by
  fin_cases s <;> rfl

/-- The witness's answer labels realize a nonconstant selected decoder, CM73. -/
theorem witness_realizable : ∀ s, witnessLabel s = !(witnessData s 0) := by
  intro s
  fin_cases s <;> rfl

/-- Both values of the selected slot appear in the actual observations, CM73. -/
theorem witness_coverage : Covers witnessData 0 := by
  intro v
  cases v
  · exact ⟨0, rfl⟩
  · exact ⟨2, rfl⟩

/-- Each wrong slot has a collision with opposite observed labels, CM73. -/
theorem witness_separation : SeparatesSlots witnessData witnessLabel 0 := by
  intro k hk
  fin_cases k
  · exact False.elim (hk rfl)
  · exact ⟨0, 2, rfl, by decide⟩
  · exact ⟨0, 3, rfl, by decide⟩

/-- Every premise of shared identification is satisfied by this finite
nonconstant example, including an attained optimum of the actual loss, CM73. -/
theorem witness_optimum :
    IsMinOn (fun A => answerLoss witnessQuery witnessData witnessLabel
      (tiedChannel witnessAlign A)) {A | Feasible A}
      (selected (0 : Fin 3) Bool.not) := by
  simp only [answerLoss_tiedChannel, witness_aligned]
  exact selected_isMinOn witnessData witnessLabel 0 Bool.not witness_realizable

/-- Every feasible CE optimizer, not just the displayed parameter witness,
answers all dictionaries under both queries correctly, CM73. -/
theorem witness_all_inputs {A : Channel Unit (Fin 3) Bool Bool} (hA : Feasible A)
    (hmin : IsMinOn (fun B => answerLoss witnessQuery witnessData witnessLabel
      (tiedChannel witnessAlign B)) {B | Feasible B} A)
    (q : Bool) (D : Fin 3 → Bool) (c : Bool) :
    logits (tiedChannel witnessAlign A) q D c =
      if c = !(alignedData witnessAlign q D 0) then 1 else 0 := by
  apply trained_shared_logits witnessAlign witnessQuery witnessData witnessLabel
    0 Bool.not ?_ ?_ ?_ hA hmin
  · simpa only [witness_aligned] using witness_realizable
  · simpa only [witness_aligned] using witness_coverage
  · simpa only [witness_aligned] using witness_separation

/-- The query and the complete dictionary used below are both unobserved,
CM73. This is not merely rescoring a training record. -/
theorem witness_unobserved :
    (∀ s, witnessQuery s ≠ true) ∧ (∀ s, witnessData s ≠ unseenData) := by
  decide

/-- The previously unobserved input has a strict correct raw answer for every
empirical optimizer in the declared shared domain, CM73. -/
theorem witness_unseen_answer {A : Channel Unit (Fin 3) Bool Bool} (hA : Feasible A)
    (hmin : IsMinOn (fun B => answerLoss witnessQuery witnessData witnessLabel
      (tiedChannel witnessAlign B)) {B | Feasible B} A) :
    logits (tiedChannel witnessAlign A) true unseenData true <
      logits (tiedChannel witnessAlign A) true unseenData false := by
  rw [witness_all_inputs hA hmin, witness_all_inputs hA hmin]
  norm_num [alignedData, witnessAlign, unseenData]

/-- Ordinary CE distinguishes routing at the same fixed total confidence,
CM73. Swapping only the learned routing slot strictly increases empirical loss. -/
theorem witness_loss_changes_with_routing :
    answerLoss (fun _ : Fin 4 => ()) witnessData witnessLabel
        (selected (0 : Fin 3) Bool.not) <
      answerLoss (fun _ : Fin 4 => ()) witnessData witnessLabel
        (selected (1 : Fin 3) Bool.not) := by
  apply lt_of_not_ge
  intro h
  have he := eq_selected_of_loss_le witnessData witnessLabel 0 Bool.not
    witness_realizable witness_coverage witness_separation (feasible_selected 1 Bool.not) h
  have hz := congrArg (fun A => A () 0 false true) he
  norm_num [selected, Forward.channelOf] at hz

/-- Both routing witnesses have exactly the same total logit mass on every
input, CM73; the loss improvement is not a change in a confidence scalar. -/
theorem witness_fixed_confidence (D : Fin 3 → Bool) :
    (∑ c, logits (selected (0 : Fin 3) Bool.not) () D c) = 1 ∧
    (∑ c, logits (selected (1 : Fin 3) Bool.not) () D c) = 1 := by
  exact ⟨sum_logits (feasible_selected 0 Bool.not) () D,
    sum_logits (feasible_selected 1 Bool.not) () D⟩

example := eq_selected_of_loss_le witnessData witnessLabel 0 Bool.not witness_realizable
  witness_coverage witness_separation (feasible_selected 0 Bool.not) le_rfl

example := trained_shared_logits witnessAlign witnessQuery witnessData witnessLabel 0 Bool.not
  (by simpa only [witness_aligned] using witness_realizable)
  (by simpa only [witness_aligned] using witness_coverage)
  (by simpa only [witness_aligned] using witness_separation)
  (feasible_selected 0 Bool.not) witness_optimum true unseenData false

example := trained_shared_strict witnessAlign witnessQuery witnessData witnessLabel 0 Bool.not
  (by simpa only [witness_aligned] using witness_realizable)
  (by simpa only [witness_aligned] using witness_coverage)
  (by simpa only [witness_aligned] using witness_separation)
  (feasible_selected 0 Bool.not) witness_optimum true unseenData true
  (by decide)

example := witness_all_inputs (feasible_selected 0 Bool.not) witness_optimum true unseenData false

example := witness_unseen_answer (feasible_selected 0 Bool.not) witness_optimum

end Transformer.Grokking.FlatSpectrum.ConvexChannel.SharedLearning
