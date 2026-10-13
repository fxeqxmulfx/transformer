import Transformer.GPTMini.Convex.Structured.LearnedParityRows

/-!
# Actual learned full parity weights satisfy sufficient raw logit gaps

Source: kernel-checked exact dyadic checkpoint rows and the original
shared parameter reads. This derives all needed raw logit inequalities
for the actual learned parameter family, leaving every other coordinate
arbitrary. Row selector and target equalities are static data-layout
facts, independently checked against the original raw parity vocabulary.
They are not correctly inferred states, correct logits or correct
encoder premises for a full-model theorem.

All nine physical transitions and all fifteen actual required value
channels meet gap eleven. Actual initial and branch inequalities were
already recovered from the same learned snapshots. The full real
probabilistic model must subsequently use those facts through its true
path/channel/branch likelihood and checked tensor/integer transfer.
No learned matrix is replaced by a hard transition in inference.
The stored coordinates are from completed ordinary-AdamW runs; this
result does not supply an optimizer-convergence or IEEE arithmetic proof.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis
open scoped Classical
noncomputable section

/-- A needed real transition row has the sufficient raw gap, from the genuine saved lookup and independently correct data-index mapping.
Source: actual row-major learned read, kernel-checked rational rivals and original parity transition target; remaining parameters are arbitrary. -/
theorem learnedParity_transition_gap (run : Fin 6) (other : BindingParameters 68 19)
    (token : Fin 68) (previous : Fin 6) (row : Fin 9)
    (hrow : learnedParityTransitionRow token previous = some row)
    (htarget : learnedParityRowTarget ⟨row.val + 1, by have hr := row.isLt; omega⟩ = paritySharedRule token previous) :
    RowLogitGap (fun next => sharedTransitionRead token previous next (learnedParityParameters run other).1)
      (paritySharedRule token previous) 11 := by
  intro rival hr
  simp_rw [learnedParity_transition_read run other token previous _ row hrow]
  have hri := row.isLt
  have hsmall : row.val + 1 < 10 := by omega
  have hw : rival.val < learnedParityRowWidth ⟨row.val + 1, by omega⟩ := by
    unfold learnedParityRowWidth
    rw [ite_eq_left hsmall]
    exact rival.isLt
  have hg := learnedParity_rows_real_gap run ⟨row.val + 1, by omega⟩ rival hw
    (by rw [htarget]; exact hr)
  rw [htarget] at hg
  exact hg

example : learnedParityTransitionRow (1 : Fin 68) 0 = some 0 ∧
    learnedParityRowTarget 1 = paritySharedRule 1 0 := by decide

/-- All nine actual needed learned transition rows meet the sufficient real gap at every saved run and arbitrary remaining fields.
Source: exact static checkpoint row mapping and independently checked BOS/bit/SEP/answer targets, without an assumed correctly inferred state. -/
theorem learnedParity_transition_gaps (run : Fin 6) (other : BindingParameters 68 19) :
    ParityTransitionLogitGaps (fun token state next => sharedTransitionRead token state next
      (learnedParityParameters run other).1) 11 := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · exact learnedParity_transition_gap run other 1 0 0 (by decide) (by decide)
  · rintro bit phase (rfl | rfl)
    · cases bit with
      | false => exact learnedParity_transition_gap run other 21 0 1 (by decide) (by decide)
      | true => exact learnedParity_transition_gap run other 22 0 3 (by decide) (by decide)
    · cases bit with
      | false => exact learnedParity_transition_gap run other 21 1 2 (by decide) (by decide)
      | true => exact learnedParity_transition_gap run other 22 1 4 (by decide) (by decide)
  · rintro phase (rfl | rfl)
    · exact learnedParity_transition_gap run other 18 0 5 (by decide) (by decide)
    · exact learnedParity_transition_gap run other 18 1 6 (by decide) (by decide)
  · exact learnedParity_transition_gap run other 24 2 7 (by decide) (by decide)
  · exact learnedParity_transition_gap run other 25 3 8 (by decide) (by decide)

/-- Every required actual learned value-channel row meets the sufficient gap against all four genuine output rivals.
Source: exact emission-coordinate recovery, independent raw EVEN/ODD/EOS digit identities and kernel-checked rational channel inequalities. -/
theorem learnedParity_emission_gap (run : Fin 6) (other : BindingParameters 68 19)
    (state : Fin 6) (h : Fin 5) (hs : 2 ≤ state.val ∧ state.val ≤ 4) :
    RowLogitGap (fun d => sharedEmissionRead state h d (learnedParityParameters run other).1)
      (outputDigit (vocabularyCode (by decide) (paritySharedLabel state)) h) 11 := by
  intro rival hr
  let r : Fin 6 := ⟨rival.val, by have hd := rival.isLt; omega⟩
  have hw : r.val < learnedParityRowWidth (learnedParityEmissionRow state h hs) := by
    rw [learnedParity_emission_width state h hs]
    exact rival.isLt
  have hn : r ≠ learnedParityRowTarget (learnedParityEmissionRow state h hs) := by
    rw [learnedParity_emission_target state h hs]
    intro he
    have hv := congrArg (fun x : Fin 6 => x.val) he
    apply hr
    apply Fin.ext
    exact hv
  have hg := learnedParity_rows_real_gap run (learnedParityEmissionRow state h hs) r hw hn
  rw [learnedParity_emission_target state h hs] at hg
  simp_rw [learnedParity_emission_read run other state h _ hs]
  exact hg

example : 2 ≤ (3 : Fin 6).val ∧ (3 : Fin 6).val ≤ 4 := by decide

/-- The complete real learned parameter family satisfies every sufficient initial/needed-transition/value/branch gap condition.
Source: actual saved checkpoint coordinates and kernel-checked rational inequalities; no desired output or correctly encoded prefix is a premise. -/
theorem learnedParity_model_logit_gaps (run : Fin 6) (other : BindingParameters 68 19) :
    ParityModelLogitGaps (learnedParityParameters run other) 11 := by
  refine ⟨learnedParity_initial_gap run other, learnedParity_transition_gaps run other, ?_,
    learnedParity_branch_gap run other⟩
  intro state hstate h
  have hs : 2 ≤ state.val ∧ state.val ≤ 4 := by
    rcases hstate with rfl | rfl | rfl <;> decide
  exact learnedParity_emission_gap run other state h hs

/-- The same actual learned rows therefore satisfy the real normalized confidence conditions used by the complete model.
Source: true learned softmax gap tails and the entire sufficient raw-coordinate certificate, not a numerical probability estimate. -/
theorem learnedParity_model_rows (run : Fin 6) (other : BindingParameters 68 19) :
    ParityModelRows (learnedParityParameters run other) (6 * Real.exp (-11)) (6 * Real.exp (-11))
      (4 * Real.exp (-11)) (2 * Real.exp (-11)) := by
  exact parityModelLogitGaps_rows _ 11 (learnedParity_model_logit_gaps run other)

/-- The actual shared initial confidence follows for every completed run independently of all unused pointer/position fields.
Source: normalized complete-model row certificate derived from exact learned coordinates. -/
theorem learnedParity_initial_confidence (run : Fin 6) (other : BindingParameters 68 19) :
    1 - 6 * Real.exp (-11) ≤ stateRow (fun state => sharedInitialRead state (learnedParityParameters run other).1) 0 := by
  exact (learnedParity_model_rows run other).1

/-- The actual learned branch confidence also follows from the same certified saved coordinates without forcing a task switch.
Source: complete-model row certificate and the genuine positive normalized learned two-head mixture. -/
theorem learnedParity_branch_confidence (run : Fin 6) (other : BindingParameters 68 19) :
    1 - 2 * Real.exp (-11) ≤ mixedHeadWeight (learnedParityParameters run other) 0 := by
  exact (learnedParity_model_rows run other).2.2.2

/-- The actual selected learned chronological path of every valid raw parity prefix has the derived saved-row confidence.
Source: independently valid original raw grammar, the nine certified learned rows and genuine stochastic chronological factors. -/
theorem learnedParity_path_confidence (mode : Mode) (run : Fin 6) (other : BindingParameters 68 19)
    (tokens : List (Fin 68)) (hprefix : TaskPrefix mode .parity (decodeTokens tokens)) :
    1 - (tokens.length : ℝ) * (6 * Real.exp (-11)) ≤ conditionalStatePath
      (fun token state next => sharedTransitionRead token state next (learnedParityParameters run other).1)
      0 tokens (referencePath paritySharedRule 0 tokens) := by
  exact parityFinite_valid_path mode tokens hprefix _ _ (learnedParity_model_rows run other).2.1

example : TaskPrefix .hard .parity (decodeTokens [(1 : Fin 68), 22, 18]) :=
  ⟨[true], by decide, by decide, by decide, Or.inl rfl⟩

/-- Every needed true raw endpoint value row also has its derived learned normalized confidence.
Source: independent original counted endpoint and the fifteen certified actual saved value rows; no inferred endpoint is assumed correct. -/
theorem learnedParity_endpoint_values (mode : Mode) (run : Fin 6) (other : BindingParameters 68 19)
    (tokens : List (Fin 68)) (hprefix : TaskPrefix mode .parity (decodeTokens tokens)) :
    ∀ h, 1 - 4 * Real.exp (-11) ≤ stateRow (fun d =>
      sharedEmissionRead (referenceRun paritySharedRule 0 tokens) h d (learnedParityParameters run other).1)
      (outputDigit (vocabularyCode (by decide) (paritySharedLabel (referenceRun paritySharedRule 0 tokens))) h) := by
  exact parityModelRows_valid_values mode _ _ _ _ _ tokens hprefix (learnedParity_model_rows run other)

example : TaskPrefix .easy .parity (decodeTokens [(1 : Fin 68), 22, 18, 25]) :=
  ⟨[true], by decide, by decide, by decide, Or.inr rfl⟩

end
end Transformer.GPTMini.Convex.Structured
