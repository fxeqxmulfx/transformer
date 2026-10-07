import Transformer.GPTMini.Convex.Structured.ParityRows

/-!
# Finite learned rows derive actual raw parity path confidence

Source: Basis.Parity §no-scratchpad grammar and plan.md §4. The finite
predicate checks the actual learned initial state, nine needed transition
rows, fifteen needed value rows and learned state-branch probability.
It leaves every other row and all binding parameters unrestricted.

Raw grammar validation is used only to derive which complete data path
and final value rows need bounding. It is never an inference argument.
The real raw finite token lists are proved equal to either the original
prompt or the original correct-label-before-EOS prefix; the independently
counted endpoint is consequently two, three or four. Confidence of the
actual stochastic path follows from those finite row checks.

A genuine finite full parameter assignment satisfies the whole predicate.
This supplies real hypotheses for subsequent arbitrary-parameter tensor
correctness, without assuming a correct encoder or correct output logits.
The predicate is a property of its explicit weights and deficit bounds;
it neither contains a theorem claim nor fixes any row to a hard rule.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis
open scoped Classical
noncomputable section

/-- Finite confidence conditions on actual learned parity rows, including only reachable output endpoints.
Source: genuine learned initial/transition/emission/branch probabilities; reference labels specify checks against raw data only. -/
def ParityModelRows (θ : BindingParameters 68 19) (deltaInitial deltaTransition deltaValue deltaHead : ℝ) : Prop :=
  1 - deltaInitial ≤ stateRow (fun s => sharedInitialRead s θ.1) 0 ∧
  ParityTransitionRows (fun token state next => sharedTransitionRead token state next θ.1) deltaTransition ∧
  (∀ state : Fin 6, state = 2 ∨ state = 3 ∨ state = 4 → ∀ h,
    1 - deltaValue ≤ stateRow (fun d => sharedEmissionRead state h d θ.1)
      (outputDigit (vocabularyCode (by decide) (paritySharedLabel state)) h)) ∧
  1 - deltaHead ≤ mixedHeadWeight θ 0

/-- Every checked finite parity input is exactly an original raw prompt or original correct-label completion prefix.
Source: independently validated ParityPrefix grammar and both checked finite/raw encoding round trips. -/
theorem parityFinite_valid_forms (mode : Mode) (tokens : List (Fin 68))
    (hprefix : TaskPrefix mode .parity (decodeTokens tokens)) :
    ∃ bits : List Bool, tokens = parityFinitePrompt bits ∨ tokens = parityFiniteCompletion bits := by
  obtain ⟨bits, _, _, _, hshape⟩ := hprefix
  refine ⟨bits, ?_⟩
  rcases hshape with hp | hc
  · have he := congrArg (encodeTokens 68) hp
    rw [encode_decode, ← parityFinitePrompt_decode, encode_decode] at he
    injection he with he
    exact Or.inl he
  · have he := congrArg (encodeTokens 68) hc
    rw [encode_decode, ← parityFiniteCompletion_decode, encode_decode] at he
    injection he with he
    exact Or.inr he

example : TaskPrefix .hard .parity (decodeTokens [(1 : Fin 68), 22, 21, 22, 18, 24]) :=
  ⟨[true, false, true], by decide, by decide, by decide, Or.inr rfl⟩

/-- Actual independent data paths of valid finite inputs require only the three independently computed final value states.
Source: exact original finite token forms and counted prompt/completion endpoints, not a learned inferred endpoint premise. -/
theorem parityFinite_valid_endpoint (mode : Mode) (tokens : List (Fin 68))
    (hprefix : TaskPrefix mode .parity (decodeTokens tokens)) :
    referenceRun paritySharedRule 0 tokens = 2 ∨ referenceRun paritySharedRule 0 tokens = 3 ∨
      referenceRun paritySharedRule 0 tokens = 4 := by
  obtain ⟨bits, hp | hc⟩ := parityFinite_valid_forms mode tokens hprefix
  · rw [hp, parityFinitePrompt_run]
    by_cases h : bits.count true % 2 = 1
    · rw [ite_eq_left h]
      exact Or.inr (Or.inl rfl)
    · rw [ite_eq_right h]
      exact Or.inl rfl
  · rw [hc, parityFiniteCompletion_run]
    exact Or.inr (Or.inr rfl)

example : TaskPrefix .easy .parity (decodeTokens [(1 : Fin 68), 22, 18]) :=
  ⟨[true], by decide, by decide, by decide, Or.inl rfl⟩

/-- The nine finite learned transition checks derive actual complete path confidence for every raw valid parity prefix.
Source: original raw grammar, true selected chronological path factors and length-dependent row-deficit accumulation. -/
theorem parityFinite_valid_path (mode : Mode) (tokens : List (Fin 68))
    (hprefix : TaskPrefix mode .parity (decodeTokens tokens)) (transition : Fin 68 → Fin 6 → Fin 6 → ℝ)
    (delta : ℝ) (hrows : ParityTransitionRows transition delta) :
    1 - (tokens.length : ℝ) * delta ≤ conditionalStatePath transition 0 tokens
      (referencePath paritySharedRule 0 tokens) := by
  obtain ⟨bits, hp | hc⟩ := parityFinite_valid_forms mode tokens hprefix
  · rw [hp]
    exact parityFinitePrompt_path_lower transition delta hrows bits
  · rw [hc]
    exact parityFiniteCompletion_path_lower transition delta hrows bits

example : TaskPrefix .easy .parity (decodeTokens [(1 : Fin 68), 22, 18]) ∧
    ParityTransitionRows (sharpReferenceTable paritySharedRule 12) (6 * Real.exp (-12)) :=
  ⟨⟨[true], by decide, by decide, by decide, Or.inl rfl⟩, parityTransitionRows_sharp 12⟩

/-- The finite reachable value checks cover the actual independently labeled endpoint of every valid raw prefix.
Source: original counted endpoint semantics and the fifteen needed learned emission rows, without assuming a correct inferred endpoint. -/
theorem parityModelRows_valid_values (mode : Mode) (θ : BindingParameters 68 19)
    (deltaInitial deltaTransition deltaValue deltaHead : ℝ) (tokens : List (Fin 68))
    (hprefix : TaskPrefix mode .parity (decodeTokens tokens))
    (hrows : ParityModelRows θ deltaInitial deltaTransition deltaValue deltaHead) :
    ∀ h, 1 - deltaValue ≤ stateRow (fun d => sharedEmissionRead (referenceRun paritySharedRule 0 tokens) h d θ.1)
      (outputDigit (vocabularyCode (by decide) (paritySharedLabel (referenceRun paritySharedRule 0 tokens))) h) := by
  have hend := parityFinite_valid_endpoint mode tokens hprefix
  intro h
  exact hrows.2.2.1 _ hend h

/-- The true finite full parity witness satisfies all needed learned rows with their explicit normalized row-size tails.
Source: actual shared coordinate reads and derived mixedReference_local_rows, with no unrelated-token conditions required. -/
theorem basisParity_model_rows (mode : Mode) :
    ParityModelRows (basisMixedParameters mode .parity) (6 * Real.exp (-referenceGain))
      (6 * Real.exp (-referenceGain)) (4 * Real.exp (-referenceGain)) (2 * Real.exp (-referenceGain)) := by
  obtain ⟨hi, ht, hv, hh⟩ := mixedReference_local_rows (C := 19) paritySharedRule
    (fun state => vocabularyCode (by decide) (paritySharedLabel state)) 0 referenceGain
  refine ⟨hi, ?_, ?_, hh⟩
  · refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · exact ht 1 0
    · intro bit phase hphase
      rcases hphase with rfl | rfl
      · exact ht (if bit then 22 else 21) 0
      · exact ht (if bit then 22 else 21) 1
    · intro phase hphase
      rcases hphase with rfl | rfl
      · exact ht 18 0
      · exact ht 18 1
    · exact ht 24 2
    · exact ht 25 3
  · intro state hstate h
    rcases hstate with rfl | rfl | rfl
    · exact hv 2 h
    · exact hv 3 h
    · exact hv 4 h

example : TaskPrefix .easy .parity (decodeTokens [(1 : Fin 68), 22, 18]) ∧
    ParityModelRows (basisMixedParameters .easy .parity) (6 * Real.exp (-referenceGain))
      (6 * Real.exp (-referenceGain)) (4 * Real.exp (-referenceGain)) (2 * Real.exp (-referenceGain)) :=
  ⟨⟨[true], by decide, by decide, by decide, Or.inl rfl⟩, basisParity_model_rows .easy⟩

/-- The genuine finite parameter witness meets the complete nineteen-position sufficient error budget.
Source: actual row-size constants 2 + 6 + 19*6 + 5*4 = 142 and the independently proved real reference-gain tail. -/
theorem basisParity_rows_budget : (0 : ℝ) ≤ 6 * Real.exp (-referenceGain) ∧
    2 * Real.exp (-referenceGain) + 6 * Real.exp (-referenceGain) +
      19 * (6 * Real.exp (-referenceGain)) + 5 * (4 * Real.exp (-referenceGain)) < (1 / 11 : ℝ) := by
  have he := Real.exp_pos (-referenceGain)
  have hb := mixedReferenceGain_confidence
  constructor <;> nlinarith

end
end Transformer.GPTMini.Convex.Structured
