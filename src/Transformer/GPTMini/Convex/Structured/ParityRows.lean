import Transformer.GPTMini.Convex.Structured.ParityDataList

/-!
# Only the nine transitions required by the true parity grammar

Source: plan.md §4 and ParityReference's complete raw counted semantics.
The finite conditions read actual unrestricted learned softmax rows:
BOS at start zero, four bit/phase cases, two SEP/phase cases, and the
two correct answer/completion cases. Rows for invalid tokens/phases
are unrestricted. No deterministic reference state enters inference.

These nine conditions derive confidence for the actual selected full
chronological learned path of every raw prompt and correct answer input,
with error proportional to its true physical length. Correct output
channels, learned mixing and real tensor/integer transfer follow later.
The finite sharp witness satisfies the conditions, while arbitrary
learned checkpoints still need their actual row bounds checked.
The predicate bounds normalized probabilities, rather than fixing whole
logit rows or their normalizing constants. Output values and mixing
remain free and are checked independently. The selected path is a true
configuration of the same stochastic model; its confidence is derived
here without an assumed correctly inferred endpoint.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis
open scoped Classical
noncomputable section

/-- Predicate on nine actual learned transition rows, restricted to the independently valid raw phases.
Source: BOS/bit/SEP/answer transitions in Basis.Parity and ParityReference; no unrelated vocabulary row is constrained. -/
def ParityTransitionRows (transition : Fin 68 → Fin 6 → Fin 6 → ℝ) (delta : ℝ) : Prop :=
  1 - delta ≤ stateRow (transition 1 0) 0 ∧
  (∀ bit : Bool, ∀ phase : Fin 6, phase = 0 ∨ phase = 1 →
    1 - delta ≤ stateRow (transition (if bit then 22 else 21) phase)
      (paritySharedRule (if bit then 22 else 21) phase)) ∧
  (∀ phase : Fin 6, phase = 0 ∨ phase = 1 →
    1 - delta ≤ stateRow (transition 18 phase) (paritySharedRule 18 phase)) ∧
  1 - delta ≤ stateRow (transition 24 2) 4 ∧ 1 - delta ≤ stateRow (transition 25 3) 4

/-- The concrete finite true-logit assignment satisfies the entire restricted nine-row predicate.
Source: actual sharpReferenceTable row reads and stateRow_sharp's derived normalized probabilities, not a hard transition replacement. -/
theorem parityTransitionRows_sharp (gain : ℝ) :
    ParityTransitionRows (sharpReferenceTable paritySharedRule gain) (6 * Real.exp (-gain)) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · change 1 - 6 * Real.exp (-gain) ≤ stateRow (sharpRowLogits (0 : Fin 6) gain) 0
    exact stateRow_sharp (0 : Fin 6) gain
  · rintro bit phase (rfl | rfl)
    · exact stateRow_sharp (paritySharedRule (if bit then 22 else 21) 0) gain
    · exact stateRow_sharp (paritySharedRule (if bit then 22 else 21) 1) gain
  · rintro phase (rfl | rfl)
    · exact stateRow_sharp (paritySharedRule 18 0) gain
    · exact stateRow_sharp (paritySharedRule 18 1) gain
  · change 1 - 6 * Real.exp (-gain) ≤ stateRow (sharpRowLogits (4 : Fin 6) gain) 4
    exact stateRow_sharp (4 : Fin 6) gain
  · change 1 - 6 * Real.exp (-gain) ≤ stateRow (sharpRowLogits (4 : Fin 6) gain) 4
    exact stateRow_sharp (4 : Fin 6) gain

/-- The four actual bit rows suffice for every genuine counted bit-path factor, from either counted bit phase.
Source: independent finite bit-step semantics and the true chronological learned path, with a linear actual-length error budget. -/
theorem parityFiniteBits_path_lower (transition : Fin 68 → Fin 6 → Fin 6 → ℝ) (delta : ℝ)
    (hrows : ParityTransitionRows transition delta) (count : ℕ) (bits : List Bool) :
    1 - (bits.length : ℝ) * delta ≤ conditionalStatePath transition (parityReferencePhase count)
      (parityFiniteBits bits) (referencePath paritySharedRule (parityReferencePhase count) (parityFiniteBits bits)) := by
  induction bits generalizing count with
  | nil =>
      simp only [parityFiniteBits, List.map, List.length_nil, conditionalStatePath,
        Nat.cast_zero, zero_mul, sub_zero]
      norm_num
  | cons bit bits ih =>
      have hphase : parityReferencePhase count = 0 ∨ parityReferencePhase count = 1 := by
        unfold parityReferencePhase
        split_ifs
        · exact Or.inr rfl
        · exact Or.inl rfl
      have hp := hrows.2.1 bit (parityReferencePhase count) hphase
      have htail := ih (count + if bit then 1 else 0)
      have htailu := conditionalStatePath_le_one transition (parityReferencePhase (count + if bit then 1 else 0))
        (parityFiniteBits bits) (referencePath paritySharedRule (parityReferencePhase (count + if bit then 1 else 0)) (parityFiniteBits bits))
      rw [parityFiniteBits_step] at hp
      have hm := probability_mul_lower (stateRow (transition (if bit then 22 else 21) (parityReferencePhase count))
        (parityReferencePhase (count + if bit then 1 else 0))) _ delta ((bits.length : ℝ) * delta)
        (stateRow_le_one _ _) htailu hp htail
      change 1 - (((bits.length + 1 : ℕ) : ℝ) * delta) ≤
        stateRow (transition (if bit then 22 else 21) (parityReferencePhase count))
          (paritySharedRule (if bit then 22 else 21) (parityReferencePhase count)) *
        conditionalStatePath transition (paritySharedRule (if bit then 22 else 21) (parityReferencePhase count))
          (parityFiniteBits bits) (referencePath paritySharedRule
            (paritySharedRule (if bit then 22 else 21) (parityReferencePhase count)) (parityFiniteBits bits))
      rw [parityFiniteBits_step, Nat.cast_add, Nat.cast_one]
      linarith

example : ParityTransitionRows (sharpReferenceTable paritySharedRule 12) (6 * Real.exp (-12)) :=
  parityTransitionRows_sharp 12

/-- The required BOS, four bit and two SEP rows bound every actual full prompt path, retaining the original delimiter.
Source: genuine concatenated-path factorization and the independently derived finite bit endpoint, without assuming a correct inferred state. -/
theorem parityFinitePrompt_path_lower (transition : Fin 68 → Fin 6 → Fin 6 → ℝ) (delta : ℝ)
    (hrows : ParityTransitionRows transition delta) (bits : List Bool) :
    1 - ((parityFinitePrompt bits).length : ℝ) * delta ≤ conditionalStatePath transition 0
      (parityFinitePrompt bits) (referencePath paritySharedRule 0 (parityFinitePrompt bits)) := by
  have hz : parityReferencePhase 0 = (0 : Fin 6) := by decide
  have hrun : referenceRun paritySharedRule 0 (parityFiniteBits bits) = parityReferencePhase (bits.count true) := by
    simpa only [hz, Nat.zero_add] using parityFiniteBits_run 0 bits
  have hphase : parityReferencePhase (bits.count true) = 0 ∨ parityReferencePhase (bits.count true) = 1 := by
    unfold parityReferencePhase
    split_ifs
    · exact Or.inr rfl
    · exact Or.inl rfl
  have hs : 1 - delta ≤ conditionalStatePath transition (parityReferencePhase (bits.count true)) [18]
      (referencePath paritySharedRule (parityReferencePhase (bits.count true)) [18]) := by
    change 1 - delta ≤ stateRow (transition 18 (parityReferencePhase (bits.count true)))
      (paritySharedRule 18 (parityReferencePhase (bits.count true))) * 1
    simpa only [mul_one] using hrows.2.2.1 _ hphase
  have hb := parityFiniteBits_path_lower transition delta hrows 0 bits
  rw [hz] at hb
  have ht : 1 - ((bits.length : ℝ) * delta + delta) ≤ conditionalStatePath transition 0
      (parityFiniteBits bits ++ [18]) (referencePath paritySharedRule 0 (parityFiniteBits bits ++ [18])) := by
    rw [referencePath_probability_append, hrun]
    exact probability_mul_lower _ _ _ _ (conditionalStatePath_le_one _ _ _ _) (conditionalStatePath_le_one _ _ _ _) hb hs
  have hm := probability_mul_lower (stateRow (transition 1 0) 0) _ delta
    ((bits.length : ℝ) * delta + delta) (stateRow_le_one _ _)
    (conditionalStatePath_le_one transition 0 (parityFiniteBits bits ++ [18]) _) hrows.1 ht
  change 1 - ((parityFinitePrompt bits).length : ℝ) * delta ≤ stateRow (transition 1 0) 0 *
    conditionalStatePath transition 0 (parityFiniteBits bits ++ [18]) (referencePath paritySharedRule 0 (parityFiniteBits bits ++ [18]))
  have hlen : (parityFinitePrompt bits).length = bits.length + 2 := by
    simp only [parityFinitePrompt, parityFiniteBits, List.length_cons, List.length_append, List.length_map, List.length_nil]
  rw [hlen, Nat.cast_add]
  norm_num only [Nat.cast_ofNat]
  nlinarith

example : ParityTransitionRows (sharpReferenceTable paritySharedRule 12) (6 * Real.exp (-12)) :=
  parityTransitionRows_sharp 12

/-- Both required answer rows also certify the actual second prefix's full path to completion.
Source: original correct counted label, independently proved label phase, and real learned path composition through that boundary. -/
theorem parityFiniteCompletion_path_lower (transition : Fin 68 → Fin 6 → Fin 6 → ℝ) (delta : ℝ)
    (hrows : ParityTransitionRows transition delta) (bits : List Bool) :
    1 - ((parityFiniteCompletion bits).length : ℝ) * delta ≤ conditionalStatePath transition 0
      (parityFiniteCompletion bits) (referencePath paritySharedRule 0 (parityFiniteCompletion bits)) := by
  have hl : 1 - delta ≤ conditionalStatePath transition (referenceRun paritySharedRule 0 (parityFinitePrompt bits))
      [parityFiniteLabel bits] (referencePath paritySharedRule (referenceRun paritySharedRule 0 (parityFinitePrompt bits)) [parityFiniteLabel bits]) := by
    rw [parityFinitePrompt_run]
    unfold parityFiniteLabel
    by_cases h : bits.count true % 2 = 1
    · rw [ite_eq_left h, ite_eq_left h]
      change 1 - delta ≤ stateRow (transition 25 3) 4 * 1
      simpa only [mul_one] using hrows.2.2.2.2
    · rw [ite_eq_right h, ite_eq_right h]
      change 1 - delta ≤ stateRow (transition 24 2) 4 * 1
      simpa only [mul_one] using hrows.2.2.2.1
  have hp := parityFinitePrompt_path_lower transition delta hrows bits
  have hm := probability_mul_lower _ _ (((parityFinitePrompt bits).length : ℝ) * delta) delta
    (conditionalStatePath_le_one _ _ _ _) (conditionalStatePath_le_one _ _ _ _) hp hl
  unfold parityFiniteCompletion
  rw [referencePath_probability_append]
  simp only [List.length_append, List.length_cons, List.length_nil, Nat.cast_add, Nat.cast_one, Nat.cast_zero]
  nlinarith

example : ParityTransitionRows (sharpReferenceTable paritySharedRule 12) (6 * Real.exp (-12)) :=
  parityTransitionRows_sharp 12

end
end Transformer.GPTMini.Convex.Structured
