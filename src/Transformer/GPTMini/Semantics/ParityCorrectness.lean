import Transformer.GPTMini.Semantics.ParityReadout

/-!
# Complete original GPTMini parity correctness through List Int -> List Int

Source: Bits.prompt/Parity.solve and Basis's 1..16-bit recipes at cbafbe9,
the archived GPTMini.forward at f11b6e2, and the exact operator/margin
proofs in ParityConstruction through ParityReadout. The given ordinary
two-layer, 64-wide parameter assignment emits the correct label and then
EOS for every validated raw prefix, through the actual checked greedy
adapter. No correct logits, prepared semantic values or task-solving
encoder is an input to this proof. The result holds for both Basis modes.

This verifies the existing real-arithmetic/shared-epsilon formal model;
it makes no claim about floating-point rounding, unequal Python default
epsilons, AdamW finding these weights, or full depth/recall correctness.
Universal correctness is stronger than the sampled 99-percent criterion.
-/

namespace Transformer.GPTMini.Semantics

open TokenInterface Transformer.Basis

/-- Actual last-row greedy logits select the correct label on every bounded raw prompt.
Source: the full original forward's derived strict margin and the actual finite decoder. -/
theorem decoder_prompt_best (eps : ℝ) (heps : 0 < eps) (hsmall : eps ≤ 1 / 64)
    (bits : List Bool) (hlen : bits.length ≤ 16) :
    bestToken countConfig.vocab_pos (lastLogits countConfig (decoderParams 1024 131072) eps
      ⟨1, by decide⟩ (finiteBitTokens bits ++ [⟨18, by decide⟩])) = finiteParityLabel bits := by
  apply bestToken_of_strict
  intro token hne
  simpa only [lastLogits, finiteParityPrompt, finiteBitTokens, List.length_append,
    List.length_map, List.length_singleton] using decoder_prompt_logits eps heps hsmall bits hlen token hne

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 / 64 ∧
    [false].length ≤ 16 := by norm_num

/-- Actual greedy logits select EOS after every bounded supplied correct label.
Source: the complete answer-state computation, simultaneous EOS margin and finite-vocabulary decoder. -/
theorem decoder_answer_best (eps : ℝ) (heps : 0 < eps) (hsmall : eps ≤ 1 / 64)
    (bits : List Bool) (hlen : bits.length ≤ 16) :
    bestToken countConfig.vocab_pos (lastLogits countConfig (decoderParams 1024 131072) eps
      ⟨1, by decide⟩ ((finiteBitTokens bits ++ [⟨18, by decide⟩]) ++ [finiteParityLabel bits])) =
        ⟨17, by decide⟩ := by
  apply bestToken_of_strict
  intro token hne
  simpa only [lastLogits, finiteParityPrompt, finiteBitTokens, List.cons_append,
    List.length_append, List.length_map, List.length_singleton, Nat.add_assoc] using
    decoder_answer_logits eps heps hsmall bits hlen token hne

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 / 64 ∧
    [true, false].length ≤ 16 := by norm_num

/-- The checked public integer function preserves the raw prompt and appends its computed label.
Source: actual encode/decode round trip, actual context cap and the proved model logit decoder. -/
theorem decoder_prompt_tokens (eps : ℝ) (heps : 0 < eps) (hsmall : eps ≤ 1 / 64)
    (bits : List Bool) (hlen : bits.length ≤ 16) :
    tokenFunction countConfig (decoderParams 1024 131072) eps (parityPrompt bits) =
      parityPrompt bits ++ [parityLabel bits] := by
  have hcap : (finiteBitTokens bits ++ [(⟨18, by decide⟩ : Fin countConfig.vocab_size)]).length + 1 ≤
      countConfig.max_seq_len := by
    simp only [finiteBitTokens, List.length_append, List.length_map, List.length_singleton]
    omega
  have h := tokenFunction_decode_cons countConfig (decoderParams 1024 131072) eps
    ⟨1, by decide⟩ (finiteBitTokens bits ++ [⟨18, by decide⟩]) hcap
  have hd := finiteParityPrompt_decode bits
  dsimp only [finiteParityPrompt] at hd
  rw [hd, decoder_prompt_best eps heps hsmall bits hlen, finiteParityLabel_decode] at h
  exact h

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 / 64 ∧
    [true].length ≤ 16 := by norm_num

/-- The checked integer function appends EOS while retaining the supplied label and all input tokens.
Source: the full second supervision position, with all nineteen context slots allowed. -/
theorem decoder_answer_tokens (eps : ℝ) (heps : 0 < eps) (hsmall : eps ≤ 1 / 64)
    (bits : List Bool) (hlen : bits.length ≤ 16) :
    tokenFunction countConfig (decoderParams 1024 131072) eps
        (parityPrompt bits ++ [parityLabel bits]) =
      parityPrompt bits ++ [parityLabel bits, eos] := by
  have hcap : ((finiteBitTokens bits ++ [(⟨18, by decide⟩ : Fin countConfig.vocab_size)]) ++
      [finiteParityLabel bits]).length + 1 ≤
      countConfig.max_seq_len := by
    simp only [finiteBitTokens, List.length_append, List.length_map, List.length_singleton]
    omega
  have h := tokenFunction_decode_cons countConfig (decoderParams 1024 131072) eps
    ⟨1, by decide⟩ ((finiteBitTokens bits ++ [⟨18, by decide⟩]) ++ [finiteParityLabel bits]) hcap
  have hd : decodeTokens (⟨1, by decide⟩ ::
      ((finiteBitTokens bits ++ [⟨18, by decide⟩]) ++ [finiteParityLabel bits])) =
      parityPrompt bits ++ [parityLabel bits] := by
    simpa only [finiteParityPrompt, List.cons_append] using finiteParityAnswer_decode bits
  rw [hd, decoder_answer_best eps heps hsmall bits hlen] at h
  change tokenFunction countConfig (decoderParams 1024 131072) eps
    (parityPrompt bits ++ [parityLabel bits]) = (parityPrompt bits ++ [parityLabel bits]) ++ [eos] at h
  simpa only [List.append_assoc, List.cons_append, List.nil_append] using h

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 / 64 ∧
    [false, false].length ≤ 16 := by norm_num

/-- The given original small softmax GPTMini solves the complete raw parity grammar in either Basis mode.
Source: both actual supervised-position predictions above, not existence of weights or assumed correct logits. -/
theorem decoder_solves_parity (eps : ℝ) (heps : 0 < eps) (hsmall : eps ≤ 1 / 64)
    (mode : Mode) :
    SolvesTask (tokenFunction countConfig (decoderParams 1024 131072) eps) mode .parity := by
  intro tokens hprefix
  obtain ⟨bits, hmin, hmax, _, hform⟩ := hprefix
  have hbits : bits ≠ [] := by
    intro h
    simp [h] at hmin
  change tokenFunction countConfig (decoderParams 1024 131072) eps tokens = parityFunction tokens
  rcases hform with rfl | rfl
  · rw [decoder_prompt_tokens eps heps hsmall bits hmax, parityFunction_prompt bits hbits]
  · rw [decoder_answer_tokens eps heps hsmall bits hmax, parityFunction_answer bits hbits]

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 / 64 := by norm_num

/-- Two ordinary calls freely generate the correct label and EOS for every benchmark word.
Source: the actual integer model function and the complete SolvesTask parity theorem. -/
theorem decoder_parity_twice (eps : ℝ) (heps : 0 < eps) (hsmall : eps ≤ 1 / 64)
    (bits : List Bool) (hmin : 1 ≤ bits.length) (hmax : bits.length ≤ 16) :
    let f := tokenFunction countConfig (decoderParams 1024 131072) eps
    f (f (parityPrompt bits)) = parityPrompt bits ++ [parityLabel bits, eos] :=
  solvesTask_parity_twice _ .easy (decoder_solves_parity eps heps hsmall .easy) bits hmin hmax

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 / 64 ∧
    1 ≤ ([true, false] : List Bool).length ∧ ([true, false] : List Bool).length ≤ 16 := by norm_num

/-- The reference RMS epsilon gives a concrete complete parity solution in both recipes.
Source: RMSNorm(eps=1e-5) at f11b6e2; the formal head epsilon is still shared as documented above. -/
theorem decoder_reference_rms_solves_parity (mode : Mode) :
    SolvesTask (tokenFunction countConfig (decoderParams 1024 131072) (1 / 100000)) mode .parity :=
  decoder_solves_parity _ (by norm_num) (by norm_num) mode

/-- The maximal-count boundary freely generates EVEN then EOS using the original nineteen-slot context.
Source: the benchmark's sixteen-ONE word, including the answer token at the final permitted input position. -/
theorem decoder_sixteen_one_boundary :
    let f := tokenFunction countConfig (decoderParams 1024 131072) (1 / 100000)
    f (f (parityPrompt (List.replicate 16 true))) =
      parityPrompt (List.replicate 16 true) ++ [evenToken, eos] := by
  have h := decoder_parity_twice (1 / 100000) (by norm_num) (by norm_num)
    (List.replicate 16 true) (by decide) (by decide)
  have hl : parityLabel (List.replicate 16 true) = evenToken := by decide
  simpa only [hl] using h

/-!
Both calls above use the same weights. The first predicted token, rather
than an external teacher label, becomes the second input. All permitted
bit words are covered by decoder_solves_parity; this boundary check also
records the maximum count and the maximum second-position context size.
-/

end Transformer.GPTMini.Semantics
