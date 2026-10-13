import Transformer.GPTMini.Semantics.ParityConstruction

/-!
# Exact raw parity inputs to the original decoder FFN

Source: Bits.prompt/Parity.solve at cbafbe9 and the complete original
parameter family in ParityConstruction. The first FFN input is defined
by actual embedding and attention operators. Its raw-token formulas below
are derived from the proved simultaneous ONE/BOS head, including its real
prenorm multiplier. No prepared semantic value array is a hypothesis.

Prompt and supplied-answer states differ in completion phase and in the
tied label code. The normalized count retains the same integer quotient
at both positions. The full two-layer hidden state is connected to this
actual FFN input and the actual ReLU2 matrices. Proving that its final
greedy tied readout emits the required token still requires quantitative
margin bounds; this module does not assume correct logits.
-/

namespace Transformer.GPTMini.Semantics

open TokenInterface Transformer.Basis

/-- The real residual input to the active first FFN, using the proved equality of its projected embeddings.
Source: original blockForward; every argument enters the actual causal attention calculation. -/
noncomputable def decoderInput (eps : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin countConfig.vocab_size) (i : Fin T) : EucSpace 64 :=
  decoderEmbedding (tokens i) +
    attnSubLayer countConfig (ratioParams.blocks ⟨0, by decide⟩).attn eps positions
      (embed countConfig ratioParams tokens) i

/-- The complete actual two-layer state is this residual input plus its true prenorm/FFN output.
Source: the proved original-model unrolling and the inactive second block, not an existence argument. -/
theorem decoder_full_state (parityScale eosScale eps : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin countConfig.vocab_size) (i : Fin T) :
    hidden countConfig (decoderParams parityScale eosScale) eps positions tokens 2 i =
      decoderInput eps positions tokens i +
        relu2FFN completionFFNIn (completionFFNOut parityScale eosScale)
          (rmsNormEps eps (decoderInput eps positions tokens i)) := by
  unfold decoderInput
  rw [decoder_hidden_two, decoder_hidden_one]

/-- The actual pre-FFN residual contains raw ONE/BOS counts together with its new tied embedding code.
Source: the original first attention calculation and ratio_block_state, for every final zero-value token. -/
theorem decoderInput_raw (eps : ℝ) (head last : Fin countConfig.vocab_size)
    (body : List (Fin countConfig.vocab_size)) (hlast : last.val ≠ 22 ∧ last.val ≠ 1) :
    decoderInput eps (fun j => (j.val : ℝ)) (head :: (body ++ [last])).get
        ⟨body.length + 1, by simp⟩ =
      decoderEmbedding last +
        ((((decodeTokens (head :: (body ++ [last]))).count oneBit : ℝ) /
          (body.length + 2 : ℕ) * countScale eps) • controlUnit +
        (((decodeTokens (head :: (body ++ [last]))).count bos : ℝ) /
          (body.length + 2 : ℕ) * countScale eps) • bosUnit) := by
  have hlastToken : (head :: (body ++ [last])).get ⟨body.length + 1, by simp⟩ = last := by
    simp only [List.get_eq_getElem, List.getElem_cons_succ,
      List.getElem_append_right (le_refl body.length), Nat.sub_self, List.getElem_cons_zero]
  have hs := ratio_block_state eps head last body hlast
  rw [ratio_first_state, hlastToken] at hs
  change ratioEmbedding last + _ = ratioEmbedding last + _ at hs
  have ha := add_left_cancel hs
  unfold decoderInput
  rw [hlastToken]
  exact congrArg (fun update => decoderEmbedding last + update) ha

example : (⟨18, by decide⟩ : Fin countConfig.vocab_size).val ≠ 22 ∧
    (⟨18, by decide⟩ : Fin countConfig.vocab_size).val ≠ 1 := by decide

/-- Every raw parity prompt supplies the exact count/BOS representation and positive completion phase.
Source: Bits.prompt, with its unique BOS, actual bits and final SEP. -/
theorem decoderInput_prompt (eps : ℝ) (bits : List Bool) :
    decoderInput eps (fun j => (j.val : ℝ)) (finiteParityPrompt bits).get
        ⟨bits.length + 1, by simp [finiteParityPrompt, finiteBitTokens]⟩ =
      phaseUnit +
        (((bits.count true : ℝ) / (bits.length + 2 : ℕ) * countScale eps) • controlUnit +
          ((1 : ℝ) / (bits.length + 2 : ℕ) * countScale eps) • bosUnit) := by
  have h := decoderInput_raw eps ⟨1, by decide⟩ ⟨18, by decide⟩
    (finiteBitTokens bits) (by decide)
  have hd := finiteParityPrompt_decode bits
  dsimp only [finiteParityPrompt] at hd
  rw [hd, parityPrompt_one_count, parityPrompt_bos_count] at h
  rw [show decoderEmbedding ⟨18, by decide⟩ = phaseUnit from rfl] at h
  simpa only [finiteParityPrompt, finiteBitTokens, List.length_map, Nat.cast_one] using h

/-- The supplied correct label preserves its sign code and negative phase alongside the same integer count.
Source: the second Parity.solve position, including SEP and the actual supplied answer as input tokens. -/
theorem decoderInput_answer (eps : ℝ) (bits : List Bool) :
    decoderInput eps (fun j => (j.val : ℝ))
        (finiteParityPrompt bits ++ [finiteParityLabel bits]).get
        ⟨bits.length + 2, by simp [finiteParityPrompt, finiteBitTokens]⟩ =
      (-phaseUnit + countSign (bits.count true) • parityUnit) +
        (((bits.count true : ℝ) / (bits.length + 3 : ℕ) * countScale eps) • controlUnit +
          ((1 : ℝ) / (bits.length + 3 : ℕ) * countScale eps) • bosUnit) := by
  have hform : ⟨1, by decide⟩ ::
      ((finiteBitTokens bits ++ [⟨18, by decide⟩]) ++ [finiteParityLabel bits]) =
      finiteParityPrompt bits ++ [finiteParityLabel bits] := by
    rw [finiteParityPrompt, List.cons_append]
  have h := decoderInput_raw eps ⟨1, by decide⟩ (finiteParityLabel bits)
    (finiteBitTokens bits ++ [⟨18, by decide⟩])
    ⟨(ratio_answer_embedding bits).1, (ratio_answer_embedding bits).2.1⟩
  have hd : decodeTokens (⟨1, by decide⟩ ::
      ((finiteBitTokens bits ++ [⟨18, by decide⟩]) ++ [finiteParityLabel bits])) =
      parityPrompt bits ++ [parityLabel bits] := by
    rw [hform, finiteParityAnswer_decode]
  rw [hd, (parityAnswer_counts bits).1, (parityAnswer_counts bits).2,
    decoder_label_embedding] at h
  simpa only [finiteParityPrompt, finiteBitTokens, List.length_map, List.length_append,
    List.length_cons, List.length_nil, List.cons_append, Nat.add_assoc, Nat.reduceAdd,
    Nat.cast_one] using h

/-- The raw prompt's actual first FFN input retains phase +1.
Source: the full prompt-state identity, with orthogonal ONE/BOS residual directions. -/
theorem decoderInput_prompt_phase (eps : ℝ) (bits : List Bool) :
    phaseProbe (decoderInput eps (fun j => (j.val : ℝ)) (finiteParityPrompt bits).get
      ⟨bits.length + 1, by simp [finiteParityPrompt, finiteBitTokens]⟩) = 1 := by
  rw [decoderInput_prompt]
  simp [phaseProbe, innerSL_apply_apply, inner_add_right, phaseUnit, controlUnit,
    bosUnit, EuclideanSpace.inner_single_left]

/-- The supplied-answer first FFN input retains phase -1 despite its nonzero tied parity code.
Source: the full raw answer-state identity; the new label and count coordinates are orthogonal to phase. -/
theorem decoderInput_answer_phase (eps : ℝ) (bits : List Bool) :
    phaseProbe (decoderInput eps (fun j => (j.val : ℝ))
      (finiteParityPrompt bits ++ [finiteParityLabel bits]).get
      ⟨bits.length + 2, by simp [finiteParityPrompt, finiteBitTokens]⟩) = -1 := by
  rw [decoderInput_answer]
  simp [phaseProbe, innerSL_apply_apply, inner_add_right, phaseUnit, controlUnit,
    bosUnit, parityUnit, EuclideanSpace.inner_single_left]

/-- The true raw prompt input has a positive BOS denominator, including zero-ONE words.
Source: its exact two-channel state and RMSNorm's positive unit-embedding multiplier. -/
theorem decoderInput_prompt_bos_pos (eps : ℝ) (heps : 0 < eps) (bits : List Bool) :
    0 < inner (𝕜 := ℝ) bosUnit
      (decoderInput eps (fun j => (j.val : ℝ)) (finiteParityPrompt bits).get
        ⟨bits.length + 1, by simp [finiteParityPrompt, finiteBitTokens]⟩) := by
  rw [decoderInput_prompt]
  simp only [inner_add_right, real_inner_smul_right]
  have hscale := countScale_pos eps heps
  simp only [bosUnit, controlUnit, phaseUnit, EuclideanSpace.inner_single_left,
    PiLp.single_apply]
  norm_num
  positivity

example : (0 : ℝ) < 1 / 100000 := by norm_num

/-!
These identities are universal in the raw bit word and retain the exact
normalization coefficient. Their proofs use only checked token encoding
and the actual parameter matrices. At the model's supervised domain the
word length is one through sixteen; quantitative output margins must be
uniform across that complete range and across both phases.
-/

end Transformer.GPTMini.Semantics
