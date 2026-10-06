import Transformer.GPTMini.Semantics.DenominatorBlock

/-!
# Exact internal counting features for every raw parity word

Source: Bits.prompt and Parity.solve at cbafbe9, and the original
GPTMini block at f11b6e2. The finite serialization below round-trips the
actual integer token list. It supplies only BOS, raw bits, SEP and, on
the second supervised prefix, the already supplied correct answer token.

The actual first block recovers the ONE count from two hidden coordinates
for every bit word. The first and second supervision phases retain +1
and -1. These are universal raw-input feature computations, not selected
examples and not encoder hypotheses. A bounded parity/EOS FFN and the
full-stack tied readout still need to be implemented and verified.
No correctness of final logits or optimization is inferred from a count.
-/

namespace Transformer.GPTMini.Semantics

open TokenInterface Transformer.Basis

/-- Encode only raw ZERO/ONE vocabulary entries, retaining every bit position.
Source: Bits.prompt at cbafbe9, without index hints or a prepared count. -/
def finiteBitTokens (bits : List Bool) : List (Fin countConfig.vocab_size) :=
  bits.map fun bit => if bit then ⟨22, by decide⟩ else ⟨21, by decide⟩

/-- The exact finite-index version of the raw BOS/bits/SEP input.
Source: the checked vocabulary representation used by GPTMini.forward. -/
def finiteParityPrompt (bits : List Bool) : List (Fin countConfig.vocab_size) :=
  ⟨1, by decide⟩ :: (finiteBitTokens bits ++ [⟨18, by decide⟩])

/-- Encode the supplied correct label on the second supervised position.
Source: ParityPrefix's continuation grammar; this token is input, not a computed model prediction. -/
def finiteParityLabel (bits : List Bool) : Fin countConfig.vocab_size :=
  if bits.count true % 2 = 1 then ⟨25, by decide⟩ else ⟨24, by decide⟩

/-- The finite serializer reconstructs the actual bit IDs exactly.
Source: the Boolean-to-vocabulary map in Bits.prompt, not a task-solving transform. -/
theorem finiteBitTokens_decode (bits : List Bool) :
    decodeTokens (finiteBitTokens bits) = bitTokens bits := by
  simp only [finiteBitTokens, decodeTokens, bitTokens, List.map_map]
  congr 1
  funext bit
  cases bit <;> rfl

/-- Raw integer prompts and the model's finite arrays contain precisely the same tokens.
Source: the checked List Int adapter, including BOS and SEP. -/
theorem finiteParityPrompt_decode (bits : List Bool) :
    decodeTokens (finiteParityPrompt bits) = parityPrompt bits := by
  change decodeTokens (⟨1, by decide⟩ :: (finiteBitTokens bits ++ [⟨18, by decide⟩])) = _
  simp only [decodeTokens, List.map_cons, List.map_append]
  rw [← decodeTokens, finiteBitTokens_decode]
  rfl

/-- The input label is encoded without truncation or changing its actual ID.
Source: EVEN=24 and ODD=25 at cbafbe9. -/
theorem finiteParityLabel_decode (bits : List Bool) :
    ((finiteParityLabel bits).val : ℤ) = parityLabel bits := by
  unfold finiteParityLabel parityLabel
  split_ifs <;> rfl

/-- The supplied-answer finite prefix likewise round-trips the complete raw list.
Source: the second supervision position in Parity.solve. -/
theorem finiteParityAnswer_decode (bits : List Bool) :
    decodeTokens (finiteParityPrompt bits ++ [finiteParityLabel bits]) =
      parityPrompt bits ++ [parityLabel bits] := by
  simp only [decodeTokens, List.map_append, List.map_cons, List.map_nil]
  rw [← decodeTokens, finiteParityPrompt_decode, finiteParityLabel_decode]

/-- The final first-block prompt state retains phase and both independent count channels.
Source: the actual ratioParams embedding, prenorm, QKV, softmax, XSA and residual, for all bit words. -/
theorem ratio_prompt_state (eps : ℝ) (bits : List Bool) :
    hidden countConfig ratioParams eps (fun j => (j.val : ℝ))
        (finiteParityPrompt bits).get 1 ⟨bits.length + 1, by simp [finiteParityPrompt, finiteBitTokens]⟩ =
      phaseUnit +
        (((bits.count true : ℝ) / (bits.length + 2 : ℕ) * countScale eps) • controlUnit +
          ((1 : ℝ) / (bits.length + 2 : ℕ) * countScale eps) • bosUnit) := by
  have h := ratio_block_state eps ⟨1, by decide⟩ ⟨18, by decide⟩
    (finiteBitTokens bits) (by decide)
  have hdecode := finiteParityPrompt_decode bits
  dsimp only [finiteParityPrompt] at hdecode
  rw [hdecode, parityPrompt_one_count, parityPrompt_bos_count] at h
  rw [show ratioParams.embedding ⟨18, by decide⟩ = phaseUnit from rfl] at h
  simpa only [finiteParityPrompt, finiteBitTokens, List.length_map, Nat.cast_one] using h

/-- A raw prompt's two hidden coordinates recover its integer ONE count with no length input.
Source: the unique BOS grammar and the actual two-channel first block. -/
theorem ratio_prompt_recovers (eps : ℝ) (heps : 0 < eps) (bits : List Bool) :
    inner (𝕜 := ℝ) controlUnit
        (hidden countConfig ratioParams eps (fun j => (j.val : ℝ))
          (finiteParityPrompt bits).get 1 ⟨bits.length + 1, by simp [finiteParityPrompt, finiteBitTokens]⟩) /
      inner (𝕜 := ℝ) bosUnit
        (hidden countConfig ratioParams eps (fun j => (j.val : ℝ))
          (finiteParityPrompt bits).get 1 ⟨bits.length + 1, by simp [finiteParityPrompt, finiteBitTokens]⟩) =
      (bits.count true : ℝ) := by
  have hdecode := finiteParityPrompt_decode bits
  dsimp only [finiteParityPrompt] at hdecode
  have hbos : (decodeTokens (⟨1, by decide⟩ ::
      (finiteBitTokens bits ++ [⟨18, by decide⟩]))).count bos = 1 := by
    rw [hdecode, parityPrompt_bos_count]
  have h := ratio_block_recovers eps heps ⟨1, by decide⟩ ⟨18, by decide⟩
    (finiteBitTokens bits) (by decide) hbos
  rw [hdecode, parityPrompt_one_count] at h
  simpa only [finiteParityPrompt, finiteBitTokens, List.length_map] using h

example : (0 : ℝ) < 1 / 100000 := by norm_num

/-- The label phase's last token has zero ONE/BOS values and embedding -phaseUnit.
Source: the correct supplied EVEN/ODD token, with neither label contributing a count. -/
theorem ratio_answer_embedding (bits : List Bool) :
    (finiteParityLabel bits).val ≠ 22 ∧ (finiteParityLabel bits).val ≠ 1 ∧
      ratioParams.embedding (finiteParityLabel bits) = -phaseUnit := by
  unfold finiteParityLabel
  split_ifs <;> simp [ratioParams, ratioEmbedding, phaseEmbedding]

/-- Every supplied correct-answer prefix retains the same integer count in its own internal quotient.
Source: Parity.solve's EOS position, not an assumption that the model already predicted the label. -/
theorem ratio_answer_recovers (eps : ℝ) (heps : 0 < eps) (bits : List Bool) :
    inner (𝕜 := ℝ) controlUnit
        (hidden countConfig ratioParams eps (fun j => (j.val : ℝ))
          (finiteParityPrompt bits ++ [finiteParityLabel bits]).get 1
          ⟨bits.length + 2, by simp [finiteParityPrompt, finiteBitTokens]⟩) /
      inner (𝕜 := ℝ) bosUnit
        (hidden countConfig ratioParams eps (fun j => (j.val : ℝ))
          (finiteParityPrompt bits ++ [finiteParityLabel bits]).get 1
          ⟨bits.length + 2, by simp [finiteParityPrompt, finiteBitTokens]⟩) =
      (bits.count true : ℝ) := by
  have hform : ⟨1, by decide⟩ ::
      ((finiteBitTokens bits ++ [⟨18, by decide⟩]) ++ [finiteParityLabel bits]) =
      finiteParityPrompt bits ++ [finiteParityLabel bits] := by
    rw [finiteParityPrompt, List.cons_append]
  have hbos : (decodeTokens (⟨1, by decide⟩ ::
      ((finiteBitTokens bits ++ [⟨18, by decide⟩]) ++ [finiteParityLabel bits]))).count bos = 1 := by
    rw [hform, finiteParityAnswer_decode, (parityAnswer_counts bits).2]
  have h := ratio_block_recovers eps heps ⟨1, by decide⟩ (finiteParityLabel bits)
    (finiteBitTokens bits ++ [⟨18, by decide⟩])
    ⟨(ratio_answer_embedding bits).1, (ratio_answer_embedding bits).2.1⟩ hbos
  have hcount : (decodeTokens (⟨1, by decide⟩ ::
      ((finiteBitTokens bits ++ [⟨18, by decide⟩]) ++ [finiteParityLabel bits]))).count oneBit =
      bits.count true := by
    rw [hform, finiteParityAnswer_decode, (parityAnswer_counts bits).1]
  rw [hcount] at h
  simpa only [finiteParityPrompt, finiteBitTokens, List.length_map, List.length_append,
    List.length_cons, List.length_nil, List.cons_append, Nat.add_assoc, Nat.reduceAdd] using h

example : (0 : ℝ) < 1 / 100000 := by norm_num

/-- Both active output channels lie in the phase probe's kernel in every actual block.
Source: the independent ONE/BOS residual directions and the unchanged output matrix types. -/
theorem ratio_output_kernel :
    (∀ b, phaseProbe.comp (ratioParams.blocks b).attn.W_o = 0) ∧
      (∀ b, phaseProbe.comp (ratioParams.blocks b).ffn.W_out = 0) := by
  constructor
  · intro b
    ext x
    simp [ratioParams, ratioOutput, phaseOutput, phaseProbe, ContinuousLinearMap.comp_apply,
      ContinuousLinearMap.smulRight_apply, innerSL_apply_apply,
      phaseUnit, controlUnit, bosUnit, EuclideanSpace.inner_single_left]
  · intro b
    simp [ratioParams]

/-- The repaired concrete model retains completion phase through its entire actual stack.
Source: the proved local output-matrix equations and GPTMini.hidden's block induction. -/
theorem ratio_hidden_phase_preserved (eps : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin countConfig.vocab_size) (L : ℕ) (i : Fin T) :
    phaseProbe (hidden countConfig ratioParams eps positions tokens L i) =
      phaseProbe (ratioParams.embedding (tokens i)) :=
  hidden_probe_preserved countConfig ratioParams eps phaseProbe
    ratio_output_kernel.1 ratio_output_kernel.2 positions tokens L i

end Transformer.GPTMini.Semantics
