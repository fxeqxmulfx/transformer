import Transformer.GPTMini.Semantics.DenominatorHead

/-!
# Recover the raw count from the actual residual stream

Source: Block.forward at f11b6e2 and Basis's raw parity grammar at
cbafbe9. The nonzero output projection writes the proved ONE and BOS
head signals into separate residual coordinates, while preserving the
completion phase. Their quotient recovers the integer ONE count whenever
the raw input contains exactly one BOS and ends at a non-indicator token.

The decoder receives only the hidden state: neither context length nor
the RMS multiplier is an additional input. The unique-BOS condition is
a raw-token grammar property, proved for every parity prompt and correct
continuation in DenominatorHead. No correct logits or parity answer is a
premise. This removes the representation collision; realization of a
bounded parity/EOS decoder by the actual ReLU2 FFN is still required.
-/

namespace Transformer.GPTMini.Semantics

open TokenInterface Transformer.Basis

/-- The ordinary output matrix writes both first-head coordinates into the intended residual channels.
Source: the true contiguous head merge and the explicit rank-two W_o. -/
theorem ratioOutput_headMerge (heads : Fin countConfig.n_heads → EucSpace countConfig.head_dim) :
    ratioOutput (headMerge countConfig heads) =
      inner (𝕜 := ℝ) countDirection (heads 0) • controlUnit +
      inner (𝕜 := ℝ) bosDirection (heads 0) • bosUnit := by
  have hs : headSplit countConfig 1 = (0, ⟨1, by decide⟩) := by decide
  have hb : inner (𝕜 := ℝ) phaseUnit (headMerge countConfig heads) =
      inner (𝕜 := ℝ) bosDirection (heads 0) := by
    simp only [phaseUnit, bosDirection, EuclideanSpace.inner_single_left, headMerge_apply, hs]
  simp only [ratioOutput, add_apply, phaseOutput, ContinuousLinearMap.smulRight_apply,
    innerSL_apply_apply, phase_headMerge_probe, hb]

/-- Both normalized counts enter the whole actual first hidden state, together with the input embedding.
Source: the original prenorm/attention/XSA/residual computation and zero FFN at f11b6e2. -/
theorem ratio_block_state (eps : ℝ) (head last : Fin countConfig.vocab_size)
    (body : List (Fin countConfig.vocab_size)) (hlast : last.val ≠ 22 ∧ last.val ≠ 1) :
    hidden countConfig ratioParams eps (fun j => (j.val : ℝ))
        (head :: (body ++ [last])).get 1 ⟨body.length + 1, by simp⟩ =
      ratioParams.embedding last +
        ((((decodeTokens (head :: (body ++ [last]))).count oneBit : ℝ) /
          (body.length + 2 : ℕ) * countScale eps) • controlUnit +
        (((decodeTokens (head :: (body ++ [last]))).count bos : ℝ) /
          (body.length + 2 : ℕ) * countScale eps) • bosUnit) := by
  have hlastToken : (head :: (body ++ [last])).get ⟨body.length + 1, by simp⟩ = last := by
    simp only [List.get_eq_getElem, List.getElem_cons_succ,
      List.getElem_append_right (le_refl body.length), Nat.sub_self, List.getElem_cons_zero]
  rw [hidden, dite_eq_left (by decide : 0 < countConfig.n_layers)]
  simp only [hidden]
  unfold blockForward
  simp only [ratioParams, ffnSubLayer, relu2FFN, zero_apply, add_zero]
  rw [embed, hlastToken, attnSubLayer_eq_heads]
  change ratioEmbedding last + ratioOutput (headMerge countConfig _) = _
  rw [ratioOutput_headMerge]
  have hhead := ratio_head_raw_counts eps head last body hlast
  dsimp only [ratioParams] at hhead
  rw [hhead]
  simp only [inner_add_right, real_inner_smul_right]
  have hc : inner (𝕜 := ℝ) countDirection countDirection = 1 := by
    rw [real_inner_self_eq_norm_sq, countDirection_norm]
    norm_num
  have hb : inner (𝕜 := ℝ) bosDirection bosDirection = 1 := by
    rw [real_inner_self_eq_norm_sq, bosDirection_norm]
    norm_num
  have hcb : inner (𝕜 := ℝ) countDirection bosDirection = 0 := by
    simp [countDirection, bosDirection, EuclideanSpace.inner_single_left]
  have hbc : inner (𝕜 := ℝ) bosDirection countDirection = 0 := by
    simp [countDirection, bosDirection, EuclideanSpace.inner_single_left]
  simp only [hc, hb, hcb, hbc, mul_one, mul_zero, add_zero, zero_add]

example : (⟨18, by decide⟩ : Fin countConfig.vocab_size).val ≠ 22 ∧
    (⟨18, by decide⟩ : Fin countConfig.vocab_size).val ≠ 1 := by decide

/-- A final non-indicator embedding has zero probes in both residual count coordinates.
Source: the raw ratioEmbedding table, including SEP and either answer label. -/
theorem ratio_embedding_probes (last : Fin countConfig.vocab_size)
    (hlast : last.val ≠ 22 ∧ last.val ≠ 1) :
    inner (𝕜 := ℝ) controlUnit (ratioParams.embedding last) = 0 ∧
      inner (𝕜 := ℝ) bosUnit (ratioParams.embedding last) = 0 := by
  simp only [ratioParams, ratioEmbedding, ite_eq_right hlast.2,
    phaseEmbedding, ite_eq_right hlast.1]
  split_ifs <;>
    simp [controlUnit, bosUnit, phaseUnit, EuclideanSpace.inner_single_left]

example : (⟨24, by decide⟩ : Fin countConfig.vocab_size).val ≠ 22 ∧
    (⟨24, by decide⟩ : Fin countConfig.vocab_size).val ≠ 1 := by decide

/-- Reading the actual first residual stream gives the two normalized counters directly.
Source: the preceding full-state identity and the orthogonal embedding coordinates. -/
theorem ratio_block_probes (eps : ℝ) (head last : Fin countConfig.vocab_size)
    (body : List (Fin countConfig.vocab_size)) (hlast : last.val ≠ 22 ∧ last.val ≠ 1) :
    inner (𝕜 := ℝ) controlUnit
        (hidden countConfig ratioParams eps (fun j => (j.val : ℝ))
          (head :: (body ++ [last])).get 1 ⟨body.length + 1, by simp⟩) =
        ((decodeTokens (head :: (body ++ [last]))).count oneBit : ℝ) /
          (body.length + 2 : ℕ) * countScale eps ∧
      inner (𝕜 := ℝ) bosUnit
        (hidden countConfig ratioParams eps (fun j => (j.val : ℝ))
          (head :: (body ++ [last])).get 1 ⟨body.length + 1, by simp⟩) =
        ((decodeTokens (head :: (body ++ [last]))).count bos : ℝ) /
          (body.length + 2 : ℕ) * countScale eps := by
  rw [ratio_block_state eps head last body hlast]
  have he := ratio_embedding_probes last hlast
  simp only [inner_add_right, he.1, he.2, zero_add, real_inner_smul_right]
  simp [controlUnit, bosUnit, EuclideanSpace.inner_single_left]

example : (⟨25, by decide⟩ : Fin countConfig.vocab_size).val ≠ 22 ∧
    (⟨25, by decide⟩ : Fin countConfig.vocab_size).val ≠ 1 := by decide

/-- The raw count is recoverable from the hidden state alone, without external length or multiplier.
Source: the simultaneous ONE/BOS computation; unique BOS is a grammar property, not a desired-label premise. -/
theorem ratio_block_recovers (eps : ℝ) (heps : 0 < eps) (head last : Fin countConfig.vocab_size)
    (body : List (Fin countConfig.vocab_size)) (hlast : last.val ≠ 22 ∧ last.val ≠ 1)
    (hbos : (decodeTokens (head :: (body ++ [last]))).count bos = 1) :
    inner (𝕜 := ℝ) controlUnit
        (hidden countConfig ratioParams eps (fun j => (j.val : ℝ))
          (head :: (body ++ [last])).get 1 ⟨body.length + 1, by simp⟩) /
      inner (𝕜 := ℝ) bosUnit
        (hidden countConfig ratioParams eps (fun j => (j.val : ℝ))
          (head :: (body ++ [last])).get 1 ⟨body.length + 1, by simp⟩) =
      ((decodeTokens (head :: (body ++ [last]))).count oneBit : ℝ) := by
  have h := ratio_block_probes eps head last body hlast
  rw [h.1, h.2, hbos]
  have hs : countScale eps ≠ 0 := ne_of_gt (countScale_pos eps heps)
  have ht : ((body.length + 2 : ℕ) : ℝ) ≠ 0 := by positivity
  norm_num
  field_simp

example : (0 : ℝ) < 1 / 100000 ∧
    ((⟨18, by decide⟩ : Fin countConfig.vocab_size).val ≠ 22 ∧
      (⟨18, by decide⟩ : Fin countConfig.vocab_size).val ≠ 1) ∧
    (decodeTokens ([⟨1, by decide⟩, ⟨22, by decide⟩, ⟨21, by decide⟩,
      ⟨18, by decide⟩] : List (Fin countConfig.vocab_size))).count bos = 1 := by
  exact ⟨by norm_num, by decide, by decide⟩

/-- The unique BOS coordinate in the actual residual stream is strictly positive, including zero-ONE inputs.
Source: positive RMSNorm multiplier and finite context length; division never depends on a nonzero ONE count. -/
theorem ratio_block_denominator_pos (eps : ℝ) (heps : 0 < eps)
    (head last : Fin countConfig.vocab_size) (body : List (Fin countConfig.vocab_size))
    (hlast : last.val ≠ 22 ∧ last.val ≠ 1)
    (hbos : (decodeTokens (head :: (body ++ [last]))).count bos = 1) :
    0 < inner (𝕜 := ℝ) bosUnit
      (hidden countConfig ratioParams eps (fun j => (j.val : ℝ))
        (head :: (body ++ [last])).get 1 ⟨body.length + 1, by simp⟩) := by
  rw [(ratio_block_probes eps head last body hlast).2, hbos]
  have hs := countScale_pos eps heps
  positivity

example : (0 : ℝ) < 1 / 100000 ∧
    ((⟨18, by decide⟩ : Fin countConfig.vocab_size).val ≠ 22 ∧
      (⟨18, by decide⟩ : Fin countConfig.vocab_size).val ≠ 1) ∧
    (decodeTokens ([⟨1, by decide⟩, ⟨21, by decide⟩, ⟨18, by decide⟩] :
      List (Fin countConfig.vocab_size))).count bos = 1 := by
  exact ⟨by norm_num, by decide, by decide⟩

end Transformer.GPTMini.Semantics
