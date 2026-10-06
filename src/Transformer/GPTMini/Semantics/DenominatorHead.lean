import Transformer.GPTMini.Semantics.DenominatorEmbedding

/-!
# Uniform softmax retains count and denominator simultaneously

Source: CausalMHA.forward at f11b6e2 and the raw no-scratchpad parity
grammar at cbafbe9. The preceding finite embedding calculation is used
to evaluate the actual head. Thus its faithful ONE/BOS values are proved
from raw token IDs, rather than assumed as a prepared semantic array.

At a final token with zero value, uniform attention averages both token
indicators and XSA leaves them unchanged. Every legal parity prompt has
exactly one BOS, including its correct supplied-answer continuation.
Consequently a second internal feature records the denominator of the
normalized ONE count. A later residual/FFN proof must still show how to
decode parity and EOS. This module does not assume either desired answer.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators
open Transformer.Basis

/-- The disjoint raw ONE and BOS branches equal the sum of their indicators.
Source: the distinct cbafbe9 vocabulary IDs, with both channels retained in one head. -/
theorem ratioValue_eq_add (scale : ℝ) (token : ℤ) :
    ratioValue scale token = oneValue (scale • countDirection) token +
      (if token = bos then scale • bosDirection else 0) := by
  by_cases hone : token = oneBit
  · have hbos : token ≠ bos := by rw [hone]; decide
    simp only [ratioValue, oneValue, ite_eq_left hone, ite_eq_right hbos, add_zero]
  · simp [ratioValue, oneValue, hone]

/-- A finite array of checked tokens has exactly the same marker count as its integer decoding.
Source: the List Int adapter and the ordinary indicator sum, for any raw marker ID. -/
theorem marker_finite_sum {V d : ℕ} (tokens : List (Fin V)) (marker : ℤ)
    (direction : EucSpace d) :
    (∑ j : Fin tokens.length, if ((tokens.get j).val : ℤ) = marker then direction else 0) =
      ((decodeTokens tokens).count marker : ℝ) • direction := by
  rw [← List.sum_ofFn]
  change (List.ofFn ((fun t : Fin V => if (t.val : ℤ) = marker then direction else 0) ∘
    tokens.get)).sum = _
  rw [← List.map_ofFn, List.ofFn_get]
  have he : tokens.map (fun t => if (t.val : ℤ) = marker then direction else 0) =
      (decodeTokens tokens).map (fun t => if t = marker then direction else 0) := by
    simp only [decodeTokens, List.map_map]
    rfl
  rw [he]
  have h := List.sum_map_ite_eq (decodeTokens tokens) (fun _ => direction) (fun _ => 0) marker
  simpa only [sub_zero, List.sum_map_zero, add_zero, Nat.cast_smul_eq_nsmul] using h

/-- Summation preserves both raw counters, not only their quotient.
Source: the actual two-channel values from the preceding embedding/QKV calculation. -/
theorem ratio_values_sum (tokens : List (Fin countConfig.vocab_size)) (scale : ℝ) :
    (∑ j : Fin tokens.length, ratioValue scale ((tokens.get j).val : ℤ)) =
      ((decodeTokens tokens).count oneBit : ℝ) • (scale • countDirection) +
      ((decodeTokens tokens).count bos : ℝ) • (scale • bosDirection) := by
  simp_rw [ratioValue_eq_add]
  rw [Finset.sum_add_distrib, one_finite_tokens_sum, marker_finite_sum]

/-- The actual prenorm/QKV/reshape produces these values throughout a finite context.
Source: headAt and ratio_qkv_features, with no oracle or feature-encoding premise. -/
theorem ratio_head_forward (eps : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin countConfig.vocab_size) (i : Fin T) :
    headAt countConfig (ratioParams.blocks ⟨0, by decide⟩).attn eps positions
        (embed countConfig ratioParams tokens) 0 i =
      attentionHead countConfig 0 eps (fun _ => 0) (fun _ => 0)
        (fun j => ratioValue (countScale eps) ((tokens j).val : ℤ)) positions i := by
  have hq := funext fun j => (ratio_qkv_features eps (tokens j)).1
  have hk := funext fun j => (ratio_qkv_features eps (tokens j)).2.1
  have hv := funext fun j => (ratio_qkv_features eps (tokens j)).2.2
  dsimp only [headAt, embed, ratioParams]
  rw [hq, hk, hv]

/-- The actual head contains both normalized raw counts at any final non-ONE, non-BOS token.
Source: finite uniform softmax, true RMS multiplier and zero-self-value XSA at f11b6e2. -/
theorem ratio_head_raw_counts (eps : ℝ) (head last : Fin countConfig.vocab_size)
    (body : List (Fin countConfig.vocab_size)) (hlast : last.val ≠ 22 ∧ last.val ≠ 1) :
    headAt countConfig (ratioParams.blocks ⟨0, by decide⟩).attn eps (fun j => (j.val : ℝ))
        (embed countConfig ratioParams (head :: (body ++ [last])).get) 0
        ⟨body.length + 1, by simp⟩ =
      (((decodeTokens (head :: (body ++ [last]))).count oneBit : ℝ) /
        (body.length + 2 : ℕ)) • (countScale eps • countDirection) +
      (((decodeTokens (head :: (body ++ [last]))).count bos : ℝ) /
        (body.length + 2 : ℕ)) • (countScale eps • bosDirection) := by
  rw [ratio_head_forward]
  have hone : (last.val : ℤ) ≠ oneBit := by
    change (last.val : ℤ) ≠ 22
    exact_mod_cast hlast.1
  have hbos : (last.val : ℤ) ≠ bos := by
    change (last.val : ℤ) ≠ 1
    exact_mod_cast hlast.2
  have hself : ratioValue (countScale eps)
      (((head :: (body ++ [last])).get ⟨body.length + 1, by simp⟩).val : ℤ) = 0 := by
    simp only [List.get_eq_getElem, List.getElem_cons_succ,
      List.getElem_append_right (le_refl body.length), Nat.sub_self, List.getElem_cons_zero]
    simp only [ratioValue, ite_eq_right hone, ite_eq_right hbos]
  have hfinal (r : Fin (head :: (body ++ [last])).length) :
      r.val ≤ (⟨body.length + 1, by simp⟩ : Fin (head :: (body ++ [last])).length).val := by
    have hr := r.isLt
    simp only [List.length_cons, List.length_append, List.length_nil] at hr
    change r.val ≤ body.length + 1
    omega
  have h := head_uniform_average countConfig 0 eps
    (fun j => ratioValue (countScale eps) (((head :: (body ++ [last])).get j).val : ℤ))
    (fun j => (j.val : ℝ)) ⟨body.length + 1, by simp⟩ hfinal hself
  rw [ratio_values_sum, smul_add] at h
  simpa only [smul_smul, one_div_mul_eq_div, List.length_cons, List.length_append,
    List.length_nil, Nat.add_assoc, Nat.reduceAdd, div_mul_eq_mul_div, mul_assoc, one_mul] using h

example : (⟨18, by decide⟩ : Fin countConfig.vocab_size).val ≠ 22 ∧
    (⟨18, by decide⟩ : Fin countConfig.vocab_size).val ≠ 1 := by decide

/-- Every raw parity prompt has exactly one BOS independently of its bit length or count.
Source: Bits.prompt at cbafbe9, including the actual delimiters. -/
theorem parityPrompt_bos_count (bits : List Bool) : (parityPrompt bits).count bos = 1 := by
  have hbody : (bitTokens bits).count bos = 0 := by
    induction bits with
    | nil => rfl
    | cons bit bits ih =>
        cases bit <;> simp [bitTokens, bos, oneBit, zeroBit] at ih ⊢ <;> exact ih
  simp only [parityPrompt, List.count_append, List.count_cons, hbody]
  norm_num [bos, sep]

/-- The supplied correct label changes neither the ONE count nor the unique BOS count.
Source: the second supervised prefix in Parity.solve, with EVEN=24 and ODD=25. -/
theorem parityAnswer_counts (bits : List Bool) :
    (parityPrompt bits ++ [parityLabel bits]).count oneBit = bits.count true ∧
      (parityPrompt bits ++ [parityLabel bits]).count bos = 1 := by
  rw [List.count_append, List.count_append, parityPrompt_one_count, parityPrompt_bos_count]
  unfold parityLabel
  split_ifs <;> norm_num [List.count_cons, oneBit, bos, evenToken, oddToken]

/-- The denominator direction has unit norm in the original head.
Source: the second within-head coordinate, orthogonal to countDirection. -/
theorem bosDirection_norm : ‖bosDirection‖ = 1 := by
  simp [bosDirection, PiLp.norm_single]

/-- The actual repaired head separates the legal variable-length pair that the count/phase state collapsed.
Source: the independent BOS mass is gamma/four versus gamma/eight, while the ONE mass is unchanged. -/
theorem ratio_head_breaks_collision (eps : ℝ) (heps : 0 < eps) :
    headAt countConfig (ratioParams.blocks ⟨0, by decide⟩).attn eps (fun j => (j.val : ℝ))
        (embed countConfig ratioParams oddFractionTokens.get) 0 ⟨3, by decide⟩ ≠
      headAt countConfig (ratioParams.blocks ⟨0, by decide⟩).attn eps (fun j => (j.val : ℝ))
        (embed countConfig ratioParams evenFractionTokens.get) 0 ⟨7, by decide⟩ := by
  have hs := congrArg (fun x => inner (𝕜 := ℝ) bosDirection x)
    (ratio_head_raw_counts eps ⟨1, by decide⟩ ⟨18, by decide⟩
      [⟨22, by decide⟩, ⟨21, by decide⟩] (by decide))
  have hl := congrArg (fun x => inner (𝕜 := ℝ) bosDirection x)
    (ratio_head_raw_counts eps ⟨1, by decide⟩ ⟨18, by decide⟩
      [⟨22, by decide⟩, ⟨22, by decide⟩, ⟨21, by decide⟩,
        ⟨21, by decide⟩, ⟨21, by decide⟩, ⟨21, by decide⟩] (by decide))
  norm_num [decodeTokens, oneBit, bos, inner_add_right, real_inner_smul_right,
    bosDirection, countDirection, EuclideanSpace.inner_single_left] at hs hl
  simp only [List.cons_append, List.nil_append, List.length_cons, List.length_nil] at hs hl
  intro hstates
  have hprobe := congrArg (fun x => inner (𝕜 := ℝ) bosDirection x) hstates
  norm_num [bosDirection, EuclideanSpace.inner_single_left] at hprobe
  have hscale := hs.symm.trans (hprobe.trans hl)
  have hpos := countScale_pos eps heps
  linarith

example : (0 : ℝ) < 1 / 100000 := by norm_num

end Transformer.GPTMini.Semantics
