import Transformer.GPTMini.Semantics.CountEmbedding

/-!
# The original first block retains the raw ONE count

Source: GPTMini.forward/Block.forward at f11b6e2 and the raw Basis parity
prompt at cbafbe9. The preceding module proves the finite embedding/QKV
calculation, so no faithful-value hypothesis is needed here. The actual
first head counts raw ONE IDs, and the identity output projection carries
that signal into the actual first residual block. The positive prenorm
multiplier is included in the exact count recovery formula.

The reference code is a token indicator, not an answer label. No property
of correct logits or a task-solving predicate is assumed. This count
feature is insufficient by itself for the full task: parity decoding and
the answer/EOS phase must be implemented separately. Statements evaluate
the mathematical block at arbitrary finite lengths; the checked model
adapter additionally enforces the configured 19-token context cap.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators
open Transformer.Basis

/-- Integer decoding and finite-array summation count the same actual ONE tokens.
Source: the checked List Int interface and the preceding raw-indicator sum theorem. -/
theorem one_finite_tokens_sum {V d : ℕ} (tokens : List (Fin V)) (direction : EucSpace d) :
    (∑ j : Fin tokens.length, oneValue direction ((tokens.get j).val : ℤ)) =
      ((decodeTokens tokens).count oneBit : ℝ) • direction := by
  rw [← List.sum_ofFn]
  change (List.ofFn ((fun t : Fin V => oneValue direction (t.val : ℤ)) ∘ tokens.get)).sum = _
  rw [← List.map_ofFn, List.ofFn_get]
  have he : tokens.map (fun t => oneValue direction (t.val : ℤ)) =
      (decodeTokens tokens).map (oneValue direction) := by
    simp only [decodeTokens, List.map_map]
    rfl
  rw [he, one_values_sum]

/-- Actual prenorm and fused QKV produce this raw feature for every finite token array.
Source: CausalMHA.forward at f11b6e2, with count_qkv_features proved on the actual embeddings. -/
theorem count_head_forward (eps : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin countConfig.vocab_size) (i : Fin T) :
    headAt countConfig countAttn eps positions (embed countConfig countParams tokens) 0 i =
      attentionHead countConfig 0 eps (fun _ => 0) (fun _ => 0)
        (fun j => oneValue (countScale eps • countDirection) ((tokens j).val : ℤ)) positions i := by
  have hq := funext fun j => (count_qkv_features eps (tokens j)).1
  have hk := funext fun j => (count_qkv_features eps (tokens j)).2.1
  have hv := funext fun j => (count_qkv_features eps (tokens j)).2.2
  dsimp only [headAt, embed]
  rw [hq, hk, hv]
  rfl

/-- The actual embedding/head pipeline computes the raw count at any final non-ONE token.
Source: the original softmax, RoPE and XSA at f11b6e2; no prepared value array is a premise. -/
theorem count_head_raw_count (eps : ℝ) (head last : Fin countConfig.vocab_size)
    (body : List (Fin countConfig.vocab_size)) (hlast : last.val ≠ 22) :
    headAt countConfig countAttn eps (fun j => (j.val : ℝ))
        (embed countConfig countParams (head :: (body ++ [last])).get) 0
        ⟨body.length + 1, by simp⟩ =
      (((decodeTokens (head :: (body ++ [last]))).count oneBit : ℝ) /
        (body.length + 2 : ℕ)) • (countScale eps • countDirection) := by
  rw [count_head_forward]
  have hself : oneValue (countScale eps • countDirection)
      (((head :: (body ++ [last])).get ⟨body.length + 1, by simp⟩).val : ℤ) = 0 := by
    simp only [List.get_eq_getElem, List.getElem_cons_succ,
      List.getElem_append_right (le_refl body.length), Nat.sub_self, List.getElem_cons_zero]
    apply ite_eq_right
    change (last.val : ℤ) ≠ 22
    exact_mod_cast hlast
  have hfinal (r : Fin (head :: (body ++ [last])).length) :
      r.val ≤ (⟨body.length + 1, by simp⟩ : Fin (head :: (body ++ [last])).length).val := by
    have hr := r.isLt
    simp only [List.length_cons, List.length_append, List.length_nil] at hr
    change r.val ≤ body.length + 1
    omega
  have h := head_uniform_average countConfig 0 eps
    (fun j => oneValue (countScale eps • countDirection)
      (((head :: (body ++ [last])).get j).val : ℤ))
    (fun j => (j.val : ℝ)) ⟨body.length + 1, by simp⟩ hfinal hself
  rw [one_finite_tokens_sum, smul_smul] at h
  simpa only [one_div_mul_eq_div, List.length_cons, List.length_append, List.length_nil] using h

example : (⟨18, by decide⟩ : Fin countConfig.vocab_size).val ≠ 22 := by decide

/-- The actual first residual block carries the count signal after subtracting its retained input embedding.
Source: the identity W_o and zero FFN in countBlock, with both original residual additions retained. -/
theorem count_block_raw_count (eps : ℝ) (head last : Fin countConfig.vocab_size)
    (body : List (Fin countConfig.vocab_size)) (hlast : last.val ≠ 22) :
    headSlice countConfig
        (hidden countConfig countParams eps (fun j => (j.val : ℝ))
          (head :: (body ++ [last])).get 1 ⟨body.length + 1, by simp⟩ -
          countParams.embedding last) 0 =
      (((decodeTokens (head :: (body ++ [last]))).count oneBit : ℝ) /
        (body.length + 2 : ℕ)) • (countScale eps • countDirection) := by
  rw [hidden, dite_eq_left (by decide : 0 < countConfig.n_layers)]
  simp only [hidden]
  have hlastToken : (head :: (body ++ [last])).get ⟨body.length + 1, by simp⟩ = last := by
    simp only [List.get_eq_getElem, List.getElem_cons_succ,
      List.getElem_append_right (le_refl body.length), Nat.sub_self, List.getElem_cons_zero]
  change headSlice countConfig
    (blockForward countConfig countBlock eps (fun j => (j.val : ℝ))
      (embed countConfig countParams (head :: (body ++ [last])).get)
      ⟨body.length + 1, by simp⟩ - countParams.embedding last) 0 = _
  unfold blockForward
  simp only [countBlock, ffnSubLayer, relu2FFN, zero_apply, add_zero]
  rw [embed, hlastToken]
  have he (x y : EucSpace countConfig.d_model) : x + y - x = y := by module
  rw [he, attnSubLayer_eq_heads]
  rw [show countAttn.W_o = ContinuousLinearMap.id ℝ (EucSpace 64) from rfl,
    ContinuousLinearMap.id_apply]
  change headSlice countConfig (headMerge countConfig _) 0 = _
  rw [headSlice_headMerge]
  exact count_head_raw_count eps head last body hlast

example : (⟨24, by decide⟩ : Fin countConfig.vocab_size).val ≠ 22 := by decide

/-- The signal in the real residual stream recovers the integer raw count exactly at positive epsilon.
Source: the actual first block's calculation and its positive RMS multiplier, before any parity decoder. -/
theorem count_block_recovers (eps : ℝ) (heps : 0 < eps) (head last : Fin countConfig.vocab_size)
    (body : List (Fin countConfig.vocab_size)) (hlast : last.val ≠ 22) :
    inner (𝕜 := ℝ)
        (headSlice countConfig
          (hidden countConfig countParams eps (fun j => (j.val : ℝ))
            (head :: (body ++ [last])).get 1 ⟨body.length + 1, by simp⟩ -
            countParams.embedding last) 0) countDirection * (body.length + 2 : ℕ) /
        countScale eps = ((decodeTokens (head :: (body ++ [last]))).count oneBit : ℝ) := by
  rw [count_block_raw_count eps head last body hlast,
    real_inner_smul_left, real_inner_smul_left, real_inner_self_eq_norm_sq, countDirection_norm]
  have hs : countScale eps ≠ 0 := ne_of_gt (countScale_pos eps heps)
  have hn : ((body.length + 2 : ℕ) : ℝ) ≠ 0 := by positivity
  norm_num
  field_simp

example : (0 : ℝ) < 1 / 100000 ∧
    (⟨18, by decide⟩ : Fin countConfig.vocab_size).val ≠ 22 := ⟨by norm_num, by decide⟩

/-- The actual first block distinguishes the two one-bit Basis inputs in its hidden states.
Source: the raw ZERO/ONE parity pair; this is a property of explicit original-model parameters,
not an existence statement and not a premise that the final EVEN/ODD logits are correct. -/
theorem count_first_block_bits_distinct (eps : ℝ) (heps : 0 < eps) :
    hidden countConfig countParams eps (fun j => (j.val : ℝ))
        ([⟨1, by decide⟩, ⟨21, by decide⟩, ⟨18, by decide⟩] :
          List (Fin countConfig.vocab_size)).get 1 ⟨2, by decide⟩ ≠
      hidden countConfig countParams eps (fun j => (j.val : ℝ))
        ([⟨1, by decide⟩, ⟨22, by decide⟩, ⟨18, by decide⟩] :
          List (Fin countConfig.vocab_size)).get 1 ⟨2, by decide⟩ := by
  have hx := count_block_raw_count eps ⟨1, by decide⟩ ⟨18, by decide⟩
    [⟨21, by decide⟩] (by decide)
  have hy := count_block_raw_count eps ⟨1, by decide⟩ ⟨18, by decide⟩
    [⟨22, by decide⟩] (by decide)
  norm_num [decodeTokens, oneBit] at hx hy
  simp only [List.cons_append, List.nil_append, List.length_cons, List.length_nil] at hx hy
  intro heq
  have hsignal := congrArg (fun x => headSlice countConfig
    (x - countParams.embedding ⟨18, by decide⟩) 0) heq
  have hzero : (0 : EucSpace countConfig.head_dim) =
      (1 / 3 : ℝ) • (countScale eps • countDirection) := hx.symm.trans (hsignal.trans hy)
  have hs : 0 < countScale eps := countScale_pos eps heps
  have hnorm := congrArg norm hzero
  rw [norm_zero, norm_smul, norm_smul, countDirection_norm] at hnorm
  norm_num [Real.norm_eq_abs, abs_of_pos hs] at hnorm
  linarith

example : (0 : ℝ) < 1 / 100000 := by norm_num

end Transformer.GPTMini.Semantics
