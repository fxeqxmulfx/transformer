import Transformer.GPTMini.Semantics.RecallRawHeads

/-!
# Bind raw adjacent key/value tokens in the actual first residual

Source: MQAR.writes at cbafbe9 and the original fused QKV, RoPE,
finite softmax, XSA and W_o at f11b6e2. An actual value-token position
has zero self-value. Its immediate predecessor, when a raw key ID,
is copied with the derived positional error bound. Every query, key
and value is computed from the raw simultaneous embedding table;
the bound does not assume an already paired semantic input array.

Reading coordinates 18..25 of the genuine attention residual gives
that compact copied key. The BOS marker head runs simultaneously,
but its output cannot interfere with the copied-key slot. Thus the
first block has a genuine adjacency-sensitive binding representation
before its FFN. Gating the table region and robust final retrieval
remain necessary; a filler after a post-table query is not a write.
-/

namespace Transformer.GPTMini.Semantics

/-- The actual original first head approximately copies a raw neighboring key at a raw value position.
Source: derived RoPE gap, faithful raw QKV, complete value diameter and genuine zero-self-value XSA. -/
theorem recall_raw_predecessor_copy (eps alpha : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1)
    {T : ℕ} (hT : T ≤ 128) (tokens : Fin T → Fin recallConfig.vocab_size)
    (i selected : Fin T) (hprev : selected.val + 1 = i.val) (key value : Fin 256)
    (hkey : tokens selected = recallKeyId key) (hvalue : tokens i = recallValueId value) :
    ‖headAt recallConfig (recallFirstAttention eps alpha) eps (fun r => (r.val : ℝ))
        (fun j => recallRawEmbedding (tokens j)) 0 i -
        (recallHeadValue (recallSymbolCode key) : EucSpace recallConfig.head_dim)‖ ≤
      ((T - 1 : ℕ) : ℝ) * Real.exp (-adjacentGap alpha) * 4 := by
  rw [recall_predecessor_head eps alpha heps]
  have hs : recallHeadValue (recallSlotRead 1 (recallRawEmbedding (tokens i))) = 0 := by
    rw [hvalue]
    exact recall_raw_value_self_zero value
  have h := adjacent_head_copy alpha eps 4 hclip (by norm_num) hT
    (fun j => recallHeadValue (recallSlotRead 1 (recallRawEmbedding (tokens j)))) i selected hprev hs
    (fun j _ => recall_raw_head_value_diameter (tokens j) (tokens selected))
  rw [hkey, recall_raw_key_value] at h
  exact h

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (2 : ℕ) ≤ 128 ∧
    (0 : Fin 2).val + 1 = (1 : Fin 2).val ∧
    (fun j : Fin 2 => if j = 0 then recallKeyId 0 else recallValueId 0) 0 = recallKeyId 0 ∧
    (fun j : Fin 2 => if j = 0 then recallKeyId 0 else recallValueId 0) 1 = recallValueId 0 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by simp, by simp⟩

/-- A raw value's own compact code remains in the genuine attention residual alongside its copied key.
Source: protected original coordinates 9..16, not an externally supplied key/value pair representation. -/
theorem recall_first_residual_value_code (eps alpha : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T)
    (symbol : Fin 256) (htoken : tokens i = recallValueId symbol) :
    recallSlotRead 9 (recallRawEmbedding (tokens i) +
        attnSubLayer recallConfig (recallFirstAttention eps alpha) eps positions
          (fun j => recallRawEmbedding (tokens j)) i) = recallSymbolCode symbol := by
  ext c
  rw [recallSlotRead_at]
  have hc : (recallSlotIndex 9 c).val < 18 ∨ 27 ≤ (recallSlotIndex 9 c).val := by
    left
    change 9 + c.val < 18
    have hb := c.isLt
    omega
  rw [recall_first_attn_protected eps alpha positions (fun j => recallRawEmbedding (tokens j)) i _ hc]
  rw [← recallSlotRead_at, htoken, recallRawEmbedding_value, (recallValueEmbedding_reads symbol).2]

example : (fun _ : Fin 1 => recallValueId 0) 0 = recallValueId 0 := by rfl

/-- A raw query/key code is also preserved through the same real attention residual.
Source: protected original coordinates 1..8, allowing second-block query matching without a prepared query feature. -/
theorem recall_first_residual_key_code (eps alpha : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T)
    (symbol : Fin 256) (htoken : tokens i = recallKeyId symbol) :
    recallSlotRead 1 (recallRawEmbedding (tokens i) +
        attnSubLayer recallConfig (recallFirstAttention eps alpha) eps positions
          (fun j => recallRawEmbedding (tokens j)) i) = recallSymbolCode symbol := by
  ext c
  rw [recallSlotRead_at]
  have hc : (recallSlotIndex 1 c).val < 18 ∨ 27 ≤ (recallSlotIndex 1 c).val := by
    left
    change 1 + c.val < 18
    have hb := c.isLt
    omega
  rw [recall_first_attn_protected eps alpha positions (fun j => recallRawEmbedding (tokens j)) i _ hc]
  rw [← recallSlotRead_at, htoken, recallRawEmbedding_key, (recallKeyEmbedding_reads symbol).1]

example : (fun _ : Fin 1 => recallKeyId 0) 0 = recallKeyId 0 := by rfl

/-- Extracting the actual copied eight coordinates preserves the derived raw binding error budget.
Source: original head output and the proven nonexpansive read of its true compact value coordinates. -/
theorem recall_raw_compact_copy (eps alpha : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1)
    {T : ℕ} (hT : T ≤ 128) (tokens : Fin T → Fin recallConfig.vocab_size)
    (i selected : Fin T) (hprev : selected.val + 1 = i.val) (key value : Fin 256)
    (hkey : tokens selected = recallKeyId key) (hvalue : tokens i = recallValueId value) :
    ‖recallHeadRead (headAt recallConfig (recallFirstAttention eps alpha) eps (fun r => (r.val : ℝ))
        (fun j => recallRawEmbedding (tokens j)) 0 i) - recallSymbolCode key‖ ≤
      ((T - 1 : ℕ) : ℝ) * Real.exp (-adjacentGap alpha) * 4 := by
  have h := recallHeadRead_error
    (headAt recallConfig (recallFirstAttention eps alpha) eps (fun r => (r.val : ℝ))
      (fun j => recallRawEmbedding (tokens j)) 0 i) (recallHeadValue (recallSymbolCode key))
  rw [recallHeadRead_value] at h
  exact h.trans (recall_raw_predecessor_copy eps alpha heps hclip hT tokens i selected hprev key value hkey hvalue)

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (2 : ℕ) ≤ 128 ∧
    (0 : Fin 2).val + 1 = (1 : Fin 2).val ∧
    (fun j : Fin 2 => if j = 0 then recallKeyId 0 else recallValueId 0) 0 = recallKeyId 0 ∧
    (fun j : Fin 2 => if j = 0 then recallKeyId 0 else recallValueId 0) 1 = recallValueId 0 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by simp, by simp⟩

/-- No raw embedding contains a precomputed predecessor key in the actual destination slot.
Source: all original raw branches and the initially empty coordinates 18..25. -/
theorem recall_raw_copy_slot_zero (token : Fin recallConfig.vocab_size) :
    recallSlotRead 18 (recallRawEmbedding token) = 0 := by
  ext c
  rw [recallSlotRead_at]
  have hc : 18 ≤ (recallSlotIndex 18 c).val ∧ (recallSlotIndex 18 c).val < 35 := by
    change 18 ≤ 18 + c.val ∧ 18 + c.val < 35
    have hb := c.isLt
    omega
  rw [recallRawEmbedding_fresh token _ hc, PiLp.zero_apply]

/-- The genuine W_o writes precisely the extracted head-zero code into the actual copy slot.
Source: the simultaneous output-matrix/head-merge calculation; the BOS marker is disjoint. -/
theorem recallFirstOutput_copy_slot (heads : Fin recallConfig.n_heads → EucSpace recallConfig.head_dim) :
    recallSlotRead 18 (recallFirstOutput (headMerge recallConfig heads)) = recallHeadRead (heads 0) := by
  rw [recallFirstOutput_merge, map_add, recallSlotRead_write, map_smul]
  have hz := recallSlotRead_unit_outside 18 26 (by decide)
  change recallHeadRead (heads 0) +
    (heads 1 ⟨0, by decide⟩) • recallSlotRead 18 (EuclideanSpace.single 26 1) = _
  rw [hz, smul_zero, add_zero]

/-- The copied feature is present in the true first attention residual, not just in an isolated head.
Source: original attnSubLayer/headMerge/W_o and raw initially empty destination coordinates. -/
theorem recall_first_residual_copy_slot (eps alpha : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    recallSlotRead 18 (recallRawEmbedding (tokens i) +
        attnSubLayer recallConfig (recallFirstAttention eps alpha) eps positions
          (fun j => recallRawEmbedding (tokens j)) i) =
      recallHeadRead (headAt recallConfig (recallFirstAttention eps alpha) eps positions
        (fun j => recallRawEmbedding (tokens j)) 0 i) := by
  rw [map_add, recall_raw_copy_slot_zero, zero_add, attnSubLayer_eq_heads]
  change recallSlotRead 18 (recallFirstOutput (headMerge recallConfig _)) = _
  exact recallFirstOutput_copy_slot _

/-- A genuine raw adjacent write is represented by its copied key in the true pre-FFN residual with a derived error.
Source: every original first-attention operation, with no encoded-key or score-gap premise. -/
theorem recall_first_residual_binds (eps alpha : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1)
    {T : ℕ} (hT : T ≤ 128) (tokens : Fin T → Fin recallConfig.vocab_size)
    (i selected : Fin T) (hprev : selected.val + 1 = i.val) (key value : Fin 256)
    (hkey : tokens selected = recallKeyId key) (hvalue : tokens i = recallValueId value) :
    ‖recallSlotRead 18 (recallRawEmbedding (tokens i) +
        attnSubLayer recallConfig (recallFirstAttention eps alpha) eps (fun r => (r.val : ℝ))
          (fun j => recallRawEmbedding (tokens j)) i) - recallSymbolCode key‖ ≤
      ((T - 1 : ℕ) : ℝ) * Real.exp (-adjacentGap alpha) * 4 := by
  rw [recall_first_residual_copy_slot]
  exact recall_raw_compact_copy eps alpha heps hclip hT tokens i selected hprev key value hkey hvalue

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (2 : ℕ) ≤ 128 ∧
    (0 : Fin 2).val + 1 = (1 : Fin 2).val ∧
    (fun j : Fin 2 => if j = 0 then recallKeyId 0 else recallValueId 0) 0 = recallKeyId 0 ∧
    (fun j : Fin 2 => if j = 0 then recallKeyId 0 else recallValueId 0) 1 = recallValueId 0 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by simp, by simp⟩

end Transformer.GPTMini.Semantics
