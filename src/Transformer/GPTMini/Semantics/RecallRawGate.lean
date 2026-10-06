import Transformer.GPTMini.Semantics.RecallGateBlock

/-!
# Gate real raw table records in the full original first block

Source: MQAR's raw IDs and fixed 2P-position table at cbafbe9,
and original Block.forward at f11b6e2. Actual keys and BOS have
zero gated-key output. A raw value after the table also has exactly
zero gated output, even when a preceding query produced a copied
key. A raw table value retains the entire compact predecessor copy
times its actual positive homogeneous gate coefficient.

These outcomes include prenorm, simultaneous fused QKV/heads/W_o,
the sixteen-unit ordinary FFN and both true residual additions.
Only raw alphabet/position premises are supplied: no prepared
table indicator, copied key, gate sign or correct logits occur as
inputs. Coupling the alphabet/boundary to the complete validated
Basis parser and robust final latest-write retrieval is still needed.
-/

namespace Transformer.GPTMini.Semantics

/-- Every actual raw key/query has exactly zero gated-key slot after the real whole block.
Source: derived key-type exclusion, actual FFN zero output and the initially empty destination. -/
theorem recall_raw_gate_key (P : ℕ) (eps alpha beta : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T)
    (symbol : Fin 256) (htoken : tokens i = recallKeyId symbol) :
    recallSlotRead 27 (blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
      (fun j => recallRawEmbedding (tokens j)) i) = 0 := by
  rw [recallEncoderBlock_off P eps alpha beta heps positions tokens i
    (recallGateForm_key P eps alpha heps positions tokens i symbol htoken)]
  exact recallFirstState_gate_slot_zero eps alpha positions tokens i

example : (0 : ℝ) ≤ 0 ∧ (fun _ : Fin 1 => recallKeyId 0) 0 = recallKeyId 0 := by
  exact ⟨by norm_num, rfl⟩

/-- Actual BOS is excluded by the full block as well, including its own initial position.
Source: derived raw reserved-type exclusion, with no later-position marker assumption at BOS. -/
theorem recall_raw_gate_bos (P : ℕ) (eps alpha beta : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T)
    (htoken : tokens i = (⟨1, by decide⟩ : Fin recallConfig.vocab_size)) :
    recallSlotRead 27 (blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
      (fun j => recallRawEmbedding (tokens j)) i) = 0 := by
  rw [recallEncoderBlock_off P eps alpha beta heps positions tokens i
    (recallGateForm_bos P eps alpha heps positions tokens i htoken)]
  exact recallFirstState_gate_slot_zero eps alpha positions tokens i

example : (0 : ℝ) ≤ 0 ∧
    (fun _ : Fin 1 => (⟨1, by decide⟩ : Fin recallConfig.vocab_size)) 0 = ⟨1, by decide⟩ := by
  exact ⟨by norm_num, rfl⟩

/-- A later filler/value cannot create a false table write in the genuine first block.
Source: raw causal BOS computation, fixed table boundary, actual FFN and both residual additions. -/
theorem recall_raw_gate_after (P : ℕ) (eps alpha beta : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (first i : Fin T)
    (hfirst : first.val = 0) (hne : i ≠ first)
    (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol)
    (value : Fin 256) (hvalue : tokens i = recallValueId value) (hafter : 2 * P < i.val) :
    recallSlotRead 27 (blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
      (fun j => recallRawEmbedding (tokens j)) i) = 0 := by
  rw [recallEncoderBlock_off P eps alpha beta heps positions tokens i
    (recallGateForm_after P eps alpha heps positions tokens first i hfirst hne hbos hrange value hvalue hafter)]
  exact recallFirstState_gate_slot_zero eps alpha positions tokens i

example : (0 : ℝ) ≤ 0 ∧ (0 : Fin 2).val = 0 ∧ (1 : Fin 2) ≠ 0 ∧
    (fun j : Fin 2 => if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) 0 =
      ⟨1, by decide⟩ ∧ (∀ j : Fin 2, j ≠ 0 → ∃ symbol : Fin 256,
        (if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallKeyId symbol ∨
        (if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallValueId symbol) ∧
    (if (1 : Fin 2) = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallValueId 0 ∧
    2 * (0 : ℕ) < (1 : Fin 2).val := by
  refine ⟨by norm_num, by decide, by decide, by simp, ?_, by simp, by decide⟩
  intro j hj
  exact ⟨0, Or.inr (by simp only [ite_eq_right hj])⟩

/-- A true raw table value stores the whole genuine copied key times its actual derived gate amplitude.
Source: raw BOS/alphabet/boundary inputs determine the gate; the real block computes all encoded coordinates. -/
theorem recall_raw_gate_inside (P : ℕ) (eps alpha beta : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (first i : Fin T)
    (hfirst : first.val = 0) (hne : i ≠ first)
    (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol)
    (value : Fin 256) (hvalue : tokens i = recallValueId value) (hinside : i.val ≤ 2 * P) :
    recallSlotRead 27 (blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
        (fun j => recallRawEmbedding (tokens j)) i) =
      recallGateAmplitude P beta eps (recallFirstState eps alpha positions tokens i) •
        recallSlotRead 18 (recallFirstState eps alpha positions tokens i) := by
  exact recallEncoderBlock_on P eps alpha beta heps positions tokens i
    (recallGateForm_inside P eps alpha heps positions tokens first i hfirst hne hbos hrange value hvalue hinside)

example : (0 : ℝ) ≤ 0 ∧ (0 : Fin 2).val = 0 ∧ (1 : Fin 2) ≠ 0 ∧
    (fun j : Fin 2 => if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) 0 =
      ⟨1, by decide⟩ ∧ (∀ j : Fin 2, j ≠ 0 → ∃ symbol : Fin 256,
        (if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallKeyId symbol ∨
        (if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallValueId symbol) ∧
    (if (1 : Fin 2) = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallValueId 0 ∧
    (1 : Fin 2).val ≤ 2 * 1 := by
  refine ⟨by norm_num, by decide, by decide, by simp, ?_, by simp, by decide⟩
  intro j hj
  exact ⟨0, Or.inr (by simp only [ite_eq_right hj])⟩

/-- The genuine simultaneous full block retains the raw query/key code for its second attention layer.
Source: both original residuals preserve the complete disjoint raw key interval 1..8. -/
theorem recall_raw_gate_key_code (P : ℕ) (eps alpha beta : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T)
    (symbol : Fin 256) (htoken : tokens i = recallKeyId symbol) :
    recallSlotRead 1 (blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
      (fun j => recallRawEmbedding (tokens j)) i) = recallSymbolCode symbol := by
  ext c
  rw [recallSlotRead_at, recallEncoderBlock_raw_protected P eps alpha beta positions tokens i _ (Or.inl (by
    change 1 + c.val < 18
    have hc := c.isLt
    omega)), ← recallSlotRead_at, htoken, recallRawEmbedding_key, (recallKeyEmbedding_reads symbol).1]

example : (fun _ : Fin 1 => recallKeyId 0) 0 = recallKeyId 0 := by rfl

/-- The associated raw value code survives the real gate without being merged with its copied key.
Source: the actual protected value interval 9..16 and both original block residuals. -/
theorem recall_raw_gate_value_code (P : ℕ) (eps alpha beta : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T)
    (symbol : Fin 256) (htoken : tokens i = recallValueId symbol) :
    recallSlotRead 9 (blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
      (fun j => recallRawEmbedding (tokens j)) i) = recallSymbolCode symbol := by
  ext c
  rw [recallSlotRead_at, recallEncoderBlock_raw_protected P eps alpha beta positions tokens i _ (Or.inl (by
    change 9 + c.val < 18
    have hc := c.isLt
    omega)), ← recallSlotRead_at, htoken, recallRawEmbedding_value, (recallValueEmbedding_reads symbol).2]

example : (fun _ : Fin 1 => recallValueId 0) 0 = recallValueId 0 := by rfl

/-- The actual stored table-key amplitude is strictly positive, derived from raw input IDs rather than assumed.
Source: true RMS denominator, finite positive FFN gain and the genuine raw BOS/table-boundary gate margin. -/
theorem recall_raw_gate_amplitude_pos (P : ℕ) (eps alpha beta : ℝ) (heps : 0 < eps) (hbeta : 0 < beta) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (first i : Fin T)
    (hfirst : first.val = 0) (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol)
    (value : Fin 256) (hvalue : tokens i = recallValueId value) (hinside : i.val ≤ 2 * P) :
    0 < recallGateAmplitude P beta eps (recallFirstState eps alpha positions tokens i) := by
  apply recallGateAmplitude_pos P beta eps hbeta heps
  exact recallGateForm_inside P eps alpha heps.le positions tokens first i hfirst
    (recall_raw_value_ne_first tokens first i value hbos hvalue) hbos hrange value hvalue hinside

example : (0 : ℝ) < 1 / 100000 ∧ (0 : ℝ) < 1 ∧ (0 : Fin 2).val = 0 ∧
    (fun j : Fin 2 => if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) 0 =
      ⟨1, by decide⟩ ∧ (∀ j : Fin 2, j ≠ 0 → ∃ symbol : Fin 256,
        (if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallKeyId symbol ∨
        (if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallValueId symbol) ∧
    (if (1 : Fin 2) = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) = recallValueId 0 ∧
    (1 : Fin 2).val ≤ 2 * 1 := by
  refine ⟨by norm_num, by norm_num, by decide, by simp, ?_, by simp, by decide⟩
  intro j hj
  exact ⟨0, Or.inr (by simp only [ite_eq_right hj])⟩

end Transformer.GPTMini.Semantics
