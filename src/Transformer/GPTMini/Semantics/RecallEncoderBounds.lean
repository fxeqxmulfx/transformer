import Transformer.GPTMini.Semantics.RecallGateGain

/-!
# Bound the complete original encoder from raw table inputs

Source: raw MQAR's BOS/alphabet and adjacent table writes at cbafbe9,
with the actual original shared attention/FFN/residual at f11b6e2.
The conditional gate norm bounds are discharged at every raw
position: BOS and keys turn the gate off, values inside the table
turn it on, and later values turn it off. No encoded table indicator
or gate-sign hypothesis occurs in the uniform raw-block bound.

The finite fixed gain makes every raw table amplitude at least one.
At the derived finite predecessor temperature, each actual bound
table key also has norm at least one. Both lower bounds are derived
from raw IDs and adjacency rather than supplied as saturated-key
premises. Raw query/value slots survive the whole block. One shared
second matrix gain can now compensate the finite upper norm bound;
full validated-parser coupling and second routing/readout remain.
-/

namespace Transformer.GPTMini.Semantics

/-- Every genuine raw encoder state has a finite uniform upper norm, with no encoded gate-separation premise.
Source: actual BOS/key exclusion and inside/after-table value decisions, including both original residuals. -/
theorem recall_raw_encoder_norm (P : ℕ) (eps alpha beta : ℝ) (heps : 0 ≤ eps) (hbeta : 0 ≤ beta) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (first i : Fin T)
    (hfirst : first.val = 0) (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol) :
    ‖blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
      (fun j => recallRawEmbedding (tokens j)) i‖ ≤ 9 + 512 * beta := by
  have hoff (hgate : recallGateForm P (recallFirstState eps alpha positions tokens i) ≤ -recallTableMargin P) :
      ‖blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
        (fun j => recallRawEmbedding (tokens j)) i‖ ≤ 9 + 512 * beta := by
    rw [recallEncoderBlock_off P eps alpha beta heps positions tokens i hgate]
    linarith [recallFirstState_norm_le eps alpha heps positions tokens i]
  by_cases hi : i = first
  · apply hoff
    exact recallGateForm_bos P eps alpha heps positions tokens i (hi ▸ hbos)
  · obtain ⟨symbol, hk | hv⟩ := hrange i hi
    · exact hoff (recallGateForm_key P eps alpha heps positions tokens i symbol hk)
    · by_cases hin : i.val ≤ 2 * P
      · exact recallEncoderBlock_on_norm P eps alpha beta heps hbeta positions tokens i
          (recallGateForm_inside P eps alpha heps positions tokens first i hfirst hi hbos hrange symbol hv hin)
      · exact hoff (recallGateForm_after P eps alpha heps positions tokens first i hfirst hi hbos hrange symbol hv
          (by omega))

example : (0 : ℝ) ≤ 1 / 100000 ∧ (0 : ℝ) ≤ 1 ∧ (0 : Fin 3).val = 0 ∧
    recallPairWitness 0 0 0 = ⟨1, by decide⟩ ∧ (∀ j : Fin 3, j ≠ 0 → ∃ symbol : Fin 256,
      recallPairWitness 0 0 j = recallKeyId symbol ∨ recallPairWitness 0 0 j = recallValueId symbol) := by
  refine ⟨by norm_num, by norm_num, by decide, by simp [recallPairWitness], ?_⟩
  intro j hj
  by_cases h : j = 1
  · exact ⟨0, Or.inl (by simp [recallPairWitness, h])⟩
  · exact ⟨0, Or.inr (by simp [recallPairWitness, hj, h])⟩

/-- Every genuine raw table value has amplitude at least one with the fixed finite table-size gain.
Source: actual raw gate decision and true prenorm bound, without an amplitude or prepared-key premise. -/
theorem recall_raw_amplitude_lower (P : ℕ) (eps alpha : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (first i : Fin T)
    (hfirst : first.val = 0) (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol)
    (value : Fin 256) (hvalue : tokens i = recallValueId value) (hinside : i.val ≤ 2 * P) :
    1 ≤ recallGateAmplitude P (recallEncoderGain P) eps (recallFirstState eps alpha positions tokens i) := by
  apply recallGateAmplitude_lower P eps alpha heps hclip positions tokens i
  exact recallGateForm_inside P eps alpha heps positions tokens first i hfirst
    (recall_raw_value_ne_first tokens first i value hbos hvalue) hbos hrange value hvalue hinside

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (0 : Fin 3).val = 0 ∧
    recallPairWitness 0 0 0 = ⟨1, by decide⟩ ∧ (∀ j : Fin 3, j ≠ 0 → ∃ symbol : Fin 256,
      recallPairWitness 0 0 j = recallKeyId symbol ∨ recallPairWitness 0 0 j = recallValueId symbol) ∧
    recallPairWitness 0 0 2 = recallValueId 0 ∧ (2 : Fin 3).val ≤ 2 * 1 := by
  refine ⟨by norm_num, by norm_num, by decide, by simp [recallPairWitness], ?_,
    by simp [recallPairWitness], by decide⟩
  intro j hj
  by_cases h : j = 1
  · exact ⟨0, Or.inl (by simp [recallPairWitness, h])⟩
  · exact ⟨0, Or.inr (by simp [recallPairWitness, hj, h])⟩

/-- The stored key of every actual adjacent table write has norm at least one under one fixed finite block.
Source: derived predecessor accuracy, actual raw table gate and shared gain; no encoded norm premise is supplied. -/
theorem recall_raw_binding_unit_lower (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ}
    (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size) (first i selected : Fin T)
    (hfirst : first.val = 0) (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol)
    (hprev : selected.val + 1 = i.val) (key value : Fin 256)
    (hkey : tokens selected = recallKeyId key) (hvalue : tokens i = recallValueId value) (hinside : i.val ≤ 2 * P) :
    1 ≤ ‖recallSlotRead 27 (blockForward recallConfig
      (recallEncoderBlock P eps (recallCopyTemperature recallKeyTolerance) (recallEncoderGain P)) eps
      (fun r => (r.val : ℝ)) (fun j => recallRawEmbedding (tokens j)) i)‖ := by
  rw [recall_raw_gate_inside P eps (recallCopyTemperature recallKeyTolerance) (recallEncoderGain P) heps _
    tokens first i hfirst (recall_raw_value_ne_first tokens first i value hbos hvalue) hbos hrange value hvalue hinside]
  apply recallScaledCode_norm_lower
  · exact recall_raw_amplitude_lower P eps (recallCopyTemperature recallKeyTolerance) heps hclip _ tokens
      first i hfirst hbos hrange value hvalue hinside
  · exact recall_first_state_copy_nonzero recallKeyTolerance eps recallKeyTolerance_pos recallKeyTolerance_le_one
      heps hclip hT tokens i selected hprev key value hkey hvalue

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (3 : ℕ) ≤ 64 ∧ (0 : Fin 3).val = 0 ∧
    recallPairWitness 0 0 0 = ⟨1, by decide⟩ ∧ (∀ j : Fin 3, j ≠ 0 → ∃ symbol : Fin 256,
      recallPairWitness 0 0 j = recallKeyId symbol ∨ recallPairWitness 0 0 j = recallValueId symbol) ∧
    (1 : Fin 3).val + 1 = (2 : Fin 3).val ∧ recallPairWitness 0 0 1 = recallKeyId 0 ∧
    recallPairWitness 0 0 2 = recallValueId 0 ∧ (2 : Fin 3).val ≤ 2 * 1 := by
  refine ⟨by norm_num, by norm_num, by decide, by decide, by simp [recallPairWitness], ?_, by decide,
    by simp [recallPairWitness], by simp [recallPairWitness], by decide⟩
  intro j hj
  by_cases h : j = 1
  · exact ⟨0, Or.inl (by simp [recallPairWitness, h])⟩
  · exact ⟨0, Or.inr (by simp [recallPairWitness, hj, h])⟩

/-- Any raw-code interval below coordinate eighteen is preserved by the complete genuine first block.
Source: both actual residual output supports, with ordinary coordinate readback for every real raw entry. -/
theorem recall_raw_encoder_reads (P : ℕ) (eps alpha beta : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) (offset : Fin 57)
    (hoffset : offset.val + 8 ≤ 18) :
    recallSlotRead offset (blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
      (fun j => recallRawEmbedding (tokens j)) i) = recallSlotRead offset (recallRawEmbedding (tokens i)) := by
  ext c
  rw [recallSlotRead_at, recallSlotRead_at]
  apply recallEncoderBlock_raw_protected P eps alpha beta positions tokens i
  left
  change offset.val + c.val < 18
  have hc := c.isLt
  omega

example : (1 : Fin 57).val + 8 ≤ 18 ∧ (9 : Fin 57).val + 8 ≤ 18 := by decide

/-- A genuine raw query retains exactly zero own-value projection after the complete encoder, needed by second-head XSA.
Source: true disjoint raw key embedding and both actual residual supports, without a prepared self-value premise. -/
theorem recall_raw_encoder_query_value_zero (P : ℕ) (eps alpha beta : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T)
    (key : Fin 256) (htoken : tokens i = recallKeyId key) :
    recallSlotRead 9 (blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
      (fun j => recallRawEmbedding (tokens j)) i) = 0 := by
  rw [recall_raw_encoder_reads P eps alpha beta positions tokens i 9 (by decide), htoken, recallRawEmbedding_key]
  exact (recallKeyEmbedding_reads key).2

example : recallPairWitness 0 0 1 = recallKeyId 0 := by simp [recallPairWitness]

/-- Every actual raw value has zero query-code projection after the same full encoder, simultaneously with its bound key.
Source: the independent original key/value slots and their verified preservation through both residual additions. -/
theorem recall_raw_encoder_value_query_zero (P : ℕ) (eps alpha beta : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T)
    (value : Fin 256) (htoken : tokens i = recallValueId value) :
    recallSlotRead 1 (blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
      (fun j => recallRawEmbedding (tokens j)) i) = 0 := by
  rw [recall_raw_encoder_reads P eps alpha beta positions tokens i 1 (by decide), htoken, recallRawEmbedding_value]
  exact (recallValueEmbedding_reads value).1

example : recallPairWitness 0 0 2 = recallValueId 0 := by simp [recallPairWitness]

end Transformer.GPTMini.Semantics
