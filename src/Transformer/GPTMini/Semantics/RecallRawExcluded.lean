import Transformer.GPTMini.Semantics.RecallRawScoreGap

/-!
# Excluded raw positions cannot defeat the selected table record

Source: MQAR's BOS/table/query layout at cbafbe9 and the unchanged
encoder/second-head operators at f11b6e2. BOS and post-table value
fillers have exactly zero matching score after the real table gate,
prenorm, fused K projection, RoPE and QKNorm. Earlier raw query/key
competitors already have the same zero-score guarantee.

The actual imperfect matching table record has score at least the
positive retained gap. This derives separation from every zero-score
nonrecord; it does not assume the desired route is a maximizer. True
table competitors are handled independently by RecallRawScoreGap.
The full causal-row theorem still has to combine these cases and
bind the raw adjacency/latest-write conditions to validated parsing.
Actual finite retrieval and tied readout are subsequent obligations.
-/

namespace Transformer.GPTMini.Semantics

/-- A genuinely empty encoder table slot gives zero K after the actual next prenorm and shared fused matrix.
Source: original linear RMS scaling, ordinary second K rows and empty-slot gate laws. -/
theorem recallMatchKey_slot_zero (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (record : Fin T)
    (hslot : recallSlotRead 27 (recallMatchState P eps tokens record) = 0) :
    recallMatchKey P eps tokens record = 0 := by
  unfold recallMatchKey recallMatchQKV
  apply recallSecondQKV_key_zero
  rw [recallResidualScale_rms, map_smul, hslot, smul_zero]

example : recallSlotRead 27 (recallMatchState 0 0 (fun _ : Fin 1 => recallKeyId 0) 0) = 0 := by
  dsimp only [recallMatchState, recallEncodedState]
  exact recall_raw_gate_key 0 0 (recallCopyTemperature recallKeyTolerance) (recallEncoderGain 0)
    (by norm_num) (fun r : Fin 1 => (r.val : ℝ)) (fun _ : Fin 1 => recallKeyId 0) 0 0 rfl

/-- Raw BOS has exactly zero actual matching score, including its initial own position.
Source: actual reserved-type gate exclusion followed by genuine prenorm, K rows, rotary map and clipped score. -/
theorem recallMatchScore_raw_bos_zero (P : ℕ) (alpha eps : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query record : Fin T)
    (hbos : tokens record = (⟨1, by decide⟩ : Fin recallConfig.vocab_size)) :
    recallMatchScore P alpha eps tokens query record = 0 := by
  apply recallMatchScore_key_zero
  apply recallMatchKey_slot_zero
  exact recall_raw_gate_bos P eps _ (recallEncoderGain P) heps _ tokens record hbos

example : (0 : ℝ) ≤ 1 / 100000 ∧ recallPairWitness 0 0 0 = ⟨1, by decide⟩ := by
  exact ⟨by norm_num, by simp [recallPairWitness]⟩

/-- A genuine filler/value after the completed table has zero score even if a preceding query produced a copied key.
Source: the actual raw table cutoff in the full first block and true second prenorm/projection/score. -/
theorem recallMatchScore_raw_after_zero (P : ℕ) (alpha eps : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (first query record : Fin T)
    (hfirst : first.val = 0) (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol)
    (value : Fin 256) (hvalue : tokens record = recallValueId value) (hafter : 2 * P < record.val) :
    recallMatchScore P alpha eps tokens query record = 0 := by
  apply recallMatchScore_key_zero
  apply recallMatchKey_slot_zero
  exact recall_raw_gate_after P eps _ (recallEncoderGain P) heps _ tokens first record hfirst
    (recall_raw_value_ne_first tokens first record value hbos hvalue) hbos hrange value hvalue hafter

example : (0 : ℝ) ≤ 1 / 100000 ∧ (0 : Fin 3).val = 0 ∧ recallPairWitness 0 0 0 = ⟨1, by decide⟩ ∧
    (∀ j : Fin 3, j ≠ 0 → ∃ symbol : Fin 256,
      recallPairWitness 0 0 j = recallKeyId symbol ∨ recallPairWitness 0 0 j = recallValueId symbol) ∧
    recallPairWitness 0 0 2 = recallValueId 0 ∧ 2 * (0 : ℕ) < (2 : Fin 3).val := by
  refine ⟨by norm_num, by decide, by simp [recallPairWitness], ?_, by simp [recallPairWitness], by decide⟩
  intro j hj
  by_cases h : j = 1
  · exact ⟨0, Or.inl (by simp [recallPairWitness, h])⟩
  · exact ⟨0, Or.inr (by simp [recallPairWitness, hj, h])⟩

/-- A real raw matching table record has score at least the retained positive gap despite its actual copy error.
Source: derived true table score error and original rotary matching-versus-zero margin, with context bounds discharged. -/
theorem recall_raw_selected_score_lower (P : ℕ) (alpha eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ}
    (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size) (first query selected previous : Fin T)
    (hfirst : first.val = 0) (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol)
    (key value : Fin 256) (hquery : tokens query = recallKeyId key)
    (hprev : previous.val + 1 = selected.val) (hkey : tokens previous = recallKeyId key)
    (hvalue : tokens selected = recallValueId value) (hinside : selected.val ≤ 2 * P) :
    (0 : ℝ) ≤ recallMatchScore P alpha eps tokens query selected - recallMatchGap alpha := by
  apply recallScore_zero_gap_of_error alpha _
    (score alpha 1 (applyRope 16 10000 (query.val : ℝ) (recallRotaryCode (recallDigit key)))
      (applyRope 16 10000 (selected.val : ℝ) (recallRotaryCode (recallDigit key))))
  · exact recall_raw_score_error P alpha eps heps hclip hT tokens first query selected previous
      hfirst hbos hrange key key value hquery hprev hkey hvalue hinside
  · have hg := recall_zero_latest_gap alpha 1 (by norm_num) (recallDigit key)
      (query.val : ℝ) (selected.val : ℝ) (recall_index_distance hT selected query)
    have hz : score alpha 1 (applyRope 16 10000 (query.val : ℝ) (recallRotaryCode (recallDigit key))) 0 = 0 := by
      simp only [score, normL2, smul_zero, inner_zero_right, mul_zero]
    rw [hz] at hg
    linarith

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (3 : ℕ) ≤ 64 ∧ (0 : Fin 3).val = 0 ∧
    recallPairWitness 0 0 0 = ⟨1, by decide⟩ ∧ (∀ j : Fin 3, j ≠ 0 → ∃ symbol : Fin 256,
      recallPairWitness 0 0 j = recallKeyId symbol ∨ recallPairWitness 0 0 j = recallValueId symbol) ∧
    recallPairWitness 0 0 1 = recallKeyId 0 ∧ (1 : Fin 3).val + 1 = (2 : Fin 3).val ∧
    recallPairWitness 0 0 1 = recallKeyId 0 ∧ recallPairWitness 0 0 2 = recallValueId 0 ∧
    (2 : Fin 3).val ≤ 2 * 1 := by
  refine ⟨by norm_num, by norm_num, by decide, by decide, by simp [recallPairWitness], ?_,
    by simp [recallPairWitness], by decide, by simp [recallPairWitness], by simp [recallPairWitness], by decide⟩
  intro j hj
  by_cases h : j = 1
  · exact ⟨0, Or.inl (by simp [recallPairWitness, h])⟩
  · exact ⟨0, Or.inr (by simp [recallPairWitness, hj, h])⟩

/-- The raw protected value slot retains its original norm-two code in the real matching state.
Source: the full first-block raw slot preservation, at the fixed shared gate gain and copy temperature. -/
theorem recallMatchState_value_code (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (record : Fin T) (value : Fin 256)
    (hvalue : tokens record = recallValueId value) :
    recallSlotRead 9 (recallMatchState P eps tokens record) = recallSymbolCode value := by
  exact recall_raw_gate_value_code P eps _ (recallEncoderGain P) _ tokens record value hvalue

example : recallPairWitness 0 0 2 = recallValueId 0 := by simp [recallPairWitness]

/-- The actual selected value is its retained raw symbol with the genuine position-dependent RMS multiplier.
Source: the true fused V projection and protected raw value slot; no common or prepared value amplitude is assumed. -/
theorem recallMatchValue_value_code (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (record : Fin T) (value : Fin 256)
    (hvalue : tokens record = recallValueId value) :
    recallMatchValue P eps tokens record = recallResidualScale eps (recallMatchState P eps tokens record) •
      recallHeadValue (recallSymbolCode value) := by
  unfold recallMatchValue recallMatchQKV
  rw [recallSecondQKV_rms_value, recallMatchState_value_code P eps tokens record value hvalue]

example : recallPairWitness 0 0 2 = recallValueId 0 := by simp [recallPairWitness]

/-- Every raw vocabulary entry has value-slot norm at most two, including keys and reserved entries.
Source: the complete raw embedding table and both disjoint slot readback laws. -/
theorem recall_raw_value_slot_bound (token : Fin recallConfig.vocab_size) :
    ‖recallSlotRead 9 (recallRawEmbedding token)‖ ≤ 2 := by
  unfold recallRawEmbedding
  split_ifs
  · rw [(recallKeyEmbedding_reads _).2, norm_zero]; norm_num
  · rw [(recallValueEmbedding_reads _).2, recallSymbolCode, recallCode_norm]
  · rw [recallReservedEmbedding_reads.2, norm_zero]; norm_num

/-- Every genuine second-head value has norm at most sixteen, with no prepared value-norm hypothesis.
Source: actual prenorm scale at most eight, protected raw value slot and true fused V/head layout. -/
theorem recallMatchValue_norm_le (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (record : Fin T) :
    ‖recallMatchValue P eps tokens record‖ ≤ 16 := by
  have hs := recallEncoderBlock_scale_upper P eps (recallCopyTemperature recallKeyTolerance)
    (recallEncoderGain P) heps (fun r => (r.val : ℝ)) tokens record
  change 0 < recallResidualScale eps (recallMatchState P eps tokens record) ∧
    recallResidualScale eps (recallMatchState P eps tokens record) ≤ 8 at hs
  unfold recallMatchValue recallMatchQKV
  rw [recallSecondQKV_rms_value]
  change ‖(recallResidualScale eps (recallMatchState P eps tokens record) •
    recallHeadValue (recallSlotRead 9 (recallMatchState P eps tokens record)) : EucSpace recallConfig.head_dim)‖ ≤ 16
  rw [norm_smul_of_nonneg hs.1.le, recallHeadValue_norm]
  have hr := recall_raw_encoder_reads P eps (recallCopyTemperature recallKeyTolerance) (recallEncoderGain P)
    (fun r => (r.val : ℝ)) tokens record 9 (by decide)
  change recallSlotRead 9 (recallMatchState P eps tokens record) = recallSlotRead 9 (recallRawEmbedding (tokens record)) at hr
  rw [hr]
  exact (mul_le_mul hs.2 (recall_raw_value_slot_bound (tokens record)) (norm_nonneg _) (by norm_num)).trans (by norm_num)

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

end Transformer.GPTMini.Semantics
