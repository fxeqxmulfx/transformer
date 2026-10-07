import Transformer.GPTMini.Semantics.RecallRetrievalBlock

/-!
# A uniformly accurate genuine final recall state and protected raw readout channels

Source: original two-block residual/RMS/QKV/W_o/ReLU2 computation
at f11b6e2 and raw adjacent MQAR writes at cbafbe9. The final state
below evaluates both actual original blocks. Its retrieved value
has a genuine position-dependent positive prenorm amplitude; no
common selected-value scale is assumed or inserted into a matrix.

The fixed output gain makes every raw selected value amplitude at
least two. Its amplification of the fixed retrieval tolerance is
exactly one eighth. The actual complete final residual is therefore
within one eighth of its first-block state plus that selected raw
value code. This reference is used only in the theorem, never by
the model. Protected query/type/constant coordinates retain their
raw embeddings. Strict tied-logit comparisons, final RMS/decoder
and full validated-parser discharge remain separate obligations.
-/

namespace Transformer.GPTMini.Semantics

/-- The actual complete original second block on the actual complete first block, without a selected record or answer input.
Source: original blockForward twice, retaining genuine prenorm/heads/residuals and fixed ordinary parameters. -/
noncomputable def recallFinalState (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query : Fin T) : EucSpace 64 :=
  blockForward recallConfig (recallRetrievalBlock P eps) eps (fun r => (r.val : ℝ)) (recallMatchState P eps tokens) query

/-- The genuine selected-value amplitude produced by the fixed ordinary output gain and real next prenorm.
Source: true V scaling; the record position is an analysis argument, not an input to a learned matrix. -/
noncomputable def recallValueAmplitude (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (record : Fin T) : ℝ :=
  recallReadoutGain P eps * recallResidualScale eps (recallMatchState P eps tokens record)

/-- Compact reading commutes with every real value amplitude, including noncategorical imperfect head outputs.
Source: the evaluated ordinary first-eight-coordinate head extraction. -/
theorem recallHeadRead_smul (a : ℝ) (x : EucSpace recallConfig.head_dim) :
    recallHeadRead (a • x) = a • recallHeadRead x := by
  ext c
  rfl

/-- The actual selected V reads back as its true raw value symbol times its genuine prenorm multiplier.
Source: ordinary fused V, protected raw value slot, real head insertion and exact linear compact reading. -/
theorem recallMatchValue_read (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (record : Fin T) (value : Fin 256)
    (hvalue : tokens record = recallValueId value) :
    recallHeadRead (recallMatchValue P eps tokens record) =
      recallResidualScale eps (recallMatchState P eps tokens record) • recallSymbolCode value := by
  rw [recallMatchValue_value_code P eps tokens record value hvalue, recallHeadRead_smul, recallHeadRead_value]

example : recallPairWitness 0 0 2 = recallValueId 0 := by simp [recallPairWitness]

/-- The given fixed real output matrix amplifies its fixed retrieval tolerance to exactly one eighth.
Source: gain*secondScaleLower=2 and tolerance=secondScaleLower/16, with the positive uniform denominator derived. -/
theorem recallReadoutGain_tolerance (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) :
    recallReadoutGain P eps * recallRetrievalTolerance P eps = 1 / 8 := by
  have hm := recallReadoutGain_product P eps heps
  unfold recallRetrievalTolerance
  calc _ = (recallReadoutGain P eps * recallSecondScaleLower P eps) / 16 := by ring
    _ = 1 / 8 := by rw [hm]; norm_num

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

/-- Every true raw selected-value amplitude is at least two, despite its real record-dependent RMS denominator.
Source: uniform actual raw next-prenorm lower bound and the fixed positive ordinary output-gain product. -/
theorem recallValueAmplitude_lower (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (first record : Fin T) (hlayout : RecallRawLayout P tokens first) :
    2 ≤ recallValueAmplitude P eps tokens record := by
  have hl := recall_raw_retrieval_tolerance_bound P eps heps tokens first record hlayout
  rw [recallRetrievalTolerance_product] at hl
  unfold recallValueAmplitude
  calc 2 = recallReadoutGain P eps * recallSecondScaleLower P eps := (recallReadoutGain_product P eps heps).symm
    _ ≤ _ := mul_le_mul_of_nonneg_left hl (recallReadoutGain_pos P eps heps).le

example : (0 : ℝ) ≤ 1 / 100000 ∧ RecallRawLayout 2 recallScoreGapWitness 0 :=
  ⟨by norm_num, recallScoreGapWitness_layout⟩

/-- Every protected constant/query/type coordinate of the genuine two-block final state equals its raw token embedding.
Source: both real first-block residual supports and the second ordinary value-only write, with no encoded-coordinate premise. -/
theorem recallFinalState_protected (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query : Fin T) (c : Fin 64)
    (hc : c.val < 9 ∨ c.val = 17 ∨ 35 ≤ c.val) : recallFinalState P eps tokens query c = recallRawEmbedding (tokens query) c := by
  rw [recallFinalState, recallRetrievalBlock_protected P eps tokens query c (by omega)]
  exact recallEncoderBlock_raw_protected P eps (recallCopyTemperature recallKeyTolerance) (recallEncoderGain P)
    (fun r => (r.val : ℝ)) tokens query c (by omega)

example : (0 : Fin 64).val < 9 ∨ (0 : Fin 64).val = 17 ∨ 35 ≤ (0 : Fin 64).val := by decide

/-- The actual two-block query retains exactly its raw compact query code for the shared tied key embeddings.
Source: both genuine residual supports, followed by the actual raw key-table slot readback. -/
theorem recallFinalState_query_code (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query : Fin T) (key : Fin 256)
    (hquery : tokens query = recallKeyId key) :
    recallSlotRead 1 (recallFinalState P eps tokens query) = recallSymbolCode key := by
  have hr := (recallKeyEmbedding_reads key).1
  ext c
  rw [recallSlotRead_at, recallFinalState_protected P eps tokens query _ (by
    left; change 1 + c.val < 9; have hc := c.isLt; omega), hquery, recallRawEmbedding_key]
  simpa only [recallSlotRead_at] using congrArg (fun x : EucSpace 8 => x c) hr

example : recallScoreGapWitness 5 = recallKeyId 0 := by simp [recallScoreGapWitness]

/-- The real final query preserves the constant and key type while its reserved/value type readout axes stay zero.
Source: original two-block protected coordinates and the explicit token-local key embedding, including reserved axis seventeen. -/
theorem recallFinalState_query_flags (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query : Fin T) (key : Fin 256)
    (hquery : tokens query = recallKeyId key) :
    recallFinalState P eps tokens query 0 = 1 ∧ recallFinalState P eps tokens query 17 = 0 ∧
      recallFinalState P eps tokens query 35 = 1 ∧ recallFinalState P eps tokens query 36 = 0 ∧
      recallFinalState P eps tokens query 37 = 0 := by
  have h17 : recallKeyEmbedding key 17 = 0 := by
    rw [recallKeyEmbedding, recall_embedding_protected_coordinate 1 35 17 _ (by decide)]
    norm_num
  rw [recallFinalState_protected P eps tokens query 0 (by decide),
    recallFinalState_protected P eps tokens query 17 (by decide),
    recallFinalState_protected P eps tokens query 35 (by decide),
    recallFinalState_protected P eps tokens query 36 (by decide),
    recallFinalState_protected P eps tokens query 37 (by decide), hquery, recallRawEmbedding_key, h17]
  exact ⟨(recallKeyEmbedding_flags key).1, rfl, (recallKeyEmbedding_flags key).2⟩

example : recallScoreGapWitness 5 = recallKeyId 0 := by simp [recallScoreGapWitness]

/-- A raw query's real final value interval contains precisely the actual compact retrieval times the shared output gain.
Source: full genuine second-block slot formula and the actual first encoder's zero raw query-value channel. -/
theorem recallFinalState_query_value (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query : Fin T) (key : Fin 256)
    (hquery : tokens query = recallKeyId key) :
    recallSlotRead 9 (recallFinalState P eps tokens query) =
      recallReadoutGain P eps • recallHeadRead (recallRetrievalHead P eps tokens query) := by
  rw [recallFinalState, recallRetrievalBlock_value_read]
  have hz := recall_raw_encoder_query_value_zero P eps (recallCopyTemperature recallKeyTolerance) (recallEncoderGain P)
    (fun r => (r.val : ℝ)) tokens query key hquery
  change recallSlotRead 9 (recallMatchState P eps tokens query) = 0 at hz
  rw [hz, zero_add]

example : recallScoreGapWitness 5 = recallKeyId 0 := by simp [recallScoreGapWitness]

/-- The actual complete two-block final residual is uniformly within one eighth of its genuinely scaled selected raw value.
Source: real headAt accuracy, actual W_o/slot isometry and residual cancellation; no small final-state-error premise is assumed. -/
theorem recallFinalState_error (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ}
    (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size) (first query selected previous : Fin T)
    (hlayout : RecallRawLayout P tokens first) (key value : Fin 256) (hquery : tokens query = recallKeyId key)
    (hprev : previous.val + 1 = selected.val) (hkey : tokens previous = recallKeyId key)
    (hvalue : tokens selected = recallValueId value) (hinside : selected.val ≤ 2 * P)
    (hvisible : selected.val ≤ query.val) (hlatest : RecallRawLatestWrite P tokens selected key) :
    ‖recallFinalState P eps tokens query - (recallMatchState P eps tokens query +
      recallValueAmplitude P eps tokens selected • recallSlotWrite 9 (recallSymbolCode value))‖ ≤ (1 / 8 : ℝ) := by
  have hh := recallRetrievalHead_accuracy P eps heps hclip hT tokens first query selected previous
    hlayout key value hquery hprev hkey hvalue hinside hvisible hlatest
  have hr := recallHeadRead_error (recallRetrievalHead P eps tokens query) (recallMatchValue P eps tokens selected)
  have he : recallValueAmplitude P eps tokens selected • recallSlotWrite 9 (recallSymbolCode value) =
      recallReadoutGain P eps • recallSlotWrite 9 (recallHeadRead (recallMatchValue P eps tokens selected)) := by
    rw [recallMatchValue_read P eps tokens selected value hvalue, map_smul, smul_smul]
    rfl
  rw [recallFinalState, recallRetrievalBlock_formula, he, add_sub_add_left_eq_sub,
    ← smul_sub, norm_smul_of_nonneg (recallReadoutGain_pos P eps heps).le, ← map_sub, recallSlotWrite_norm]
  calc _ ≤ recallReadoutGain P eps * recallRetrievalTolerance P eps :=
      mul_le_mul_of_nonneg_left (hr.trans hh) (recallReadoutGain_pos P eps heps).le
    _ = 1 / 8 := recallReadoutGain_tolerance P eps heps

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (6 : ℕ) ≤ 64 ∧
    RecallRawLayout 2 recallScoreGapWitness 0 ∧ recallScoreGapWitness 5 = recallKeyId 0 ∧
    (3 : Fin 6).val + 1 = (4 : Fin 6).val ∧ recallScoreGapWitness 3 = recallKeyId 0 ∧
    recallScoreGapWitness 4 = recallValueId 1 ∧ (4 : Fin 6).val ≤ 2 * 2 ∧
    (4 : Fin 6).val ≤ (5 : Fin 6).val ∧ RecallRawLatestWrite 2 recallScoreGapWitness 4 0 := by
  exact ⟨by norm_num, by norm_num, by decide, recallScoreGapWitness_layout, by simp [recallScoreGapWitness],
    by decide, by simp [recallScoreGapWitness], by simp [recallScoreGapWitness], by decide, by decide,
    recallScoreGapWitness_latest⟩

end Transformer.GPTMini.Semantics
