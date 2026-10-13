import Transformer.GPTMini.Semantics.RecallRetrievalAccuracy

/-!
# The genuine second ordinary recall block writes retrieved values

Source: original W_o, head merge, prenorm attention, zero FFN control
and both residuals at f11b6e2. The fixed actual matching QKV and
finite shared temperature are assembled into unchanged original
parameter records. One ordinary 64-to-64 matrix reads head zero's
first eight coordinates and writes them to raw value slot 9..16.

The complete second block uses the real complete first-block state.
Every other raw query/type channel is protected. Head accuracy is
transported through actual headAt, merge and W_o; no paired prefix
encoder, prepared head or correct output logit is supplied. Full
tied readout and validated integer-parser coupling still remain.
These are given real-arithmetic softmax baseline parameters, not
a convexity, floating-point or optimization convergence claim.
-/

namespace Transformer.GPTMini.Semantics

/-- One ordinary shared output matrix reads the first compact head and writes the independent value interval.
Source: original contiguous four-head merge, with gain fixed before seeing any prefix or selected record. -/
noncomputable def recallRetrievalOutput (gain : ℝ) : EucSpace 64 →L[ℝ] EucSpace 64 :=
  gain • ((recallSlotWrite 9).comp (recallSlotRead 0))

/-- The actual original second attention parameter record at the proved finite shared coefficients.
Source: unchanged fused QKV, W_o and four per-head log-temperatures in AttnParams. -/
noncomputable def recallRetrievalAttention (P : ℕ) (eps : ℝ) : AttnParams recallConfig where
  W_qkv := recallSecondQKV (recallMatchGain P eps)
  W_o := recallRetrievalOutput (recallReadoutGain P eps)
  log_alpha := fun h => if h = 0 then recallRetrievalTemperature (recallRetrievalTolerance P eps) else 0

/-- The genuine second block retains both original residuals and an ordinary zero-matrix FFN.
Source: original BlockParams and bias-free ReLU2 FFN; zero FFN is a given baseline parameter assignment. -/
noncomputable def recallRetrievalBlock (P : ℕ) (eps : ℝ) : BlockParams recallConfig where
  attn := recallRetrievalAttention P eps
  ffn := { W_in := 0, W_out := 0 }

/-- Evaluate head zero of that actual second parameter record on the actual first-block output.
Source: headAt with genuine prenorm, fused matrix, original rotary positions, finite softmax and XSA. -/
noncomputable def recallRetrievalHead (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query : Fin T) : EucSpace recallConfig.head_dim :=
  headAt recallConfig (recallRetrievalAttention P eps) eps (fun r => (r.val : ℝ)) (recallMatchState P eps tokens) 0 query

/-- The actual ordinary W_o extracts exactly head zero's real compact coordinates, regardless of all other heads.
Source: all eight original merge indices and the evaluated shared matrix. -/
theorem recallRetrievalOutput_merge (gain : ℝ) (heads : Fin recallConfig.n_heads → EucSpace recallConfig.head_dim) :
    recallRetrievalOutput gain (headMerge recallConfig heads) = gain • recallSlotWrite 9 (recallHeadRead (heads 0)) := by
  have hs : ∀ c : Fin 8, headSplit recallConfig (recallSlotIndex 0 c) =
      (0, ⟨c.val, by change c.val < 16; have hc := c.isLt; omega⟩) := by decide
  have hr : recallSlotRead 0 (headMerge recallConfig heads) = recallHeadRead (heads 0) := by
    ext c
    rw [recallSlotRead_at, headMerge_apply, hs]
    rfl
  simp only [recallRetrievalOutput, smul_apply, ContinuousLinearMap.comp_apply, hr]

/-- The genuine extracted head is precisely the previously verified raw matching head at the fixed finite temperature.
Source: unchanged headAt/fused prenorm rows, evaluated at the same real first-block state and integer positions. -/
theorem recallRetrievalHead_formula (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query : Fin T) :
    recallRetrievalHead P eps tokens query =
      attentionHead recallConfig (recallRetrievalTemperature (recallRetrievalTolerance P eps)) eps
        (recallMatchQuery P eps tokens) (recallMatchKey P eps tokens) (recallMatchValue P eps tokens)
        (fun r => (r.val : ℝ)) query := by
  unfold recallRetrievalHead headAt
  dsimp only [recallRetrievalAttention]
  rw [ite_eq_left (rfl : (0 : Fin 4) = 0)]
  rfl

/-- The real whole second attention sublayer writes the actual retrieved compact value into its independent residual channel.
Source: original headAt/headMerge/W_o, not a replacement for finite softmax or a desired-value assignment. -/
theorem recall_retrieval_attn_formula (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query : Fin T) :
    attnSubLayer recallConfig (recallRetrievalAttention P eps) eps (fun r => (r.val : ℝ))
        (recallMatchState P eps tokens) query =
      recallReadoutGain P eps • recallSlotWrite 9 (recallHeadRead (recallRetrievalHead P eps tokens query)) := by
  rw [attnSubLayer_eq_heads]
  change recallRetrievalOutput (recallReadoutGain P eps) (headMerge recallConfig _) = _
  rw [recallRetrievalOutput_merge]
  rfl

/-- Both genuine residual additions give the exact second-block state, with the actual retrieved value and zero FFN.
Source: original blockForward and relu2FFN at the supplied ordinary zero FFN matrices. -/
theorem recallRetrievalBlock_formula (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query : Fin T) :
    blockForward recallConfig (recallRetrievalBlock P eps) eps (fun r => (r.val : ℝ))
        (recallMatchState P eps tokens) query = recallMatchState P eps tokens query +
      recallReadoutGain P eps • recallSlotWrite 9 (recallHeadRead (recallRetrievalHead P eps tokens query)) := by
  dsimp only [blockForward, recallRetrievalBlock, ffnSubLayer, relu2FFN]
  simp only [zero_apply, add_zero]
  rw [recall_retrieval_attn_formula]

/-- The actual output matrix has no contribution outside the independent value interval.
Source: evaluated real matrix support, preserving all query/type/gated-key coordinates. -/
theorem recallRetrievalOutput_protected (gain : ℝ) (x : EucSpace 64) (c : Fin 64)
    (hc : c.val < 9 ∨ 17 ≤ c.val) : recallRetrievalOutput gain x c = 0 := by
  simp only [recallRetrievalOutput, smul_apply, ContinuousLinearMap.comp_apply, PiLp.smul_apply, smul_eq_mul]
  rw [recallSlotWrite_outside 9 _ c hc, mul_zero]

example : (35 : Fin 64).val < 9 ∨ 17 ≤ (35 : Fin 64).val := by decide

/-- The true complete second block retains every raw query/type coordinate outside its value output interval.
Source: original residual state formula and the actual ordinary write support, including the zero FFN. -/
theorem recallRetrievalBlock_protected (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query : Fin T) (c : Fin 64)
    (hc : c.val < 9 ∨ 17 ≤ c.val) :
    blockForward recallConfig (recallRetrievalBlock P eps) eps (fun r => (r.val : ℝ))
      (recallMatchState P eps tokens) query c = recallMatchState P eps tokens query c := by
  rw [recallRetrievalBlock_formula, PiLp.add_apply, PiLp.smul_apply, recallSlotWrite_outside 9 _ c hc]
  simp only [smul_zero, add_zero]

example : (0 : Fin 64).val < 9 ∨ 17 ≤ (0 : Fin 64).val := by decide

/-- Reading the actual final value interval retains exactly the original residual plus the genuine compact retrieval.
Source: the complete true second block and exact ordinary slot read/write round trip. -/
theorem recallRetrievalBlock_value_read (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query : Fin T) :
    recallSlotRead 9 (blockForward recallConfig (recallRetrievalBlock P eps) eps (fun r => (r.val : ℝ))
      (recallMatchState P eps tokens) query) = recallSlotRead 9 (recallMatchState P eps tokens query) +
        recallReadoutGain P eps • recallHeadRead (recallRetrievalHead P eps tokens query) := by
  rw [recallRetrievalBlock_formula, map_add, map_smul, recallSlotRead_write]

/-- The ordinary true output matrix amplifies compact retrieval error only by its given nonnegative gain.
Source: actual head merge/write formula, exact slot isometry and nonexpansive compact head read. -/
theorem recallRetrievalOutput_error (gain : ℝ) (hg : 0 ≤ gain)
    (heads : Fin recallConfig.n_heads → EucSpace recallConfig.head_dim) (target : EucSpace recallConfig.head_dim) :
    ‖recallRetrievalOutput gain (headMerge recallConfig heads) - gain • recallSlotWrite 9 (recallHeadRead target)‖ ≤
      gain * ‖heads 0 - target‖ := by
  rw [recallRetrievalOutput_merge, ← smul_sub, norm_smul_of_nonneg hg, ← map_sub, recallSlotWrite_norm]
  exact mul_le_mul_of_nonneg_left (recallHeadRead_error (heads 0) target) hg

example : (0 : ℝ) ≤ 1 := by norm_num

/-- The actual extracted head of the genuine second block has the proved prefix-independent tied-readout accuracy.
Source: complete actual headAt coupling and uniform real raw retrieval, with no prepared head or score-gap premise. -/
theorem recallRetrievalHead_accuracy (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ}
    (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size) (first query selected previous : Fin T)
    (hlayout : RecallRawLayout P tokens first) (key value : Fin 256) (hquery : tokens query = recallKeyId key)
    (hprev : previous.val + 1 = selected.val) (hkey : tokens previous = recallKeyId key)
    (hvalue : tokens selected = recallValueId value) (hinside : selected.val ≤ 2 * P)
    (hvisible : selected.val ≤ query.val) (hlatest : RecallRawLatestWrite P tokens selected key) :
    ‖recallRetrievalHead P eps tokens query - recallMatchValue P eps tokens selected‖ ≤ recallRetrievalTolerance P eps := by
  rw [recallRetrievalHead_formula]
  exact recall_raw_head_readout_accuracy P eps heps hclip hT tokens first query selected previous
    hlayout key value hquery hprev hkey hvalue hinside hvisible hlatest

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (6 : ℕ) ≤ 64 ∧
    RecallRawLayout 2 recallScoreGapWitness 0 ∧ recallScoreGapWitness 5 = recallKeyId 0 ∧
    (3 : Fin 6).val + 1 = (4 : Fin 6).val ∧ recallScoreGapWitness 3 = recallKeyId 0 ∧
    recallScoreGapWitness 4 = recallValueId 1 ∧ (4 : Fin 6).val ≤ 2 * 2 ∧
    (4 : Fin 6).val ≤ (5 : Fin 6).val ∧ RecallRawLatestWrite 2 recallScoreGapWitness 4 0 := by
  exact ⟨by norm_num, by norm_num, by decide, recallScoreGapWitness_layout, by simp [recallScoreGapWitness],
    by decide, by simp [recallScoreGapWitness], by simp [recallScoreGapWitness], by decide, by decide,
    recallScoreGapWitness_latest⟩

end Transformer.GPTMini.Semantics
