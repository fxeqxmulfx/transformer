import Transformer.GPTMini.Semantics.RecallFinalState

/-!
# Actual tied recall scores and compact categorical margins

Source: original tied unembedding after the two residual blocks at
f11b6e2, and the real disjoint raw embedding/code layout for the
548-token MQAR vocabulary at cbafbe9. Every score identity below
evaluates ordinary inner products with those same input embeddings.
The eight-coordinate value code is retained without a new classifier.

The genuine final query has key-token scores at most six and reserved
scores exactly one, independently of retrieval leakage. A reference
value write is used only to analyze the already derived final-state
error. Its value scores are one plus the actual selected amplitude
times the compact code inner product. Identical symbols have product
four; distinct symbols have product at most three. Full raw routing
and error bounds must supply that reference; it is not a model input.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators

/-- The real slot insertion and extraction are adjoint in the original residual coordinates.
Source: the evaluated rank-one matrices and actual Euclidean inner products. -/
theorem recallSlotWrite_inner_read (offset : Fin 57) (x : EucSpace 64) (code : EucSpace 8) :
    inner (𝕜 := ℝ) x (recallSlotWrite offset code) = inner (𝕜 := ℝ) (recallSlotRead offset x) code := by
  rw [recallSlotWrite_apply, inner_sum, PiLp.inner_apply]
  simp only [real_inner_smul_right, EuclideanSpace.inner_single_right, conj_trivial,
    one_mul, RCLike.inner_apply, recallSlotRead_at]

/-- Every tied raw key score reads its actual constant, compact key interval and key-type axis.
Source: the same original input embedding table, with no separate classifier weights. -/
theorem recallKeyEmbedding_inner (x : EucSpace 64) (symbol : Fin 256) :
    inner (𝕜 := ℝ) x (recallKeyEmbedding symbol) =
      x 0 + inner (𝕜 := ℝ) (recallSlotRead 1 x) (recallSymbolCode symbol) + x 35 := by
  rw [recallKeyEmbedding, inner_add_right, inner_add_right, recallSlotWrite_inner_read]
  simp only [recallUnit, EuclideanSpace.inner_single_right, conj_trivial, one_mul]

/-- Every tied raw value score reads its own compact interval rather than the query-key interval.
Source: disjoint ordinary value embedding and true tied unembedding. -/
theorem recallValueEmbedding_inner (x : EucSpace 64) (symbol : Fin 256) :
    inner (𝕜 := ℝ) x (recallValueEmbedding symbol) =
      x 0 + inner (𝕜 := ℝ) (recallSlotRead 9 x) (recallSymbolCode symbol) + x 36 := by
  rw [recallValueEmbedding, inner_add_right, inner_add_right, recallSlotWrite_inner_read]
  simp only [recallUnit, EuclideanSpace.inner_single_right, conj_trivial, one_mul]

/-- Every tied reserved entry reads only the three explicit fallback coordinates.
Source: the complete original embedding-table branch on IDs below thirty-six. -/
theorem recallReservedEmbedding_inner (x : EucSpace 64) :
    inner (𝕜 := ℝ) x recallReservedEmbedding = x 0 + 2 * x 17 + x 37 := by
  rw [recallReservedEmbedding, inner_add_right, inner_add_right, real_inner_smul_right]
  simp only [recallUnit, EuclideanSpace.inner_single_right, conj_trivial, one_mul]

/-- Every pair of actual compact symbol codes has inner product at most four.
Source: their proved norm two and Cauchy-Schwarz, including equal symbols. -/
theorem recallSymbolCode_inner_le (a b : Fin 256) :
    inner (𝕜 := ℝ) (recallSymbolCode a) (recallSymbolCode b) ≤ 4 := by
  have h := (le_abs_self _).trans (abs_real_inner_le_norm (recallSymbolCode a) (recallSymbolCode b))
  simpa only [recallSymbolCode, recallCode_norm, show (2 : ℝ) * 2 = 4 from by norm_num] using h

/-- An identical raw value symbol contributes precisely four to its own tied code score.
Source: the evaluated norm-square of the four unit signed-axis pairs. -/
theorem recallSymbolCode_inner_self (a : Fin 256) :
    inner (𝕜 := ℝ) (recallSymbolCode a) (recallSymbolCode a) = 4 := by
  rw [real_inner_self_eq_norm_sq, recallSymbolCode, recallCode_norm_sq]

/-- Every distinct raw value code loses at least one unit of categorical inner product.
Source: the actual injective radix-four digits and the proved differing-axis margin. -/
theorem recallSymbolCode_inner_gap (a b : Fin 256) (hne : a ≠ b) :
    inner (𝕜 := ℝ) (recallSymbolCode a) (recallSymbolCode b) ≤ 3 := by
  exact recallCode_inner_gap _ _ (fun h => hne (recallDigit_injective h))

example : (0 : Fin 256) ≠ 1 := by decide

/-- The actual complete final query's tied key scores are bounded by six for all two hundred fifty-six keys.
Source: derived true query code and protected flags, not an assumed key-logit bound. -/
theorem recallFinalState_key_score (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query : Fin T) (key other : Fin 256)
    (hquery : tokens query = recallKeyId key) :
    inner (𝕜 := ℝ) (recallFinalState P eps tokens query) (recallRawEmbedding (recallKeyId other)) ≤ 6 := by
  have hf := recallFinalState_query_flags P eps tokens query key hquery
  rw [recallRawEmbedding_key, recallKeyEmbedding_inner, hf.1, hf.2.2.1,
    recallFinalState_query_code P eps tokens query key hquery]
  linarith [recallSymbolCode_inner_le key other]

example : recallScoreGapWitness 5 = recallKeyId 0 := by simp [recallScoreGapWitness]

/-- Every genuine final query gives reserved fallback tokens the same tied score one.
Source: derived true constant/reserved/type coordinates in both original blocks. -/
theorem recallFinalState_reserved_score (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query : Fin T) (key : Fin 256)
    (hquery : tokens query = recallKeyId key) :
    inner (𝕜 := ℝ) (recallFinalState P eps tokens query) recallReservedEmbedding = 1 := by
  have hf := recallFinalState_query_flags P eps tokens query key hquery
  rw [recallReservedEmbedding_inner, hf.1, hf.2.1, hf.2.2.2.2]
  norm_num

example : recallScoreGapWitness 5 = recallKeyId 0 := by simp [recallScoreGapWitness]

/-- The actual first block's query has no initial tied value preference for any of the raw values.
Source: protected constant/value type and derived zero raw query-value slot after the real encoder. -/
theorem recallMatchState_query_value_score (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query : Fin T) (key value : Fin 256)
    (hquery : tokens query = recallKeyId key) :
    inner (𝕜 := ℝ) (recallMatchState P eps tokens query) (recallValueEmbedding value) = 1 := by
  have hz := recall_raw_encoder_query_value_zero P eps (recallCopyTemperature recallKeyTolerance) (recallEncoderGain P)
    (fun r => (r.val : ℝ)) tokens query key hquery
  have hc (c : Fin 64) (h : c.val < 18 ∨ 35 ≤ c.val) :
      recallMatchState P eps tokens query c = recallRawEmbedding (tokens query) c :=
    recallEncoderBlock_raw_protected P eps (recallCopyTemperature recallKeyTolerance) (recallEncoderGain P)
      (fun r => (r.val : ℝ)) tokens query c h
  change recallSlotRead 9 (recallMatchState P eps tokens query) = 0 at hz
  rw [recallValueEmbedding_inner, hz, inner_zero_left, add_zero]
  change recallMatchState P eps tokens query 0 + recallMatchState P eps tokens query 36 = 1
  rw [hc 0 (by decide), hc 36 (by decide), hquery, recallRawEmbedding_key,
    (recallKeyEmbedding_flags key).1, (recallKeyEmbedding_flags key).2.2.1, add_zero]

example : recallScoreGapWitness 5 = recallKeyId 0 := by simp [recallScoreGapWitness]

/-- A real value-only residual write contributes exactly the compact categorical inner product to every tied value entry.
Source: ordinary slot support and adjoint readback of the same raw input value embedding. -/
theorem recallValueEmbedding_write_inner (code : EucSpace 8) (symbol : Fin 256) :
    inner (𝕜 := ℝ) (recallSlotWrite 9 code) (recallValueEmbedding symbol) =
      inner (𝕜 := ℝ) code (recallSymbolCode symbol) := by
  rw [recallValueEmbedding_inner, recallSlotRead_write,
    recallSlotWrite_outside 9 code 0 (by decide), recallSlotWrite_outside 9 code 36 (by decide)]
  simp only [zero_add, add_zero]

/-- The derived selected-value reference has exact tied scores with its genuine amplitude and every categorical value.
Source: actual first-block query residual and value-only write; this reference appears only in the final-error analysis. -/
theorem recallReference_value_score (P : ℕ) (eps a : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query : Fin T) (key answer other : Fin 256)
    (hquery : tokens query = recallKeyId key) :
    inner (𝕜 := ℝ) (recallMatchState P eps tokens query + a • recallSlotWrite 9 (recallSymbolCode answer))
      (recallRawEmbedding (recallValueId other)) =
      1 + a * inner (𝕜 := ℝ) (recallSymbolCode answer) (recallSymbolCode other) := by
  rw [recallRawEmbedding_value, inner_add_left, real_inner_smul_left,
    recallMatchState_query_value_score P eps tokens query key other hquery, recallValueEmbedding_write_inner]

example : recallScoreGapWitness 5 = recallKeyId 0 := by simp [recallScoreGapWitness]

/-- Every actual vocabulary entry amplifies an arbitrary residual error by at most three in tied unembedding.
Source: the whole raw embedding table's derived norm bound and real Cauchy-Schwarz; all 548 entries are covered. -/
theorem recallRawEmbedding_score_error (x y : EucSpace 64) (token : Fin recallConfig.vocab_size) :
    |inner (𝕜 := ℝ) x (recallRawEmbedding token) - inner (𝕜 := ℝ) y (recallRawEmbedding token)| ≤ 3 * ‖x - y‖ := by
  rw [← inner_sub_left]
  calc _ ≤ ‖x - y‖ * ‖recallRawEmbedding token‖ := abs_real_inner_le_norm _ _
    _ ≤ ‖x - y‖ * 3 := mul_le_mul_of_nonneg_left (recallRawEmbedding_norm_le token) (norm_nonneg _)
    _ = 3 * ‖x - y‖ := by ring

end Transformer.GPTMini.Semantics
