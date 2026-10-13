import Transformer.GPTMini.Semantics.RecallNormalizedInputs

/-!
# Scores of the genuine raw matching head remain close to rotary references

Source: original CausalMHA.forward at f11b6e2, with raw MQAR writes
at cbafbe9. The accessors below evaluate the real complete encoder,
actual second prenorm and shared fused matrix at the fixed first-copy
temperature. They are ordinary forward-pass components, not an oracle
or a lookup of the desired record.

Exact raw query normalization and derived table-key error give a
score error at most 2*exp(alpha)*copyTolerance. This estimate compares
the actual epsilon with a unit-clipped categorical reference without
introducing a factor 1/epsilon. A true raw query also has zero own
value and zero matching key. Latest-write comparison and full raw
parser coupling remain separate obligations.
-/

namespace Transformer.GPTMini.Semantics

/-- Evaluate the given genuine first block at the fixed shared copy temperature and raw integer positions.
Source: recallEncodedState, with no semantic answer or selected-record argument. -/
noncomputable def recallMatchState (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) : EucSpace 64 :=
  recallEncodedState P eps (recallCopyTemperature recallKeyTolerance) (fun r => (r.val : ℝ)) tokens i

/-- The complete true second fused projection, retaining the genuine per-position prenorm multiplier.
Source: ordinary 64-to-192 matrix and original RMSNorm, at the fixed shared Q/K gain. -/
noncomputable def recallMatchQKV (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) : EucSpace 192 :=
  recallSecondQKV (recallMatchGain P eps) (rmsNormEps eps (recallMatchState P eps tokens i))

/-- The actual unrotated query of original head zero.
Source: original Q chunk and sixteen-coordinate head view of the simultaneous fused projection. -/
noncomputable def recallMatchQuery (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) : EucSpace 16 :=
  headSlice recallConfig (qkvSlice recallConfig (qkvQ recallConfig) (recallMatchQKV P eps tokens i)) 0

/-- The actual unrotated gated key of original head zero.
Source: original K chunk and head view, including imperfect real predecessor copies. -/
noncomputable def recallMatchKey (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) : EucSpace 16 :=
  headSlice recallConfig (qkvSlice recallConfig (qkvK recallConfig) (recallMatchQKV P eps tokens i)) 0

/-- The independent unrotated actual value of original head zero.
Source: original V chunk, without the query/key matching gain or a semantic-value substitution. -/
noncomputable def recallMatchValue (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) : EucSpace recallConfig.head_dim :=
  headSlice recallConfig (qkvSlice recallConfig (qkvV recallConfig) (recallMatchQKV P eps tokens i)) 0

/-- The actual matching score after original rotary positions and clipped normalization.
Source: original score/preScore, before the causal mask and finite softmax. -/
noncomputable def recallMatchScore (P : ℕ) (alpha eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query record : Fin T) : ℝ :=
  score alpha eps (applyRope 16 10000 (query.val : ℝ) (recallMatchQuery P eps tokens query))
    (applyRope 16 10000 (record.val : ℝ) (recallMatchKey P eps tokens record))

/-- An exact normalized query and approximate normalized key have a temperature-scaled score error, across clipping choices.
Source: the genuine bilinear score and Cauchy-Schwarz; unit-reference query norm is derived from normL2. -/
theorem recallScore_normalized_error {d : ℕ} (alpha eps eta : ℝ) (q k referenceQ referenceK : EucSpace d)
    (hq : normL2 eps q = normL2 1 referenceQ)
    (hk : ‖normL2 eps k - normL2 1 referenceK‖ ≤ eta) :
    |score alpha eps q k - score alpha 1 referenceQ referenceK| ≤ Real.exp alpha * eta := by
  unfold score
  rw [hq, ← mul_sub, ← inner_sub_right, abs_mul, abs_of_pos (Real.exp_pos alpha)]
  have hi := abs_real_inner_le_norm (normL2 1 referenceQ) (normL2 eps k - normL2 1 referenceK)
  have hn := normL2_norm_le 1 (by norm_num) referenceQ
  have hb : |inner (𝕜 := ℝ) (normL2 1 referenceQ) (normL2 eps k - normL2 1 referenceK)| ≤ eta := by
    calc _ ≤ ‖normL2 1 referenceQ‖ * ‖normL2 eps k - normL2 1 referenceK‖ := hi
      _ ≤ 1 * ‖normL2 eps k - normL2 1 referenceK‖ :=
        mul_le_mul_of_nonneg_right hn (norm_nonneg _)
      _ ≤ eta := by rw [one_mul]; exact hk
  exact mul_le_mul_of_nonneg_left hb (Real.exp_pos alpha).le

example : normL2 1 (recallCode (fun _ => 0)) = normL2 1 (recallCode (fun _ => 0)) ∧
    ‖normL2 1 (recallCode (fun _ => 0)) - normL2 1 (recallCode (fun _ => 0))‖ ≤ (0 : ℝ) := by
  exact ⟨rfl, by rw [sub_self, norm_zero]⟩

/-- A real query/table-write score has uniformly small error from its true raw neighboring-key rotary reference.
Source: derived actual query saturation and real normalized copied-key error, with all clipping conditions discharged. -/
theorem recall_raw_score_error (P : ℕ) (alpha eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ}
    (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size) (first query record previous : Fin T)
    (hfirst : first.val = 0) (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol)
    (queryKey recordKey value : Fin 256) (hquery : tokens query = recallKeyId queryKey)
    (hprev : previous.val + 1 = record.val) (hkey : tokens previous = recallKeyId recordKey)
    (hvalue : tokens record = recallValueId value) (hinside : record.val ≤ 2 * P) :
    |recallMatchScore P alpha eps tokens query record -
      score alpha 1 (applyRope 16 10000 (query.val : ℝ) (recallRotaryCode (recallDigit queryKey)))
        (applyRope 16 10000 (record.val : ℝ) (recallRotaryCode (recallDigit recordKey)))| ≤
          2 * Real.exp alpha * recallKeyTolerance := by
  unfold recallMatchScore
  have he := recallScore_normalized_error alpha eps (2 * recallKeyTolerance)
    (applyRope 16 10000 (query.val : ℝ) (recallMatchQuery P eps tokens query))
    (applyRope 16 10000 (record.val : ℝ) (recallMatchKey P eps tokens record))
    (applyRope 16 10000 (query.val : ℝ) (recallRotaryCode (recallDigit queryKey)))
    (applyRope 16 10000 (record.val : ℝ) (recallRotaryCode (recallDigit recordKey)))
    (recall_raw_query_normalized P eps (recallCopyTemperature recallKeyTolerance) heps _
      tokens first query hfirst hbos hrange queryKey hquery)
    (recall_raw_key_normalized_error P eps heps hclip hT tokens first record previous hfirst hbos hrange
      hprev recordKey value hkey hvalue hinside)
  calc _ ≤ Real.exp alpha * (2 * recallKeyTolerance) := he
    _ = _ := by ring

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

/-- Any genuinely excluded zero matching key has exactly zero actual rotary score.
Source: original linear rotary map, clipped normalization and bilinear score, including epsilon zero. -/
theorem recallMatchScore_key_zero (P : ℕ) (alpha eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query record : Fin T)
    (hkey : recallMatchKey P eps tokens record = 0) : recallMatchScore P alpha eps tokens query record = 0 := by
  rw [recallMatchScore, hkey, rope_zero]
  simp only [score, normL2, smul_zero, inner_zero_right, mul_zero]

example : recallMatchKey 0 0 (fun _ : Fin 1 => recallKeyId 0) 0 = 0 := by
  exact recall_raw_second_query_key_zero 0 0 _ (by norm_num) _ _ 0 0 rfl

/-- Every raw query/key competitor has zero real score rather than a spurious copied binding.
Source: the complete genuine raw gate exclusion and actual prenorm/fused K projection. -/
theorem recallMatchScore_raw_key_zero (P : ℕ) (alpha eps : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query record : Fin T) (key : Fin 256)
    (hrecord : tokens record = recallKeyId key) : recallMatchScore P alpha eps tokens query record = 0 := by
  apply recallMatchScore_key_zero
  exact recall_raw_second_query_key_zero P eps _ heps _ tokens record key hrecord

example : (0 : ℝ) ≤ 1 / 100000 ∧ recallPairWitness 0 0 1 = recallKeyId 0 := by
  exact ⟨by norm_num, by simp [recallPairWitness]⟩

/-- Every actual raw query has zero own matching value at the fixed complete encoder, preserving retrieval through XSA.
Source: the genuine raw second-value-zero result, with unchanged matrix and original head dimensions. -/
theorem recallMatchValue_query_zero (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query : Fin T) (key : Fin 256)
    (hquery : tokens query = recallKeyId key) : recallMatchValue P eps tokens query = 0 := by
  exact recall_raw_second_self_value_zero P eps _ _ tokens query key hquery

example : recallPairWitness 0 0 1 = recallKeyId 0 := by simp [recallPairWitness]

end Transformer.GPTMini.Semantics
