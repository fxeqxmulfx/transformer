import Transformer.GPTMini.Semantics.RecallRawExcluded

/-!
# A full genuine recall score row from raw table and last-write conditions

Source: MQAR.writes/targets at cbafbe9 and original causal softmax
at f11b6e2. The predicates describe only raw BOS/alphabet, adjacency
inside the fixed table, and chronological last occurrence. Neither
predicate mentions a hidden state, normalized key, score, logits or
the desired model output. They must still be derived from successful
full Basis parsing; they do not define a new model encoder.

Every actual competing position is proved below the true selected
record: table values use robust copied-key geometry, whereas BOS,
keys/queries and post-table fillers have zero matching score. This
gives the complete finite-softmax tail bound at the original mask,
without a score-gap premise. The six-token overwrite control makes
all raw conditions concrete. Full validated-parser coupling, finite
value retrieval and complete tied integer readout remain.
-/

namespace Transformer.GPTMini.Semantics

/-- Raw table applicability: BOS, the two original alphabets, and the key immediately preceding each table value.
Source: the actual adjacent MQAR table region; this is a predicate of the raw token array and boundary. -/
def RecallRawLayout (P : ℕ) {T : ℕ} (tokens : Fin T → Fin recallConfig.vocab_size) (first : Fin T) : Prop :=
  first.val = 0 ∧ tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size) ∧
  (∀ j, j ≠ first → ∃ symbol : Fin 256, tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol) ∧
  ∀ record value, tokens record = recallValueId value → record.val ≤ 2 * P →
    ∃ previous key, previous.val + 1 = record.val ∧ tokens previous = recallKeyId key

/-- The selected raw table position is no earlier than every table write whose neighboring key equals the query key.
Source: MQAR's chronological last-write semantics; no model score or encoded route is part of this predicate. -/
def RecallRawLatestWrite (P : ℕ) {T : ℕ} (tokens : Fin T → Fin recallConfig.vocab_size)
    (selected : Fin T) (key : Fin 256) : Prop :=
  ∀ record previous value, previous.val + 1 = record.val → tokens previous = recallKeyId key →
    tokens record = recallValueId value → record.val ≤ 2 * P → record.val ≤ selected.val

/-- All positions in the actual matching row are separated from the raw latest matching record by a positive gap.
Source: complete actual raw encoder/second QKV, all raw competitor cases, original geometry and derived copy-error margins. -/
theorem recall_raw_row_score_gap (P : ℕ) (alpha eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ}
    (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size) (first query selected previous : Fin T)
    (hlayout : RecallRawLayout P tokens first) (key value : Fin 256) (hquery : tokens query = recallKeyId key)
    (hprev : previous.val + 1 = selected.val) (hkey : tokens previous = recallKeyId key)
    (hvalue : tokens selected = recallValueId value) (hinside : selected.val ≤ 2 * P)
    (hvisible : selected.val ≤ query.val) (hlatest : RecallRawLatestWrite P tokens selected key) :
    ∀ competitor, competitor ≠ selected → recallMatchScore P alpha eps tokens query competitor ≤
      recallMatchScore P alpha eps tokens query selected - recallMatchGap alpha := by
  rcases hlayout with ⟨hfirst, hbos, hrange, htable⟩
  have hzero := recall_raw_selected_score_lower P alpha eps heps hclip hT tokens first query selected previous
    hfirst hbos hrange key value hquery hprev hkey hvalue hinside
  intro competitor hne
  by_cases hb : competitor = first
  · rw [hb, recallMatchScore_raw_bos_zero P alpha eps heps tokens query first hbos]
    exact hzero
  · obtain ⟨symbol, hk | hv⟩ := hrange competitor hb
    · rw [recallMatchScore_raw_key_zero P alpha eps heps tokens query competitor symbol hk]
      exact hzero
    · by_cases hin : competitor.val ≤ 2 * P
      · obtain ⟨cp, ck, hp, hc⟩ := htable competitor symbol hv hin
        apply recall_raw_table_score_gap P alpha eps heps hclip hT tokens first query selected previous competitor cp
          hfirst hbos hrange key ck value symbol hquery hprev hkey hvalue hinside hp hc hv hin hvisible
        intro he
        subst ck
        have hl := hlatest competitor cp symbol hp hc hv hin
        have hi : competitor.val ≠ selected.val := fun h => hne (Fin.ext h)
        omega
      · rw [recallMatchScore_raw_after_zero P alpha eps heps tokens first query competitor hfirst hbos hrange symbol hv
          (by omega)]
        exact hzero

/-- The actual original causal softmax concentrates on the raw latest matching record, with no score-gap premise.
Source: the derived complete real score row and tail_mass_le for the genuine causal denominator. -/
theorem recall_raw_tail_mass (P : ℕ) (alpha eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ}
    (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size) (first query selected previous : Fin T)
    (hlayout : RecallRawLayout P tokens first) (key value : Fin 256) (hquery : tokens query = recallKeyId key)
    (hprev : previous.val + 1 = selected.val) (hkey : tokens previous = recallKeyId key)
    (hvalue : tokens selected = recallValueId value) (hinside : selected.val ≤ 2 * P)
    (hvisible : selected.val ≤ query.val) (hlatest : RecallRawLatestWrite P tokens selected key) :
    1 - causalAttnWeights recallConfig alpha eps
        (fun r => applyRope 16 10000 (r.val : ℝ) (recallMatchQuery P eps tokens r))
        (fun r => applyRope 16 10000 (r.val : ℝ) (recallMatchKey P eps tokens r)) query selected ≤
      ((T - 1 : ℕ) : ℝ) * Real.exp (-recallMatchGap alpha) := by
  have hg := recall_raw_row_score_gap P alpha eps heps hclip hT tokens first query selected previous
    hlayout key value hquery hprev hkey hvalue hinside hvisible hlatest
  apply tail_mass_le recallConfig alpha eps _ _ _ query selected hvisible
  intro competitor hne hcausal
  exact hg competitor hne

/-- The actual finite-softmax/XSA head retrieves the latest raw value with a uniform derived error and no routing premise.
Source: the genuine full-row tail bound, actual V norm at most sixteen and exactly zero query self-value. -/
theorem recall_raw_head_retrieval (P : ℕ) (alpha eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ}
    (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size) (first query selected previous : Fin T)
    (hlayout : RecallRawLayout P tokens first) (key value : Fin 256) (hquery : tokens query = recallKeyId key)
    (hprev : previous.val + 1 = selected.val) (hkey : tokens previous = recallKeyId key)
    (hvalue : tokens selected = recallValueId value) (hinside : selected.val ≤ 2 * P)
    (hvisible : selected.val ≤ query.val) (hlatest : RecallRawLatestWrite P tokens selected key) :
    ‖attentionHead recallConfig alpha eps (recallMatchQuery P eps tokens) (recallMatchKey P eps tokens)
        (recallMatchValue P eps tokens) (fun r => (r.val : ℝ)) query -
      (recallMatchValue P eps tokens selected : EucSpace recallConfig.head_dim)‖ ≤
      ((T - 1 : ℕ) : ℝ) * Real.exp (-recallMatchGap alpha) * 32 := by
  have ht := recall_raw_tail_mass P alpha eps heps hclip hT tokens first query selected previous
    hlayout key value hquery hprev hkey hvalue hinside hvisible hlatest
  have hd : ∀ j, j ≠ selected → ‖recallMatchValue P eps tokens j - recallMatchValue P eps tokens selected‖ ≤ 32 := by
    intro j hj
    have hn := norm_sub_le (recallMatchValue P eps tokens j) (recallMatchValue P eps tokens selected)
    linarith [recallMatchValue_norm_le P eps heps tokens j, recallMatchValue_norm_le P eps heps tokens selected]
  have ho := output_error_le recallConfig alpha eps
    (fun r => applyRope 16 10000 (r.val : ℝ) (recallMatchQuery P eps tokens r))
    (fun r => applyRope 16 10000 (r.val : ℝ) (recallMatchKey P eps tokens r))
    (recallMatchValue P eps tokens) query selected 32 hd
  have hself : (recallMatchValue P eps tokens query : EucSpace recallConfig.head_dim) = 0 :=
    recallMatchValue_query_zero P eps tokens query key hquery
  have hn : normL2 eps (recallMatchValue P eps tokens query : EucSpace recallConfig.head_dim) = 0 := by
    rw [hself, normL2, smul_zero]
  have he : attentionHead recallConfig alpha eps (recallMatchQuery P eps tokens) (recallMatchKey P eps tokens)
      (recallMatchValue P eps tokens) (fun r => (r.val : ℝ)) query =
        attnOutput recallConfig alpha eps
          (fun r => applyRope 16 10000 (r.val : ℝ) (recallMatchQuery P eps tokens r))
          (fun r => applyRope 16 10000 (r.val : ℝ) (recallMatchKey P eps tokens r))
          (recallMatchValue P eps tokens) query := by
    unfold attentionHead xsaProjection
    simp only [hn, inner_zero_right, zero_smul, sub_zero, ite_true]
    rfl
  rw [he]
  exact ho.trans (mul_le_mul_of_nonneg_right ht (by norm_num))

/-- The concrete raw overwrite control has a genuine initial BOS, raw alphabet and adjacent table layout.
Source: the two complete chronological raw records in recallScoreGapWitness, used only as proof test data. -/
theorem recallScoreGapWitness_layout : RecallRawLayout 2 recallScoreGapWitness 0 := by
  refine ⟨by decide, by simp [recallScoreGapWitness], ?_, ?_⟩
  · intro j hj
    fin_cases j
    · contradiction
    · exact ⟨0, Or.inl (by simp [recallScoreGapWitness])⟩
    · exact ⟨0, Or.inr (by simp [recallScoreGapWitness])⟩
    · exact ⟨0, Or.inl (by simp [recallScoreGapWitness])⟩
    · exact ⟨1, Or.inr (by simp [recallScoreGapWitness])⟩
    · exact ⟨0, Or.inl (by simp [recallScoreGapWitness])⟩
  · intro record value hv hin
    have he := congrArg Fin.val hv
    fin_cases record
    · norm_num [recallScoreGapWitness, recallValueId] at he; omega
    · norm_num [recallScoreGapWitness, recallKeyId, recallValueId] at he; omega
    · exact ⟨1, 0, by decide, by simp [recallScoreGapWitness]⟩
    · norm_num [recallScoreGapWitness, recallKeyId, recallValueId] at he; omega
    · exact ⟨3, 0, by decide, by simp [recallScoreGapWitness]⟩
    · norm_num [recallScoreGapWitness, recallKeyId, recallValueId] at he; omega

/-- The final table value in the overwrite control is its query key's latest raw write.
Source: the fixed two-record raw table; earlier values do not supersede the second record. -/
theorem recallScoreGapWitness_latest : RecallRawLatestWrite 2 recallScoreGapWitness 4 0 := by
  intro record previous value hp hk hv hin
  exact hin

/-- Full-row, causal-tail and genuine retrieval hypotheses hold simultaneously on the actual raw overwrite control.
Source: genuine six-token example data, without any prepared head, score gap or model-answer assumption. -/
example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (6 : ℕ) ≤ 64 ∧
    RecallRawLayout 2 recallScoreGapWitness 0 ∧ recallScoreGapWitness 5 = recallKeyId 0 ∧
    (3 : Fin 6).val + 1 = (4 : Fin 6).val ∧ recallScoreGapWitness 3 = recallKeyId 0 ∧
    recallScoreGapWitness 4 = recallValueId 1 ∧ (4 : Fin 6).val ≤ 2 * 2 ∧
    (4 : Fin 6).val ≤ (5 : Fin 6).val ∧ RecallRawLatestWrite 2 recallScoreGapWitness 4 0 := by
  exact ⟨by norm_num, by norm_num, by decide, recallScoreGapWitness_layout, by simp [recallScoreGapWitness],
    by decide, by simp [recallScoreGapWitness], by simp [recallScoreGapWitness], by decide, by decide,
    recallScoreGapWitness_latest⟩

/-- In the actual raw overwrite control, the genuine score strictly prefers the second write despite real copy errors.
Source: the complete raw routing theorem and positive retained margin, tested on different stored values. -/
theorem recallScoreGapWitness_later_score (alpha eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) :
    recallMatchScore 2 alpha eps recallScoreGapWitness 5 2 < recallMatchScore 2 alpha eps recallScoreGapWitness 5 4 := by
  have hg := recall_raw_row_score_gap 2 alpha eps heps hclip (by decide) recallScoreGapWitness 0 5 4 3
    recallScoreGapWitness_layout 0 1 (by simp [recallScoreGapWitness]) (by decide)
    (by simp [recallScoreGapWitness]) (by simp [recallScoreGapWitness]) (by decide) (by decide)
    recallScoreGapWitness_latest 2 (by decide)
  have hp := recallMatchGap_pos alpha
  linarith

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 := by norm_num

end Transformer.GPTMini.Semantics
