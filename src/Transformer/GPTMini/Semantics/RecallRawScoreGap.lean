import Transformer.GPTMini.Semantics.RecallScoreError

/-!
# Genuine raw table competitors retain a positive latest-write score gap

Source: original CausalMHA.forward at f11b6e2 and MQAR's adjacent,
chronological writes at cbafbe9. Each score in the main theorem is
evaluated by the complete real first encoder and true second QKV.
The competitor either has a different raw neighboring key or is an
earlier write of the query key. In both cases the derived rotary
margin exceeds both genuine copied-key perturbations.

One positive gap, exp(alpha)*latestMargin/2, remains for all these
real table competitors at context 64. Raw adjacency, table membership
and chronology are explicit local token conditions, not an assumed
ideal routing gap. Full validated-parser coupling must still derive
them for every legal prefix. Excluded nonrecords, finite-softmax
retrieval and complete tied readout remain subsequent obligations.
-/

namespace Transformer.GPTMini.Semantics

/-- A fixed positive real score gap after the two actual copied-key errors.
Source: the derived original latest-write margin and genuine temperature, with a conservative factor two. -/
noncomputable def recallMatchGap (alpha : ℝ) : ℝ := Real.exp alpha * recallLatestMargin / 2

/-- The retained actual routing gap is strictly positive at every finite temperature.
Source: the proved original rotary margin and positivity of the original exponential gain. -/
theorem recallMatchGap_pos (alpha : ℝ) : 0 < recallMatchGap alpha := by
  unfold recallMatchGap
  exact div_pos (mul_pos (Real.exp_pos alpha) recallLatestMargin_pos) (by norm_num)

/-- The sum of the two genuine score errors fits inside half the original latest-write gap.
Source: fixed copyTolerance=latestMargin/16, with the proved positive categorical/rotary margin. -/
theorem recallKeyTolerance_score_budget (alpha : ℝ) :
    4 * Real.exp alpha * recallKeyTolerance ≤ Real.exp alpha * recallLatestMargin / 2 := by
  have hm := mul_pos (Real.exp_pos alpha) recallLatestMargin_pos
  unfold recallKeyTolerance
  nlinarith

/-- Two derived score errors preserve a positive half-margin from a reference comparison.
Source: direct perturbation arithmetic for the actual original scores; the reference gap is discharged below. -/
theorem recallScore_gap_of_errors (alpha selected competitor referenceSelected referenceCompetitor : ℝ)
    (hselected : |selected - referenceSelected| ≤ 2 * Real.exp alpha * recallKeyTolerance)
    (hcompetitor : |competitor - referenceCompetitor| ≤ 2 * Real.exp alpha * recallKeyTolerance)
    (hgap : referenceCompetitor ≤ referenceSelected - Real.exp alpha * recallLatestMargin) :
    competitor ≤ selected - recallMatchGap alpha := by
  have hs := (abs_le.mp hselected).1
  have hc := (abs_le.mp hcompetitor).2
  have hb := recallKeyTolerance_score_budget alpha
  unfold recallMatchGap
  linarith

example : |recallLatestMargin - recallLatestMargin| ≤ 2 * Real.exp 0 * recallKeyTolerance ∧
    |(0 : ℝ) - 0| ≤ 2 * Real.exp 0 * recallKeyTolerance ∧
    (0 : ℝ) ≤ recallLatestMargin - Real.exp 0 * recallLatestMargin := by
  have ht := recallKeyTolerance_pos
  simp only [sub_self, abs_zero, Real.exp_zero, one_mul]
  exact ⟨by linarith, by linarith, le_rfl⟩

/-- A zero-score nonrecord also stays below an imperfect selected score with the same retained margin.
Source: the same actual perturbation budget, requiring only the selected reference's positive margin. -/
theorem recallScore_zero_gap_of_error (alpha selected referenceSelected : ℝ)
    (hselected : |selected - referenceSelected| ≤ 2 * Real.exp alpha * recallKeyTolerance)
    (hreference : Real.exp alpha * recallLatestMargin ≤ referenceSelected) :
    (0 : ℝ) ≤ selected - recallMatchGap alpha := by
  apply recallScore_gap_of_errors alpha selected 0 referenceSelected 0 hselected
  · rw [sub_self, abs_zero]
    exact mul_nonneg (by positivity) recallKeyTolerance_pos.le
  · linarith

example : |recallLatestMargin - recallLatestMargin| ≤ 2 * Real.exp 0 * recallKeyTolerance ∧
    Real.exp 0 * recallLatestMargin ≤ recallLatestMargin := by
  have ht := recallKeyTolerance_pos
  simp only [sub_self, abs_zero, Real.exp_zero, one_mul]
  exact ⟨by linarith, le_rfl⟩

/-- Any two actual integer positions in the original recall context have reference displacement at most sixty-four.
Source: Fin index bounds and Basis's original context cap, without an assumed geometric displacement. -/
theorem recall_index_distance {T : ℕ} (hT : T ≤ 64) (i j : Fin T) : |(i.val : ℝ) - (j.val : ℝ)| ≤ 64 := by
  have hi : (i.val : ℝ) < (T : ℝ) := by exact_mod_cast i.isLt
  have hj : (j.val : ℝ) < (T : ℝ) := by exact_mod_cast j.isLt
  have hc : (T : ℝ) ≤ 64 := by exact_mod_cast hT
  have hi0 := Nat.cast_nonneg (α := ℝ) i.val
  have hj0 := Nat.cast_nonneg (α := ℝ) j.val
  apply abs_le.mpr
  constructor <;> linarith

example : (6 : ℕ) ≤ 64 := by decide

/-- Every real raw table competitor is separated if its neighboring key differs or its matching write is earlier.
Source: actual full encoder/second scores, derived normalized error, original content/recency geometry and raw chronology. -/
theorem recall_raw_table_score_gap (P : ℕ) (alpha eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ}
    (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size)
    (first query selected selectedPrevious competitor competitorPrevious : Fin T)
    (hfirst : first.val = 0) (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol)
    (key other selectedValue competitorValue : Fin 256) (hquery : tokens query = recallKeyId key)
    (hselectedPrev : selectedPrevious.val + 1 = selected.val)
    (hselectedKey : tokens selectedPrevious = recallKeyId key)
    (hselectedValue : tokens selected = recallValueId selectedValue) (hselectedTable : selected.val ≤ 2 * P)
    (hcompetitorPrev : competitorPrevious.val + 1 = competitor.val)
    (hcompetitorKey : tokens competitorPrevious = recallKeyId other)
    (hcompetitorValue : tokens competitor = recallValueId competitorValue) (hcompetitorTable : competitor.val ≤ 2 * P)
    (hselectedVisible : selected.val ≤ query.val)
    (hlatest : other = key → competitor.val + 1 ≤ selected.val) :
    recallMatchScore P alpha eps tokens query competitor ≤
      recallMatchScore P alpha eps tokens query selected - recallMatchGap alpha := by
  apply recallScore_gap_of_errors alpha _ _
    (score alpha 1 (applyRope 16 10000 (query.val : ℝ) (recallRotaryCode (recallDigit key)))
      (applyRope 16 10000 (selected.val : ℝ) (recallRotaryCode (recallDigit key))))
    (score alpha 1 (applyRope 16 10000 (query.val : ℝ) (recallRotaryCode (recallDigit key)))
      (applyRope 16 10000 (competitor.val : ℝ) (recallRotaryCode (recallDigit other))))
  · exact recall_raw_score_error P alpha eps heps hclip hT tokens first query selected selectedPrevious
      hfirst hbos hrange key key selectedValue hquery hselectedPrev hselectedKey hselectedValue hselectedTable
  · exact recall_raw_score_error P alpha eps heps hclip hT tokens first query competitor competitorPrevious
      hfirst hbos hrange key other competitorValue hquery hcompetitorPrev hcompetitorKey hcompetitorValue hcompetitorTable
  · by_cases hk : other = key
    · subst other
      apply recall_latest_gap alpha 1 (by norm_num)
      · exact_mod_cast hlatest rfl
      · exact_mod_cast hselectedVisible
      · exact (abs_le.mp (recall_index_distance hT query competitor)).2
    · exact recall_different_latest_gap alpha 1 (by norm_num) key other (Ne.symm hk) _ _ _
        (recall_index_distance hT selected query) (recall_index_distance hT competitor query)

/-- A six-token raw overwrite control, used only to exhibit the theorem's hypotheses.
Source: MQAR's adjacent two-write hard-mode pattern; no model definition calls this witness. -/
def recallScoreGapWitness (i : Fin 6) : Fin recallConfig.vocab_size :=
  if i = 0 then ⟨1, by decide⟩ else if i = 1 ∨ i = 3 ∨ i = 5 then recallKeyId 0
  else if i = 4 then recallValueId 1 else recallValueId 0

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (6 : ℕ) ≤ 64 ∧ (0 : Fin 6).val = 0 ∧
    recallScoreGapWitness 0 = ⟨1, by decide⟩ ∧ (∀ j : Fin 6, j ≠ 0 → ∃ symbol : Fin 256,
      recallScoreGapWitness j = recallKeyId symbol ∨ recallScoreGapWitness j = recallValueId symbol) ∧
    recallScoreGapWitness 5 = recallKeyId 0 ∧ (3 : Fin 6).val + 1 = (4 : Fin 6).val ∧
    recallScoreGapWitness 3 = recallKeyId 0 ∧ recallScoreGapWitness 4 = recallValueId 1 ∧
    (4 : Fin 6).val ≤ 2 * 2 ∧ (1 : Fin 6).val + 1 = (2 : Fin 6).val ∧
    recallScoreGapWitness 1 = recallKeyId 0 ∧ recallScoreGapWitness 2 = recallValueId 0 ∧
    (2 : Fin 6).val ≤ 2 * 2 ∧ (4 : Fin 6).val ≤ (5 : Fin 6).val ∧
    ((0 : Fin 256) = 0 → (2 : Fin 6).val + 1 ≤ (4 : Fin 6).val) := by
  refine ⟨by norm_num, by norm_num, by decide, by decide, by simp [recallScoreGapWitness], ?_,
    by simp [recallScoreGapWitness], by decide, by simp [recallScoreGapWitness], by simp [recallScoreGapWitness],
    by decide, by decide, by simp [recallScoreGapWitness], by simp [recallScoreGapWitness], by decide, by decide,
    by decide⟩
  intro j hj
  fin_cases j
  · contradiction
  · exact ⟨0, Or.inl (by simp [recallScoreGapWitness])⟩
  · exact ⟨0, Or.inr (by simp [recallScoreGapWitness])⟩
  · exact ⟨0, Or.inl (by simp [recallScoreGapWitness])⟩
  · exact ⟨1, Or.inr (by simp [recallScoreGapWitness])⟩
  · exact ⟨0, Or.inl (by simp [recallScoreGapWitness])⟩

end Transformer.GPTMini.Semantics
