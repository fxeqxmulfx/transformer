import Transformer.GPTMini.Semantics.RecallReadoutCoordinates

/-!
# Strict actual recall tied readout across the entire vocabulary

Source: original tied embedding readout and final RMSNorm at f11b6e2,
with MQAR raw adjacency/chronological overwrite semantics at cbafbe9.
The full two-block output's independently proved error of at most
one eighth gives score errors at most three eighths. Its selected
value amplitude is at least two, derived using a fixed ordinary gain.

These facts yield a strict advantage over all 547 competitors: the
categorical gap defeats every wrong value, and the correct value
also exceeds all raw key and reserved scores. Final RMSNorm retains
the advantage because the protected constant proves its actual
multiplier positive. No prepared logits, selected-value embedding
input, norm bound or normalization factor is a premise. Raw layout
and last-write predicates still need discharge from the full Basis
parser, and the two blocks still need top-level model/decoder coupling.
-/

namespace Transformer.GPTMini.Semantics

/-- The genuine two-block final state always has norm at least one through its protected raw constant.
Source: actual residual supports, the complete raw embedding table and the real coordinate-norm inequality. -/
theorem recallFinalState_norm_lower (P : ℕ) (eps : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query : Fin T) :
    1 ≤ ‖recallFinalState P eps tokens query‖ := by
  have h := PiLp.norm_apply_le (recallFinalState P eps tokens query) 0
  rw [recallFinalState_protected P eps tokens query 0 (by decide), recallRawEmbedding_constant,
    Real.norm_eq_abs, abs_one] at h
  exact h

/-- The actual final normalization scale is positive, even when the real retrieval residual is large.
Source: its genuine norm-at-least-one state and the original RMS denominator, with no common-scale assumption. -/
theorem recallFinalState_scale_pos (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (query : Fin T) :
    0 < recallResidualScale eps (recallFinalState P eps tokens query) :=
  recallResidualScale_pos eps heps _ (recallFinalState_norm_lower P eps tokens query)

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

/-- The actual complete raw recall residual strictly prefers its real selected value over every other tied token.
Source: the derived full-block finite-softmax error, true selected RMS amplitude and actual whole-vocabulary score identities. -/
theorem recallFinalState_tied_margin (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ}
    (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size) (first query selected previous : Fin T)
    (hlayout : RecallRawLayout P tokens first) (key value : Fin 256) (hquery : tokens query = recallKeyId key)
    (hprev : previous.val + 1 = selected.val) (hkey : tokens previous = recallKeyId key)
    (hvalue : tokens selected = recallValueId value) (hinside : selected.val ≤ 2 * P)
    (hvisible : selected.val ≤ query.val) (hlatest : RecallRawLatestWrite P tokens selected key) :
    ∀ token : Fin recallConfig.vocab_size, token ≠ recallValueId value →
      inner (𝕜 := ℝ) (recallFinalState P eps tokens query) (recallRawEmbedding token) <
        inner (𝕜 := ℝ) (recallFinalState P eps tokens query) (recallRawEmbedding (recallValueId value)) := by
  set a := recallValueAmplitude P eps tokens selected with ha
  have hl : 2 ≤ a := recallValueAmplitude_lower P eps heps tokens first selected hlayout
  have he := recallFinalState_error P eps heps hclip hT tokens first query selected previous
    hlayout key value hquery hprev hkey hvalue hinside hvisible hlatest
  have hv (other : Fin 256) :
      |inner (𝕜 := ℝ) (recallFinalState P eps tokens query) (recallRawEmbedding (recallValueId other)) -
        (1 + a * inner (𝕜 := ℝ) (recallSymbolCode value) (recallSymbolCode other))| ≤ 3 / 8 := by
    rw [← recallReference_value_score P eps a tokens query key value other hquery]
    exact (recallRawEmbedding_score_error _ _ _).trans (by nlinarith [he])
  have hw := abs_le.mp (hv value)
  rw [recallSymbolCode_inner_self] at hw
  intro token hne
  by_cases hk : 36 ≤ token.val ∧ token.val < 292
  · let other : Fin 256 := ⟨token.val - 36, by omega⟩
    have ht : token = recallKeyId other := by apply Fin.ext; change token.val = 36 + (token.val - 36); omega
    rw [ht]
    have hb := recallFinalState_key_score P eps tokens query key other hquery
    linarith [hw.1]
  · by_cases hval : 292 ≤ token.val
    · let other : Fin 256 := ⟨token.val - 292, by have ht := token.isLt; change token.val < 548 at ht; omega⟩
      have ht : token = recallValueId other := by apply Fin.ext; change token.val = 292 + (token.val - 292); omega
      have hd : value ≠ other := by intro h; apply hne; rw [ht, ← h]
      have ho := (abs_le.mp (hv other)).2
      have hi := mul_le_mul_of_nonneg_left (recallSymbolCode_inner_gap value other hd) (by linarith : 0 ≤ a)
      rw [ht]
      linarith [hw.1]
    · rw [recallRawEmbedding, dite_eq_right hk, dite_eq_right hval,
        recallFinalState_reserved_score P eps tokens query key hquery]
      linarith [hw.1]

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (6 : ℕ) ≤ 64 ∧
    RecallRawLayout 2 recallScoreGapWitness 0 ∧ recallScoreGapWitness 5 = recallKeyId 0 ∧
    (3 : Fin 6).val + 1 = (4 : Fin 6).val ∧ recallScoreGapWitness 3 = recallKeyId 0 ∧
    recallScoreGapWitness 4 = recallValueId 1 ∧ (4 : Fin 6).val ≤ 2 * 2 ∧
    (4 : Fin 6).val ≤ (5 : Fin 6).val ∧ RecallRawLatestWrite 2 recallScoreGapWitness 4 0 := by
  exact ⟨by norm_num, by norm_num, by decide, recallScoreGapWitness_layout, by simp [recallScoreGapWitness],
    by decide, by simp [recallScoreGapWitness], by simp [recallScoreGapWitness], by decide, by decide,
    recallScoreGapWitness_latest⟩

/-- The genuine final RMSNorm preserves the derived strict recall answer margin over all checked vocabulary entries.
Source: original final normalization/tied readout, with actual positivity derived from the complete raw two-block state. -/
theorem recallFinalState_rms_margin (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ}
    (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size) (first query selected previous : Fin T)
    (hlayout : RecallRawLayout P tokens first) (key value : Fin 256) (hquery : tokens query = recallKeyId key)
    (hprev : previous.val + 1 = selected.val) (hkey : tokens previous = recallKeyId key)
    (hvalue : tokens selected = recallValueId value) (hinside : selected.val ≤ 2 * P)
    (hvisible : selected.val ≤ query.val) (hlatest : RecallRawLatestWrite P tokens selected key) :
    ∀ token : Fin recallConfig.vocab_size, token ≠ recallValueId value →
      inner (𝕜 := ℝ) (rmsNormEps eps (recallFinalState P eps tokens query)) (recallRawEmbedding token) <
        inner (𝕜 := ℝ) (rmsNormEps eps (recallFinalState P eps tokens query)) (recallRawEmbedding (recallValueId value)) := by
  intro token hne
  rw [recallResidualScale_rms, real_inner_smul_left, real_inner_smul_left]
  exact mul_lt_mul_of_pos_left (recallFinalState_tied_margin P eps heps hclip hT tokens first query selected previous
    hlayout key value hquery hprev hkey hvalue hinside hvisible hlatest token hne)
      (recallFinalState_scale_pos P eps heps tokens query)

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (6 : ℕ) ≤ 64 ∧
    RecallRawLayout 2 recallScoreGapWitness 0 ∧ recallScoreGapWitness 5 = recallKeyId 0 ∧
    (3 : Fin 6).val + 1 = (4 : Fin 6).val ∧ recallScoreGapWitness 3 = recallKeyId 0 ∧
    recallScoreGapWitness 4 = recallValueId 1 ∧ (4 : Fin 6).val ≤ 2 * 2 ∧
    (4 : Fin 6).val ≤ (5 : Fin 6).val ∧ RecallRawLatestWrite 2 recallScoreGapWitness 4 0 := by
  exact ⟨by norm_num, by norm_num, by decide, recallScoreGapWitness_layout, by simp [recallScoreGapWitness],
    by decide, by simp [recallScoreGapWitness], by simp [recallScoreGapWitness], by decide, by decide,
    recallScoreGapWitness_latest⟩

/-- Greedy decoding of the genuine final normalized tied scores returns precisely the selected raw value ID.
Source: original whole-vocabulary greedy readout and the derived strict finite-temperature answer margin, including tie exclusion. -/
theorem recallFinalState_greedy_value (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ}
    (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size) (first query selected previous : Fin T)
    (hlayout : RecallRawLayout P tokens first) (key value : Fin 256) (hquery : tokens query = recallKeyId key)
    (hprev : previous.val + 1 = selected.val) (hkey : tokens previous = recallKeyId key)
    (hvalue : tokens selected = recallValueId value) (hinside : selected.val ≤ 2 * P)
    (hvisible : selected.val ≤ query.val) (hlatest : RecallRawLatestWrite P tokens selected key) :
    TokenInterface.bestToken recallConfig.vocab_pos (fun token =>
      inner (𝕜 := ℝ) (rmsNormEps eps (recallFinalState P eps tokens query)) (recallRawEmbedding token)) =
        recallValueId value := by
  apply TokenInterface.bestToken_of_strict
  exact recallFinalState_rms_margin P eps heps hclip hT tokens first query selected previous
    hlayout key value hquery hprev hkey hvalue hinside hvisible hlatest

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (6 : ℕ) ≤ 64 ∧
    RecallRawLayout 2 recallScoreGapWitness 0 ∧ recallScoreGapWitness 5 = recallKeyId 0 ∧
    (3 : Fin 6).val + 1 = (4 : Fin 6).val ∧ recallScoreGapWitness 3 = recallKeyId 0 ∧
    recallScoreGapWitness 4 = recallValueId 1 ∧ (4 : Fin 6).val ≤ 2 * 2 ∧
    (4 : Fin 6).val ≤ (5 : Fin 6).val ∧ RecallRawLatestWrite 2 recallScoreGapWitness 4 0 := by
  exact ⟨by norm_num, by norm_num, by decide, recallScoreGapWitness_layout, by simp [recallScoreGapWitness],
    by decide, by simp [recallScoreGapWitness], by simp [recallScoreGapWitness], by decide, by decide,
    recallScoreGapWitness_latest⟩

/-- The actual six-token overwrite control decodes its later different-valued write, using finite shared original parameters.
Source: raw overwrite witness [BOS,key0,value0,key0,value1,key0] and the complete genuine tied readout theorem. -/
theorem recallScoreGapWitness_prediction (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) :
    TokenInterface.bestToken recallConfig.vocab_pos (fun token =>
      inner (𝕜 := ℝ) (rmsNormEps eps (recallFinalState 2 eps recallScoreGapWitness 5)) (recallRawEmbedding token)) =
        recallValueId 1 := by
  exact recallFinalState_greedy_value 2 eps heps hclip (by decide) recallScoreGapWitness 0 5 4 3
    recallScoreGapWitness_layout 0 1 (by simp [recallScoreGapWitness]) (by decide)
    (by simp [recallScoreGapWitness]) (by simp [recallScoreGapWitness]) (by decide) (by decide) recallScoreGapWitness_latest

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 := by norm_num

end Transformer.GPTMini.Semantics
