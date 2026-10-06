import Transformer.GPTMini.Semantics.RecallCopyAccuracy

/-!
# A uniformly accurate nonzero raw binding in the full original block

Source: MQAR's adjacent raw writes at cbafbe9 and the original
softmax GPTMini at f11b6e2. The fixed finite predecessor temperature
uses the derived global key tolerance. Actual raw table values store
their neighbor's compact key with relative error at most that
tolerance; no small encoder-error hypothesis is supplied.

The result includes the simultaneous marker/table gate, real RMS
scales, true FFN and both residual additions. With positive finite
gain, every such stored key has positive norm and a positive actual
amplitude. This supplies directions for the genuine next Q/K layer
without pretending every table value has the same RMS multiplier.
Complete validated-parser coupling and second-block saturation,
routing/value/readout margins remain separate obligations.
-/

namespace Transformer.GPTMini.Semantics

/-- A minimal raw BOS/key/value table used only to witness the local proof hypotheses.
Source: the original raw integer intervals; this is example data, not a model encoder or oracle. -/
def recallPairWitness (key value : Fin 256) (j : Fin 3) : Fin recallConfig.vocab_size :=
  if j = 0 then ⟨1, by decide⟩ else if j = 1 then recallKeyId key else recallValueId value

/-- The actual first full block stores a true neighboring raw key with the fixed derived relative error.
Source: genuine raw adjacency copy, finite shared temperature, actual table gate and complete original block. -/
theorem recall_raw_binding_error (P : ℕ) (eps beta : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1)
    {T : ℕ} (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size) (first i selected : Fin T)
    (hfirst : first.val = 0) (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol)
    (hprev : selected.val + 1 = i.val) (key value : Fin 256)
    (hkey : tokens selected = recallKeyId key) (hvalue : tokens i = recallValueId value) (hinside : i.val ≤ 2 * P) :
    let alpha := recallCopyTemperature recallKeyTolerance
    let a := recallGateAmplitude P beta eps (recallFirstState eps alpha (fun r => (r.val : ℝ)) tokens i)
    ‖recallSlotRead 27 (blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps (fun r => (r.val : ℝ))
        (fun j => recallRawEmbedding (tokens j)) i) - a • recallSymbolCode key‖ ≤ |a| * recallKeyTolerance := by
  dsimp only
  rw [recall_raw_gate_inside P eps (recallCopyTemperature recallKeyTolerance) beta heps _ tokens first i
    hfirst (recall_raw_value_ne_first tokens first i value hbos hvalue) hbos hrange value hvalue hinside,
    ← smul_sub, norm_smul, Real.norm_eq_abs]
  exact mul_le_mul_of_nonneg_left
    (recall_first_state_accuracy recallKeyTolerance eps recallKeyTolerance_pos heps hclip hT tokens i selected
      hprev key value hkey hvalue) (abs_nonneg _)

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (3 : ℕ) ≤ 64 ∧
    (0 : Fin 3).val = 0 ∧ recallPairWitness 0 0 0 = ⟨1, by decide⟩ ∧
    (∀ j : Fin 3, j ≠ 0 → ∃ symbol : Fin 256,
      recallPairWitness 0 0 j = recallKeyId symbol ∨ recallPairWitness 0 0 j = recallValueId symbol) ∧
    (1 : Fin 3).val + 1 = (2 : Fin 3).val ∧ recallPairWitness 0 0 1 = recallKeyId 0 ∧
    recallPairWitness 0 0 2 = recallValueId 0 ∧ (2 : Fin 3).val ≤ 2 * 1 := by
  refine ⟨by norm_num, by norm_num, by decide, by decide, by simp [recallPairWitness], ?_, by decide,
    by simp [recallPairWitness], by simp [recallPairWitness], by decide⟩
  intro j hj
  by_cases h : j = 1
  · exact ⟨0, Or.inl (by simp [recallPairWitness, h])⟩
  · exact ⟨0, Or.inr (by simp [recallPairWitness, hj, h])⟩

/-- The genuine stored raw table key has norm at least its actual strictly positive amplitude.
Source: derived finite copy tolerance at most one, exact original gate scaling and a norm-two raw reference key. -/
theorem recall_raw_binding_norm (P : ℕ) (eps beta : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1) (hbeta : 0 < beta)
    {T : ℕ} (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size) (first i selected : Fin T)
    (hfirst : first.val = 0) (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol)
    (hprev : selected.val + 1 = i.val) (key value : Fin 256)
    (hkey : tokens selected = recallKeyId key) (hvalue : tokens i = recallValueId value) (hinside : i.val ≤ 2 * P) :
    let alpha := recallCopyTemperature recallKeyTolerance
    recallGateAmplitude P beta eps (recallFirstState eps alpha (fun r => (r.val : ℝ)) tokens i) ≤
      ‖recallSlotRead 27 (blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps (fun r => (r.val : ℝ))
        (fun j => recallRawEmbedding (tokens j)) i)‖ := by
  dsimp only
  have hpos := recall_raw_gate_amplitude_pos P eps (recallCopyTemperature recallKeyTolerance) beta heps hbeta
    (fun r => (r.val : ℝ)) tokens first i hfirst hbos hrange value hvalue hinside
  have hnorm := recall_first_state_copy_nonzero recallKeyTolerance eps recallKeyTolerance_pos recallKeyTolerance_le_one
    heps.le hclip hT tokens i selected hprev key value hkey hvalue
  rw [recall_raw_gate_inside P eps (recallCopyTemperature recallKeyTolerance) beta heps.le _ tokens first i
    hfirst (recall_raw_value_ne_first tokens first i value hbos hvalue) hbos hrange value hvalue hinside,
    norm_smul, Real.norm_eq_abs, abs_of_pos hpos]
  simpa only [mul_one] using mul_le_mul_of_nonneg_left hnorm hpos.le

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧ (3 : ℕ) ≤ 64 ∧
    (0 : Fin 3).val = 0 ∧ recallPairWitness 0 0 0 = ⟨1, by decide⟩ ∧
    (∀ j : Fin 3, j ≠ 0 → ∃ symbol : Fin 256,
      recallPairWitness 0 0 j = recallKeyId symbol ∨ recallPairWitness 0 0 j = recallValueId symbol) ∧
    (1 : Fin 3).val + 1 = (2 : Fin 3).val ∧ recallPairWitness 0 0 1 = recallKeyId 0 ∧
    recallPairWitness 0 0 2 = recallValueId 0 ∧ (2 : Fin 3).val ≤ 2 * 1 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by decide, by decide, by simp [recallPairWitness], ?_, by decide,
    by simp [recallPairWitness], by simp [recallPairWitness], by decide⟩
  intro j hj
  by_cases h : j = 1
  · exact ⟨0, Or.inl (by simp [recallPairWitness, h])⟩
  · exact ⟨0, Or.inr (by simp [recallPairWitness, hj, h])⟩

/-- Every true stored table key has strictly positive norm under the given finite original block parameters.
Source: actual positive gate amplitude and the norm lower bound derived from raw finite-temperature copying. -/
theorem recall_raw_binding_positive (P : ℕ) (eps beta : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1) (hbeta : 0 < beta)
    {T : ℕ} (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size) (first i selected : Fin T)
    (hfirst : first.val = 0) (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol)
    (hprev : selected.val + 1 = i.val) (key value : Fin 256)
    (hkey : tokens selected = recallKeyId key) (hvalue : tokens i = recallValueId value) (hinside : i.val ≤ 2 * P) :
    0 < ‖recallSlotRead 27 (blockForward recallConfig (recallEncoderBlock P eps
        (recallCopyTemperature recallKeyTolerance) beta) eps (fun r => (r.val : ℝ))
      (fun j => recallRawEmbedding (tokens j)) i)‖ := by
  have ha := recall_raw_gate_amplitude_pos P eps (recallCopyTemperature recallKeyTolerance) beta heps hbeta
    (fun r => (r.val : ℝ)) tokens first i hfirst hbos hrange value hvalue hinside
  exact lt_of_lt_of_le ha (recall_raw_binding_norm P eps beta heps hclip hbeta hT tokens first i selected
    hfirst hbos hrange hprev key value hkey hvalue hinside)

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧ (3 : ℕ) ≤ 64 ∧
    (0 : Fin 3).val = 0 ∧ recallPairWitness 0 0 0 = ⟨1, by decide⟩ ∧
    (∀ j : Fin 3, j ≠ 0 → ∃ symbol : Fin 256,
      recallPairWitness 0 0 j = recallKeyId symbol ∨ recallPairWitness 0 0 j = recallValueId symbol) ∧
    (1 : Fin 3).val + 1 = (2 : Fin 3).val ∧ recallPairWitness 0 0 1 = recallKeyId 0 ∧
    recallPairWitness 0 0 2 = recallValueId 0 ∧ (2 : Fin 3).val ≤ 2 * 1 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by decide, by decide, by simp [recallPairWitness], ?_, by decide,
    by simp [recallPairWitness], by simp [recallPairWitness], by decide⟩
  intro j hj
  by_cases h : j = 1
  · exact ⟨0, Or.inl (by simp [recallPairWitness, h])⟩
  · exact ⟨0, Or.inr (by simp [recallPairWitness, hj, h])⟩

/-- The same given finite-temperature block retains the true raw value code alongside its bound key.
Source: simultaneous original residual layout and the actual raw value interval, not a paired semantic input. -/
theorem recall_raw_binding_value_code (P : ℕ) (eps beta : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) (value : Fin 256)
    (hvalue : tokens i = recallValueId value) :
    recallSlotRead 9 (blockForward recallConfig (recallEncoderBlock P eps
        (recallCopyTemperature recallKeyTolerance) beta) eps (fun r => (r.val : ℝ))
      (fun j => recallRawEmbedding (tokens j)) i) = recallSymbolCode value := by
  exact recall_raw_gate_value_code P eps (recallCopyTemperature recallKeyTolerance) beta _ tokens i value hvalue

example : recallPairWitness 0 0 2 = recallValueId 0 := by simp [recallPairWitness]

/-- The given finite block also retains a real raw query code for subsequent learned query-key matching.
Source: the complete original first block's protected key slot, with the derived copy temperature fixed globally. -/
theorem recall_raw_binding_query_code (P : ℕ) (eps beta : ℝ) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) (key : Fin 256)
    (hkey : tokens i = recallKeyId key) :
    recallSlotRead 1 (blockForward recallConfig (recallEncoderBlock P eps
        (recallCopyTemperature recallKeyTolerance) beta) eps (fun r => (r.val : ℝ))
      (fun j => recallRawEmbedding (tokens j)) i) = recallSymbolCode key := by
  exact recall_raw_gate_key_code P eps (recallCopyTemperature recallKeyTolerance) beta _ tokens i key hkey

example : recallPairWitness 0 0 1 = recallKeyId 0 := by simp [recallPairWitness]

end Transformer.GPTMini.Semantics
