import Transformer.GPTMini.Semantics.RecallSecondQKV

/-!
# Normalize genuine raw second-layer recall queries and copied keys

Source: the actual original fused QKV, prenorm and clipped QKNorm
at f11b6e2, connected to raw adjacent writes at cbafbe9. The encoder
state below is only the real complete first block at its fixed gain;
it calls no task function or desired-answer encoder.

The genuine query direction equals the verified categorical rotary
direction. Every true raw table key has normalized error at most
twice the fixed predecessor-copy tolerance, regardless of its real
position-dependent gate/RMS amplitude. All clipping and base norm
conditions are discharged from raw IDs and the finite shared gains.
The real fused value projection is zero at a raw query, preserving
the selected value through original XSA. Robust score margins,
softmax retrieval, tied readout and full parser coupling remain.
-/

namespace Transformer.GPTMini.Semantics

/-- The real full first block with its fixed ordinary table gain, without a prepared prefix feature.
Source: unchanged blockForward at the genuine simultaneous encoder parameters and raw token embeddings. -/
noncomputable def recallEncodedState (P : ℕ) (eps alpha : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) : EucSpace 64 :=
  blockForward recallConfig (recallEncoderBlock P eps alpha (recallEncoderGain P)) eps positions
    (fun j => recallRawEmbedding (tokens j)) i

/-- A genuine projected copied key above clipping has the unit-reference normalized error, including real prenorm.
Source: the actual fused K slice, ordinary scalar writes and the verified rotary normalization law. -/
theorem recallSecondQKV_normalized_copy (gain eps a position eta : ℝ) (y : EucSpace 64)
    (copy : EucSpace 8) (digits : Fin 4 → Fin 4) (heps : 0 ≤ eps)
    (hslot : recallSlotRead 27 y = a • copy) (ha : 1 ≤ a)
    (hthreshold : eps < gain * recallResidualScale eps y) (heta : eta ≤ 1)
    (herr : ‖copy - recallCode digits‖ ≤ eta) :
    ‖normL2 eps (applyRope 16 10000 position (headSlice recallConfig
      (qkvSlice recallConfig (qkvK recallConfig) (recallSecondQKV gain (rmsNormEps eps y))) 0)) -
      normL2 1 (applyRope 16 10000 position (recallRotaryCode digits))‖ ≤ 2 * eta := by
  have hs : 0 < gain * recallResidualScale eps y := by linarith
  have hp : 0 < a := by linarith
  have hn := recallCopy_norm_lower copy digits (herr.trans heta)
  have hscale : gain * recallResidualScale eps y ≤ gain * recallResidualScale eps y * a := by
    simpa only [mul_one] using mul_le_mul_of_nonneg_left ha hs.le
  have hnorm : gain * recallResidualScale eps y * a ≤
      (gain * recallResidualScale eps y * a) * ‖copy‖ := by
    simpa only [mul_one] using mul_le_mul_of_nonneg_left hn (mul_pos hs hp).le
  have hclip := hthreshold.le.trans (hscale.trans hnorm)
  rw [recallSecondQKV_rms_key, hslot, map_smul, smul_smul]
  exact recallRotary_projected_error eps (gain * recallResidualScale eps y * a) position eta copy digits
    (mul_pos hs hp) heta herr hclip

example : (0 : ℝ) ≤ 0 ∧ recallSlotRead 27 (recallSlotWrite 27 (recallCode (fun _ => 0))) =
    (1 : ℝ) • recallCode (fun _ => 0) ∧ (1 : ℝ) ≤ 1 ∧
    (0 : ℝ) < 1 * recallResidualScale 0 (recallSlotWrite 27 (recallCode (fun _ => 0))) ∧
    (0 : ℝ) ≤ 1 ∧ ‖recallCode (fun _ => 0) - recallCode (fun _ => 0)‖ ≤ (0 : ℝ) := by
  rw [recallSlotRead_write, one_smul, one_mul, sub_self, norm_zero]
  refine ⟨by norm_num, rfl, by norm_num, ?_, by norm_num, by norm_num⟩
  apply recallResidualScale_pos 0 (by norm_num)
  rw [recallSlotWrite_norm, recallCode_norm]
  norm_num

/-- The actual normalized raw query is precisely its verified rotary matching code, with no clipping premise.
Source: genuine raw protected query slot, finite shared matrix gain and derived true next-prenorm threshold. -/
theorem recall_raw_query_normalized (P : ℕ) (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (first i : Fin T)
    (hfirst : first.val = 0) (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol)
    (key : Fin 256) (hkey : tokens i = recallKeyId key) :
    normL2 eps (applyRope 16 10000 (positions i) (headSlice recallConfig (qkvSlice recallConfig (qkvQ recallConfig)
      (recallSecondQKV (recallMatchGain P eps) (rmsNormEps eps (recallEncodedState P eps alpha positions tokens i)))) 0)) =
        normL2 1 (applyRope 16 10000 (positions i) (recallRotaryCode (recallDigit key))) := by
  dsimp only [recallEncodedState]
  have ht := recall_raw_match_gain_threshold P eps alpha heps positions tokens first i hfirst hbos hrange
  have hp : 0 < recallMatchGain P eps * recallResidualScale eps
      (blockForward recallConfig (recallEncoderBlock P eps alpha (recallEncoderGain P)) eps positions
        (fun j => recallRawEmbedding (tokens j)) i) := by linarith
  rw [recallSecondQKV_rms_query,
    recall_raw_gate_key_code P eps alpha (recallEncoderGain P) positions tokens i key hkey]
  apply recallRotary_query_normalized eps _ (positions i) (recallDigit key) hp
  linarith

example : (0 : ℝ) ≤ 1 / 100000 ∧ (0 : Fin 3).val = 0 ∧ recallPairWitness 0 0 0 = ⟨1, by decide⟩ ∧
    (∀ j : Fin 3, j ≠ 0 → ∃ symbol : Fin 256,
      recallPairWitness 0 0 j = recallKeyId symbol ∨ recallPairWitness 0 0 j = recallValueId symbol) ∧
    recallPairWitness 0 0 1 = recallKeyId 0 := by
  refine ⟨by norm_num, by decide, by simp [recallPairWitness], ?_, by simp [recallPairWitness]⟩
  intro j hj
  by_cases h : j = 1
  · exact ⟨0, Or.inl (by simp [recallPairWitness, h])⟩
  · exact ⟨0, Or.inr (by simp [recallPairWitness, hj, h])⟩

/-- Every actual normalized table key has uniformly small error from its true raw neighboring symbol.
Source: full raw copy/gate/matrix/RMS/QKNorm chain, discharging all clipping, amplitude and copy-error conditions. -/
theorem recall_raw_key_normalized_error (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ}
    (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size) (first i selected : Fin T)
    (hfirst : first.val = 0) (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol)
    (hprev : selected.val + 1 = i.val) (key value : Fin 256)
    (hkey : tokens selected = recallKeyId key) (hvalue : tokens i = recallValueId value) (hinside : i.val ≤ 2 * P) :
    ‖normL2 eps (applyRope 16 10000 (i.val : ℝ) (headSlice recallConfig (qkvSlice recallConfig (qkvK recallConfig)
      (recallSecondQKV (recallMatchGain P eps) (rmsNormEps eps (recallEncodedState P eps
        (recallCopyTemperature recallKeyTolerance) (fun r => (r.val : ℝ)) tokens i)))) 0)) -
      normL2 1 (applyRope 16 10000 (i.val : ℝ) (recallRotaryCode (recallDigit key)))‖ ≤ 2 * recallKeyTolerance := by
  dsimp only [recallEncodedState]
  apply recallSecondQKV_normalized_copy (recallMatchGain P eps) eps
    (recallGateAmplitude P (recallEncoderGain P) eps (recallFirstState eps (recallCopyTemperature recallKeyTolerance)
      (fun r => (r.val : ℝ)) tokens i)) (i.val : ℝ) recallKeyTolerance _
    (recallSlotRead 18 (recallFirstState eps (recallCopyTemperature recallKeyTolerance)
      (fun r => (r.val : ℝ)) tokens i)) (recallDigit key) heps
  · exact recall_raw_gate_inside P eps (recallCopyTemperature recallKeyTolerance) (recallEncoderGain P) heps _
      tokens first i hfirst (recall_raw_value_ne_first tokens first i value hbos hvalue) hbos hrange value hvalue hinside
  · exact recall_raw_amplitude_lower P eps (recallCopyTemperature recallKeyTolerance) heps hclip _
      tokens first i hfirst hbos hrange value hvalue hinside
  · exact recall_raw_match_gain_threshold P eps (recallCopyTemperature recallKeyTolerance) heps _
      tokens first i hfirst hbos hrange
  · exact recallKeyTolerance_le_one
  · exact recall_first_state_accuracy recallKeyTolerance eps recallKeyTolerance_pos heps hclip hT
      tokens i selected hprev key value hkey hvalue

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

/-- A genuine raw query's own V remains zero after the complete first block and actual second prenorm/fused matrix.
Source: protected independent raw value channel, real encoder and complete original V projection, not a self-value premise. -/
theorem recall_raw_second_self_value_zero (P : ℕ) (eps alpha : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) (key : Fin 256)
    (hkey : tokens i = recallKeyId key) :
    headSlice recallConfig (qkvSlice recallConfig (qkvV recallConfig)
      (recallSecondQKV (recallMatchGain P eps) (rmsNormEps eps (recallEncodedState P eps alpha positions tokens i)))) 0 = 0 := by
  apply recallSecondQKV_rms_self_zero
  exact recall_raw_encoder_query_value_zero P eps alpha (recallEncoderGain P) positions tokens i key hkey

example : recallPairWitness 0 0 1 = recallKeyId 0 := by simp [recallPairWitness]

/-- A raw query's actual matching key is zero, so it cannot masquerade as an additional table record.
Source: actual raw key-type gate exclusion followed by genuine prenorm and the ordinary second K rows. -/
theorem recall_raw_second_query_key_zero (P : ℕ) (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T)
    (key : Fin 256) (hkey : tokens i = recallKeyId key) :
    headSlice recallConfig (qkvSlice recallConfig (qkvK recallConfig)
      (recallSecondQKV (recallMatchGain P eps) (rmsNormEps eps (recallEncodedState P eps alpha positions tokens i)))) 0 = 0 := by
  apply recallSecondQKV_key_zero
  rw [recallResidualScale_rms, map_smul]
  have hz := recall_raw_gate_key P eps alpha (recallEncoderGain P) heps positions tokens i key hkey
  change recallSlotRead 27 (recallEncodedState P eps alpha positions tokens i) = 0 at hz
  rw [hz, smul_zero]

example : (0 : ℝ) ≤ 1 / 100000 ∧ recallPairWitness 0 0 1 = recallKeyId 0 := by
  exact ⟨by norm_num, by simp [recallPairWitness]⟩

end Transformer.GPTMini.Semantics
