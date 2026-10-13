import Transformer.GPTMini.Semantics.RecallEncoderBounds

/-!
# One finite shared second-projection gain reaches actual QKNorm clipping

Source: ordinary shared QKV at f11b6e2 and the derived actual raw
encoder bounds. Its fixed table gain yields uniform whole-block
norm bound M=9+512*gain. The next genuine RMS multiplier is therefore
at least 8/sqrt(M^2+64*epsilon), a positive number independent of
the input, record position or stored key's amplitude.

One fixed finite gain (1+epsilon) divided by that lower multiplier
makes the actual product of QKV gain and next RMS scale exceed
epsilon at every raw position. Genuine copied table keys have norm
at least one and true queries have norm two, so this bound supplies
their saturation conditions. The complete fused second QKV and
robust score/readout still require actual operator coupling; no
input-dependent matrix or prepared common RMS scale is inserted.
-/

namespace Transformer.GPTMini.Semantics

/-- The uniform actual first-block norm bound at the fixed finite table gain.
Source: the derived raw whole-block bound, independent of any prefix or record. -/
noncomputable def recallEncoderNormBound (P : ℕ) : ℝ := 9 + 512 * recallEncoderGain P

/-- This fixed norm bound is positive and at least nine at every table size.
Source: the given ordinary table gain is strictly positive. -/
theorem recallEncoderNormBound_lower (P : ℕ) : 9 ≤ recallEncoderNormBound P := by
  have h := recallEncoderGain_pos P
  unfold recallEncoderNormBound
  linarith

/-- A fixed lower bound for the next actual prenorm multiplier, retaining the original epsilon.
Source: the genuine width-sixty-four RMS denominator and the derived complete raw encoder bound. -/
noncomputable def recallSecondScaleLower (P : ℕ) (eps : ℝ) : ℝ :=
  8 / Real.sqrt ((recallEncoderNormBound P) ^ 2 + 64 * eps)

/-- This fixed lower multiplier is strictly positive for nonnegative epsilon.
Source: the positive finite bound M and the original actual RMS denominator. -/
theorem recallSecondScaleLower_pos (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) :
    0 < recallSecondScaleLower P eps := by
  have hM := recallEncoderNormBound_lower P
  unfold recallSecondScaleLower
  apply div_pos (by norm_num)
  apply Real.sqrt_pos.mpr
  nlinarith

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

/-- The ordinary shared matching projection gain depends only on table size and model epsilon.
Source: sufficient finite gain compensating the genuine next-prenorm lower bound, without an input argument. -/
noncomputable def recallMatchGain (P : ℕ) (eps : ℝ) : ℝ := (1 + eps) / recallSecondScaleLower P eps

/-- The chosen shared Q/K matrix coefficient is positive at every nonnegative model epsilon.
Source: positive finite numerator and the derived fixed second-prenorm lower bound. -/
theorem recallMatchGain_pos (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) : 0 < recallMatchGain P eps := by
  unfold recallMatchGain
  exact div_pos (by linarith) (recallSecondScaleLower_pos P eps heps)

example : (0 : ℝ) ≤ 0 := by norm_num

/-- The exact product of the fixed gain and lower prenorm multiplier is one plus epsilon.
Source: the ordinary coefficient and proved nonzero denominator, rather than a per-input scale cancellation. -/
theorem recallMatchGain_product (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) :
    recallMatchGain P eps * recallSecondScaleLower P eps = 1 + eps := by
  unfold recallMatchGain
  exact div_mul_cancel₀ _ (ne_of_gt (recallSecondScaleLower_pos P eps heps))

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

/-- Genuine RMS at a bounded norm-at-least-one residual is no smaller than the fixed worst-denominator multiplier.
Source: the actual square-root denominator and monotonicity, with both norm conditions kept explicit locally. -/
theorem recallResidualScale_lower (M eps : ℝ) (heps : 0 ≤ eps) (x : EucSpace 64)
    (hx : 1 ≤ ‖x‖) (hu : ‖x‖ ≤ M) :
    8 / Real.sqrt (M ^ 2 + 64 * eps) ≤ recallResidualScale eps x := by
  have hd : 0 < ‖x‖ ^ 2 + 64 * eps := by nlinarith [norm_nonneg x]
  have hsq : ‖x‖ ^ 2 + 64 * eps ≤ M ^ 2 + 64 * eps := by nlinarith [norm_nonneg x]
  unfold recallResidualScale
  exact div_le_div_of_nonneg_left (by norm_num) (Real.sqrt_pos.mpr hd) (Real.sqrt_le_sqrt hsq)

example : (0 : ℝ) ≤ 1 / 100000 ∧ 1 ≤ ‖recallUnit 0‖ ∧ ‖recallUnit 0‖ ≤ (9 : ℝ) := by
  norm_num [recallUnit, PiLp.norm_single]

/-- Every true raw encoder state has the fixed positive second-prenorm lower bound, with no norm or gate premise.
Source: actual raw alphabet/BOS, whole first-block bound and preserved real constant at the fixed shared gain. -/
theorem recall_raw_second_scale_lower (P : ℕ) (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (first i : Fin T)
    (hfirst : first.val = 0) (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol) :
    recallSecondScaleLower P eps ≤ recallResidualScale eps
      (blockForward recallConfig (recallEncoderBlock P eps alpha (recallEncoderGain P)) eps positions
        (fun j => recallRawEmbedding (tokens j)) i) := by
  apply recallResidualScale_lower (recallEncoderNormBound P) eps heps
  · exact recallEncoderBlock_norm_lower P eps alpha (recallEncoderGain P) positions tokens i
  · exact recall_raw_encoder_norm P eps alpha (recallEncoderGain P) heps (recallEncoderGain_pos P).le
      positions tokens first i hfirst hbos hrange

example : (0 : ℝ) ≤ 1 / 100000 ∧ (0 : Fin 3).val = 0 ∧ recallPairWitness 0 0 0 = ⟨1, by decide⟩ ∧
    (∀ j : Fin 3, j ≠ 0 → ∃ symbol : Fin 256,
      recallPairWitness 0 0 j = recallKeyId symbol ∨ recallPairWitness 0 0 j = recallValueId symbol) := by
  refine ⟨by norm_num, by decide, by simp [recallPairWitness], ?_⟩
  intro j hj
  by_cases h : j = 1
  · exact ⟨0, Or.inl (by simp [recallPairWitness, h])⟩
  · exact ⟨0, Or.inr (by simp [recallPairWitness, hj, h])⟩

/-- One fixed actual Q/K coefficient times the true next RMS multiplier exceeds epsilon at every raw position.
Source: the derived worst-prefix bound, with neither encoded norm nor a prepared RMS multiplier assumed. -/
theorem recall_raw_match_gain_threshold (P : ℕ) (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (first i : Fin T)
    (hfirst : first.val = 0) (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol) :
    eps < recallMatchGain P eps * recallResidualScale eps
      (blockForward recallConfig (recallEncoderBlock P eps alpha (recallEncoderGain P)) eps positions
        (fun j => recallRawEmbedding (tokens j)) i) := by
  have hgain := recallMatchGain_pos P eps heps
  have hl := recall_raw_second_scale_lower P eps alpha heps positions tokens first i hfirst hbos hrange
  calc eps < 1 + eps := by linarith
    _ = recallMatchGain P eps * recallSecondScaleLower P eps := (recallMatchGain_product P eps heps).symm
    _ ≤ recallMatchGain P eps * recallResidualScale eps
        (blockForward recallConfig (recallEncoderBlock P eps alpha (recallEncoderGain P)) eps positions
          (fun j => recallRawEmbedding (tokens j)) i) := mul_le_mul_of_nonneg_left hl hgain.le

example : (0 : ℝ) ≤ 0 ∧ (0 : Fin 3).val = 0 ∧ recallPairWitness 0 0 0 = ⟨1, by decide⟩ ∧
    (∀ j : Fin 3, j ≠ 0 → ∃ symbol : Fin 256,
      recallPairWitness 0 0 j = recallKeyId symbol ∨ recallPairWitness 0 0 j = recallValueId symbol) := by
  refine ⟨by norm_num, by decide, by simp [recallPairWitness], ?_⟩
  intro j hj
  by_cases h : j = 1
  · exact ⟨0, Or.inl (by simp [recallPairWitness, h])⟩
  · exact ⟨0, Or.inr (by simp [recallPairWitness, hj, h])⟩

/-- Every raw value's genuine next-prenorm compact code has a positive uniform lower norm and finite upper norm.
Source: actual protected value slot, true position-dependent RMS scale and the shared raw-state bounds. -/
theorem recall_raw_prenorm_value_bounds (P : ℕ) (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (first i : Fin T)
    (hfirst : first.val = 0) (hbos : tokens first = (⟨1, by decide⟩ : Fin recallConfig.vocab_size))
    (hrange : ∀ j, j ≠ first → ∃ symbol : Fin 256,
      tokens j = recallKeyId symbol ∨ tokens j = recallValueId symbol)
    (value : Fin 256) (hvalue : tokens i = recallValueId value) :
    let x := blockForward recallConfig (recallEncoderBlock P eps alpha (recallEncoderGain P)) eps positions
      (fun j => recallRawEmbedding (tokens j)) i
    2 * recallSecondScaleLower P eps ≤ ‖recallSlotRead 9 (rmsNormEps eps x)‖ ∧
      ‖recallSlotRead 9 (rmsNormEps eps x)‖ ≤ 16 := by
  dsimp only
  have hs := recallEncoderBlock_scale_upper P eps alpha (recallEncoderGain P) heps positions tokens i
  have hl := recall_raw_second_scale_lower P eps alpha heps positions tokens first i hfirst hbos hrange
  rw [recallResidualScale_rms, map_smul,
    recall_raw_gate_value_code P eps alpha (recallEncoderGain P) positions tokens i value hvalue,
    norm_smul_of_nonneg hs.1.le, recallSymbolCode, recallCode_norm]
  constructor <;> linarith [hs.2]

example : (0 : ℝ) ≤ 1 / 100000 ∧ (0 : Fin 3).val = 0 ∧ recallPairWitness 0 0 0 = ⟨1, by decide⟩ ∧
    (∀ j : Fin 3, j ≠ 0 → ∃ symbol : Fin 256,
      recallPairWitness 0 0 j = recallKeyId symbol ∨ recallPairWitness 0 0 j = recallValueId symbol) ∧
    recallPairWitness 0 0 2 = recallValueId 0 := by
  refine ⟨by norm_num, by decide, by simp [recallPairWitness], ?_, by simp [recallPairWitness]⟩
  intro j hj
  by_cases h : j = 1
  · exact ⟨0, Or.inl (by simp [recallPairWitness, h])⟩
  · exact ⟨0, Or.inr (by simp [recallPairWitness, hj, h])⟩

end Transformer.GPTMini.Semantics
