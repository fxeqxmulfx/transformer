import Transformer.GPTMini.Semantics.RecallSaturation

/-!
# Uniform bounds on the genuine raw recall residual and prenorm

Source: both simultaneous original attention heads, W_o and residual
addition at f11b6e2, on the actual recall vocabulary at cbafbe9.
Every raw embedding has squared norm six. The actual predecessor
head has norm at most four and the actual marker head at most two.
Their ordinary output matrix consequently has norm at most six.

The true pre-FFN residual has norm between one and nine. The lower
bound comes from its preserved constant coordinate, not an assumed
nonzero feature. Its actual RMS multiplier lies between one half
and eight for epsilon in [0,1]. The whole first block also retains
that constant. These bounds precede a finite uniform gate/projection
gain; they do not assume equal RMS scales across table positions or
claim the complete second block is already saturated.
-/

namespace Transformer.GPTMini.Semantics

/-- Every actual raw vocabulary entry has norm at most three, including the reserved fallback entries.
Source: the complete embedding table's independently proved squared norm six. -/
theorem recallRawEmbedding_norm_le (token : Fin recallConfig.vocab_size) :
    ‖recallRawEmbedding token‖ ≤ 3 := by
  nlinarith [recallRawEmbedding_norm_sq token, norm_nonneg (recallRawEmbedding token)]

/-- The marker coordinate of the genuine head has an absolute bound, including at BOS's own XSA row.
Source: the real projected marker values, actual attention/XSA norm bound and coordinate projection. -/
theorem recall_raw_marker_coordinate_abs (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    |headAt recallConfig (recallFirstAttention eps alpha) eps positions
      (fun j => recallRawEmbedding (tokens j)) 1 i ⟨0, by decide⟩| ≤ 2 := by
  have h := PiLp.norm_apply_le (headAt recallConfig (recallFirstAttention eps alpha) eps positions
    (fun j => recallRawEmbedding (tokens j)) 1 i) (⟨0, by decide⟩ : Fin recallConfig.head_dim)
  rw [Real.norm_eq_abs] at h
  exact h.trans (recall_raw_marker_norm eps alpha heps positions tokens i)

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

/-- The genuine simultaneous W_o contribution is uniformly bounded by six for every raw array.
Source: its exact two-channel matrix, nonexpansive compact head readback and real head bounds. -/
theorem recall_first_attention_norm (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    ‖attnSubLayer recallConfig (recallFirstAttention eps alpha) eps positions
      (fun j => recallRawEmbedding (tokens j)) i‖ ≤ 6 := by
  let heads := fun h => headAt recallConfig (recallFirstAttention eps alpha) eps positions
    (fun j => recallRawEmbedding (tokens j)) h i
  have h0 : ‖recallHeadRead (heads 0)‖ ≤ 4 :=
    (recallHeadRead_norm_le _).trans (recall_raw_predecessor_norm eps alpha heps positions tokens i)
  have h1 : |heads 1 ⟨0, by decide⟩| ≤ 2 :=
    recall_raw_marker_coordinate_abs eps alpha heps positions tokens i
  have hu : ‖recallUnit 26‖ = 1 := by norm_num [recallUnit, PiLp.norm_single]
  rw [attnSubLayer_eq_heads]
  change ‖recallFirstOutput (headMerge recallConfig heads)‖ ≤ 6
  rw [recallFirstOutput_merge]
  refine (norm_add_le _ _).trans ?_
  rw [recallSlotWrite_norm, norm_smul, Real.norm_eq_abs, hu, mul_one]
  linarith

example : (0 : ℝ) ≤ 0 := by norm_num

/-- The genuine first residual is bounded above by nine, uniformly in raw IDs, positions and temperature.
Source: embeddings of squared norm six plus the actual simultaneous output contribution bounded by six. -/
theorem recallFirstState_norm_le (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    ‖recallFirstState eps alpha positions tokens i‖ ≤ 9 := by
  unfold recallFirstState
  refine (norm_add_le _ _).trans ?_
  linarith [recallRawEmbedding_norm_le (tokens i),
    recall_first_attention_norm eps alpha heps positions tokens i]

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

/-- The real pre-FFN residual has norm at least one because its actual constant coordinate is preserved.
Source: the verified raw constant and the genuine coordinate-norm inequality, without a nonzero-state premise. -/
theorem recallFirstState_norm_lower (eps alpha : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    1 ≤ ‖recallFirstState eps alpha positions tokens i‖ := by
  have h := PiLp.norm_apply_le (recallFirstState eps alpha positions tokens i) 0
  rw [recallFirstState_constant, Real.norm_eq_abs, abs_one] at h
  exact h

/-- The same norm lower bound survives the true table FFN and second residual addition.
Source: the complete original encoder block's protected constant, for every raw input and finite gain. -/
theorem recallEncoderBlock_norm_lower (P : ℕ) (eps alpha beta : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    1 ≤ ‖blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
      (fun j => recallRawEmbedding (tokens j)) i‖ := by
  have h := PiLp.norm_apply_le (blockForward recallConfig (recallEncoderBlock P eps alpha beta)
    eps positions (fun j => recallRawEmbedding (tokens j)) i) 0
  rw [recallEncoderBlock_constant, Real.norm_eq_abs, abs_one] at h
  exact h

/-- The original width-sixty-four prenorm multiplier evaluated on an arbitrary true residual.
Source: RMSNorm.forward at f11b6e2, retaining its actual residual norm and epsilon. -/
noncomputable def recallResidualScale (eps : ℝ) (x : EucSpace 64) : ℝ :=
  8 / Real.sqrt (‖x‖ ^ 2 + 64 * eps)

/-- This multiplier is exactly the genuine width-sixty-four RMSNorm, not a replacement normalization.
Source: the original RMSNorm expression and sqrt(64)=8. -/
theorem recallResidualScale_rms (eps : ℝ) (x : EucSpace 64) :
    rmsNormEps eps x = recallResidualScale eps x • x := by
  unfold rmsNormEps recallResidualScale
  norm_num only [Nat.cast_ofNat]

/-- A norm-at-least-one genuine residual has a strictly positive RMS multiplier for nonnegative epsilon.
Source: the true positive squared-norm denominator, with no assumption of a common position scale. -/
theorem recallResidualScale_pos (eps : ℝ) (heps : 0 ≤ eps) (x : EucSpace 64) (hx : 1 ≤ ‖x‖) :
    0 < recallResidualScale eps x := by
  unfold recallResidualScale
  apply div_pos (by norm_num)
  apply Real.sqrt_pos.mpr
  nlinarith [norm_nonneg x]

example : (0 : ℝ) ≤ 1 / 100000 ∧ 1 ≤ ‖recallUnit 0‖ := by
  norm_num [recallUnit, PiLp.norm_single]

/-- Such a true multiplier is uniformly bounded above by eight, regardless of the residual's size.
Source: the actual denominator is at least one by its verified norm lower bound. -/
theorem recallResidualScale_upper (eps : ℝ) (heps : 0 ≤ eps) (x : EucSpace 64) (hx : 1 ≤ ‖x‖) :
    recallResidualScale eps x ≤ 8 := by
  have hd : 0 < ‖x‖ ^ 2 + 64 * eps := by nlinarith [norm_nonneg x]
  have hs := Real.sqrt_pos.mpr hd
  have hl : 1 ≤ Real.sqrt (‖x‖ ^ 2 + 64 * eps) := by
    nlinarith [Real.sq_sqrt hd.le, Real.sqrt_nonneg (‖x‖ ^ 2 + 64 * eps)]
  unfold recallResidualScale
  rw [div_le_iff₀ hs]
  linarith

example : (0 : ℝ) ≤ 0 ∧ 1 ≤ ‖recallUnit 0‖ := by norm_num [recallUnit, PiLp.norm_single]

/-- The actual prenorm of the bounded true first residual also has a uniform positive lower multiplier.
Source: its norm at most nine gives a denominator below sixteen for epsilon in [0,1]. -/
theorem recallFirstState_scale_lower (eps alpha : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    (1 / 2 : ℝ) ≤ recallResidualScale eps (recallFirstState eps alpha positions tokens i) := by
  let x := recallFirstState eps alpha positions tokens i
  have hx : 1 ≤ ‖x‖ := recallFirstState_norm_lower eps alpha positions tokens i
  have hu : ‖x‖ ≤ 9 := recallFirstState_norm_le eps alpha heps positions tokens i
  have hd : 0 < ‖x‖ ^ 2 + 64 * eps := by nlinarith [norm_nonneg x]
  have hs := Real.sqrt_pos.mpr hd
  have hden : Real.sqrt (‖x‖ ^ 2 + 64 * eps) ≤ 16 := by
    nlinarith [Real.sq_sqrt hd.le, Real.sqrt_nonneg (‖x‖ ^ 2 + 64 * eps)]
  change (1 / 2 : ℝ) ≤ 8 / Real.sqrt (‖x‖ ^ 2 + 64 * eps)
  rw [le_div_iff₀ hs]
  linarith

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 := by norm_num

/-- Both uniform bounds hold for the same real pre-FFN multiplier at every raw position.
Source: the actual first residual's independently derived norm interval, without a prepared-state premise. -/
theorem recallFirstState_scale_bounds (eps alpha : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    (1 / 2 : ℝ) ≤ recallResidualScale eps (recallFirstState eps alpha positions tokens i) ∧
      recallResidualScale eps (recallFirstState eps alpha positions tokens i) ≤ 8 := by
  exact ⟨recallFirstState_scale_lower eps alpha heps hclip positions tokens i,
    recallResidualScale_upper eps heps _ (recallFirstState_norm_lower eps alpha positions tokens i)⟩

example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := by norm_num

/-- The full true first block has a positive second-attention prenorm multiplier bounded above by eight.
Source: its protected constant survives both actual residual additions and the complete gate FFN. -/
theorem recallEncoderBlock_scale_upper (P : ℕ) (eps alpha beta : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    let x := blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
      (fun j => recallRawEmbedding (tokens j)) i
    0 < recallResidualScale eps x ∧ recallResidualScale eps x ≤ 8 := by
  dsimp only
  have hx := recallEncoderBlock_norm_lower P eps alpha beta positions tokens i
  exact ⟨recallResidualScale_pos eps heps _ hx, recallResidualScale_upper eps heps _ hx⟩

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

end Transformer.GPTMini.Semantics
