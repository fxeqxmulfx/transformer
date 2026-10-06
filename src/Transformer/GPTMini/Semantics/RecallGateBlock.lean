import Transformer.GPTMini.Semantics.RecallGateFFN

/-!
# The simultaneous original first block with its table gate

Source: original Block.forward at f11b6e2, using the given shared
raw embedding/QKV/head merge/W_o and sixteen-unit ReLU2 matrices.
The whole block fits the original 64-wide stream and 256-unit FFN.
The copied key, raw key/value codes, causal marker and type channels
coexist; gated output goes only to the initially empty slot 27..34.

The local positive/negative gate conditions below are precisely those
already derived from raw IDs and the table boundary. Their use here
does not redefine blockForward or supply an oracle. Genuine residual
addition is included, and the actual position-dependent RMS square
is retained in the amplitude. Complete validated-prefix coupling and
robust second-block retrieval/readout still remain to be discharged.
-/

namespace Transformer.GPTMini.Semantics

/-- A single original block combines both actual attention heads and the ordinary table-gating FFN.
Source: unchanged BlockParams/Block.forward, with no widening or alternate nonlinear operator. -/
noncomputable def recallEncoderBlock (P : ℕ) (eps alpha beta : ℝ) : BlockParams recallConfig where
  attn := recallFirstAttention eps alpha
  ffn := recallGateParameters P beta

/-- The amplitude is the actual homogeneous FFN coefficient, including its true prenorm multiplier.
Source: the evaluated original ReLU2 gate, not an assumed fixed scale for table records. -/
noncomputable def recallGateAmplitude (P : ℕ) (beta eps : ℝ) (x : EucSpace 64) : ℝ :=
  beta * (8 / Real.sqrt (‖x‖ ^ 2 + 64 * eps)) ^ 2 * recallGateForm P x

/-- The real whole block leaves every nongated coordinate equal to its actual attention residual.
Source: genuine second residual addition and complete support of the evaluated original FFN. -/
theorem recallEncoderBlock_protected (P : ℕ) (eps alpha beta : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) (c : Fin 64)
    (hc : c.val < 27 ∨ 35 ≤ c.val) :
    blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
        (fun j => recallRawEmbedding (tokens j)) i c = recallFirstState eps alpha positions tokens i c := by
  dsimp only [blockForward, ffnSubLayer, recallEncoderBlock, recallGateParameters]
  change (recallFirstState eps alpha positions tokens i +
    relu2FFN (recallGateIn P) (recallGateOut P beta)
      (rmsNormEps eps (recallFirstState eps alpha positions tokens i))) c = _
  rw [PiLp.add_apply, recallGateFFN_protected P beta _ c hc, add_zero]

example : (36 : Fin 64).val < 27 ∨ 35 ≤ (36 : Fin 64).val := by decide

/-- The simultaneous full original block preserves every raw code, constant and type coordinate.
Source: both real residual additions and the disjoint actual output supports of attention and FFN. -/
theorem recallEncoderBlock_raw_protected (P : ℕ) (eps alpha beta : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) (c : Fin 64)
    (hc : c.val < 18 ∨ 35 ≤ c.val) :
    blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
        (fun j => recallRawEmbedding (tokens j)) i c = recallRawEmbedding (tokens i) c := by
  rw [recallEncoderBlock_protected P eps alpha beta positions tokens i c (by omega)]
  exact recallFirstState_protected eps alpha positions tokens i c (by omega)

example : (0 : Fin 64).val < 18 ∨ 35 ≤ (0 : Fin 64).val := by decide

/-- The complete actual first block retains a constant one at every raw position.
Source: both true residual additions preserve the raw embedding's independently verified constant axis. -/
theorem recallEncoderBlock_constant (P : ℕ) (eps alpha beta : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
      (fun j => recallRawEmbedding (tokens j)) i 0 = 1 := by
  rw [recallEncoderBlock_raw_protected P eps alpha beta positions tokens i 0 (by decide)]
  exact recallRawEmbedding_constant (tokens i)

/-- The genuine marker coordinate computed by attention also survives the original FFN residual.
Source: the complete table-gate output matrix is supported only on coordinates 27..34. -/
theorem recallEncoderBlock_marker (P : ℕ) (eps alpha beta : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
      (fun j => recallRawEmbedding (tokens j)) i 26 = recallFirstState eps alpha positions tokens i 26 := by
  exact recallEncoderBlock_protected P eps alpha beta positions tokens i 26 (by decide)

/-- Every raw-token first attention residual has an empty gated-key destination before the genuine FFN.
Source: initially empty raw coordinates 27..34 and the predecessor/marker output support 18..26. -/
theorem recallFirstState_gate_slot_zero (eps alpha : ℝ) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    recallSlotRead 27 (recallFirstState eps alpha positions tokens i) = 0 := by
  ext c
  rw [recallSlotRead_at, recallFirstState_protected eps alpha positions tokens i _ (Or.inr (by
    change 27 ≤ 27 + c.val
    omega))]
  rw [recallRawEmbedding_fresh (tokens i) _ (by
    change 18 ≤ 27 + c.val ∧ 27 + c.val < 35
    have hc := c.isLt
    omega), PiLp.zero_apply]

/-- A separated negative gate makes the entire actual block's FFN contribution zero on raw inputs.
Source: real pre-FFN coordinate bounds are derived by the actual raw head, not supplied as an encoder premise. -/
theorem recallEncoderBlock_off (P : ℕ) (eps alpha beta : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T)
    (hgate : recallGateForm P (recallFirstState eps alpha positions tokens i) ≤ -recallTableMargin P) :
    blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
        (fun j => recallRawEmbedding (tokens j)) i = recallFirstState eps alpha positions tokens i := by
  dsimp only [blockForward, ffnSubLayer, recallEncoderBlock, recallGateParameters]
  change recallFirstState eps alpha positions tokens i +
    relu2FFN (recallGateIn P) (recallGateOut P beta)
      (rmsNormEps eps (recallFirstState eps alpha positions tokens i)) = _
  rw [recallGateFFN_off P beta eps _ hgate
    (recallFirstState_copy_coordinate eps alpha heps positions tokens i), add_zero]

example : (0 : ℝ) ≤ 0 ∧
    recallGateForm 0 (recallFirstState 0 0 (fun _ : Fin 1 => 0) (fun _ => recallKeyId 0) 0) ≤
      -recallTableMargin 0 := by
  exact ⟨by norm_num, recallGateForm_key 0 0 0 (by norm_num) _ _ 0 0 rfl⟩

/-- On a separated positive table value, the actual full first block stores precisely the scaled compact copy.
Source: true residual addition into an initially empty destination and the real prenorm/FFN formula. -/
theorem recallEncoderBlock_on (P : ℕ) (eps alpha beta : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T)
    (hgate : recallTableMargin P ≤ recallGateForm P (recallFirstState eps alpha positions tokens i)) :
    recallSlotRead 27 (blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
        (fun j => recallRawEmbedding (tokens j)) i) =
      recallGateAmplitude P beta eps (recallFirstState eps alpha positions tokens i) •
        recallSlotRead 18 (recallFirstState eps alpha positions tokens i) := by
  have hz := recallFirstState_gate_slot_zero eps alpha positions tokens i
  have hon := recallGateFFN_read_on P beta eps (recallFirstState eps alpha positions tokens i) hgate
    (recallFirstState_copy_coordinate eps alpha heps positions tokens i)
  dsimp only [recallFirstState] at hz hon
  dsimp only [blockForward, ffnSubLayer, recallEncoderBlock, recallGateParameters]
  rw [map_add, hz, zero_add, hon]
  unfold recallGateAmplitude recallFirstState
  rfl

example : (0 : ℝ) ≤ 0 ∧
    recallTableMargin 1 ≤ recallGateForm 1 (recallFirstState 0 0 (fun j : Fin 2 => (j.val : ℝ))
      (fun j => if j = 0 then (⟨1, by decide⟩ : Fin recallConfig.vocab_size) else recallValueId 0) 1) := by
  refine ⟨by norm_num, ?_⟩
  apply recallGateForm_inside 1 0 0 (by norm_num) _ _ 0 1 (by decide) (by decide) (by simp) (value := 0)
  · intro j hj
    exact ⟨0, Or.inr (by simp only [ite_eq_right hj])⟩
  · simp
  · decide

/-- The stated actual gate amplitude is positive whenever its finite gain and genuine normalization are positive.
Source: real RMS denominator and the derived strictly positive table margin. -/
theorem recallGateAmplitude_pos (P : ℕ) (beta eps : ℝ) (hbeta : 0 < beta) (heps : 0 < eps)
    (x : EucSpace 64) (hgate : recallTableMargin P ≤ recallGateForm P x) :
    0 < recallGateAmplitude P beta eps x := by
  have hg : 0 < recallGateForm P x := lt_of_lt_of_le (recallTableMargin_pos P) hgate
  have hd : 0 < ‖x‖ ^ 2 + 64 * eps := by positivity
  have hs : 0 < Real.sqrt (‖x‖ ^ 2 + 64 * eps) := Real.sqrt_pos.mpr hd
  unfold recallGateAmplitude
  positivity

example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000 ∧
    recallTableMargin 0 ≤ recallGateForm 0 (recallUnit 26) := by
  refine ⟨by norm_num, by norm_num, ?_⟩
  norm_num [recallGateForm_apply, recallTableMargin, recallTableThreshold, recallUnit, PiLp.single_apply]

end Transformer.GPTMini.Semantics
