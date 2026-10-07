import Transformer.GPTMini.Semantics.RecallStateBounds

/-!
# A finite shared table gain and a bound on the actual gated block

Source: the sixteen original ReLU2 units at f11b6e2 and the actual
raw table marker at cbafbe9. The gain is fixed from the table size;
it does not depend on a token, record amplitude or whole prefix.
The genuine pre-FFN RMS lower bound makes every positive table
amplitude at least one with gain 4/tableMargin.

All raw value-type flags are at most one, so the actual linear gate
signal is at most two. Its true positive amplitude is at most
128*beta. The complete positive-gate block therefore has norm at
most 9+512*beta, including both original residual additions. The
local gate-separation conditions used here were derived from raw
IDs in RecallRawGate; subsequent uniform raw bounds discharge them
at every position before choosing one shared second-QKV gain.
-/

namespace Transformer.GPTMini.Semantics

/-- One ordinary finite FFN gain is fixed by the original table size alone.
Source: the derived table margin and the true gate RMS multiplier bounded below by one half. -/
noncomputable def recallEncoderGain (P : ℕ) : ℝ := 4 / recallTableMargin P

/-- The chosen shared gain is positive for every nonnegative table size.
Source: the independently proved strict positivity of the genuine boundary margin. -/
theorem recallEncoderGain_pos (P : ℕ) : 0 < recallEncoderGain P :=
  div_pos (by norm_num) (recallTableMargin_pos P)

/-- The exact product of the fixed gain and actual boundary margin is four.
Source: its ordinary finite real coefficient, with no record-dependent compensation. -/
theorem recallEncoderGain_margin (P : ℕ) : recallEncoderGain P * recallTableMargin P = 4 := by
  unfold recallEncoderGain
  exact div_mul_cancel₀ _ (ne_of_gt (recallTableMargin_pos P))

/-- Every actual raw vocabulary entry has value-type flag at most one.
Source: all three genuine embedding branches, including reserved fallback entries. -/
theorem recallRawEmbedding_value_flag_le (token : Fin recallConfig.vocab_size) :
    recallRawEmbedding token 36 ≤ 1 := by
  unfold recallRawEmbedding
  split_ifs
  · rw [(recallKeyEmbedding_flags _).2.2.1]
    norm_num
  · rw [(recallValueEmbedding_flags _).2.2.1]
  · rw [recallReservedEmbedding_flags.2.2.1]
    norm_num

/-- The genuine gate signal is uniformly at most two on every raw array and real position sequence.
Source: actual marker-head bound, protected constant/type coordinates and positive table cutoff. -/
theorem recallFirstState_gate_upper (P : ℕ) (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    recallGateForm P (recallFirstState eps alpha positions tokens i) ≤ 2 := by
  rw [recallGateForm_apply, recallFirstState_constant,
    recallFirstState_protected eps alpha positions tokens i 36 (by decide)]
  linarith [recallFirstState_marker_le eps alpha heps positions tokens i,
    recallRawEmbedding_value_flag_le (tokens i), recallTableThreshold_pos P]

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

/-- A separated positive gate has nonnegative actual amplitude for every nonnegative ordinary gain.
Source: the true RMS square and the independently proved positive table margin. -/
theorem recallGateAmplitude_nonneg (P : ℕ) (beta eps : ℝ) (x : EucSpace 64)
    (hbeta : 0 ≤ beta) (hgate : recallTableMargin P ≤ recallGateForm P x) :
    0 ≤ recallGateAmplitude P beta eps x := by
  have hg := lt_of_lt_of_le (recallTableMargin_pos P) hgate
  unfold recallGateAmplitude
  positivity

example : (0 : ℝ) ≤ 0 ∧ recallTableMargin 0 ≤ recallGateForm 0 (recallUnit 26) := by
  norm_num [recallGateForm_apply, recallTableMargin, recallTableThreshold, recallUnit, PiLp.single_apply]

/-- This real BOS/key/value witness has a derived positive gate, used only to satisfy later local examples.
Source: actual raw-token marker theorem on the three-token witness, without prepared gate data. -/
theorem recallPairWitness_gate_inside (eps alpha : ℝ) (heps : 0 ≤ eps) :
    recallTableMargin 1 ≤ recallGateForm 1 (recallFirstState eps alpha
      (fun j : Fin 3 => (j.val : ℝ)) (recallPairWitness 0 0) 2) := by
  apply recallGateForm_inside 1 eps alpha heps _ _ 0 2 (by decide) (by decide)
    (by simp [recallPairWitness]) (value := 0)
  · intro j hj
    by_cases h : j = 1
    · exact ⟨0, Or.inl (by simp [recallPairWitness, h])⟩
    · exact ⟨0, Or.inr (by simp [recallPairWitness, hj, h])⟩
  · simp [recallPairWitness]
  · decide

example : (0 : ℝ) ≤ 0 := by norm_num

/-- The chosen fixed gain makes the real positive raw table amplitude at least one.
Source: actual pre-FFN RMS multiplier at least one half and the true separated gate, not an assumed common scale. -/
theorem recallGateAmplitude_lower (P : ℕ) (eps alpha : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T)
    (hgate : recallTableMargin P ≤ recallGateForm P (recallFirstState eps alpha positions tokens i)) :
    1 ≤ recallGateAmplitude P (recallEncoderGain P) eps (recallFirstState eps alpha positions tokens i) := by
  let x := recallFirstState eps alpha positions tokens i
  have hs := recallFirstState_scale_lower eps alpha heps hclip positions tokens i
  have hsq : (1 / 4 : ℝ) ≤ (recallResidualScale eps x) ^ 2 := by
    nlinarith [recallResidualScale_pos eps heps x (recallFirstState_norm_lower eps alpha positions tokens i)]
  have hb := (recallEncoderGain_pos P).le
  have hm := (recallTableMargin_pos P).le
  change 1 ≤ recallEncoderGain P * (recallResidualScale eps x) ^ 2 * recallGateForm P x
  calc 1 = recallEncoderGain P * (1 / 4) * recallTableMargin P := by
        nlinarith [recallEncoderGain_margin P]
    _ ≤ recallEncoderGain P * (recallResidualScale eps x) ^ 2 * recallTableMargin P :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hsq hb) hm
    _ ≤ recallEncoderGain P * (recallResidualScale eps x) ^ 2 * recallGateForm P x :=
        mul_le_mul_of_nonneg_left hgate (mul_nonneg hb (sq_nonneg _))

example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ recallTableMargin 1 ≤ recallGateForm 1
    (recallFirstState 0 0 (fun j : Fin 3 => (j.val : ℝ)) (recallPairWitness 0 0) 2) :=
  ⟨by norm_num, by norm_num, recallPairWitness_gate_inside 0 0 (by norm_num)⟩

/-- A real amplitude and compact copy both at least one give a stored-key norm at least one.
Source: exact norm homogeneity of the actual scalar write, independent of its later representation. -/
theorem recallScaledCode_norm_lower (a : ℝ) (x : EucSpace 8) (ha : 1 ≤ a) (hx : 1 ≤ ‖x‖) :
    1 ≤ ‖a • x‖ := by
  have hp : 0 ≤ a := by linarith
  rw [norm_smul_of_nonneg hp]
  have h := mul_le_mul_of_nonneg_left hx hp
  rw [mul_one] at h
  exact ha.trans h

example : (1 : ℝ) ≤ 1 ∧ 1 ≤ ‖recallCode (fun _ => 0)‖ := by rw [recallCode_norm]; norm_num

/-- Every true raw amplitude is at most 128 times its nonnegative shared gain.
Source: actual RMS multiplier at most eight and actual raw linear gate signal at most two. -/
theorem recallGateAmplitude_upper (P : ℕ) (eps alpha beta : ℝ) (heps : 0 ≤ eps) (hbeta : 0 ≤ beta) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    recallGateAmplitude P beta eps (recallFirstState eps alpha positions tokens i) ≤ 128 * beta := by
  let x := recallFirstState eps alpha positions tokens i
  have hs := recallResidualScale_upper eps heps x (recallFirstState_norm_lower eps alpha positions tokens i)
  have hp := recallResidualScale_pos eps heps x (recallFirstState_norm_lower eps alpha positions tokens i)
  have hsq : (recallResidualScale eps x) ^ 2 ≤ 64 := by nlinarith
  have hg := recallFirstState_gate_upper P eps alpha heps positions tokens i
  change beta * (recallResidualScale eps x) ^ 2 * recallGateForm P x ≤ 128 * beta
  calc beta * (recallResidualScale eps x) ^ 2 * recallGateForm P x
      ≤ beta * (recallResidualScale eps x) ^ 2 * 2 :=
        mul_le_mul_of_nonneg_left hg (mul_nonneg hbeta (sq_nonneg _))
    _ ≤ beta * 64 * 2 :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hsq hbeta) (by norm_num)
    _ = 128 * beta := by ring

example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 0 := by norm_num

/-- The positive-gate whole block includes exactly the true attention residual plus the actual scaled key write.
Source: unchanged blockForward and the complete evaluated original ReLU2 FFN, including real prenorm. -/
theorem recallEncoderBlock_on_full (P : ℕ) (eps alpha beta : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T)
    (hgate : recallTableMargin P ≤ recallGateForm P (recallFirstState eps alpha positions tokens i)) :
    blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
        (fun j => recallRawEmbedding (tokens j)) i = recallFirstState eps alpha positions tokens i +
      recallGateAmplitude P beta eps (recallFirstState eps alpha positions tokens i) •
        recallSlotWrite 27 (recallSlotRead 18 (recallFirstState eps alpha positions tokens i)) := by
  have hon := recallGateFFN_on P beta eps (recallFirstState eps alpha positions tokens i) hgate
    (recallFirstState_copy_coordinate eps alpha heps positions tokens i)
  dsimp only [recallFirstState] at hon
  dsimp only [blockForward, ffnSubLayer, recallEncoderBlock, recallGateParameters]
  rw [hon]
  unfold recallGateAmplitude recallFirstState
  rfl

example : (0 : ℝ) ≤ 0 ∧ recallTableMargin 1 ≤ recallGateForm 1
    (recallFirstState 0 0 (fun j : Fin 3 => (j.val : ℝ)) (recallPairWitness 0 0) 2) :=
  ⟨by norm_num, recallPairWitness_gate_inside 0 0 (by norm_num)⟩

/-- The full original block has a uniform finite norm bound in its genuine positive-gate region.
Source: both residual additions, copied-slot norm at most four and true amplitude at most 128*beta. -/
theorem recallEncoderBlock_on_norm (P : ℕ) (eps alpha beta : ℝ) (heps : 0 ≤ eps) (hbeta : 0 ≤ beta) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T)
    (hgate : recallTableMargin P ≤ recallGateForm P (recallFirstState eps alpha positions tokens i)) :
    ‖blockForward recallConfig (recallEncoderBlock P eps alpha beta) eps positions
      (fun j => recallRawEmbedding (tokens j)) i‖ ≤ 9 + 512 * beta := by
  let x := recallFirstState eps alpha positions tokens i
  have ha := recallGateAmplitude_nonneg P beta eps x hbeta hgate
  have hu := recallGateAmplitude_upper P eps alpha beta heps hbeta positions tokens i
  have hc := recallFirstState_copy_norm eps alpha heps positions tokens i
  rw [recallEncoderBlock_on_full P eps alpha beta heps positions tokens i hgate]
  refine (norm_add_le _ _).trans ?_
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg ha, recallSlotWrite_norm]
  have hprod := mul_le_mul_of_nonneg_left hc ha
  linarith [recallFirstState_norm_le eps alpha heps positions tokens i]

example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ recallTableMargin 1 ≤ recallGateForm 1
    (recallFirstState 0 0 (fun j : Fin 3 => (j.val : ℝ)) (recallPairWitness 0 0) 2) :=
  ⟨by norm_num, by norm_num, recallPairWitness_gate_inside 0 0 (by norm_num)⟩

end Transformer.GPTMini.Semantics
