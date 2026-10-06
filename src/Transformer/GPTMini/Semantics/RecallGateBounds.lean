import Transformer.GPTMini.Semantics.RecallGateMatrices
import Transformer.GPTMini.Semantics.RecallRawCopy

/-!
# Bound the real first residual before table gating

Source: the original attention/FFN residual ordering at f11b6e2 and
the actual raw MQAR vocabulary at cbafbe9. All coordinates below
come from embedding plus the simultaneous predecessor and BOS heads.
No paired-key array, external length or desired gate output is an
input to the model. The copied coordinates have a uniform bound four
from the true raw value bound and XSA, irrespective of temperature.

The marker coordinate has a coarse bound two for every vocabulary
array. Together with the protected type flag this already excludes
all nonvalues, including the initial BOS where reciprocal-prefix
mass need not describe XSA. Exact positive/negative table decisions
for later value positions use the separately derived raw BOS formula.
-/

namespace Transformer.GPTMini.Semantics

/-- This is precisely the original first attention residual, before its FFN.
Source: blockForward at f11b6e2 with the already realized shared raw attention matrices. -/
noncomputable def recallFirstState (eps alpha : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) : EucSpace 64 :=
  recallRawEmbedding (tokens i) + attnSubLayer recallConfig (recallFirstAttention eps alpha)
    eps positions (fun j => recallRawEmbedding (tokens j)) i

/-- The actual predecessor head has norm at most four on every checked raw-token array.
Source: every raw projected value has norm at most two; the original XSA can at most double it. -/
theorem recall_raw_predecessor_norm (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    ‖headAt recallConfig (recallFirstAttention eps alpha) eps positions
      (fun j => recallRawEmbedding (tokens j)) 0 i‖ ≤ 4 := by
  rw [recall_predecessor_head eps alpha heps]
  have h := attentionHead_norm_le recallConfig alpha eps heps
    (fun _ => adjacentQuery) (fun _ => adjacentDirection) _ positions i 2
    (fun j => recall_raw_head_value_bound (tokens j))
  norm_num only at h
  exact h

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

/-- The real compact predecessor-copy slot also has norm at most four.
Source: the genuine W_o and the proved nonexpansive coordinate extraction. -/
theorem recallFirstState_copy_norm (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    ‖recallSlotRead 18 (recallFirstState eps alpha positions tokens i)‖ ≤ 4 := by
  rw [recallFirstState, recall_first_residual_copy_slot]
  exact (recallHeadRead_norm_le _).trans (recall_raw_predecessor_norm eps alpha heps positions tokens i)

example : (0 : ℝ) ≤ 0 := by norm_num

/-- Every actual copied coordinate has the scalar bound needed by the finite quadratic gate.
Source: the true slot norm and the coordinate projection inequality. -/
theorem recallFirstState_copy_coordinate (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) (c : Fin 8) :
    |recallFirstState eps alpha positions tokens i (recallSlotIndex 18 c)| ≤ 4 := by
  rw [← recallSlotRead_at]
  have h := PiLp.norm_apply_le (recallSlotRead 18 (recallFirstState eps alpha positions tokens i)) c
  rw [Real.norm_eq_abs] at h
  exact h.trans (recallFirstState_copy_norm eps alpha heps positions tokens i)

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

/-- The actual attention residual retains every protected input coordinate exactly.
Source: the simultaneous copy/marker W_o is supported only on coordinates 18..26. -/
theorem recallFirstState_protected (eps alpha : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) (c : Fin 64)
    (hc : c.val < 18 ∨ 27 ≤ c.val) :
    recallFirstState eps alpha positions tokens i c = recallRawEmbedding (tokens i) c := by
  unfold recallFirstState
  exact recall_first_attn_protected eps alpha positions (fun j => recallRawEmbedding (tokens j)) i c hc

example : (36 : Fin 64).val < 18 ∨ 27 ≤ (36 : Fin 64).val := by decide

/-- The gate's constant input is derived from every raw vocabulary entry.
Source: actual protected raw constant coordinate zero and the true first residual. -/
theorem recallFirstState_constant (eps alpha : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    recallFirstState eps alpha positions tokens i 0 = 1 := by
  rw [recallFirstState_protected eps alpha positions tokens i 0 (by decide)]
  exact recallRawEmbedding_constant (tokens i)

/-- Actual raw value IDs retain their own value-type flag in the gate input.
Source: the disjoint raw vocabulary and protected type axis 36. -/
theorem recallFirstState_value_flag (eps alpha : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) (symbol : Fin 256)
    (htoken : tokens i = recallValueId symbol) : recallFirstState eps alpha positions tokens i 36 = 1 := by
  rw [recallFirstState_protected eps alpha positions tokens i 36 (by decide), htoken, recallRawEmbedding_value]
  exact (recallValueEmbedding_flags symbol).2.2.1

example : (fun _ : Fin 1 => recallValueId 0) 0 = recallValueId 0 := by rfl

/-- A genuine raw query/key has zero value-type flag, even if its copied slot is nonzero.
Source: actual key embedding and the protected type axis. -/
theorem recallFirstState_key_flag (eps alpha : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) (symbol : Fin 256)
    (htoken : tokens i = recallKeyId symbol) : recallFirstState eps alpha positions tokens i 36 = 0 := by
  rw [recallFirstState_protected eps alpha positions tokens i 36 (by decide), htoken, recallRawEmbedding_key]
  exact (recallKeyEmbedding_flags symbol).2.2.1

example : (fun _ : Fin 1 => recallKeyId 0) 0 = recallKeyId 0 := by rfl

/-- Initial BOS has zero value-type flag as well.
Source: the original reserved/BOS entry, separately from any later-position marker formula. -/
theorem recallFirstState_bos_flag (eps alpha : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T)
    (htoken : tokens i = (⟨1, by decide⟩ : Fin recallConfig.vocab_size)) :
    recallFirstState eps alpha positions tokens i 36 = 0 := by
  rw [recallFirstState_protected eps alpha positions tokens i 36 (by decide), htoken, recallRawEmbedding_bos]
  exact recallReservedEmbedding_flags.2.2.1

example : (fun _ : Fin 1 => (⟨1, by decide⟩ : Fin recallConfig.vocab_size)) 0 = ⟨1, by decide⟩ := by rfl

/-- Every actual raw marker value has norm at most one.
Source: the complete three-branch raw vocabulary, with flag one only for the reserved group. -/
theorem recall_raw_marker_value_norm (token : Fin recallConfig.vocab_size) :
    ‖recallRawEmbedding token 37 • recallMarkerDirection‖ ≤ 1 := by
  unfold recallRawEmbedding
  split_ifs
  · rw [(recallKeyEmbedding_flags _).2.2.2, zero_smul, norm_zero]; norm_num
  · rw [(recallValueEmbedding_flags _).2.2.2, zero_smul, norm_zero]; norm_num
  · rw [recallReservedEmbedding_flags.2.2.2, one_smul, recallMarkerDirection_norm]

/-- The genuine marker head has norm at most two on any raw array, including the BOS position.
Source: the original uniform head/XSA and the derived complete raw marker-value bound. -/
theorem recall_raw_marker_norm (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    ‖headAt recallConfig (recallFirstAttention eps alpha) eps positions
      (fun j => recallRawEmbedding (tokens j)) 1 i‖ ≤ 2 := by
  rw [recall_marker_head eps alpha heps]
  change ‖attentionHead recallConfig 0 eps (fun _ => 0) (fun _ => 0)
    (fun j => recallRawEmbedding (tokens j) 37 • recallMarkerDirection) positions i‖ ≤ 2
  simpa only [mul_one] using attentionHead_norm_le recallConfig 0 eps heps
    (fun _ => 0) (fun _ => 0) _ positions i 1 (fun j => recall_raw_marker_value_norm (tokens j))

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

/-- The true marker coordinate is at most two on every raw input, including initial BOS.
Source: the genuine W_o coordinate and the actual marker head's derived norm bound. -/
theorem recallFirstState_marker_le (eps alpha : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (positions : Fin T → ℝ) (tokens : Fin T → Fin recallConfig.vocab_size) (i : Fin T) :
    recallFirstState eps alpha positions tokens i 26 ≤ 2 := by
  rw [recallFirstState, PiLp.add_apply, recallRawEmbedding_fresh (tokens i) 26 (by decide),
    zero_add, attnSubLayer_eq_heads]
  change recallFirstOutput (headMerge recallConfig _) 26 ≤ 2
  rw [recallFirstOutput_marker]
  have hc := PiLp.norm_apply_le
    (headAt recallConfig (recallFirstAttention eps alpha) eps positions
      (fun j => recallRawEmbedding (tokens j)) 1 i) (⟨0, by decide⟩ : Fin recallConfig.head_dim)
  rw [Real.norm_eq_abs] at hc
  exact (le_abs_self _).trans (hc.trans (recall_raw_marker_norm eps alpha heps positions tokens i))

example : (0 : ℝ) ≤ 0 := by norm_num

end Transformer.GPTMini.Semantics
