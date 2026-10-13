import Transformer.GPTMini.Semantics.RecallRawGate
import Transformer.GPTMini.Semantics.RecallLatestGap

/-!
# A finite shared predecessor temperature for any positive copy tolerance

Source: original finite softmax/QKNorm/RoPE/XSA at f11b6e2,
the actual raw recall head and the true context cap 64 at cbafbe9.
The previously derived error bound is not left as an assumed small
copy error: an explicit finite log-temperature now enforces any
positive requested tolerance uniformly over all raw adjacent writes.

The conservative choice uses exp(x) >= x, so it is a real-arithmetic
capacity bound, not a claim of numerically optimal weights, floating-
point equivalence or successful AdamW training. Its temperature
depends only on the desired global tolerance, not on a token prefix.
The genuine pre-FFN residual's compact copy inherits that accuracy.
At tolerance at most one, every copied table key has nonzero norm;
the actual positive table gate can therefore preserve its direction.
-/

namespace Transformer.GPTMini.Semantics

/-- One fixed finite original log-temperature controls all raw prefixes at context 64.
Source: actual predecessor cosine margin, raw value diameter four and at most 63 competitors. -/
noncomputable def recallCopyTemperature (eta : ℝ) : ℝ :=
  Real.log (252 / ((1 - Real.cos adjacentFrequency) * eta))

/-- The chosen actual temperature has exactly the finite positional gap needed by the uniform tolerance bound.
Source: genuine exp(log_alpha), not a hard attention limit or a supplied score-gap input. -/
theorem recallCopyTemperature_gap (eta : ℝ) (heta : 0 < eta) :
    adjacentGap (recallCopyTemperature eta) = 252 / eta := by
  have hd := adjacent_cosine_gap_pos
  unfold adjacentGap recallCopyTemperature
  rw [Real.exp_log (by positivity)]
  field_simp

example : (0 : ℝ) < 1 / 1000000 := by norm_num

/-- The actual finite softmax tail bound is at most the requested tolerance for the entire original context.
Source: derived real gap and exp(x) >= x+1; no convergence or infinite-temperature assumption. -/
theorem recallCopyTemperature_error (eta : ℝ) (heta : 0 < eta) (T : ℕ) (hT : T ≤ 64) :
    ((T - 1 : ℕ) : ℝ) * Real.exp (-adjacentGap (recallCopyTemperature eta)) * 4 ≤ eta := by
  rw [recallCopyTemperature_gap eta heta, Real.exp_neg]
  have hcount : ((T - 1 : ℕ) : ℝ) ≤ 63 := by exact_mod_cast (by omega : T - 1 ≤ 63)
  have he : 252 / eta ≤ Real.exp (252 / eta) := by linarith [Real.add_one_le_exp (252 / eta)]
  have hbudget : 252 ≤ eta * Real.exp (252 / eta) := by
    simpa only [mul_comm] using (div_le_iff₀ heta).mp he
  have hinv : 252 * (Real.exp (252 / eta))⁻¹ ≤ eta := by
    rw [← div_eq_mul_inv, div_le_iff₀ (Real.exp_pos _)]
    exact hbudget
  have hcount' := mul_le_mul_of_nonneg_right hcount (by positivity : 0 ≤ (Real.exp (252 / eta))⁻¹)
  nlinarith

example : (0 : ℝ) < 1 / 1000000 ∧ (64 : ℕ) ≤ 64 := by norm_num

/-- The complete true sixteen-coordinate predecessor head has the same requested accuracy.
Source: the actual raw whole-head copy estimate and the proved finite shared-temperature bound. -/
theorem recall_raw_head_accuracy (eta eps : ℝ) (heta : 0 < eta) (heps : 0 ≤ eps) (hclip : eps ≤ 1)
    {T : ℕ} (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size) (i selected : Fin T)
    (hprev : selected.val + 1 = i.val) (key value : Fin 256)
    (hkey : tokens selected = recallKeyId key) (hvalue : tokens i = recallValueId value) :
    ‖headAt recallConfig (recallFirstAttention eps (recallCopyTemperature eta)) eps
        (fun r => (r.val : ℝ)) (fun j => recallRawEmbedding (tokens j)) 0 i -
      (recallHeadValue (recallSymbolCode key) : EucSpace recallConfig.head_dim)‖ ≤ eta := by
  exact (recall_raw_predecessor_copy eps (recallCopyTemperature eta) heps hclip (by omega)
    tokens i selected hprev key value hkey hvalue).trans (recallCopyTemperature_error eta heta T hT)

example : (0 : ℝ) < 1 / 1000000 ∧ (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    (2 : ℕ) ≤ 64 ∧ (0 : Fin 2).val + 1 = (1 : Fin 2).val ∧
    (fun j : Fin 2 => if j = 0 then recallKeyId 0 else recallValueId 0) 0 = recallKeyId 0 ∧
    (fun j : Fin 2 => if j = 0 then recallKeyId 0 else recallValueId 0) 1 = recallValueId 0 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by decide, by decide, by simp, by simp⟩

/-- The genuine compact original head achieves the requested positive accuracy on every actual adjacent raw pair.
Source: all raw QKV/value/gap premises are derived by the original head, with one finite shared temperature. -/
theorem recall_raw_copy_accuracy (eta eps : ℝ) (heta : 0 < eta) (heps : 0 ≤ eps) (hclip : eps ≤ 1)
    {T : ℕ} (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size) (i selected : Fin T)
    (hprev : selected.val + 1 = i.val) (key value : Fin 256)
    (hkey : tokens selected = recallKeyId key) (hvalue : tokens i = recallValueId value) :
    ‖recallHeadRead (headAt recallConfig (recallFirstAttention eps (recallCopyTemperature eta)) eps
        (fun r => (r.val : ℝ)) (fun j => recallRawEmbedding (tokens j)) 0 i) - recallSymbolCode key‖ ≤ eta := by
  exact (recall_raw_compact_copy eps (recallCopyTemperature eta) heps hclip (by omega) tokens i selected hprev key value hkey hvalue).trans
    (recallCopyTemperature_error eta heta T hT)

example : (0 : ℝ) < 1 / 1000000 ∧ (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    (2 : ℕ) ≤ 64 ∧ (0 : Fin 2).val + 1 = (1 : Fin 2).val ∧
    (fun j : Fin 2 => if j = 0 then recallKeyId 0 else recallValueId 0) 0 = recallKeyId 0 ∧
    (fun j : Fin 2 => if j = 0 then recallKeyId 0 else recallValueId 0) 1 = recallValueId 0 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by decide, by decide, by simp, by simp⟩

/-- The real first attention residual has the same uniform tolerance at every true raw adjacent write.
Source: actual W_o and initially empty destination, so a semantic copied-key input is not assumed. -/
theorem recall_first_state_accuracy (eta eps : ℝ) (heta : 0 < eta) (heps : 0 ≤ eps) (hclip : eps ≤ 1)
    {T : ℕ} (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size) (i selected : Fin T)
    (hprev : selected.val + 1 = i.val) (key value : Fin 256)
    (hkey : tokens selected = recallKeyId key) (hvalue : tokens i = recallValueId value) :
    ‖recallSlotRead 18 (recallFirstState eps (recallCopyTemperature eta) (fun r => (r.val : ℝ)) tokens i) -
      recallSymbolCode key‖ ≤ eta := by
  rw [recallFirstState, recall_first_residual_copy_slot]
  exact recall_raw_copy_accuracy eta eps heta heps hclip hT tokens i selected hprev key value hkey hvalue

example : (0 : ℝ) < 1 / 1000000 ∧ (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    (2 : ℕ) ≤ 64 ∧ (0 : Fin 2).val + 1 = (1 : Fin 2).val ∧
    (fun j : Fin 2 => if j = 0 then recallKeyId 0 else recallValueId 0) 0 = recallKeyId 0 ∧
    (fun j : Fin 2 => if j = 0 then recallKeyId 0 else recallValueId 0) 1 = recallValueId 0 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by decide, by decide, by simp, by simp⟩

/-- A genuine copied norm-two raw key has norm at least one at tolerance at most one.
Source: the derived actual finite-temperature copy accuracy and the reverse norm triangle inequality. -/
theorem recall_first_state_copy_nonzero (eta eps : ℝ) (heta : 0 < eta) (hsmall : eta ≤ 1)
    (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ} (hT : T ≤ 64)
    (tokens : Fin T → Fin recallConfig.vocab_size) (i selected : Fin T)
    (hprev : selected.val + 1 = i.val) (key value : Fin 256)
    (hkey : tokens selected = recallKeyId key) (hvalue : tokens i = recallValueId value) :
    1 ≤ ‖recallSlotRead 18 (recallFirstState eps (recallCopyTemperature eta) (fun r => (r.val : ℝ)) tokens i)‖ := by
  have herr := recall_first_state_accuracy eta eps heta heps hclip hT tokens i selected hprev key value hkey hvalue
  have hn := norm_sub_norm_le
    (recallSymbolCode key) (recallSlotRead 18 (recallFirstState eps (recallCopyTemperature eta) (fun r => (r.val : ℝ)) tokens i))
  rw [norm_sub_rev] at hn
  have hk : ‖recallSymbolCode key‖ = 2 := recallCode_norm _
  rw [hk] at hn
  linarith

example : (0 : ℝ) < 1 / 1000000 ∧ (1 / 1000000 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 / 100000 ∧
    (1 / 100000 : ℝ) ≤ 1 ∧ (2 : ℕ) ≤ 64 ∧ (0 : Fin 2).val + 1 = (1 : Fin 2).val ∧
    (fun j : Fin 2 => if j = 0 then recallKeyId 0 else recallValueId 0) 0 = recallKeyId 0 ∧
    (fun j : Fin 2 => if j = 0 then recallKeyId 0 else recallValueId 0) 1 = recallValueId 0 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by decide, by decide, by simp, by simp⟩

/-- A fixed global key-copy tolerance leaves room inside the derived actual latest-write margin.
Source: a conservative sixteenth of the proved uniform normalized rotary gap. -/
noncomputable def recallKeyTolerance : ℝ := recallLatestMargin / 16

/-- This chosen shared tolerance is strictly positive.
Source: the actual frequency-derived latest-write margin, independent of any raw prefix. -/
theorem recallKeyTolerance_pos : 0 < recallKeyTolerance := by
  have hm := recallLatestMargin_pos
  unfold recallKeyTolerance
  positivity

/-- The same tolerance is small enough to ensure every genuine copied key remains nonzero.
Source: its real rotary margin and the independently derived content-margin upper bound. -/
theorem recallKeyTolerance_le_one : recallKeyTolerance ≤ 1 := by
  unfold recallKeyTolerance
  linarith [recallLatestMargin_le_content]

/-- An eightfold perturbation budget still leaves at least half the original latest-write margin.
Source: the stated fixed tolerance, with the actual positive margin retained in the conclusion. -/
theorem recallKeyTolerance_budget : 8 * recallKeyTolerance ≤ recallLatestMargin / 2 := by
  unfold recallKeyTolerance
  linarith

end Transformer.GPTMini.Semantics
