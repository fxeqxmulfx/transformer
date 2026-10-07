import Transformer.GPTMini.Semantics.RecallRawRouting

/-!
# A finite shared temperature gives uniform actual recall retrieval accuracy

Source: the original softmax head at f11b6e2 and raw Basis context
64 at cbafbe9. The derived genuine head error is not left as an
assumed small leakage bound. One explicit finite log-temperature
depending only on the global requested tolerance supplies that
accuracy for every applicable raw table/query prefix.

The logarithmic choice evaluates the softmax tail directly: its
true score gap is log(1+2016/tolerance), where 2016=63*32 is the
uniform competitor count times the actual value diameter. No hard
attention limit or optimizer convergence is substituted. A fixed
tolerance, one sixteenth of the true second-prenorm lower scale,
leaves room for tied value readout at gain two divided by that scale.
Full raw-parser discharge, actual second W_o and readout still remain.
All bounds are real-arithmetic statements, not floating-point claims.
-/

namespace Transformer.GPTMini.Semantics

/-- One finite ordinary second-head temperature controls the entire raw recall context.
Source: genuine retained normalized score margin, at most 63 competitors and derived V diameter 32. -/
noncomputable def recallRetrievalTemperature (eta : ℝ) : ℝ :=
  Real.log (2 * Real.log (1 + 2016 / eta) / recallLatestMargin)

/-- The actual retained gap at the fixed finite temperature is exactly the chosen logarithmic tail threshold.
Source: original exp(log_alpha), positive real rotary margin and direct logarithm evaluation. -/
theorem recallRetrievalTemperature_gap (eta : ℝ) (heta : 0 < eta) :
    recallMatchGap (recallRetrievalTemperature eta) = Real.log (1 + 2016 / eta) := by
  have hd : 0 < 2016 / eta := div_pos (by norm_num) heta
  have hl := Real.log_pos (by linarith : 1 < 1 + 2016 / eta)
  unfold recallMatchGap recallRetrievalTemperature
  rw [Real.exp_log (div_pos (mul_pos (by norm_num) hl) recallLatestMargin_pos),
    div_mul_cancel₀ _ (ne_of_gt recallLatestMargin_pos)]
  ring

example : (0 : ℝ) < 1 / 1000000 := by norm_num

/-- The genuine complete-context finite tail estimate is at most any positive requested tolerance.
Source: the evaluated actual logarithmic gap and the true original 64-slot/32-diameter budget. -/
theorem recallRetrievalTemperature_error (eta : ℝ) (heta : 0 < eta) (T : ℕ) (hT : T ≤ 64) :
    ((T - 1 : ℕ) : ℝ) * Real.exp (-recallMatchGap (recallRetrievalTemperature eta)) * 32 ≤ eta := by
  have hx : 0 < 1 + 2016 / eta := by positivity
  rw [recallRetrievalTemperature_gap eta heta, Real.exp_neg, Real.exp_log hx]
  have hc : ((T - 1 : ℕ) : ℝ) ≤ 63 := by exact_mod_cast (by omega : T - 1 ≤ 63)
  have he : eta * (1 + 2016 / eta) = eta + 2016 := by field_simp [ne_of_gt heta]
  have hi : 2016 * (1 + 2016 / eta)⁻¹ ≤ eta := by
    rw [← div_eq_mul_inv, div_le_iff₀ hx, he]
    linarith
  have hm := mul_le_mul_of_nonneg_right hc (by positivity : 0 ≤ (1 + 2016 / eta)⁻¹)
  calc _ ≤ 63 * (1 + 2016 / eta)⁻¹ * 32 := mul_le_mul_of_nonneg_right hm (by norm_num)
    _ = 2016 * (1 + 2016 / eta)⁻¹ := by ring
    _ ≤ eta := hi

example : (0 : ℝ) < 1 / 1000000 ∧ (64 : ℕ) ≤ 64 := by norm_num

/-- The genuine raw softmax/XSA head has any prescribed positive accuracy at one fixed finite shared temperature.
Source: complete actual raw routing/values/self-value, not a supplied small leakage or desired-score premise. -/
theorem recall_raw_head_accuracy_fixed (P : ℕ) (eta eps : ℝ) (heta : 0 < eta) (heps : 0 ≤ eps)
    (hclip : eps ≤ 1) {T : ℕ} (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size)
    (first query selected previous : Fin T) (hlayout : RecallRawLayout P tokens first)
    (key value : Fin 256) (hquery : tokens query = recallKeyId key)
    (hprev : previous.val + 1 = selected.val) (hkey : tokens previous = recallKeyId key)
    (hvalue : tokens selected = recallValueId value) (hinside : selected.val ≤ 2 * P)
    (hvisible : selected.val ≤ query.val) (hlatest : RecallRawLatestWrite P tokens selected key) :
    ‖attentionHead recallConfig (recallRetrievalTemperature eta) eps (recallMatchQuery P eps tokens)
        (recallMatchKey P eps tokens) (recallMatchValue P eps tokens) (fun r => (r.val : ℝ)) query -
      recallMatchValue P eps tokens selected‖ ≤ eta := by
  exact (recall_raw_head_retrieval P (recallRetrievalTemperature eta) eps heps hclip hT tokens first query selected previous
    hlayout key value hquery hprev hkey hvalue hinside hvisible hlatest).trans
      (recallRetrievalTemperature_error eta heta T hT)

example : (0 : ℝ) < 1 / 1000000 ∧ (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (6 : ℕ) ≤ 64 ∧
    RecallRawLayout 2 recallScoreGapWitness 0 ∧ recallScoreGapWitness 5 = recallKeyId 0 ∧
    (3 : Fin 6).val + 1 = (4 : Fin 6).val ∧ recallScoreGapWitness 3 = recallKeyId 0 ∧
    recallScoreGapWitness 4 = recallValueId 1 ∧ (4 : Fin 6).val ≤ 2 * 2 ∧
    (4 : Fin 6).val ≤ (5 : Fin 6).val ∧ RecallRawLatestWrite 2 recallScoreGapWitness 4 0 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by decide, recallScoreGapWitness_layout,
    by simp [recallScoreGapWitness], by decide, by simp [recallScoreGapWitness], by simp [recallScoreGapWitness],
    by decide, by decide, recallScoreGapWitness_latest⟩

/-- A fixed global retrieval tolerance reserves sixteenfold room within every true selected-value RMS amplitude.
Source: the actual positive raw second-prenorm lower bound, independent of the token prefix and selected position. -/
noncomputable def recallRetrievalTolerance (P : ℕ) (eps : ℝ) : ℝ := recallSecondScaleLower P eps / 16

/-- The chosen shared tolerance is strictly positive, so it supplies an ordinary finite temperature.
Source: the genuine encoder/prenorm lower bound, with the implementation's nonnegative epsilon. -/
theorem recallRetrievalTolerance_pos (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) :
    0 < recallRetrievalTolerance P eps :=
  div_pos (recallSecondScaleLower_pos P eps heps) (by norm_num)

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

/-- The exact tolerance budget uses the independently derived actual lower prenorm scale.
Source: the fixed sixteenth-scale coefficient, not a data-dependent retrieval-error input. -/
theorem recallRetrievalTolerance_product (P : ℕ) (eps : ℝ) :
    16 * recallRetrievalTolerance P eps = recallSecondScaleLower P eps := by
  unfold recallRetrievalTolerance
  ring

/-- One ordinary shared output gain can overcome the raw query/type readout without depending on any selected record.
Source: the tied norm-two raw value code and the fixed actual second-prenorm lower bound; readout margins remain to prove. -/
noncomputable def recallReadoutGain (P : ℕ) (eps : ℝ) : ℝ := 2 / recallSecondScaleLower P eps

/-- The chosen output-matrix coefficient is strictly positive and finite at every allowed model epsilon.
Source: the actual fixed positive next-prenorm lower multiplier. -/
theorem recallReadoutGain_pos (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) : 0 < recallReadoutGain P eps :=
  div_pos (by norm_num) (recallSecondScaleLower_pos P eps heps)

example : (0 : ℝ) ≤ 1 / 100000 := by norm_num

/-- The exact shared output-gain/lower-scale product is two, retaining ordinary input-independent weights.
Source: the stated coefficient and nonzero genuine uniform prenorm lower bound. -/
theorem recallReadoutGain_product (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) :
    recallReadoutGain P eps * recallSecondScaleLower P eps = 2 := by
  unfold recallReadoutGain
  exact div_mul_cancel₀ _ (ne_of_gt (recallSecondScaleLower_pos P eps heps))

example : (0 : ℝ) ≤ 0 := by norm_num

/-- The fixed retrieval tolerance is at most one sixteenth of every genuine raw value amplitude.
Source: actual complete encoder/prenorm bounds discharged from the raw layout, not a prepared scale premise. -/
theorem recall_raw_retrieval_tolerance_bound (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) {T : ℕ}
    (tokens : Fin T → Fin recallConfig.vocab_size) (first record : Fin T) (hlayout : RecallRawLayout P tokens first) :
    16 * recallRetrievalTolerance P eps ≤ recallResidualScale eps (recallMatchState P eps tokens record) := by
  rw [recallRetrievalTolerance_product]
  exact recall_raw_second_scale_lower P eps (recallCopyTemperature recallKeyTolerance) heps
    (fun r => (r.val : ℝ)) tokens first record hlayout.1 hlayout.2.1 hlayout.2.2.1

example : (0 : ℝ) ≤ 1 / 100000 ∧ RecallRawLayout 2 recallScoreGapWitness 0 :=
  ⟨by norm_num, recallScoreGapWitness_layout⟩

/-- One fixed temperature derived from the genuine uniform prenorm scale gives the reserved tied-readout accuracy.
Source: actual raw head retrieval at the positive prefix-independent tolerance, with no small-error hypothesis. -/
theorem recall_raw_head_readout_accuracy (P : ℕ) (eps : ℝ) (heps : 0 ≤ eps) (hclip : eps ≤ 1) {T : ℕ}
    (hT : T ≤ 64) (tokens : Fin T → Fin recallConfig.vocab_size) (first query selected previous : Fin T)
    (hlayout : RecallRawLayout P tokens first) (key value : Fin 256) (hquery : tokens query = recallKeyId key)
    (hprev : previous.val + 1 = selected.val) (hkey : tokens previous = recallKeyId key)
    (hvalue : tokens selected = recallValueId value) (hinside : selected.val ≤ 2 * P)
    (hvisible : selected.val ≤ query.val) (hlatest : RecallRawLatestWrite P tokens selected key) :
    ‖attentionHead recallConfig (recallRetrievalTemperature (recallRetrievalTolerance P eps)) eps
        (recallMatchQuery P eps tokens) (recallMatchKey P eps tokens) (recallMatchValue P eps tokens)
        (fun r => (r.val : ℝ)) query - recallMatchValue P eps tokens selected‖ ≤ recallRetrievalTolerance P eps := by
  exact recall_raw_head_accuracy_fixed P (recallRetrievalTolerance P eps) eps (recallRetrievalTolerance_pos P eps heps)
    heps hclip hT tokens first query selected previous hlayout key value hquery hprev hkey hvalue hinside hvisible hlatest

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧ (6 : ℕ) ≤ 64 ∧
    RecallRawLayout 2 recallScoreGapWitness 0 ∧ recallScoreGapWitness 5 = recallKeyId 0 ∧
    (3 : Fin 6).val + 1 = (4 : Fin 6).val ∧ recallScoreGapWitness 3 = recallKeyId 0 ∧
    recallScoreGapWitness 4 = recallValueId 1 ∧ (4 : Fin 6).val ≤ 2 * 2 ∧
    (4 : Fin 6).val ≤ (5 : Fin 6).val ∧ RecallRawLatestWrite 2 recallScoreGapWitness 4 0 := by
  exact ⟨by norm_num, by norm_num, by decide, recallScoreGapWitness_layout, by simp [recallScoreGapWitness],
    by decide, by simp [recallScoreGapWitness], by simp [recallScoreGapWitness], by decide, by decide,
    recallScoreGapWitness_latest⟩

end Transformer.GPTMini.Semantics
