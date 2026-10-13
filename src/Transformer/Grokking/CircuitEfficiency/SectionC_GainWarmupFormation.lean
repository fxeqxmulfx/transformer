import Transformer.Grokking.CircuitEfficiency.SectionC_GainWarmupBounds

/-!
# Actual circuit formation with zero-rate first native warmup

Sources: Varma et al., arXiv:2309.02390v1, section 3 and appendix C,
one-zero-factor formation and product CE; retained native AdamW
and completed-update warmup at lab commits 88aa892/0033b1b.
The actual first rate is zero. Physical parameters and the complete
fixed-table CE input are unchanged, but buffers and clock advance.
A positive partner generates strictly negative first momentum and
strictly positive variance even at that zero-rate update.
The original betas are 0.9 and 0.98; the second rate is 0.0001.
Every gradient comes from the current complete finite-class CE.
The general activation result allows retained nonpositive moments
already at initialization, and does not assume that they are zero.

At the second actual rate, every coordinate with a positive initial
partner becomes positive in the native sign region. Source-style
zero-first-factor seeds therefore form positive Gen and Mem product
scores at clock two, without an assigned future gradient stream or
a restarted optimizer. Formation does not select the winning circuit.

Fixed physical gains, plain complete CE, shared clipping and uniform
native decoupled decay differ from the source's coupled norm-cost GD.
First-input equality uses the same fixed table at both evaluations;
changing minibatches and learned GPTMini mechanisms need separate
bridges. These exact-real finite-clock results supply neither a
transition time, attraction, nor numerical or stochastic transfer.
Frozen runs, checkpoints, moments and measurements are unchanged.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Zero-rate warmup preserves physical parameters and its actual
complete CE callback, while retaining the numerical optimizer state.
Sources: appendix C fixed train table and native schedule at 0033b1b;
the equality does not extend to different sampled batches. -/
theorem gain_native_warmup_first_feedback (remaining : ℕ) (genGain memGain bound eps : ℝ)
    (initial : NativeSubweightState) :
    (∀ i, (gainNativeWarmupPath remaining genGain memGain bound eps initial 1 i).parameter = (initial i).parameter) ∧
    appliedGainNativeGradient remaining genGain memGain bound
      (gainNativeWarmupPath remaining genGain memGain bound eps initial 1) =
        appliedGainNativeGradient remaining genGain memGain bound initial := by
  have hp : ∀ i, (gainNativeWarmupPath remaining genGain memGain bound eps initial 1 i).parameter = (initial i).parameter := by
    intro i
    change (1 - grokkingWarmupRate 0 * (1 / 10)) * (initial i).parameter - grokkingWarmupRate 0 * _ = _
    rw [grokking_warmup_rate_profile.1]
    ring
  exact ⟨hp, gain_native_equal_parameters_applied_gradient remaining genGain memGain bound _ initial hp⟩

/-- A positive partner creates real negative momentum, positive
variance and clock one on the zero-rate first update. Sources:
appendix C partner derivatives and native retained buffers at 88aa892;
neither physical movement nor positive epsilon is needed for buffers. -/
theorem gain_native_seeded_warmup_first_buffers (remaining : ℕ) (genGain memGain bound eps : ℝ)
    (parameters : (ℝ × ℝ) × (ℝ × ℝ)) (i : Fin 4) (hclip : 0 < bound)
    (hg : 0 < nativeFactorGain genGain memGain i)
    (hp : 0 < (seededNativeSubweights parameters (nativeFactorPartner i)).parameter) :
    (gainNativeWarmupPath remaining genGain memGain bound eps (seededNativeSubweights parameters) 1 i).moment < 0 ∧
    0 < (gainNativeWarmupPath remaining genGain memGain bound eps (seededNativeSubweights parameters) 1 i).variance ∧
    (gainNativeWarmupPath remaining genGain memGain bound eps (seededNativeSubweights parameters) 1 i).clock = 1 := by
  let initial := seededNativeSubweights parameters
  let gradient := appliedGainNativeGradient remaining genGain memGain bound initial i
  have hnegative : gradient < 0 := gain_native_applied_gradient_negative remaining genGain memGain bound initial i hclip hg hp
  have hsquare : 0 < gradient * gradient := mul_pos_of_neg_of_neg hnegative hnegative
  have hz : (initial i).moment = 0 ∧ (initial i).variance = 0 ∧ (initial i).clock = 0 := by
    fin_cases i <;> exact ⟨rfl, rfl, rfl⟩
  change (9 / 10) * (initial i).moment + (1 - 9 / 10) * gradient < 0 ∧
    0 < (49 / 50) * (initial i).variance + (1 - 49 / 50) * gradient ^ 2 ∧ (initial i).clock + 1 = 1
  rw [hz.1, hz.2.1, hz.2.2]
  refine ⟨by nlinarith only [hnegative], by nlinarith only [hsquare], by norm_num⟩

example : (0 : ℝ) < 1 ∧ 0 < nativeFactorGain 3 2 0 ∧
    0 < (seededNativeSubweights ((0, 1 / 200), (0, 1)) (nativeFactorPartner 0)).parameter := by
  norm_num [nativeFactorGain, nativeFactorPartner, seededNativeSubweights, seededScalarState]

/-- A positive initial partner activates the corresponding actual
coordinate at the second warmup clock with retained moments.
Sources: appendix C one-zero-factor formation and native warmup at
0033b1b; the current negative CE input follows from physical parameters. -/
theorem gain_native_warmup_coordinate_forms (remaining : ℕ) (genGain memGain bound eps : ℝ)
    (initial : NativeSubweightState) (i : Fin 4)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound) (he : 0 < eps)
    (hs : NonnegativeNativeState initial) (hp : 0 < (initial (nativeFactorPartner i)).parameter) :
    PositiveScalarState (gainNativeWarmupPath remaining genGain memGain bound eps initial 2 i) := by
  let state := gainNativeWarmupPath remaining genGain memGain bound eps initial 1
  have hn : NonnegativeNativeState state := gain_native_scheduled_nonnegative_path remaining genGain memGain bound
    (9 / 10) (49 / 50) eps (1 / 10) grokkingWarmupRate initial 1 hgen hmem hclip
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) he
    (fun n => ⟨(grokking_warmup_rate_legal n).1, (grokking_warmup_rate_legal n).2.2⟩) hs
  have hpartner : 0 < (state (nativeFactorPartner i)).parameter := by
    rw [(gain_native_warmup_first_feedback remaining genGain memGain bound eps initial).1]
    exact hp
  have hg := gain_native_applied_gradient_negative remaining genGain memGain bound state i hclip
    (gain_native_factor_gain_pos genGain memGain hgen hmem i) hpartner
  change PositiveScalarState (scalarNativeStep (9 / 10) (49 / 50) eps (1 / 10) (grokkingWarmupRate 1)
    (state i) (appliedGainNativeGradient remaining genGain memGain bound state i))
  rw [grokking_warmup_rate_profile.2.1]
  exact scalar_negative_gradient_activates _ _ _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) he (by norm_num) (by norm_num) (hn i) hg

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    0 < (seededNativeSubweights ((0, 1 / 200), (0, 1)) (nativeFactorPartner 0)).parameter := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [seededNativeSubweights, seededScalarState, nativeFactorPartner]⟩

/-- Both source-style zero-first-factor circuit scores become
strictly positive at actual warmup clock two. Sources: section 3
formation and appendix C product readouts, with native warmup at
0033b1b; simultaneous formation does not establish Gen dominance. -/
theorem gain_native_warmup_source_scores_form (remaining : ℕ) (genGain memGain bound eps b d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound) (he : 0 < eps)
    (hb : 0 < b) (hd : 0 < d) :
    let state := gainNativeWarmupPath remaining genGain memGain bound eps (seededNativeSubweights ((0, b), (0, d))) 2
    0 < physicalCircuitScore genGain (state 0).parameter (state 1).parameter ∧
      0 < physicalCircuitScore memGain (state 2).parameter (state 3).parameter := by
  let initial := seededNativeSubweights ((0, b), (0, d))
  let first := gainNativeWarmupPath remaining genGain memGain bound eps initial 1
  have hs : NonnegativeNativeState initial := native_seeded_nonnegative _ _ _ _ le_rfl (le_of_lt hb) le_rfl (le_of_lt hd)
  have hn : NonnegativeNativeState first := gain_native_scheduled_nonnegative_path remaining genGain memGain bound
    (9 / 10) (49 / 50) eps (1 / 10) grokkingWarmupRate initial 1 hgen hmem hclip
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) he
    (fun n => ⟨(grokking_warmup_rate_legal n).1, (grokking_warmup_rate_legal n).2.2⟩) hs
  have hactivated (i : Fin 4) (hi : 0 < (initial (nativeFactorPartner i)).parameter) :=
    gain_native_warmup_coordinate_forms remaining genGain memGain bound eps initial i hgen hmem hclip he hs hi
  have hpreserved (i : Fin 4) (hi : 0 < (initial i).parameter) :
      0 < (gainNativeWarmupPath remaining genGain memGain bound eps initial 2 i).parameter := by
    have hfirst : 0 < (first i).parameter := by
      rw [(gain_native_warmup_first_feedback remaining genGain memGain bound eps initial).1]
      exact hi
    have hf := gain_native_parameter_decay_floor remaining genGain memGain bound (9 / 10) (49 / 50) eps (1 / 10)
      (grokkingWarmupRate 1) first i hclip (le_of_lt (gain_native_factor_gain_pos _ _ hgen hmem i))
      (hn _).1 (by norm_num) (by norm_num) he (grokking_warmup_rate_legal 1).1 (hn i).2.1
    have hk : 0 < 1 - grokkingWarmupRate 1 * (1 / 10) := by rw [grokking_warmup_rate_profile.2.1]; norm_num
    exact lt_of_lt_of_le (mul_pos hk hfirst) hf
  have hzero := hactivated 0 (by simpa [initial, seededNativeSubweights, seededScalarState, nativeFactorPartner] using hb)
  have htwo := hactivated 2 (by simpa [initial, seededNativeSubweights, seededScalarState, nativeFactorPartner] using hd)
  have hone := hpreserved 1 (by simpa [initial, seededNativeSubweights, seededScalarState] using hb)
  have hthree := hpreserved 3 (by simpa [initial, seededNativeSubweights, seededScalarState] using hd)
  exact ⟨mul_pos hgen (mul_pos hzero.1 hone), mul_pos hmem (mul_pos htwo.1 hthree)⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    (0 : ℝ) < 1 / 200 ∧ (0 : ℝ) < 1 := by norm_num

end Transformer.Grokking.CircuitEfficiency
