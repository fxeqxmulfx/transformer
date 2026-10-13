import Transformer.Grokking.CircuitEfficiency.SectionC_GainLimitBalance
import Transformer.Grokking.CircuitEfficiency.SectionC_NativePositiveBalance

/-!
# Physical efficiency orders positive native balanced allocations

Sources: Varma et al., arXiv:2309.02390v1, section 3, Efficiency,
and appendix C product/test logits; native AdamW at lab commit
4e10267. Solve the actual normalized CE balance at a strictly
positive physical parameter point with two unequal fixed gains.
One uniform decay and one shared clipped CE scale apply to all
four coordinates. No stronger Gen logit or successful test margin
is inserted as a premise or as part of a predicate definition.

The actual pair equations force balanced factors, positive decay,
larger Gen coordinates and a strictly larger Gen logit when Gen's
physical gain is larger. All held-out competitors are then below
the correct class. A nonzero true-CE point witnesses all hypotheses
with gains 3/2, Gen factors 1 and Mem factors 1/2, epsilon chosen
from the actual clipped CE scale, and uniform decay 1/2.

This is necessary positive-interior allocation, not attraction or
path convergence; the witness will require a separate retained
trajectory proof. Degree-two fixed tables, chosen witness epsilon,
the source's coupled norm-cost/GD deviation and absent learned
stochastic/numerical GPTMini transfer remain explicit.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Unequal physical gains admit a nonzero actual native balanced
point. Sources: appendix C's true CE and native normalization at
4e10267; this point is not declared an attracting or converging path. -/
theorem gain_native_efficient_point_balance (remaining : ℕ) (bound : ℝ) (hclip : 0 < bound) :
    let state := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let scale := gainCEGradientScale remaining 3 2 bound state
    0 < 3 * scale ∧ ∀ i, (1 / 2 : ℝ) * (state i).parameter +
      appliedGainNativeGradient remaining 3 2 bound state i /
        (|appliedGainNativeGradient remaining 3 2 bound state i| + 3 * scale) = 0 := by
  dsimp only
  let state := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
  let scale := gainCEGradientScale remaining 3 2 bound state
  have hs : 0 < scale := gain_ce_gradient_scale_pos remaining 3 2 bound state hclip
  have hp : ∀ i : Fin 4, (state i).parameter = ![1, 1, 1 / 2, 1 / 2] i := by
    intro i
    fin_cases i <;> rfl
  have hg : ∀ i, appliedGainNativeGradient remaining 3 2 bound state i = ![-3 * scale, -3 * scale, -scale, -scale] i := by
    intro i
    rw [gain_native_applied_gradient_scale, hp]
    change -(scale * nativeFactorGain 3 2 i * ![1, 1, 1 / 2, 1 / 2] (nativeFactorPartner i)) = _
    fin_cases i <;> norm_num [nativeFactorGain, nativeFactorPartner] <;> ring
  refine ⟨by positivity, ?_⟩
  intro i
  change (1 / 2 : ℝ) * (state i).parameter +
    appliedGainNativeGradient remaining 3 2 bound state i / (|appliedGainNativeGradient remaining 3 2 bound state i| + 3 * scale) = 0
  rw [hg, hp]
  fin_cases i
  · change (1 / 2 : ℝ) * 1 + (-3 * scale) / (|-3 * scale| + 3 * scale) = 0
    rw [abs_of_neg (by linarith : -3 * scale < 0)]
    field_simp
    ring
  · change (1 / 2 : ℝ) * 1 + (-3 * scale) / (|-3 * scale| + 3 * scale) = 0
    rw [abs_of_neg (by linarith : -3 * scale < 0)]
    field_simp
    ring
  · change (1 / 2 : ℝ) * (1 / 2) + (-scale) / (|-scale| + 3 * scale) = 0
    rw [abs_of_neg (by linarith : -scale < 0)]
    field_simp
    ring
  · change (1 / 2 : ℝ) * (1 / 2) + (-scale) / (|-scale| + 3 * scale) = 0
    rw [abs_of_neg (by linarith : -scale < 0)]
    field_simp
    ring

example : (0 : ℝ) < 1 := by norm_num

/-- Positive actual native balance selects the more efficient
physical circuit in both parameter amplitude and held-out logit.
Sources: section 3 Efficiency and appendix C test products, corrected
to the explicit forward/uniform-native-decay equations at 4e10267. -/
theorem gain_native_positive_balanced_allocation (remaining : ℕ) (genGain memGain bound eps decay : ℝ)
    (state : NativeSubweightState) (hmem : 0 < memGain) (hgain : memGain < genGain)
    (hclip : 0 < bound) (he : 0 < eps) (hp : ∀ i, 0 < (state i).parameter)
    (hb : ∀ i, decay * (state i).parameter + appliedGainNativeGradient remaining genGain memGain bound state i /
      (|appliedGainNativeGradient remaining genGain memGain bound state i| + eps) = 0) :
    0 < decay ∧ (state 0).parameter = (state 1).parameter ∧
      (state 2).parameter = (state 3).parameter ∧ (state 2).parameter < (state 0).parameter ∧
      physicalCircuitScore memGain (state 2).parameter (state 3).parameter <
        physicalCircuitScore genGain (state 0).parameter (state 1).parameter := by
  let scale := gainCEGradientScale remaining genGain memGain bound state
  have hs : 0 < scale := gain_ce_gradient_scale_pos remaining genGain memGain bound state hclip
  have hgen : 0 < genGain := lt_trans hmem hgain
  have hpositive : ∀ i, 0 < nativeFactorGain genGain memGain i := by
    intro i
    fin_cases i
    · exact hgen
    · exact hgen
    · exact hmem
    · exact hmem
  have hn : ∀ i, decay * (state i).parameter =
      (scale * nativeFactorGain genGain memGain i) * (state (nativeFactorPartner i)).parameter /
        ((scale * nativeFactorGain genGain memGain i) * (state (nativeFactorPartner i)).parameter + eps) := by
    intro i
    have hi := hb i
    rw [abs_of_neg (gain_native_applied_gradient_negative remaining _ _ _ state i hclip (hpositive i) (hp _)),
      gain_native_applied_gradient_scale, neg_neg, neg_div] at hi
    linarith only [hi]
  have hg := native_positive_pair_balance (scale * genGain) eps decay (state 0).parameter (state 1).parameter
    (mul_pos hs hgen) he (hp 0) (hp 1) (by simpa [nativeFactorGain, nativeFactorPartner] using hn 0)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hn 1)
  have hm := native_positive_pair_balance (scale * memGain) eps decay (state 2).parameter (state 3).parameter
    (mul_pos hs hmem) he (hp 2) (hp 3) (by simpa [nativeFactorGain, nativeFactorPartner] using hn 2)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hn 3)
  have hdiff : (decay * scale * genGain * memGain) * ((state 0).parameter - (state 2).parameter) =
      decay * eps * (genGain - memGain) := by
    have ha := congrArg (fun t : ℝ => t * memGain) hg.2.2
    have hb := congrArg (fun t : ℝ => t * genGain) hm.2.2
    nlinarith only [ha, hb]
  have hfactor : (state 2).parameter < (state 0).parameter := by
    apply sub_pos.mp
    apply (mul_pos_iff_of_pos_left (mul_pos (mul_pos (mul_pos hg.1 hs) hgen) hmem)).mp
    rw [hdiff]
    exact mul_pos (mul_pos hg.1 he) (sub_pos.mpr hgain)
  have hsq : (state 2).parameter * (state 2).parameter < (state 0).parameter * (state 0).parameter := by
    have ht := mul_pos (sub_pos.mpr hfactor) (show 0 < (state 0).parameter + (state 2).parameter by linarith [hp 0, hp 2])
    nlinarith only [ht]
  refine ⟨hg.1, hg.2.1, hm.2.1, hfactor, ?_⟩
  unfold physicalCircuitScore
  rw [← hg.2.1, ← hm.2.1]
  exact lt_trans (mul_lt_mul_of_pos_left hsq hmem) (mul_lt_mul_of_pos_right hgain (mul_pos (hp 0) (hp 0)))

example :
    let state := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let scale := gainCEGradientScale 111 3 2 1 state
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ 0 < 3 * scale ∧
    (∀ i : Fin 4, 0 < (state i).parameter) ∧
    (∀ i, (1 / 2 : ℝ) * (state i).parameter + appliedGainNativeGradient 111 3 2 1 state i /
      (|appliedGainNativeGradient 111 3 2 1 state i| + 3 * scale) = 0) := by
  dsimp only
  have hw := gain_native_efficient_point_balance 111 1 (by norm_num)
  refine ⟨by norm_num, by norm_num, by norm_num, hw.1, ?_, hw.2⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- Every class is strictly beaten by the correct held-out class at
such a positive native balance. Sources: appendix C test-table logits
and section 3 Efficiency; a successful margin is derived, not supplied. -/
theorem gain_native_positive_balanced_heldout_correct (remaining : ℕ) (genGain memGain bound eps decay : ℝ)
    (state : NativeSubweightState) (hmem : 0 < memGain) (hgain : memGain < genGain)
    (hclip : 0 < bound) (he : 0 < eps) (hp : ∀ i, 0 < (state i).parameter)
    (hb : ∀ i, decay * (state i).parameter + appliedGainNativeGradient remaining genGain memGain bound state i /
      (|appliedGainNativeGradient remaining genGain memGain bound state i| + eps) = 0) :
    Transformer.Grokking.NaiveLoss.StrictCorrect (heldoutTableLogits remaining
      (physicalCircuitScore genGain (state 0).parameter (state 1).parameter)
      (physicalCircuitScore memGain (state 2).parameter (state 3).parameter)) 0 := by
  have ha := gain_native_positive_balanced_allocation remaining _ _ _ _ _ state hmem hgain hclip he hp hb
  exact heldout_table_strict_correct remaining _ _ (mul_pos (lt_trans hmem hgain) (mul_pos (hp 0) (hp 1))) ha.2.2.2.2

example :
    let state := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let scale := gainCEGradientScale 111 3 2 1 state
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ 0 < 3 * scale ∧
    (∀ i : Fin 4, 0 < (state i).parameter) ∧
    (∀ i, (1 / 2 : ℝ) * (state i).parameter + appliedGainNativeGradient 111 3 2 1 state i /
      (|appliedGainNativeGradient 111 3 2 1 state i| + 3 * scale) = 0) := by
  dsimp only
  have hw := gain_native_efficient_point_balance 111 1 (by norm_num)
  refine ⟨by norm_num, by norm_num, by norm_num, hw.1, ?_, hw.2⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

end Transformer.Grokking.CircuitEfficiency
