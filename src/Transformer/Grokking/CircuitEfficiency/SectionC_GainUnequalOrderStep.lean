import Transformer.Grokking.CircuitEfficiency.SectionC_GainMassOrder

/-!
# A quantified unequal-pair native step preserves Gen mass dominance

Sources: Varma et al., arXiv:2309.02390v1, section 3's physical
efficiency and appendix C's unequal product inputs; exact actual-CE
native balancing corrections and initialized estimates at 68d90fa.

For Gen mass at least Mem mass, the balanced-proxy difference has a
decay term in the mass difference and a positive efficiency term in
Mem mass. Subtract the true unequal Gen error. If error per Gen mass
is at most half the efficiency gap, and its rate-weighted value at
most half remaining decay, both surviving terms are nonnegative and
the Mem-mass efficiency term is strictly positive.

Thus the true unequal next Gen mass is strictly greater than Mem's.
Mem's own nonnegative correction only strengthens this conclusion.
Apply the comparison to the actual retained native step with its
original shared clipped CE scale and the positive tolerance derived
from the current physical box. No individual parameter limit, future
successful reference or alternate balanced trajectory is assumed.

This current update statement still needs initialized tail generation
and iteration for permanent selection. Exact-real fixed tables, zero
betas and uniform native decay differ from appendix C's coupled-cost
GD and preserved learned nonzero-beta GPTMini. Product/decision order
and positive limiting margin do not follow from mass order alone.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Small explicit current Gen relative correction preserves strict
next mass dominance for unequal actual-form pairs. Sources: appendix C
partner inputs and the native balancing law at 68d90fa; no future
error bound or parameter convergence is supplied as a hypothesis. -/
theorem gain_pair_mass_order_step (lower scale eps genGain memGain ceiling decay rate a b c d : ℝ)
    (hlower : 0 < lower) (hscale : lower ≤ scale) (hunit : scale ≤ 1) (he : 0 < eps)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hdseed : 0 ≤ d)
    (hmMass : 0 < c + d) (horder : c + d ≤ a + b) (hceiling : c + d ≤ 2 * ceiling)
    (hgapError : gainPairBalancingError genGain scale eps a b / (a + b) ≤
      gainBalancedRelativeGap lower eps genGain memGain ceiling / 2)
    (hdecayError : rate * (gainPairBalancingError genGain scale eps a b / (a + b)) ≤ (1 - rate * decay) / 2) :
    (gainPairStep memGain scale eps decay rate c d).1 + (gainPairStep memGain scale eps decay rate c d).2 <
      (gainPairStep genGain scale eps decay rate a b).1 + (gainPairStep genGain scale eps decay rate a b).2 := by
  have hs := lt_of_lt_of_le hlower hscale
  have hg := lt_trans hmem hgain
  have hgMass := lt_of_lt_of_le hmMass horder
  have hproxy := gain_balanced_mass_proxy_order_margin lower scale eps genGain memGain ceiling decay rate
    (a + b) (c + d) hlower hscale hunit he hmem hgain (le_of_lt heta) hmMass horder hceiling
  have hdiff : 0 ≤ (a + b) - (c + d) := by linarith only [horder]
  have hremaining : 0 ≤ 1 - rate * decay -
      rate * (gainPairBalancingError genGain scale eps a b / (a + b)) := by
    linarith only [hd, hdecayError]
  have hbase := mul_nonneg hremaining hdiff
  have hgapRemaining : 0 < gainBalancedRelativeGap lower eps genGain memGain ceiling -
      gainPairBalancingError genGain scale eps a b / (a + b) := by
    linarith only [hproxy.1, hgapError]
  have hboost := mul_pos (mul_pos heta hgapRemaining) hmMass
  have hcancel := div_mul_cancel₀ (gainPairBalancingError genGain scale eps a b) (ne_of_gt hgMass)
  have hcancelRate : rate * (gainPairBalancingError genGain scale eps a b / (a + b) * (a + b)) =
      rate * gainPairBalancingError genGain scale eps a b := by rw [hcancel]
  have hmemError := mul_nonneg (le_of_lt heta)
    (gain_pair_balancing_error_nonneg memGain scale eps c d (le_of_lt hmem) (le_of_lt hs) he hc hdseed)
  rw [gain_pair_step_balanced_proxy memGain scale eps decay rate c d (le_of_lt hmem) (le_of_lt hs) he hc hdseed,
    gain_pair_step_balanced_proxy genGain scale eps decay rate a b (le_of_lt hg) (le_of_lt hs) he ha hb]
  nlinarith only [hproxy.2, hbase, hboost, hcancelRate, hmemError]

example : (0 : ℝ) < 1 / 10 ∧ (1 / 10 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 / 1000 ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    (0 : ℝ) < 1 + 1 ∧ (1 : ℝ) + 1 ≤ 1 + 1 ∧ (1 : ℝ) + 1 ≤ 2 * 2 ∧
    gainPairBalancingError 3 (1 / 2) 1 1 1 / (1 + 1) ≤ gainBalancedRelativeGap (1 / 10) 1 3 2 2 / 2 ∧
    (1 / 1000 : ℝ) * (gainPairBalancingError 3 (1 / 2) 1 1 1 / (1 + 1)) ≤
      (1 - (1 / 1000 : ℝ) * (1 / 10)) / 2 := by
  norm_num [gainPairBalancingError, gainBalancedRelativeGap]

/-- An explicit positive current tolerance preserves unequal mass
order on the actual retained native step. Sources: appendix C true
CE and native partner feedback at 68d90fa; original clipping/scale
bounds and both error budgets are derived from the current point. -/
theorem gain_native_unequal_order_step (remaining : ℕ) (genGain memGain bound ceiling eps decay rate : ℝ)
    (state : NativeSubweightState) (hmem : 0 < memGain) (hgain : memGain < genGain)
    (hclip : 0 < bound) (he : 0 < eps) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (hp : ∀ i, 0 ≤ (state i).parameter ∧ (state i).parameter ≤ ceiling)
    (hmMass : 0 < (state 2).parameter + (state 3).parameter)
    (horder : (state 2).parameter + (state 3).parameter ≤ (state 0).parameter + (state 1).parameter)
    (herror : gainPairBalancingError genGain (gainCEGradientScale remaining genGain memGain bound state) eps
      (state 0).parameter (state 1).parameter / ((state 0).parameter + (state 1).parameter) ≤
        gainUnequalOrderTolerance (gainCEFeedbackFloor remaining genGain memGain bound ceiling)
          eps genGain memGain ceiling decay rate) :
    let next := gainNativeStep remaining genGain memGain bound 0 0 eps decay rate state
    (next 2).parameter + (next 3).parameter < (next 0).parameter + (next 1).parameter := by
  have hg := lt_trans hmem hgain
  have hc : 0 ≤ ceiling := le_trans (hp 0).1 (hp 0).2
  have hl := gain_ce_feedback_floor_pos remaining genGain memGain bound ceiling (le_of_lt hg) hc hclip
  have hs := gain_ce_gradient_scale_box_floor remaining genGain memGain bound ceiling state
    hg hmem (le_of_lt hgain) hc hclip hp
  have hu := (gain_ce_gradient_scale_unit_interval remaining genGain memGain bound state hclip).2
  have ht := gain_unequal_order_tolerance_bounds _ eps genGain memGain ceiling decay rate hl he hmem hgain hc heta hd
  have hdecayError := le_trans (mul_le_mul_of_nonneg_left herror (le_of_lt heta)) ht.2.2
  have hmBox : (state 2).parameter + (state 3).parameter ≤ 2 * ceiling := by linarith only [(hp 2).2, (hp 3).2]
  have hform := gain_native_zero_beta_pair_step remaining genGain memGain bound eps decay rate state
    hg hmem hclip (fun i => (hp i).1)
  have hnext := gain_pair_mass_order_step _ _ eps genGain memGain ceiling decay rate _ _ _ _
    hl hs (le_of_lt hu) he hmem hgain heta hd (hp 0).1 (hp 1).1 (hp 2).1 (hp 3).1
    hmMass horder hmBox (le_trans herror ht.2.1) hdecayError
  rw [← hform.1, ← hform.2] at hnext
  exact hnext

example :
    let state := seededNativeSubweights ((1, 1), (1, 1))
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧
    0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    (∀ i, 0 ≤ (state i).parameter ∧ (state i).parameter ≤ 2) ∧
    0 < (state 2).parameter + (state 3).parameter ∧
    (state 2).parameter + (state 3).parameter ≤ (state 0).parameter + (state 1).parameter ∧
    gainPairBalancingError 3 (gainCEGradientScale 0 3 2 1 state) 1
      (state 0).parameter (state 1).parameter / ((state 0).parameter + (state 1).parameter) ≤
        gainUnequalOrderTolerance (gainCEFeedbackFloor 0 3 2 1 2) 1 3 2 2 (1 / 10) (1 / 1000) := by
  dsimp only
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    ?_, by norm_num [seededNativeSubweights, seededScalarState],
    by norm_num [seededNativeSubweights, seededScalarState], ?_⟩
  · intro i
    fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]
  · have hl := gain_ce_feedback_floor_pos 0 3 2 1 2 (by norm_num) (by norm_num) (by norm_num)
    have ht := gain_unequal_order_tolerance_bounds _ 1 3 2 2 (1 / 10) (1 / 1000) hl
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    norm_num [seededNativeSubweights, seededScalarState, gainPairBalancingError]
    exact le_of_lt ht.1

/-- Positive current pair mass forms two positive factors after one
actual-form native step, including either zero-first-factor ordering.
Sources: section 3's compositional initialization and appendix C's
partner partials, with native normalization at 68d90fa. This formation
does not assert a successful physical product or a positive limit. -/
theorem gain_pair_step_factors_positive (gain scale eps decay rate a b : ℝ)
    (hg : 0 < gain) (hs : 0 < scale) (he : 0 < eps) (heta : 0 < rate)
    (hd : 0 < 1 - rate * decay) (ha : 0 ≤ a) (hb : 0 ≤ b) (hmass : 0 < a + b) :
    0 < (gainPairStep gain scale eps decay rate a b).1 ∧
      0 < (gainPairStep gain scale eps decay rate a b).2 := by
  have hda : 0 < scale * gain * a + eps := by positivity
  have hdb : 0 < scale * gain * b + eps := by positivity
  have hia : 0 ≤ rate * (scale * gain * a) / (scale * gain * a + eps) := by positivity
  have hib : 0 ≤ rate * (scale * gain * b) / (scale * gain * b + eps) := by positivity
  have hbaseA := mul_nonneg (le_of_lt hd) ha
  have hbaseB := mul_nonneg (le_of_lt hd) hb
  unfold gainPairStep
  dsimp only
  constructor
  · by_cases haPos : 0 < a
    · have hbase := mul_pos hd haPos
      linarith only [hbase, hib]
    · have haZero : a = 0 := by linarith only [ha, haPos]
      have hbPos : 0 < b := by linarith only [hmass, haZero]
      have hinput : 0 < rate * (scale * gain * b) / (scale * gain * b + eps) := by positivity
      linarith only [hbaseA, hinput]
  · by_cases hbPos : 0 < b
    · have hbase := mul_pos hd hbPos
      linarith only [hbase, hia]
    · have hbZero : b = 0 := by linarith only [hb, hbPos]
      have haPos : 0 < a := by linarith only [hmass, hbZero]
      have hinput : 0 < rate * (scale * gain * a) / (scale * gain * a + eps) := by positivity
      linarith only [hbaseB, hinput]

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧
    0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 0 + 1 := by norm_num

end Transformer.Grokking.CircuitEfficiency
