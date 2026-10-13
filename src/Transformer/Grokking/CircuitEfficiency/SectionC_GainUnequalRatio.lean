import Transformer.Grokking.CircuitEfficiency.SectionC_GainUnequalGap

/-!
# Actual unequal-pair Gen/Mem mass-ratio growth after a current error budget

Sources: Varma et al., arXiv:2309.02390v1, section 3's competing
compositional factors and appendix C's true CE partials; retained
native normalization and initialized balancing errors at d655756.

Convert the quantitative current relative-mass advantage into a
fixed multiplicative ratio increment greater than one while Gen
has no more pair mass than Mem. The next Mem mass is positive,
and its current relative multiplier has the native epsilon ceiling.
Both facts follow from the actual-form unequal partner update.

Then instantiate this comparison on the actual current retained
native state. Derive the shared positive clipped CE floor from the
physical box and use the original current forward, not an alternate
balanced feedback. The explicit Gen correction is kept in the
hypotheses as a numerical current inequality; it will be generated
after a finite start by the proved initialized relative-error limit.

Neither theorem assumes a future successful path or finite parameter
limit. They prove a single actual update, so a finite crossing and
permanent task correctness still require iteration. Exact reals,
fixed tables, zero betas and uniform native decay differ from the
source's coupled-cost GD and preserved learned nonzero-beta GPTMini.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Actual-form unequal Gen/Mem mass ratios gain the fixed half-gap
factor while Gen is weaker and its current relative correction fits
the budget. Sources: appendix C's product partials and native proxy
law at d655756; the shared current scale can change between steps. -/
theorem gain_pair_mass_ratio_growth_step (lower scale eps genGain memGain ceiling decay rate a b c d : ℝ)
    (hlower : 0 < lower) (hscale : lower ≤ scale) (hunit : scale ≤ 1) (he : 0 < eps)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hdseed : 0 ≤ d)
    (hgMass : 0 < a + b) (hmMass : 0 < c + d) (hweak : a + b ≤ c + d)
    (hceiling : c + d ≤ 2 * ceiling)
    (herror : gainPairBalancingError genGain scale eps a b / (a + b) ≤
      gainBalancedRelativeGap lower eps genGain memGain ceiling / 2) :
    gainUnequalRatioGrowth lower eps genGain memGain ceiling decay rate * ((a + b) / (c + d)) ≤
      ((gainPairStep genGain scale eps decay rate a b).1 + (gainPairStep genGain scale eps decay rate a b).2) /
        ((gainPairStep memGain scale eps decay rate c d).1 + (gainPairStep memGain scale eps decay rate c d).2) := by
  have hs := lt_of_lt_of_le hlower hscale
  have hgap := gain_pair_mass_relative_gap lower scale eps genGain memGain ceiling decay rate a b c d
    hlower hscale hunit he hmem hgain heta ha hb hc hdseed hgMass hmMass hweak hceiling herror
  have hmPos := gain_pair_step_mass_positive memGain scale eps decay rate c d
    (le_of_lt hmem) (le_of_lt hs) he (le_of_lt heta) hd hc hdseed hmMass
  have hmUpper := gain_pair_step_relative_mass_ceiling memGain scale eps decay rate c d
    (le_of_lt hmem) (le_of_lt hs) hunit he (le_of_lt heta) hc hdseed hmMass
  have hinc : 0 ≤ rate * gainBalancedRelativeGap lower eps genGain memGain ceiling / 2 :=
    div_nonneg (mul_nonneg (le_of_lt heta) (le_of_lt hgap.1)) (by norm_num)
  have hratio := gain_mass_ratio_from_relative_advantage (a + b) (c + d) _ _
    (1 - rate * decay + rate * memGain / eps) (rate * gainBalancedRelativeGap lower eps genGain memGain ceiling / 2)
    hgMass hmMass hmPos hinc hmUpper hgap.2
  have hupper : 0 < 1 - rate * decay + rate * memGain / eps := by positivity
  have heq : gainUnequalRatioGrowth lower eps genGain memGain ceiling decay rate =
      1 + (rate * gainBalancedRelativeGap lower eps genGain memGain ceiling / 2) /
        (1 - rate * decay + rate * memGain / eps) := by
    unfold gainUnequalRatioGrowth
    field_simp [ne_of_gt hupper]
  rw [heq]
  exact hratio

example : (0 : ℝ) < 1 / 10 ∧ (1 / 10 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 / 1000 ∧
    0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    (0 : ℝ) ≤ 1 / 4 ∧ (0 : ℝ) ≤ 1 / 4 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    (0 : ℝ) < 1 / 4 + 1 / 4 ∧ (0 : ℝ) < 1 + 1 ∧ (1 / 4 : ℝ) + 1 / 4 ≤ 1 + 1 ∧
    (1 : ℝ) + 1 ≤ 2 * 2 ∧
    gainPairBalancingError 3 (1 / 2) 1 (1 / 4) (1 / 4) / (1 / 4 + 1 / 4) ≤
      gainBalancedRelativeGap (1 / 10) 1 3 2 2 / 2 := by
  norm_num [gainPairBalancingError, gainBalancedRelativeGap]

/-- The actual native unequal Gen/Mem mass ratio grows by the fixed
half-gap factor under an explicit current Gen error budget. Sources:
appendix C true CE and retained native feedback at d655756; scale
and clipping bounds are generated from the current physical state. -/
theorem gain_native_unequal_ratio_growth (remaining : ℕ) (genGain memGain bound ceiling eps decay rate : ℝ)
    (state : NativeSubweightState) (hmem : 0 < memGain) (hgain : memGain < genGain)
    (hclip : 0 < bound) (he : 0 < eps) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (hp : ∀ i, 0 ≤ (state i).parameter ∧ (state i).parameter ≤ ceiling)
    (hgMass : 0 < (state 0).parameter + (state 1).parameter)
    (hmMass : 0 < (state 2).parameter + (state 3).parameter)
    (hweak : (state 0).parameter + (state 1).parameter ≤ (state 2).parameter + (state 3).parameter)
    (herror : gainPairBalancingError genGain (gainCEGradientScale remaining genGain memGain bound state) eps
      (state 0).parameter (state 1).parameter / ((state 0).parameter + (state 1).parameter) ≤
        gainBalancedRelativeGap (gainCEFeedbackFloor remaining genGain memGain bound ceiling)
          eps genGain memGain ceiling / 2) :
    let next := gainNativeStep remaining genGain memGain bound 0 0 eps decay rate state
    gainUnequalRatioGrowth (gainCEFeedbackFloor remaining genGain memGain bound ceiling)
      eps genGain memGain ceiling decay rate *
        (((state 0).parameter + (state 1).parameter) / ((state 2).parameter + (state 3).parameter)) ≤
          ((next 0).parameter + (next 1).parameter) / ((next 2).parameter + (next 3).parameter) := by
  have hg := lt_trans hmem hgain
  have hc : 0 ≤ ceiling := le_trans (hp 0).1 (hp 0).2
  have hl := gain_ce_feedback_floor_pos remaining genGain memGain bound ceiling (le_of_lt hg) hc hclip
  have hs := gain_ce_gradient_scale_box_floor remaining genGain memGain bound ceiling state
    hg hmem (le_of_lt hgain) hc hclip hp
  have hu := (gain_ce_gradient_scale_unit_interval remaining genGain memGain bound state hclip).2
  have hform := gain_native_zero_beta_pair_step remaining genGain memGain bound eps decay rate state
    hg hmem hclip (fun i => (hp i).1)
  have hmBox : (state 2).parameter + (state 3).parameter ≤ 2 * ceiling := by
    linarith only [(hp 2).2, (hp 3).2]
  have hratio := gain_pair_mass_ratio_growth_step _ _ eps genGain memGain ceiling decay rate _ _ _ _
    hl hs (le_of_lt hu) he hmem hgain heta hd (hp 0).1 (hp 1).1 (hp 2).1 (hp 3).1 hgMass hmMass hweak hmBox herror
  rw [← hform.1, ← hform.2] at hratio
  exact hratio

example :
    let state := seededNativeSubweights ((1 / 4, 1 / 4), (1, 1))
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧
    0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    (∀ i, 0 ≤ (state i).parameter ∧ (state i).parameter ≤ 2) ∧
    0 < (state 0).parameter + (state 1).parameter ∧
    0 < (state 2).parameter + (state 3).parameter ∧
    (state 0).parameter + (state 1).parameter ≤ (state 2).parameter + (state 3).parameter ∧
    gainPairBalancingError 3 (gainCEGradientScale 0 3 2 1 state) 1
      (state 0).parameter (state 1).parameter / ((state 0).parameter + (state 1).parameter) ≤
        gainBalancedRelativeGap (gainCEFeedbackFloor 0 3 2 1 2) 1 3 2 2 / 2 := by
  dsimp only
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    ?_, by norm_num [seededNativeSubweights, seededScalarState],
    by norm_num [seededNativeSubweights, seededScalarState],
    by norm_num [seededNativeSubweights, seededScalarState], ?_⟩
  · intro i
    fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]
  · have hl := gain_ce_feedback_floor_pos 0 3 2 1 2 (by norm_num) (by norm_num) (by norm_num)
    have hg : 0 < gainBalancedRelativeGap (gainCEFeedbackFloor 0 3 2 1 2) 1 3 2 2 := by
      unfold gainBalancedRelativeGap
      positivity
    norm_num [seededNativeSubweights, seededScalarState, gainPairBalancingError]
    exact le_of_lt (div_pos hg (by norm_num))

/-- The same current error budget gives a strict increase of the
unequal pair mass ratio, not merely a nondecreasing step. Sources:
appendix C efficiency and actual-form native comparison at d655756;
this does not assert a finite crossing or future successful tail. -/
theorem gain_pair_mass_ratio_strict_step (lower scale eps genGain memGain ceiling decay rate a b c d : ℝ)
    (hlower : 0 < lower) (hscale : lower ≤ scale) (hunit : scale ≤ 1) (he : 0 < eps)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hdseed : 0 ≤ d)
    (hgMass : 0 < a + b) (hmMass : 0 < c + d) (hweak : a + b ≤ c + d)
    (hceiling : c + d ≤ 2 * ceiling)
    (herror : gainPairBalancingError genGain scale eps a b / (a + b) ≤
      gainBalancedRelativeGap lower eps genGain memGain ceiling / 2) :
    (a + b) / (c + d) <
      ((gainPairStep genGain scale eps decay rate a b).1 + (gainPairStep genGain scale eps decay rate a b).2) /
        ((gainPairStep memGain scale eps decay rate c d).1 + (gainPairStep memGain scale eps decay rate c d).2) := by
  have hbound := gain_pair_mass_ratio_growth_step lower scale eps genGain memGain ceiling decay rate a b c d
    hlower hscale hunit he hmem hgain heta hd ha hb hc hdseed hgMass hmMass hweak hceiling herror
  have hbox : 0 ≤ ceiling := by linarith only [hmMass, hceiling]
  have hq := gain_unequal_ratio_growth_one_lt lower eps genGain memGain ceiling decay rate
    hlower he hmem hgain hbox heta hd
  have hstrict := mul_lt_mul_of_pos_right hq (div_pos hgMass hmMass)
  simpa only [one_mul] using lt_of_lt_of_le hstrict hbound

example : (0 : ℝ) < 1 / 10 ∧ (1 / 10 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 / 1000 ∧
    0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    (0 : ℝ) ≤ 1 / 4 ∧ (0 : ℝ) ≤ 1 / 4 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    (0 : ℝ) < 1 / 4 + 1 / 4 ∧ (0 : ℝ) < 1 + 1 ∧ (1 / 4 : ℝ) + 1 / 4 ≤ 1 + 1 ∧
    (1 : ℝ) + 1 ≤ 2 * 2 ∧
    gainPairBalancingError 3 (1 / 2) 1 (1 / 4) (1 / 4) / (1 / 4 + 1 / 4) ≤
      gainBalancedRelativeGap (1 / 10) 1 3 2 2 / 2 := by
  norm_num [gainPairBalancingError, gainBalancedRelativeGap]

end Transformer.Grokking.CircuitEfficiency
