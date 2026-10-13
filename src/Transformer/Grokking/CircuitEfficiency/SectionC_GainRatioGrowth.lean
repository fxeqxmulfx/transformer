import Transformer.Grokking.CircuitEfficiency.SectionC_GainRelativeRate

/-!
# Uniform multiplicative ratio growth before the native balance crossing

Sources: Varma et al., arXiv:2309.02390v1, section 3's competing
circuits and appendix C's product CE; actual zero-beta amplitude
reduction at lab commit efe2dd3. Convert the checked positive
relative-rate gap into a factor greater than one for Gen/Mem.

The native epsilon ceiling bounds the current Mem multiplier.
Divide the rate gap by that ceiling and keep positive remaining
decay. This produces a uniform ratio increment while Gen is no
stronger, even when the actual shared CE scale changes each step.
The scale floor and finite box will be derived from initialization
on the same actual path, not assumed as a separate input stream.

Equal positive pair factors and legal zero betas specialize the
source's unbalanced seeds and the preserved nonzero-beta GPTMini
settings. Fixed physical readouts and uniform decoupled native
decay differ from appendix C's coupled assigned circuit norm/GD.
The finite-time path crossing and persistent held-out ordering
remain subsequent obligations; this file proves one actual-form step.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- Uniform candidate ratio multiplier from the checked native rate
gap and Mem multiplier ceiling. Sources: appendix C's gained CE and
native normalization at efe2dd3; no parameter path is defined here. -/
noncomputable def gainBalancedRatioGrowth (lower eps genGain memGain ceiling decay rate : ℝ) : ℝ :=
  1 + rate * gainBalancedRelativeGap lower eps genGain memGain ceiling /
    (1 - rate * decay + rate * memGain / eps)

/-- The derived numerical ratio multiplier is strictly greater than
one. Sources: actual native rate estimates at efe2dd3; the conditions
concern fixed constants and contain no future limit or crossing. -/
theorem gain_balanced_ratio_growth_one_lt (lower eps genGain memGain ceiling decay rate : ℝ)
    (hlower : 0 < lower) (he : 0 < eps) (hmem : 0 < memGain) (hgain : memGain < genGain)
    (hc : 0 ≤ ceiling) (heta : 0 < rate) (hd : 0 < 1 - rate * decay) :
    1 < gainBalancedRatioGrowth lower eps genGain memGain ceiling decay rate := by
  have hg : 0 < genGain := lt_trans hmem hgain
  have hgap : 0 < gainBalancedRelativeGap lower eps genGain memGain ceiling := by
    unfold gainBalancedRelativeGap
    positivity
  have hden : 0 < 1 - rate * decay + rate * memGain / eps := by positivity
  have hpart := div_pos (mul_pos heta hgap) hden
  unfold gainBalancedRatioGrowth
  linarith only [hpart]

example : (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧
    (0 : ℝ) ≤ 2 ∧ (0 : ℝ) < 1 / 1000 ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) := by
  norm_num

/-- The actual-form Mem factor multiplier is positive and below its
fixed epsilon ceiling. Source: the native zero-beta relative-rate
formula at efe2dd3; scale and factor hypotheses describe the current point. -/
theorem gain_balanced_mem_multiplier_interval (gain scale eps decay rate factor : ℝ)
    (hg : 0 ≤ gain) (hs : 0 ≤ scale) (hunit : scale ≤ 1) (he : 0 < eps)
    (heta : 0 ≤ rate) (hd : 0 < 1 - rate * decay) (hf : 0 ≤ factor) :
    0 < 1 - rate * decay + rate * gainBalancedRelativeRate gain scale eps factor ∧
      1 - rate * decay + rate * gainBalancedRelativeRate gain scale eps factor ≤
        1 - rate * decay + rate * gain / eps := by
  have hn := gain_balanced_relative_rate_nonneg gain scale eps factor hg hs he hf
  have hm := mul_nonneg heta hn
  have hc := gain_balanced_relative_rate_ceiling gain scale eps factor hg hs hunit he hf
  have hu := mul_le_mul_of_nonneg_left hc heta
  constructor
  · linarith only [hd, hm]
  · have heq : rate * (gain / eps) = rate * gain / eps := by ring
    rw [heq] at hu
    linarith only [hu]

example : (0 : ℝ) ≤ 2 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧ (0 : ℝ) ≤ 1 := by
  norm_num

/-- The actual-form next amplitude ratio gains the fixed factor while
Gen has not caught Mem. Sources: appendix C's gained CE and the proved
native zero-beta recurrence at efe2dd3; the current scale may vary
between applications and no future positive reference is supplied. -/
theorem gain_balanced_ratio_growth_step (lower scale eps genGain memGain ceiling decay rate a b : ℝ)
    (hlower : 0 < lower) (hscale : lower ≤ scale) (hunit : scale ≤ 1) (he : 0 < eps)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (ha : 0 ≤ a) (hb : 0 < b) (hab : a ≤ b) (hceiling : b ≤ ceiling) :
    gainBalancedRatioGrowth lower eps genGain memGain ceiling decay rate * (a / b) ≤
      gainBalancedFactorStep genGain scale eps decay rate a /
        gainBalancedFactorStep memGain scale eps decay rate b := by
  have hs : 0 < scale := lt_of_lt_of_le hlower hscale
  let gap := gainBalancedRelativeGap lower eps genGain memGain ceiling
  let genRate := gainBalancedRelativeRate genGain scale eps a
  let memRate := gainBalancedRelativeRate memGain scale eps b
  let ceilingRate := 1 - rate * decay + rate * memGain / eps
  have hgap := gain_balanced_relative_gap_floor lower scale eps genGain memGain ceiling a b
    hlower hscale hunit he hmem hgain ha hab hceiling
  have hm := gain_balanced_mem_multiplier_interval memGain scale eps decay rate b
    (le_of_lt hmem) (le_of_lt hs) hunit he (le_of_lt heta) hd (le_of_lt hb)
  have hupper : 0 < ceilingRate := by
    dsimp only [ceilingRate]
    positivity
  have hgapPos : 0 < gap := hgap.1
  have hpart : 0 ≤ rate * gap / ceilingRate := by positivity
  have hweighted := mul_le_mul_of_nonneg_left hm.2 hpart
  have hcancel := div_mul_cancel₀ (rate * gap) (ne_of_gt hupper)
  have hrate := mul_le_mul_of_nonneg_left hgap.2 (le_of_lt heta)
  have hmultiplier : gainBalancedRatioGrowth lower eps genGain memGain ceiling decay rate *
      (1 - rate * decay + rate * memRate) ≤ 1 - rate * decay + rate * genRate := by
    change (1 + rate * gap / ceilingRate) * (1 - rate * decay + rate * memRate) ≤
      1 - rate * decay + rate * genRate
    change rate * gap / ceilingRate * (1 - rate * decay + rate * memRate) ≤
      rate * gap / ceilingRate * ceilingRate at hweighted
    change rate * gap ≤ rate * (genRate - memRate) at hrate
    nlinarith only [hweighted, hcancel, hrate]
  rw [gain_balanced_factor_step_multiplier, gain_balanced_factor_step_multiplier]
  have hden : 0 < b * (1 - rate * decay + rate * memRate) := mul_pos hb hm.1
  apply (le_div_iff₀ hden).mpr
  have hgroup : gainBalancedRatioGrowth lower eps genGain memGain ceiling decay rate * (a / b) *
      (b * (1 - rate * decay + rate * memRate)) =
      a * (gainBalancedRatioGrowth lower eps genGain memGain ceiling decay rate *
        (1 - rate * decay + rate * memRate)) := by
    field_simp [ne_of_gt hb]
  rw [hgroup]
  exact mul_le_mul_of_nonneg_left hmultiplier ha

example : (0 : ℝ) < 1 / 10 ∧ (1 / 10 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 / 1000 ∧
    0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧ (0 : ℝ) ≤ 1 / 4 ∧ (0 : ℝ) < 1 ∧
    (1 / 4 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 2 := by
  norm_num

/-- The same ratio growth holds for the actual native step and its
actual current CE. Sources: appendix C's true partials and retained
native AdamW at efe2dd3; derive scale bounds from the present physical
box, leaving no separately supplied callback or future success premise. -/
theorem gain_native_balanced_ratio_growth (remaining : ℕ) (genGain memGain bound ceiling eps decay rate : ℝ)
    (state : Transformer.Grokking.CircuitEfficiency.NativeSubweightState)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hclip : 0 < bound)
    (he : 0 < eps) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (hp : ∀ i, 0 ≤ (state i).parameter ∧ (state i).parameter ≤ ceiling)
    (hs : NativePairSymmetry state) (hb : 0 < (state 2).parameter)
    (hweak : (state 0).parameter ≤ (state 2).parameter) :
    gainBalancedRatioGrowth (gainCEFeedbackFloor remaining genGain memGain bound ceiling)
      eps genGain memGain ceiling decay rate * ((state 0).parameter / (state 2).parameter) ≤
        (gainNativeStep remaining genGain memGain bound 0 0 eps decay rate state 0).parameter /
          (gainNativeStep remaining genGain memGain bound 0 0 eps decay rate state 2).parameter := by
  have hg : 0 < genGain := lt_trans hmem hgain
  have hc : 0 ≤ ceiling := le_trans (hp 0).1 (hp 0).2
  have hl := gain_ce_feedback_floor_pos remaining genGain memGain bound ceiling (le_of_lt hg) hc hclip
  have hslope := gain_ce_gradient_scale_box_floor remaining genGain memGain bound ceiling state
    hg hmem (le_of_lt hgain) hc hclip hp
  have hu := (gain_ce_gradient_scale_unit_interval remaining genGain memGain bound state hclip).2
  have hstep := gain_native_zero_beta_balanced_step remaining genGain memGain bound eps decay rate state
    hg hmem hclip (fun i => (hp i).1) hs
  rw [hstep.1, hstep.2]
  exact gain_balanced_ratio_growth_step _ _ eps genGain memGain ceiling decay rate _ _
    hl hslope (le_of_lt hu) he hmem hgain heta hd (hp 0).1 hb hweak (hp 2).2

example :
    let state := seededNativeSubweights ((1 / 4, 1 / 4), (1, 1))
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧
    0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    (∀ i, 0 ≤ (state i).parameter ∧ (state i).parameter ≤ 2) ∧ NativePairSymmetry state ∧
    0 < (state 2).parameter ∧ (state 0).parameter ≤ (state 2).parameter := by
  dsimp only
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    ?_, ⟨rfl, rfl⟩, by norm_num [seededNativeSubweights, Transformer.Grokking.AdamW.seededScalarState],
    by norm_num [seededNativeSubweights, Transformer.Grokking.AdamW.seededScalarState]⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, Transformer.Grokking.AdamW.seededScalarState]

end Transformer.Grokking.CircuitEfficiency
