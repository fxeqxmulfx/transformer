import Transformer.Grokking.CircuitEfficiency.SectionC_GainPairMass
import Transformer.Grokking.CircuitEfficiency.SectionC_GainPairContraction

/-!
# Actual current native relative-asymmetry contraction

Sources: Varma et al., arXiv:2309.02390v1, section 3's unbalanced
factor seeds and appendix C's product CE; native feedback at lab
commit 620ad1a. Derive the shared current input floor from the actual
full physical box, including the common norm-two clipping factor.

Both actual pair sums dominate one positive Mem-based lower
multiplier. Its ratio to pure decay is strictly below one at positive
rate. At the sufficient fixed small rate, both actual relative pair
differences contract by this common ratio in one native step.
No independent callback, future pair balance or parameter limit is
assumed. The initialized box must still be generated on the same
trajectory before this pointwise estimate can be iterated.

This exact-real fixed-table, legal zero-beta property does not prove
positive limiting mass, loss convergence, Gen/Mem allocation or a
learned-transformer detector. Uniform decoupled native decay differs
from appendix C's coupled-cost GD and preserved GPTMini settings.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Relative physical pair difference, including the totalized zero
denominator value. Sources: appendix C's two factors at 620ad1a;
all uses requiring positive mass state and prove it explicitly. -/
noncomputable def gainPairRelativeDifference (a b : ℝ) : ℝ := |a - b| / (a + b)

/-- Uniform relative-difference ratio from the actual box CE floor
and lower-efficiency Mem mass multiplier. Sources: appendix C true
feedback and native estimates at 620ad1a; no path is prescribed. -/
noncomputable def gainNativeRelativeRatio (remaining : ℕ)
    (genGain memGain bound ceiling eps decay rate : ℝ) : ℝ :=
  (1 - rate * decay) / gainPairMassMultiplier
    (gainCEFeedbackFloor remaining genGain memGain bound ceiling) memGain eps decay rate ceiling

/-- The same actual CE/box lower multiplier bounds both unequal pair
sums. Sources: appendix C's physical product partials and native
step at 620ad1a; every scale premise is derived from actual feedback. -/
theorem gain_native_pair_mass_floor (remaining : ℕ) (genGain memGain bound ceiling eps decay rate : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (he : 0 < eps) (heta : 0 ≤ rate)
    (hp : ∀ i, 0 ≤ (state i).parameter ∧ (state i).parameter ≤ ceiling) :
    let next := gainNativeStep remaining genGain memGain bound 0 0 eps decay rate state
    let multiplier := gainPairMassMultiplier
      (gainCEFeedbackFloor remaining genGain memGain bound ceiling) memGain eps decay rate ceiling
    multiplier * ((state 0).parameter + (state 1).parameter) ≤ (next 0).parameter + (next 1).parameter ∧
      multiplier * ((state 2).parameter + (state 3).parameter) ≤ (next 2).parameter + (next 3).parameter := by
  have hc : 0 ≤ ceiling := le_trans (hp 0).1 (hp 0).2
  have hl := gain_ce_feedback_floor_pos remaining genGain memGain bound ceiling (le_of_lt hgen) hc hclip
  have hs := gain_ce_gradient_scale_box_floor remaining genGain memGain bound ceiling state hgen hmem hgain hc hclip hp
  have hu := (gain_ce_gradient_scale_unit_interval remaining genGain memGain bound state hclip).2
  have hform := gain_native_zero_beta_pair_step remaining genGain memGain bound eps decay rate state
    hgen hmem hclip (fun i => (hp i).1)
  have hg := gain_pair_step_mass_floor _ _ genGain eps decay rate ceiling _ _
    hl hs (le_of_lt hu) hgen he heta (hp 0).1 (hp 1).1 (hp 0).2 (hp 1).2
  have hm := gain_pair_step_mass_floor _ _ memGain eps decay rate ceiling _ _
    hl hs (le_of_lt hu) hmem he heta (hp 2).1 (hp 3).1 (hp 2).2 (hp 3).2
  have horder := gain_pair_mass_multiplier_gain_mono _ genGain memGain eps decay rate ceiling
    (le_of_lt hl) (le_of_lt hmem) hgain he heta hc
  rw [← hform.1] at hg
  rw [← hform.2] at hm
  have hsum : 0 ≤ (state 0).parameter + (state 1).parameter := by
    linarith only [(hp 0).1, (hp 1).1]
  exact ⟨le_trans
    (mul_le_mul_of_nonneg_right horder hsum) hg, hm⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ ∀ i : Fin 4,
      0 ≤ (seededNativeSubweights ((0, 1 / 200), (0, 1)) i).parameter ∧
        (seededNativeSubweights ((0, 1 / 200), (0, 1)) i).parameter ≤ 2 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- The actual-box relative-asymmetry multiplier lies in [0,1).
Source: actual clipped CE floor and native mass normalization at
620ad1a; its positive denominator is derived from static constants. -/
theorem gain_native_relative_ratio_interval (remaining : ℕ) (genGain memGain bound ceiling eps decay rate : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound) (hc : 0 ≤ ceiling)
    (he : 0 < eps) (heta : 0 < rate) (hd : 0 ≤ 1 - rate * decay) :
    0 ≤ gainNativeRelativeRatio remaining genGain memGain bound ceiling eps decay rate ∧
      gainNativeRelativeRatio remaining genGain memGain bound ceiling eps decay rate < 1 := by
  have hl := gain_ce_feedback_floor_pos remaining genGain memGain bound ceiling (le_of_lt hgen) hc hclip
  have hmult := gain_pair_mass_multiplier_interval _ memGain eps decay rate ceiling hl hmem he heta hc hd
  exact hmult.2.2

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) := by
  norm_num

/-- Actual unequal native pairs with positive current masses both
contract relative asymmetry by the common fixed box ratio. Sources:
section 3's compositional seeds and true native CE at 620ad1a;
current scale/sign/denominator conditions are derived, not supplied. -/
theorem gain_native_relative_difference_step (remaining : ℕ) (genGain memGain bound ceiling eps decay rate : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (he : 0 < eps) (heta : 0 < rate)
    (hp : ∀ i, 0 ≤ (state i).parameter ∧ (state i).parameter ≤ ceiling)
    (hgMass : 0 < (state 0).parameter + (state 1).parameter)
    (hmMass : 0 < (state 2).parameter + (state 3).parameter)
    (hsmall : rate * (decay + genGain / eps) ≤ 1) :
    let next := gainNativeStep remaining genGain memGain bound 0 0 eps decay rate state
    let q := gainNativeRelativeRatio remaining genGain memGain bound ceiling eps decay rate
    gainPairRelativeDifference (next 0).parameter (next 1).parameter ≤
        q * gainPairRelativeDifference (state 0).parameter (state 1).parameter ∧
      gainPairRelativeDifference (next 2).parameter (next 3).parameter ≤
        q * gainPairRelativeDifference (state 2).parameter (state 3).parameter := by
  have hpart := mul_nonneg (le_of_lt heta) (div_nonneg (le_of_lt hgen) (le_of_lt he))
  have hd : 0 ≤ 1 - rate * decay := by nlinarith only [hsmall, hpart]
  have hc : 0 ≤ ceiling := le_trans (hp 0).1 (hp 0).2
  have hl := gain_ce_feedback_floor_pos remaining genGain memGain bound ceiling (le_of_lt hgen) hc hclip
  have hs := gain_ce_gradient_scale_box_floor remaining genGain memGain bound ceiling state hgen hmem hgain hc hclip hp
  have hu := (gain_ce_gradient_scale_unit_interval remaining genGain memGain bound state hclip).2
  have hform := gain_native_zero_beta_pair_step remaining genGain memGain bound eps decay rate state
    hgen hmem hclip (fun i => (hp i).1)
  have hmSmall : rate * (decay + memGain / eps) ≤ 1 := by
    have hweighted := mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right hgain (le_of_lt he)) (le_of_lt heta)
    nlinarith only [hweighted, hsmall]
  have hg := gain_pair_step_relative_difference _ _ genGain eps decay rate ceiling _ _ hl hs (le_of_lt hu)
    hgen he heta hd (hp 0).1 (hp 1).1 (hp 0).2 (hp 1).2 hgMass hsmall
  have hm := gain_pair_step_relative_difference _ _ memGain eps decay rate ceiling _ _ hl hs (le_of_lt hu)
    hmem he heta hd (hp 2).1 (hp 3).1 (hp 2).2 (hp 3).2 hmMass hmSmall
  have hmult := gain_pair_mass_multiplier_interval _ memGain eps decay rate ceiling hl hmem he heta hc hd
  have horder := gain_pair_mass_multiplier_gain_mono _ genGain memGain eps decay rate ceiling
    (le_of_lt hl) (le_of_lt hmem) hgain he (le_of_lt heta) hc
  have hratio := div_le_div_of_nonneg_left hd hmult.1 horder
  rw [← hform.1] at hg
  rw [← hform.2] at hm
  exact ⟨le_trans hg (mul_le_mul_of_nonneg_right hratio (div_nonneg (abs_nonneg _) (le_of_lt hgMass))), hm⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ (∀ i : Fin 4,
      0 ≤ (seededNativeSubweights ((0, 1 / 200), (0, 1)) i).parameter ∧
        (seededNativeSubweights ((0, 1 / 200), (0, 1)) i).parameter ≤ 2) ∧
    0 < (seededNativeSubweights ((0, 1 / 200), (0, 1)) 0).parameter + (seededNativeSubweights ((0, 1 / 200), (0, 1)) 1).parameter ∧
    0 < (seededNativeSubweights ((0, 1 / 200), (0, 1)) 2).parameter + (seededNativeSubweights ((0, 1 / 200), (0, 1)) 3).parameter ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_,
    by norm_num [seededNativeSubweights, seededScalarState], by norm_num [seededNativeSubweights, seededScalarState], by norm_num⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

end Transformer.Grokking.CircuitEfficiency
