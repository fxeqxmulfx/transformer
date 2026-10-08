import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdRegime
import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryMass

/-!
# Actual clipped CE feedback has its cold value as a global ceiling

Sources: Varma et al., arXiv:2309.02390v1, section 3's CE/decay
competition and appendix C product derivatives; original native
clipping at ca253e4 and complete retained masses at ada36f3.

Nonnegative physical gains and factors make the true train score
nonnegative. Its finite-class CE slope is then at most the uniform
initial slope. The actual clipping multiplier is at most its value
at zero raw norm, including the source's additive 1e-6 strip.
Consequently the computed cold CE magnitude is a global upper bound
on the actual shared coefficient throughout this numerical region.

Use this sharper value in the true total retained first-moment
recurrence and in every actual applied-input ceiling. Greater Gen
gain times cold coefficient replaces the previous looser gain-only
constant. These are current generated-callback bounds, not a prescribed
gradient stream, a future cold limit or an invented optimizer.

The attained cold value prepares an exact retained collapse threshold.
Contraction still requires the actual completed-clock denominator floor
and a suitable parameter/moment weight. No stability, phase classification,
confidence selection or learned GPTMini transfer is proved here.
Fixed gained tables/plain CE/uniform native decay differ from the
source's coupled-cost GD; a thermodynamic system-size limit is separate.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Nonnegative physical factors and gains give nonnegative actual
train score. Sources: appendix C product logits and gained physical
forward at 2150464; buffers do not enter this current numerical law. -/
theorem gain_native_nonnegative_total_score (genGain memGain : ℝ) (state : NativeSubweightState)
    (hgen : 0 ≤ genGain) (hmem : 0 ≤ memGain) (hp : ∀ i, 0 ≤ (state i).parameter) :
    0 ≤ gainNativeTotalScore genGain memGain state := by
  unfold gainNativeTotalScore physicalCircuitScore
  exact add_nonneg (mul_nonneg hgen (mul_nonneg (hp 0) (hp 1)))
    (mul_nonneg hmem (mul_nonneg (hp 2) (hp 3)))

example : (0 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 2 ∧
    ∀ i, 0 ≤ (seededNativeSubweights ((0, 1 / 200), (0, 1)) i).parameter := by
  refine ⟨by norm_num, by norm_num, ?_⟩
  intro i
  exact (native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) i).1

/-- The computed cold coefficient bounds the original shared clipped
CE magnitude at every nonnegative physical point. Sources: appendix C
uniform CE slope and native norm-plus-1e-6 clipping at ca253e4;
the upper bound includes the actual gain-dependent raw gradient norm. -/
theorem gain_ce_gradient_scale_cold_ceiling (remaining : ℕ) (genGain memGain bound : ℝ)
    (state : NativeSubweightState) (hgen : 0 ≤ genGain) (hmem : 0 ≤ memGain)
    (hclip : 0 < bound) (hp : ∀ i, 0 ≤ (state i).parameter) :
    gainCEGradientScale remaining genGain memGain bound state ≤ coldGainCEGradientScale remaining bound := by
  have hscore := gain_native_nonnegative_total_score genGain memGain state hgen hmem hp
  have hexp := Real.exp_le_exp.mpr hscore
  rw [Real.exp_zero] at hexp
  have hslope : ((remaining : ℝ) + 1) /
      (Real.exp (gainNativeTotalScore genGain memGain state) + (remaining : ℝ) + 1) ≤
        ((remaining : ℝ) + 1) / ((remaining : ℝ) + 2) :=
    div_le_div_of_nonneg_left (by positivity) (by positivity) (by linarith only [hexp])
  have hnorm : 0 ≤ coordinateGradientNorm (rawGainNativeGradient remaining genGain memGain state) :=
    Real.sqrt_nonneg _
  have hstrip := div_le_div_of_nonneg_left (le_of_lt hclip) (by norm_num : (0 : ℝ) < 1 / 1000000)
    (show (1 / 1000000 : ℝ) ≤ coordinateGradientNorm (rawGainNativeGradient remaining genGain memGain state) + 1 / 1000000 by
      linarith only [hnorm])
  have hfactor : coordinateClipFactor bound (rawGainNativeGradient remaining genGain memGain state) ≤
      min 1 (bound / (1 / 1000000)) := by
    unfold coordinateClipFactor
    exact le_min (min_le_left _ _) (le_trans (min_le_right _ _) hstrip)
  have hcold : 0 ≤ min 1 (bound / (1 / 1000000)) := le_of_lt (lt_min (by norm_num) (div_pos hclip (by norm_num)))
  unfold gainCEGradientScale coldGainCEGradientScale
  exact mul_le_mul hfactor hslope (by positivity) hcold

example : (0 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    ∀ i, 0 ≤ (seededNativeSubweights ((0, 1 / 200), (0, 1)) i).parameter := by
  refine ⟨by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  exact (native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) i).1

/-- The exact cold ceiling sharpens actual total first-moment
feedback. Sources: appendix C true partner derivatives and retained
native insertion at ada36f3; old retained moments remain in the law. -/
theorem gain_native_cold_moment_mass_ceiling (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (h1 : b1 ≤ 1) (hs : NonnegativeNativeState state) :
    gainNativeNegativeMomentMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) ≤
      b1 * gainNativeNegativeMomentMass state + (1 - b1) *
        (coldGainCEGradientScale remaining bound * genGain) * gainNativeParameterMass state := by
  have hp : ∀ i, 0 ≤ (state i).parameter := fun i => (hs i).1
  have hGen := add_nonneg (hp 0) (hp 1)
  have hMem := add_nonneg (hp 2) (hp 3)
  have hweighted := add_nonneg (mul_nonneg (le_of_lt hgen) hGen) (mul_nonneg (le_of_lt hmem) hMem)
  have hgap := mul_nonneg (show 0 ≤ genGain - memGain by linarith only [hgain]) hMem
  have hweights : genGain * ((state 0).parameter + (state 1).parameter) +
      memGain * ((state 2).parameter + (state 3).parameter) ≤ genGain * gainNativeParameterMass state := by
    unfold gainNativeParameterMass
    nlinarith only [hgap]
  have hcold := gain_ce_gradient_scale_cold_ceiling remaining genGain memGain bound state (le_of_lt hgen) (le_of_lt hmem) hclip hp
  have hh := mul_le_mul hcold hweights hweighted
    (le_of_lt (cold_gain_ce_gradient_scale_pos remaining bound hclip))
  have hscaled := mul_le_mul_of_nonneg_left hh (show 0 ≤ 1 - b1 by linarith only [h1])
  rw [gain_native_moment_mass_step]
  nlinarith only [hscaled]

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (9 / 10 : ℝ) ≤ 1 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

/-- Every actual negative coordinate input is bounded by cold
coefficient times greater gain times present total parameter mass.
Sources: appendix C applied partner CE and native clipping at ca253e4;
no future input convergence or instantaneous buffer match is supplied. -/
theorem gain_native_cold_applied_gradient_mass_cap (remaining : ℕ) (genGain memGain bound : ℝ)
    (state : NativeSubweightState) (i : Fin 4) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (hs : NonnegativeNativeState state) :
    0 ≤ -appliedGainNativeGradient remaining genGain memGain bound state i ∧
      -appliedGainNativeGradient remaining genGain memGain bound state i ≤
        (coldGainCEGradientScale remaining bound * genGain) * gainNativeParameterMass state := by
  have hp := (hs (nativeFactorPartner i)).1
  have hg := le_of_lt (gain_native_factor_gain_pos genGain memGain hgen hmem i)
  have hi : nativeFactorGain genGain memGain i ≤ genGain := by
    fin_cases i
    · exact le_rfl
    · exact le_rfl
    · exact hgain
    · exact hgain
  have hcold := gain_ce_gradient_scale_cold_ceiling remaining genGain memGain bound state
    (le_of_lt hgen) (le_of_lt hmem) hclip (fun j => (hs j).1)
  have hfirst := mul_le_mul_of_nonneg_right hcold (mul_nonneg hg hp)
  have hgainNow := mul_le_mul_of_nonneg_right hi hp
  have hmassNow := mul_le_mul_of_nonneg_left
    ((gain_native_masses_nonnegative state hs).2.2 (nativeFactorPartner i)) (le_of_lt hgen)
  have hsecond := mul_le_mul_of_nonneg_left (le_trans hgainNow hmassNow)
    (le_of_lt (cold_gain_ce_gradient_scale_pos remaining bound hclip))
  rw [gain_native_applied_gradient_scale, neg_neg]
  refine ⟨mul_nonneg (mul_nonneg
    (le_of_lt (gain_ce_gradient_scale_pos remaining genGain memGain bound state hclip)) hg) hp, ?_⟩
  simpa only [mul_assoc] using le_trans hfirst hsecond

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

end Transformer.Grokking.CircuitEfficiency

