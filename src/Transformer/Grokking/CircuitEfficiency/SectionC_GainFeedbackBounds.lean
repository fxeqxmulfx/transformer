import Transformer.Grokking.CircuitEfficiency.SectionC_GainBounds

/-!
# Actual raw CE and logit ceilings from a physical parameter box

Sources: Varma et al., arXiv:2309.02390v1, appendix C's two-factor
multiclass CE, and the gained native feedback at lab commit 59aff2c.
Bound the actual raw coordinate inputs, their full norm and the
current train score on nonnegative physical factors with a finite
common ceiling. The Gen readout is at least the Mem readout.

These are consequences of the true current product CE, not an
independently chosen gradient stream or a future convergence
assumption. Retained moments and clocks are unrestricted here:
they do not enter this current-parameter callback. The separate
native bound theorem supplies such a box on initialized paths.

These bounds will give a uniform positive lower bound on the actual
clipped CE scale on the same generated path. A finite box alone
does not establish convergence; the checked native period-two
example remains compatible with these estimates. Physical gains,
fixed tables and uniform decoupled AdamW differ from appendix C's
assigned coupled norm cost and from learned GPTMini features.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Actual raw coordinate inputs lie between minus the Gen ceiling
and zero. Sources: appendix C's exact CE chain rule and the gained
forward at 59aff2c; all four present physical factors enter the callback. -/
theorem gain_native_raw_coordinate_box (remaining : ℕ) (genGain memGain ceiling : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (horder : memGain ≤ genGain)
    (hp : ∀ i, 0 ≤ (state i).parameter ∧ (state i).parameter ≤ ceiling) :
    ∀ i, -genGain * ceiling ≤ rawGainNativeGradient remaining genGain memGain state i ∧
      rawGainNativeGradient remaining genGain memGain state i ≤ 0 := by
  let slope := ((remaining : ℝ) + 1) /
    (Real.exp (gainNativeTotalScore genGain memGain state) + (remaining : ℝ) + 1)
  have hs : 0 ≤ slope := by dsimp only [slope]; positivity
  have hden : 0 < Real.exp (gainNativeTotalScore genGain memGain state) + (remaining : ℝ) + 1 := by positivity
  have hu : slope ≤ 1 := by
    apply (div_le_iff₀ hden).mpr
    have hexp := Real.exp_pos (gainNativeTotalScore genGain memGain state)
    linarith only [hexp]
  intro i
  have hg : 0 ≤ nativeFactorGain genGain memGain i :=
    le_of_lt (gain_native_factor_gain_pos genGain memGain hgen hmem i)
  have hgain : nativeFactorGain genGain memGain i ≤ genGain := by
    fin_cases i
    · exact le_rfl
    · exact le_rfl
    · exact horder
    · exact horder
  have hproduct : 0 ≤ nativeFactorGain genGain memGain i * (state (nativeFactorPartner i)).parameter :=
    mul_nonneg hg (hp _).1
  have hceiling := mul_le_mul hgain (hp (nativeFactorPartner i)).2 (hp _).1 (le_of_lt hgen)
  have hscaled := mul_le_mul_of_nonneg_right hu hproduct
  have hnonneg := mul_nonneg hs hproduct
  have heq : rawGainNativeGradient remaining genGain memGain state i =
      -(slope * (nativeFactorGain genGain memGain i * (state (nativeFactorPartner i)).parameter)) := by
    unfold rawGainNativeGradient
    dsimp only [slope]
    ring
  rw [heq]
  constructor <;> nlinarith only [hceiling, hscaled, hnonneg]

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧
    ∀ i : Fin 4, 0 ≤ (seededNativeSubweights ((1, 2), (3, 4)) i).parameter ∧
      (seededNativeSubweights ((1, 2), (3, 4)) i).parameter ≤ 100 := by
  refine ⟨by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- The actual raw full norm is at most twice the Gen factor ceiling.
Sources: appendix C's four partial derivatives and native norm-two
clipping at 59aff2c; this is the norm before any shared multiplier. -/
theorem gain_native_raw_norm_box (remaining : ℕ) (genGain memGain ceiling : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (horder : memGain ≤ genGain) (hc : 0 ≤ ceiling)
    (hp : ∀ i, 0 ≤ (state i).parameter ∧ (state i).parameter ≤ ceiling) :
    coordinateGradientNorm (rawGainNativeGradient remaining genGain memGain state) ≤ 2 * genGain * ceiling := by
  let gradient := rawGainNativeGradient remaining genGain memGain state
  have hb : 0 ≤ genGain * ceiling := mul_nonneg (le_of_lt hgen) hc
  have hsq : ∀ i, gradient i ^ 2 ≤ (genGain * ceiling) ^ 2 := by
    intro i
    have h := gain_native_raw_coordinate_box remaining genGain memGain ceiling state hgen hmem horder hp i
    change -genGain * ceiling ≤ gradient i ∧ gradient i ≤ 0 at h
    have hleft : 0 ≤ gradient i + genGain * ceiling := by nlinarith only [h.1]
    have hright : 0 ≤ genGain * ceiling - gradient i := by linarith only [hb, h.2]
    have hm := mul_nonneg hleft hright
    nlinarith only [hm]
  have hsum : (∑ i : Fin 4, gradient i ^ 2) ≤ (2 * genGain * ceiling) ^ 2 := by
    norm_num [Fin.sum_univ_succ] at ⊢
    nlinarith only [hsq 0, hsq 1, hsq 2, hsq 3]
  have hs := Real.sqrt_le_sqrt hsum
  have hbound : 0 ≤ 2 * genGain * ceiling := by positivity
  simpa only [coordinateGradientNorm, Real.sqrt_sq_eq_abs, abs_of_nonneg hbound] using hs

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 100 ∧
    ∀ i : Fin 4, 0 ≤ (seededNativeSubweights ((1, 2), (3, 4)) i).parameter ∧
      (seededNativeSubweights ((1, 2), (3, 4)) i).parameter ≤ 100 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- Actual gained train logits have a finite ceiling on the same
physical box. Source: appendix C's product forward with section 3's
efficiency represented by positive physical readouts at 59aff2c. -/
theorem gain_native_total_score_box (genGain memGain ceiling : ℝ)
    (state : NativeSubweightState) (hgen : 0 ≤ genGain) (hmem : 0 ≤ memGain)
    (hc : 0 ≤ ceiling)
    (hp : ∀ i, 0 ≤ (state i).parameter ∧ (state i).parameter ≤ ceiling) :
    gainNativeTotalScore genGain memGain state ≤ (genGain + memGain) * ceiling ^ 2 := by
  have hgpair := mul_le_mul (hp 0).2 (hp 1).2 (hp 1).1 hc
  have hmpair := mul_le_mul (hp 2).2 (hp 3).2 (hp 3).1 hc
  have hg := mul_le_mul_of_nonneg_left hgpair hgen
  have hm := mul_le_mul_of_nonneg_left hmpair hmem
  unfold gainNativeTotalScore physicalCircuitScore
  nlinarith only [hg, hm]

example : (0 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 2 ∧ (0 : ℝ) ≤ 100 ∧
    ∀ i : Fin 4, 0 ≤ (seededNativeSubweights ((1, 2), (3, 4)) i).parameter ∧
      (seededNativeSubweights ((1, 2), (3, 4)) i).parameter ≤ 100 := by
  refine ⟨by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- Native shared clipping has a numerical lower bound
on a finite nonnegative physical box. Sources: appendix C's true CE
and native clip_grad_norm_ at 59aff2c, retaining norm plus 1e-6. -/
theorem gain_native_clip_factor_box (remaining : ℕ) (genGain memGain bound ceiling : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (horder : memGain ≤ genGain) (hc : 0 ≤ ceiling) (hb : 0 ≤ bound)
    (hp : ∀ i, 0 ≤ (state i).parameter ∧ (state i).parameter ≤ ceiling) :
    min 1 (bound / (2 * genGain * ceiling + 1 / 1000000)) ≤
      coordinateClipFactor bound (rawGainNativeGradient remaining genGain memGain state) := by
  have hn := gain_native_raw_norm_box remaining genGain memGain ceiling state hgen hmem horder hc hp
  have hd : 0 < coordinateGradientNorm (rawGainNativeGradient remaining genGain memGain state) +
      1 / 1000000 := by
    unfold coordinateGradientNorm
    positivity
  unfold coordinateClipFactor
  apply min_le_min le_rfl
  exact div_le_div_of_nonneg_left hb hd (by linarith only [hn])

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) ≤ 100 ∧ (0 : ℝ) ≤ 1 ∧
    ∀ i : Fin 4, 0 ≤ (seededNativeSubweights ((1, 2), (3, 4)) i).parameter ∧
      (seededNativeSubweights ((1, 2), (3, 4)) i).parameter ≤ 100 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

end Transformer.Grokking.CircuitEfficiency
