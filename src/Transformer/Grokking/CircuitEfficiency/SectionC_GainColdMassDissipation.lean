import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdMassMonotone
import Transformer.Grokking.CircuitEfficiency.SectionC_GainFeedbackFloor
import Transformer.Grokking.AdamW.ScalarInputDissipation

/-!
# Actual critical mass dissipation from true CE inputs

Sources: Varma et al., arXiv:2309.02390v1, section 3 CE/decay
competition and appendix C product partials; native input-square
dissipation at c555a63, cold ceiling at 696630b and complete retained
mass observers at ada36f3. The actual callback and clipping remain.

With legal first beta zero, one actual input bounded away from zero
forces an explicit loss of total physical mass at or above the cold
threshold. Sum the true scalar adaptive estimates, retaining every
second buffer and completed clock. The other coordinates need no
independent input floors. Nonnegative rate is allowed in this observer
law; a strict mass decrement needs positive rate.

A current physical box and positive lower mass generate a positive
input floor for at least one of the four actual coordinates. This
comes from the true CE floor, each physical gain and its partner.
No external gradient stream or future input tail is prescribed.

These are current estimates. Actual critical convergence still needs
the initialized mass limit and physical box to exclude a positive
limit. First beta zero is an explicit legal mathematical control, not
a reset of the frozen beta1=0.9 transformer experiments. The second
buffer is retained. Fixed tables/plain CE/uniform native decay differ
from appendix C's assigned coupled norm-cost GD. Exact-real proofs
do not establish learned GPTMini or thermodynamic system-size transfer.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- One actual negative input floor generates a positive total-mass
dissipation gap at or above the cold threshold. Sources: appendix C
partner inputs, native dissipation at c555a63 and cold ceiling at
696630b; every coordinate uses the original CE callback and buffers. -/
theorem gain_native_zero_first_beta_cold_mass_gap (remaining : ℕ)
    (genGain memGain bound b2 eps decay rate lower : ℝ) (state : NativeSubweightState) (i : Fin 4)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 ≤ rate)
    (hs : NonnegativeNativeState state) (hl : 0 < lower)
    (hg : appliedGainNativeGradient remaining genGain memGain bound state i ≤ -lower)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    let gap := lower / eps - lower / (eps + Real.sqrt ((1 - b2) * lower ^ 2))
    0 < gap ∧
      gainNativeParameterMass (gainNativeStep remaining genGain memGain bound 0 b2 eps decay rate state) ≤
        gainNativeParameterMass state - rate * gap := by
  dsimp only
  let gap := lower / eps - lower / (eps + Real.sqrt ((1 - b2) * lower ^ 2))
  let next := gainNativeStep remaining genGain memGain bound 0 b2 eps decay rate state
  let linear := fun j : Fin 4 => (1 - rate * decay) * (state j).parameter + rate * (-(next j).moment) / eps
  have hcoordinate : ∀ j, (next j).parameter ≤ linear j := by
    intro j
    have hj := gain_native_applied_gradient_nonpos remaining genGain memGain bound state j hclip
      (le_of_lt (gain_native_factor_gain_pos genGain memGain hgen hmem j)) (hs _).1
    have hh := scalar_parameter_denominator_ceiling 0 b2 eps decay rate _ (state j)
      (by norm_num) (by norm_num) he heta (hs j).2.1 hj
    simpa only [linear, next, gainNativeStep, sub_zero, one_mul] using hh
  obtain ⟨hgap, hstrong⟩ := scalar_zero_first_beta_adaptive_gap b2 eps decay rate
    (appliedGainNativeGradient remaining genGain memGain bound state i) lower (state i)
    hb2 h2 (hs i).2.2 he heta hl hg
  have hstrongLinear : (next i).parameter ≤ linear i - rate * gap := by
    convert hstrong using 1 <;> dsimp only [linear, next, gap, gainNativeStep, scalarNativeStep]
    ring
  have hsum : gainNativeParameterMass next ≤
      (linear 0 + linear 1) + (linear 2 + linear 3) - rate * gap := by
    unfold gainNativeParameterMass
    fin_cases i
    · change (next 0).parameter ≤ linear 0 - rate * gap at hstrongLinear
      convert add_le_add (add_le_add hstrongLinear (hcoordinate 1))
        (add_le_add (hcoordinate 2) (hcoordinate 3)) using 1
      ring
    · change (next 1).parameter ≤ linear 1 - rate * gap at hstrongLinear
      convert add_le_add (add_le_add (hcoordinate 0) hstrongLinear)
        (add_le_add (hcoordinate 2) (hcoordinate 3)) using 1
      ring
    · change (next 2).parameter ≤ linear 2 - rate * gap at hstrongLinear
      convert add_le_add (add_le_add (hcoordinate 0) (hcoordinate 1))
        (add_le_add hstrongLinear (hcoordinate 3)) using 1
      ring
    · change (next 3).parameter ≤ linear 3 - rate * gap at hstrongLinear
      convert add_le_add (add_le_add (hcoordinate 0) (hcoordinate 1))
        (add_le_add (hcoordinate 2) hstrongLinear) using 1
      ring
  have hidentity : (linear 0 + linear 1) + (linear 2 + linear 3) =
      (1 - rate * decay) * gainNativeParameterMass state + rate * (gainNativeNegativeMomentMass next / eps) := by
    dsimp only [linear, gainNativeParameterMass, gainNativeNegativeMomentMass]
    ring
  rw [hidentity] at hsum
  have hm := gain_native_cold_moment_mass_ceiling remaining genGain memGain bound 0 b2 eps decay rate state
    hgen hmem hgain hclip (by norm_num) hs
  simp only [zero_mul, sub_zero, one_mul, zero_add] at hm
  have hmoment := mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right hm (le_of_lt he)) heta
  have hscaled := mul_le_mul_of_nonneg_right hthreshold (gain_native_masses_nonnegative state hs).1
  have hfeedback : (coldGainCEGradientScale remaining bound * genGain * gainNativeParameterMass state) / eps ≤
      decay * gainNativeParameterMass state := (div_le_iff₀ he).mpr (by nlinarith only [hscaled])
  have hlast := mul_le_mul_of_nonneg_left hfeedback heta
  exact ⟨hgap, by nlinarith only [hsum, hmoment, hlast]⟩

example :
    let state := seededNativeSubweights ((0, 1 / 200), (0, 1))
    let lower := -appliedGainNativeGradient 0 3 2 1 state 0 / 2
    (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
      (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
      NonnegativeNativeState state ∧ 0 < lower ∧
      appliedGainNativeGradient 0 3 2 1 state 0 ≤ -lower ∧
      coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  dsimp only
  have hscale := gain_ce_gradient_scale_pos 0 3 2 1 (seededNativeSubweights ((0, 1 / 200), (0, 1))) (by norm_num)
  have hinput : appliedGainNativeGradient 0 3 2 1 (seededNativeSubweights ((0, 1 / 200), (0, 1))) 0 < 0 := by
    rw [gain_native_applied_gradient_scale]
    change -(gainCEGradientScale 0 3 2 1 (seededNativeSubweights ((0, 1 / 200), (0, 1))) * 3 * (1 / 200)) < 0
    nlinarith only [hscale]
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by linarith only [hinput], by linarith only [hinput], by norm_num [coldGainCEGradientScale]⟩

/-- A physical box and positive current mass force at least one
actual CE input above a quantitative positive floor. Sources:
appendix C's four partner partials and complete clipped CE box floor
at ee02740; no future input or successful classification is a premise. -/
theorem gain_native_box_large_applied_input (remaining : ℕ)
    (genGain memGain bound ceiling lower : ℝ) (state : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain)
    (hclip : 0 < bound) (hc : 0 ≤ ceiling) (hl : 0 < lower)
    (hp : ∀ i, 0 ≤ (state i).parameter ∧ (state i).parameter ≤ ceiling)
    (hmass : lower ≤ gainNativeParameterMass state) :
    0 < gainCEFeedbackFloor remaining genGain memGain bound ceiling * memGain * lower / 4 ∧
      ∃ i, gainCEFeedbackFloor remaining genGain memGain bound ceiling * memGain * lower / 4 ≤
        -appliedGainNativeGradient remaining genGain memGain bound state i := by
  have hf := gain_ce_feedback_floor_pos remaining genGain memGain bound ceiling (le_of_lt hgen) hc hclip
  have hfloor : ∀ i, gainCEFeedbackFloor remaining genGain memGain bound ceiling * memGain *
      (state (nativeFactorPartner i)).parameter ≤ -appliedGainNativeGradient remaining genGain memGain bound state i := by
    intro i
    have hg : memGain ≤ nativeFactorGain genGain memGain i := by
      fin_cases i
      · exact hgain
      · exact hgain
      · exact le_rfl
      · exact le_rfl
    have hfirst := mul_le_mul_of_nonneg_left hg (le_of_lt hf)
    have hsecond := mul_le_mul_of_nonneg_right hfirst (hp (nativeFactorPartner i)).1
    exact le_trans hsecond (gain_native_applied_gradient_box_floor remaining genGain memGain bound ceiling state i
      hgen hmem hgain hc hclip hp)
  refine ⟨by positivity, ?_⟩
  by_contra h
  push Not at h
  have hmassScaled := mul_le_mul_of_nonneg_left hmass (mul_nonneg (le_of_lt hf) (le_of_lt hmem))
  have h0 := hfloor 0
  have h1 := hfloor 1
  have h2 := hfloor 2
  have h3 := hfloor 3
  change _ * memGain * (state 1).parameter ≤ _ at h0
  change _ * memGain * (state 0).parameter ≤ _ at h1
  change _ * memGain * (state 3).parameter ≤ _ at h2
  change _ * memGain * (state 2).parameter ≤ _ at h3
  unfold gainNativeParameterMass at hmassScaled
  nlinarith only [h0, h1, h2, h3, h 0, h 1, h 2, h 3, hmassScaled]

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    (∀ i : Fin 4, 0 ≤ (seededNativeSubweights ((0, 1 / 200), (0, 1)) i).parameter ∧
      (seededNativeSubweights ((0, 1 / 200), (0, 1)) i).parameter ≤ 2) ∧
    (1 : ℝ) ≤ gainNativeParameterMass (seededNativeSubweights ((0, 1 / 200), (0, 1))) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_, ?_⟩
  · intro i
    fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]
  · norm_num [gainNativeParameterMass, seededNativeSubweights, seededScalarState]

end Transformer.Grokking.CircuitEfficiency
