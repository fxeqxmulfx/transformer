import Transformer.Grokking.CircuitEfficiency.SectionC_GainGrowthEnvelope

/-!
# Current total parameter and retained-moment mass laws

Sources: Varma et al., arXiv:2309.02390v1, appendix C's four product
partials and section 3 CE/decay competition; PyTorch 2.14.1 retained
native AdamW/clipping as ported at lab commit ada36f3.

Keep the sum of all four physical parameters and the negative sum of
all four retained first moments. Native insertion gives the exact
next negative moment mass, with the actual shared CE/clipping scale.
That scale lies below one; greater Gen gain bounds the complete next
moment mass by beta times its old mass plus (1-beta) times Gen gain
and current total parameter mass.

The actual corrected-denominator epsilon floor then bounds the next
parameter mass by remaining decay times its current mass plus the
floor step scale times the newly inserted negative moment mass.
Both buffers and each completed clock remain in the original update.
Current numerical signs are assumed; no future convergence or
instantaneous variance/gradient matching is imposed.

These laws prepare a static retained-memory contraction argument.
They alone do not show that a contraction factor exists, convergence,
task success or a GPTMini grokking event. Fixed physical tables and
uniform decoupled native decay differ from appendix C's coupled-cost
GD. No numerical-kernel or learned-transformer transfer is asserted.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Total physical mass in both learned factor pairs. Sources:
appendix C's four coordinates; no confidence or success is encoded. -/
def gainNativeParameterMass (state : NativeSubweightState) : ℝ :=
  ((state 0).parameter + (state 1).parameter) + ((state 2).parameter + (state 3).parameter)

/-- Negative total retained first-moment mass. Source: native AdamW
at ada36f3; this quantity does not substitute the current CE inputs. -/
def gainNativeNegativeMomentMass (state : NativeSubweightState) : ℝ :=
  -(((state 0).moment + (state 1).moment) + ((state 2).moment + (state 3).moment))

/-- Present numerical signs make both total masses nonnegative and
the parameter mass covers every coordinate. Sources: appendix C factors
and native retained sign region at ada36f3; no future mass is assumed. -/
theorem gain_native_masses_nonnegative (state : NativeSubweightState) (hs : NonnegativeNativeState state) :
    0 ≤ gainNativeParameterMass state ∧ 0 ≤ gainNativeNegativeMomentMass state ∧
      ∀ i, (state i).parameter ≤ gainNativeParameterMass state := by
  dsimp only [gainNativeParameterMass, gainNativeNegativeMomentMass]
  refine ⟨by linarith only [(hs 0).1, (hs 1).1, (hs 2).1, (hs 3).1],
    by linarith only [(hs 0).2.1, (hs 1).2.1, (hs 2).2.1, (hs 3).2.1], ?_⟩
  intro i
  fin_cases i
  · change (state 0).parameter ≤ _
    linarith only [(hs 1).1, (hs 2).1, (hs 3).1]
  · change (state 1).parameter ≤ _
    linarith only [(hs 0).1, (hs 2).1, (hs 3).1]
  · change (state 2).parameter ≤ _
    linarith only [(hs 0).1, (hs 1).1, (hs 3).1]
  · change (state 3).parameter ≤ _
    linarith only [(hs 0).1, (hs 1).1, (hs 2).1]

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Total native moment insertion is exactly the current gained
partner mass times the original shared CE/clipping scale. Sources:
appendix C actual partials and first-moment recurrence at ada36f3;
old retained moments, both gains and every partner are included. -/
theorem gain_native_moment_mass_step (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState) :
    gainNativeNegativeMomentMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) =
      b1 * gainNativeNegativeMomentMass state + (1 - b1) * gainCEGradientScale remaining genGain memGain bound state *
        (genGain * ((state 0).parameter + (state 1).parameter) +
          memGain * ((state 2).parameter + (state 3).parameter)) := by
  unfold gainNativeNegativeMomentMass gainNativeStep
  simp only [scalarNativeStep, gain_native_applied_gradient_scale]
  change -(((b1 * (state 0).moment + (1 - b1) * -(gainCEGradientScale remaining genGain memGain bound state * genGain * (state 1).parameter)) +
    (b1 * (state 1).moment + (1 - b1) * -(gainCEGradientScale remaining genGain memGain bound state * genGain * (state 0).parameter))) +
    ((b1 * (state 2).moment + (1 - b1) * -(gainCEGradientScale remaining genGain memGain bound state * memGain * (state 3).parameter)) +
    (b1 * (state 3).moment + (1 - b1) * -(gainCEGradientScale remaining genGain memGain bound state * memGain * (state 2).parameter)))) = _
  ring

/-- The actual newly retained total negative moment has the sharper
beta-dependent ceiling from current total parameter mass. Sources:
appendix C product partials and native CE at ada36f3; the current
shared scale bound is derived, without an external gradient stream. -/
theorem gain_native_moment_mass_ceiling (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (h1 : b1 ≤ 1) (hs : NonnegativeNativeState state) :
    gainNativeNegativeMomentMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) ≤
      b1 * gainNativeNegativeMomentMass state + (1 - b1) * genGain * gainNativeParameterMass state := by
  have hgm := add_nonneg (hs 0).1 (hs 1).1
  have hmm := add_nonneg (hs 2).1 (hs 3).1
  have hn := add_nonneg (mul_nonneg (le_of_lt hgen) hgm) (mul_nonneg (le_of_lt hmem) hmm)
  have hgap := mul_nonneg (show 0 ≤ genGain - memGain by linarith only [hgain]) hmm
  have hweights : genGain * ((state 0).parameter + (state 1).parameter) +
      memGain * ((state 2).parameter + (state 3).parameter) ≤ genGain * gainNativeParameterMass state := by
    unfold gainNativeParameterMass
    nlinarith only [hgap]
  have hc := gain_ce_gradient_scale_unit_interval remaining genGain memGain bound state hclip
  have hscale := mul_le_mul_of_nonneg_right (le_of_lt hc.2) hn
  simp only [one_mul] at hscale
  have hw := mul_le_mul_of_nonneg_left (le_trans hscale hweights)
    (show 0 ≤ 1 - b1 by linarith only [h1])
  rw [gain_native_moment_mass_step]
  nlinarith only [hw]

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (9 / 10 : ℝ) ≤ 1 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

/-- The complete actual parameter-mass ceiling retains the new first
moments and the true corrected-denominator floor. Sources: native
AdamW at ada36f3 and appendix C actual partner inputs; neither current
variance matching nor zero-beta or future sign data are assumed. -/
theorem gain_native_parameter_mass_ceiling (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps) (heta : 0 ≤ rate)
    (hs : NonnegativeNativeState state) :
    gainNativeParameterMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) ≤
      (1 - rate * decay) * gainNativeParameterMass state + rate / ((1 - b1) * eps) *
        gainNativeNegativeMomentMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) := by
  have hcoordinate : ∀ i, (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state i).parameter ≤
      (1 - rate * decay) * (state i).parameter + rate / ((1 - b1) * eps) *
        (-(gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state i).moment) := by
    intro i
    have hg := gain_native_applied_gradient_nonpos remaining genGain memGain bound state i hclip
      (le_of_lt (gain_native_factor_gain_pos genGain memGain hgen hmem i)) (hs _).1
    have hu := scalar_parameter_denominator_ceiling b1 b2 eps decay rate _ (state i) hb1 h1 he heta (hs i).2.1 hg
    dsimp only [gainNativeStep]
    convert hu using 1
    ring
  convert add_le_add (add_le_add (hcoordinate 0) (hcoordinate 1)) (add_le_add (hcoordinate 2) (hcoordinate 3)) using 1
  · rfl
  · dsimp only [gainNativeParameterMass, gainNativeNegativeMomentMass]
    ring

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

/-- Every actual applied coordinate input is covered by greater gain
times current total parameter mass. Sources: appendix C true partials
and shared clipping at ada36f3; no prescribed input or future bound is
assumed, preparing buffer limits from generated parameter collapse. -/
theorem gain_native_applied_gradient_mass_cap (remaining : ℕ) (genGain memGain bound : ℝ)
    (state : NativeSubweightState) (i : Fin 4) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (hs : NonnegativeNativeState state) :
    0 ≤ -appliedGainNativeGradient remaining genGain memGain bound state i ∧
      -appliedGainNativeGradient remaining genGain memGain bound state i ≤ genGain * gainNativeParameterMass state := by
  have hc := gain_ce_gradient_scale_unit_interval remaining genGain memGain bound state hclip
  have hi : nativeFactorGain genGain memGain i ≤ genGain := by
    fin_cases i
    · exact le_rfl
    · exact le_rfl
    · exact hgain
    · exact hgain
  have hg := le_of_lt (gain_native_factor_gain_pos genGain memGain hgen hmem i)
  have hp := (hs (nativeFactorPartner i)).1
  have hn := mul_nonneg hg hp
  have hu := mul_le_mul_of_nonneg_right (le_of_lt hc.2) hn
  simp only [one_mul] at hu
  have hgainBound := mul_le_mul_of_nonneg_right hi hp
  have hmass := mul_le_mul_of_nonneg_left
    ((gain_native_masses_nonnegative state hs).2.2 (nativeFactorPartner i)) (le_of_lt hgen)
  rw [gain_native_applied_gradient_scale, neg_neg]
  refine ⟨mul_nonneg (mul_nonneg (le_of_lt hc.1) hg) hp, ?_⟩
  simpa only [mul_assoc] using le_trans hu (le_trans hgainBound hmass)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

end Transformer.Grokking.CircuitEfficiency
