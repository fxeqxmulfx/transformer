import Transformer.Grokking.CircuitEfficiency.SectionC_GainEnvelope
import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdRegime
import Transformer.Grokking.AdamW.ScalarDenominator

/-!
# A finite-step upper envelope for actual retained Gen formation

Sources: Varma et al., arXiv:2309.02390v1, section 3's Slow vs fast
learning and appendix C's two-factor CE; native AdamW/clipping at
lab commit 73c78eb. Keep both physical factors and both retained
first moments in the growth quantity. The true clipped CE scale is
less than one and the actual denominator has its epsilon floor.

Derive the upper envelope from the present state, rather than assuming
future gradient or buffer bounds. The growth factor is deliberately
loose and can be large at small epsilon. Subsequent finite-budget
comparisons must derive an appropriately small positive seed.
These are exact-real, fixed-table, uniform decoupled-decay statements.
They do not prove parameter convergence, later success, a learned
transformer circuit, or a thermodynamic transition. Appendix C uses
GD and a coupled assigned norm cost; that optimizer is not substituted.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Current Gen mass plus its retained negative moment mass at the
epsilon-floor step scale. Sources: appendix C factors and native
AdamW at 73c78eb; this numerical quantity encodes no task success. -/
noncomputable def gainGenGrowthWeight (b1 eps rate : ℝ) (state : NativeSubweightState) : ℝ :=
  gainGenParameterMass state + rate / ((1 - b1) * eps) * gainGenNegativeMomentMass state

/-- A loose uniform growth factor using the actual gain and native
epsilon-floor scale. Sources: appendix C forward and AdamW at 73c78eb. -/
noncomputable def gainGenGrowthFactor (b1 eps rate genGain : ℝ) : ℝ :=
  2 + 2 * (rate / ((1 - b1) * eps)) * genGain

/-- The numerical upper factor is at least two in the intended native
region. Sources: appendix C positive gain and AdamW at 73c78eb;
this permits comparison of different clocks through a fixed budget.
It asserts a bound on the envelope factor, not on actual growth. -/
theorem gain_gen_growth_factor_two_le (b1 eps rate genGain : ℝ)
    (h1 : b1 < 1) (he : 0 < eps) (heta : 0 ≤ rate) (hg : 0 ≤ genGain) :
    2 ≤ gainGenGrowthFactor b1 eps rate genGain := by
  have ht : 0 ≤ rate / ((1 - b1) * eps) := div_nonneg heta (by positivity)
  have hp := mul_nonneg ht hg
  unfold gainGenGrowthFactor
  linarith only [hp]

example : (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) ≤ 3 := by
  norm_num

/-- Numerical nonnegativity makes the growth weight cover the actual
Gen parameter mass. Sources: appendix C factors and retained signs
at 73c78eb; no positive or convergent future mass is assumed. -/
theorem gain_gen_growth_weight_covers_mass (b1 eps rate : ℝ) (state : NativeSubweightState)
    (h1 : b1 < 1) (he : 0 < eps) (heta : 0 ≤ rate) (hs : NonnegativeNativeState state) :
    0 ≤ gainGenParameterMass state ∧
      gainGenParameterMass state ≤ gainGenGrowthWeight b1 eps rate state := by
  have hm := gain_gen_masses_nonnegative state hs
  have ht : 0 ≤ rate / ((1 - b1) * eps) := div_nonneg heta (by positivity)
  have hw := mul_nonneg ht hm.2
  exact ⟨hm.1, by unfold gainGenGrowthWeight; linarith only [hw]⟩

example : (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

/-- Actual Gen moment insertion is at most the old negative moment
mass plus gained parameter mass. Sources: appendix C's true partner
derivatives and native clipping at 73c78eb; the CE scale bound is derived. -/
theorem gain_gen_moment_mass_ceiling (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 ≤ 1) (hs : NonnegativeNativeState state) :
    gainGenNegativeMomentMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) ≤
      gainGenNegativeMomentMass state + genGain * gainGenParameterMass state := by
  rw [gain_gen_moment_mass_step]
  have hm := gain_gen_masses_nonnegative state hs
  have hc := gain_ce_gradient_scale_unit_interval remaining genGain memGain bound state hclip
  have hret := mul_nonneg (show 0 ≤ 1 - b1 by linarith only [h1]) hm.2
  have hmiss := mul_nonneg (mul_nonneg
    (show 0 ≤ 1 - (1 - b1) * gainCEGradientScale remaining genGain memGain bound state by nlinarith [hc.1, hc.2])
    (le_of_lt hgen)) hm.1
  nlinarith only [hret, hmiss]

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) ≤ 1 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

/-- Sum the two actual epsilon-floor increment ceilings with the
new retained moments. Sources: native AdamW at 73c78eb and appendix
C's actual CE; no variance/current-gradient matching is assumed. -/
theorem gain_gen_parameter_mass_ceiling (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps) (heta : 0 ≤ rate)
    (hs : NonnegativeNativeState state) :
    gainGenParameterMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) ≤
      (1 - rate * decay) * gainGenParameterMass state + rate / ((1 - b1) * eps) *
        gainGenNegativeMomentMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) := by
  have hg0 := gain_native_applied_gradient_nonpos remaining genGain memGain bound state 0 hclip
    (show 0 ≤ nativeFactorGain genGain memGain 0 from le_of_lt hgen) (hs _).1
  have hg1 := gain_native_applied_gradient_nonpos remaining genGain memGain bound state 1 hclip
    (show 0 ≤ nativeFactorGain genGain memGain 1 from le_of_lt hgen) (hs _).1
  have ha := scalar_parameter_denominator_ceiling b1 b2 eps decay rate _ (state 0) hb1 h1 he heta (hs 0).2.1 hg0
  have hb := scalar_parameter_denominator_ceiling b1 b2 eps decay rate _ (state 1) hb1 h1 he heta (hs 1).2.1 hg1
  convert add_le_add ha hb using 1
  · rfl
  · dsimp only [gainGenParameterMass, gainGenNegativeMomentMass, gainNativeStep]
    ring

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

/-- The actual retained Gen growth weight has a uniform one-step
upper factor at nonnegative decay. Sources: section 3 Slow vs fast
learning and native AdamW at 73c78eb; this is an upper rate, not an
assumed monotone parameter path or an eventual successful margin. -/
theorem gain_gen_growth_weight_step (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps) (heta : 0 ≤ rate) (hdecay : 0 ≤ decay)
    (hs : NonnegativeNativeState state) :
    gainGenGrowthWeight b1 eps rate (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) ≤
      gainGenGrowthFactor b1 eps rate genGain * gainGenGrowthWeight b1 eps rate state := by
  let t := rate / ((1 - b1) * eps)
  have ht : 0 ≤ t := div_nonneg heta (by positivity)
  have hm := gain_gen_masses_nonnegative state hs
  have hu := gain_gen_parameter_mass_ceiling remaining genGain memGain bound b1 b2 eps decay rate state
    hgen hclip hb1 h1 he heta hs
  have hw := gain_gen_moment_mass_ceiling remaining genGain memGain bound b1 b2 eps decay rate state
    hgen hclip hb1 (le_of_lt h1) hs
  have hret := mul_le_mul_of_nonneg_left hw (show 0 ≤ 2 * t by positivity)
  have hshrink := mul_nonneg (mul_nonneg heta hdecay) hm.1
  have hextra := mul_nonneg (mul_nonneg (mul_nonneg ht ht) (le_of_lt hgen)) hm.2
  change gainGenParameterMass _ ≤ (1 - rate * decay) * gainGenParameterMass state +
    t * gainGenNegativeMomentMass _ at hu
  change gainGenParameterMass _ + t * gainGenNegativeMomentMass _ ≤
    (2 + 2 * t * genGain) * (gainGenParameterMass state + t * gainGenNegativeMomentMass state)
  nlinarith only [hu, hret, hshrink, hextra, hm.1]

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) ≤ 1 / 10 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

end Transformer.Grokking.CircuitEfficiency
