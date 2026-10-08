import Transformer.Grokking.CircuitEfficiency.SectionC_GainSigns
import Transformer.Grokking.CircuitEfficiency.SectionC_GainGradientLimits
import Transformer.Grokking.AdamW.PairEnvelope

/-!
# Actual gained CE generates the retained-pair envelope

Sources: Varma et al., arXiv:2309.02390v1, section 3 and appendix C;
native AdamW/clipping at lab commit 2150464. Gen parameter mass is
the sum of its two physical factors; negative moment mass retains
both actual first moments. Derive their feedback directly from
the current true CE inputs and simultaneous native update.

Finite physical parameter convergence gives the actual shared CE
scale limit. A lower coefficient and upper actual denominators then
give the two explicit envelope inequalities checked in PairEnvelope.
No future gradient stream or instantaneous buffer match is supplied.
The application still must derive eventual bounds and convergence.

Uniform decoupled decay, fixed physical gains and plain CE differ
from appendix C's GD/coupled norm cost. These numerical feedback
identities do not prove source-seed attraction, a delayed crossing,
or learned stochastic/floating-point GPTMini generalization.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Sum of Gen's current physical factors. Source: appendix C's
two-factor coordinates; no successful-margin property is encoded. -/
def gainGenParameterMass (state : NativeSubweightState) : ℝ :=
  (state 0).parameter + (state 1).parameter

/-- Negative sum of Gen's retained first moments. Source: native
AdamW at 2150464; current CE inputs do not replace these buffers. -/
def gainGenNegativeMomentMass (state : NativeSubweightState) : ℝ :=
  -((state 0).moment + (state 1).moment)

/-- Current parameter limits determine the actual shared clipped
CE scale limit. Sources: appendix C true CE and native clipping at
2150464; no coefficient convergence is independently presumed. -/
theorem gain_ce_gradient_scale_tendsto (remaining : ℕ) (genGain memGain bound : ℝ)
    (state : ℕ → NativeSubweightState) (reference : NativeSubweightState)
    (hp : ∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter)) :
    Tendsto (fun n => gainCEGradientScale remaining genGain memGain bound (state n)) atTop
      (nhds (gainCEGradientScale remaining genGain memGain bound reference)) := by
  have hs := gain_native_total_score_tendsto genGain memGain state reference hp
  have hc := gain_native_clip_factor_tendsto remaining genGain memGain bound state reference hp
  have hd := (hs.rexp.add_const (remaining : ℝ)).add_const 1
  have hn : Real.exp (gainNativeTotalScore genGain memGain reference) + (remaining : ℝ) + 1 ≠ 0 := by positivity
  have ht : Tendsto (fun _ : ℕ => (remaining : ℝ) + 1) atTop (nhds ((remaining : ℝ) + 1)) := tendsto_const_nhds
  simpa only [gainCEGradientScale, Pi.div_def] using hc.mul (ht.div hd hn)

example : ∀ i : Fin 4, Tendsto (fun n : ℕ =>
    ({ parameter := (i.val : ℝ) + 1, moment := 0, variance := 0, clock := n } : ScalarState).parameter)
    atTop (nhds ((i.val : ℝ) + 1)) := by
  intro i
  exact tendsto_const_nhds

/-- Actual numerical signs give nonnegative parameter and negative
moment masses. Sources: appendix C coordinates and native sign
region at 2150464; positive limiting mass is not a sign premise. -/
theorem gain_gen_masses_nonnegative (state : NativeSubweightState) (hs : NonnegativeNativeState state) :
    0 ≤ gainGenParameterMass state ∧ 0 ≤ gainGenNegativeMomentMass state := by
  unfold gainGenParameterMass gainGenNegativeMomentMass
  constructor
  · linarith [(hs 0).1, (hs 1).1]
  · linarith [(hs 0).2.1, (hs 1).2.1]

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) := by
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- The true gained CE inserts exactly its present parameter mass
into the retained negative moment mass. Sources: appendix C product
partials and native first-moment recurrence at 2150464. -/
theorem gain_gen_moment_mass_step (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState) :
    gainGenNegativeMomentMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) =
      b1 * gainGenNegativeMomentMass state + (1 - b1) *
        (gainCEGradientScale remaining genGain memGain bound state * genGain) * gainGenParameterMass state := by
  unfold gainGenNegativeMomentMass gainNativeStep
  simp only [scalarNativeStep, gain_native_applied_gradient_scale]
  change -((b1 * (state 0).moment + (1 - b1) *
    -(gainCEGradientScale remaining genGain memGain bound state * genGain * (state 1).parameter)) +
    (b1 * (state 1).moment + (1 - b1) *
      -(gainCEGradientScale remaining genGain memGain bound state * genGain * (state 0).parameter))) = _
  unfold gainGenParameterMass
  ring

/-- A bound on the actual present CE coefficient gives the moment
envelope inequality. Sources: appendix C partner feedback and native
AdamW at 2150464; no independent future input is substituted. -/
theorem gain_gen_moment_mass_floor (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate coefficient : ℝ)
    (state : NativeSubweightState) (h1 : b1 ≤ 1) (hs : NonnegativeNativeState state)
    (hc : coefficient ≤ gainCEGradientScale remaining genGain memGain bound state * genGain) :
    b1 * gainGenNegativeMomentMass state + (1 - b1) * coefficient * gainGenParameterMass state ≤
      gainGenNegativeMomentMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) := by
  rw [gain_gen_moment_mass_step]
  have hu := (gain_gen_masses_nonnegative state hs).1
  have ht := mul_nonneg (mul_nonneg (show 0 ≤ 1 - b1 by linarith)
    (show 0 ≤ gainCEGradientScale remaining genGain memGain bound state * genGain - coefficient by linarith)) hu
  nlinarith only [ht]

example : (9 / 10 : ℝ) ≤ 1 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    gainCEGradientScale 111 3 2 1 (seededNativeSubweights ((0, 1 / 200), (0, 1))) * 3 ≤
      gainCEGradientScale 111 3 2 1 (seededNativeSubweights ((0, 1 / 200), (0, 1))) * 3 := by
  exact ⟨by norm_num, native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), le_rfl⟩

/-- Actual corrected denominator ceilings yield the parameter-mass
envelope bound. Sources: appendix C true CE and native AdamW at
2150464; both first moments, variances and completed clocks remain. -/
theorem gain_gen_parameter_mass_floor (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate ceiling : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps) (heta : 0 ≤ rate) (hd : 0 < ceiling)
    (hs : NonnegativeNativeState state)
    (hd0 : nextBufferDenominator b1 b2 eps (state 0).variance
      (appliedGainNativeGradient remaining genGain memGain bound state 0) (state 0).clock ≤ ceiling)
    (hd1 : nextBufferDenominator b1 b2 eps (state 1).variance
      (appliedGainNativeGradient remaining genGain memGain bound state 1) (state 1).clock ≤ ceiling) :
    ceiling * (1 - rate * decay) * gainGenParameterMass state + rate *
      gainGenNegativeMomentMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) ≤
        ceiling * gainGenParameterMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) := by
  have hg0 := gain_native_applied_gradient_nonpos remaining genGain memGain bound state 0 hclip
    (show 0 ≤ nativeFactorGain genGain memGain 0 from le_of_lt hgen) (hs _).1
  have hg1 := gain_native_applied_gradient_nonpos remaining genGain memGain bound state 1 hclip
    (show 0 ≤ nativeFactorGain genGain memGain 1 from le_of_lt hgen) (hs _).1
  have ha := scalar_parameter_denominator_floor b1 b2 eps decay rate _ ceiling (state 0) hb1 h1 he heta
    (hs 0).2.1 hg0 hd0
  have hb := scalar_parameter_denominator_floor b1 b2 eps decay rate _ ceiling (state 1) hb1 h1 he heta
    (hs 1).2.1 hg1 hd1
  have ht := mul_le_mul_of_nonneg_left (add_le_add ha hb) (le_of_lt hd)
  have hcancel0 := div_mul_cancel₀
    (rate * (-(scalarNativeStep b1 b2 eps decay rate (state 0) (appliedGainNativeGradient remaining genGain memGain bound state 0)).moment)) (ne_of_gt hd)
  have hcancel1 := div_mul_cancel₀
    (rate * (-(scalarNativeStep b1 b2 eps decay rate (state 1) (appliedGainNativeGradient remaining genGain memGain bound state 1)).moment)) (ne_of_gt hd)
  unfold gainGenParameterMass gainGenNegativeMomentMass gainNativeStep
  nlinarith only [ht, hcancel0, hcancel1]

example :
    let state := seededNativeSubweights ((0, 1 / 200), (0, 1))
    let d0 := nextBufferDenominator (9 / 10) (49 / 50) 1 (state 0).variance
      (appliedGainNativeGradient 111 3 2 1 state 0) (state 0).clock
    let d1 := nextBufferDenominator (9 / 10) (49 / 50) 1 (state 1).variance
      (appliedGainNativeGradient 111 3 2 1 state 1) (state 1).clock
    (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
      (0 : ℝ) ≤ 1 / 1000 ∧ 0 < d0 + d1 ∧ NonnegativeNativeState state ∧ d0 ≤ d0 + d1 ∧ d1 ≤ d0 + d1 := by
  dsimp only
  let state := seededNativeSubweights ((0, 1 / 200), (0, 1))
  have h0 := next_buffer_denominator_pos (9 / 10) (49 / 50) 1 (state 0).variance
    (appliedGainNativeGradient 111 3 2 1 state 0) (state 0).clock (by norm_num) (by norm_num) (by norm_num)
  have h1 := next_buffer_denominator_pos (9 / 10) (49 / 50) 1 (state 1).variance
    (appliedGainNativeGradient 111 3 2 1 state 1) (state 1).clock (by norm_num) (by norm_num) (by norm_num)
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), ?_, ?_⟩
  · linarith only [h0, h1]
  · linarith only [h1]
  · linarith only [h0]

end Transformer.Grokking.CircuitEfficiency
