import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdCeiling
import Transformer.Grokking.AdamW.PairContraction

/-!
# Actual complete denominator floors for cold contraction

Sources: Varma et al., arXiv:2309.02390v1, section 3's CE/decay
competition and appendix C's four partner partials; native denominator
at 79f4fb0, retained mass at ada36f3 and cold ceiling at 696630b.

A supplied positive floor for all four current corrected denominators
bounds actual next total parameter mass by the remaining decay mass
plus rate times the newly retained negative moment mass over that floor.
The sum keeps the actual first-moment insertions and original feedback.

For any floor strictly below epsilon, the actual completed clocks
generate a common eventual denominator floor. Valid first beta gives
a vanishing correction power; the nonnegative square-root contribution
then covers the epsilon correction term. This uses no parameter,
gradient or variance convergence, nor a reset of either buffer.
The second beta and arbitrary retained states are unrestricted for
this lower bound; sign invariance in its application needs legal data.

These laws prepare an actual retained contraction at the strict cold
threshold. They do not assert the critical case, circuit selection or
learned transformer generalization. Exact-real fixed gained tables,
plain CE and uniform decoupled native decay differ from the source's
assigned coupled norm-cost GD. No optimizer or checkpoint is modified,
and no thermodynamic system-size scaling is encoded.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- A common actual denominator floor caps total native parameter
movement with newly retained moments. Sources: appendix C partner
partials and native denominator at 79f4fb0; the bound is current,
not an assumed future gradient or parameter limit. -/
theorem gain_native_parameter_mass_floor_denominator (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate floor : ℝ) (state : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 ≤ 1) (heta : 0 ≤ rate) (hd : 0 < floor)
    (hs : NonnegativeNativeState state)
    (hden : ∀ i, floor ≤ nextBufferDenominator b1 b2 eps (state i).variance
      (appliedGainNativeGradient remaining genGain memGain bound state i) (state i).clock) :
    gainNativeParameterMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) ≤
      (1 - rate * decay) * gainNativeParameterMass state + rate *
        (gainNativeNegativeMomentMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) / floor) := by
  have hcoordinate : ∀ i, (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state i).parameter ≤
      (1 - rate * decay) * (state i).parameter + rate *
        (-(gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state i).moment) / floor := by
    intro i
    have hg := gain_native_applied_gradient_nonpos remaining genGain memGain bound state i hclip
      (le_of_lt (gain_native_factor_gain_pos genGain memGain hgen hmem i)) (hs _).1
    have hn := scalar_native_moment_nonpos b1 b2 eps decay rate _ (state i) hb1 h1 (hs i).2.1 hg
    have hh := div_le_div_of_nonneg_left (mul_nonneg heta (neg_nonneg.mpr hn)) hd (hden i)
    rw [show gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state i =
      scalarNativeStep b1 b2 eps decay rate (state i)
        (appliedGainNativeGradient remaining genGain memGain bound state i) from rfl,
      scalar_parameter_denominator]
    linarith only [hh]
  convert add_le_add (add_le_add (hcoordinate 0) (hcoordinate 1))
    (add_le_add (hcoordinate 2) (hcoordinate 3)) using 1
  · rfl
  · unfold gainNativeParameterMass gainNativeNegativeMomentMass
    ring

example :
    let state := seededNativeSubweights ((0, 1 / 200), (0, 1))
    (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) ≤ 1 ∧
      (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) < 1 / 10 ∧ NonnegativeNativeState state ∧
      ∀ i, (1 / 10 : ℝ) ≤ nextBufferDenominator (9 / 10) (49 / 50) 1 (state i).variance
        (appliedGainNativeGradient 0 3 2 1 state i) (state i).clock := by
  dsimp only
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), ?_⟩
  intro i
  convert next_buffer_denominator_floor (9 / 10) (49 / 50) 1
    (seededNativeSubweights ((0, 1 / 200), (0, 1)) i).variance
    (appliedGainNativeGradient 0 3 2 1 (seededNativeSubweights ((0, 1 / 200), (0, 1))) i)
    (seededNativeSubweights ((0, 1 / 200), (0, 1)) i).clock
    (by norm_num) (by norm_num) (by norm_num) using 1; norm_num

/-- Any strict sub-epsilon floor eventually holds for all complete
actual denominators. Source: native corrected update at 79f4fb0;
completed-clock growth alone generates the tail, without any input,
physical-factor or retained-buffer limit premise. -/
theorem gain_native_denominator_tail_floor (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate floor : ℝ) (initial : NativeSubweightState)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hd : floor < eps) :
    ∃ start : ℕ, ∀ n, start ≤ n → ∀ i,
      floor ≤ nextBufferDenominator b1 b2 eps
        (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n i).variance
        (appliedGainNativeGradient remaining genGain memGain bound
          (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n) i)
        (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n i).clock := by
  let state := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
  have htail : ∀ i : Fin 4, ∀ᶠ n in atTop,
      floor ≤ nextBufferDenominator b1 b2 eps (state n i).variance
        (appliedGainNativeGradient remaining genGain memGain bound (state n) i) (state n i).clock := by
    intro i
    have hstep : ∀ n, state (n + 1) i = scalarNativeStep b1 b2 eps decay rate (state n i)
        (appliedGainNativeGradient remaining genGain memGain bound (state n) i) := fun _ => rfl
    have hclock := scalar_completed_clock_tendsto b1 b2 eps decay rate
      (fun n => state n i) (fun n => appliedGainNativeGradient remaining genGain memGain bound (state n) i) hstep
    have hpower := (tendsto_pow_atTop_nhds_zero_of_lt_one hb1 h1).comp hclock
    have hepsilon : Tendsto (fun n => (1 - b1 ^ ((state n i).clock + 1)) * eps) atTop (nhds eps) := by
      have hconst : Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 1) := tendsto_const_nhds
      simpa only [Function.comp_apply, sub_zero, one_mul] using (hconst.sub hpower).mul_const eps
    filter_upwards [hepsilon.eventually_const_lt hd] with n hn
    have hcorrection := le_of_lt (bias_correction_positive b1 ((state n i).clock + 1) hb1 h1 (by omega))
    have hroot := mul_nonneg hcorrection (Real.sqrt_nonneg
      ((b2 * (state n i).variance + (1 - b2) *
        appliedGainNativeGradient remaining genGain memGain bound (state n) i ^ 2) /
        (1 - b2 ^ ((state n i).clock + 1))))
    unfold nextBufferDenominator
    nlinarith only [hroot, hn]
  obtain ⟨start, hall⟩ := eventually_atTop.mp
    ((htail 0).and ((htail 1).and ((htail 2).and (htail 3))))
  refine ⟨start, ?_⟩
  intro n hn i
  have hh := hall n hn
  fin_cases i
  · exact hh.1
  · exact hh.2.1
  · exact hh.2.2.1
  · exact hh.2.2.2

example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (1 / 2 : ℝ) < 1 := by norm_num

/-- Legal initialized native CE dynamics generate both complete
retained mass envelopes on one tail. Sources: section 3 competition,
appendix C partials, native AdamW at 79f4fb0 and cold ceiling at
696630b; future convergence and buffer replacement are not premises. -/
theorem gain_native_cold_mass_tail_envelopes (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate floor : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 ≤ rate) (hkeep : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial) (hd : 0 < floor) (hfloor : floor < eps) :
    let state := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    ∃ start : ℕ, ∀ n, start ≤ n →
      gainNativeNegativeMomentMass (state (n + 1)) ≤ b1 * gainNativeNegativeMomentMass (state n) +
        (1 - b1) * (coldGainCEGradientScale remaining bound * genGain) * gainNativeParameterMass (state n) ∧
      gainNativeParameterMass (state (n + 1)) ≤ (1 - rate * decay) * gainNativeParameterMass (state n) +
        rate * (gainNativeNegativeMomentMass (state (n + 1)) / floor) := by
  dsimp only
  obtain ⟨start, htail⟩ := gain_native_denominator_tail_floor remaining genGain memGain bound b1 b2 eps decay rate floor initial
    hb1 h1 hfloor
  refine ⟨start, ?_⟩
  intro n hn
  have hsign := gain_native_nonnegative_path remaining genGain memGain bound b1 b2 eps decay rate initial n
    hgen hmem hclip hb1 h1 hb2 h2 he heta hkeep hs
  exact ⟨gain_native_cold_moment_mass_ceiling remaining genGain memGain bound b1 b2 eps decay rate _
    hgen hmem hgain hclip (le_of_lt h1) hsign,
    gain_native_parameter_mass_floor_denominator remaining genGain memGain bound b1 b2 eps decay rate floor _
      hgen hmem hclip hb1 (le_of_lt h1) heta hd hsign (htail n hn)⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * 100 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) < 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num, by norm_num⟩

end Transformer.Grokking.CircuitEfficiency
