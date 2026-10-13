import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdWeightedLimit

/-!
# Actual retained weighted dissipation at the cold threshold

Sources: Varma et al., arXiv:2309.02390v1, section 3 CE/decay
competition and appendix C product partials; weighted native clock
budget at 4748aba and retained adaptive gap at 9d87822.

One actual CE input bounded away from zero, together with its actual
denominator above epsilon, gives a positive decrement of the weighted
physical parameter/first-moment mass. The other three coordinates
retain their numerical native clock-error bounds. The complete shared
clipped CE callback, both buffers and completed clock stay unchanged.

The current input and denominator premises are explicit. A later
initialized-path application must generate them from a positive
physical mass, the derived CE box floor and its completed clocks.
They are not prescribed future inputs or parameter convergence.

At or above the cold threshold, the linear parameter/retained-moment
terms cancel in the weighted upper bound. Geometric clock errors
remain, but the nonzero-input decrement can dominate them on a tail.
This prepares the actual critical-collapse proof with nonzero betas;
the current estimate alone neither establishes convergence nor selects
Gen over Mem. Nonnegative rate is allowed here.

Fixed gained tables/plain CE/uniform decoupled native decay differ
from appendix C's assigned coupled norm-cost GD. The formulas use
exact reals and non-AMSGrad AdamW. No frozen optimizer or checkpoint
is modified, and no learned GPTMini or thermodynamic size transfer
is asserted by this four-coordinate current law.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- A current large actual input and strict denominator floor give
weighted native dissipation with an explicit clock error. Sources:
appendix C true partials, retained gap at 9d87822 and weighted
clock budget at 4748aba; the whole feedback step remains native. -/
theorem gain_native_cold_weighted_mass_gap (remaining clock : ℕ)
    (genGain memGain bound b1 b2 eps decay rate lower floor : ℝ) (state : NativeSubweightState) (i : Fin 4)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps) (heta : 0 ≤ rate)
    (hs : NonnegativeNativeState state) (hbound : ∀ j, -(state j).moment ≤ bound)
    (hclock : ∀ j, (state j).clock = clock) (hl : 0 < lower)
    (hg : appliedGainNativeGradient remaining genGain memGain bound state i ≤ -lower) (hf : eps < floor)
    (hden : floor ≤ nextBufferDenominator b1 b2 eps (state i).variance
      (appliedGainNativeGradient remaining genGain memGain bound state i) (state i).clock)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    let gap := ((1 - b1) * lower) / eps - ((1 - b1) * lower) / floor
    0 < gap ∧ coldGainNativeWeightedMass b1 eps rate
      (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) ≤
        coldGainNativeWeightedMass b1 eps rate state + (4 * rate * bound) * b1 ^ clock -
          ((1 - b1) * eps) * rate * gap := by
  dsimp only
  let gap := ((1 - b1) * lower) / eps - ((1 - b1) * lower) / floor
  let next := gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state
  let weighted := fun j : Fin 4 => ((1 - b1) * eps) * (next j).parameter + rate * b1 * (-(next j).moment)
  let linear := fun j : Fin 4 => ((1 - b1) * eps) * (1 - rate * decay) * (state j).parameter +
    rate * (-(next j).moment) + rate * bound * b1 ^ clock
  have hcoordinate : ∀ j, weighted j ≤ linear j := by
    intro j
    have hj := gain_native_applied_gradient_nonpos remaining genGain memGain bound state j hclip
      (le_of_lt (gain_native_factor_gain_pos genGain memGain hgen hmem j)) (hs _).1
    have hmag := gain_native_applied_gradient_abs_bound remaining genGain memGain bound state j hclip
    rw [abs_of_nonpos hj] at hmag
    have hold := mul_le_mul_of_nonneg_left (hbound j) hb1
    have hnew := mul_le_mul_of_nonneg_left hmag (show 0 ≤ 1 - b1 by linarith only [h1])
    have hnext : -(scalarNativeStep b1 b2 eps decay rate (state j)
        (appliedGainNativeGradient remaining genGain memGain bound state j)).moment ≤ bound := by
      change -(b1 * (state j).moment + (1 - b1) * appliedGainNativeGradient remaining genGain memGain bound state j) ≤ bound
      nlinarith only [hold, hnew]
    simpa only [weighted, linear, next, gainNativeStep, hclock j] using
      scalar_parameter_moment_clock_ceiling b1 b2 eps decay rate _ bound (state j) hb1 h1 he heta (hs j).2.1 hj hnext
  obtain ⟨hgap, hstrong⟩ := scalar_retained_input_adaptive_gap b1 b2 eps decay rate
    (appliedGainNativeGradient remaining genGain memGain bound state i) lower floor (state i)
    hb1 h1 (hs i).2.1 he heta hl hg hf hden
  have ha : 0 < (1 - b1) * eps := mul_pos (by linarith only [h1]) he
  have hscaled := mul_le_mul_of_nonneg_left hstrong (le_of_lt ha)
  have herror := mul_nonneg (mul_nonneg heta (le_of_lt hclip)) (pow_nonneg hb1 clock)
  have hstrongLinear : weighted i ≤ linear i - ((1 - b1) * eps) * rate * gap := by
    have hidentity : ((1 - b1) * eps) *
        ((1 - rate * decay) * (state i).parameter + rate * ((-(next i).moment) / eps - gap)) +
        rate * b1 * (-(next i).moment) =
          ((1 - b1) * eps) * (1 - rate * decay) * (state i).parameter +
            rate * (-(next i).moment) - ((1 - b1) * eps) * rate * gap := by
      field_simp [ne_of_gt he]
      ring
    change ((1 - b1) * eps) * (next i).parameter ≤
      ((1 - b1) * eps) * ((1 - rate * decay) * (state i).parameter +
        rate * ((-(next i).moment) / eps - gap)) at hscaled
    dsimp only [weighted, linear]
    nlinarith only [hscaled, hidentity, herror]
  have hsum : (weighted 0 + weighted 1) + (weighted 2 + weighted 3) ≤
      (linear 0 + linear 1) + (linear 2 + linear 3) - ((1 - b1) * eps) * rate * gap := by
    fin_cases i
    · change weighted 0 ≤ linear 0 - ((1 - b1) * eps) * rate * gap at hstrongLinear
      convert add_le_add (add_le_add hstrongLinear (hcoordinate 1)) (add_le_add (hcoordinate 2) (hcoordinate 3)) using 1
      ring
    · change weighted 1 ≤ linear 1 - ((1 - b1) * eps) * rate * gap at hstrongLinear
      convert add_le_add (add_le_add (hcoordinate 0) hstrongLinear) (add_le_add (hcoordinate 2) (hcoordinate 3)) using 1
      ring
    · change weighted 2 ≤ linear 2 - ((1 - b1) * eps) * rate * gap at hstrongLinear
      convert add_le_add (add_le_add (hcoordinate 0) (hcoordinate 1)) (add_le_add hstrongLinear (hcoordinate 3)) using 1
      ring
    · change weighted 3 ≤ linear 3 - ((1 - b1) * eps) * rate * gap at hstrongLinear
      convert add_le_add (add_le_add (hcoordinate 0) (hcoordinate 1)) (add_le_add (hcoordinate 2) hstrongLinear) using 1
      ring
  have hweighted : (weighted 0 + weighted 1) + (weighted 2 + weighted 3) = coldGainNativeWeightedMass b1 eps rate next := by
    dsimp only [weighted, coldGainNativeWeightedMass, gainNativeParameterMass, gainNativeNegativeMomentMass]
    ring
  have hlinear : (linear 0 + linear 1) + (linear 2 + linear 3) =
      ((1 - b1) * eps) * (1 - rate * decay) * gainNativeParameterMass state +
        rate * gainNativeNegativeMomentMass next + (4 * rate * bound) * b1 ^ clock := by
    dsimp only [linear, gainNativeParameterMass, gainNativeNegativeMomentMass]
    ring
  rw [hweighted, hlinear] at hsum
  have hm := mul_le_mul_of_nonneg_left
    (gain_native_cold_moment_mass_ceiling remaining genGain memGain bound b1 b2 eps decay rate state
      hgen hmem hgain hclip (le_of_lt h1) hs) heta
  have hthresholdScaled := mul_le_mul_of_nonneg_right hthreshold (gain_native_masses_nonnegative state hs).1
  have hfeedback := mul_le_mul_of_nonneg_left hthresholdScaled (mul_nonneg heta (show 0 ≤ 1 - b1 by linarith only [h1]))
  unfold coldGainNativeWeightedMass at hsum ⊢
  exact ⟨hgap, by nlinarith only [hsum, hm, hfeedback]⟩

example :
    let state := seededNativeSubweights ((0, 1 / 200), (0, 1))
    let lower := -appliedGainNativeGradient 0 3 2 1 state 0 / 2
    (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
      (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
      NonnegativeNativeState state ∧ (∀ j, -(state j).moment ≤ 1) ∧ (∀ j, (state j).clock = 0) ∧
      0 < lower ∧ appliedGainNativeGradient 0 3 2 1 state 0 ≤ -lower ∧ 1 < 1 + lower ∧
      1 + lower ≤ nextBufferDenominator 0 0 1 (state 0).variance
        (appliedGainNativeGradient 0 3 2 1 state 0) (state 0).clock ∧
      coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  dsimp only
  have hscale := gain_ce_gradient_scale_pos 0 3 2 1 (seededNativeSubweights ((0, 1 / 200), (0, 1))) (by norm_num)
  have hinput : appliedGainNativeGradient 0 3 2 1 (seededNativeSubweights ((0, 1 / 200), (0, 1))) 0 < 0 := by
    rw [gain_native_applied_gradient_scale]
    change -(gainCEGradientScale 0 3 2 1 (seededNativeSubweights ((0, 1 / 200), (0, 1))) * 3 * (1 / 200)) < 0
    nlinarith only [hscale]
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), ?_, ?_,
    by linarith only [hinput], by linarith only [hinput], by linarith only [hinput], ?_, by norm_num [coldGainCEGradientScale]⟩
  · intro j
    fin_cases j <;> norm_num [seededNativeSubweights, seededScalarState]
  · intro j
    fin_cases j <;> norm_num [seededNativeSubweights, seededScalarState]
  · simp only [nextBufferDenominator,
      zero_pow (by omega : (seededNativeSubweights ((0, 1 / 200), (0, 1)) 0).clock + 1 ≠ 0),
      zero_mul, sub_zero, one_mul, zero_add, div_one,
      Real.sqrt_sq_eq_abs, abs_of_nonpos (le_of_lt hinput)]
    linarith only [hinput]

end Transformer.Grokking.CircuitEfficiency
