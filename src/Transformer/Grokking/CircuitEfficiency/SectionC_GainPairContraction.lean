import Transformer.Grokking.CircuitEfficiency.SectionC_GainPairDifference
import Mathlib.Analysis.SpecificLimits.Basic

/-!
# Actual unbalanced native pair difference decay

Sources: Varma et al., arXiv:2309.02390v1, section 3's unbalanced
zero-first-factor initialization and appendix C's true product CE;
native zero-beta reduction at lab commit dadb5ac.

Identify both unequal physical pairs with their true partner-input
updates, deriving the shared scale from actual clipped CE. A fixed
sufficient rate condition makes both absolute differences contract
at every step. Initialize nonnegative retained states and derive
the entire native path bound kappa^n times the initial difference.
At positive rate/decay both differences therefore tend to zero,
without requiring convergence of any individual physical parameter.

This admits the source's zero-first-factor seeds and does not supply
future balance or an independent gradient callback. Absolute balance
alone does not prove positive mass or successful held-out selection:
both factors could shrink together. Relative balance/competition
remain separate obligations. Fixed gained tables, legal zero betas,
exact reals and uniform native decay differ from coupled-cost GD
and preserved learned GPTMini with nonzero betas.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW Filter

/-- Both actual native unequal pairs equal their explicit partner
updates. Sources: appendix C partials and zero-beta native coordinates
at dadb5ac; retained buffers/clocks are unrestricted at this point. -/
theorem gain_native_zero_beta_pair_step (remaining : ℕ) (genGain memGain bound eps decay rate : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hclip : 0 < bound) (hp : ∀ i, 0 ≤ (state i).parameter) :
    let next := gainNativeStep remaining genGain memGain bound 0 0 eps decay rate state
    let scale := gainCEGradientScale remaining genGain memGain bound state
    (next 0 |>.parameter, next 1 |>.parameter) =
        gainPairStep genGain scale eps decay rate (state 0).parameter (state 1).parameter ∧
      (next 2 |>.parameter, next 3 |>.parameter) =
        gainPairStep memGain scale eps decay rate (state 2).parameter (state 3).parameter := by
  constructor
  · apply Prod.ext
    · exact gain_native_zero_beta_parameter remaining genGain memGain bound eps decay rate state 0 hgen hmem hclip hp
    · exact gain_native_zero_beta_parameter remaining genGain memGain bound eps decay rate state 1 hgen hmem hclip hp
  · apply Prod.ext
    · exact gain_native_zero_beta_parameter remaining genGain memGain bound eps decay rate state 2 hgen hmem hclip hp
    · exact gain_native_zero_beta_parameter remaining genGain memGain bound eps decay rate state 3 hgen hmem hclip hp

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧
    ∀ i : Fin 4, 0 ≤ (seededNativeSubweights ((0, 1 / 200), (0, 1)) i).parameter := by
  refine ⟨by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- True clipped native feedback contracts both physical differences
under a fixed initial-data-independent sufficient rate condition.
Sources: appendix C true CE and native AdamW at dadb5ac; the shared
scale/unit bound is derived from the actual forward, not assumed. -/
theorem gain_native_pair_difference_contraction (remaining : ℕ) (genGain memGain bound eps decay rate : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (he : 0 < eps) (heta : 0 ≤ rate)
    (hp : ∀ i, 0 ≤ (state i).parameter) (hsmall : rate * (decay + genGain / eps) ≤ 1) :
    let next := gainNativeStep remaining genGain memGain bound 0 0 eps decay rate state
    |(next 0).parameter - (next 1).parameter| ≤ (1 - rate * decay) * |(state 0).parameter - (state 1).parameter| ∧
      |(next 2).parameter - (next 3).parameter| ≤ (1 - rate * decay) * |(state 2).parameter - (state 3).parameter| := by
  have hscale := gain_ce_gradient_scale_unit_interval remaining genGain memGain bound state hclip
  have hform := gain_native_zero_beta_pair_step remaining genGain memGain bound eps decay rate state hgen hmem hclip hp
  have hweighted := mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right hgain (le_of_lt he)) heta
  have hmSmall : rate * (decay + memGain / eps) ≤ 1 := by nlinarith only [hweighted, hsmall]
  have hg := gain_pair_step_difference_contraction genGain _ eps decay rate _ _
    (le_of_lt hgen) (le_of_lt hscale.1) (le_of_lt hscale.2) he heta (hp 0) (hp 1) hsmall
  have hm := gain_pair_step_difference_contraction memGain _ eps decay rate _ _
    (le_of_lt hmem) (le_of_lt hscale.1) (le_of_lt hscale.2) he heta (hp 2) (hp 3) hmSmall
  rw [← hform.1] at hg
  rw [← hform.2] at hm
  exact ⟨hg, hm⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ (∀ i : Fin 4, 0 ≤ (seededNativeSubweights ((0, 1 / 200), (0, 1)) i).parameter) ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_, by norm_num⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- Initial nonnegative retained data generate the full physical
difference ceiling at every native clock. Sources: section 3's
unbalanced seeds and actual feedback at dadb5ac; no future trajectory,
pair symmetry, parameter bound or convergence is supplied as a premise. -/
theorem gain_native_pair_difference_path_bound (remaining : ℕ) (genGain memGain bound eps decay rate : ℝ)
    (initial : NativeSubweightState) (n : ℕ) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (he : 0 < eps) (heta : 0 ≤ rate)
    (hs : NonnegativeNativeState initial) (hsmall : rate * (decay + genGain / eps) ≤ 1) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate initial
    |(path n 0).parameter - (path n 1).parameter| ≤
        (1 - rate * decay) ^ n * |(initial 0).parameter - (initial 1).parameter| ∧
      |(path n 2).parameter - (path n 3).parameter| ≤
        (1 - rate * decay) ^ n * |(initial 2).parameter - (initial 3).parameter| := by
  have hpart := mul_nonneg heta (div_nonneg (le_of_lt hgen) (le_of_lt he))
  have hd : 0 ≤ 1 - rate * decay := by nlinarith only [hsmall, hpart]
  induction n with
  | zero => simp only [gainNativePath, pow_zero, one_mul, le_refl, and_self]
  | succ n ih =>
    let state := gainNativePath remaining genGain memGain bound 0 0 eps decay rate initial n
    have hn := gain_native_nonnegative_path remaining genGain memGain bound 0 0 eps decay rate initial n
      hgen hmem hclip (by norm_num) (by norm_num) (by norm_num) (by norm_num) he heta hd hs
    have hstep := gain_native_pair_difference_contraction remaining genGain memGain bound eps decay rate state
      hgen hmem hgain hclip he heta (fun i => (hn i).1) hsmall
    have hg := le_trans hstep.1 (mul_le_mul_of_nonneg_left ih.1 hd)
    have hm := le_trans hstep.2 (mul_le_mul_of_nonneg_left ih.2 hd)
    constructor
    · simpa only [gainNativePath, pow_succ, mul_assoc, mul_comm, mul_left_comm] using hg
    · simpa only [gainNativePath, pow_succ, mul_assoc, mul_comm, mul_left_comm] using hm

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), by norm_num⟩

/-- Both actual physical pair differences tend to zero at positive
native decay/rate, without finite limits for individual parameters.
Sources: section 3's unbalanced factors and actual recurrence at
dadb5ac; absolute pair balance alone is not positive mass or test success. -/
theorem gain_native_pair_difference_tendsto_zero (remaining : ℕ) (genGain memGain bound eps decay rate : ℝ)
    (initial : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hgain : memGain ≤ genGain) (hclip : 0 < bound) (he : 0 < eps)
    (hdecay : 0 < decay) (heta : 0 < rate) (hs : NonnegativeNativeState initial)
    (hsmall : rate * (decay + genGain / eps) ≤ 1) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate initial
    Tendsto (fun n => |(path n 0).parameter - (path n 1).parameter|) atTop (nhds 0) ∧
      Tendsto (fun n => |(path n 2).parameter - (path n 3).parameter|) atTop (nhds 0) := by
  have hpart := mul_nonneg (le_of_lt heta) (div_nonneg (le_of_lt hgen) (le_of_lt he))
  have hd : 0 ≤ 1 - rate * decay := by nlinarith only [hsmall, hpart]
  have hu : 1 - rate * decay < 1 := by nlinarith only [mul_pos heta hdecay]
  have hpow := tendsto_pow_atTop_nhds_zero_of_lt_one hd hu
  have hpath := fun n => gain_native_pair_difference_path_bound remaining genGain memGain bound eps decay rate initial n
    hgen hmem hgain hclip he (le_of_lt heta) hs hsmall
  constructor
  · apply squeeze_zero (fun n => abs_nonneg _) (fun n => (hpath n).1)
    simpa using hpow.mul_const (|(initial 0).parameter - (initial 1).parameter|)
  · apply squeeze_zero (fun n => abs_nonneg _) (fun n => (hpath n).2)
    simpa using hpow.mul_const (|(initial 2).parameter - (initial 3).parameter|)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), by norm_num⟩

end Transformer.Grokking.CircuitEfficiency
