import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdReturns
import Transformer.Grokking.CircuitEfficiency.SectionC_GainFeedbackFloor
import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryCoefficients

/-!
# Uniform actual partner mixing at small retained native decay

Sources: Varma et al., arXiv:2309.02390v1, section 3 formation and
appendix C's physical product partials; native retained coefficients
at d3bb8b8, initialized parameter/buffer box at 3a44336, full actual
CE floor at ee02740 and weak-decay recurrence at lab commit e8766b6.

Keep the source-style Gen (0,seed), Mem (0,1) with nonnegative seed,
beta1=0.9, beta2=0.98, cap=1, epsilon=1e-8, decay=0.1 and rate=0.001.
The complete actual corrected denominator is at most eight whenever
the retained variance is at most one: clipping bounds the new input,
and the completed second correction is at least 1-beta2.

Initialized native bounds generate that variance ceiling at every
clock and a finite physical box. The actual clipped CE has a positive
numerical box floor. Combine it with the denominator ceiling and gain
at least two to give a fixed positive coefficient for every current
partner, without assuming future input or variance matching.

The original native parameter law keeps its nonnegative negative-
moment contribution and remaining pure-decay contribution. Both new
factors therefore cover a fixed positive fraction of the preceding
pair's full parameter sum. The fraction is below 1/40000 and may be
extremely small because of the loose exponential CE box estimate.
It is generated, rather than supplied as a trajectory premise.

This is uniform pair mixing, not equal-factor convergence, efficient
Gen selection or positive confidence. The next bridge must translate
recurrent physical mass into actual product logits. Fixed gained
tables/plain CE/uniform decoupled native decay differ from appendix C's
coupled norm-cost GD and learned stochastic/floating-point GPTMini.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Original small-decay initialized numerical path. Sources:
section 3's zero first factors and native cold regime at 6a99a48;
class count and partner seed enter the actual callback and initial data. -/
noncomputable def smallDecayGainPath (remaining : ℕ) (seed : ℝ) : ℕ → NativeSubweightState :=
  gainNativePath remaining 3 2 1 (9 / 10) (49 / 50) (1 / 100000000)
    (1 / 10) (1 / 1000) (seededNativeSubweights ((0, seed), (0, 1)))

/-- A retained variance ceiling gives a complete native denominator
ceiling, including both completed-clock corrections and actual input.
Sources: PyTorch AdamW denominator at 79f4fb0 and full clipping at
65ce284; no instantaneous variance/gradient match is supplied. -/
theorem small_decay_gain_denominator_ceiling (remaining : ℕ) (state : NativeSubweightState) (i : Fin 4)
    (hvariance : (state i).variance ≤ 1) :
    nextBufferDenominator (9 / 10) (49 / 50) (1 / 100000000) (state i).variance
      (appliedGainNativeGradient remaining 3 2 1 state i) (state i).clock ≤ 8 := by
  let gradient := appliedGainNativeGradient remaining 3 2 1 state i
  have hclip := gain_native_applied_gradient_abs_bound remaining 3 2 1 state i (by norm_num)
  have habs : -1 ≤ gradient ∧ gradient ≤ 1 := abs_le.mp hclip
  have hsquare : gradient ^ 2 ≤ 1 := by
    have hh := mul_nonneg (show 0 ≤ gradient + 1 by linarith only [habs.1])
      (show 0 ≤ 1 - gradient by linarith only [habs.2])
    nlinarith only [hh]
  have hp2 := pow_le_of_le_one (by norm_num : (0 : ℝ) ≤ 49 / 50)
    (by norm_num : (49 / 50 : ℝ) ≤ 1) (show (state i).clock + 1 ≠ 0 by omega)
  have hcorrection : 0 < 1 - (49 / 50 : ℝ) ^ ((state i).clock + 1) := by linarith only [hp2]
  have hratio : ((49 / 50 : ℝ) * (state i).variance + (1 - 49 / 50) * gradient ^ 2) /
      (1 - (49 / 50 : ℝ) ^ ((state i).clock + 1)) ≤ 50 := by
    apply (div_le_iff₀ hcorrection).mpr
    nlinarith only [hvariance, hsquare, hp2]
  have hsqrt : Real.sqrt (((49 / 50 : ℝ) * (state i).variance + (1 - 49 / 50) * gradient ^ 2) /
      (1 - (49 / 50 : ℝ) ^ ((state i).clock + 1))) ≤ 15 / 2 := by
    apply Real.sqrt_le_iff.mpr
    exact ⟨by norm_num, by nlinarith only [hratio]⟩
  have hp1 := pow_nonneg (by norm_num : (0 : ℝ) ≤ 9 / 10) ((state i).clock + 1)
  have hm := mul_le_mul_of_nonneg_right (show 1 - (9 / 10 : ℝ) ^ ((state i).clock + 1) ≤ 1 by linarith only [hp1])
    (show 0 ≤ Real.sqrt (((49 / 50 : ℝ) * (state i).variance + (1 - 49 / 50) * gradient ^ 2) /
      (1 - (49 / 50 : ℝ) ^ ((state i).clock + 1))) + 1 / 100000000 by positivity)
  dsimp only [nextBufferDenominator]
  change (1 - (9 / 10 : ℝ) ^ ((state i).clock + 1)) *
    (Real.sqrt (((49 / 50 : ℝ) * (state i).variance + (1 - 49 / 50) * gradient ^ 2) /
      (1 - (49 / 50 : ℝ) ^ ((state i).clock + 1))) + 1 / 100000000) ≤ 8
  nlinarith only [hm, hsqrt]

example : (seededNativeSubweights ((0, 1 / 200), (0, 1)) 0).variance ≤ 1 := by
  norm_num [seededNativeSubweights, seededScalarState]

/-- Initialized actual bounds generate a uniform positive partner
coefficient and all numerical signs at every clock. Sources: appendix C
product inputs and native box/CE/coefficients at 3a44336/ee02740/d3bb8b8;
the completed denominator ceiling is derived from retained variance. -/
theorem small_decay_gain_uniform_partner_floor (remaining : ℕ) (seed : ℝ) (hseed : 0 ≤ seed) :
    ∃ coefficient : ℝ, 0 < coefficient ∧ coefficient < 1 / 40000 ∧
      ∀ n, NonnegativeNativeState (smallDecayGainPath remaining seed n) ∧ ∀ i : Fin 4,
        coefficient ≤ gainNativePartnerCoefficient remaining 3 2 1 (9 / 10) (49 / 50)
          (1 / 100000000) (1 / 1000) (smallDecayGainPath remaining seed n) i := by
  obtain ⟨ceiling, hceiling, hbox⟩ := gain_native_seeded_bounded remaining 3 2 1
    (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000) 0 seed 0 1
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) le_rfl hseed le_rfl (by norm_num)
  let lower := gainCEFeedbackFloor remaining 3 2 1 ceiling
  have hpositive : 0 < lower := gain_ce_feedback_floor_pos remaining 3 2 1 ceiling
    (by norm_num) (le_of_lt hceiling) (by norm_num)
  have hscale : ∀ n, lower ≤ gainCEGradientScale remaining 3 2 1 (smallDecayGainPath remaining seed n) := by
    intro n
    exact gain_ce_gradient_scale_box_floor remaining 3 2 1 ceiling _
      (by norm_num) (by norm_num) (by norm_num) (le_of_lt hceiling) (by norm_num)
      (fun i => ⟨((hbox n).1 i).1, ((hbox n).2 i).1⟩)
  have hunit : lower < 1 := lt_of_le_of_lt (hscale 0)
    (gain_ce_gradient_scale_unit_interval remaining 3 2 1 (smallDecayGainPath remaining seed 0) (by norm_num)).2
  refine ⟨lower / 40000, by positivity, by nlinarith only [hunit], ?_⟩
  intro n
  refine ⟨(hbox n).1, ?_⟩
  intro i
  have hd := next_buffer_denominator_pos (9 / 10) (49 / 50) (1 / 100000000)
    (smallDecayGainPath remaining seed n i).variance
    (appliedGainNativeGradient remaining 3 2 1 (smallDecayGainPath remaining seed n) i)
    (smallDecayGainPath remaining seed n i).clock (by norm_num) (by norm_num) (by norm_num)
  have hupper := small_decay_gain_denominator_ceiling remaining (smallDecayGainPath remaining seed n) i
    (by simpa [smallDecayGainPath] using ((hbox n).2 i).2.2)
  have hgain : (2 : ℝ) ≤ nativeFactorGain 3 2 i := by fin_cases i <;> norm_num [nativeFactorGain]
  have hweighted := mul_le_mul_of_nonneg_left hgain
    (le_of_lt (gain_ce_gradient_scale_pos remaining 3 2 1 (smallDecayGainPath remaining seed n) (by norm_num)))
  have hden := mul_le_mul_of_nonneg_left hupper (show 0 ≤ lower / 40000 by positivity)
  unfold gainNativePartnerCoefficient
  apply (le_div_iff₀ hd).mpr
  nlinarith only [hden, hweighted, hscale n]

example : (0 : ℝ) ≤ 1 / 200 := by norm_num

/-- Each original next physical factor covers a fixed positive
fraction of its preceding complete pair sum. Sources: appendix C
partner partials and actual retained parameter law at d3bb8b8;
moments, variances and growing correction clocks are not reset. -/
theorem small_decay_gain_uniform_pair_mixing (remaining : ℕ) (seed : ℝ) (hseed : 0 ≤ seed) :
    ∃ coefficient : ℝ, 0 < coefficient ∧ coefficient < 1 / 40000 ∧
      ∀ n, NonnegativeNativeState (smallDecayGainPath remaining seed n) ∧ ∀ i : Fin 4,
        coefficient * ((smallDecayGainPath remaining seed n i).parameter +
          (smallDecayGainPath remaining seed n (nativeFactorPartner i)).parameter) ≤
            (smallDecayGainPath remaining seed (n + 1) i).parameter := by
  obtain ⟨coefficient, hc, hsmall, hfloor⟩ := small_decay_gain_uniform_partner_floor remaining seed hseed
  refine ⟨coefficient, hc, hsmall, ?_⟩
  intro n
  refine ⟨(hfloor n).1, ?_⟩
  intro i
  let state := smallDecayGainPath remaining seed n
  have hs : NonnegativeNativeState state := (hfloor n).1
  have hd := next_buffer_denominator_pos (9 / 10) (49 / 50) (1 / 100000000) (state i).variance
    (appliedGainNativeGradient remaining 3 2 1 state i) (state i).clock (by norm_num) (by norm_num) (by norm_num)
  have hm : 0 ≤ gainNativeMomentCoefficient remaining 3 2 1 (9 / 10) (49 / 50) (1 / 100000000) (1 / 1000) state i := by
    unfold gainNativeMomentCoefficient
    exact div_nonneg (by norm_num) (le_of_lt hd)
  have hmemory := mul_nonneg hm (show 0 ≤ -(state i).moment by linarith only [(hs i).2.1])
  have hown := mul_nonneg (show 0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) - coefficient by linarith only [hsmall]) (hs i).1
  have hpartner := mul_le_mul_of_nonneg_right ((hfloor n).2 i) (hs (nativeFactorPartner i)).1
  have heq := gain_native_parameter_memory_partner remaining 3 2 1 (9 / 10) (49 / 50)
    (1 / 100000000) (1 / 10) (1 / 1000) state i
  change coefficient * ((state i).parameter + (state (nativeFactorPartner i)).parameter) ≤
    (gainNativeStep remaining 3 2 1 (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000) state i).parameter
  nlinarith only [hmemory, hown, hpartner, heq]

example : (0 : ℝ) ≤ 1 / 200 := by norm_num

end Transformer.Grokking.CircuitEfficiency
