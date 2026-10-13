import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryMass
import Transformer.Grokking.AdamW.PairComparison

/-!
# Actual competing pair envelopes with retained native moments

Sources: Varma et al., arXiv:2309.02390v1, section 3's Gen/Mem
competition and appendix C's product CE; actual Gen envelope at
2150464, full native mass at ada36f3 and comparison at 3bab55a.

Define Mem's parameter sum and negative retained first-moment sum.
Its exact new moment inserts the same actual clipped CE scale as Gen,
times Mem gain and current Mem parameter mass. An actual complete
denominator floor bounds Mem's adaptive parameter increment above.
An actual Gen denominator ceiling gives the opposing lower envelope,
now in the divided form needed by the retained comparison theorem.

The supplied bounds concern current denominators, including variance
and completed-clock bias corrections. Future bounds, moment matching,
parameter limits and successful outputs are not premises or conclusions
of these current numerical laws. A subsequent application must derive
the bounds from the actual initialized path before iterating them.

Fixed gained tables/plain CE/uniform native decoupled decay differ
from the source's coupled assigned norm cost and gradient descent.
Both native buffers remain untouched; no replacement optimizer,
learned GPTMini transfer or floating-point guarantee is asserted.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Mem's current physical parameter sum. Sources: appendix C's
two Mem factors and native actual masses at ada36f3. -/
def gainMemParameterMass (state : NativeSubweightState) : ℝ :=
  (state 2).parameter + (state 3).parameter

/-- Negative sum of Mem's actual retained first moments. Sources:
native insertion at ada36f3 and appendix C's two Mem factors. -/
def gainMemNegativeMomentMass (state : NativeSubweightState) : ℝ :=
  -((state 2).moment + (state 3).moment)

/-- Present numerical signs make both Mem masses nonnegative.
Sources: appendix C factors and native sign region at 65ce284;
this says nothing about their future size or trained dominance. -/
theorem gain_mem_masses_nonnegative (state : NativeSubweightState) (hs : NonnegativeNativeState state) :
    0 ≤ gainMemParameterMass state ∧ 0 ≤ gainMemNegativeMomentMass state := by
  unfold gainMemParameterMass gainMemNegativeMomentMass
  constructor
  · linarith only [(hs 2).1, (hs 3).1]
  · linarith only [(hs 2).2.1, (hs 3).2.1]

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) := by
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- The exact Mem moment insertion uses current true CE feedback.
Sources: appendix C partner derivatives and retained native insertion
at ada36f3; both original first moments survive the step. -/
theorem gain_mem_moment_mass_step (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState) :
    gainMemNegativeMomentMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) =
      b1 * gainMemNegativeMomentMass state + (1 - b1) *
        (gainCEGradientScale remaining genGain memGain bound state * memGain) * gainMemParameterMass state := by
  unfold gainMemNegativeMomentMass gainNativeStep
  simp only [scalarNativeStep, gain_native_applied_gradient_scale]
  change -((b1 * (state 2).moment + (1 - b1) *
    -(gainCEGradientScale remaining genGain memGain bound state * memGain * (state 3).parameter)) +
    (b1 * (state 3).moment + (1 - b1) *
      -(gainCEGradientScale remaining genGain memGain bound state * memGain * (state 2).parameter))) = _
  unfold gainMemParameterMass
  ring

/-- A floor for both actual Mem denominators gives its retained
parameter-mass ceiling. Sources: appendix C products and native
denominator formula at 79f4fb0; no instantaneous variance match is used. -/
theorem gain_mem_parameter_mass_floor_denominator (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate floor : ℝ) (state : NativeSubweightState)
    (hmem : 0 ≤ memGain) (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 ≤ 1)
    (heta : 0 ≤ rate) (hd : 0 < floor) (hs : NonnegativeNativeState state)
    (hd2 : floor ≤ nextBufferDenominator b1 b2 eps (state 2).variance
      (appliedGainNativeGradient remaining genGain memGain bound state 2) (state 2).clock)
    (hd3 : floor ≤ nextBufferDenominator b1 b2 eps (state 3).variance
      (appliedGainNativeGradient remaining genGain memGain bound state 3) (state 3).clock) :
    gainMemParameterMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) ≤
      (1 - rate * decay) * gainMemParameterMass state + rate *
        (gainMemNegativeMomentMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) / floor) := by
  have hcoordinate : ∀ i : Fin 4, i = 2 ∨ i = 3 →
      (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state i).parameter ≤
        (1 - rate * decay) * (state i).parameter + rate *
          (-(gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state i).moment) / floor := by
    intro i hi
    have hg : appliedGainNativeGradient remaining genGain memGain bound state i ≤ 0 := by
      apply gain_native_applied_gradient_nonpos remaining genGain memGain bound state i hclip
      · rcases hi with rfl | rfl <;> exact hmem
      · exact (hs _).1
    have hn := scalar_native_moment_nonpos b1 b2 eps decay rate _ (state i) hb1 h1 (hs i).2.1 hg
    have hdi : floor ≤ nextBufferDenominator b1 b2 eps (state i).variance
        (appliedGainNativeGradient remaining genGain memGain bound state i) (state i).clock := by
      rcases hi with rfl | rfl
      · exact hd2
      · exact hd3
    have hdiv := div_le_div_of_nonneg_left (mul_nonneg heta (neg_nonneg.mpr hn)) hd hdi
    rw [show gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state i =
      scalarNativeStep b1 b2 eps decay rate (state i) (appliedGainNativeGradient remaining genGain memGain bound state i) from rfl,
      scalar_parameter_denominator]
    linarith only [hdiv]
  have hh := add_le_add (hcoordinate 2 (Or.inl rfl)) (hcoordinate 3 (Or.inr rfl))
  convert hh using 1
  · rfl
  · unfold gainMemParameterMass gainMemNegativeMomentMass
    ring

example :
    let state := seededNativeSubweights ((0, 1 / 200), (0, 1))
    (0 : ℝ) ≤ 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) ≤ 1 ∧
      (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) < 1 / 10 ∧ NonnegativeNativeState state ∧
      (1 / 10 : ℝ) ≤ nextBufferDenominator (9 / 10) (49 / 50) 1 (state 2).variance
        (appliedGainNativeGradient 0 3 2 1 state 2) (state 2).clock ∧
      (1 / 10 : ℝ) ≤ nextBufferDenominator (9 / 10) (49 / 50) 1 (state 3).variance
        (appliedGainNativeGradient 0 3 2 1 state 3) (state 3).clock := by
  dsimp only
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), ?_, ?_⟩
  · convert next_buffer_denominator_floor (9 / 10) (49 / 50) 1
      (seededNativeSubweights ((0, 1 / 200), (0, 1)) 2).variance
      (appliedGainNativeGradient 0 3 2 1 (seededNativeSubweights ((0, 1 / 200), (0, 1))) 2)
      (seededNativeSubweights ((0, 1 / 200), (0, 1)) 2).clock (by norm_num) (by norm_num) (by norm_num) using 1; norm_num
  · convert next_buffer_denominator_floor (9 / 10) (49 / 50) 1
      (seededNativeSubweights ((0, 1 / 200), (0, 1)) 3).variance
      (appliedGainNativeGradient 0 3 2 1 (seededNativeSubweights ((0, 1 / 200), (0, 1))) 3)
      (seededNativeSubweights ((0, 1 / 200), (0, 1)) 3).clock (by norm_num) (by norm_num) (by norm_num) using 1; norm_num

/-- The actual Gen denominator ceiling gives its divided retained
parameter envelope. Sources: appendix C and native Gen mass floor
at 2150464; this is the same original update with a positive divisor. -/
theorem gain_gen_parameter_mass_ceiling_denominator (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate ceiling : ℝ) (state : NativeSubweightState)
    (hgen : 0 < genGain) (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1)
    (he : 0 < eps) (heta : 0 ≤ rate) (hd : 0 < ceiling) (hs : NonnegativeNativeState state)
    (hd0 : nextBufferDenominator b1 b2 eps (state 0).variance
      (appliedGainNativeGradient remaining genGain memGain bound state 0) (state 0).clock ≤ ceiling)
    (hd1 : nextBufferDenominator b1 b2 eps (state 1).variance
      (appliedGainNativeGradient remaining genGain memGain bound state 1) (state 1).clock ≤ ceiling) :
    (1 - rate * decay) * gainGenParameterMass state + rate *
      (gainGenNegativeMomentMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) / ceiling) ≤
        gainGenParameterMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) := by
  have hh := gain_gen_parameter_mass_floor remaining genGain memGain bound b1 b2 eps decay rate ceiling state
    hgen hclip hb1 h1 he heta hd hs hd0 hd1
  calc
    _ = (ceiling * (1 - rate * decay) * gainGenParameterMass state + rate *
        gainGenNegativeMomentMass (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state)) / ceiling := by
      field_simp [ne_of_gt hd]
    _ ≤ _ := (div_le_iff₀ hd).mpr (by nlinarith only [hh])

example :
    let state := seededNativeSubweights ((0, 0), (0, 0))
    (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧
      (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) < 2 ∧ NonnegativeNativeState state ∧
      nextBufferDenominator 0 0 1 (state 0).variance (appliedGainNativeGradient 0 3 2 1 state 0) (state 0).clock ≤ 2 ∧
      nextBufferDenominator 0 0 1 (state 1).variance (appliedGainNativeGradient 0 3 2 1 state 1) (state 1).clock ≤ 2 := by
  dsimp only
  have hg : ∀ i, appliedGainNativeGradient 0 3 2 1 (seededNativeSubweights ((0, 0), (0, 0))) i = 0 := by
    intro i
    rw [gain_native_applied_gradient_scale]
    fin_cases i <;> norm_num [nativeFactorGain, nativeFactorPartner, seededNativeSubweights, seededScalarState]
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num), ?_, ?_⟩ <;>
    simp only [hg] <;> norm_num [nextBufferDenominator, seededNativeSubweights, seededScalarState]

end Transformer.Grokking.CircuitEfficiency
