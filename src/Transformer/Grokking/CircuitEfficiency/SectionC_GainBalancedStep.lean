import Transformer.Grokking.CircuitEfficiency.SectionC_GainFeedbackFloor
import Transformer.Grokking.AdamW.ScalarDenominator

/-!
# Actual balanced-pair native recurrence

Sources: Varma et al., arXiv:2309.02390v1, appendix C's two-factor
CE and section 3's competing circuits; native AdamW at lab commit
3695354. Derive the actual nonnegative zero-beta coordinate formula
and the reduced equal-factor recurrence using the true current CE
scale, retaining both buffer insertions and completed clocks.

Equal complete native states within each physical pair stay equal
under the actual gained CE at every finite clock, for arbitrary
fixed betas. The gains may differ between Gen and Mem. The seeded
zero-beta specialization therefore reduces to two real factor
amplitudes without prescribing either future scale or future margin.

Equal positive pair factors are a specialization, distinct from the
source's zero first factor and positive partner initialization. Zero
betas are a legal native configuration, differing from the preserved
nonzero-beta GPTMini runs. Fixed physical readouts, plain CE and
uniform decoupled decay differ from appendix C's assigned circuit
norm/GD model. This reduction alone proves no later selection.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Equality of full native states within each physical factor pair.
Sources: appendix C pair order and the gained native port at 3695354;
this predicate constrains parameters, buffers and clocks of its argument. -/
def NativePairSymmetry (state : NativeSubweightState) : Prop :=
  state 0 = state 1 ∧ state 2 = state 3

/-- Reduced actual zero-beta factor update at a current CE scale.
Sources: appendix C equal-factor chain rule and native AdamW at
3695354; correspondence with the actual feedback is proved below. -/
noncomputable def gainBalancedFactorStep (gain scale eps decay rate factor : ℝ) : ℝ :=
  (1 - rate * decay) * factor + rate * (scale * gain * factor) / (scale * gain * factor + eps)

/-- Actual nonnegative zero-beta coordinates use their current partner
and the full true clipped CE scale. Sources: appendix C's partials and
native AdamW at 3695354; no retained buffer or clock is reset by hand. -/
theorem gain_native_zero_beta_parameter (remaining : ℕ) (genGain memGain bound eps decay rate : ℝ)
    (state : NativeSubweightState) (i : Fin 4) (hgen : 0 < genGain) (hmem : 0 < memGain)
    (hclip : 0 < bound) (hp : ∀ j, 0 ≤ (state j).parameter) :
    (gainNativeStep remaining genGain memGain bound 0 0 eps decay rate state i).parameter =
      (1 - rate * decay) * (state i).parameter + rate *
        (gainCEGradientScale remaining genGain memGain bound state * nativeFactorGain genGain memGain i *
          (state (nativeFactorPartner i)).parameter) /
        (gainCEGradientScale remaining genGain memGain bound state * nativeFactorGain genGain memGain i *
          (state (nativeFactorPartner i)).parameter + eps) := by
  have hg := gain_native_applied_gradient_nonpos remaining genGain memGain bound state i hclip
    (le_of_lt (gain_native_factor_gain_pos genGain memGain hgen hmem i)) (hp _)
  change (scalarNativeStep 0 0 eps decay rate (state i) _).parameter = _
  rw [scalar_zero_betas_parameter, abs_of_nonpos hg, gain_native_applied_gradient_scale]
  ring

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧
    ∀ i : Fin 4, 0 ≤ (seededNativeSubweights ((1, 2), (3, 4)) i).parameter := by
  refine ⟨by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- True shared CE feedback preserves both full within-pair equalities.
Sources: appendix C equal partner derivatives and native AdamW at
3695354; this identity includes retained moments, variances and clocks. -/
theorem gain_native_pair_symmetric_step (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState) (hs : NativePairSymmetry state) :
    NativePairSymmetry (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) := by
  constructor
  · have hg : appliedGainNativeGradient remaining genGain memGain bound state 0 =
        appliedGainNativeGradient remaining genGain memGain bound state 1 := by
      rw [gain_native_applied_gradient_scale, gain_native_applied_gradient_scale]
      change -(gainCEGradientScale remaining genGain memGain bound state * genGain * (state 1).parameter) =
        -(gainCEGradientScale remaining genGain memGain bound state * genGain * (state 0).parameter)
      rw [hs.1]
    change scalarNativeStep b1 b2 eps decay rate (state 0) _ = scalarNativeStep b1 b2 eps decay rate (state 1) _
    rw [hs.1, hg]
  · have hg : appliedGainNativeGradient remaining genGain memGain bound state 2 =
        appliedGainNativeGradient remaining genGain memGain bound state 3 := by
      rw [gain_native_applied_gradient_scale, gain_native_applied_gradient_scale]
      change -(gainCEGradientScale remaining genGain memGain bound state * memGain * (state 3).parameter) =
        -(gainCEGradientScale remaining genGain memGain bound state * memGain * (state 2).parameter)
      rw [hs.2]
    change scalarNativeStep b1 b2 eps decay rate (state 2) _ = scalarNativeStep b1 b2 eps decay rate (state 3) _
    rw [hs.2, hg]

example : NativePairSymmetry (seededNativeSubweights ((1, 1), (2, 2))) := by
  exact ⟨rfl, rfl⟩

/-- Equal factor seeds keep full within-pair native equality at every
finite clock. Sources: appendix C paired factors and native CE at
3695354; actual buffers are carried through this entire induction. -/
theorem gain_native_equal_pair_path (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate a b : ℝ)
    (n : ℕ) : NativePairSymmetry (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate
      (seededNativeSubweights ((a, a), (b, b))) n) := by
  induction n with
  | zero => exact ⟨rfl, rfl⟩
  | succ n ih => exact gain_native_pair_symmetric_step remaining genGain memGain bound b1 b2 eps decay rate _ ih

/-- At a balanced present point, both actual zero-beta pair amplitudes
follow the reduced factor step with their shared actual CE multiplier.
Sources: appendix C product partials and native AdamW at 3695354. -/
theorem gain_native_zero_beta_balanced_step (remaining : ℕ) (genGain memGain bound eps decay rate : ℝ)
    (state : NativeSubweightState) (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound)
    (hp : ∀ i, 0 ≤ (state i).parameter) (hs : NativePairSymmetry state) :
    (gainNativeStep remaining genGain memGain bound 0 0 eps decay rate state 0).parameter =
        gainBalancedFactorStep genGain (gainCEGradientScale remaining genGain memGain bound state)
          eps decay rate (state 0).parameter ∧
      (gainNativeStep remaining genGain memGain bound 0 0 eps decay rate state 2).parameter =
        gainBalancedFactorStep memGain (gainCEGradientScale remaining genGain memGain bound state)
          eps decay rate (state 2).parameter := by
  rw [gain_native_zero_beta_parameter remaining genGain memGain bound eps decay rate state 0 hgen hmem hclip hp,
    gain_native_zero_beta_parameter remaining genGain memGain bound eps decay rate state 2 hgen hmem hclip hp]
  change _ = (1 - rate * decay) * (state 0).parameter + rate *
      (_ * genGain * (state 0).parameter) / (_ * genGain * (state 0).parameter + eps) ∧
    _ = (1 - rate * decay) * (state 2).parameter + rate *
      (_ * memGain * (state 2).parameter) / (_ * memGain * (state 2).parameter + eps)
  rw [hs.1, hs.2]
  exact ⟨rfl, rfl⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧
    (∀ i : Fin 4, 0 ≤ (seededNativeSubweights ((1, 1), (2, 2)) i).parameter) ∧
    NativePairSymmetry (seededNativeSubweights ((1, 1), (2, 2))) := by
  refine ⟨by norm_num, by norm_num, by norm_num, ?_, ⟨rfl, rfl⟩⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

/-- Initialized actual balanced paths obey both factor equations at
every clock. Sources: appendix C CE and native AdamW at 3695354;
nonnegative physical signs and pair symmetry are derived from the seeds. -/
theorem gain_native_seeded_balanced_recurrence (remaining : ℕ) (genGain memGain bound eps decay rate a b : ℝ)
    (n : ℕ) (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound) (he : 0 < eps)
    (heta : 0 ≤ rate) (hd : 0 ≤ 1 - rate * decay) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, a), (b, b)))
    (path (n + 1) 0).parameter = gainBalancedFactorStep genGain
        (gainCEGradientScale remaining genGain memGain bound (path n)) eps decay rate (path n 0).parameter ∧
      (path (n + 1) 2).parameter = gainBalancedFactorStep memGain
        (gainCEGradientScale remaining genGain memGain bound (path n)) eps decay rate (path n 2).parameter := by
  have hs := native_seeded_nonnegative a a b b ha ha hb hb
  have hn := gain_native_nonnegative_path remaining genGain memGain bound 0 0 eps decay rate
    (seededNativeSubweights ((a, a), (b, b))) n hgen hmem hclip
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) he heta hd hs
  exact gain_native_zero_beta_balanced_step remaining genGain memGain bound eps decay rate _ hgen hmem hclip
    (fun i => (hn i).1) (gain_native_equal_pair_path remaining genGain memGain bound 0 0 eps decay rate a b n)

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    (0 : ℝ) ≤ 1 / 200 ∧ (0 : ℝ) ≤ 1 := by
  norm_num

end Transformer.Grokking.CircuitEfficiency
