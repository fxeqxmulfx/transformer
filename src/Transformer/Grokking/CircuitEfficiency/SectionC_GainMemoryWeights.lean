import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryCoefficients

/-!
# Exact pair weights for the original native retained-memory update

Sources: Varma et al., arXiv:2309.02390v1, appendix C product factors
and section 3 competing circuits; native coefficients at lab commit
85a2d66. Add a numerical multiple of the negative first-moment sum
to a physical pair's parameter sum. This keeps both retained moments,
current full denominators, shared clipping/CE, second moments and clocks.

The actual next weight is a positive-coordinate linear combination
with computed current coefficients. Bounds on those coefficients give
bounds on the original next pair weight, not on a replaced gradient
stream. Numerical nonnegative signs and coefficient intervals are current
hypotheses here; their initialized tails must be generated separately.

A pair weight is a measurement of the original path, not a new optimizer
or a success predicate. Its dominance does not imply a physical product
margin without additional pair balance or individual-factor bounds.
Fixed gained tables and uniform decoupled native decay differ from
appendix C's assigned circuit cost/GD. No learned GPTMini transfer,
finite precision statement or thermodynamic limit is asserted.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Current parameter sum plus weighted negative retained moments of
one true partner pair. Sources: appendix C factors and native buffers
at 85a2d66; every argument occurs in this numerical measurement. -/
def gainNativePairMemoryMass (weight : ℝ) (state : NativeSubweightState) (i : Fin 4) : ℝ :=
  (state i).parameter + (state (nativeFactorPartner i)).parameter +
    weight * (-(state i).moment - (state (nativeFactorPartner i)).moment)

/-- Current negative-moment coefficient after adding the retained
moment measurement. Source: exact native coefficients at 85a2d66. -/
noncomputable def gainNativeWeightedMomentCoefficient (remaining : ℕ)
    (genGain memGain bound b1 b2 eps rate weight : ℝ) (state : NativeSubweightState) (i : Fin 4) : ℝ :=
  gainNativeMomentCoefficient remaining genGain memGain bound b1 b2 eps rate state i + weight * b1

/-- Current partner coefficient including the weight's actual new
moment insertion and remaining decay. Sources: appendix C actual
partials and native coefficients at 85a2d66. -/
noncomputable def gainNativeWeightedParameterCoefficient (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate weight : ℝ) (state : NativeSubweightState) (i : Fin 4) : ℝ :=
  (1 - rate * decay) + gainNativePartnerCoefficient remaining genGain memGain bound b1 b2 eps rate state i +
    weight * (1 - b1) * gainCEGradientScale remaining genGain memGain bound state * nativeFactorGain genGain memGain i

/-- The four concrete product partners form an involution. Source:
appendix C's two separate factor pairs; this is a finite index identity. -/
theorem gain_native_partner_involution (i : Fin 4) : nativeFactorPartner (nativeFactorPartner i) = i := by
  fin_cases i <;> rfl

/-- Exact original next pair weight in present parameters and
retained first moments. Sources: appendix C product partials and
native update at 85a2d66; no denominator approximation is made. -/
theorem gain_native_pair_memory_step (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate weight : ℝ) (state : NativeSubweightState) (i : Fin 4) :
    gainNativePairMemoryMass weight (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) i =
      gainNativeWeightedParameterCoefficient remaining genGain memGain bound b1 b2 eps decay rate weight state (nativeFactorPartner i) *
        (state i).parameter +
      gainNativeWeightedParameterCoefficient remaining genGain memGain bound b1 b2 eps decay rate weight state i *
        (state (nativeFactorPartner i)).parameter +
      gainNativeWeightedMomentCoefficient remaining genGain memGain bound b1 b2 eps rate weight state i * (-(state i).moment) +
      gainNativeWeightedMomentCoefficient remaining genGain memGain bound b1 b2 eps rate weight state (nativeFactorPartner i) *
        (-(state (nativeFactorPartner i)).moment) := by
  have hparameter := gain_native_parameter_memory_partner remaining genGain memGain bound b1 b2 eps decay rate state
  unfold gainNativePairMemoryMass
  rw [hparameter i, hparameter (nativeFactorPartner i), gain_native_partner_involution]
  have hmoment : ∀ j, (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state j).moment =
      b1 * (state j).moment - (1 - b1) * gainCEGradientScale remaining genGain memGain bound state *
        nativeFactorGain genGain memGain j * (state (nativeFactorPartner j)).parameter := by
    intro j
    dsimp only [gainNativeStep, scalarNativeStep]
    rw [gain_native_applied_gradient_scale]
    ring
  rw [hmoment i, hmoment (nativeFactorPartner i), gain_native_partner_involution]
  unfold gainNativeWeightedParameterCoefficient gainNativeWeightedMomentCoefficient
  ring

/-- A nonnegative numerical weight covers its true physical pair
mass. Sources: appendix C parameters and native signs at 85a2d66;
no positive or convergent future weight is supplied. -/
theorem gain_native_pair_memory_mass_covers (weight : ℝ) (state : NativeSubweightState) (i : Fin 4)
    (hw : 0 ≤ weight) (hs : NonnegativeNativeState state) :
    0 ≤ (state i).parameter + (state (nativeFactorPartner i)).parameter ∧
      (state i).parameter + (state (nativeFactorPartner i)).parameter ≤ gainNativePairMemoryMass weight state i := by
  have hn : 0 ≤ -(state i).moment - (state (nativeFactorPartner i)).moment := by
    linarith only [(hs i).2.1, (hs (nativeFactorPartner i)).2.1]
  have hp := mul_nonneg hw hn
  unfold gainNativePairMemoryMass
  exact ⟨add_nonneg (hs i).1 (hs (nativeFactorPartner i)).1, by linarith only [hp]⟩

example : (0 : ℝ) ≤ 2 / 25 ∧ NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨by norm_num, native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)⟩

/-- Current coefficient intervals bound the original next pair
weight on both sides. Sources: exact native pair law at 85a2d66 and
appendix C coordinates; interval generation along a path is separate.
No sign of the comparison factors is needed for this current law. -/
theorem gain_native_pair_memory_comparison (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate weight lower upper : ℝ) (state : NativeSubweightState) (i : Fin 4)
    (hs : NonnegativeNativeState state)
    (hp : ∀ j, j = i ∨ j = nativeFactorPartner i →
      lower ≤ gainNativeWeightedParameterCoefficient remaining genGain memGain bound b1 b2 eps decay rate weight state j ∧
      gainNativeWeightedParameterCoefficient remaining genGain memGain bound b1 b2 eps decay rate weight state j ≤ upper)
    (hm : ∀ j, j = i ∨ j = nativeFactorPartner i →
      lower * weight ≤ gainNativeWeightedMomentCoefficient remaining genGain memGain bound b1 b2 eps rate weight state j ∧
      gainNativeWeightedMomentCoefficient remaining genGain memGain bound b1 b2 eps rate weight state j ≤ upper * weight) :
    lower * gainNativePairMemoryMass weight state i ≤
      gainNativePairMemoryMass weight (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) i ∧
      gainNativePairMemoryMass weight (gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate state) i ≤
        upper * gainNativePairMemoryMass weight state i := by
  have hpi := hp i (Or.inl rfl)
  have hpj := hp (nativeFactorPartner i) (Or.inr rfl)
  have hmi := hm i (Or.inl rfl)
  have hmj := hm (nativeFactorPartner i) (Or.inr rfl)
  have hlo := add_le_add (add_le_add (mul_le_mul_of_nonneg_right hpj.1 (hs i).1)
    (mul_le_mul_of_nonneg_right hpi.1 (hs (nativeFactorPartner i)).1))
    (add_le_add (mul_le_mul_of_nonneg_right hmi.1 (show 0 ≤ -(state i).moment by linarith only [(hs i).2.1]))
      (mul_le_mul_of_nonneg_right hmj.1 (show 0 ≤ -(state (nativeFactorPartner i)).moment by linarith only [(hs (nativeFactorPartner i)).2.1])))
  have hhi := add_le_add (add_le_add (mul_le_mul_of_nonneg_right hpj.2 (hs i).1)
    (mul_le_mul_of_nonneg_right hpi.2 (hs (nativeFactorPartner i)).1))
    (add_le_add (mul_le_mul_of_nonneg_right hmi.2 (show 0 ≤ -(state i).moment by linarith only [(hs i).2.1]))
      (mul_le_mul_of_nonneg_right hmj.2 (show 0 ≤ -(state (nativeFactorPartner i)).moment by linarith only [(hs (nativeFactorPartner i)).2.1])))
  rw [gain_native_pair_memory_step]
  unfold gainNativePairMemoryMass
  constructor <;> nlinarith only [hlo, hhi]

example :
    let state := seededNativeSubweights ((0, 0), (0, 0))
    NonnegativeNativeState state ∧
    (∀ j : Fin 4, j = 0 ∨ j = nativeFactorPartner 0 →
      (9 / 10 : ℝ) ≤ gainNativeWeightedParameterCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) (2 / 25) state j ∧
      gainNativeWeightedParameterCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) (2 / 25) state j ≤ 51 / 50) ∧
    (∀ j : Fin 4, j = 0 ∨ j = nativeFactorPartner 0 →
      (9 / 10 : ℝ) * (2 / 25) ≤ gainNativeWeightedMomentCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 (1 / 1000) (2 / 25) state j ∧
      gainNativeWeightedMomentCoefficient 0 3 2 1 (9 / 10) (49 / 50) 1 (1 / 1000) (2 / 25) state j ≤ (51 / 50) * (2 / 25)) := by
  dsimp only
  have hs : NonnegativeNativeState (seededNativeSubweights ((0, 0), (0, 0))) :=
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  have hc := gain_ce_gradient_scale_zero_parameters 0 3 2 1 (seededNativeSubweights ((0, 0), (0, 0))) (by intro j; fin_cases j <;> rfl)
  norm_num [coldGainCEGradientScale] at hc
  have hg : ∀ j, appliedGainNativeGradient 0 3 2 1 (seededNativeSubweights ((0, 0), (0, 0))) j = 0 := by
    intro j
    rw [gain_native_applied_gradient_scale, hc]
    fin_cases j <;> norm_num [nativeFactorPartner, seededNativeSubweights, seededScalarState]
  refine ⟨hs, ?_, ?_⟩ <;> intro j hj <;>
    simp only [gainNativeWeightedParameterCoefficient, gainNativeWeightedMomentCoefficient,
      gainNativePartnerCoefficient, gainNativeMomentCoefficient, hc, hg] <;> fin_cases j <;>
    norm_num [nextBufferDenominator, nativeFactorGain, nativeFactorPartner, seededNativeSubweights, seededScalarState] at *

end Transformer.Grokking.CircuitEfficiency
