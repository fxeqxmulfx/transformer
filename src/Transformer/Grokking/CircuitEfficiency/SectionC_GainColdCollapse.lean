import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdContraction
import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdInstability

/-!
# Original cold parameter/input limits and strict threshold classification

Sources: Varma et al., arXiv:2309.02390v1, section 3's CE/decay
competition and appendix C product CE; actual strict collapse at
b750b29 and initialized opposite cold obstruction at 6a99a48.

Above the strict cold threshold, generated nonnegative total parameter
mass squeezes every actual physical coordinate and every actual clipped
CE input to zero. The original retained buffers and correction clocks
are unchanged; their limits can now be derived from these inputs.

For positive initialized Gen data, greater Gen gain, legal beta memory,
positive rate/decay/epsilon and strictly positive remaining decay, the
two directions give one classification away from critical equality:
total physical parameter mass tends to zero exactly when decay times
epsilon is greater than cold coefficient times Gen gain.

The positive Gen partner excludes the exact absent-Gen invariant,
whose training improvement does not establish held-out progress.
Class count and the native clip strip remain in the threshold;
neither gain nor beta is replaced by a fitted growth exponent.

The excluded equality is explicit. Below threshold, failure to tend
to zero does not assert a positive limit, stable Gen dominance or
held-out success. This is a fixed-table exact-real native result;
learned stochastic/numerical GPTMini and thermodynamic size scaling
are separate. Uniform native decay/plain CE differ from appendix C's
assigned coupled norm-cost GD. No optimizer or checkpoint is modified.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Every actual physical parameter tends to zero above the strict
cold threshold. Sources: appendix C coordinates and actual retained
mass collapse at b750b29; coordinate convergence is squeezed from
generated initialized total mass, not supplied as a premise. -/
theorem gain_native_strict_cold_parameters_tendsto (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hkeep : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial)
    (hgap : coldGainCEGradientScale remaining bound * genGain < decay * eps) :
    ∀ i, Tendsto (fun n => (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n i).parameter)
      atTop (nhds 0) := by
  have hmass := (gain_native_strict_cold_masses_tendsto remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep hs hgap).1
  have hsign := fun n => gain_native_nonnegative_path remaining genGain memGain bound b1 b2 eps decay rate initial n
    hgen hmem hclip hb1 h1 hb2 h2 he (le_of_lt heta) hkeep hs
  intro i
  exact squeeze_zero (fun n => (hsign n i).1)
    (fun n => (gain_native_masses_nonnegative _ (hsign n)).2.2 i) hmass

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * 2 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    coldGainCEGradientScale 0 1 * 3 < (2 : ℝ) * 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [coldGainCEGradientScale]⟩

/-- Every actual clipped CE coordinate input vanishes above the strict
cold threshold. Sources: appendix C partner derivatives, actual cold
input cap at 696630b and initialized native mass collapse at b750b29;
no external convergent gradient stream or buffer reset is inserted. -/
theorem gain_native_strict_cold_inputs_tendsto (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hkeep : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial)
    (hgap : coldGainCEGradientScale remaining bound * genGain < decay * eps) :
    ∀ i, Tendsto (fun n => appliedGainNativeGradient remaining genGain memGain bound
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n) i) atTop (nhds 0) := by
  have hmass := (gain_native_strict_cold_masses_tendsto remaining genGain memGain bound b1 b2 eps decay rate initial
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep hs hgap).1
  have hsign := fun n => gain_native_nonnegative_path remaining genGain memGain bound b1 b2 eps decay rate initial n
    hgen hmem hclip hb1 h1 hb2 h2 he (le_of_lt heta) hkeep hs
  have hupper : Tendsto (fun n => (coldGainCEGradientScale remaining bound * genGain) *
      gainNativeParameterMass (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n))
      atTop (nhds 0) := by simpa only [mul_zero] using hmass.const_mul (coldGainCEGradientScale remaining bound * genGain)
  intro i
  have hcap := fun n => gain_native_cold_applied_gradient_mass_cap remaining genGain memGain bound _ i
    hgen hmem hgain hclip (hsign n)
  have ht := squeeze_zero (fun n => (hcap n).1) (fun n => (hcap n).2) hupper
  simpa only [neg_neg, neg_zero] using ht.neg

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * 2 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    coldGainCEGradientScale 0 1 * 3 < (2 : ℝ) * 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [coldGainCEGradientScale]⟩

/-- Away from explicit critical equality, original initialized total
mass collapses exactly above the cold decay threshold. Sources:
section 3 competition, appendix C CE, native noncollapse at 6a99a48
and collapse at b750b29; no future convergence or task-success premise. -/
theorem gain_native_strict_cold_threshold (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate) (hkeep : 0 < 1 - rate * decay)
    (hs : NonnegativeNativeState initial) (hi : 0 < (initial 1).parameter)
    (hne : decay * eps ≠ coldGainCEGradientScale remaining bound * genGain) :
    Tendsto (fun n => gainNativeParameterMass
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n)) atTop (nhds 0) ↔
        coldGainCEGradientScale remaining bound * genGain < decay * eps := by
  constructor
  · intro hmass
    by_contra hstrong
    have hweak := lt_of_le_of_ne (le_of_not_gt hstrong) hne
    exact gain_native_cold_regime_not_mass_collapse remaining genGain memGain bound b1 b2 eps decay rate initial
      hmem hgain hclip hb1 h1 hb2 h2 he hdecay heta hkeep hs hi hweak hmass
  · intro hstrong
    exact (gain_native_strict_cold_masses_tendsto remaining genGain memGain bound b1 b2 eps decay rate initial
      (lt_trans hmem hgain) hmem (le_of_lt hgain) hclip hb1 h1 hb2 h2 he heta (le_of_lt hkeep) hs hstrong).1

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧
    (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) < 1 - (1 / 1000) * 2 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    0 < (seededNativeSubweights ((0, 1 / 200), (0, 1)) 1).parameter ∧
    (2 : ℝ) * 1 ≠ coldGainCEGradientScale 0 1 * 3 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [seededNativeSubweights, seededScalarState], by norm_num [coldGainCEGradientScale]⟩

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) < 1 - (1 / 1000) * 1 ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    0 < (seededNativeSubweights ((0, 1 / 200), (0, 1)) 1).parameter ∧
    (1 : ℝ) * 1 ≠ coldGainCEGradientScale 0 1 * 3 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [seededNativeSubweights, seededScalarState], by norm_num [coldGainCEGradientScale]⟩

end Transformer.Grokking.CircuitEfficiency
