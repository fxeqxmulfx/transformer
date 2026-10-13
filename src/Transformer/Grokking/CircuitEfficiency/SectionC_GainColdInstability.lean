import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdLimits
import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryMass

/-!
# Static weak decay excludes actual cold parameter collapse

Sources: Varma et al., arXiv:2309.02390v1, section 3 CE/decay
competition and appendix C's product feedback; retained native AdamW
at lab commit 92002ec, actual cold-feedback boundary at ca253e4.

Initial numerical signs and one positive Gen partner generate the
complete actual retained path. If decay times epsilon is below the
computed cold Gen coefficient, all four physical parameters cannot
tend to zero. A hypothetical cold limit supplies its own actual input,
buffer and denominator limits; retained positive-mass feedback then
contradicts that limit. No convergence premise survives in the result.

The sum of all four nonnegative parameters cannot tend to zero either:
its collapse would squeeze each actual parameter to zero. Specialize
this obstruction to source-style Gen (0,seed), Mem (0,1), beta1=0.9,
beta2=0.98, rate=0.001, cap=1, decay=0.1 and epsilon=1e-8. These native
optimizer constants meet the cold threshold for every finite class
count and every positive seed. Both buffers and completed clocks evolve
without resets; initial products and logits are zero.

This excludes one collapse mechanism and does not prove attraction,
positive confidence, delayed held-out selection or convergence to a
finite point. Fixed physical gains 3/2 and tables remain a model, not
a learned-head certificate for the ordinary transformer. Uniform native
decoupled decay/plain CE differ from the source's coupled norm-cost GD.
All statements concern exact reals, not floating-point kernel errors.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Initial positive Gen data under the static cold threshold exclude
collapse of all actual physical parameters, with no future convergence
premise. Sources: section 3 formation, appendix C CE and retained native
boundary at 92002ec/ca253e4; no successful reference is supplied. -/
theorem gain_native_cold_regime_not_parameter_collapse (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (hs : NonnegativeNativeState initial) (hi : 0 < (initial 1).parameter)
    (hregime : decay * eps < coldGainCEGradientScale remaining bound * genGain) :
    ¬∀ i, Tendsto (fun n => (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n i).parameter)
      atTop (nhds 0) := by
  intro hp
  let reference := seededNativeSubweights ((0, 0), (0, 0))
  have hlimit : ∀ i, Tendsto
      (fun n => (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n i).parameter)
      atTop (nhds (reference i).parameter) := by
    intro i
    fin_cases i
    · simpa [reference, seededNativeSubweights, seededScalarState] using hp 0
    · simpa [reference, seededNativeSubweights, seededScalarState] using hp 1
    · simpa [reference, seededNativeSubweights, seededScalarState] using hp 2
    · simpa [reference, seededNativeSubweights, seededScalarState] using hp 3
  have hpositive := gain_native_source_cold_regime_gen_mass remaining genGain memGain bound b1 b2 eps decay rate
    initial reference hmem hgain hclip hb1 h1 hb2 h2 he hdecay heta hd hs hi hlimit hregime
  norm_num [gainGenParameterMass, reference, seededNativeSubweights, seededScalarState] at hpositive

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧
    (0 : ℝ) < 1 - (1 / 1000) * (1 / 10) ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    0 < (seededNativeSubweights ((0, 1 / 200), (0, 1)) 1).parameter ∧
    (1 / 10 : ℝ) * (1 / 100000000) < coldGainCEGradientScale 0 1 * 3 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [seededNativeSubweights, seededScalarState], cold_gain_small_decay_epsilon_regime 0⟩

/-- The complete actual nonnegative physical parameter mass cannot
tend to zero in the static cold regime. Sources: section 3's norm
competition and appendix C factors; native retained mass at ada36f3
and cold boundary at 92002ec. This is not a positive limiting mass claim. -/
theorem gain_native_cold_regime_not_mass_collapse (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (hs : NonnegativeNativeState initial) (hi : 0 < (initial 1).parameter)
    (hregime : decay * eps < coldGainCEGradientScale remaining bound * genGain) :
    ¬Tendsto (fun n => gainNativeParameterMass
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n)) atTop (nhds 0) := by
  intro hmass
  have hsign : ∀ n, NonnegativeNativeState
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n) := fun n =>
    gain_native_nonnegative_path remaining genGain memGain bound b1 b2 eps decay rate initial n
      (lt_trans hmem hgain) hmem hclip hb1 h1 hb2 h2 he (le_of_lt heta) (le_of_lt hd) hs
  apply gain_native_cold_regime_not_parameter_collapse remaining genGain memGain bound b1 b2 eps decay rate
    initial hmem hgain hclip hb1 h1 hb2 h2 he hdecay heta hd hs hi hregime
  intro i
  exact squeeze_zero (fun n => (hsign n i).1)
    (fun n => (gain_native_masses_nonnegative _ (hsign n)).2.2 i) hmass

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧
    (0 : ℝ) < 1 - (1 / 1000) * (1 / 10) ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    0 < (seededNativeSubweights ((0, 1 / 200), (0, 1)) 1).parameter ∧
    (1 / 10 : ℝ) * (1 / 100000000) < coldGainCEGradientScale 0 1 * 3 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [seededNativeSubweights, seededScalarState], cold_gain_small_decay_epsilon_regime 0⟩

/-- Source-style double-zero products with the small native decay and
epsilon cannot have vanishing total physical mass, for any finite class
count. Sources: section 3 initial factors and appendix C CE; computed
cold native threshold at ca253e4, no future state or limit hypothesis. -/
theorem gain_native_small_decay_source_not_mass_collapse (remaining : ℕ) (seed : ℝ) (hseed : 0 < seed) :
    ¬Tendsto (fun n => gainNativeParameterMass (gainNativePath remaining 3 2 1
      (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
      (seededNativeSubweights ((0, seed), (0, 1))) n)) atTop (nhds 0) := by
  apply gain_native_cold_regime_not_mass_collapse remaining 3 2 1
    (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
    (seededNativeSubweights ((0, seed), (0, 1)))
  · norm_num
  · norm_num
  · norm_num
  · norm_num
  · norm_num
  · norm_num
  · norm_num
  · norm_num
  · norm_num
  · norm_num
  · norm_num
  · exact native_seeded_nonnegative 0 seed 0 1 le_rfl (le_of_lt hseed) le_rfl (by norm_num)
  · change 0 < seed
    exact hseed
  · exact cold_gain_small_decay_epsilon_regime remaining

example : (0 : ℝ) < 1 / 200 := by norm_num

/-- The same source-style small-decay path cannot have zero limits
for all four parameters. Sources: section 3's slow positive Gen seed
and appendix C product inputs; native cold threshold at ca253e4.
This remains an obstruction, without asserting existence of any limit. -/
theorem gain_native_small_decay_source_not_parameter_collapse (remaining : ℕ) (seed : ℝ) (hseed : 0 < seed) :
    ¬∀ i, Tendsto (fun n => (gainNativePath remaining 3 2 1
      (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
      (seededNativeSubweights ((0, seed), (0, 1))) n i).parameter) atTop (nhds 0) := by
  apply gain_native_cold_regime_not_parameter_collapse remaining 3 2 1
    (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
    (seededNativeSubweights ((0, seed), (0, 1)))
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (native_seeded_nonnegative 0 seed 0 1 le_rfl (le_of_lt hseed) le_rfl (by norm_num))
  · change 0 < seed
    exact hseed
  · exact cold_gain_small_decay_epsilon_regime remaining

example : (0 : ℝ) < 1 / 200 := by norm_num

end Transformer.Grokking.CircuitEfficiency
