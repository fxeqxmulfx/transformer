import Transformer.Grokking.CircuitEfficiency.SectionC_GainReferenceBalance
import Transformer.Grokking.CircuitEfficiency.SectionC_GainPositiveLimits

/-!
# Efficient Gen selection from an initial positive partner

Sources: Varma et al., arXiv:2309.02390v1, section 3's formation
and efficiency ingredients and appendix C product CE; native
AdamW/clipping at lab commit 9e13187. Replace four positive
reference-factor premises by numerical initial signs, one positive
initial Gen partner, actual finite parameter convergence and one
positive Mem reference factor. Gen reference positivity, native
balance, efficient allocation and eventual test success are derived.

The first Gen factor may start at zero. The positive remaining
decay factor preserves its partner at finite clocks; the retained
feedback argument excludes the Mem-only finite boundary. Both
buffers and completed clocks remain in every actual update.

Finite convergence and a nonzero Mem limit remain explicit. This
is not global attraction, exclusion of the fully cold limit, a
delay estimate or a learned stochastic/numerical GPTMini theorem.
Physical readout gains encode efficiency here; appendix C instead
assigns circuit norm costs and uses GD. Both use fixed tables.
The joint witnesses are actual nonzero retained balanced paths.
Their derived convergence does not establish convergence from the
source's smaller Gen seed or attraction from other initial states.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- One positive initial Gen partner gives positive mass at every
finite actual clock. Sources: section 3's small seeds and native
AdamW at 9e13187; positive future limiting mass is not assumed. -/
theorem gain_native_source_positive_mass (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) (n : ℕ) (hmem : 0 < memGain) (hgain : memGain < genGain)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 ≤ rate) (hd : 0 < 1 - rate * decay)
    (hs : NonnegativeNativeState initial) (hi : 0 < (initial 1).parameter) :
    0 < gainGenParameterMass (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n) := by
  have hgen := lt_trans hmem hgain
  have hn := gain_native_nonnegative_path remaining genGain memGain bound b1 b2 eps decay rate initial n
    hgen hmem hclip hb1 h1 hb2 h2 he heta (le_of_lt hd) hs
  have hp := gain_native_positive_coordinate_path remaining genGain memGain bound b1 b2 eps decay rate initial n 1
    hgen hmem hclip hb1 h1 hb2 h2 he heta hd hs hi
  unfold gainGenParameterMass
  linarith only [(hn 0).1, hp.1]

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    (0 : ℝ) < 1 - (1 / 1000) * (1 / 10) ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    0 < (seededNativeSubweights ((0, 1 / 200), (0, 1)) 1).parameter := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [seededNativeSubweights, seededScalarState]⟩

/-- A convergent actual source-sign path with a live Mem reference
has all reference factors positive. Sources: section 3 Efficiency
and appendix C CE, with native retained noncollapse at 9e13187;
neither positive Gen limits nor a successful output margin are premises. -/
theorem gain_native_source_live_mem_positive_limits (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial reference : NativeSubweightState) (hmem : 0 < memGain) (hgain : memGain < genGain)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (hs : NonnegativeNativeState initial) (hi : 0 < (initial 1).parameter)
    (hp : ∀ i, Tendsto (fun n => (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n i).parameter)
      atTop (nhds (reference i).parameter)) (hm : 0 < (reference 2).parameter) :
    ∀ i, 0 < (reference i).parameter := by
  let state := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
  have hgen := lt_trans hmem hgain
  have hstep : ∀ n, state (n + 1) = gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate (state n) := fun _ => rfl
  have hsign : ∀ n, NonnegativeNativeState (state n) := fun n =>
    gain_native_nonnegative_path remaining genGain memGain bound b1 b2 eps decay rate initial n
      hgen hmem hclip hb1 h1 hb2 h2 he (le_of_lt heta) (le_of_lt hd) hs
  have hmass : ∀ n, 0 < gainGenParameterMass (state n) := fun n =>
    gain_native_source_positive_mass remaining genGain memGain bound b1 b2 eps decay rate initial n
      hmem hgain hclip hb1 h1 hb2 h2 he (le_of_lt heta) hd hs hi
  have href := gain_native_nonnegative_parameter_limit state reference hsign hp
  have hbalance : ∀ i, decay * (reference i).parameter + appliedGainNativeGradient remaining genGain memGain bound reference i /
      (|appliedGainNativeGradient remaining genGain memGain bound reference i| + eps) = 0 := fun i =>
    gain_native_closed_limit_balance remaining genGain memGain bound b1 b2 eps decay rate state reference i
      hstep hb1 h1 hb2 h2 he heta hp
  have hthreshold := gain_native_mem_live_feedback_threshold remaining genGain memGain bound eps decay reference
    hmem hgain hclip he href hm hbalance
  have hpositive := gain_gen_mass_positive_limit remaining genGain memGain bound b1 b2 eps decay rate state reference
    hstep hgen hclip hb1 h1 hb2 h2 he hthreshold.1 (le_of_lt heta) hsign hmass hp hthreshold.2.2
  exact gain_native_live_pairs_positive remaining genGain memGain bound eps decay reference
    hmem hgain hclip he href hpositive hm hbalance

example :
    let point := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let eps := 3 * gainCEGradientScale 111 3 2 1 point
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ 0 < eps ∧ (0 : ℝ) < 1 / 1000 ∧
    (0 : ℝ) < 1 - (1 / 1000) * (1 / 2) ∧ NonnegativeNativeState point ∧ 0 < (point 1).parameter ∧
    (∀ i, Tendsto (fun n => (gainNativePath 111 3 2 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000) point n i).parameter)
      atTop (nhds (point i).parameter)) ∧ 0 < (point 2).parameter := by
  dsimp only
  have hw := gain_native_efficient_point_balance 111 1 (by norm_num)
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, hw.1, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [seededNativeSubweights, seededScalarState], ?_, by norm_num [seededNativeSubweights, seededScalarState]⟩
  intro i
  exact gain_native_balanced_path_parameter_tendsto 111 3 2 1 (9 / 10) (49 / 50) _ (1 / 2) (1 / 1000)
    ((1, 1), (1 / 2, 1 / 2)) i (by norm_num) (by norm_num) (by norm_num) (by norm_num) hw.2

/-- The actual source-sign efficient path eventually beats every
held-out competitor under finite convergence and a live Mem limit.
Sources: appendix C test products and section 3 Efficiency, native
AdamW at 9e13187; Gen limiting positivity and the test margin are derived. -/
theorem gain_native_source_live_mem_eventually_correct (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial reference : NativeSubweightState) (hmem : 0 < memGain) (hgain : memGain < genGain)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (hs : NonnegativeNativeState initial) (hi : 0 < (initial 1).parameter)
    (hp : ∀ i, Tendsto (fun n => (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n i).parameter)
      atTop (nhds (reference i).parameter)) (hm : 0 < (reference 2).parameter) :
    let state := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    ∀ᶠ n in atTop, Transformer.Grokking.NaiveLoss.StrictCorrect (heldoutTableLogits remaining
      (physicalCircuitScore genGain (state n 0).parameter (state n 1).parameter)
      (physicalCircuitScore memGain (state n 2).parameter (state n 3).parameter)) 0 := by
  have hpos := gain_native_source_live_mem_positive_limits remaining genGain memGain bound b1 b2 eps decay rate
    initial reference hmem hgain hclip hb1 h1 hb2 h2 he heta hd hs hi hp hm
  exact gain_native_closed_eventually_heldout_correct remaining genGain memGain bound b1 b2 eps decay rate
    (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial) reference
    (fun _ => rfl) hmem hgain hclip hb1 h1 hb2 h2 he heta hpos hp

example :
    let point := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let eps := 3 * gainCEGradientScale 111 3 2 1 point
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ 0 < eps ∧ (0 : ℝ) < 1 / 1000 ∧
    (0 : ℝ) < 1 - (1 / 1000) * (1 / 2) ∧ NonnegativeNativeState point ∧ 0 < (point 1).parameter ∧
    (∀ i, Tendsto (fun n => (gainNativePath 111 3 2 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000) point n i).parameter)
      atTop (nhds (point i).parameter)) ∧ 0 < (point 2).parameter := by
  dsimp only
  have hw := gain_native_efficient_point_balance 111 1 (by norm_num)
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, hw.1, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [seededNativeSubweights, seededScalarState], ?_, by norm_num [seededNativeSubweights, seededScalarState]⟩
  intro i
  exact gain_native_balanced_path_parameter_tendsto 111 3 2 1 (9 / 10) (49 / 50) _ (1 / 2) (1 / 1000)
    ((1, 1), (1 / 2, 1 / 2)) i (by norm_num) (by norm_num) (by norm_num) (by norm_num) hw.2

end Transformer.Grokking.CircuitEfficiency
