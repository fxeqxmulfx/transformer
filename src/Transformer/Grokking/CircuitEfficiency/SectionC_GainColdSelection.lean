import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdLimits

/-!
# Eventual efficient Gen decisions under a static native regime

Sources: Varma et al., arXiv:2309.02390v1, section 3 and appendix C;
actual gained CE/native retained feedback at lab commit 45c1826.
Combine initial positive-partner persistence, necessary native
balance and boundary exclusion with actual physical output limits.
The static regime uses only decay, epsilon, cap, class count and
Gen physical gain; no nonzero future Mem or Gen factor is assumed.

Conditional on actual finite parameter convergence, the reference
Gen score and Gen-minus-Mem margin are positive. The actual margin
converges to that value. Consequently every held-out competing class
is strictly beaten at all sufficiently late native clocks. Mem may
have zero reference factors, so positive-interior allocation alone
cannot supply the final argument.

Global parameter convergence and a finite delay are still open.
The source's smaller-Gen seed is admitted but is not proved attracted
to the numerical balanced witnesses. Both buffers and growing clocks
remain in the update; tables/gains are fixed, and plain CE/native
decoupled decay differ from the source's norm-cost GD. Learned
stochastic/numerical GPTMini generalization is not identified with
this conditional fixed-table decision theorem.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Actual finite physical limits have positive Gen score and test
margin under the static cold-feedback regime. Sources: section 3
Efficiency and appendix C products, with native feedback at 45c1826;
the actual successful margin and its limit are derived, not premises. -/
theorem gain_native_source_cold_regime_margin_limit (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial reference : NativeSubweightState) (hmem : 0 < memGain) (hgain : memGain < genGain)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (hs : NonnegativeNativeState initial) (hi : 0 < (initial 1).parameter)
    (hp : ∀ i, Tendsto (fun n =>
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n i).parameter)
      atTop (nhds (reference i).parameter))
    (hregime : decay * eps < coldGainCEGradientScale remaining bound * genGain) :
    let state := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    let gen := physicalCircuitScore genGain (reference 0).parameter (reference 1).parameter
    let mem := physicalCircuitScore memGain (reference 2).parameter (reference 3).parameter
    0 < gen ∧ 0 < gen - mem ∧ Tendsto (fun n =>
      physicalCircuitScore genGain (state n 0).parameter (state n 1).parameter -
        physicalCircuitScore memGain (state n 2).parameter (state n 3).parameter)
      atTop (nhds (gen - mem)) := by
  dsimp only
  let state := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
  have hstep : ∀ n, state (n + 1) =
      gainNativeStep remaining genGain memGain bound b1 b2 eps decay rate (state n) := fun _ => rfl
  have hmass := gain_native_source_cold_regime_gen_mass remaining genGain memGain bound b1 b2 eps decay rate
    initial reference hmem hgain hclip hb1 h1 hb2 h2 he hdecay heta hd hs hi hp hregime
  have hsign : ∀ n, NonnegativeNativeState (state n) := fun n =>
    gain_native_nonnegative_path remaining genGain memGain bound b1 b2 eps decay rate initial n
      (lt_trans hmem hgain) hmem hclip hb1 h1 hb2 h2 he (le_of_lt heta) (le_of_lt hd) hs
  have href := gain_native_nonnegative_parameter_limit state reference hsign hp
  have hbalance : ∀ i, decay * (reference i).parameter +
      appliedGainNativeGradient remaining genGain memGain bound reference i /
      (|appliedGainNativeGradient remaining genGain memGain bound reference i| + eps) = 0 := fun i =>
    gain_native_closed_limit_balance remaining genGain memGain bound b1 b2 eps decay rate state reference i
      hstep hb1 h1 hb2 h2 he heta hp
  have hm := gain_native_nonnegative_gen_margin remaining genGain memGain bound eps decay reference
    hmem hgain hclip he href hmass hbalance
  refine ⟨hm.1, sub_pos.mpr hm.2, ?_⟩
  have hg := ((hp 0).mul (hp 1)).const_mul genGain
  have hmemLimit := ((hp 2).mul (hp 3)).const_mul memGain
  simpa only [physicalCircuitScore] using hg.sub hmemLimit

example :
    let point := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let eps := 3 * gainCEGradientScale 111 3 2 1 point
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ 0 < eps ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 1000 ∧
    (0 : ℝ) < 1 - (1 / 1000) * (1 / 2) ∧ NonnegativeNativeState point ∧
    0 < (point 1).parameter ∧ (∀ i, Tendsto (fun n =>
      (gainNativePath 111 3 2 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000) point n i).parameter)
      atTop (nhds (point i).parameter)) ∧ (1 / 2) * eps < coldGainCEGradientScale 111 1 * 3 := by
  dsimp only
  have hw := gain_native_efficient_point_cold_regime
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, hw.1, hw.2.1, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _
      (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [seededNativeSubweights, seededScalarState], ?_, hw.2.2.1⟩
  intro i
  exact gain_native_balanced_path_parameter_tendsto 111 3 2 1 (9 / 10) (49 / 50) _ (1 / 2) (1 / 1000)
    ((1, 1), (1 / 2, 1 / 2)) i
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) hw.2.2.2

/-- Initial Gen partner data and the static native regime yield
eventual strict held-out correctness on the actual path, conditional
on finite parameter convergence. Sources: appendix C test logits
and section 3 Efficiency at 45c1826; no future pair positivity or
successful-margin hypothesis is used, and zero Mem is allowed. -/
theorem gain_native_source_cold_regime_eventually_correct (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial reference : NativeSubweightState) (hmem : 0 < memGain) (hgain : memGain < genGain)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (hs : NonnegativeNativeState initial) (hi : 0 < (initial 1).parameter)
    (hp : ∀ i, Tendsto (fun n =>
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n i).parameter)
      atTop (nhds (reference i).parameter))
    (hregime : decay * eps < coldGainCEGradientScale remaining bound * genGain) :
    let state := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
    ∀ᶠ n in atTop, Transformer.Grokking.NaiveLoss.StrictCorrect (heldoutTableLogits remaining
      (physicalCircuitScore genGain (state n 0).parameter (state n 1).parameter)
      (physicalCircuitScore memGain (state n 2).parameter (state n 3).parameter)) 0 := by
  have ha := gain_native_source_cold_regime_margin_limit remaining genGain memGain bound b1 b2 eps decay rate
    initial reference hmem hgain hclip hb1 h1 hb2 h2 he hdecay heta hd hs hi hp hregime
  have hg : Tendsto (fun n => physicalCircuitScore genGain
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n 0).parameter
      (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n 1).parameter)
      atTop (nhds (physicalCircuitScore genGain (reference 0).parameter (reference 1).parameter)) := by
    simpa only [physicalCircuitScore] using ((hp 0).mul (hp 1)).const_mul genGain
  have hzero : Tendsto (fun _ : ℕ => (0 : ℝ)) atTop (nhds 0) := tendsto_const_nhds
  have hpositive := hzero.eventually_lt hg ha.1
  have hmargin := hzero.eventually_lt ha.2.2 ha.2.1
  exact (hpositive.and hmargin).mono fun n hh =>
    heldout_table_strict_correct remaining _ _ hh.1 (sub_pos.mp hh.2)

example :
    let point := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let eps := 3 * gainCEGradientScale 111 3 2 1 point
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ 0 < eps ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 1000 ∧
    (0 : ℝ) < 1 - (1 / 1000) * (1 / 2) ∧ NonnegativeNativeState point ∧
    0 < (point 1).parameter ∧ (∀ i, Tendsto (fun n =>
      (gainNativePath 111 3 2 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000) point n i).parameter)
      atTop (nhds (point i).parameter)) ∧ (1 / 2) * eps < coldGainCEGradientScale 111 1 * 3 := by
  dsimp only
  have hw := gain_native_efficient_point_cold_regime
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, hw.1, hw.2.1, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _
      (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [seededNativeSubweights, seededScalarState], ?_, hw.2.2.1⟩
  intro i
  exact gain_native_balanced_path_parameter_tendsto 111 3 2 1 (9 / 10) (49 / 50) _ (1 / 2) (1 / 1000)
    ((1, 1), (1 / 2, 1 / 2)) i
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) hw.2.2.2

end Transformer.Grokking.CircuitEfficiency
