import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdRegime

/-!
# Efficient Gen finite limits under a static cold-feedback regime

Sources: Varma et al., arXiv:2309.02390v1, section 3 Efficiency
and appendix C product CE; retained native AdamW/clipping at lab
commit 92002ec. A static condition on decay, epsilon, class count,
clip cap and Gen gain replaces the previous live-Mem-limit premise.
No positive Gen reference factor or held-out margin is supplied.

Assuming actual finite parameter convergence, initial numerical
signs and one positive Gen partner, Gen limiting mass is positive.
If it were zero, a live Mem reference supplies the feedback threshold
by native balance; otherwise all four reference parameters vanish,
and the exact cold coefficient supplies it by the static regime.
The retained noncollapse argument rules out both boundary cases.

At a nonnegative actual normalized balance, positive Gen mass also
gives a positive Gen score and Gen strictly beats Mem. Mem may be
zero or positive: zero Mem needs no positive-interior theorem.
Global convergence, attraction and delayed crossing remain open.
Tables/gains are fixed; plain CE/uniform native decay differ from
the source's norm-cost GD and learned stochastic/numerical GPTMini.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- A static cold-feedback regime excludes both zero-Gen boundary
limits from the actual positive-partner path. Sources: section 3
formation and appendix C CE, with native feedback at 92002ec;
finite convergence remains explicit, and no live Mem limit is assumed. -/
theorem gain_native_source_cold_regime_gen_mass (remaining : ℕ) (genGain memGain bound b1 b2 eps decay rate : ℝ)
    (initial reference : NativeSubweightState) (hmem : 0 < memGain) (hgain : memGain < genGain)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (hs : NonnegativeNativeState initial) (hi : 0 < (initial 1).parameter)
    (hp : ∀ i, Tendsto (fun n => (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial n i).parameter)
      atTop (nhds (reference i).parameter))
    (hregime : decay * eps < coldGainCEGradientScale remaining bound * genGain) :
    0 < gainGenParameterMass reference := by
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
  by_contra hnot
  have hzeroGen : (reference 0).parameter = 0 ∧ (reference 1).parameter = 0 := by
    have hmassNonpos := le_of_not_gt hnot
    unfold gainGenParameterMass at hmassNonpos
    constructor <;> linarith [href 0, href 1]
  have hthreshold : decay * eps < gainCEGradientScale remaining genGain memGain bound reference * genGain := by
    by_cases hm : (reference 2).parameter = 0
    · have hn := gain_native_nonnegative_partner_balance remaining genGain memGain bound eps decay reference
        hgen hmem hclip href hbalance
      have hzero3 : (reference 3).parameter = 0 := by
        have hz : decay * (reference 3).parameter = 0 := by
          simpa [nativeFactorGain, nativeFactorPartner, hm] using hn 3
        exact (mul_eq_zero.mp hz).resolve_left (ne_of_gt hdecay)
      have hall : ∀ i, (reference i).parameter = 0 := by
        intro i
        fin_cases i
        · exact hzeroGen.1
        · exact hzeroGen.2
        · exact hm
        · exact hzero3
      rw [gain_ce_gradient_scale_zero_parameters remaining genGain memGain bound reference hall]
      exact hregime
    · have hpositiveMem : 0 < (reference 2).parameter := lt_of_le_of_ne (href 2) (Ne.symm hm)
      exact (gain_native_mem_live_feedback_threshold remaining genGain memGain bound eps decay reference
        hmem hgain hclip he href hpositiveMem hbalance).2.2
  exact hnot (gain_gen_mass_positive_limit remaining genGain memGain bound b1 b2 eps decay rate state reference
    hstep hgen hclip hb1 h1 hb2 h2 he hdecay (le_of_lt heta) hsign hmass hp hthreshold)

example :
    let point := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let eps := 3 * gainCEGradientScale 111 3 2 1 point
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ 0 < eps ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 1000 ∧
    (0 : ℝ) < 1 - (1 / 1000) * (1 / 2) ∧ NonnegativeNativeState point ∧ 0 < (point 1).parameter ∧
    (∀ i, Tendsto (fun n => (gainNativePath 111 3 2 1 (9 / 10) (49 / 50) eps (1 / 2) (1 / 1000) point n i).parameter)
      atTop (nhds (point i).parameter)) ∧ (1 / 2) * eps < coldGainCEGradientScale 111 1 * 3 := by
  dsimp only
  have hw := gain_native_efficient_point_cold_regime
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, hw.1, hw.2.1, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [seededNativeSubweights, seededScalarState], ?_, hw.2.2.1⟩
  intro i
  exact gain_native_balanced_path_parameter_tendsto 111 3 2 1 (9 / 10) (49 / 50) _ (1 / 2) (1 / 1000)
    ((1, 1), (1 / 2, 1 / 2)) i (by norm_num) (by norm_num) (by norm_num) (by norm_num) hw.2.2.2

/-- Positive Gen mass at any nonnegative native balanced reference
gives positive Gen score and a strict Gen-minus-Mem margin.
Sources: section 3 Efficiency and appendix C test products at 92002ec;
zero Mem is allowed rather than hidden behind positive-interior premises. -/
theorem gain_native_nonnegative_gen_margin (remaining : ℕ) (genGain memGain bound eps decay : ℝ)
    (state : NativeSubweightState) (hmem : 0 < memGain) (hgain : memGain < genGain)
    (hclip : 0 < bound) (he : 0 < eps) (hp : ∀ i, 0 ≤ (state i).parameter)
    (hg : 0 < gainGenParameterMass state)
    (hb : ∀ i, decay * (state i).parameter + appliedGainNativeGradient remaining genGain memGain bound state i /
      (|appliedGainNativeGradient remaining genGain memGain bound state i| + eps) = 0) :
    0 < physicalCircuitScore genGain (state 0).parameter (state 1).parameter ∧
      physicalCircuitScore memGain (state 2).parameter (state 3).parameter <
        physicalCircuitScore genGain (state 0).parameter (state 1).parameter := by
  have hgen := lt_trans hmem hgain
  have hn := gain_native_nonnegative_partner_balance remaining genGain memGain bound eps decay state hgen hmem hclip hp hb
  have hs := gain_ce_gradient_scale_pos remaining genGain memGain bound state hclip
  have hGenPair := native_nonnegative_pair_balance
    (gainCEGradientScale remaining genGain memGain bound state * genGain) eps decay (state 0).parameter (state 1).parameter
    (mul_pos hs hgen) he (hp 0) (hp 1)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hn 0)
    (by simpa [nativeFactorGain, nativeFactorPartner] using hn 1)
  rcases hGenPair with hz | ⟨_, hp0, hp1, _, _⟩
  · exfalso
    unfold gainGenParameterMass at hg
    linarith [hz.1, hz.2]
  · have hscore : 0 < physicalCircuitScore genGain (state 0).parameter (state 1).parameter := mul_pos hgen (mul_pos hp0 hp1)
    have hMemPair := native_nonnegative_pair_balance
      (gainCEGradientScale remaining genGain memGain bound state * memGain) eps decay (state 2).parameter (state 3).parameter
      (mul_pos hs hmem) he (hp 2) (hp 3)
      (by simpa [nativeFactorGain, nativeFactorPartner] using hn 2)
      (by simpa [nativeFactorGain, nativeFactorPartner] using hn 3)
    rcases hMemPair with hzMem | ⟨_, hp2, hp3, _, _⟩
    · refine ⟨hscore, ?_⟩
      simpa only [physicalCircuitScore, hzMem.1, hzMem.2, mul_zero] using hscore
    · have hpositive : ∀ i : Fin 4, 0 < (state i).parameter := by
        intro i
        fin_cases i
        · exact hp0
        · exact hp1
        · exact hp2
        · exact hp3
      exact ⟨hscore, (gain_native_positive_balanced_allocation remaining genGain memGain bound eps decay
        state hmem hgain hclip he hpositive hb).2.2.2.2⟩

example :
    let state := seededNativeSubweights ((1, 1), (1 / 2, 1 / 2))
    let eps := 3 * gainCEGradientScale 111 3 2 1 state
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ 0 < eps ∧ (∀ i, 0 ≤ (state i).parameter) ∧
    0 < gainGenParameterMass state ∧
    (∀ i, (1 / 2 : ℝ) * (state i).parameter + appliedGainNativeGradient 111 3 2 1 state i /
      (|appliedGainNativeGradient 111 3 2 1 state i| + eps) = 0) := by
  dsimp only
  have hw := gain_native_efficient_point_cold_regime
  refine ⟨by norm_num, by norm_num, by norm_num, hw.1, ?_,
    by norm_num [gainGenParameterMass, seededNativeSubweights, seededScalarState], hw.2.2.2⟩
  intro i
  fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState]

end Transformer.Grokking.CircuitEfficiency
