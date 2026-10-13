import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdMassDissipation

/-!
# Actual critical collapse with legal first beta zero

Sources: Varma et al., arXiv:2309.02390v1, section 3 CE/decay
competition and appendix C product CE; initialized native mass
monotonicity at 7f76ed7 and true CE-generated dissipation at 1abd0fa.

At or above the cold greater-gain threshold, the actual initialized
zero-first-beta mass has a finite nonnegative limit and a generated
physical box. A positive limit would supply a uniform positive actual
input floor through the true clipped CE feedback. Its squared native
variance insertion then forces a fixed positive mass decrement,
contradicting convergence. Thus total physical mass tends to zero,
including exact critical equality, without a future limit premise.

Every physical coordinate tends to zero by nonnegative mass coverage.
The retained second beta, variance and completed clocks remain in the
original recurrence; no instantaneous variance matching, partner mixing,
variance ceiling or successful classifier is supplied. Positive rate is
essential: the preceding observer laws allowed a stationary zero rate.
Remaining decay need only be nonnegative, including zero.

This is a legal first-beta-zero control of fixed gained tables with
plain CE and uniform decoupled native AdamW. It does not reset or replace
the frozen beta1=0.9 experiments. Nonzero-first-beta critical convergence
needs its own weighted dissipation argument. Appendix C's assigned
coupled norm-cost GD differs; learned stochastic/numerical GPTMini,
relative Gen/Mem selection and thermodynamic size scaling are separate.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- The actual initialized total physical mass collapses at or above
the cold threshold with legal first beta zero. Sources: appendix C
true partner CE and generated mass/dissipation at 7f76ed7/1abd0fa;
no parameter, gradient, buffer or denominator convergence is assumed. -/
theorem gain_native_zero_first_beta_cold_mass_tendsto (remaining : ℕ)
    (genGain memGain bound b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hkeep : 0 ≤ 1 - rate * decay) (hs : NonnegativeNativeState initial)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    Tendsto (fun n => gainNativeParameterMass
      (gainNativePath remaining genGain memGain bound 0 b2 eps decay rate initial n)) atTop (nhds 0) := by
  let state := gainNativePath remaining genGain memGain bound 0 b2 eps decay rate initial
  let mass := fun n => gainNativeParameterMass (state n)
  obtain ⟨value, hnonnegative, ht⟩ := gain_native_zero_first_beta_cold_mass_limit remaining genGain memGain bound b2 eps decay rate initial
    hgen hmem hgain hclip hb2 h2 he (le_of_lt heta) hkeep hs hthreshold
  change Tendsto mass atTop (nhds value) at ht
  have hzero : value = 0 := by
    by_contra h
    have hv : 0 < value := lt_of_le_of_ne hnonnegative (Ne.symm h)
    let ceiling := gainNativeParameterMass initial
    have hc : 0 ≤ ceiling := (gain_native_masses_nonnegative initial hs).1
    have hf := gain_ce_feedback_floor_pos remaining genGain memGain bound ceiling (le_of_lt hgen) hc hclip
    let input := gainCEFeedbackFloor remaining genGain memGain bound ceiling * memGain * (value / 2) / 4
    have hinput : 0 < input := by dsimp only [input]; positivity
    let gap := input / eps - input / (eps + Real.sqrt ((1 - b2) * input ^ 2))
    have hsquare : 0 < input ^ 2 := by simpa only [pow_two] using mul_pos hinput hinput
    have hroot := Real.sqrt_pos.mpr (mul_pos (show 0 < 1 - b2 by linarith only [h2]) hsquare)
    have hdiv := div_lt_div_of_pos_left hinput he
      (show eps < eps + Real.sqrt ((1 - b2) * input ^ 2) by linarith only [hroot])
    have hgap : 0 < gap := by dsimp only [gap]; linarith only [hdiv]
    have hdecrement := mul_pos heta hgap
    obtain ⟨start, htail⟩ := eventually_atTop.mp
      ((ht.eventually_const_lt (show value / 2 < value by linarith only [hv])).and
        ((ht.eventually_const_lt (show value - rate * gap / 4 < value by linarith only [hdecrement])).and
          (ht.eventually_lt_const (show value < value + rate * gap / 4 by linarith only [hdecrement]))))
    have hsign := gain_native_nonnegative_path remaining genGain memGain bound 0 b2 eps decay rate initial start
      hgen hmem hclip (by norm_num) (by norm_num) hb2 h2 he (le_of_lt heta) hkeep hs
    have hbox : ∀ i, 0 ≤ (state start i).parameter ∧ (state start i).parameter ≤ ceiling := by
      intro i
      exact ⟨(hsign i).1, gain_native_zero_first_beta_cold_parameter_box remaining genGain memGain bound b2 eps decay rate initial start i
        hgen hmem hgain hclip hb2 h2 he (le_of_lt heta) hkeep hs hthreshold⟩
    obtain ⟨hpositive, i, hi⟩ := gain_native_box_large_applied_input remaining genGain memGain bound ceiling (value / 2) (state start)
      hgen hmem hgain hclip hc (by linarith only [hv]) hbox (le_of_lt (htail start le_rfl).1)
    have hnegative : appliedGainNativeGradient remaining genGain memGain bound (state start) i ≤ -input := by
      change input ≤ -appliedGainNativeGradient remaining genGain memGain bound (state start) i at hi
      linarith only [hi]
    have hstep := (gain_native_zero_first_beta_cold_mass_gap remaining genGain memGain bound b2 eps decay rate input (state start) i
      hgen hmem hgain hclip hb2 h2 he (le_of_lt heta) hsign hpositive hnegative hthreshold).2
    change mass (start + 1) ≤ mass start - rate * gap at hstep
    have hbefore := (htail start le_rfl).2.2
    have hafter := (htail (start + 1) (by omega)).2.1
    nlinarith only [hstep, hbefore, hafter, hdecrement]
  simpa only [hzero] using ht

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧
    (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [coldGainCEGradientScale]⟩

/-- Every actual initialized physical coordinate tends to zero at
or above the cold threshold with legal first beta zero. Sources:
appendix C's four factors and the generated critical mass collapse
above; coordinate convergence follows from sign-preserving coverage. -/
theorem gain_native_zero_first_beta_cold_parameters_tendsto (remaining : ℕ)
    (genGain memGain bound b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hkeep : 0 ≤ 1 - rate * decay) (hs : NonnegativeNativeState initial)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    ∀ i, Tendsto (fun n => (gainNativePath remaining genGain memGain bound 0 b2 eps decay rate initial n i).parameter)
      atTop (nhds 0) := by
  have hm := gain_native_zero_first_beta_cold_mass_tendsto remaining genGain memGain bound b2 eps decay rate initial
    hgen hmem hgain hclip hb2 h2 he heta hkeep hs hthreshold
  have hsign := fun n => gain_native_nonnegative_path remaining genGain memGain bound 0 b2 eps decay rate initial n
    hgen hmem hclip (by norm_num) (by norm_num) hb2 h2 he (le_of_lt heta) hkeep hs
  intro i
  exact squeeze_zero (fun n => (hsign n i).1)
    (fun n => (gain_native_masses_nonnegative _ (hsign n)).2.2 i) hm

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧
    (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [coldGainCEGradientScale]⟩

/-- Every true clipped CE input vanishes at or above the threshold
with legal first beta zero. Sources: appendix C partner derivatives,
actual cold input cap at 696630b and critical mass collapse above;
no independent convergent gradient stream is supplied. -/
theorem gain_native_zero_first_beta_cold_inputs_tendsto (remaining : ℕ)
    (genGain memGain bound b2 eps decay rate : ℝ) (initial : NativeSubweightState)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hkeep : 0 ≤ 1 - rate * decay) (hs : NonnegativeNativeState initial)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    ∀ i, Tendsto (fun n => appliedGainNativeGradient remaining genGain memGain bound
      (gainNativePath remaining genGain memGain bound 0 b2 eps decay rate initial n) i) atTop (nhds 0) := by
  have hm := gain_native_zero_first_beta_cold_mass_tendsto remaining genGain memGain bound b2 eps decay rate initial
    hgen hmem hgain hclip hb2 h2 he heta hkeep hs hthreshold
  have hsign := fun n => gain_native_nonnegative_path remaining genGain memGain bound 0 b2 eps decay rate initial n
    hgen hmem hclip (by norm_num) (by norm_num) hb2 h2 he (le_of_lt heta) hkeep hs
  have hupper : Tendsto (fun n => (coldGainCEGradientScale remaining bound * genGain) *
      gainNativeParameterMass (gainNativePath remaining genGain memGain bound 0 b2 eps decay rate initial n))
      atTop (nhds 0) := by simpa only [mul_zero] using hm.const_mul (coldGainCEGradientScale remaining bound * genGain)
  intro i
  have hcap := fun n => gain_native_cold_applied_gradient_mass_cap remaining genGain memGain bound _ i
    hgen hmem hgain hclip (hsign n)
  have ht := squeeze_zero (fun n => (hcap n).1) (fun n => (hcap n).2) hupper
  simpa only [neg_neg, neg_zero] using ht.neg

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧
    (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num,
    native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [coldGainCEGradientScale]⟩

end Transformer.Grokking.CircuitEfficiency
