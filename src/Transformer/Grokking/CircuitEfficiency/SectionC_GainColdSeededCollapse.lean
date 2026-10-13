import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdWeightedDissipation
import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdMassDissipation

/-!
# Actual initialized critical collapse with retained native betas

Sources: Varma et al., arXiv:2309.02390v1, section 3 CE/decay
competition and appendix C product partials; actual weighted limit
at 4748aba and retained weighted dissipation at dd45898.

For standard nonnegative zero-buffer seeds, total physical mass
tends to zero at or above the greater-gain cold threshold, including
critical equality. Both legal native betas are arbitrary. In
particular beta1=0.9 and beta2=0.98 retain their actual histories.

Positive decay and clipping generate a finite physical box. Above
any fixed positive mass level, its true CE floor gives a nonzero
actual coordinate input. Late completed clocks make that coordinate's
retained denominator strictly exceed epsilon, producing a fixed
positive weighted decrement. The other coordinates contribute only
the geometric clock error. The actual weighted observer already
converges, so this decrement contradicts its late small increments.

No future parameter, input, buffer or denominator convergence is a
premise. Physical mass need not be monotone. The input floor is
generated only when the current mass exceeds the chosen level;
there is no independently prescribed gradient stream. Positive rate
is necessary and the remaining-decay factor may be zero.

This extends the legal first-beta-zero control at aadb3b7 to the
original retained moments. Fixed gained tables/plain CE/uniform
decoupled native decay differ from appendix C's assigned coupled
norm-cost GD. Exact reals and non-AMSGrad AdamW remain explicit.
Absolute collapse does not determine eventual relative Gen/Mem
selection or the transition time. Learned stochastic/numerical
GPTMini and thermodynamic size scaling remain separate claims.
Frozen transformer experiments and optimizers are unchanged.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- The actual initialized physical mass vanishes at or above the
cold threshold for arbitrary legal retained betas. Sources: appendix
C true partner CE, generated weighted limit at 4748aba and actual
decrement at dd45898; no future convergence is supplied as input. -/
theorem gain_native_seeded_cold_mass_tendsto (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hkeep : 0 ≤ 1 - rate * decay) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    Tendsto (fun n => gainNativeParameterMass (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate
      (seededNativeSubweights ((a, b), (c, d))) n)) atTop (nhds 0) := by
  let initial := seededNativeSubweights ((a, b), (c, d))
  let state := gainNativePath remaining genGain memGain bound b1 b2 eps decay rate initial
  let mass := fun n => gainNativeParameterMass (state n)
  let weighted := fun n => coldGainNativeWeightedMass b1 eps rate (state n)
  have hpositive := lt_of_lt_of_le
    (mul_pos (cold_gain_ce_gradient_scale_pos remaining bound hclip) hgen) hthreshold
  have hdecay : 0 < decay := by nlinarith only [hpositive, he]
  obtain ⟨ceiling, hceiling, hbounds⟩ := gain_native_seeded_bounded remaining genGain memGain bound b1 b2 eps decay rate a b c d
    hgen hmem hclip hb1 h1 hb2 h2 he hdecay (le_of_lt heta) hkeep ha hb hc hd
  obtain ⟨value, _, hw⟩ := gain_native_seeded_cold_weighted_mass_limit remaining genGain memGain bound b1 b2 eps decay rate a b c d
    hgen hmem hgain hclip hb1 h1 hb2 h2 he (le_of_lt heta) hkeep ha hb hc hd hthreshold
  change Tendsto weighted atTop (nhds value) at hw
  have hmass : ∀ n, 0 ≤ mass n := fun n => (gain_native_masses_nonnegative (state n) (hbounds n).1).1
  apply tendsto_order.mpr
  constructor
  · intro level hlevel
    exact Eventually.of_forall (fun n => lt_of_lt_of_le hlevel (hmass n))
  · intro level hlevel
    have hf := gain_ce_feedback_floor_pos remaining genGain memGain bound ceiling (le_of_lt hgen) (le_of_lt hceiling) hclip
    let input := gainCEFeedbackFloor remaining genGain memGain bound ceiling * memGain * level / 4
    have hinput : 0 < input := by dsimp only [input]; positivity
    have hsquare : 0 < input ^ 2 := by simpa only [pow_two] using mul_pos hinput hinput
    have hroot := Real.sqrt_pos.mpr (mul_pos (show 0 < 1 - b2 by linarith only [h2]) hsquare)
    let floor := eps + Real.sqrt ((1 - b2) * input ^ 2) / 2
    have hfloor : eps < floor := by dsimp only [floor]; linarith only [hroot]
    have hsubfloor : floor < eps + Real.sqrt ((1 - b2) * input ^ 2) := by dsimp only [floor]; linarith only [hroot]
    let gap := ((1 - b1) * input) / eps - ((1 - b1) * input) / floor
    have hdiv := div_lt_div_of_pos_left (mul_pos (show 0 < 1 - b1 by linarith only [h1]) hinput) he hfloor
    have hgap : 0 < gap := by dsimp only [gap]; linarith only [hdiv]
    let decrement := ((1 - b1) * eps) * rate * gap
    have hdecrement : 0 < decrement := by dsimp only [decrement]; positivity
    obtain ⟨clockStart, hdenTail⟩ := next_buffer_retained_input_floor_tail b1 b2 eps input floor
      hb1 h1 hb2 h2 (le_of_lt hinput) hsubfloor
    have herror : Tendsto (fun n : ℕ => (4 * rate * bound) * b1 ^ n) atTop (nhds 0) := by
      simpa only [mul_zero] using (tendsto_pow_atTop_nhds_zero_of_lt_one hb1 h1).const_mul (4 * rate * bound)
    obtain ⟨timeStart, htail⟩ := eventually_atTop.mp
      ((hw.eventually_lt_const (show value < value + decrement / 4 by linarith only [hdecrement])).and
        ((hw.eventually_const_lt (show value - decrement / 4 < value by linarith only [hdecrement])).and
          (herror.eventually_lt_const (show 0 < decrement / 4 by linarith only [hdecrement]))))
    apply eventually_atTop.mpr
    refine ⟨max timeStart clockStart, ?_⟩
    intro n hn
    have hnTime : timeStart ≤ n := by omega
    have hnClock : clockStart ≤ n := by omega
    by_contra h
    have hmassLower : level ≤ mass n := le_of_not_gt h
    have hbox : ∀ i, 0 ≤ (state n i).parameter ∧ (state n i).parameter ≤ ceiling :=
      fun i => ⟨((hbounds n).1 i).1, ((hbounds n).2 i).1⟩
    obtain ⟨_, i, hi⟩ := gain_native_box_large_applied_input remaining genGain memGain bound ceiling level (state n)
      hgen hmem hgain hclip (le_of_lt hceiling) hlevel hbox hmassLower
    change input ≤ -appliedGainNativeGradient remaining genGain memGain bound (state n) i at hi
    have hnegative : appliedGainNativeGradient remaining genGain memGain bound (state n) i ≤ -input := by linarith only [hi]
    have hnonpos : appliedGainNativeGradient remaining genGain memGain bound (state n) i ≤ 0 := by linarith only [hnegative, hinput]
    have hclock : ∀ j, (state n j).clock = n := by
      intro j
      rw [gain_native_path_clocks]
      fin_cases j <;> simp [initial, seededNativeSubweights, seededScalarState]
    have hden : floor ≤ nextBufferDenominator b1 b2 eps (state n i).variance
        (appliedGainNativeGradient remaining genGain memGain bound (state n) i) (state n i).clock := by
      rw [hclock i]
      exact hdenTail n hnClock _ _ ((hbounds n).1 i).2.2 (by simpa only [abs_of_nonpos hnonpos] using hi)
    have hstep := (gain_native_cold_weighted_mass_gap remaining n genGain memGain bound b1 b2 eps decay rate input floor (state n) i
      hgen hmem hgain hclip hb1 h1 he (le_of_lt heta) (hbounds n).1 (fun j => ((hbounds n).2 j).2.1)
      hclock hinput hnegative hfloor hden hthreshold).2
    change weighted (n + 1) ≤ weighted n + (4 * rate * bound) * b1 ^ n - decrement at hstep
    have hbefore := (htail n hnTime).1
    have hafter := (htail (n + 1) (by omega)).2.1
    have herr := (htail n hnTime).2.2
    nlinarith only [hstep, hbefore, hafter, herr, hdecrement]

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  norm_num [coldGainCEGradientScale]

/-- Every actual initialized physical coordinate vanishes, including
critical equality with nonzero first beta. Sources: appendix C's
four factors and the generated mass convergence above; numerical
sign induction covers each coordinate by total physical mass. -/
theorem gain_native_seeded_cold_parameters_tendsto (remaining : ℕ)
    (genGain memGain bound b1 b2 eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hkeep : 0 ≤ 1 - rate * decay) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hthreshold : coldGainCEGradientScale remaining bound * genGain ≤ decay * eps) :
    ∀ i, Tendsto (fun n => (gainNativePath remaining genGain memGain bound b1 b2 eps decay rate
      (seededNativeSubweights ((a, b), (c, d))) n i).parameter) atTop (nhds 0) := by
  have hm := gain_native_seeded_cold_mass_tendsto remaining genGain memGain bound b1 b2 eps decay rate a b c d
    hgen hmem hgain hclip hb1 h1 hb2 h2 he heta hkeep ha hb hc hd hthreshold
  have hsign := fun n => gain_native_nonnegative_path remaining genGain memGain bound b1 b2 eps decay rate
    (seededNativeSubweights ((a, b), (c, d))) n hgen hmem hclip hb1 h1 hb2 h2 he (le_of_lt heta) hkeep
      (native_seeded_nonnegative a b c d ha hb hc hd)
  intro i
  exact squeeze_zero (fun n => (hsign n i).1)
    (fun n => (gain_native_masses_nonnegative _ (hsign n)).2.2 i) hm

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 1 - (1 / 1000) * (3 / 2) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    coldGainCEGradientScale 0 1 * 3 ≤ (3 / 2 : ℝ) * 1 := by
  norm_num [coldGainCEGradientScale]

end Transformer.Grokking.CircuitEfficiency
