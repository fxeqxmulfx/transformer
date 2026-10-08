import Transformer.Grokking.CircuitEfficiency.SectionC_GainSmallDecayAbsent
import Transformer.Grokking.CircuitEfficiency.SectionC_GainPairEnvelopes

/-!
# Generated original weak-decay pair data near an absent-Gen boundary

Sources: Varma et al., arXiv:2309.02390v1, section 3 slow formation
and appendix C product CE; original retained controls at 7977534,
comparison at 3bab55a and actual pair envelopes at 7a2f5b7.

A positive source-style Gen partner gives positive Gen parameter
and retained negative-moment masses at every successor. For every
native coordinate, completed clocks generate an eventual denominator
floor 4/5 times epsilon, independently of parameter/input convergence.

If Gen parameter mass tends to zero, the unit-interval actual CE
scale bounds both of its actual applied inputs by three times that
mass. They therefore tend to zero without any Mem limit premise.
Their retained variances and correction clocks then generate both
Gen denominator ceilings at 6/5 times epsilon. The ratio of this
ceiling to the common floor is exactly the physical gain ratio 3/2.

The hypothetical collapse lemmas allow seed zero: the fully absent
actual Gen trajectory supplies a jointly satisfying example. No
positive-seed future collapse is postulated as an input to a theorem
whose hypotheses would already be contradictory. The next step must
combine generated tails and the positive current comparison cone.

These are original native histories without resets or an assumed Mem
attractor. They do not yet assert winning Gen dominance, confidence
stability, learned GPTMini behavior or a thermodynamic system limit.
Fixed tables and uniform native decay differ from coupled-cost GD.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Actual absent-Gen mass tends to zero through its retained state
invariance. Sources: appendix C products and original control at
7977534; this supplies a nonempty hypothetical-boundary example. -/
theorem small_decay_gain_absent_gen_mass_tendsto (remaining : ℕ) :
    Tendsto (fun n => gainGenParameterMass (smallDecayGainPath remaining 0 n)) atTop (nhds 0) := by
  apply Tendsto.congr _ tendsto_const_nhds
  intro n
  rw [gainGenParameterMass, (small_decay_gain_absent_gen_history remaining n).1,
    (small_decay_gain_absent_gen_history remaining n).2]
  norm_num [zeroScalarStateAt]

/-- Initial positive Gen partner generates positive actual parameter
and negative retained-moment masses at every successor. Sources:
section 3 formation, appendix C true partials and native signs at
65ce284; a positive future mass is a conclusion, not a reference. -/
theorem small_decay_gain_gen_masses_positive_successor (remaining : ℕ) (seed : ℝ) (n : ℕ)
    (hseed : 0 < seed) :
    0 < gainGenParameterMass (smallDecayGainPath remaining seed (n + 1)) ∧
      0 < gainGenNegativeMomentMass (smallDecayGainPath remaining seed (n + 1)) := by
  obtain ⟨_, _, _, hsign⟩ := small_decay_gain_uniform_partner_floor remaining seed (le_of_lt hseed)
  have hp1 : ∀ k, 0 < (smallDecayGainPath remaining seed k 1).parameter := by
    intro k
    exact (gain_native_positive_coordinate_path remaining 3 2 1 (9 / 10) (49 / 50)
      (1 / 100000000) (1 / 10) (1 / 1000) (seededNativeSubweights ((0, seed), (0, 1))) k 1
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num)
      (native_seeded_nonnegative _ _ _ _ le_rfl (le_of_lt hseed) le_rfl (by norm_num)) hseed).1
  have hp : ∀ k, 0 < gainGenParameterMass (smallDecayGainPath remaining seed k) := by
    intro k
    unfold gainGenParameterMass
    linarith only [((hsign k).1 0).1, hp1 k]
  refine ⟨hp (n + 1), ?_⟩
  have hm := (gain_gen_masses_nonnegative _ (hsign n).1).2
  have hs := gain_ce_gradient_scale_pos remaining 3 2 1 (smallDecayGainPath remaining seed n) (by norm_num)
  change 0 < gainGenNegativeMomentMass (gainNativeStep remaining 3 2 1 (9 / 10) (49 / 50)
    (1 / 100000000) (1 / 10) (1 / 1000) (smallDecayGainPath remaining seed n))
  rw [gain_gen_moment_mass_step]
  have hpn := hp n
  have hi : 0 < (1 - (9 / 10 : ℝ)) *
      (gainCEGradientScale remaining 3 2 1 (smallDecayGainPath remaining seed n) * 3) *
        gainGenParameterMass (smallDecayGainPath remaining seed n) := by positivity
  nlinarith only [hm, hi]

example : (0 : ℝ) < 1 / 200 := by norm_num

/-- Every original coordinate has a generated eventual near-epsilon
denominator floor. Source: native completed-clock denominator at
79f4fb0; no convergence of inputs, variance or physical factors is assumed. -/
theorem small_decay_gain_denominator_tail_floor (remaining : ℕ) (seed : ℝ) (i : Fin 4) :
    ∃ start : ℕ, ∀ n, start ≤ n → (4 / 5 : ℝ) * (1 / 100000000) ≤
      nextBufferDenominator (9 / 10) (49 / 50) (1 / 100000000) (smallDecayGainPath remaining seed n i).variance
        (appliedGainNativeGradient remaining 3 2 1 (smallDecayGainPath remaining seed n) i)
        (smallDecayGainPath remaining seed n i).clock := by
  have hstep : ∀ n, smallDecayGainPath remaining seed (n + 1) i =
      scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
        (smallDecayGainPath remaining seed n i) (appliedGainNativeGradient remaining 3 2 1 (smallDecayGainPath remaining seed n) i) :=
    fun _ => rfl
  have hclock := scalar_completed_clock_tendsto (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
    (fun n => smallDecayGainPath remaining seed n i)
    (fun n => appliedGainNativeGradient remaining 3 2 1 (smallDecayGainPath remaining seed n) i) hstep
  have hpower := (tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num : (0 : ℝ) ≤ 9 / 10)
    (by norm_num : (9 / 10 : ℝ) < 1)).comp hclock
  have hfloor : Tendsto (fun n => (1 - (9 / 10 : ℝ) ^ ((smallDecayGainPath remaining seed n i).clock + 1)) *
      (1 / 100000000)) atTop (nhds (1 / 100000000 : ℝ)) := by
    have hconst : Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 1) := tendsto_const_nhds
    simpa only [Function.comp_apply, sub_zero, one_mul] using
      (hconst.sub hpower).mul_const (1 / 100000000)
  obtain ⟨start, htail⟩ := eventually_atTop.mp (hfloor.eventually_const_lt (by norm_num :
    (4 / 5 : ℝ) * (1 / 100000000) < 1 / 100000000))
  refine ⟨start, ?_⟩
  intro n hn
  have hb := le_of_lt (bias_correction_positive (9 / 10) ((smallDecayGainPath remaining seed n i).clock + 1)
    (by norm_num) (by norm_num) (by omega))
  have hs := mul_nonneg hb (Real.sqrt_nonneg
    (((49 / 50 : ℝ) * (smallDecayGainPath remaining seed n i).variance + (1 - 49 / 50) *
      appliedGainNativeGradient remaining 3 2 1 (smallDecayGainPath remaining seed n) i ^ 2) /
      (1 - (49 / 50 : ℝ) ^ ((smallDecayGainPath remaining seed n i).clock + 1))))
  unfold nextBufferDenominator
  nlinarith only [hs, htail n hn]

/-- Gen-only physical mass collapse forces both actual Gen inputs
to zero without a Mem convergence premise. Sources: appendix C
partner partials and the unit-interval actual scale at 65ce284. -/
theorem small_decay_gain_gen_collapse_inputs (remaining : ℕ) (seed : ℝ) (hseed : 0 ≤ seed)
    (hcollapse : Tendsto (fun n => gainGenParameterMass (smallDecayGainPath remaining seed n)) atTop (nhds 0)) :
    ∀ i : Fin 4, i = 0 ∨ i = 1 → Tendsto
      (fun n => appliedGainNativeGradient remaining 3 2 1 (smallDecayGainPath remaining seed n) i) atTop (nhds 0) := by
  obtain ⟨_, _, _, hsign⟩ := small_decay_gain_uniform_partner_floor remaining seed hseed
  have hupper : Tendsto (fun n => 3 * gainGenParameterMass (smallDecayGainPath remaining seed n)) atTop (nhds 0) := by
    simpa only [mul_zero] using hcollapse.const_mul 3
  intro i hi
  have hbound : ∀ n, 0 ≤ -appliedGainNativeGradient remaining 3 2 1 (smallDecayGainPath remaining seed n) i ∧
      -appliedGainNativeGradient remaining 3 2 1 (smallDecayGainPath remaining seed n) i ≤
        3 * gainGenParameterMass (smallDecayGainPath remaining seed n) := by
    intro n
    have hp := ((hsign n).1 (nativeFactorPartner i)).1
    have hmass : (smallDecayGainPath remaining seed n (nativeFactorPartner i)).parameter ≤
        gainGenParameterMass (smallDecayGainPath remaining seed n) := by
      rcases hi with rfl | rfl
      · change (smallDecayGainPath remaining seed n 1).parameter ≤
          (smallDecayGainPath remaining seed n 0).parameter + (smallDecayGainPath remaining seed n 1).parameter
        linarith only [((hsign n).1 0).1]
      · change (smallDecayGainPath remaining seed n 0).parameter ≤
          (smallDecayGainPath remaining seed n 0).parameter + (smallDecayGainPath remaining seed n 1).parameter
        linarith only [((hsign n).1 1).1]
    have hscale := gain_ce_gradient_scale_unit_interval remaining 3 2 1 (smallDecayGainPath remaining seed n) (by norm_num)
    have heq : appliedGainNativeGradient remaining 3 2 1 (smallDecayGainPath remaining seed n) i =
        -(gainCEGradientScale remaining 3 2 1 (smallDecayGainPath remaining seed n) * 3 *
          (smallDecayGainPath remaining seed n (nativeFactorPartner i)).parameter) := by
      rw [gain_native_applied_gradient_scale]
      rcases hi with rfl | rfl <;> rfl
    have ht := mul_le_mul_of_nonneg_right (le_of_lt hscale.2) (show 0 ≤ 3 *
      (smallDecayGainPath remaining seed n (nativeFactorPartner i)).parameter by positivity)
    rw [heq, neg_neg]
    refine ⟨mul_nonneg (mul_nonneg (le_of_lt hscale.1) (by norm_num)) hp, ?_⟩
    nlinarith only [ht, hmass]
  have ht := squeeze_zero (fun n => (hbound n).1) (fun n => (hbound n).2) hupper
  simpa only [neg_neg, neg_zero] using ht.neg

example : (0 : ℝ) ≤ 0 ∧ Tendsto (fun n => gainGenParameterMass (smallDecayGainPath 0 0 n)) atTop (nhds 0) :=
  ⟨le_rfl, small_decay_gain_absent_gen_mass_tendsto 0⟩

/-- Hypothetical Gen-only collapse generates both actual retained
Gen denominator ceilings. Sources: appendix C feedback and native
vanishing-input denominator law at 79f4fb0; Mem need not converge. -/
theorem small_decay_gain_gen_collapse_denominator_tail (remaining : ℕ) (seed : ℝ) (hseed : 0 ≤ seed)
    (hcollapse : Tendsto (fun n => gainGenParameterMass (smallDecayGainPath remaining seed n)) atTop (nhds 0)) :
    ∃ start : ℕ, ∀ n, start ≤ n → ∀ i : Fin 4, i = 0 ∨ i = 1 →
      nextBufferDenominator (9 / 10) (49 / 50) (1 / 100000000) (smallDecayGainPath remaining seed n i).variance
        (appliedGainNativeGradient remaining 3 2 1 (smallDecayGainPath remaining seed n) i)
        (smallDecayGainPath remaining seed n i).clock ≤ (6 / 5 : ℝ) * (1 / 100000000) := by
  have hinput := small_decay_gain_gen_collapse_inputs remaining seed hseed hcollapse
  have htail : ∀ i : Fin 4, i = 0 ∨ i = 1 → ∀ᶠ n in atTop,
      nextBufferDenominator (9 / 10) (49 / 50) (1 / 100000000) (smallDecayGainPath remaining seed n i).variance
        (appliedGainNativeGradient remaining 3 2 1 (smallDecayGainPath remaining seed n) i)
        (smallDecayGainPath remaining seed n i).clock < (6 / 5 : ℝ) * (1 / 100000000) := by
    intro i hi
    exact scalar_zero_next_denominator_tail (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000) _
      (fun n => smallDecayGainPath remaining seed n i) _ (fun _ => rfl)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (hinput i hi) (by norm_num)
  obtain ⟨start, hh⟩ := eventually_atTop.mp ((htail 0 (Or.inl rfl)).and (htail 1 (Or.inr rfl)))
  refine ⟨start, ?_⟩
  intro n hn i hi
  rcases hi with rfl | rfl
  · exact le_of_lt (hh n hn).1
  · exact le_of_lt (hh n hn).2

example : (0 : ℝ) ≤ 0 ∧ Tendsto (fun n => gainGenParameterMass (smallDecayGainPath 0 0 n)) atTop (nhds 0) :=
  ⟨le_rfl, small_decay_gain_absent_gen_mass_tendsto 0⟩

end Transformer.Grokking.CircuitEfficiency
