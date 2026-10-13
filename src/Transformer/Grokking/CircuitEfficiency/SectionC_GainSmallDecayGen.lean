import Transformer.Grokking.CircuitEfficiency.SectionC_GainSmallDecayBoundary

/-!
# Original weak-decay Gen cannot asymptotically disappear

Sources: Varma et al., arXiv:2309.02390v1, section 3's competing
circuits and appendix C product CE; original comparison at 3bab55a,
opposing envelopes at 7a2f5b7 and generated boundary tails at fd2347b.

Keep beta1=0.9, beta2=0.98, epsilon=1e-8, decay=0.1, rate=0.001,
cap=1, gains 3/2, Gen (0,seed), Mem (0,1), and both native buffers.
For every positive seed, Gen parameter mass cannot tend to zero,
without a Mem parameter limit or any future success as a premise.

A hypothetical collapse generates Gen denominator ceilings
6/5*epsilon and Mem floors 4/5*epsilon. At a retained positive clock,
actual Gen parameter and negative-moment masses choose a positive
comparison factor. The generated native envelopes preserve both
scaled inequalities over the tail. Hence Mem mass would also tend
to zero, contradicting the initialized total-mass cold obstruction.

The component-specific Gen mass therefore reaches a fixed positive
level arbitrarily late. Uniform actual partner mixing transfers its
excursions to the actual gained Gen product logit. Mem-only training
cannot account for these component measurements as it could for
total norm, train accuracy, train confidence or train CE.

Recurrence permits oscillation: neither a uniform positive Gen tail
floor nor eventual Gen/Mem dominance or correct held-out answers is
claimed. The circuit is a fixed gained table; learned head formation
and stochastic/floating-point GPTMini transfer remain open. Uniform
native decay differs from the source's coupled assigned norm-cost GD.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Actual original gained Gen product at the supplied clock.
Sources: appendix C Gen forward and initialized native path at 6304fbe;
this numerical observer does not encode a successful task predicate. -/
noncomputable def smallDecayGainGenScore (remaining : ℕ) (seed : ℝ) (n : ℕ) : ℝ :=
  physicalCircuitScore 3 (smallDecayGainPath remaining seed n 0).parameter
    (smallDecayGainPath remaining seed n 1).parameter

/-- Gen cannot collapse on the actual original weak-decay path.
Sources: section 3 efficiency, appendix C actual partner CE and native
boundary comparison at 3bab55a/fd2347b; no Mem convergence is assumed. -/
theorem small_decay_gain_gen_not_mass_collapse (remaining : ℕ) (seed : ℝ) (hseed : 0 < seed) :
    ¬Tendsto (fun n => gainGenParameterMass (smallDecayGainPath remaining seed n)) atTop (nhds 0) := by
  intro hcollapse
  let state := smallDecayGainPath remaining seed
  obtain ⟨startGen, hGen⟩ := small_decay_gain_gen_collapse_denominator_tail remaining seed (le_of_lt hseed) hcollapse
  obtain ⟨startMem2, hMem2⟩ := small_decay_gain_denominator_tail_floor remaining seed 2
  obtain ⟨startMem3, hMem3⟩ := small_decay_gain_denominator_tail_floor remaining seed 3
  obtain ⟨_, _, _, hsign⟩ := small_decay_gain_uniform_partner_floor remaining seed (le_of_lt hseed)
  let start := startGen + startMem2 + startMem3 + 1
  have hp : 0 < gainGenParameterMass (state start) ∧ 0 < gainGenNegativeMomentMass (state start) :=
    small_decay_gain_gen_masses_positive_successor remaining seed (startGen + startMem2 + startMem3) hseed
  have hm := gain_mem_masses_nonnegative (state start) (hsign start).1
  obtain ⟨factor, hfactor, hparameter, hmoment⟩ := retained_pair_comparison_initial_factor
    (gainGenParameterMass (state start)) (gainMemParameterMass (state start))
    (gainGenNegativeMomentMass (state start)) (gainMemNegativeMomentMass (state start)) (3 / 2)
    hp.1 hp.2 hm.1 hm.2 (by norm_num)
  have hstep : ∀ n, state (n + 1) = gainNativeStep remaining 3 2 1 (9 / 10) (49 / 50)
      (1 / 100000000) (1 / 10) (1 / 1000) (state n) := fun _ => rfl
  have hcone := retained_pair_comparison_tail (9 / 10) (1 - (1 / 1000) * (1 / 10)) (1 / 1000)
    ((6 / 5) * (1 / 100000000)) ((4 / 5) * (1 / 100000000)) (3 / 2) factor
    (fun n => gainCEGradientScale remaining 3 2 1 (state n) * 2)
    (fun n => gainGenParameterMass (state n)) (fun n => gainMemParameterMass (state n))
    (fun n => gainGenNegativeMomentMass (state n)) (fun n => gainMemNegativeMomentMass (state n)) start
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (le_of_lt hfactor) (fun n => mul_nonneg
      (le_of_lt (gain_ce_gradient_scale_pos remaining 3 2 1 (state n) (by norm_num))) (by norm_num))
    (by norm_num) hparameter hmoment ?_ ?_ ?_ ?_
  · have hupper : Tendsto (fun n => gainGenParameterMass (state n) / factor) atTop (nhds 0) := by
      simpa only [zero_div] using hcollapse.div_const factor
    have htail : ∀ᶠ n in atTop, gainMemParameterMass (state n) ≤ gainGenParameterMass (state n) / factor := by
      apply eventually_atTop.mpr
      refine ⟨start, ?_⟩
      intro n hn
      have heq : start + (n - start) = n := by omega
      have hh : factor * gainMemParameterMass (state n) ≤ gainGenParameterMass (state n) := by
        simpa only [heq] using (hcone (n - start)).1
      exact (le_div_iff₀ hfactor).mpr (by simpa only [mul_comm] using hh)
    have hMemCollapse := squeeze_zero' (Eventually.of_forall (fun n =>
      (gain_mem_masses_nonnegative (state n) (hsign n).1).1)) htail hupper
    have htotal : Tendsto (fun n => gainNativeParameterMass (state n)) atTop (nhds 0) := by
      simpa only [gainNativeParameterMass, gainGenParameterMass, gainMemParameterMass, zero_add] using
        hcollapse.add hMemCollapse
    exact gain_native_small_decay_source_not_mass_collapse remaining seed hseed htotal
  · intro n _
    rw [hstep n, gain_gen_moment_mass_step]
    ring
  · intro n _
    rw [hstep n, gain_mem_moment_mass_step]
  · intro n hn
    rw [hstep n]
    exact gain_gen_parameter_mass_ceiling_denominator remaining 3 2 1 (9 / 10) (49 / 50)
      (1 / 100000000) (1 / 10) (1 / 1000) ((6 / 5) * (1 / 100000000)) (state n)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (hsign n).1 (hGen n (by dsimp [start] at hn; omega) 0 (Or.inl rfl))
      (hGen n (by dsimp [start] at hn; omega) 1 (Or.inr rfl))
  · intro n hn
    rw [hstep n]
    exact gain_mem_parameter_mass_floor_denominator remaining 3 2 1 (9 / 10) (49 / 50)
      (1 / 100000000) (1 / 10) (1 / 1000) ((4 / 5) * (1 / 100000000)) (state n)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (hsign n).1
      (hMem2 n (by dsimp [start] at hn; omega)) (hMem3 n (by dsimp [start] at hn; omega))

example : (0 : ℝ) < 1 / 200 := by norm_num

/-- A component-specific positive Gen mass level occurs arbitrarily
late on the original path. Sources: section 3 internal circuit strength
and native boundary comparison at 3bab55a/fd2347b; no limit is supplied. -/
theorem small_decay_gain_gen_mass_returns (remaining : ℕ) (seed : ℝ) (hseed : 0 < seed) :
    ∃ level : ℝ, 0 < level ∧ ∀ start : ℕ, ∃ n : ℕ, start ≤ n ∧
      level ≤ gainGenParameterMass (smallDecayGainPath remaining seed n) := by
  obtain ⟨_, _, _, hsign⟩ := small_decay_gain_uniform_partner_floor remaining seed (le_of_lt hseed)
  exact nonnegative_noncollapse_recurrent_level _
    (fun n => (gain_gen_masses_nonnegative _ (hsign n).1).1)
    (small_decay_gain_gen_not_mass_collapse remaining seed hseed)

example : (0 : ℝ) < 1 / 200 := by norm_num

/-- A fixed positive level of the actual Gen product logit occurs
arbitrarily late, with both factors learned by retained native AdamW.
Sources: appendix C product forward and actual pair mixing at 6304fbe;
this does not imply that this logit exceeds the competing Mem logit. -/
theorem small_decay_gain_gen_score_returns (remaining : ℕ) (seed : ℝ) (hseed : 0 < seed) :
    ∃ level : ℝ, 0 < level ∧ ∀ start : ℕ, ∃ n : ℕ, start ≤ n ∧
      level ≤ smallDecayGainGenScore remaining seed n := by
  obtain ⟨mass, hmass, hreturn⟩ := small_decay_gain_gen_mass_returns remaining seed hseed
  obtain ⟨coefficient, hc, _, hmix⟩ := small_decay_gain_uniform_pair_mixing remaining seed (le_of_lt hseed)
  refine ⟨3 * (coefficient * mass) ^ 2, by positivity, ?_⟩
  intro start
  obtain ⟨n, hn, hmassNow⟩ := hreturn start
  have hsum := mul_le_mul_of_nonneg_left hmassNow (le_of_lt hc)
  have hleft : coefficient * mass ≤ (smallDecayGainPath remaining seed (n + 1) 0).parameter := by
    have hh := (hmix n).2 0
    change coefficient * gainGenParameterMass (smallDecayGainPath remaining seed n) ≤ _ at hh
    exact le_trans hsum hh
  have hright : coefficient * mass ≤ (smallDecayGainPath remaining seed (n + 1) 1).parameter := by
    have hh := (hmix n).2 1
    have hcomm : coefficient * gainGenParameterMass (smallDecayGainPath remaining seed n) ≤
        (smallDecayGainPath remaining seed (n + 1) 1).parameter := by
      convert hh using 1
      change coefficient * ((smallDecayGainPath remaining seed n 0).parameter +
        (smallDecayGainPath remaining seed n 1).parameter) =
          coefficient * ((smallDecayGainPath remaining seed n 1).parameter + (smallDecayGainPath remaining seed n 0).parameter)
      ring
    exact le_trans hsum hcomm
  have hproduct := mul_le_mul hleft hright (le_of_lt (mul_pos hc hmass)) (((hmix (n + 1)).1 0).1)
  refine ⟨n + 1, by omega, ?_⟩
  unfold smallDecayGainGenScore physicalCircuitScore
  nlinarith only [hproduct]

example : (0 : ℝ) < 1 / 200 := by norm_num

end Transformer.Grokking.CircuitEfficiency
