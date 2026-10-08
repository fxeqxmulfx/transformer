import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryCompetition

/-!
# Initialized powers for actual competing retained-memory measurements

Sources: Varma et al., arXiv:2309.02390v1, section 3 competing
circuits and appendix C factors; original legal-beta pair rate tails
at lab commit cc9fd5f.

The fixed configuration has equal remaining decay and beta1, both
0.9. True nonpositive clipped-CE inputs give an unconditional 0.9
floor for the complete Gen parameter/moment measurement. Iterate
that current floor from the original full retained initialization.
The initialization may contain nonzero retained buffers and clocks.
A positive initial numerical Gen measurement stays positive at every
finite clock, even though its absolute limit can be zero.

The separately generated tail rates 0.911 and 0.910 give Gen lower
and Mem upper powers measured from the same derived start clock.
No future convergence, coefficient interval, successful margin or
external gradient stream is independently supplied.

These powers prepare a relative ratio limit. Complete measurements
include retained first moments; physical product selection and
confidence require further arguments. Fixed gained tables/native
decay differ from appendix C coupled-cost GD. Learned stochastic/
floating-point GPTMini transfer is not asserted.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- The original current Gen parameter/moment measurement stays
above 0.9 times itself. Sources: appendix C true partner partials and
native sign/decay laws at cc9fd5f; equal beta1 and remaining decay
are the explicit pinned constants, not a future positivity premise. -/
theorem fixed_gain_gen_memory_step_floor (state : NativeSubweightState)
    (hs : NonnegativeNativeState state) :
    (9 / 10 : ℝ) * fixedGainGenMemoryMass state ≤
      fixedGainGenMemoryMass (gainNativeStep 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) state) := by
  have h0 := gain_native_parameter_decay_floor 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) state 0
    (by norm_num) (by norm_num [nativeFactorGain]) (hs 1).1
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (hs 0).2.1
  have h1 := gain_native_parameter_decay_floor 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) state 1
    (by norm_num) (by norm_num [nativeFactorGain]) (hs 0).1
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (hs 1).2.1
  have hm := gain_gen_moment_mass_step 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) state
  have hextra := mul_nonneg (le_of_lt (gain_ce_gradient_scale_pos 0 3 2 1 state (by norm_num)))
    (gain_gen_masses_nonnegative state hs).1
  unfold gainGenNegativeMomentMass gainGenParameterMass at hm hextra
  unfold fixedGainGenMemoryMass gainNativePairMemoryMass
  change (9 / 10 : ℝ) * ((state 0).parameter + (state 1).parameter + (2 / 25) * (-(state 0).moment - (state 1).moment)) ≤ _
  change _ ≤ (gainNativeStep 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) state 0).parameter +
    (gainNativeStep 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) state 1).parameter +
      (2 / 25) * (-(gainNativeStep 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) state 0).moment -
        (gainNativeStep 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) state 1).moment)
  nlinarith only [h0, h1, hm, hextra]

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Initial full retained signs generate the complete Gen floor
power and nonnegative Mem measurements at every actual clock. Sources:
appendix C factors and native current floor at cc9fd5f; no positive
or convergent future Gen measurement is assumed. -/
theorem fixed_gain_memory_initial_power_floor (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) :
    ∀ n, (9 / 10 : ℝ) ^ n * fixedGainGenMemoryMass initial ≤
      fixedGainGenMemoryMass (fixedGainMemoryPath initial n) ∧
      0 ≤ fixedGainMemMemoryMass (fixedGainMemoryPath initial n) := by
  have hg : ∀ n, (9 / 10 : ℝ) ^ n * fixedGainGenMemoryMass initial ≤
      fixedGainGenMemoryMass (fixedGainMemoryPath initial n) := by
    intro n
    induction n with
    | zero =>
      change (9 / 10 : ℝ) ^ 0 * fixedGainGenMemoryMass initial ≤ fixedGainGenMemoryMass initial
      simp only [pow_zero, one_mul, le_refl]
    | succ n ih =>
      have hstep := fixed_gain_gen_memory_step_floor (fixedGainMemoryPath initial n) (fixed_gain_memory_signs initial hs n)
      have hl := le_trans (mul_le_mul_of_nonneg_left ih (by norm_num : (0 : ℝ) ≤ 9 / 10)) hstep
      convert hl using 1
      · rw [pow_succ]
        ring
      · rfl
  intro n
  have hm := gain_native_pair_memory_mass_covers (19 / 200) (fixedGainMemoryPath initial n) 2
    (by norm_num) (fixed_gain_memory_signs initial hs n)
  exact ⟨hg n, le_trans hm.1 hm.2⟩

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- A positive present Gen parameter/moment measurement stays
positive at every finite original clock. Sources: section 3 small Gen
seeds and native power floor at cc9fd5f; positive finite values do not
imply a strictly positive limiting parameter or measurement. -/
theorem fixed_gain_gen_memory_positive_path (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) (hgen : 0 < fixedGainGenMemoryMass initial) :
    ∀ n, 0 < fixedGainGenMemoryMass (fixedGainMemoryPath initial n) := by
  intro n
  have hp := mul_pos (pow_pos (by norm_num : (0 : ℝ) < 9 / 10) n) hgen
  exact lt_of_lt_of_le hp (fixed_gain_memory_initial_power_floor initial hs n).1

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) ∧
    0 < fixedGainGenMemoryMass (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact ⟨native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    by norm_num [fixedGainGenMemoryMass, gainNativePairMemoryMass, seededNativeSubweights, seededScalarState, nativeFactorPartner]⟩

/-- The generated common tail gives full actual Gen lower and Mem
upper powers from its original retained state. Sources: section 3
relative competition and native pair rates at cc9fd5f; the start is
produced by the initialized path and is not a future input hypothesis. -/
theorem fixed_gain_memory_tail_power_bounds (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) :
    ∃ start : ℕ, ∀ k,
      (911 / 1000 : ℝ) ^ k * fixedGainGenMemoryMass (fixedGainMemoryPath initial start) ≤
        fixedGainGenMemoryMass (fixedGainMemoryPath initial (start + k)) ∧
      fixedGainMemMemoryMass (fixedGainMemoryPath initial (start + k)) ≤
        (91 / 100 : ℝ) ^ k * fixedGainMemMemoryMass (fixedGainMemoryPath initial start) := by
  obtain ⟨start, htail⟩ := fixed_gain_memory_pair_rate_tail initial hs
  refine ⟨start, ?_⟩
  intro k
  induction k with
  | zero =>
    simp only [pow_zero, one_mul, Nat.add_zero, le_refl, and_self]
  | succ k ih =>
    have hstep := htail (start + k) (by omega)
    constructor
    · calc
        (911 / 1000 : ℝ) ^ (k + 1) * fixedGainGenMemoryMass (fixedGainMemoryPath initial start) =
            (911 / 1000) * ((911 / 1000) ^ k * fixedGainGenMemoryMass (fixedGainMemoryPath initial start)) := by
          rw [pow_succ]
          ring
        _ ≤ (911 / 1000) * fixedGainGenMemoryMass (fixedGainMemoryPath initial (start + k)) :=
          mul_le_mul_of_nonneg_left ih.1 (by norm_num)
        _ ≤ fixedGainGenMemoryMass (fixedGainMemoryPath initial (start + (k + 1))) := hstep.1
    · calc
        fixedGainMemMemoryMass (fixedGainMemoryPath initial (start + (k + 1))) ≤
            (91 / 100) * fixedGainMemMemoryMass (fixedGainMemoryPath initial (start + k)) := hstep.2.2
        _ ≤ (91 / 100) * ((91 / 100) ^ k * fixedGainMemMemoryMass (fixedGainMemoryPath initial start)) :=
          mul_le_mul_of_nonneg_left ih.2 (by norm_num)
        _ = (91 / 100 : ℝ) ^ (k + 1) * fixedGainMemMemoryMass (fixedGainMemoryPath initial start) := by
          rw [pow_succ]
          ring

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.CircuitEfficiency
