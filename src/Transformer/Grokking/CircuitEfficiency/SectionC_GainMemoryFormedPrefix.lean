import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryFormation
import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryDelay

/-!
# Arbitrary wrong-test prefixes after true double-zero native formation

Sources: Varma et al., arXiv:2309.02390v1, section 3 Slow vs fast
learning and appendix C initially zero first factors; original uniform
formed Mem amplitude, Gen envelope and full-state shift at lab commit
fe86232, actual retained wrong-prefix comparison at f31f874.

Keep one binary task and the original gains 3/2, beta1=0.9,
beta2=0.98, epsilon/cap 1, decay 100 and rate 0.001. Both products
are initially zero: Gen=(0,seed), Mem=(0,1), with fresh initial buffers.
For every prescribed budget choose only a positive Gen partner seed
at most one. The uniform actual Mem amplitude formed at clock one
and the retained Gen envelope then generate a uniquely wrong held-out
prefix at clocks one through budget+1. Gen nevertheless forms a
positive product immediately and stays positive at every successor
clock, and training decisions are strictly correct from clock one.

The prefix comparison starts at the actual full clock-one state,
retaining both buffers and the completed bias-correction clock. The
path shift is exact and does not replace it with a fresh seeded state.
No future weak-Gen margin, clipping stream, convergence, successful
reference or delay premise is supplied. The same original initialized
path later selects the true held-out class permanently.

Exact reals, fixed gained tables and uniform decoupled native decay
differ from appendix C coupled-cost GD and learned stochastic/numerical
GPTMini. The loose positive seeds can fall outside floating-point range.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss

/-- A bounded positive seed meets the clock-one growth-adjusted initial
comparison for every budget and positive formed Mem amplitude. Sources:
section 3 small initial partners and actual native envelopes at fe86232;
only initial data vary, not the task or native optimizer configuration. -/
theorem fixed_gain_memory_bounded_formed_delay_seed (budget : ℕ) (amplitude : ℝ) (ha : 0 < amplitude) :
    ∃ seed : ℝ, 0 < seed ∧ seed ≤ 1 ∧
      3 * (gainGenGrowthFactor (9 / 10) 1 (1 / 1000) 3 ^ budget * ((103 / 50 : ℝ) * seed)) ^ 2 <
        2 * ((1 - (1 / 1000 : ℝ) * 100) ^ budget * amplitude) ^ 2 := by
  obtain ⟨tiny, htiny, hmargin⟩ := exists_positive_gain_delay_seed budget 3 2 (9 / 10) 1 100 (1 / 1000) amplitude
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) ha
  let seed := min 1 ((50 / 103 : ℝ) * tiny)
  have hseed : 0 < seed := lt_min (by norm_num) (mul_pos (by norm_num) htiny)
  have hunit : seed ≤ 1 := min_le_left _ _
  have hsmall : seed ≤ (50 / 103 : ℝ) * tiny := min_le_right _ _
  have hformed : (103 / 50 : ℝ) * seed ≤ tiny := by linarith only [hsmall]
  have hr : 0 ≤ gainGenGrowthFactor (9 / 10) 1 (1 / 1000) 3 := by norm_num [gainGenGrowthFactor]
  have hupper := mul_le_mul_of_nonneg_left hformed (pow_nonneg hr budget)
  have hnn : 0 ≤ gainGenGrowthFactor (9 / 10) 1 (1 / 1000) 3 ^ budget * ((103 / 50 : ℝ) * seed) :=
    mul_nonneg (pow_nonneg hr budget) (mul_nonneg (by norm_num) (le_of_lt hseed))
  have htn : 0 ≤ gainGenGrowthFactor (9 / 10) 1 (1 / 1000) 3 ^ budget * tiny :=
    mul_nonneg (pow_nonneg hr budget) (le_of_lt htiny)
  have hsquare := mul_le_mul hupper hupper hnn htn
  have hscore := mul_le_mul_of_nonneg_left hsquare (show (0 : ℝ) ≤ 3 by norm_num)
  refine ⟨seed, hseed, hunit, ?_⟩
  exact lt_of_le_of_lt (by simpa only [pow_two] using hscore) hmargin

example : (0 : ℝ) < 1 := by norm_num

/-- Both first factors start at zero, yet actual formation is followed
by an arbitrary uniquely wrong-test prefix with successor-clock strict
train correctness and a positive formed Gen product. Sources: section
3 untrained initialization, appendix C products and original retained
formation/prefix laws at fe86232/f31f874; future weak scores are derived. -/
theorem fixed_gain_memory_double_zero_wrong_prefix (budget : ℕ) :
    ∃ seed : ℝ, 0 < seed ∧ seed ≤ 1 ∧
      let initial := seededNativeSubweights ((0, seed), (0, 1))
      (∀ n, 0 < physicalCircuitScore 3 (fixedGainMemoryPath initial (n + 1) 0).parameter
        (fixedGainMemoryPath initial (n + 1) 1).parameter) ∧
      (∀ n, StrictCorrect (fixedGainMemoryTrainLogits initial (n + 1)) 0) ∧
      ∀ n, n ≤ budget → StrictCorrect (fixedGainMemoryHeldoutLogits initial (n + 1)) 1 := by
  obtain ⟨amplitude, ha, hmem⟩ := fixed_gain_memory_uniform_mem_formation
  obtain ⟨seed, hseed, hunit, hmargin⟩ := fixed_gain_memory_bounded_formed_delay_seed budget amplitude ha
  let initial := seededNativeSubweights ((0, seed), (0, 1))
  let formed := fixedGainMemoryPath initial 1
  have hs : NonnegativeNativeState initial := native_seeded_nonnegative 0 seed 0 1
    le_rfl (le_of_lt hseed) le_rfl (by norm_num)
  have hsf : NonnegativeNativeState formed := fixed_gain_memory_signs initial hs 1
  have hmf := hmem seed (le_of_lt hseed) hunit
  have hweight := fixed_gain_memory_seed_gen_formation_ceiling seed (le_of_lt hseed)
  have hmasses := gain_gen_growth_weight_covers_mass (9 / 10) 1 (1 / 1000) formed
    (by norm_num) (by norm_num) (by norm_num) hsf
  have hwpos : 0 ≤ gainGenGrowthWeight (9 / 10) 1 (1 / 1000) formed := le_trans hmasses.1 hmasses.2
  have hr : 0 ≤ gainGenGrowthFactor (9 / 10) 1 (1 / 1000) 3 := by norm_num [gainGenGrowthFactor]
  have hupper := mul_le_mul_of_nonneg_left hweight (pow_nonneg hr budget)
  have hnn : 0 ≤ gainGenGrowthFactor (9 / 10) 1 (1 / 1000) 3 ^ budget *
      gainGenGrowthWeight (9 / 10) 1 (1 / 1000) formed := mul_nonneg (pow_nonneg hr budget) hwpos
  have hsn : 0 ≤ gainGenGrowthFactor (9 / 10) 1 (1 / 1000) 3 ^ budget * ((103 / 50 : ℝ) * seed) :=
    mul_nonneg (pow_nonneg hr budget) (mul_nonneg (by norm_num) (le_of_lt hseed))
  have hsquare := mul_le_mul hupper hupper hnn hsn
  have hactualMargin : 3 * (gainGenGrowthFactor (9 / 10) 1 (1 / 1000) 3 ^ budget *
      gainGenGrowthWeight (9 / 10) 1 (1 / 1000) formed) ^ 2 <
        2 * ((1 - (1 / 1000 : ℝ) * 100) ^ budget * amplitude) ^ 2 := by
    exact lt_of_le_of_lt (by simpa only [pow_two] using
      mul_le_mul_of_nonneg_left hsquare (show (0 : ℝ) ≤ 3 by norm_num)) hmargin
  have hprefix := gain_native_train_correct_wrong_test_prefix 0 budget 3 2 1
    (9 / 10) (49 / 50) 1 100 (1 / 1000) amplitude formed
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) hsf ha hmf.1 hmf.2 hactualMargin
  have hformation : ∀ n, 0 < physicalCircuitScore 3 (fixedGainMemoryPath initial (n + 1) 0).parameter
      (fixedGainMemoryPath initial (n + 1) 1).parameter := by
    intro n
    have hf := gain_native_positive_gen_partner_activates 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) initial n
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) hs hseed
    have hp := gain_native_positive_coordinate_path 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) initial (n + 1) 1
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) hs hseed
    exact mul_pos (by norm_num) (mul_pos hf.1 hp.1)
  refine ⟨seed, hseed, hunit, hformation, ?_, ?_⟩
  · intro n
    have hsnow := fixed_gain_memory_signs initial hs (n + 1)
    have hmemNN : 0 ≤ physicalCircuitScore 2 (fixedGainMemoryPath initial (n + 1) 2).parameter
        (fixedGainMemoryPath initial (n + 1) 3).parameter :=
      mul_nonneg (by norm_num) (mul_nonneg (hsnow 2).1 (hsnow 3).1)
    apply (train_table_strict_correct_iff 0 _ _).2
    linarith only [hformation n, hmemNN]
  · intro n hn
    have hw := (hprefix n hn).2
    have hshift := fixed_gain_memory_path_shift_one initial n
    unfold fixedGainMemoryHeldoutLogits
    rw [hshift]
    exact hw

/-- Source-style double-zero first factors admit every finite wrong-test
budget after actual formation and later permanent true held-out answers.
Sources: section 3 slow formation, appendix C product tables and original
initialized selection at 4c76b7d, with uniform formation at fe86232;
all future properties concern the same full retained native path. -/
theorem fixed_gain_memory_double_zero_delayed_accuracy (budget : ℕ) :
    ∃ seed : ℝ, 0 < seed ∧ seed ≤ 1 ∧
      let initial := seededNativeSubweights ((0, seed), (0, 1))
      (∀ n, 0 < physicalCircuitScore 3 (fixedGainMemoryPath initial (n + 1) 0).parameter
        (fixedGainMemoryPath initial (n + 1) 1).parameter) ∧
      (∀ n, StrictCorrect (fixedGainMemoryTrainLogits initial (n + 1)) 0) ∧
      (∀ n, n ≤ budget → StrictCorrect (fixedGainMemoryHeldoutLogits initial (n + 1)) 1) ∧
      ∃ start : ℕ, budget + 1 < start ∧
        ∀ n, start ≤ n → StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 0 := by
  obtain ⟨seed, hseed, hunit, hformation, htrain, hwrong⟩ := fixed_gain_memory_double_zero_wrong_prefix budget
  let initial := seededNativeSubweights ((0, seed), (0, 1))
  have hs : NonnegativeNativeState initial := native_seeded_nonnegative 0 seed 0 1
    le_rfl (le_of_lt hseed) le_rfl (by norm_num)
  have hg : 0 < fixedGainGenMemoryMass initial := by
    simpa [fixedGainGenMemoryMass, gainNativePairMemoryMass, initial,
      seededNativeSubweights, seededScalarState, nativeFactorPartner] using hseed
  obtain ⟨start, hcorrect⟩ := fixed_gain_memory_eventual_correct initial hs hg
  refine ⟨seed, hseed, hunit, hformation, htrain, hwrong, start + budget + 2, by omega, ?_⟩
  intro n hn
  exact (hcorrect n (by omega)).2

end Transformer.Grokking.CircuitEfficiency
