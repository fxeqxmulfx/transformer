import Transformer.Grokking.CircuitEfficiency.SectionC_GainSmallDecayMixing

/-!
# Recurrent actual product logits from weak-decay internal mass

Sources: Varma et al., arXiv:2309.02390v1, section 3's CE/decay
competition and appendix C's product train logits; original small-decay
uniform mixing at lab commit 6304fbe and recurrent mass at e8766b6.

When each next factor covers a fraction q of its preceding pair sum,
the gained next training score is at least q^2 times the square of
total preceding physical mass. This uses all four actual factors and
Gen/Mem gains 3/2, with no pair symmetry or successful test margin.

The initialized native path generates its uniform positive fraction.
For every positive Gen seed, recurrent internal mass therefore gives
a fixed positive actual training-score level reached arbitrarily late.
Its training score cannot tend to zero. No physical parameter or input
convergence is supplied. The original buffers, variance insertions and
completed-clock bias corrections remain in the path.

At every successor clock the actual training answer is uniquely
correct, even with a zero Gen seed: the positive initial Mem partner
persists and the generated mixing forms its product. Clock zero ties
remain excluded. Train correctness does not establish learned-rule
selection on held-out inputs; both hardcoded circuits fit training.

The recurrent score level is existential and may be extremely small.
It does not give a positive uniform tail gap, stable confidence or
grokking time. Fixed physical tables/plain CE and native uniform decay
differ from appendix C's coupled norm-cost GD and learned stochastic/
floating-point GPTMini. Confidence transfer is the next separate step.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss

/-- Actual total gained training target score of the original native
small-decay path. Sources: appendix C training product logits and
source-style retained path at 6304fbe; this is a numerical observer. -/
noncomputable def smallDecayGainTrainScore (remaining : ℕ) (seed : ℝ) (n : ℕ) : ℝ :=
  gainNativeTotalScore 3 2 (smallDecayGainPath remaining seed n)

/-- Current pair mixing bounds the next true gained training score
by the square of preceding full mass. Sources: appendix C product
composition and physical gains at 6304fbe; no symmetry, optimizer
limit or correct held-out ordering is assumed in this algebra bridge. -/
theorem gain_total_score_pair_mass_floor (coefficient : ℝ) (state next : NativeSubweightState)
    (hc : 0 ≤ coefficient) (hs : ∀ i, 0 ≤ (state i).parameter)
    (hmix : ∀ i, coefficient * ((state i).parameter + (state (nativeFactorPartner i)).parameter) ≤ (next i).parameter) :
    coefficient ^ 2 * gainNativeParameterMass state ^ 2 ≤ gainNativeTotalScore 3 2 next := by
  have h0 := hmix 0
  have h1 := hmix 1
  have h2 := hmix 2
  have h3 := hmix 3
  change coefficient * ((state 0).parameter + (state 1).parameter) ≤ (next 0).parameter at h0
  change coefficient * ((state 1).parameter + (state 0).parameter) ≤ (next 1).parameter at h1
  change coefficient * ((state 2).parameter + (state 3).parameter) ≤ (next 2).parameter at h2
  change coefficient * ((state 3).parameter + (state 2).parameter) ≤ (next 3).parameter at h3
  have h1' : coefficient * ((state 0).parameter + (state 1).parameter) ≤ (next 1).parameter := by linarith only [h1]
  have h3' : coefficient * ((state 2).parameter + (state 3).parameter) ≤ (next 3).parameter := by linarith only [h3]
  have hg := mul_nonneg hc (add_nonneg (hs 0) (hs 1))
  have hm := mul_nonneg hc (add_nonneg (hs 2) (hs 3))
  have hGen := mul_le_mul h0 h1' hg (le_trans hg h0)
  have hMem := mul_le_mul h2 h3' hm (le_trans hm h2)
  have hgap := mul_nonneg (sq_nonneg coefficient)
    (sq_nonneg (((state 0).parameter + (state 1).parameter) - ((state 2).parameter + (state 3).parameter)))
  have hGenSquare := sq_nonneg (coefficient * ((state 0).parameter + (state 1).parameter))
  unfold gainNativeParameterMass gainNativeTotalScore physicalCircuitScore
  nlinarith only [hGen, hMem, hgap, hGenSquare]

example : (0 : ℝ) ≤ 1 / 2 ∧
    (∀ i : Fin 4, 0 ≤ (seededNativeSubweights ((1, 1), (1, 1)) i).parameter) ∧
    (∀ i : Fin 4, (1 / 2 : ℝ) * ((seededNativeSubweights ((1, 1), (1, 1)) i).parameter +
      (seededNativeSubweights ((1, 1), (1, 1)) (nativeFactorPartner i)).parameter) ≤
        (seededNativeSubweights ((1, 1), (1, 1)) i).parameter) := by
  refine ⟨by norm_num, ?_, ?_⟩ <;> intro i <;>
    fin_cases i <;> norm_num [seededNativeSubweights, seededScalarState, nativeFactorPartner]

/-- The original initialized path generates a uniform positive next-
score floor from preceding mass. Sources: section 3's formed factors,
appendix C true logits and actual retained mixing at 6304fbe; no future
coefficient, boundedness, symmetry or score floor is supplied. -/
theorem small_decay_gain_uniform_score_floor (remaining : ℕ) (seed : ℝ) (hseed : 0 ≤ seed) :
    ∃ coefficient : ℝ, 0 < coefficient ∧ ∀ n,
      coefficient ^ 2 * gainNativeParameterMass (smallDecayGainPath remaining seed n) ^ 2 ≤
        smallDecayGainTrainScore remaining seed (n + 1) := by
  obtain ⟨coefficient, hc, _, hmix⟩ := small_decay_gain_uniform_pair_mixing remaining seed hseed
  refine ⟨coefficient, hc, ?_⟩
  intro n
  exact gain_total_score_pair_mass_floor coefficient _ _ (le_of_lt hc)
    (fun i => ((hmix n).1 i).1) (hmix n).2

example : (0 : ℝ) ≤ 1 / 200 := by norm_num

/-- Actual train target logits reach a fixed positive level at
arbitrarily late clocks on the initialized weak-decay path. Sources:
section 3 CE competition, appendix C product logits, recurrent mass
at e8766b6 and actual mixing at 6304fbe; a future limit is not a premise. -/
theorem small_decay_gain_train_score_returns (remaining : ℕ) (seed : ℝ) (hseed : 0 < seed) :
    ∃ level : ℝ, 0 < level ∧ ∀ start : ℕ, ∃ n : ℕ, start ≤ n ∧
      level ≤ smallDecayGainTrainScore remaining seed n := by
  obtain ⟨coefficient, hc, hscore⟩ := small_decay_gain_uniform_score_floor remaining seed (le_of_lt hseed)
  obtain ⟨mass, hmass, hreturn⟩ := gain_native_small_decay_source_mass_returns remaining seed hseed
  refine ⟨coefficient ^ 2 * mass ^ 2, by positivity, ?_⟩
  intro start
  obtain ⟨n, hn, hlevel⟩ := hreturn start
  change mass ≤ gainNativeParameterMass (smallDecayGainPath remaining seed n) at hlevel
  have hd := mul_nonneg
    (show 0 ≤ gainNativeParameterMass (smallDecayGainPath remaining seed n) - mass by linarith only [hlevel])
    (show 0 ≤ gainNativeParameterMass (smallDecayGainPath remaining seed n) + mass by linarith only [hlevel, hmass])
  have hw := mul_nonneg (sq_nonneg coefficient) hd
  refine ⟨n + 1, by omega, ?_⟩
  nlinarith only [hw, hscore n]

example : (0 : ℝ) < 1 / 200 := by norm_num

/-- The true original training score cannot tend to zero at these
small native constants. Sources: section 3 CE/decay competition and
actual original mixing at 6304fbe; this does not assert a positive
uniform tail floor, parameter convergence or true held-out selection. -/
theorem small_decay_gain_train_score_not_collapse (remaining : ℕ) (seed : ℝ) (hseed : 0 < seed) :
    ¬Tendsto (smallDecayGainTrainScore remaining seed) atTop (nhds 0) := by
  intro hzero
  obtain ⟨level, hlevel, hreturn⟩ := small_decay_gain_train_score_returns remaining seed hseed
  obtain ⟨start, htail⟩ := eventually_atTop.mp (hzero.eventually_lt_const hlevel)
  obtain ⟨n, hn, hscore⟩ := hreturn start
  exact (not_le_of_gt (htail n hn)) hscore

example : (0 : ℝ) < 1 / 200 := by norm_num

/-- Actual train answers are uniquely correct at every successor
clock, including a zero Gen seed. Sources: appendix C common train
labels and original mixing at 6304fbe; clock zero still has tied logits,
and correctness does not distinguish Mem from a generalizing rule. -/
theorem small_decay_gain_train_correct_successors (remaining : ℕ) (seed : ℝ) (n : ℕ) (hseed : 0 ≤ seed) :
    StrictCorrect (trainTableLogits remaining
      (physicalCircuitScore 3 (smallDecayGainPath remaining seed (n + 1) 0).parameter
        (smallDecayGainPath remaining seed (n + 1) 1).parameter)
      (physicalCircuitScore 2 (smallDecayGainPath remaining seed (n + 1) 2).parameter
        (smallDecayGainPath remaining seed (n + 1) 3).parameter)) 0 := by
  obtain ⟨coefficient, hc, hscore⟩ := small_decay_gain_uniform_score_floor remaining seed hseed
  have hp := (gain_native_positive_coordinate_path remaining 3 2 1
    (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
    (seededNativeSubweights ((0, seed), (0, 1))) n 3
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (native_seeded_nonnegative 0 seed 0 1 le_rfl hseed le_rfl (by norm_num))
    (by norm_num [seededNativeSubweights, seededScalarState])).1
  have hs := gain_native_nonnegative_path remaining 3 2 1
    (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
    (seededNativeSubweights ((0, seed), (0, 1))) n
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (native_seeded_nonnegative 0 seed 0 1 le_rfl hseed le_rfl (by norm_num))
  have hmass : 0 < gainNativeParameterMass (smallDecayGainPath remaining seed n) :=
    lt_of_lt_of_le hp ((gain_native_masses_nonnegative _ hs).2.2 3)
  have hpositive : 0 < coefficient ^ 2 * gainNativeParameterMass (smallDecayGainPath remaining seed n) ^ 2 := by positivity
  apply (train_table_strict_correct_iff remaining _ _).mpr
  exact lt_of_lt_of_le hpositive (hscore n)

example : (0 : ℝ) ≤ 0 := by norm_num

end Transformer.Grokking.CircuitEfficiency
