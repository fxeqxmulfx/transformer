import Transformer.Grokking.CircuitEfficiency.SectionC_GainColdInstability

/-!
# Recurrent internal mass and squared norm under weak native decay

Sources: Varma et al., arXiv:2309.02390v1, section 3's parameter-norm
competition and appendix C's four physical factors; initialized retained
cold obstruction at lab commit 6a99a48. Translate exclusion of a zero
limit into actual numerical measurements at arbitrarily late clocks.

A nonnegative real sequence that does not tend to zero has a fixed
positive level reached beyond every prescribed clock. Apply this fact
to the complete original native parameter mass, without assuming a
finite positive limit, monotone training, or a successful reference.

The squared physical Euclidean norm is the sum of the four actual
parameter squares. An explicit four-coordinate Cauchy bound relates
the square of total parameter mass to four times this squared norm.
The source-style weak-decay initialized path therefore has a positive
squared-norm level reached at arbitrarily late clocks, and this norm
cannot tend to zero either. Beta1=0.9, beta2=0.98, decay=0.1, rate=0.001,
epsilon=1e-8 and cap=1 are retained; only class count and positive Gen
partner seed vary. Mem begins (0,1), Gen (0,seed), with zero buffers.

The level is existential rather than an estimated transition scale.
Recurrent excursions permit oscillations and do not give a positive
uniform tail floor, positive confidence, or eventual true test answers.
Mass and squared norm remain internal numerical observers. A separate
actual partner-mixing argument must connect them to product logits.
Fixed physical gains/tables and uniform native decay differ from the
source's coupled norm-cost GD and learned stochastic/numerical GPTMini.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- Nonnegative numerical noncollapse forces a fixed positive level
at arbitrarily late clocks. Sources: section 3's norm competition,
with the real order-topology formulation of a zero limit; this is a
general measurement lemma, not a supplied native trajectory. -/
theorem nonnegative_noncollapse_recurrent_level (measurement : ℕ → ℝ)
    (hpositive : ∀ n, 0 ≤ measurement n)
    (hnot : ¬Tendsto measurement atTop (nhds 0)) :
    ∃ level : ℝ, 0 < level ∧ ∀ start : ℕ, ∃ n : ℕ, start ≤ n ∧ level ≤ measurement n := by
  by_contra hnone
  push Not at hnone
  apply hnot
  apply tendsto_order.mpr
  constructor
  · intro lower hlower
    exact Eventually.of_forall (fun n => lt_of_lt_of_le hlower (hpositive n))
  · intro upper hupper
    obtain ⟨start, htail⟩ := hnone upper hupper
    exact (eventually_ge_atTop start).mono (fun n hn => htail n hn)

example : (∀ _ : ℕ, (0 : ℝ) ≤ 1) ∧
    ¬Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 0) := by
  refine ⟨fun _ => by norm_num, ?_⟩
  intro hzero
  have hconstant : Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 1) := tendsto_const_nhds
  have heq := tendsto_nhds_unique hconstant hzero
  norm_num at heq

/-- Squared physical parameter norm of the four actual product factors.
Sources: section 3 parameter size and appendix C coordinates; retained
buffers and clocks are separate fields, not part of this norm. -/
def gainNativeSquaredParameterNorm (state : NativeSubweightState) : ℝ :=
  ((state 0).parameter ^ 2 + (state 1).parameter ^ 2) +
    ((state 2).parameter ^ 2 + (state 3).parameter ^ 2)

/-- The measured squared physical norm is nonnegative at any state.
Sources: section 3 parameter norm and appendix C factor coordinates;
no parameter signs or numerical success conditions are required. -/
theorem gain_native_squared_parameter_norm_nonnegative (state : NativeSubweightState) :
    0 ≤ gainNativeSquaredParameterNorm state := by
  unfold gainNativeSquaredParameterNorm
  exact add_nonneg (add_nonneg (sq_nonneg _) (sq_nonneg _))
    (add_nonneg (sq_nonneg _) (sq_nonneg _))

/-- Four-coordinate mass obeys the explicit squared Euclidean bound.
Sources: section 3's parameter norm and appendix C's four factors;
the elementary Cauchy bound needs neither signs nor future limits. -/
theorem gain_native_parameter_mass_sq_le_norm (state : NativeSubweightState) :
    gainNativeParameterMass state ^ 2 ≤ 4 * gainNativeSquaredParameterNorm state := by
  have hg := sq_nonneg ((state 0).parameter - (state 1).parameter)
  have hm := sq_nonneg ((state 2).parameter - (state 3).parameter)
  have hp := sq_nonneg (((state 0).parameter + (state 1).parameter) -
    ((state 2).parameter + (state 3).parameter))
  unfold gainNativeParameterMass gainNativeSquaredParameterNorm
  nlinarith only [hg, hm, hp]

/-- The initialized original weak-decay path has a fixed positive
total-mass level at arbitrarily late clocks. Sources: section 3's
positive slow seed and appendix C feedback; actual static noncollapse
at 6a99a48, no future convergence or mass level is independently supplied. -/
theorem gain_native_small_decay_source_mass_returns (remaining : ℕ) (seed : ℝ) (hseed : 0 < seed) :
    ∃ level : ℝ, 0 < level ∧ ∀ start : ℕ, ∃ n : ℕ, start ≤ n ∧
      level ≤ gainNativeParameterMass (gainNativePath remaining 3 2 1
        (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
        (seededNativeSubweights ((0, seed), (0, 1))) n) := by
  apply nonnegative_noncollapse_recurrent_level
  · intro n
    have hs := gain_native_nonnegative_path remaining 3 2 1
      (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
      (seededNativeSubweights ((0, seed), (0, 1))) n
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (native_seeded_nonnegative 0 seed 0 1 le_rfl (le_of_lt hseed) le_rfl (by norm_num))
    exact (gain_native_masses_nonnegative _ hs).1
  · exact gain_native_small_decay_source_not_mass_collapse remaining seed hseed

example : (0 : ℝ) < 1 / 200 := by norm_num

/-- A positive squared physical-norm level is reached arbitrarily
late on that same original path. Sources: section 3's norm competition,
appendix C factors and actual noncollapse at 6a99a48; recurrence does
not assert a positive tail lower bound or a finite limiting norm. -/
theorem gain_native_small_decay_source_squared_norm_returns (remaining : ℕ) (seed : ℝ) (hseed : 0 < seed) :
    ∃ level : ℝ, 0 < level ∧ ∀ start : ℕ, ∃ n : ℕ, start ≤ n ∧
      level ≤ gainNativeSquaredParameterNorm (gainNativePath remaining 3 2 1
        (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
        (seededNativeSubweights ((0, seed), (0, 1))) n) := by
  obtain ⟨level, hlevel, hreturn⟩ := gain_native_small_decay_source_mass_returns remaining seed hseed
  refine ⟨level ^ 2 / 4, by positivity, ?_⟩
  intro start
  obtain ⟨n, hn, hmass⟩ := hreturn start
  refine ⟨n, hn, ?_⟩
  let state := gainNativePath remaining 3 2 1 (9 / 10) (49 / 50)
    (1 / 100000000) (1 / 10) (1 / 1000) (seededNativeSubweights ((0, seed), (0, 1))) n
  have hdiff := mul_nonneg
    (show 0 ≤ gainNativeParameterMass state - level by linarith only [hmass])
    (show 0 ≤ gainNativeParameterMass state + level by linarith only [hmass, hlevel])
  have hnorm := gain_native_parameter_mass_sq_le_norm state
  change level ^ 2 / 4 ≤ gainNativeSquaredParameterNorm state
  nlinarith only [hdiff, hnorm]

example : (0 : ℝ) < 1 / 200 := by norm_num

/-- The actual squared physical norm cannot tend to zero under these
small-decay initial data. Sources: section 3 parameter-norm competition
and actual noncollapse at 6a99a48; neither convergence elsewhere nor
confidence or a correct test margin is assumed or inferred. -/
theorem gain_native_small_decay_source_not_squared_norm_collapse (remaining : ℕ) (seed : ℝ) (hseed : 0 < seed) :
    ¬Tendsto (fun n => gainNativeSquaredParameterNorm (gainNativePath remaining 3 2 1
      (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
      (seededNativeSubweights ((0, seed), (0, 1))) n)) atTop (nhds 0) := by
  intro hzero
  obtain ⟨level, hlevel, hreturn⟩ := gain_native_small_decay_source_squared_norm_returns remaining seed hseed
  obtain ⟨start, htail⟩ := eventually_atTop.mp (hzero.eventually_lt_const hlevel)
  obtain ⟨n, hn, hnorm⟩ := hreturn start
  exact (not_le_of_gt (htail n hn)) hnorm

example : (0 : ℝ) < 1 / 200 := by norm_num

end Transformer.Grokking.CircuitEfficiency
