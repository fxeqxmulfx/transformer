import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryDelay
import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryLoss

/-!
# Delayed permanent native accuracy with uniform limiting confidence

Sources: Varma et al., arXiv:2309.02390v1, section 3 confidence versus
decisions and appendix C true train/test CE; original initialized
nonzero-beta delayed accuracy at lab commit 917fc73 and complete
retained-beta loss/probability limits at 2c2a85b.

Keep the same binary task, physical gains 3/2 and native constants:
beta1=0.9, beta2=0.98, epsilon/cap 1, decay 100 and rate 0.001.
The actual full softmax probability observer uses both class logits.
Initial full signs generate train and held-out CE limits log 2 and
every held-out class probability limit 1/2, without supplied future
convergence or successful references.

Combine these limits with arbitrary delayed permanent accuracy on the
very same retained path. Only a positive first-zero Gen partner seed
varies with the prescribed budget; Mem initially has factors (1,1).
Train decisions remain strictly correct, held-out decisions are wrong
through the budget and become permanently correct later. Nevertheless
held-out CE does not tend to zero, and no fixed confidence threshold
above 1/2 holds on a tail. Strict decisions and confidence are distinct.

This is an exact-real accuracy counterexample under sufficient strong
decay, not a proof of low training CE or numerical robustness. Fixed
tables and uniform decoupled native decay differ from appendix C's
coupled-cost GD. Learned stochastic/floating-point GPTMini transfer
and the weaker-decay positive-confidence regime remain open.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss
open scoped BigOperators

/-- Actual complete binary held-out softmax probability at a native clock.
Sources: appendix C test-logit sum and the fixed retained readout at
917fc73; both classes and every numerical argument enter the forward. -/
noncomputable def fixedGainMemoryHeldoutProbability
    (initial : NativeSubweightState) (n : ℕ) (k : Fin 2) : ℝ :=
  Real.exp (fixedGainMemoryHeldoutLogits initial n k) /
    ∑ j, Real.exp (fixedGainMemoryHeldoutLogits initial n j)

/-- Initial numerical signs generate actual uniform train/test CE and
held-out confidence limits on the original legal-beta fixed path.
Sources: appendix C complete CE and initialized retained-beta limits
at 2c2a85b; a future limit or true decision is not an independent premise. -/
theorem fixed_gain_memory_ce_confidence_limits (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) :
    Tendsto (fun n => crossEntropy (fixedGainMemoryTrainLogits initial n) 0) atTop (nhds (Real.log 2)) ∧
      Tendsto (fun n => crossEntropy (fixedGainMemoryHeldoutLogits initial n) 0) atTop (nhds (Real.log 2)) ∧
      ∀ k, Tendsto (fun n => fixedGainMemoryHeldoutProbability initial n k) atTop (nhds (1 / 2)) := by
  have hc := gain_native_memory_ce_tendsto_uniform 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) initial
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) hs (by norm_num)
  have hp := gain_native_memory_probabilities_tendsto_uniform 0 3 2 1 (9 / 10) (49 / 50) 1 100 (1 / 1000) initial
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) hs (by norm_num)
  have htrain : ∀ x y : ℝ, trainTableLogits 0 (x + y) 0 = trainTableLogits 0 x y := by
    intro x y
    funext k
    simp only [trainTableLogits, add_zero]
  refine ⟨?_, ?_, ?_⟩
  · simpa only [fixedGainMemoryTrainLogits, fixedGainMemoryPath, tableTrainCE,
      gainNativeTotalScore, htrain, Nat.cast_zero, zero_add] using hc.1
  · simpa only [fixedGainMemoryHeldoutLogits, fixedGainMemoryPath, Nat.cast_zero, zero_add] using hc.2
  · simpa only [fixedGainMemoryHeldoutProbability, fixedGainMemoryHeldoutLogits,
      fixedGainMemoryPath, Nat.cast_zero, zero_add] using hp

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Every budget has an original nonzero-memory delayed-accuracy path
with actual uniform CE/confidence limits at the same native clocks.
Sources: section 3 delayed generalization and appendix C true CE,
extended by initialized formation/selection at 917fc73 and loss at
2c2a85b; no future gradient, reference, success or confidence is supplied. -/
theorem fixed_gain_memory_delayed_accuracy_with_uniform_confidence (budget : ℕ) :
    ∃ seed : ℝ, 0 < seed ∧
      let initial := seededNativeSubweights ((0, seed), (1, 1))
      (∀ n, StrictCorrect (fixedGainMemoryTrainLogits initial n) 0) ∧
      (∀ n, 0 < physicalCircuitScore 3 (fixedGainMemoryPath initial (n + 1) 0).parameter
        (fixedGainMemoryPath initial (n + 1) 1).parameter) ∧
      (∀ n, n ≤ budget → StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 1) ∧
      (∃ start : ℕ, budget < start ∧ ∀ n, start ≤ n → StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 0) ∧
      Tendsto (fun n => crossEntropy (fixedGainMemoryTrainLogits initial n) 0) atTop (nhds (Real.log 2)) ∧
      Tendsto (fun n => crossEntropy (fixedGainMemoryHeldoutLogits initial n) 0) atTop (nhds (Real.log 2)) ∧
      ∀ k, Tendsto (fun n => fixedGainMemoryHeldoutProbability initial n k) atTop (nhds (1 / 2)) := by
  obtain ⟨seed, hseed, htrain, hformation, hwrong, htrue⟩ := fixed_gain_memory_arbitrarily_delayed_accuracy budget
  have hs := native_seeded_nonnegative 0 seed 1 1 le_rfl (le_of_lt hseed) (by norm_num) (by norm_num)
  have hlimits := fixed_gain_memory_ce_confidence_limits _ hs
  exact ⟨seed, hseed, htrain, hformation, hwrong, htrue, hlimits⟩

/-- Arbitrarily delayed permanent strict accuracy does not force zero
held-out CE, even with original retained nonzero beta buffers. Sources:
section 3 decisions/confidence and appendix C true CE, with the fixed
initialized original counterexample at 917fc73 and uniform limits at
2c2a85b. All task and native constants stay fixed as the budget grows. -/
theorem fixed_gain_memory_delayed_accuracy_without_zero_ce (budget : ℕ) :
    ∃ seed : ℝ, 0 < seed ∧
      let initial := seededNativeSubweights ((0, seed), (1, 1))
      (∀ n, StrictCorrect (fixedGainMemoryTrainLogits initial n) 0) ∧
      (∀ n, n ≤ budget → StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 1) ∧
      (∃ start : ℕ, budget < start ∧ ∀ n, start ≤ n → StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 0) ∧
      ¬Tendsto (fun n => crossEntropy (fixedGainMemoryHeldoutLogits initial n) 0) atTop (nhds 0) := by
  obtain ⟨seed, hseed, htrain, _, hwrong, htrue, _, hce, _⟩ :=
    fixed_gain_memory_delayed_accuracy_with_uniform_confidence budget
  refine ⟨seed, hseed, htrain, hwrong, htrue, ?_⟩
  intro hzero
  have hpositive : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have heq := tendsto_nhds_unique hce hzero
  linarith only [hpositive, heq]

/-- No fixed target softmax confidence above one half holds on a tail
of the original initialized fixed path. Sources: section 3 confidence
competition and complete held-out probability limits at 2c2a85b;
this obstruction concerns the actual probability, not an accuracy proxy. -/
theorem fixed_gain_memory_no_positive_confidence_tail (initial : NativeSubweightState)
    (hs : NonnegativeNativeState initial) :
    ¬∃ confidence : ℝ, (1 / 2 : ℝ) < confidence ∧ ∃ start : ℕ, ∀ n, start ≤ n →
      confidence ≤ fixedGainMemoryHeldoutProbability initial n 0 := by
  rintro ⟨confidence, hc, start, htail⟩
  have hp := (fixed_gain_memory_ce_confidence_limits initial hs).2.2 0
  obtain ⟨after, hbelow⟩ := eventually_atTop.mp (hp.eventually_lt_const hc)
  have hlow := hbelow (start + after) (by omega)
  have hhigh := htail (start + after) (by omega)
  linarith only [hlow, hhigh]

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (1, 1))) := by
  exact native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- On the same arbitrarily delayed permanent-correct native path,
every target confidence threshold above one half is eventually missed
at every later clock. Sources: section 3 accuracy/confidence and appendix
C actual probabilities, with initialized delayed selection at 917fc73
and loss limits at 2c2a85b; accuracy and threshold clocks are generated. -/
theorem fixed_gain_memory_delayed_accuracy_below_confidence_thresholds (budget : ℕ) :
    ∃ seed : ℝ, 0 < seed ∧
      let initial := seededNativeSubweights ((0, seed), (1, 1))
      (∀ n, StrictCorrect (fixedGainMemoryTrainLogits initial n) 0) ∧
      (∀ n, n ≤ budget → StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 1) ∧
      ∃ start : ℕ, budget < start ∧
        (∀ n, start ≤ n → StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 0) ∧
        ∀ confidence : ℝ, (1 / 2 : ℝ) < confidence →
          ∃ after : ℕ, start ≤ after ∧ ∀ n, after ≤ n →
            fixedGainMemoryHeldoutProbability initial n 0 < confidence := by
  obtain ⟨seed, hseed, htrain, _, hwrong, ⟨start, hlate, htrue⟩, _, _, hp⟩ :=
    fixed_gain_memory_delayed_accuracy_with_uniform_confidence budget
  refine ⟨seed, hseed, htrain, hwrong, start, hlate, htrue, ?_⟩
  intro confidence hc
  obtain ⟨after, hbelow⟩ := eventually_atTop.mp ((hp 0).eventually_lt_const hc)
  exact ⟨start + after, by omega, fun n hn => hbelow n (by omega)⟩

end Transformer.Grokking.CircuitEfficiency
