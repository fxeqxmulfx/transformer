import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemorySelection
import Transformer.Grokking.CircuitEfficiency.SectionC_GainDelaySeeds

/-!
# Arbitrarily delayed permanent accuracy with original nonzero-beta memory

Sources: Varma et al., arXiv:2309.02390v1, section 3 slow versus fast
learning and appendix C product logits; native arbitrary wrong prefixes,
formation and initialized permanent selection at lab commit 4c76b7d.

Keep one binary task and one original native configuration: gains 3/2,
beta1=0.9, beta2=0.98, epsilon/cap 1, decay 100 and rate 0.001.
For every prescribed finite budget, choose only a positive initial Gen
partner seed; its first factor is zero and Mem starts at (1,1).
The actual clipped CE forms a positive Gen product at step one, yet
Mem uniquely gives wrong held-out answers through the entire budget.
Training decisions are strictly correct at every finite clock.

The same original path eventually selects the correct held-out class
permanently. Its generated permanent start and its least true held-out
clock both exceed the prescribed budget. No future parameter limit,
coefficient interval, margin, successful reference or gradient stream
is supplied. Both retained histories and full correction clocks are kept.

This is arbitrary delay in exact-real accuracy with native nonzero-beta
memory. It does not assert positive limiting confidence, a stable floating
point decision, learned heads or a thermodynamic/system-size limit.
Static strong decay makes the selected margin vanish. Fixed gained tables
and uniform decoupled native decay differ from appendix C coupled-cost
GD. Learned stochastic/numerical GPTMini transfer remains open.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss

/-- Original fixed binary train logits at a current retained clock.
Sources: appendix C gained product readout and native path at 4c76b7d;
this numerical forward contains no decision or success predicate. -/
noncomputable def fixedGainMemoryTrainLogits (initial : NativeSubweightState) (n : ℕ) : Fin 2 → ℝ :=
  trainTableLogits 0
    (physicalCircuitScore 3 (fixedGainMemoryPath initial n 0).parameter (fixedGainMemoryPath initial n 1).parameter)
    (physicalCircuitScore 2 (fixedGainMemoryPath initial n 2).parameter (fixedGainMemoryPath initial n 3).parameter)

/-- Original fixed binary held-out logits at a current retained clock.
Sources: appendix C gained Gen/Mem readout and native path at 4c76b7d;
all arguments enter the actual physical forward. -/
noncomputable def fixedGainMemoryHeldoutLogits (initial : NativeSubweightState) (n : ℕ) : Fin 2 → ℝ :=
  heldoutTableLogits 0
    (physicalCircuitScore 3 (fixedGainMemoryPath initial n 0).parameter (fixedGainMemoryPath initial n 1).parameter)
    (physicalCircuitScore 2 (fixedGainMemoryPath initial n 2).parameter (fixedGainMemoryPath initial n 3).parameter)

/-- Every budget admits a single initialized original native path
with all-clock strict train correctness, positive Gen formation,
uniquely wrong test prefix and later permanent true test correctness.
Sources: section 3 slow formation, appendix C actual logits and native
formation/selection at 4c76b7d; only the initial Gen partner varies. -/
theorem fixed_gain_memory_arbitrarily_delayed_accuracy (budget : ℕ) :
    ∃ seed : ℝ, 0 < seed ∧
      let initial := seededNativeSubweights ((0, seed), (1, 1))
      (∀ n, StrictCorrect (fixedGainMemoryTrainLogits initial n) 0) ∧
      (∀ n, 0 < physicalCircuitScore 3 (fixedGainMemoryPath initial (n + 1) 0).parameter
        (fixedGainMemoryPath initial (n + 1) 1).parameter) ∧
      (∀ n, n ≤ budget → StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 1) ∧
      ∃ start : ℕ, budget < start ∧ ∀ n, start ≤ n → StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 0 := by
  obtain ⟨seed, hseed, hformation, hprefix⟩ := gain_native_arbitrary_wrong_test_prefix 0 3 2 1
    (9 / 10) (49 / 50) 1 100 (1 / 1000) 1 (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) budget
  let initial := seededNativeSubweights ((0, seed), (1, 1))
  have hs : NonnegativeNativeState initial := native_seeded_nonnegative _ _ _ _
    (by norm_num) (le_of_lt hseed) (by norm_num) (by norm_num)
  have hg : 0 < fixedGainGenMemoryMass initial := by
    simpa [fixedGainGenMemoryMass, gainNativePairMemoryMass, initial,
      seededNativeSubweights, seededScalarState, nativeFactorPartner] using hseed
  obtain ⟨start, hcorrect⟩ := fixed_gain_memory_eventual_correct initial hs hg
  have hwrong : ∀ n, n ≤ budget → StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 1 :=
    fun n hn => (hprefix n hn).2
  have htrue : ∀ n, start ≤ n → StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 0 :=
    fun n hn => (hcorrect n hn).2
  have hlate : budget < start := by
    by_contra h
    have hn : start ≤ budget := by omega
    have h0 := htrue start le_rfl 1 (by decide)
    have h1 := hwrong start hn 0 (by decide)
    linarith only [h0, h1]
  refine ⟨seed, hseed, ?_⟩
  dsimp only
  refine ⟨?_, hformation, hwrong, start, hlate, htrue⟩
  intro n
  unfold fixedGainMemoryTrainLogits
  apply (train_table_strict_correct_iff 0 _ _).2
  cases n with
  | zero =>
    change 0 < physicalCircuitScore 3 0 seed + physicalCircuitScore 2 1 1
    norm_num [physicalCircuitScore]
  | succ n =>
    have hgen : 0 < physicalCircuitScore 3 (fixedGainMemoryPath initial (n + 1) 0).parameter
        (fixedGainMemoryPath initial (n + 1) 1).parameter := hformation n
    have hsnow := fixed_gain_memory_signs initial hs (n + 1)
    have hmem : 0 ≤ physicalCircuitScore 2 (fixedGainMemoryPath initial (n + 1) 2).parameter
        (fixedGainMemoryPath initial (n + 1) 3).parameter := mul_nonneg (by norm_num) (mul_nonneg (hsnow 2).1 (hsnow 3).1)
    linarith only [hgen, hmem]

/-- The same initialized delayed native path has a least true
held-out clock beyond the budget, with no earlier strict correct
answer. Sources: section 3 delayed generalization and native complete
accuracy at 4c76b7d; least success is derived, not equated with the
later permanent-correctness start or supplied as a future premise. -/
theorem fixed_gain_memory_delayed_first_success (budget : ℕ) :
    ∃ seed : ℝ, 0 < seed ∧
      let initial := seededNativeSubweights ((0, seed), (1, 1))
      (∀ n, StrictCorrect (fixedGainMemoryTrainLogits initial n) 0) ∧
      (∀ n, 0 < physicalCircuitScore 3 (fixedGainMemoryPath initial (n + 1) 0).parameter
        (fixedGainMemoryPath initial (n + 1) 1).parameter) ∧
      (∀ n, n ≤ budget → StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 1) ∧
      (∃ start : ℕ, budget < start ∧ ∀ n, start ≤ n → StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 0) ∧
      ∃ first : ℕ, budget < first ∧ StrictCorrect (fixedGainMemoryHeldoutLogits initial first) 0 ∧
        (∀ n, n < first → ¬StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 0) ∧
        ∀ n, StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 0 → first ≤ n := by
  classical
  obtain ⟨seed, hseed, htrain, hformation, hwrong, ⟨start, hlate, htrue⟩⟩ :=
    fixed_gain_memory_arbitrarily_delayed_accuracy budget
  let initial := seededNativeSubweights ((0, seed), (1, 1))
  have hex : ∃ n, StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 0 := ⟨start, htrue start le_rfl⟩
  let first := Nat.find hex
  have hfirst : StrictCorrect (fixedGainMemoryHeldoutLogits initial first) 0 := Nat.find_spec hex
  have hfLate : budget < first := by
    by_contra h
    have hn : first ≤ budget := by omega
    have h0 := hfirst 1 (by decide)
    have h1 := hwrong first hn 0 (by decide)
    linarith only [h0, h1]
  refine ⟨seed, hseed, ?_⟩
  dsimp only
  exact ⟨htrain, hformation, hwrong, ⟨start, hlate, htrue⟩, first, hfLate, hfirst,
    fun n hn => Nat.find_min hex hn, fun n hsuccess => Nat.find_min' hex hsuccess⟩

/-- Positive initialized Gen partners admit no uniform finite
correct held-out start on this fixed native task/configuration. Sources:
section 3 slow formation and native arbitrary delay at 4c76b7d; this
is a time/initial-seed statement, not a thermodynamic size limit. -/
theorem fixed_gain_memory_no_uniform_success_start :
    ¬∃ start : ℕ, ∀ seed : ℝ, 0 < seed → ∀ n, start ≤ n →
      StrictCorrect (fixedGainMemoryHeldoutLogits (seededNativeSubweights ((0, seed), (1, 1))) n) 0 := by
  rintro ⟨start, hstart⟩
  obtain ⟨seed, hseed, _, _, hwrong, _⟩ := fixed_gain_memory_arbitrarily_delayed_accuracy start
  have htrue := hstart seed hseed start le_rfl
  have hw := hwrong start le_rfl
  have h0 := htrue 1 (by decide)
  have h1 := hw 0 (by decide)
  linarith only [h0, h1]

end Transformer.Grokking.CircuitEfficiency
