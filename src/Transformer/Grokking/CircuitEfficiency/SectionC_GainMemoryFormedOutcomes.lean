import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryFormedPrefix
import Transformer.Grokking.CircuitEfficiency.SectionC_GainMemoryConfidence

/-!
# Least success and confidence limits after original double-zero formation

Sources: Varma et al., arXiv:2309.02390v1, section 3 delayed
learning and appendix C initial product logits/true CE; original
nonzero-beta double-zero delayed accuracy at lab commit c59d32c,
full native CE/confidence limits at 5e4a49e.

Gen=(0,seed) and Mem=(0,1) initially have zero products. One fixed
binary task and legal retained native configuration admit arbitrary
wrong-test prefixes after actual formation, followed by permanent
strict true decisions. Derive the least strictly correct held-out
clock beyond budget+1, without confusing it with the later permanent
start. At clock zero all actual logits tie, so it is not a true success.

On the very same initialized path, true train/held-out CE approach
log 2 and both actual held-out softmax class probabilities approach
1/2; held-out CE does not tend to zero. Every fixed confidence
threshold above one half is eventually
missed at every later clock, even while answers are permanently right.
Bounded positive Gen partners have no uniform finite success start.

The physical gains are 3/2, clipping cap and epsilon 1, decay 100
and rate 0.001. Both classes and all native constants remain fixed
as the delay budget grows. This is a time/initial-seed family; no
thermodynamic size variable or finite-size scaling claim is supplied.

No future parameter/input convergence, successful reference, weak
margin or confidence limit is independently supplied. The result
concerns exact-real decisions under strong native decay, not low
training CE or positive confidence robustness. The task has fixed
physical gained tables, not learned heads. Uniform decoupled AdamW
with beta1=0.9/beta2=0.98 differs from appendix C coupled-cost GD;
weaker-decay attraction and stochastic/numerical GPTMini transfer remain open.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss

/-- After double-zero original native formation, the least strictly
true held-out clock exceeds the prescribed post-formation budget.
Sources: section 3 delayed learning and original actual decisions at
c59d32c; least success is derived and is distinct from permanent onset. -/
theorem fixed_gain_memory_double_zero_delayed_first_success (budget : ℕ) :
    ∃ seed : ℝ, 0 < seed ∧ seed ≤ 1 ∧
      let initial := seededNativeSubweights ((0, seed), (0, 1))
      (∀ n, StrictCorrect (fixedGainMemoryTrainLogits initial (n + 1)) 0) ∧
      (∀ n, n ≤ budget → StrictCorrect (fixedGainMemoryHeldoutLogits initial (n + 1)) 1) ∧
      (∃ start : ℕ, budget + 1 < start ∧ ∀ n, start ≤ n → StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 0) ∧
      ∃ first : ℕ, budget + 1 < first ∧ StrictCorrect (fixedGainMemoryHeldoutLogits initial first) 0 ∧
        (∀ n, n < first → ¬StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 0) ∧
        ∀ n, StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 0 → first ≤ n := by
  classical
  obtain ⟨seed, hseed, hunit, _, htrain, hwrong, ⟨start, hlate, htrue⟩⟩ :=
    fixed_gain_memory_double_zero_delayed_accuracy budget
  let initial := seededNativeSubweights ((0, seed), (0, 1))
  have hex : ∃ n, StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 0 := ⟨start, htrue start le_rfl⟩
  let first := Nat.find hex
  have hfirst : StrictCorrect (fixedGainMemoryHeldoutLogits initial first) 0 := Nat.find_spec hex
  have hlateFirst : budget + 1 < first := by
    by_contra h
    have hn : first ≤ budget + 1 := by omega
    by_cases hz : first = 0
    · have h0 := hfirst 1 (by decide)
      norm_num [hz, fixedGainMemoryHeldoutLogits, fixedGainMemoryPath, gainNativePath,
        initial, seededNativeSubweights, seededScalarState, physicalCircuitScore, heldoutTableLogits] at h0
    · have hclock : (first - 1) + 1 = first := by omega
      have h1 := hwrong (first - 1) (by omega) 0 (by decide)
      rw [hclock] at h1
      have h0 := hfirst 1 (by decide)
      linarith only [h0, h1]
  refine ⟨seed, hseed, hunit, htrain, hwrong, ⟨start, hlate, htrue⟩, first, hlateFirst, hfirst, ?_, ?_⟩
  · exact fun n hn => Nat.find_min hex hn
  · exact fun n hc => Nat.find_min' hex hc

/-- The same original double-zero delayed-accuracy path has uniform
true CE and complete softmax confidence limits. Sources: appendix C
actual train/test loss and native initialized formation at c59d32c,
combined with retained-beta full confidence laws at 5e4a49e. -/
theorem fixed_gain_memory_double_zero_delay_with_uniform_confidence (budget : ℕ) :
    ∃ seed : ℝ, 0 < seed ∧ seed ≤ 1 ∧
      let initial := seededNativeSubweights ((0, seed), (0, 1))
      (∀ n, StrictCorrect (fixedGainMemoryTrainLogits initial (n + 1)) 0) ∧
      (∀ n, n ≤ budget → StrictCorrect (fixedGainMemoryHeldoutLogits initial (n + 1)) 1) ∧
      (∃ start : ℕ, budget + 1 < start ∧ ∀ n, start ≤ n → StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 0) ∧
      Tendsto (fun n => crossEntropy (fixedGainMemoryTrainLogits initial n) 0) atTop (nhds (Real.log 2)) ∧
      Tendsto (fun n => crossEntropy (fixedGainMemoryHeldoutLogits initial n) 0) atTop (nhds (Real.log 2)) ∧
      ∀ k, Tendsto (fun n => fixedGainMemoryHeldoutProbability initial n k) atTop (nhds (1 / 2)) := by
  obtain ⟨seed, hseed, hunit, _, htrain, hwrong, htrue⟩ := fixed_gain_memory_double_zero_delayed_accuracy budget
  have hs := native_seeded_nonnegative 0 seed 0 1 le_rfl (le_of_lt hseed) le_rfl (by norm_num)
  have hlimits := fixed_gain_memory_ce_confidence_limits _ hs
  exact ⟨seed, hseed, hunit, htrain, hwrong, htrue, hlimits⟩

/-- Bounded positive initialized Gen partners have no uniform finite
true held-out start after original double-zero formation. Sources:
section 3 slow partners and native actual delayed accuracy at c59d32c;
this is a fixed task/time/initial-data claim, not a system-size limit. -/
theorem fixed_gain_memory_double_zero_no_uniform_success_start :
    ¬∃ start : ℕ, ∀ seed : ℝ, 0 < seed → seed ≤ 1 → ∀ n, start ≤ n →
      StrictCorrect (fixedGainMemoryHeldoutLogits (seededNativeSubweights ((0, seed), (0, 1))) n) 0 := by
  rintro ⟨start, hstart⟩
  obtain ⟨seed, hseed, hunit, _, _, hwrong, _⟩ := fixed_gain_memory_double_zero_delayed_accuracy start
  have ht := hstart seed hseed hunit (start + 1) (by omega) 1 (by decide)
  have hw := hwrong start le_rfl 0 (by decide)
  linarith only [ht, hw]

/-- Every fixed above-uniform confidence threshold is eventually missed
on the same double-zero delayed permanently correct native path.
Sources: section 3 decisions/confidence and actual complete probabilities
at 5e4a49e, with generated original formation and selection at c59d32c. -/
theorem fixed_gain_memory_double_zero_delay_below_confidence_thresholds (budget : ℕ) :
    ∃ seed : ℝ, 0 < seed ∧ seed ≤ 1 ∧
      let initial := seededNativeSubweights ((0, seed), (0, 1))
      (∀ n, StrictCorrect (fixedGainMemoryTrainLogits initial (n + 1)) 0) ∧
      (∀ n, n ≤ budget → StrictCorrect (fixedGainMemoryHeldoutLogits initial (n + 1)) 1) ∧
      ∃ start : ℕ, budget + 1 < start ∧
        (∀ n, start ≤ n → StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 0) ∧
        ∀ confidence : ℝ, (1 / 2 : ℝ) < confidence → ∃ after : ℕ, start ≤ after ∧
          ∀ n, after ≤ n → fixedGainMemoryHeldoutProbability initial n 0 < confidence := by
  obtain ⟨seed, hseed, hunit, htrain, hwrong, ⟨start, hlate, htrue⟩, _, _, hp⟩ :=
    fixed_gain_memory_double_zero_delay_with_uniform_confidence budget
  refine ⟨seed, hseed, hunit, htrain, hwrong, start, hlate, htrue, ?_⟩
  intro confidence hc
  obtain ⟨after, hbelow⟩ := eventually_atTop.mp ((hp 0).eventually_lt_const hc)
  exact ⟨start + after, by omega, fun n hn => hbelow n (by omega)⟩

/-- Source-style original native double-zero delayed permanent accuracy
does not force vanishing held-out CE. Sources: section 3 distinction
between decisions and confidence, appendix C true loss and the native
initialized delay at c59d32c with full confidence limits at 5e4a49e;
all native/task constants stay fixed as the prescribed budget grows. -/
theorem fixed_gain_memory_double_zero_delay_without_zero_ce (budget : ℕ) :
    ∃ seed : ℝ, 0 < seed ∧ seed ≤ 1 ∧
      let initial := seededNativeSubweights ((0, seed), (0, 1))
      (∀ n, StrictCorrect (fixedGainMemoryTrainLogits initial (n + 1)) 0) ∧
      (∀ n, n ≤ budget → StrictCorrect (fixedGainMemoryHeldoutLogits initial (n + 1)) 1) ∧
      (∃ start : ℕ, budget + 1 < start ∧ ∀ n, start ≤ n → StrictCorrect (fixedGainMemoryHeldoutLogits initial n) 0) ∧
      ¬Tendsto (fun n => crossEntropy (fixedGainMemoryHeldoutLogits initial n) 0) atTop (nhds 0) := by
  obtain ⟨seed, hseed, hunit, htrain, hwrong, htrue, _, hce, _⟩ :=
    fixed_gain_memory_double_zero_delay_with_uniform_confidence budget
  refine ⟨seed, hseed, hunit, htrain, hwrong, htrue, ?_⟩
  intro hzero
  have hpositive : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have heq := tendsto_nhds_unique hce hzero
  linarith only [hpositive, heq]

end Transformer.Grokking.CircuitEfficiency
