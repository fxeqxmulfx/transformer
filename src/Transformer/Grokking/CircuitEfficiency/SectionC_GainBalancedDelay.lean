import Transformer.Grokking.CircuitEfficiency.SectionC_GainBalancedSelection
import Transformer.Grokking.CircuitEfficiency.SectionC_GainDelaySeeds

/-!
# Arbitrarily delayed permanent accuracy generalization on an actual native path

Sources: Varma et al., arXiv:2309.02390v1, section 3's efficiency
and slow-learning ingredients and appendix C's product train/test
tables; native AdamW/clipping at lab commit dc9177d.

Keep the task, gains, clipping cap, epsilon, decay, rate and Mem seed
fixed. For every finite budget choose a positive balanced Gen seed.
The resulting single actual path has bounded physical parameters and
both retained buffers, fits training decisions at every finite clock,
uniquely selects the wrong held-out Mem class throughout the budget,
and permanently selects the true held-out class after a finite start.
The later successful start is derived, not supplied. No parameter
convergence, future callback or successful reference is assumed.

This exact-real accuracy formulation specializes to equal positive
pair factors and legal zero betas. Mem is already train-fitted at
initialization; both lookup tables are fixed, rather than learned
transformer computations. It differs from the source's initially
zero first factors, coupled assigned norm cost and GD. The chosen
seed may fall below floating-point range. The later tail start is
not asserted to be the first correct clock or a uniform time bound;
CE-loss convergence and transfer to preserved GPTMini remain open.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss

/-- A fixed task and fixed native constants admit any finite duration
of train-correct/test-wrong decisions followed by permanently correct
held-out decisions on the very same bounded path. Sources: section 3,
appendix C's actual tables and dc9177d. Only the positive balanced
Gen seed and its derived parameter ceiling/tail start vary with budget. -/
theorem gain_native_balanced_arbitrary_delayed_generalization (remaining : ℕ)
    (genGain memGain bound eps decay rate amplitude : ℝ)
    (hmem : 0 < memGain) (hbetter : memGain < genGain) (hclip : 0 < bound)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate)
    (hd : 0 < 1 - rate * decay) (ha : 0 < amplitude) :
    ∀ budget : ℕ, ∃ seed ceiling : ℝ, 0 < seed ∧ 0 < ceiling ∧
      let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
        (seededNativeSubweights ((seed, seed), (amplitude, amplitude)))
      (∀ n, NonnegativeNativeState (path n) ∧ ∀ i,
        (path n i).parameter ≤ ceiling ∧ -(path n i).moment ≤ bound ∧ (path n i).variance ≤ bound ^ 2) ∧
      (∀ n, StrictCorrect (trainTableLogits remaining
        (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0) ∧
      (∀ n, n ≤ budget → StrictCorrect (heldoutTableLogits remaining
        (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter))
          (0 : Fin (remaining + 1)).succ) ∧
      ∃ start : ℕ, budget < start ∧ ∀ n, start ≤ n →
        StrictCorrect (heldoutTableLogits remaining
          (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
          (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0 := by
  intro budget
  have hgen := lt_trans hmem hbetter
  obtain ⟨massSeed, hmassSeed, hmargin⟩ := exists_positive_gain_delay_seed budget genGain memGain 0 eps decay rate amplitude
    hgen hmem (by norm_num) he (le_of_lt heta) hd ha
  let seed := massSeed / 2
  have hseed : 0 < seed := div_pos hmassSeed (by norm_num)
  let initial := seededNativeSubweights ((seed, seed), (amplitude, amplitude))
  have hs : NonnegativeNativeState initial :=
    native_seeded_nonnegative _ _ _ _ (le_of_lt hseed) (le_of_lt hseed) (le_of_lt ha) (le_of_lt ha)
  have hweight : gainGenGrowthWeight 0 eps rate initial = massSeed := by
    change (massSeed / 2 + massSeed / 2) + rate / ((1 - 0) * eps) * -(0 + 0) = massSeed
    ring
  obtain ⟨ceiling, hceiling, hbounded⟩ := gain_native_seeded_bounded remaining genGain memGain bound 0 0 eps decay rate
    seed seed amplitude amplitude hgen hmem hclip (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    he hdecay (le_of_lt heta) (le_of_lt hd) (le_of_lt hseed) (le_of_lt hseed) (le_of_lt ha) (le_of_lt ha)
  have hprefix := gain_native_train_correct_wrong_test_prefix remaining budget genGain memGain bound 0 0 eps decay rate
    amplitude initial hgen hmem hclip (by norm_num) (by norm_num) (by norm_num) (by norm_num) he
    (le_of_lt heta) (le_of_lt hdecay) hd hs ha le_rfl le_rfl (by simpa only [hweight] using hmargin)
  obtain ⟨start, htail⟩ := gain_native_balanced_eventual_correct remaining genGain memGain bound eps decay rate seed amplitude
    hmem hbetter hclip he hdecay heta hd hseed ha
  refine ⟨seed, ceiling, hseed, hceiling, hbounded, ?_, ?_, start + budget + 1, by omega, ?_⟩
  · intro n
    exact gain_native_balanced_train_correct remaining genGain memGain bound eps decay rate seed amplitude n
      hgen hmem hclip he (le_of_lt heta) hd hseed ha
  · intro n hn
    exact (hprefix n hn).2
  · intro n hn
    exact (htail n (by omega)).2

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧
    0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧ (0 : ℝ) < 1 := by
  norm_num

/-- One concrete native configuration on each fixed source-style
table has arbitrarily delayed permanent accuracy generalization.
Sources: section 3's small seeds and actual native model at dc9177d;
the fixed zero betas/epsilon-one toy constants differ from GPTMini,
and the initial positive balanced seed may be smaller than machine range. -/
theorem fixed_native_balanced_arbitrary_delayed_generalization (remaining : ℕ) :
    ∀ budget : ℕ, ∃ seed ceiling : ℝ, 0 < seed ∧ 0 < ceiling ∧
      let path := gainNativePath remaining 3 2 1 0 0 1 (1 / 10) (1 / 1000)
        (seededNativeSubweights ((seed, seed), (1, 1)))
      (∀ n, NonnegativeNativeState (path n) ∧ ∀ i,
        (path n i).parameter ≤ ceiling ∧ -(path n i).moment ≤ 1 ∧ (path n i).variance ≤ 1 ^ 2) ∧
      (∀ n, StrictCorrect (trainTableLogits remaining
        (physicalCircuitScore 3 (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore 2 (path n 2).parameter (path n 3).parameter)) 0) ∧
      (∀ n, n ≤ budget → StrictCorrect (heldoutTableLogits remaining
        (physicalCircuitScore 3 (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore 2 (path n 2).parameter (path n 3).parameter))
          (0 : Fin (remaining + 1)).succ) ∧
      ∃ start : ℕ, budget < start ∧ ∀ n, start ≤ n →
        StrictCorrect (heldoutTableLogits remaining
          (physicalCircuitScore 3 (path n 0).parameter (path n 1).parameter)
          (physicalCircuitScore 2 (path n 2).parameter (path n 3).parameter)) 0 := by
  exact gain_native_balanced_arbitrary_delayed_generalization remaining 3 2 1 1 (1 / 10) (1 / 1000) 1
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- The first actual strictly correct held-out clock is finite and
exceeds any prescribed budget for a generated positive balanced seed.
Sources: section 3's delayed accuracy description and appendix C's
actual tables at dc9177d. Minimality concerns every native update,
not only checkpoints; it does not rule out ties or claim that this
first correct update is already the permanent-success tail start. -/
theorem gain_native_balanced_first_correct_after_budget (remaining : ℕ)
    (genGain memGain bound eps decay rate amplitude : ℝ)
    (hmem : 0 < memGain) (hbetter : memGain < genGain) (hclip : 0 < bound)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate)
    (hd : 0 < 1 - rate * decay) (ha : 0 < amplitude) (budget : ℕ) :
    ∃ seed : ℝ, 0 < seed ∧
      let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
        (seededNativeSubweights ((seed, seed), (amplitude, amplitude)))
      let correct := fun n => StrictCorrect (heldoutTableLogits remaining
        (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0
      ∃ first : ℕ, budget < first ∧ correct first ∧ ∀ n, n < first → ¬correct n := by
  classical
  obtain ⟨seed, _, hseed, _, _, _, hprefix, start, _, htail⟩ :=
    gain_native_balanced_arbitrary_delayed_generalization remaining genGain memGain bound eps decay rate amplitude
      hmem hbetter hclip he hdecay heta hd ha budget
  let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
    (seededNativeSubweights ((seed, seed), (amplitude, amplitude)))
  let correct := fun n => StrictCorrect (heldoutTableLogits remaining
    (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
    (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0
  have hex : ∃ n, correct n := ⟨start, htail start le_rfl⟩
  have hfirst : correct (Nat.find hex) := Nat.find_spec hex
  have hlate : budget < Nat.find hex := by
    by_contra hn
    have hw := hprefix (Nat.find hex) (by omega)
    have hwrong := hw 0 (fun heq => Fin.succ_ne_zero (0 : Fin (remaining + 1)) heq.symm)
    have hright := hfirst (0 : Fin (remaining + 1)).succ (Fin.succ_ne_zero _)
    linarith only [hwrong, hright]
  exact ⟨seed, hseed, Nat.find hex, hlate, hfirst, fun n hn => Nat.find_min hex hn⟩

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧
    0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧ (0 : ℝ) < 1 := by
  norm_num

end Transformer.Grokking.CircuitEfficiency
