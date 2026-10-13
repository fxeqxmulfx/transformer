import Transformer.Grokking.CircuitEfficiency.SectionC_GainUnequalSelection
import Transformer.Grokking.CircuitEfficiency.SectionC_GainDelaySeeds

/-!
# Arbitrarily delayed permanent native accuracy generalization with unequal Gen

Sources: Varma et al., arXiv:2309.02390v1, section 3's unbalanced
compositional seeds, efficiency and slow-learning ingredients, and
appendix C's product train/test tables; actual native CE at e8a4910.

Fix the task and native constants. For every finite budget choose
one positive Gen partner seed, with its first factor exactly zero.
The same initialized actual path has bounded physical parameters
and both retained buffers, strictly fits train decisions at every
finite clock, forms a positive Gen product from step one, uniquely
selects the wrong held-out Mem class through the budget, and
permanently selects the true class after a generated later start.
The first strictly correct held-out clock is also derived beyond
the budget, without claiming it is the permanent-success start.

Neither equal Gen factors nor individual parameter convergence is
assumed. All sign, scale, box, balance, error and successful-margin
premises are initialized consequences of the original closed native
feedback. Clipping, clock and buffers are retained; no future gradient
stream, teacher or successful reference is supplied.

Mem starts with equal positive factors and is already train-fitted,
unlike the source's zero-first Mem factor. Both tables are fixed.
Exact reals, zero betas, sufficient small rate and uniform decoupled
decay differ from coupled-cost GD and learned nonzero-beta GPTMini.
The seed may be below machine range; accuracy success need not give
positive limiting confidence or zero limiting CE loss.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss

/-- One fixed actual task/native configuration has any finite
train-correct/test-wrong prefix followed by permanent held-out success
on the same bounded unequal-Gen path. Sources: section 3, appendix C
tables and native CE at e8a4910; only initial Gen partner data vary. -/
theorem gain_native_unequal_arbitrary_delayed_generalization (remaining : ℕ)
    (genGain memGain bound eps decay rate amplitude : ℝ)
    (hmem : 0 < memGain) (hbetter : memGain < genGain) (hclip : 0 < bound)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate)
    (hsmall : rate * (decay + genGain / eps) ≤ 1) (ha : 0 < amplitude) :
    ∀ budget : ℕ, ∃ seed ceiling : ℝ, 0 < seed ∧ 0 < ceiling ∧
      let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
        (seededNativeSubweights ((0, seed), (amplitude, amplitude)))
      (∀ n, NonnegativeNativeState (path n) ∧ ∀ i,
        (path n i).parameter ≤ ceiling ∧ -(path n i).moment ≤ bound ∧ (path n i).variance ≤ bound ^ 2) ∧
      (∀ n, StrictCorrect (trainTableLogits remaining
        (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0) ∧
      (∀ n, 0 < physicalCircuitScore genGain (path (n + 1) 0).parameter (path (n + 1) 1).parameter) ∧
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
  have hd := gain_small_rate_decay_remaining_positive genGain eps decay rate hgen he heta hsmall
  obtain ⟨seed, hseed, hpositive, hprefix⟩ := gain_native_arbitrary_wrong_test_prefix remaining
    genGain memGain bound 0 0 eps decay rate amplitude hmem hbetter hclip
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) he heta (le_of_lt hdecay) hd ha budget
  have hgMass : 0 < (0 : ℝ) + seed := by simpa only [zero_add] using hseed
  have hmMass : 0 < amplitude + amplitude := add_pos ha ha
  obtain ⟨ceiling, hceiling, _, _, hbounded, _, _⟩ := gain_native_seeded_relative_path_bound remaining
    genGain memGain bound eps decay rate 0 seed amplitude amplitude hgen hmem (le_of_lt hbetter) hclip he hdecay heta
    le_rfl (le_of_lt hseed) (le_of_lt ha) (le_of_lt ha) hgMass hmMass hsmall
  obtain ⟨start, htail⟩ := gain_native_seeded_unequal_eventual_correct remaining
    genGain memGain bound eps decay rate 0 seed amplitude amplitude hmem hbetter hclip he hdecay heta
    le_rfl (le_of_lt hseed) (le_of_lt ha) (le_of_lt ha) hgMass hmMass hsmall
  refine ⟨seed, ceiling, hseed, hceiling, hbounded, ?_, hpositive, ?_, start + budget + 1, by omega, ?_⟩
  · intro n
    cases n with
    | zero =>
      apply (train_table_strict_correct_iff remaining _ _).mpr
      change 0 < genGain * (0 * seed) + memGain * (amplitude * amplitude)
      simp only [mul_zero, zero_mul, zero_add]
      exact mul_pos hmem (mul_pos ha ha)
    | succ n =>
      exact gain_native_seeded_unequal_train_successor remaining genGain memGain bound eps decay rate
        0 seed amplitude amplitude hgen hmem (le_of_lt hbetter) hclip he hdecay heta
        le_rfl (le_of_lt hseed) (le_of_lt ha) (le_of_lt ha) hgMass hmMass hsmall n
  · intro n hn
    exact (hprefix n hn).2
  · intro n hn
    exact (htail n (by omega)).2

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 ∧ (0 : ℝ) < 1 := by norm_num

/-- One pinned legal native configuration admits arbitrary delayed
permanent accuracy generalization with first-zero Gen initialization.
Sources: section 3's slow partner seeds and actual CE at e8a4910;
this is an exact-real fixed-table possibility, not a GPTMini clock bound. -/
theorem fixed_native_unequal_arbitrary_delayed_generalization (remaining : ℕ) :
    ∀ budget : ℕ, ∃ seed ceiling : ℝ, 0 < seed ∧ 0 < ceiling ∧
      let path := gainNativePath remaining 3 2 1 0 0 1 (1 / 10) (1 / 1000)
        (seededNativeSubweights ((0, seed), (1, 1)))
      (∀ n, NonnegativeNativeState (path n) ∧ ∀ i,
        (path n i).parameter ≤ ceiling ∧ -(path n i).moment ≤ 1 ∧ (path n i).variance ≤ 1 ^ 2) ∧
      (∀ n, StrictCorrect (trainTableLogits remaining
        (physicalCircuitScore 3 (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore 2 (path n 2).parameter (path n 3).parameter)) 0) ∧
      (∀ n, 0 < physicalCircuitScore 3 (path (n + 1) 0).parameter (path (n + 1) 1).parameter) ∧
      (∀ n, n ≤ budget → StrictCorrect (heldoutTableLogits remaining
        (physicalCircuitScore 3 (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore 2 (path n 2).parameter (path n 3).parameter))
          (0 : Fin (remaining + 1)).succ) ∧
      ∃ start : ℕ, budget < start ∧ ∀ n, start ≤ n →
        StrictCorrect (heldoutTableLogits remaining
          (physicalCircuitScore 3 (path n 0).parameter (path n 1).parameter)
          (physicalCircuitScore 2 (path n 2).parameter (path n 3).parameter)) 0 := by
  exact gain_native_unequal_arbitrary_delayed_generalization remaining 3 2 1 1 (1 / 10) (1 / 1000) 1
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- The first strictly correct actual held-out clock exists beyond
every budget for a generated first-zero Gen partner seed. Sources:
section 3 delayed accuracy and appendix C true logits at e8a4910;
least-clock minimality does not imply permanent correctness from it. -/
theorem gain_native_unequal_first_correct_after_budget (remaining : ℕ)
    (genGain memGain bound eps decay rate amplitude : ℝ)
    (hmem : 0 < memGain) (hbetter : memGain < genGain) (hclip : 0 < bound)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate)
    (hsmall : rate * (decay + genGain / eps) ≤ 1) (ha : 0 < amplitude) (budget : ℕ) :
    ∃ seed : ℝ, 0 < seed ∧
      let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
        (seededNativeSubweights ((0, seed), (amplitude, amplitude)))
      let correct := fun n => StrictCorrect (heldoutTableLogits remaining
        (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0
      ∃ first : ℕ, budget < first ∧ correct first ∧ ∀ n, n < first → ¬correct n := by
  classical
  obtain ⟨seed, _, hseed, _, _, _, _, hprefix, start, _, htail⟩ :=
    gain_native_unequal_arbitrary_delayed_generalization remaining genGain memGain bound eps decay rate amplitude
      hmem hbetter hclip he hdecay heta hsmall ha budget
  let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
    (seededNativeSubweights ((0, seed), (amplitude, amplitude)))
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
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 ∧ (0 : ℝ) < 1 := by norm_num

end Transformer.Grokking.CircuitEfficiency
