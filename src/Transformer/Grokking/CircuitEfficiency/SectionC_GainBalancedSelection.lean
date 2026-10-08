import Transformer.Grokking.CircuitEfficiency.SectionC_GainBalancedOrder

/-!
# Persistent actual task decisions without parameter convergence

Sources: Varma et al., arXiv:2309.02390v1, appendix C's Gen/Mem
train and held-out logits; native AdamW at lab commit 91557cd.
Translate the derived balanced factor ordering into a quantitative
physical score gap and strict correctness against every class.

Positive balanced initialized native factors give strictly correct
training decisions at every finite clock. Greater Gen gain and the
proved finite crossing/persistent factor order give strictly correct
held-out decisions at every clock after a generated finite start.
No future margin, successful reference or finite parameter limit
is supplied as a hypothesis. All decisions use the actual forward.

The current lower margin scales with Mem's current squared factor;
it is not a uniform positive limiting margin. These exact-real,
fixed gained-table, equal-positive-pair, legal zero-beta results differ
from appendix C's coupled norm/GD and unbalanced seeds, and from the
preserved learned nonzero-beta GPTMini. Loss convergence, numerical
classification and a measured transition time remain separate.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss

/-- Ordered balanced factors have a positive quantitative actual Gen
score advantage. Sources: appendix C's product forward and section 3
physical gain efficiency at 91557cd; retained states enter only through
their actual current parameters and within-pair equality. -/
theorem gain_native_balanced_order_margin (genGain memGain : ℝ) (state : NativeSubweightState)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hs : NativePairSymmetry state)
    (hb : 0 < (state 2).parameter) (horder : (state 2).parameter ≤ (state 0).parameter) :
    0 < (genGain - memGain) * ((state 2).parameter * (state 2).parameter) ∧
      (genGain - memGain) * ((state 2).parameter * (state 2).parameter) ≤
        physicalCircuitScore genGain (state 0).parameter (state 1).parameter -
          physicalCircuitScore memGain (state 2).parameter (state 3).parameter := by
  have hg : 0 < genGain := lt_trans hmem hgain
  have ha : 0 < (state 0).parameter := lt_of_lt_of_le hb horder
  have hsq := mul_le_mul horder horder (le_of_lt hb) (le_of_lt ha)
  have hscaled := mul_le_mul_of_nonneg_left hsq (le_of_lt hg)
  have hgap : 0 < genGain - memGain := by linarith only [hgain]
  constructor
  · exact mul_pos hgap (mul_pos hb hb)
  · rw [← hs.1, ← hs.2]
    unfold physicalCircuitScore
    nlinarith only [hscaled]

example :
    let state := seededNativeSubweights ((1, 1), (1, 1))
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ NativePairSymmetry state ∧
    0 < (state 2).parameter ∧ (state 2).parameter ≤ (state 0).parameter := by
  dsimp only
  exact ⟨by norm_num, by norm_num, ⟨rfl, rfl⟩,
    by norm_num [seededNativeSubweights, seededScalarState], by norm_num [seededNativeSubweights, seededScalarState]⟩

/-- The actual ordered physical point is strictly correct on train
and held-out tables. Source: appendix C's true logits at 91557cd;
derive all-class correctness from the quantitative score advantage. -/
theorem gain_native_balanced_order_correct (remaining : ℕ) (genGain memGain : ℝ) (state : NativeSubweightState)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hs : NativePairSymmetry state)
    (hb : 0 < (state 2).parameter) (horder : (state 2).parameter ≤ (state 0).parameter) :
    StrictCorrect (trainTableLogits remaining
      (physicalCircuitScore genGain (state 0).parameter (state 1).parameter)
      (physicalCircuitScore memGain (state 2).parameter (state 3).parameter)) 0 ∧
      StrictCorrect (heldoutTableLogits remaining
        (physicalCircuitScore genGain (state 0).parameter (state 1).parameter)
        (physicalCircuitScore memGain (state 2).parameter (state 3).parameter)) 0 := by
  have hg : 0 < genGain := lt_trans hmem hgain
  have ha : 0 < (state 0).parameter := lt_of_lt_of_le hb horder
  have hx : 0 < physicalCircuitScore genGain (state 0).parameter (state 1).parameter := by
    rw [← hs.1]
    exact mul_pos hg (mul_pos ha ha)
  have hy : 0 < physicalCircuitScore memGain (state 2).parameter (state 3).parameter := by
    rw [← hs.2]
    exact mul_pos hmem (mul_pos hb hb)
  have hm := gain_native_balanced_order_margin genGain memGain state hmem hgain hs hb horder
  have hscore : physicalCircuitScore memGain (state 2).parameter (state 3).parameter <
      physicalCircuitScore genGain (state 0).parameter (state 1).parameter := by linarith only [hm.1, hm.2]
  exact ⟨(train_table_strict_correct_iff remaining _ _).mpr (by linarith only [hx, hy]),
    heldout_table_strict_correct remaining _ _ hx hscore⟩

example :
    let state := seededNativeSubweights ((1, 1), (1, 1))
    (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ NativePairSymmetry state ∧
    0 < (state 2).parameter ∧ (state 2).parameter ≤ (state 0).parameter := by
  dsimp only
  exact ⟨by norm_num, by norm_num, ⟨rfl, rfl⟩,
    by norm_num [seededNativeSubweights, seededScalarState], by norm_num [seededNativeSubweights, seededScalarState]⟩

/-- Actual positive balanced initialized paths fit training decisions
at every finite clock. Sources: appendix C's training tables and
native sign preservation at 91557cd; no current/future Gen dominance
or convergence premise is needed for this training-only statement. -/
theorem gain_native_balanced_train_correct (remaining : ℕ) (genGain memGain bound eps decay rate a b : ℝ)
    (n : ℕ) (hgen : 0 < genGain) (hmem : 0 < memGain) (hclip : 0 < bound) (he : 0 < eps)
    (heta : 0 ≤ rate) (hd : 0 < 1 - rate * decay) (ha : 0 < a) (hb : 0 < b) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, a), (b, b)))
    StrictCorrect (trainTableLogits remaining
      (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
      (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0 := by
  have hp : ∀ i, 0 < (gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, a), (b, b))) n i).parameter := by
    intro i
    exact (gain_native_seeded_balanced_positive remaining genGain memGain bound eps decay rate a b n i
      hgen hmem hclip he heta hd ha hb).1
  apply (train_table_strict_correct_iff remaining _ _).mpr
  exact add_pos (mul_pos hgen (mul_pos (hp 0) (hp 1))) (mul_pos hmem (mul_pos (hp 2) (hp 3)))

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    (0 : ℝ) < 1 / 200 ∧ (0 : ℝ) < 1 := by
  norm_num

/-- Actual positive balanced native paths permanently solve held-out
decisions after a derived finite start. Sources: appendix C's actual
task logits and native AdamW at 91557cd; no finite parameter limit,
future successful reference or margin is supplied as a hypothesis. -/
theorem gain_native_balanced_eventual_correct (remaining : ℕ) (genGain memGain bound eps decay rate a b : ℝ)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hclip : 0 < bound)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (ha : 0 < a) (hb : 0 < b) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, a), (b, b)))
    ∃ start, ∀ n, start ≤ n →
      StrictCorrect (trainTableLogits remaining
        (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0 ∧
        StrictCorrect (heldoutTableLogits remaining
          (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
          (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0 := by
  obtain ⟨start, horder⟩ := gain_native_balanced_order_tail remaining genGain memGain bound eps decay rate a b
    hmem hgain hclip he hdecay heta hd ha hb
  refine ⟨start, ?_⟩
  intro n hn
  have hg : 0 < genGain := lt_trans hmem hgain
  have hp := (gain_native_seeded_balanced_positive remaining genGain memGain bound eps decay rate a b n 2
    hg hmem hclip he (le_of_lt heta) hd ha hb).1
  exact gain_native_balanced_order_correct remaining genGain memGain _ hmem hgain
    (gain_native_equal_pair_path remaining genGain memGain bound 0 0 eps decay rate a b n) hp (le_of_lt (horder n hn))

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 10 ∧
    (0 : ℝ) < 1 / 1000 ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    (0 : ℝ) < 1 / 200 ∧ (0 : ℝ) < 1 := by
  norm_num

end Transformer.Grokking.CircuitEfficiency
