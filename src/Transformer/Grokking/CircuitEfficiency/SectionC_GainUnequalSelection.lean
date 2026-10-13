import Transformer.Grokking.CircuitEfficiency.SectionC_GainMassBalanceScore

/-!
# Permanent actual task correctness from unequal native initialization

Sources: Varma et al., arXiv:2309.02390v1, section 3's unbalanced
compositional seeds and appendix C's true product train/test logits;
closed native feedback, mass selection and relative balance at 483b2b1.

At every successor clock positive actual factors give strictly correct
training decisions. Permanent Gen mass dominance and vanishing squared
relative Gen asymmetry meet the quantitative score-gap criterion after
one derived finite start. The same initialized actual path then has a
positive current physical margin and strictly correct train/held-out
decisions against every class at all later clocks.

This removes both equal-factor initialization and individual parameter
convergence from this zero-beta selection route. Initial nonnegative
positive-sum data and sufficient numerical small rate generate all
future sign, box, mass, error, dominance and balance premises. The
source-style first-zero-factor family is instantiated with fixed legal
native constants and any positive Gen partner seed.

Only current margins and permanent exact-real decisions are proved.
Their lower bound may vanish if Mem mass shrinks. Positive limiting
mass/margin, CE convergence and measured formation time are separate.
Fixed tables, zero betas and uniform native decay differ from the
source's coupled-cost GD and preserved learned nonzero-beta GPTMini.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW Transformer.Grokking.NaiveLoss Filter

/-- Actual unequal initialized native paths fit training decisions
at every successor clock, including first-zero-factor seeds. Sources:
appendix C training logits and native formation at 483b2b1; training
correctness needs no Gen efficiency advantage or future convergence. -/
theorem gain_native_seeded_unequal_train_successor (remaining : ℕ)
    (genGain memGain bound eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain)
    (hclip : 0 < bound) (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hdseed : 0 ≤ d)
    (hgMass : 0 < a + b) (hmMass : 0 < c + d)
    (hsmall : rate * (decay + genGain / eps) ≤ 1) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, b), (c, d)))
    ∀ n, StrictCorrect (trainTableLogits remaining
      (physicalCircuitScore genGain (path (n + 1) 0).parameter (path (n + 1) 1).parameter)
      (physicalCircuitScore memGain (path (n + 1) 2).parameter (path (n + 1) 3).parameter)) 0 := by
  have hp := gain_native_seeded_unequal_positive_successor remaining genGain memGain bound eps decay rate a b c d
    hgen hmem hgain hclip he hdecay heta ha hb hc hdseed hgMass hmMass hsmall
  dsimp only
  intro n
  apply (train_table_strict_correct_iff remaining _ _).mpr
  exact add_pos (mul_pos hgen (mul_pos (hp n 0) (hp n 1))) (mul_pos hmem (mul_pos (hp n 2) (hp n 3)))

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 0 + 1 / 200 ∧ (0 : ℝ) < 0 + 1 ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by norm_num

/-- The initialized unequal actual path eventually has a positive
quantitative physical Gen margin at every clock. Sources: appendix C
product forward and native estimates at 483b2b1; current NN states,
mass dominance and the relative-balance budget are all derived. -/
theorem gain_native_seeded_unequal_margin_tail (remaining : ℕ)
    (genGain memGain bound eps decay rate a b c d : ℝ)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hclip : 0 < bound)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hdseed : 0 ≤ d)
    (hgMass : 0 < a + b) (hmMass : 0 < c + d)
    (hsmall : rate * (decay + genGain / eps) ≤ 1) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, b), (c, d)))
    ∃ start, ∀ n, start ≤ n → NonnegativeNativeState (path n) ∧
      0 < (genGain - memGain) * ((path n 2).parameter + (path n 3).parameter) ^ 2 / 8 ∧
      (genGain - memGain) * ((path n 2).parameter + (path n 3).parameter) ^ 2 / 8 ≤
        physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter -
          physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter := by
  have hg := lt_trans hmem hgain
  obtain ⟨_, _, _, _, hbox, hmass, _⟩ := gain_native_seeded_relative_path_bound remaining
    genGain memGain bound eps decay rate a b c d hg hmem (le_of_lt hgain) hclip he hdecay heta
    ha hb hc hdseed hgMass hmMass hsmall
  obtain ⟨massStart, horder⟩ := gain_native_seeded_unequal_order_tail remaining
    genGain memGain bound eps decay rate a b c d hmem hgain hclip he hdecay heta
    ha hb hc hdseed hgMass hmMass hsmall
  have hbalance := gain_native_seeded_relative_difference_tendsto_zero remaining
    genGain memGain bound eps decay rate a b c d hg hmem (le_of_lt hgain) hclip he hdecay heta
    ha hb hc hdseed hgMass hmMass hsmall
  have hbudget : 0 < (genGain - memGain) / (2 * genGain) := by
    have hd : 0 < genGain - memGain := by linarith only [hgain]
    exact div_pos hd (by positivity)
  have hsquare := hbalance.1.pow 2
  simp only [zero_pow (by norm_num : (2 : ℕ) ≠ 0)] at hsquare
  obtain ⟨balanceStart, hb⟩ := eventually_atTop.mp (hsquare.eventually_lt_const hbudget)
  refine ⟨massStart + balanceStart, ?_⟩
  intro n hn
  have hm := gain_pair_mass_balance_score_margin genGain memGain _ _ _ _ hmem hgain (hmass n).2
    (le_of_lt (horder n (by omega))) (le_of_lt (hb n (by omega)))
  exact ⟨(hbox n).1, hm.1, hm.2⟩

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 0 + 1 / 200 ∧ (0 : ℝ) < 0 + 1 ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by norm_num

/-- Initialized unequal native paths permanently solve train and
held-out decisions after a generated finite clock. Sources: appendix C
actual logits and native feedback at 483b2b1; no parameter limit,
future margin or successful reference is supplied as a hypothesis. -/
theorem gain_native_seeded_unequal_eventual_correct (remaining : ℕ)
    (genGain memGain bound eps decay rate a b c d : ℝ)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hclip : 0 < bound)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hdseed : 0 ≤ d)
    (hgMass : 0 < a + b) (hmMass : 0 < c + d)
    (hsmall : rate * (decay + genGain / eps) ≤ 1) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, b), (c, d)))
    ∃ start, ∀ n, start ≤ n →
      StrictCorrect (trainTableLogits remaining
        (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0 ∧
        StrictCorrect (heldoutTableLogits remaining
          (physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter)
          (physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter)) 0 := by
  let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
    (seededNativeSubweights ((a, b), (c, d)))
  obtain ⟨start, hmargin⟩ := gain_native_seeded_unequal_margin_tail remaining
    genGain memGain bound eps decay rate a b c d hmem hgain hclip he hdecay heta
    ha hb hc hdseed hgMass hmMass hsmall
  refine ⟨start, ?_⟩
  intro n hn
  obtain ⟨hs, hpositive, hlower⟩ := hmargin n hn
  change 0 < (genGain - memGain) * ((path n 2).parameter + (path n 3).parameter) ^ 2 / 8 at hpositive
  change (genGain - memGain) * ((path n 2).parameter + (path n 3).parameter) ^ 2 / 8 ≤
    physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter -
      physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter at hlower
  have hy : 0 ≤ physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter :=
    mul_nonneg (le_of_lt hmem) (mul_nonneg (hs 2).1 (hs 3).1)
  have hscore : physicalCircuitScore memGain (path n 2).parameter (path n 3).parameter <
      physicalCircuitScore genGain (path n 0).parameter (path n 1).parameter := by linarith only [hpositive, hlower]
  have hx := lt_of_le_of_lt hy hscore
  exact ⟨(train_table_strict_correct_iff remaining _ _).mpr (by linarith only [hx, hy]),
    heldout_table_strict_correct remaining _ _ hx hscore⟩

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 0 + 1 / 200 ∧ (0 : ℝ) < 0 + 1 ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by norm_num

/-- The pinned first-zero-factor family permanently solves the actual
train/held-out task for every positive Gen partner. Sources: section 3
initialization and native CE at 483b2b1; task/native constants are
fixed and no positive limiting weights or future success are assumed. -/
theorem fixed_native_unequal_eventual_correct (remaining : ℕ) (seed : ℝ) (hseed : 0 < seed) :
    let path := gainNativePath remaining 3 2 1 0 0 1 (1 / 10) (1 / 1000)
      (seededNativeSubweights ((0, seed), (0, 1)))
    ∃ start, ∀ n, start ≤ n →
      StrictCorrect (trainTableLogits remaining (physicalCircuitScore 3 (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore 2 (path n 2).parameter (path n 3).parameter)) 0 ∧
      StrictCorrect (heldoutTableLogits remaining (physicalCircuitScore 3 (path n 0).parameter (path n 1).parameter)
        (physicalCircuitScore 2 (path n 2).parameter (path n 3).parameter)) 0 := by
  exact gain_native_seeded_unequal_eventual_correct remaining 3 2 1 1 (1 / 10) (1 / 1000) 0 seed 0 1
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (le_of_lt hseed) (by norm_num) (by norm_num)
    (by simpa only [zero_add] using hseed) (by norm_num) (by norm_num)

example : (0 : ℝ) < 1 / 200 := by norm_num

end Transformer.Grokking.CircuitEfficiency
