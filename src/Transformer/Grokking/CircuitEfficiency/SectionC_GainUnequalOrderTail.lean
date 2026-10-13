import Transformer.Grokking.CircuitEfficiency.SectionC_GainUnequalOrderStep

/-!
# Permanent actual unequal Gen mass dominance from initial data

Sources: Varma et al., arXiv:2309.02390v1, section 3's unbalanced
compositional seeds and appendix C's true gained product partials;
closed retained native feedback and initialized bounds at 68d90fa.

Initialized nonnegative positive-sum pairs generate one finite box,
positive masses and vanishing normalized Gen balancing error. Choose
the positive order tolerance derived from the actual box/CE floor.
After its generated finite start, every actual mass-ordered point
has a strictly mass-ordered next native step. The proved crossing
beyond every budget supplies a crossing after that same error start.
Iteration therefore yields permanent strict Gen mass dominance.

No individual parameter limit, future error bound, successful reference
or prescribed gradient stream is a hypothesis. The source-style
zero-first-factor family is explicitly instantiated. At every successor
clock, each actual pair also has two positive factors, derived from
its positive mass and actual partner update rather than assumed.

These are mass and formation statements; physical product/decision
selection needs the relative-balance result as well. Exact reals,
zero betas, fixed tables, sufficient small rate and uniform native
decay differ from the source's coupled-cost GD and learned GPTMini.
No positive limiting mass, loss convergence or measured clock is claimed.
The finite start depends on initialization and the fixed constants.
It need not be uniformly bounded over positive seeds tending to zero.
Retained clocks and buffers remain those of the native recurrence.
Selection uses its actual current signs, sums and clipped CE inputs.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Actual initialized unequal native paths have permanent strict
Gen mass dominance after a derived finite start. Sources: section 3's
unbalanced factors and original native CE at 68d90fa; all box, error,
mass, crossing and preservation premises are initialized consequences. -/
theorem gain_native_seeded_unequal_order_tail (remaining : ℕ)
    (genGain memGain bound eps decay rate a b c d : ℝ)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hclip : 0 < bound)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hdseed : 0 ≤ d)
    (hgMass : 0 < a + b) (hmMass : 0 < c + d)
    (hsmall : rate * (decay + genGain / eps) ≤ 1) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, b), (c, d)))
    ∃ start, ∀ n, start ≤ n →
      (path n 2).parameter + (path n 3).parameter < (path n 0).parameter + (path n 1).parameter := by
  let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
    (seededNativeSubweights ((a, b), (c, d)))
  have hg := lt_trans hmem hgain
  have hd := gain_small_rate_decay_remaining_positive genGain eps decay rate hg he heta hsmall
  obtain ⟨ceiling, hceiling, _, _, hbox, hmass, _⟩ := gain_native_seeded_relative_path_bound remaining
    genGain memGain bound eps decay rate a b c d hg hmem (le_of_lt hgain) hclip he hdecay heta
    ha hb hc hdseed hgMass hmMass hsmall
  let lower := gainCEFeedbackFloor remaining genGain memGain bound ceiling
  let tolerance := gainUnequalOrderTolerance lower eps genGain memGain ceiling decay rate
  have hlower : 0 < lower := gain_ce_feedback_floor_pos remaining genGain memGain bound ceiling
    (le_of_lt hg) (le_of_lt hceiling) hclip
  have ht := gain_unequal_order_tolerance_bounds lower eps genGain memGain ceiling decay rate
    hlower he hmem hgain (le_of_lt hceiling) heta hd
  obtain ⟨errorStart, herr⟩ := gain_native_seeded_relative_error_tail remaining genGain memGain bound eps decay rate
    a b c d tolerance hg hmem (le_of_lt hgain) hclip he hdecay heta ha hb hc hdseed hgMass hmMass hsmall ht.1
  obtain ⟨crossing, hcrossStart, hcross⟩ := gain_native_seeded_unequal_crossing_after remaining
    genGain memGain bound eps decay rate a b c d hmem hgain hclip he hdecay heta
    ha hb hc hdseed hgMass hmMass hsmall errorStart
  have htail : ∀ k, (path (crossing + k) 2).parameter + (path (crossing + k) 3).parameter <
      (path (crossing + k) 0).parameter + (path (crossing + k) 1).parameter := by
    intro k
    induction k with
    | zero => simpa only [Nat.add_zero] using hcross
    | succ k ih =>
      have hn : errorStart ≤ crossing + k := by omega
      have hp : ∀ i, 0 ≤ (path (crossing + k) i).parameter ∧ (path (crossing + k) i).parameter ≤ ceiling := by
        intro i
        exact ⟨((hbox (crossing + k)).1 i).1, ((hbox (crossing + k)).2 i).1⟩
      exact gain_native_unequal_order_step remaining genGain memGain bound ceiling eps decay rate (path (crossing + k))
        hmem hgain hclip he heta hd hp (hmass (crossing + k)).2 (le_of_lt ih) (herr (crossing + k) hn).1
  refine ⟨crossing, ?_⟩
  intro n hn
  have hindex : crossing + (n - crossing) = n := by omega
  simpa only [hindex] using htail (n - crossing)

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 0 + 1 / 200 ∧ (0 : ℝ) < 0 + 1 ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by norm_num

/-- The pinned legal first-zero-factor family has permanent actual
Gen mass dominance for every positive Gen partner. Sources: section 3
initialization and original native CE at 68d90fa; fixed task/native
constants require no successful future reference or parameter limit. -/
theorem fixed_native_unequal_order_tail (remaining : ℕ) (seed : ℝ) (hseed : 0 < seed) :
    let path := gainNativePath remaining 3 2 1 0 0 1 (1 / 10) (1 / 1000)
      (seededNativeSubweights ((0, seed), (0, 1)))
    ∃ start, ∀ n, start ≤ n →
      (path n 2).parameter + (path n 3).parameter < (path n 0).parameter + (path n 1).parameter := by
  exact gain_native_seeded_unequal_order_tail remaining 3 2 1 1 (1 / 10) (1 / 1000) 0 seed 0 1
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (le_of_lt hseed) (by norm_num) (by norm_num)
    (by simpa only [zero_add] using hseed) (by norm_num) (by norm_num)

example : (0 : ℝ) < 1 / 200 := by norm_num

/-- Every actual initialized unequal pair has two positive factors
at every successor clock. Sources: section 3 compositional seeds and
native partner normalization at 68d90fa; derive positive sums/signs
from initialization, including either zero-factor ordering. -/
theorem gain_native_seeded_unequal_positive_successor (remaining : ℕ)
    (genGain memGain bound eps decay rate a b c d : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain)
    (hclip : 0 < bound) (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hdseed : 0 ≤ d)
    (hgMass : 0 < a + b) (hmMass : 0 < c + d)
    (hsmall : rate * (decay + genGain / eps) ≤ 1) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, b), (c, d)))
    ∀ n (i : Fin 4), 0 < (path (n + 1) i).parameter := by
  let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
    (seededNativeSubweights ((a, b), (c, d)))
  have hd := gain_small_rate_decay_remaining_positive genGain eps decay rate hgen he heta hsmall
  obtain ⟨_, _, _, _, hbox, hmass, _⟩ := gain_native_seeded_relative_path_bound remaining
    genGain memGain bound eps decay rate a b c d hgen hmem hgain hclip he hdecay heta
    ha hb hc hdseed hgMass hmMass hsmall
  dsimp only
  intro n i
  have hp : ∀ j, 0 ≤ (path n j).parameter := fun j => ((hbox n).1 j).1
  have hs := gain_ce_gradient_scale_pos remaining genGain memGain bound (path n) hclip
  have hform := gain_native_zero_beta_pair_step remaining genGain memGain bound eps decay rate (path n) hgen hmem hclip hp
  have hg := gain_pair_step_factors_positive genGain _ eps decay rate _ _ hgen hs he heta hd (hp 0) (hp 1) (hmass n).1
  have hm := gain_pair_step_factors_positive memGain _ eps decay rate _ _ hmem hs he heta hd (hp 2) (hp 3) (hmass n).2
  rw [← hform.1] at hg
  rw [← hform.2] at hm
  fin_cases i
  · exact hg.1
  · exact hg.2
  · exact hm.1
  · exact hm.2

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 0 + 1 / 200 ∧ (0 : ℝ) < 0 + 1 ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by norm_num

end Transformer.Grokking.CircuitEfficiency
