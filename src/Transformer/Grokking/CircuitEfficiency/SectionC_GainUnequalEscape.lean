import Transformer.Grokking.CircuitEfficiency.SectionC_GainUnequalGrowthTail
import Mathlib.Analysis.SpecificLimits.Basic

/-!
# Actual unequal native Gen mass crosses Mem after every finite budget

Sources: Varma et al., arXiv:2309.02390v1, section 3's competing
unbalanced products and appendix C's true CE partials; closed native
zero-beta dynamics and quantitative ratio estimates at bb934a5.

The actual initialized path derives a finite relative-growth start.
If Gen never exceeds Mem after a given budget, its positive mass
ratio after both starts grows at least as a fixed q>1 raised to the
number of actual updates. Yet that same ratio is at most one.
Unbounded powers contradict this, yielding a strict mass crossing
at some finite clock beyond every prescribed budget.

There is no individual parameter-limit, future successful reference,
prescribed gradient stream or attraction hypothesis. Source-style
first-zero-factor seeds satisfy the static nonnegative positive-sum
conditions. All update, clipping and CE terms remain the original
actual feedback. This is a mass statement; physical product ordering
and permanently correct held-out decisions require additional work.

Exact reals, fixed gained tables, zero betas and sufficient small
rate differ from appendix C's coupled-cost GD and learned GPTMini.
The finite clock is qualitative, not a numerical or FLOP bound.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW Filter

/-- Actual unequal native Gen mass strictly exceeds Mem mass at a
generated finite clock beyond every budget. Sources: section 3's
unbalanced seeds and native feedback at bb934a5; the box, positive
masses and eventual growth bound are derived from initialization. -/
theorem gain_native_seeded_unequal_crossing_after (remaining : ℕ)
    (genGain memGain bound eps decay rate a b c d : ℝ)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hclip : 0 < bound)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hdseed : 0 ≤ d)
    (hgMass : 0 < a + b) (hmMass : 0 < c + d)
    (hsmall : rate * (decay + genGain / eps) ≤ 1) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, b), (c, d)))
    ∀ budget : ℕ, ∃ n, budget ≤ n ∧
      (path n 2).parameter + (path n 3).parameter < (path n 0).parameter + (path n 1).parameter := by
  dsimp only
  let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
    (seededNativeSubweights ((a, b), (c, d)))
  intro budget
  change ∃ n, budget ≤ n ∧ (path n 2).parameter + (path n 3).parameter <
    (path n 0).parameter + (path n 1).parameter
  by_contra hnever
  push Not at hnever
  obtain ⟨ceiling, _, _, hq, _, hmass, start, hstep⟩ := gain_native_seeded_unequal_growth_tail remaining
    genGain memGain bound eps decay rate a b c d hmem hgain hclip he hdecay heta
    ha hb hc hdseed hgMass hmMass hsmall
  let growth := gainUnequalRatioGrowth (gainCEFeedbackFloor remaining genGain memGain bound ceiling)
    eps genGain memGain ceiling decay rate
  let base := start + budget
  let ratio := fun n => ((path n 0).parameter + (path n 1).parameter) /
    ((path n 2).parameter + (path n 3).parameter)
  have hr : 0 < ratio base := div_pos (hmass base).1 (hmass base).2
  have hpower : ∀ k, growth ^ k * ratio base ≤ ratio (base + k) := by
    intro k
    induction k with
    | zero => simp only [pow_zero, one_mul, Nat.add_zero, le_refl]
    | succ k ih =>
      have hn : start ≤ base + k := by dsimp only [base]; omega
      have hb : budget ≤ base + k := by dsimp only [base]; omega
      have hweak := hnever (base + k) hb
      have hcurrent := hstep (base + k) hn hweak
      change growth * ratio (base + k) ≤ ratio ((base + k) + 1) at hcurrent
      calc
        growth ^ (k + 1) * ratio base = growth * (growth ^ k * ratio base) := by rw [pow_succ]; ring
        _ ≤ growth * ratio (base + k) := mul_le_mul_of_nonneg_left ih (by linarith only [hq])
        _ ≤ ratio ((base + k) + 1) := hcurrent
        _ = ratio (base + (k + 1)) := rfl
  obtain ⟨k, hk⟩ := ((tendsto_pow_atTop_atTop_of_one_lt hq).eventually_gt_atTop (1 / ratio base)).exists
  have hlarge : 1 < growth ^ k * ratio base := (div_lt_iff₀ hr).mp hk
  have hb : budget ≤ base + k := by dsimp only [base]; omega
  have hupper : ratio (base + k) ≤ 1 := by
    apply (div_le_iff₀ (hmass (base + k)).2).mpr
    simpa only [one_mul] using hnever (base + k) hb
  linarith only [hlarge, hpower k, hupper]

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 0 + 1 / 200 ∧ (0 : ℝ) < 0 + 1 ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by norm_num

/-- One fixed legal native configuration crosses from source-style
first-zero Gen/Mem seeds beyond every finite budget for every positive
Gen partner. Sources: section 3 initialization and native CE at
bb934a5; only initialization varies, not a future reference path. -/
theorem fixed_native_unequal_crossing_after (remaining : ℕ) (seed : ℝ) (hseed : 0 < seed) :
    let path := gainNativePath remaining 3 2 1 0 0 1 (1 / 10) (1 / 1000)
      (seededNativeSubweights ((0, seed), (0, 1)))
    ∀ budget : ℕ, ∃ n, budget ≤ n ∧
      (path n 2).parameter + (path n 3).parameter < (path n 0).parameter + (path n 1).parameter := by
  exact gain_native_seeded_unequal_crossing_after remaining 3 2 1 1 (1 / 10) (1 / 1000) 0 seed 0 1
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (le_of_lt hseed) (by norm_num) (by norm_num)
    (by simpa only [zero_add] using hseed) (by norm_num) (by norm_num)

example : (0 : ℝ) < 1 / 200 := by norm_num

/-- No actual initialized unequal native path in this regime stays
Gen-weaker at all clocks after any finite start. Sources: section 3
efficiency and native CE at bb934a5; this rules out a weak tail but
does not yet establish permanent strict dominance or product order. -/
theorem gain_native_seeded_unequal_no_weak_tail (remaining : ℕ)
    (genGain memGain bound eps decay rate a b c d : ℝ)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hclip : 0 < bound)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hdseed : 0 ≤ d)
    (hgMass : 0 < a + b) (hmMass : 0 < c + d)
    (hsmall : rate * (decay + genGain / eps) ≤ 1) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, b), (c, d)))
    ¬ ∃ start, ∀ n, start ≤ n →
      (path n 0).parameter + (path n 1).parameter ≤ (path n 2).parameter + (path n 3).parameter := by
  dsimp only
  intro ⟨start, hweak⟩
  obtain ⟨n, hn, hcross⟩ := gain_native_seeded_unequal_crossing_after remaining
    genGain memGain bound eps decay rate a b c d hmem hgain hclip he hdecay heta
    ha hb hc hdseed hgMass hmMass hsmall start
  exact (not_lt_of_ge (hweak n hn)) hcross

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 0 + 1 / 200 ∧ (0 : ℝ) < 0 + 1 ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by norm_num

/-- The pinned zero-first-factor family cannot remain Gen-weaker
forever after any finite start. Sources: section 3's initialization
and original native CE at bb934a5; this is still a mass statement,
without a hidden parameter limit or permanent decision premise. -/
theorem fixed_native_unequal_no_weak_tail (remaining : ℕ) (seed : ℝ) (hseed : 0 < seed) :
    let path := gainNativePath remaining 3 2 1 0 0 1 (1 / 10) (1 / 1000)
      (seededNativeSubweights ((0, seed), (0, 1)))
    ¬ ∃ start, ∀ n, start ≤ n →
      (path n 0).parameter + (path n 1).parameter ≤ (path n 2).parameter + (path n 3).parameter := by
  dsimp only
  intro ⟨start, hweak⟩
  obtain ⟨n, hn, hcross⟩ := fixed_native_unequal_crossing_after remaining seed hseed start
  exact (not_lt_of_ge (hweak n hn)) hcross

example : (0 : ℝ) < 1 / 200 := by norm_num

end Transformer.Grokking.CircuitEfficiency
