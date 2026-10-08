import Transformer.Grokking.CircuitEfficiency.SectionC_GainUnequalRatio

/-!
# Generated unequal-pair relative-growth tails on actual native paths

Sources: Varma et al., arXiv:2309.02390v1, section 3's unbalanced
zero-first-factor initialization and appendix C's coupled product
partials; native clipped feedback and same-scale estimates at bb934a5.

Initialized nonnegative pairs with positive sums generate their full
physical/buffer box and positive masses. Their actual normalized
balancing errors tend to zero. Therefore every fixed positive error
tolerance is met at all clocks after a derived finite start.

Choose the tolerance as half the balanced efficiency gap generated
from that actual box and its actual positive clipped-CE floor. The
initialized sufficient small rate gives positive remaining decay.
The original actual native ratio step then gains one fixed factor
greater than one at every late clock where Gen mass is no greater
than Mem mass. None of these future bounds is an input hypothesis.

Individual parameter convergence, a supplied gradient stream and a
future successful circuit/margin are unnecessary. This is a derived
growth tail, not yet a crossing or persistent task-decision theorem.
Exact-real fixed tables, zero betas and uniform native decay differ
from appendix C's assigned coupled norm/GD and learned GPTMini.
The quantitative CE floor is loose; the start is qualitative rather
than a measured or floating-point grokking clock certificate.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW Filter

/-- Any positive numerical tolerance eventually bounds both actual
normalized pair errors, from initial nonnegative positive-sum data.
Sources: section 3's unequal factors and native estimates at bb934a5;
there is no future box, gradient callback or balance hypothesis. -/
theorem gain_native_seeded_relative_error_tail (remaining : ℕ)
    (genGain memGain bound eps decay rate a b c d allowance : ℝ)
    (hgen : 0 < genGain) (hmem : 0 < memGain) (hgain : memGain ≤ genGain)
    (hclip : 0 < bound) (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hdseed : 0 ≤ d)
    (hgMass : 0 < a + b) (hmMass : 0 < c + d)
    (hsmall : rate * (decay + genGain / eps) ≤ 1) (hallowance : 0 < allowance) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, b), (c, d)))
    let scale := fun n => gainCEGradientScale remaining genGain memGain bound (path n)
    ∃ start, ∀ n, start ≤ n →
      gainPairBalancingError genGain (scale n) eps (path n 0).parameter (path n 1).parameter /
        ((path n 0).parameter + (path n 1).parameter) ≤ allowance ∧
      gainPairBalancingError memGain (scale n) eps (path n 2).parameter (path n 3).parameter /
        ((path n 2).parameter + (path n 3).parameter) ≤ allowance := by
  have hlim := gain_native_seeded_proxy_error_tendsto_zero remaining genGain memGain bound eps decay rate a b c d
    hgen hmem hgain hclip he hdecay heta ha hb hc hdseed hgMass hmMass hsmall
  obtain ⟨genStart, hg⟩ := eventually_atTop.mp (hlim.1.eventually_lt_const hallowance)
  obtain ⟨memStart, hm⟩ := eventually_atTop.mp (hlim.2.eventually_lt_const hallowance)
  refine ⟨genStart + memStart, ?_⟩
  intro n hn
  exact ⟨le_of_lt (hg n (by omega)), le_of_lt (hm n (by omega))⟩

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 2 ∧ (2 : ℝ) ≤ 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 0 + 1 / 200 ∧ (0 : ℝ) < 0 + 1 ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 ∧ (0 : ℝ) < 1 / 100 := by norm_num

/-- Initial unequal positive-sum native seeds generate one finite box,
positive masses and a late fixed multiplicative Gen/Mem ratio advantage
while Gen is weaker. Sources: section 3's unbalanced seeds and original
closed native CE at bb934a5; no future successful path is supplied. -/
theorem gain_native_seeded_unequal_growth_tail (remaining : ℕ)
    (genGain memGain bound eps decay rate a b c d : ℝ)
    (hmem : 0 < memGain) (hgain : memGain < genGain) (hclip : 0 < bound)
    (he : 0 < eps) (hdecay : 0 < decay) (heta : 0 < rate)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hdseed : 0 ≤ d)
    (hgMass : 0 < a + b) (hmMass : 0 < c + d)
    (hsmall : rate * (decay + genGain / eps) ≤ 1) :
    let path := gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, b), (c, d)))
    ∃ ceiling : ℝ, 0 < ceiling ∧
      let growth := gainUnequalRatioGrowth (gainCEFeedbackFloor remaining genGain memGain bound ceiling)
        eps genGain memGain ceiling decay rate
      0 < 1 - rate * decay ∧ 1 < growth ∧
      (∀ n, NonnegativeNativeState (path n) ∧ ∀ i,
        (path n i).parameter ≤ ceiling ∧ -(path n i).moment ≤ bound ∧ (path n i).variance ≤ bound ^ 2) ∧
      (∀ n, 0 < (path n 0).parameter + (path n 1).parameter ∧
        0 < (path n 2).parameter + (path n 3).parameter) ∧
      ∃ start, ∀ n, start ≤ n →
        (path n 0).parameter + (path n 1).parameter ≤ (path n 2).parameter + (path n 3).parameter →
          growth * (((path n 0).parameter + (path n 1).parameter) /
            ((path n 2).parameter + (path n 3).parameter)) ≤
              ((path (n + 1) 0).parameter + (path (n + 1) 1).parameter) /
                ((path (n + 1) 2).parameter + (path (n + 1) 3).parameter) := by
  have hg := lt_trans hmem hgain
  have hd := gain_small_rate_decay_remaining_positive genGain eps decay rate hg he heta hsmall
  obtain ⟨ceiling, hceiling, _, _, hbox, hmass, _⟩ := gain_native_seeded_relative_path_bound remaining
    genGain memGain bound eps decay rate a b c d hg hmem (le_of_lt hgain) hclip he hdecay heta
    ha hb hc hdseed hgMass hmMass hsmall
  let lower := gainCEFeedbackFloor remaining genGain memGain bound ceiling
  let gap := gainBalancedRelativeGap lower eps genGain memGain ceiling
  have hlower : 0 < lower := gain_ce_feedback_floor_pos remaining genGain memGain bound ceiling
    (le_of_lt hg) (le_of_lt hceiling) hclip
  have hgap : 0 < gap := by
    unfold gap gainBalancedRelativeGap
    positivity
  have hq := gain_unequal_ratio_growth_one_lt lower eps genGain memGain ceiling decay rate
    hlower he hmem hgain (le_of_lt hceiling) heta hd
  obtain ⟨start, herror⟩ := gain_native_seeded_relative_error_tail remaining genGain memGain bound eps decay rate
    a b c d (gap / 2) hg hmem (le_of_lt hgain) hclip he hdecay heta ha hb hc hdseed hgMass hmMass hsmall
    (div_pos hgap (by norm_num))
  refine ⟨ceiling, hceiling, hd, hq, hbox, hmass, start, ?_⟩
  intro n hn hweak
  have hp : ∀ i, 0 ≤ (gainNativePath remaining genGain memGain bound 0 0 eps decay rate
      (seededNativeSubweights ((a, b), (c, d))) n i).parameter ∧
      (gainNativePath remaining genGain memGain bound 0 0 eps decay rate
        (seededNativeSubweights ((a, b), (c, d))) n i).parameter ≤ ceiling := by
    intro i
    exact ⟨((hbox n).1 i).1, ((hbox n).2 i).1⟩
  exact gain_native_unequal_ratio_growth remaining genGain memGain bound ceiling eps decay rate _
    hmem hgain hclip he heta hd hp (hmass n).1 (hmass n).2 hweak (herror n hn).1

example : (0 : ℝ) < 2 ∧ (2 : ℝ) < 3 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 10 ∧ (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 0 + 1 / 200 ∧ (0 : ℝ) < 0 + 1 ∧
    (1 / 1000 : ℝ) * (1 / 10 + 3 / 1) ≤ 1 := by norm_num

/-- One fixed legal native configuration generates a late ratio
advantage from the source-style zero-first-factor seeds for every
positive Gen partner. Sources: section 3 initialization and native
feedback at bb934a5; this fixes hyperparameters and supplies no
future error tolerance, balance, convergence or successful margin. -/
theorem fixed_native_unequal_growth_tail (remaining : ℕ) (seed : ℝ) (hseed : 0 < seed) :
    let path := gainNativePath remaining 3 2 1 0 0 1 (1 / 10) (1 / 1000)
      (seededNativeSubweights ((0, seed), (0, 1)))
    ∃ ceiling : ℝ, 0 < ceiling ∧
      let growth := gainUnequalRatioGrowth (gainCEFeedbackFloor remaining 3 2 1 ceiling)
        1 3 2 ceiling (1 / 10) (1 / 1000)
      1 < growth ∧ ∃ start, ∀ n, start ≤ n →
        (path n 0).parameter + (path n 1).parameter ≤ (path n 2).parameter + (path n 3).parameter →
          growth * (((path n 0).parameter + (path n 1).parameter) /
            ((path n 2).parameter + (path n 3).parameter)) ≤
              ((path (n + 1) 0).parameter + (path (n + 1) 1).parameter) /
                ((path (n + 1) 2).parameter + (path (n + 1) 3).parameter) := by
  obtain ⟨ceiling, hceiling, _, hq, _, _, start, hstep⟩ := gain_native_seeded_unequal_growth_tail remaining
    3 2 1 1 (1 / 10) (1 / 1000) 0 seed 0 1
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (le_of_lt hseed) (by norm_num) (by norm_num)
    (by simpa only [zero_add] using hseed) (by norm_num) (by norm_num)
  exact ⟨ceiling, hceiling, hq, start, hstep⟩

example : (0 : ℝ) < 1 / 200 := by norm_num

end Transformer.Grokking.CircuitEfficiency
