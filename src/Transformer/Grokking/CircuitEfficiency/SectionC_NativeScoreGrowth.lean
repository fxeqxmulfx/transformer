import Transformer.Grokking.CircuitEfficiency.SectionC_NativeLimitBalance
import Transformer.Grokking.CircuitEfficiency.SectionC_NativeDescent
import Mathlib.Topology.Order.MonotoneConvergence

/-!
# Unbounded native training score from actual positive source seeds

Sources: Varma et al., arXiv:2309.02390v1, appendix C's product CE
and one-zero-factor initialization; retained native AdamW/clipping at
lab commit a3cc86c. At zero parameter decay, the actual formed factors
increase. A hypothetical bounded score would bound every coordinate
using its positive first-formed partner, giving finite monotone limits.
The actual native limit law would force those positive limits to zero,
a contradiction. Thus the current training score tends to infinity.

No parameter or gradient convergence is supplied for this path. All
four moments and clocks are retained, with the actual CE callbacks and
shared clipping. The conclusion holds at any constant positive rate
in this particular sign region; it does not analyze positive decay,
learned stochastic GPTMini features or floating-point kernels. The
paper's GD/coupled efficiency penalty is not substituted for AdamW.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW

/-- A present factor times its partner is no larger than the total
training score in the formed positive region. Source: appendix C,
sim-overall-logits; both actual product contributions are retained. -/
theorem native_positive_partner_product_le_score (state : NativeSubweightState) (i : Fin 4)
    (hs : PositiveNativeState state) :
    (state i).parameter * (state (nativeFactorPartner i)).parameter ≤ nativeTotalScore state := by
  have hg := mul_nonneg (le_of_lt (hs 0).1) (le_of_lt (hs 1).1)
  have hm := mul_nonneg (le_of_lt (hs 2).1) (le_of_lt (hs 3).1)
  fin_cases i <;> norm_num [nativeFactorPartner, nativeTotalScore] <;> nlinarith

example : PositiveNativeState (seededNativeSubweights ((1, 1), (1, 1))) := by
  intro i
  fin_cases i <;> exact seeded_scalar_positive _ (by norm_num)

/-- Every actual coordinate strictly grows after source-seed
formation at zero decay. Sources: appendix C positive second seeds
and native AdamW at a3cc86c; neither partners nor moments are frozen. -/
theorem native_no_decay_source_parameter_strictMono (remaining : ℕ)
    (bound b1 b2 eps rate genSeed memSeed : ℝ) (i : Fin 4)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hg : 0 < genSeed) (hm : 0 < memSeed) :
    StrictMono (fun n => (nativeSubweightPath remaining bound b1 b2 eps 0 rate
      (seededNativeSubweights ((0, genSeed), (0, memSeed))) (n + 1) i).parameter) := by
  apply strictMono_nat_of_lt_succ
  intro n
  exact native_no_decay_parameter_increase remaining bound b1 b2 eps rate _ i hclip hb1 h1 he heta
    (native_one_zero_seed_path_positive remaining bound b1 b2 eps 0 rate genSeed memSeed n
      hclip hb1 h1 hb2 h2 he heta (by norm_num) hg hm)

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) < 1 / 200 ∧ (0 : ℝ) < 1 := by norm_num

/-- A bounded actual native score would give positive finite
coordinate limits, contradicting the closed no-decay limit law.
Sources: appendix C's product CE and native AdamW at a3cc86c;
convergence is derived only under the hypothetical bounded score. -/
theorem native_no_decay_source_score_unbounded (remaining : ℕ)
    (bound b1 b2 eps rate genSeed memSeed : ℝ)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hg : 0 < genSeed) (hm : 0 < memSeed) :
    ¬BddAbove (Set.range (fun n => nativeTotalScore (nativeSubweightPath remaining
      bound b1 b2 eps 0 rate (seededNativeSubweights ((0, genSeed), (0, memSeed))) (n + 1)))) := by
  let state := fun n => nativeSubweightPath remaining bound b1 b2 eps 0 rate
    (seededNativeSubweights ((0, genSeed), (0, memSeed))) (n + 1)
  have hs : ∀ n, PositiveNativeState (state n) := fun n =>
    native_one_zero_seed_path_positive remaining bound b1 b2 eps 0 rate genSeed memSeed n
      hclip hb1 h1 hb2 h2 he heta (by norm_num) hg hm
  have hmono : ∀ i, Monotone (fun n => (state n i).parameter) := fun i =>
    (native_no_decay_source_parameter_strictMono remaining bound b1 b2 eps rate genSeed memSeed i
      hclip hb1 h1 hb2 h2 he heta hg hm).monotone
  rintro ⟨boundScore, hboundScore⟩
  have hbounded : ∀ i, BddAbove (Set.range (fun n => (state n i).parameter)) := by
    intro i
    refine ⟨boundScore / (state 0 (nativeFactorPartner i)).parameter, ?_⟩
    rintro _ ⟨n, rfl⟩
    apply (le_div_iff₀ (hs 0 (nativeFactorPartner i)).1).mpr
    calc
      (state n i).parameter * (state 0 (nativeFactorPartner i)).parameter ≤
          (state n i).parameter * (state n (nativeFactorPartner i)).parameter :=
        mul_le_mul_of_nonneg_left (hmono _ (Nat.zero_le n)) (le_of_lt (hs n i).1)
      _ ≤ nativeTotalScore (state n) := native_positive_partner_product_le_score (state n) i (hs n)
      _ ≤ boundScore := hboundScore (Set.mem_range_self n)
  let reference : NativeSubweightState := fun i => seededScalarState (⨆ n, (state n i).parameter)
  have hp : ∀ i, Tendsto (fun n => (state n i).parameter) atTop (nhds (reference i).parameter) := by
    intro i
    exact tendsto_atTop_ciSup (hmono i) (hbounded i)
  have hzero := native_no_decay_finite_limit_zero remaining bound b1 b2 eps rate state reference
    (fun n => rfl) hclip hb1 h1 hb2 h2 he heta hp
  have hpositive : 0 < (reference 0).parameter :=
    lt_of_lt_of_le (hs 0 0).1 (le_ciSup (hbounded 0) 0)
  rw [hzero 0] at hpositive
  exact lt_irrefl 0 hpositive

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) < 1 / 200 ∧ (0 : ℝ) < 1 := by norm_num

/-- The full current training-score path tends to positive infinity,
including the finite initialization prefix. Sources: appendix C's
source seeds and native AdamW at a3cc86c; this is a dynamics result,
not an assumed divergent successful gradient or logit stream. -/
theorem native_no_decay_source_score_tendsto_atTop (remaining : ℕ)
    (bound b1 b2 eps rate genSeed memSeed : ℝ)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hg : 0 < genSeed) (hm : 0 < memSeed) :
    Tendsto (fun n => nativeTotalScore (nativeSubweightPath remaining bound b1 b2 eps 0 rate
      (seededNativeSubweights ((0, genSeed), (0, memSeed))) n)) atTop atTop := by
  apply (tendsto_add_atTop_iff_nat 1).mp
  have hmono : Monotone (fun n => nativeTotalScore (nativeSubweightPath remaining
      bound b1 b2 eps 0 rate (seededNativeSubweights ((0, genSeed), (0, memSeed))) (n + 1))) := by
    apply StrictMono.monotone
    apply strictMono_nat_of_lt_succ
    intro n
    exact native_no_decay_score_increase remaining bound b1 b2 eps rate _ hclip hb1 h1 he heta
      (native_one_zero_seed_path_positive remaining bound b1 b2 eps 0 rate genSeed memSeed n
        hclip hb1 h1 hb2 h2 he heta (by norm_num) hg hm)
  apply hmono.tendsto_atTop_atTop
  intro score
  obtain ⟨_, ⟨n, rfl⟩, hn⟩ := (not_bddAbove_iff.mp
    (native_no_decay_source_score_unbounded remaining bound b1 b2 eps rate genSeed memSeed
      hclip hb1 h1 hb2 h2 he heta hg hm)) score
  exact ⟨n, le_of_lt hn⟩

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) < 1 / 200 ∧ (0 : ℝ) < 1 := by norm_num

/-- These actual positive-seed no-decay trajectories have no finite
four-parameter limit. Sources: appendix C's product CE and native
AdamW at a3cc86c; finite-point continuity conflicts with the derived
unbounded score, rather than with an assumed buffer or clock limit. -/
theorem native_no_decay_source_no_finite_parameter_limit (remaining : ℕ)
    (bound b1 b2 eps rate genSeed memSeed : ℝ) (reference : NativeSubweightState)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hg : 0 < genSeed) (hm : 0 < memSeed) :
    ¬(∀ i, Tendsto (fun n => (nativeSubweightPath remaining bound b1 b2 eps 0 rate
      (seededNativeSubweights ((0, genSeed), (0, memSeed))) n i).parameter)
      atTop (nhds (reference i).parameter)) := by
  intro hp
  have hs := native_no_decay_source_score_tendsto_atTop remaining bound b1 b2 eps rate genSeed memSeed
    hclip hb1 h1 hb2 h2 he heta hg hm
  exact not_tendsto_nhds_of_tendsto_atTop hs (nativeTotalScore reference)
    (native_total_score_tendsto _ reference hp)

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) < 1 / 200 ∧ (0 : ℝ) < 1 := by norm_num

end Transformer.Grokking.CircuitEfficiency
