import Transformer.Grokking.CircuitEfficiency.SectionC_NativeScoreGrowth
import Mathlib.Order.Filter.AtTopBot.Archimedean

/-!
# Actual native train CE can vanish while held-out CE stays positive

Sources: Varma et al., arXiv:2309.02390v1, appendix C's train/test
CE tables and product seeds; retained native AdamW/clipping at lab
commit 7681b4f. Normalize the actual train CE using exp(-score), then
transfer the derived no-decay score growth to train CE tending to zero.
Equal Gen/Mem source seeds keep their full-state symmetry under the
actual retained update, so true held-out CE is always at least log two
and cannot tend to zero. The correct held-out class also remains tied
with the actual wrong Mem class, preventing a unique correct argmax.

This is an actual-feedback counterexample to automatic rule selection
from arbitrarily good train confidence and persistent native updates.
It uses plain CE, zero decay and fixed tables, rather than appendix C's
cost-asymmetric coupled penalty/GD. No stochastic learned GPTMini or
finite-precision transfer, onset time or universal grokking detector is
claimed. Tie handling is not replaced by an assigned accuracy value.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.AdamW
open scoped BigOperators

/-- Normalize true multiclass train CE without discarding its q-1
competitors. Source: appendix C train-loss formula, q=remaining+2;
this identity is valid at every finite real score. -/
theorem table_train_ce_exp_neg (remaining : ℕ) (score : ℝ) :
    tableTrainCE remaining score = Real.log (1 + ((remaining : ℝ) + 1) * Real.exp (-score)) := by
  have hf : Real.exp score + (remaining : ℝ) + 1 =
      Real.exp score * (1 + ((remaining : ℝ) + 1) * Real.exp (-score)) := by
    rw [Real.exp_neg]
    field_simp
    ring
  have hp : 0 < 1 + ((remaining : ℝ) + 1) * Real.exp (-score) := by positivity
  rw [table_train_ce_formula, hf, Real.log_mul (Real.exp_ne_zero score) (ne_of_gt hp), Real.log_exp]
  ring

/-- True finite-class train CE tends to zero when its total correct
score tends to infinity. Source: appendix C train CE; the dynamics
application below must derive this score limit from actual updates. -/
theorem table_train_ce_tendsto_zero (remaining : ℕ) (score : ℕ → ℝ)
    (hs : Tendsto score atTop atTop) :
    Tendsto (fun n => tableTrainCE remaining (score n)) atTop (nhds 0) := by
  have he : Tendsto (fun n => Real.exp (-score n)) atTop (nhds 0) := by
    simpa only [Function.comp_def] using Real.tendsto_exp_neg_atTop_nhds_zero.comp hs
  have hd : Tendsto (fun n => 1 + ((remaining : ℝ) + 1) * Real.exp (-score n)) atTop (nhds 1) := by
    simpa only [mul_zero, add_zero] using (he.const_mul ((remaining : ℝ) + 1)).const_add 1
  simpa only [table_train_ce_exp_neg, Real.log_one] using hd.log (by norm_num : (1 : ℝ) ≠ 0)

example : Tendsto (fun n : ℕ => (n : ℝ)) atTop atTop := tendsto_natCast_atTop_atTop

/-- Actual retained native CE tends to zero on all positive second-
seed no-decay paths. Sources: appendix C product seeds and native
AdamW at 7681b4f; no future gradient, parameter or score convergence
premise is supplied, and moments and clocks are never reset. -/
theorem native_no_decay_source_train_ce_tendsto_zero (remaining : ℕ)
    (bound b1 b2 eps rate genSeed memSeed : ℝ)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hg : 0 < genSeed) (hm : 0 < memSeed) :
    Tendsto (fun n => tableTrainCE remaining (nativeTotalScore (nativeSubweightPath remaining
      bound b1 b2 eps 0 rate (seededNativeSubweights ((0, genSeed), (0, memSeed))) n))) atTop (nhds 0) := by
  exact table_train_ce_tendsto_zero remaining _
    (native_no_decay_source_score_tendsto_atTop remaining bound b1 b2 eps rate genSeed memSeed
      hclip hb1 h1 hb2 h2 he heta hg hm)

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) < 1 / 200 ∧ (0 : ℝ) < 1 := by norm_num

/-- Equal actual native Gen/Mem seeds never yield a zero held-out CE
limit. Sources: appendix C test table and retained native symmetry at
7681b4f; the exact log-two floor is independent of optimizer settings. -/
theorem native_equal_seed_path_no_heldout_zero (remaining : ℕ)
    (bound b1 b2 eps decay rate first second : ℝ) :
    ¬Tendsto (fun n =>
      let state := nativeSubweightPath remaining bound b1 b2 eps decay rate
        (seededNativeSubweights ((first, second), (first, second))) n
      Transformer.Grokking.NaiveLoss.crossEntropy (heldoutTableLogits remaining
        ((state 0).parameter * (state 1).parameter) ((state 2).parameter * (state 3).parameter)) 0)
      atTop (nhds 0) := by
  intro ht
  have hf : Real.log 2 ≤ 0 := ge_of_tendsto ht
    (Eventually.of_forall (native_equal_seed_path_heldout_ce_floor remaining bound b1 b2 eps decay rate first second))
  have hp : 0 < Real.log 2 := Real.log_pos (by norm_num)
  linarith

/-- Equal actual circuit states retain a wrong held-out class tied
with the correct class at every native clock. Sources: appendix C
test logits and native symmetry at 7681b4f; this excludes unique
correct argmax without assigning an arbitrary tie-breaking accuracy. -/
theorem native_equal_seed_path_not_strict_heldout_correct (remaining : ℕ)
    (bound b1 b2 eps decay rate first second : ℝ) (n : ℕ) :
    let state := nativeSubweightPath remaining bound b1 b2 eps decay rate
      (seededNativeSubweights ((first, second), (first, second))) n
    ¬Transformer.Grokking.NaiveLoss.StrictCorrect (heldoutTableLogits remaining
      ((state 0).parameter * (state 1).parameter) ((state 2).parameter * (state 3).parameter)) 0 := by
  dsimp only
  intro hc
  have hw := hc (0 : Fin (remaining + 1)).succ (Fin.succ_ne_zero _)
  have hs := native_symmetric_products _
    (native_equal_seed_path_symmetry remaining bound b1 b2 eps decay rate first second n)
  simp only [heldoutTableLogits, Fin.succ_ne_zero, ite_false, ite_true] at hw
  rw [hs] at hw
  exact lt_irrefl _ hw

/-- Arbitrarily good actual train confidence coexists with permanent
held-out loss and decision obstructions on a valid native path.
Sources: appendix C equal-speed seeds and native AdamW at 7681b4f;
this strengthens continued descent to a derived zero training-CE limit. -/
theorem native_symmetric_train_zero_with_test_obstruction (remaining : ℕ)
    (bound b1 b2 eps rate seed : ℝ)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hseed : 0 < seed) :
    Tendsto (fun n => tableTrainCE remaining (nativeTotalScore (nativeSubweightPath remaining
      bound b1 b2 eps 0 rate (seededNativeSubweights ((0, seed), (0, seed))) n))) atTop (nhds 0) ∧
    (∀ n : ℕ,
      let state := nativeSubweightPath remaining bound b1 b2 eps 0 rate
        (seededNativeSubweights ((0, seed), (0, seed))) n
      Real.log 2 ≤ Transformer.Grokking.NaiveLoss.crossEntropy (heldoutTableLogits remaining
        ((state 0).parameter * (state 1).parameter) ((state 2).parameter * (state 3).parameter)) 0 ∧
      ¬Transformer.Grokking.NaiveLoss.StrictCorrect (heldoutTableLogits remaining
        ((state 0).parameter * (state 1).parameter) ((state 2).parameter * (state 3).parameter)) 0) := by
  refine ⟨native_no_decay_source_train_ce_tendsto_zero remaining bound b1 b2 eps rate seed seed
    hclip hb1 h1 hb2 h2 he heta hseed hseed, ?_⟩
  intro n
  exact ⟨native_equal_seed_path_heldout_ce_floor remaining bound b1 b2 eps 0 rate 0 seed n,
    native_equal_seed_path_not_strict_heldout_correct remaining bound b1 b2 eps 0 rate 0 seed n⟩

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) < 1 := by norm_num

/-- Equal Gen/Mem test logits cap actual softmax confidence in the
correct class at one half. Source: appendix C's test-probability
formula; the denominator is the sum over all remaining+2 classes,
not an assigned binary surrogate. This quantifies the loss floor. -/
theorem table_equal_weights_target_probability_bound (remaining : ℕ) (weight : ℝ) :
    Real.exp (heldoutTableLogits remaining weight weight 0) /
      (∑ k, Real.exp (heldoutTableLogits remaining weight weight k)) ≤ 1 / 2 := by
  rw [heldout_table_exp_sum]
  simp only [heldoutTableLogits, ite_true]
  apply (div_le_iff₀ (by positivity :
    0 < Real.exp weight + Real.exp weight + (remaining : ℝ))).mpr
  have hr : 0 ≤ (remaining : ℝ) := by positivity
  linarith

end Transformer.Grokking.CircuitEfficiency
