import Transformer.Grokking.CircuitEfficiency.SectionC_NativeSymmetry
import Mathlib.Analysis.Calculus.Deriv.MeanValue

/-!
# Strict native training-CE descent without eventual rule selection

Sources: Varma et al., arXiv:2309.02390v1, appendix C's product CE
and Gen/Mem tables; native AdamW/clipping at lab commit 2c052c9.
In the formed positive-factor/nonpositive-moment region, zero parameter
decay makes every coordinate increase at any positive rate. Both actual
products and the total training score increase; actual multiclass CE
strictly decreases. No small-rate assumption or moment reset is needed
for this particular monotone positive-factor CE model.

Equal positive second seeds give correct training decisions after the
first update and decreasing train CE thereafter, but held-out CE never
falls below log two. This refutes automatic eventual rule selection from
component formation and continued train-loss decrease. It does not prove
train CE tends to zero, handle positive decay, or contradict the paper's
cost-asymmetric GD penalty. Learned GPTMini/finite-precision transfer
remains open; admissible beta2 is explicit for the actual path result.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Actual unpenalized multiclass CE is strictly decreasing in total
training score. Source: appendix C train CE; its globally negative
actual derivative is already checked, including all q-1 competitors. -/
theorem table_train_ce_strictAnti (remaining : ℕ) : StrictAnti (tableTrainCE remaining) := by
  apply strictAnti_of_deriv_neg
  intro score
  rw [(table_train_ce_deriv remaining score).deriv]
  exact table_train_ce_slope_negative remaining score

/-- Without parameter decay, an actual positive factor strictly grows
from its current CE partial and retained nonpositive moment. Sources:
appendix C product CE and native AdamW at 2c052c9. The inequality needs
no beta2 restriction; physical variance on paths is checked separately. -/
theorem native_no_decay_parameter_increase (remaining : ℕ) (bound b1 b2 eps rate : ℝ)
    (state : NativeSubweightState) (i : Fin 4) (hclip : 0 < bound)
    (hb : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hs : PositiveNativeState state) :
    (state i).parameter < (nativeSubweightStep remaining bound b1 b2 eps 0 rate state i).parameter := by
  have hg := native_applied_gradient_negative remaining bound state i hclip (hs _).1
  have hf := scalar_strict_parameter_decay_floor b1 b2 eps 0 rate _ (state i)
    hb h1 he heta (hs i).2.1 hg
  simpa only [nativeSubweightStep, mul_zero, sub_zero, one_mul] using hf

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) < 1 / 1000 ∧
    PositiveNativeState (seededNativeSubweights ((1, 1 / 200), (1, 1))) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  fin_cases i <;> exact seeded_scalar_positive _ (by norm_num)

/-- Both actual product logits strictly increase in this zero-decay
region. Source: appendix C, sim-overall-logits; simultaneous current
CE updates of all four factors are used, rather than frozen partners. -/
theorem native_no_decay_products_increase (remaining : ℕ) (bound b1 b2 eps rate : ℝ)
    (state : NativeSubweightState) (hclip : 0 < bound)
    (hb : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hs : PositiveNativeState state) :
    let next := nativeSubweightStep remaining bound b1 b2 eps 0 rate state
    (state 0).parameter * (state 1).parameter < (next 0).parameter * (next 1).parameter ∧
      (state 2).parameter * (state 3).parameter < (next 2).parameter * (next 3).parameter := by
  have h0 := native_no_decay_parameter_increase remaining bound b1 b2 eps rate state 0 hclip hb h1 he heta hs
  have h1' := native_no_decay_parameter_increase remaining bound b1 b2 eps rate state 1 hclip hb h1 he heta hs
  have h2 := native_no_decay_parameter_increase remaining bound b1 b2 eps rate state 2 hclip hb h1 he heta hs
  have h3 := native_no_decay_parameter_increase remaining bound b1 b2 eps rate state 3 hclip hb h1 he heta hs
  exact ⟨mul_lt_mul_of_pos h0 h1' (hs 0).1 (lt_trans (hs 1).1 h1'),
    mul_lt_mul_of_pos h2 h3 (hs 2).1 (lt_trans (hs 3).1 h3)⟩

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) < 1 / 1000 ∧
    PositiveNativeState (seededNativeSubweights ((1, 1 / 200), (1, 1))) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  fin_cases i <;> exact seeded_scalar_positive _ (by norm_num)

/-- Both growing products raise the actual correct training score.
Source: appendix C train logits and the closed simultaneous native
update at 2c052c9; zero parameter decay is essential to this estimate. -/
theorem native_no_decay_score_increase (remaining : ℕ) (bound b1 b2 eps rate : ℝ)
    (state : NativeSubweightState) (hclip : 0 < bound)
    (hb : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hs : PositiveNativeState state) :
    nativeTotalScore state < nativeTotalScore (nativeSubweightStep remaining bound b1 b2 eps 0 rate state) := by
  have hp := native_no_decay_products_increase remaining bound b1 b2 eps rate state hclip hb h1 he heta hs
  unfold nativeTotalScore
  linarith [hp.1, hp.2]

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) < 1 / 1000 ∧
    PositiveNativeState (seededNativeSubweights ((1, 1 / 200), (1, 1))) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  fin_cases i <;> exact seeded_scalar_positive _ (by norm_num)

/-- Actual training CE strictly drops after a zero-decay native step
in this formed-factor region. Sources: appendix C CE and native AdamW
at 2c052c9; this conditional descent is not a universal GPTMini claim. -/
theorem native_no_decay_train_ce_decrease (remaining : ℕ) (bound b1 b2 eps rate : ℝ)
    (state : NativeSubweightState) (hclip : 0 < bound)
    (hb : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps) (heta : 0 < rate)
    (hs : PositiveNativeState state) :
    tableTrainCE remaining (nativeTotalScore (nativeSubweightStep remaining bound b1 b2 eps 0 rate state)) <
      tableTrainCE remaining (nativeTotalScore state) := by
  exact table_train_ce_strictAnti remaining
    (native_no_decay_score_increase remaining bound b1 b2 eps rate state hclip hb h1 he heta hs)

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) < 1 / 1000 ∧
    PositiveNativeState (seededNativeSubweights ((1, 1 / 200), (1, 1))) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  intro i
  fin_cases i <;> exact seeded_scalar_positive _ (by norm_num)

/-- Train CE decreases between any two finite clocks after initial
formation on the actual zero-decay path. Sources: appendix C positive
second seeds and native AdamW at 2c052c9; no limit value is inferred. -/
theorem native_no_decay_path_train_ce_strictAnti (remaining : ℕ)
    (bound b1 b2 eps rate genSeed memSeed : ℝ)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hg : 0 < genSeed) (hm : 0 < memSeed) :
    StrictAnti (fun n => tableTrainCE remaining (nativeTotalScore (nativeSubweightPath remaining
      bound b1 b2 eps 0 rate (seededNativeSubweights ((0, genSeed), (0, memSeed))) (n + 1)))) := by
  apply strictAnti_nat_of_succ_lt
  intro n
  have hs := native_one_zero_seed_path_positive remaining bound b1 b2 eps 0 rate genSeed memSeed n
    hclip hb1 h1 hb2 h2 he heta (by norm_num) hg hm
  exact native_no_decay_train_ce_decrease remaining bound b1 b2 eps rate _ hclip hb1 h1 he heta hs

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) < 1 / 200 ∧ (0 : ℝ) < 1 := by norm_num

/-- Perfect formed training decisions and continuing CE decrease can
coexist with permanent held-out CE at least log two. Sources: appendix C
equal-speed seeds and native zero-decay path at 2c052c9; loss improvement
alone is insufficient for eventual Gen/Mem symmetry breaking. -/
theorem native_symmetric_train_descent_with_test_ce_floor (remaining : ℕ)
    (bound b1 b2 eps rate seed : ℝ)
    (hclip : 0 < bound) (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 < rate) (hseed : 0 < seed) :
    (∀ n : ℕ,
      let state := nativeSubweightPath remaining bound b1 b2 eps 0 rate
        (seededNativeSubweights ((0, seed), (0, seed))) (n + 1)
      Transformer.Grokking.NaiveLoss.StrictCorrect (trainTableLogits remaining
        ((state 0).parameter * (state 1).parameter) ((state 2).parameter * (state 3).parameter)) 0) ∧
    StrictAnti (fun n => tableTrainCE remaining (nativeTotalScore (nativeSubweightPath remaining
      bound b1 b2 eps 0 rate (seededNativeSubweights ((0, seed), (0, seed))) (n + 1)))) ∧
    (∀ n : ℕ,
      let state := nativeSubweightPath remaining bound b1 b2 eps 0 rate
        (seededNativeSubweights ((0, seed), (0, seed))) n
      Real.log 2 ≤ Transformer.Grokking.NaiveLoss.crossEntropy (heldoutTableLogits remaining
        ((state 0).parameter * (state 1).parameter) ((state 2).parameter * (state 3).parameter)) 0) := by
  refine ⟨?_, native_no_decay_path_train_ce_strictAnti remaining bound b1 b2 eps rate seed seed
    hclip hb1 h1 hb2 h2 he heta hseed hseed, ?_⟩
  · intro n
    exact native_one_zero_seed_path_train_correct remaining bound b1 b2 eps 0 rate seed seed n
      hclip hb1 h1 hb2 h2 he heta (by norm_num) hseed hseed
  · intro n
    exact native_equal_seed_path_heldout_ce_floor remaining bound b1 b2 eps 0 rate 0 seed n

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) < 1 := by norm_num

end Transformer.Grokking.CircuitEfficiency
