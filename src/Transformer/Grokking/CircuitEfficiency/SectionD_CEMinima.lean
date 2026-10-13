import Transformer.Grokking.CircuitEfficiency.SectionD_CEExistence
import Transformer.Grokking.CircuitEfficiency.SectionD_SuperlinearAllocation

/-!
# Correct fixed-table decisions at attained multiclass CE minima

Source: Varma et al., arXiv:2309.02390v1, appendix D, Theorem
efficiency-circuit-weights-logits case 2, and appendix C's Gen/Mem
tables. Actual CE rules out the zero minimum. The now-proved attainment
and marginal-cost balance give positive weights and the source's ratio.
When Gen has strictly smaller normalized cost, every nonnegative global
minimum selects the correct held-out class among all remaining+2 labels.

The fixed circuit tables, norm-scaling model and coupled penalty remain
explicit. This establishes their equilibrium prediction, including the
113-class simulation, rather than convergence or circuit discovery by
native AdamW. No delayed event or training time is assumed or concluded.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- The actual finite-class CE budget cannot have a zero-sum nonnegative
minimum for r>1. Source: arXiv:2309.02390v1, appendix D case 2;
appendix C's CE removes the earlier input nonzero-sum premise. -/
theorem table_budget_minimizer_nonzero (remaining : ℕ)
    (multiplier firstCost secondCost exponent x y : ℝ) (hr : 1 < exponent)
    (hx : 0 ≤ x) (hy : 0 ≤ y)
    (hmin : ∀ u v : ℝ, 0 ≤ u → 0 ≤ v →
      tableBudgetLoss remaining multiplier firstCost secondCost exponent x y ≤
        tableBudgetLoss remaining multiplier firstCost secondCost exponent u v) :
    0 < x + y := by
  by_contra hn
  have hz : x = 0 := by linarith
  have hw : y = 0 := by linarith
  subst x
  subst y
  obtain ⟨epsilon, he, hi⟩ := table_budget_origin_improvable
    remaining multiplier firstCost secondCost exponent hr
  have hm := hmin epsilon 0 he.le (by norm_num)
  linarith

example : ∃ x y : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 < x + y := by
  obtain ⟨x, y, hx, hy, hm⟩ := source_113_class_effective_minimum_exists
  exact ⟨x, y, hx, hy, table_budget_minimizer_nonzero
    111 (1 / 400) 1 4 (5 / 3) x y (by norm_num) hx hy hm⟩

/-- Every nonnegative minimum of actual CE uses both positive-cost
circuits when r>1. Source: arXiv:2309.02390v1, appendix D case 2;
positivity is derived from actual feasible improvements, without a
nonzero minimum hypothesis or an assigned stationary equation. -/
theorem table_budget_minimizer_positive_weights (remaining : ℕ)
    (multiplier firstCost secondCost exponent x y : ℝ)
    (hl : 0 < multiplier) (h0 : 0 < firstCost) (h1 : 0 < secondCost) (hr : 1 < exponent)
    (hx : 0 ≤ x) (hy : 0 ≤ y)
    (hmin : ∀ u v : ℝ, 0 ≤ u → 0 ≤ v →
      tableBudgetLoss remaining multiplier firstCost secondCost exponent x y ≤
        tableBudgetLoss remaining multiplier firstCost secondCost exponent u v) :
    0 < x ∧ 0 < y := by
  have hs := table_budget_minimizer_nonzero remaining multiplier firstCost secondCost exponent x y hr hx hy hmin
  exact superlinear_minimizer_positive_weights (tableTrainCE remaining)
    multiplier firstCost secondCost exponent x y hl h0 h1 hr hx hy hs hmin

example : ∃ x y : ℝ, 0 < x ∧ 0 < y := by
  obtain ⟨x, y, hx, hy, hm⟩ := source_113_class_effective_minimum_exists
  have hp := table_budget_minimizer_positive_weights 111 (1 / 400) 1 4 (5 / 3) x y
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) hx hy hm
  exact ⟨x, y, hp⟩

/-- Actual CE minima obey the marginal-cost equality. Source:
arXiv:2309.02390v1, appendix D case 2; Fermat's theorem applies to
the derived interior allocation, rather than presuming the equality. -/
theorem table_budget_minimizer_cost_balance (remaining : ℕ)
    (multiplier firstCost secondCost exponent x y : ℝ)
    (hl : 0 < multiplier) (h0 : 0 < firstCost) (h1 : 0 < secondCost) (hr : 1 < exponent)
    (hx : 0 ≤ x) (hy : 0 ≤ y)
    (hmin : ∀ u v : ℝ, 0 ≤ u → 0 ≤ v →
      tableBudgetLoss remaining multiplier firstCost secondCost exponent x y ≤
        tableBudgetLoss remaining multiplier firstCost secondCost exponent u v) :
    firstCost * x ^ (exponent - 1) = secondCost * y ^ (exponent - 1) := by
  have hs := table_budget_minimizer_nonzero remaining multiplier firstCost secondCost exponent x y hr hx hy hmin
  exact superlinear_minimizer_cost_balance (tableTrainCE remaining)
    multiplier firstCost secondCost exponent x y hl h0 h1 hr hx hy hs hmin

example : ∃ x y : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ x ^ ((5 / 3 : ℝ) - 1) = 4 * y ^ ((5 / 3 : ℝ) - 1) := by
  obtain ⟨x, y, hx, hy, hm⟩ := source_113_class_effective_minimum_exists
  have hb := table_budget_minimizer_cost_balance 111 (1 / 400) 1 4 (5 / 3) x y
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) hx hy hm
  exact ⟨x, y, hx, hy, by simpa only [one_mul] using hb⟩

/-- The source's inverse-cost ratio holds at every attained actual CE
minimum. Source: arXiv:2309.02390v1, appendix D, final case-2 formula. -/
theorem table_budget_minimizer_weight_ratio (remaining : ℕ)
    (multiplier firstCost secondCost exponent x y : ℝ)
    (hl : 0 < multiplier) (h0 : 0 < firstCost) (h1 : 0 < secondCost) (hr : 1 < exponent)
    (hx : 0 ≤ x) (hy : 0 ≤ y)
    (hmin : ∀ u v : ℝ, 0 ≤ u → 0 ≤ v →
      tableBudgetLoss remaining multiplier firstCost secondCost exponent x y ≤
        tableBudgetLoss remaining multiplier firstCost secondCost exponent u v) :
    x / y = (secondCost / firstCost) ^ ((exponent - 1)⁻¹) := by
  have hs := table_budget_minimizer_nonzero remaining multiplier firstCost secondCost exponent x y hr hx hy hmin
  exact superlinear_minimizer_weight_ratio (tableTrainCE remaining)
    multiplier firstCost secondCost exponent x y hl h0 h1 hr hx hy hs hmin

example : ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x / y = (4 : ℝ) ^ (((5 / 3 : ℝ) - 1)⁻¹) := by
  obtain ⟨x, y, hx, hy, hm⟩ := source_113_class_effective_minimum_exists
  have hp := table_budget_minimizer_positive_weights 111 (1 / 400) 1 4 (5 / 3) x y
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) hx hy hm
  have hr := table_budget_minimizer_weight_ratio 111 (1 / 400) 1 4 (5 / 3) x y
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) hx hy hm
  exact ⟨x, y, hp.1, hp.2, by simpa only [div_one] using hr⟩

/-- A strictly more efficient Gen circuit makes every actual CE minimum
correct on the full train and test logit tables. Source: arXiv:2309.02390v1,
section 3's efficiency mechanism, appendix C tables and appendix D case 2. -/
theorem table_budget_minimizer_train_and_test_correct (remaining : ℕ)
    (multiplier firstCost secondCost exponent x y : ℝ)
    (hl : 0 < multiplier) (h0 : 0 < firstCost) (heff : firstCost < secondCost) (hr : 1 < exponent)
    (hx : 0 ≤ x) (hy : 0 ≤ y)
    (hmin : ∀ u v : ℝ, 0 ≤ u → 0 ≤ v →
      tableBudgetLoss remaining multiplier firstCost secondCost exponent x y ≤
        tableBudgetLoss remaining multiplier firstCost secondCost exponent u v) :
    Transformer.Grokking.NaiveLoss.StrictCorrect (trainTableLogits remaining x y) 0 ∧
      Transformer.Grokking.NaiveLoss.StrictCorrect (heldoutTableLogits remaining x y) 0 := by
  have h1 : 0 < secondCost := h0.trans heff
  have hp := table_budget_minimizer_positive_weights remaining multiplier firstCost secondCost
    exponent x y hl h0 h1 hr hx hy hmin
  have hb := table_budget_minimizer_cost_balance remaining multiplier firstCost secondCost
    exponent x y hl h0 h1 hr hx hy hmin
  have hm := cost_balance_positive_heldout_margin firstCost secondCost exponent x y h0 heff hr hx hp.2 hb
  refine ⟨(train_table_strict_correct_iff remaining x y).mpr (by linarith), ?_⟩
  exact heldout_table_strict_correct remaining x y hp.1 (by linarith)

example : ∃ x y : ℝ,
    Transformer.Grokking.NaiveLoss.StrictCorrect (trainTableLogits 111 x y) 0 ∧
    Transformer.Grokking.NaiveLoss.StrictCorrect (heldoutTableLogits 111 x y) 0 := by
  obtain ⟨x, y, hx, hy, hm⟩ := source_113_class_effective_minimum_exists
  exact ⟨x, y, table_budget_minimizer_train_and_test_correct 111 (1 / 400) 1 4 (5 / 3) x y
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) hx hy hm⟩

/-- Attainment and correct decisions are jointly realized with the
source's class count, normalized norms and scaling exponent. Source:
arXiv:2309.02390v1, appendix C's Table and appendix D's alpha/p
normalization; this does not claim the simulation's delayed trajectory. -/
theorem source_113_class_correct_minimum_exists :
    ∃ x y : ℝ, 0 < x ∧ 0 < y ∧
      (∀ u v : ℝ, 0 ≤ u → 0 ≤ v →
        tableBudgetLoss 111 (1 / 400) 1 4 (5 / 3) x y ≤
          tableBudgetLoss 111 (1 / 400) 1 4 (5 / 3) u v) ∧
      Transformer.Grokking.NaiveLoss.StrictCorrect (trainTableLogits 111 x y) 0 ∧
      Transformer.Grokking.NaiveLoss.StrictCorrect (heldoutTableLogits 111 x y) 0 := by
  obtain ⟨x, y, hx, hy, hm⟩ := source_113_class_effective_minimum_exists
  have hp := table_budget_minimizer_positive_weights 111 (1 / 400) 1 4 (5 / 3) x y
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) hx hy hm
  have hc := table_budget_minimizer_train_and_test_correct 111 (1 / 400) 1 4 (5 / 3) x y
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) hx hy hm
  exact ⟨x, y, hp.1, hp.2, hm, hc⟩

end Transformer.Grokking.CircuitEfficiency
