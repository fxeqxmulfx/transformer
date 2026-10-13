import Transformer.Grokking.CircuitEfficiency.SectionD_CEBudgetBasic
import Mathlib.Topology.Order.Compact

/-!
# Attainment of the actual multiclass CE budget

Source: Varma et al., arXiv:2309.02390v1, appendices C and D. The
source's allocation theorem refers to a global minimum. For actual
finite-class CE, positive multiplier and positive costs, prove that a
nonnegative global minimum exists when the power exponent is at least
one. Continuity and explicit growing cost bounds reduce the unbounded
quadrant to a compact rectangle containing the origin.

This supplies minimizer attainment for the stated fixed-table objective.
It does not assert that gradient descent on product subweights, or
native AdamW on GPTMini, reaches any global minimum. No finite training
time or delayed-generalization event is encoded in the objective.
The source-specific example uses appendix D's multiplier alpha/p:
0.005/2=1/400. Appendix C's displayed LossWD omits this factor; the
example is therefore not identified with an exact simulator normalization.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- A sufficiently expensive first weight costs more than the all-tied
origin. Source: arXiv:2309.02390v1, appendix D, attainment extension
for appendix C's actual CE; the positive multiplier/cost are explicit. -/
theorem table_budget_first_large (remaining : ℕ)
    (multiplier firstCost secondCost exponent x y : ℝ)
    (hl : 0 < multiplier) (h0 : 0 < firstCost) (h1 : 0 ≤ secondCost)
    (hr : 1 ≤ exponent) (hx : 1 ≤ x) (hy : 0 ≤ y)
    (hb : Real.log ((remaining : ℝ) + 2) < multiplier * firstCost * x) :
    tableBudgetLoss remaining multiplier firstCost secondCost exponent 0 0 <
      tableBudgetLoss remaining multiplier firstCost secondCost exponent x y := by
  have hp := Real.self_le_rpow_of_one_le hx hr
  have hc := mul_le_mul_of_nonneg_left hp (mul_nonneg hl.le h0.le)
  have hlower := (table_budget_cost_lower_bounds remaining multiplier firstCost secondCost
    exponent x y hl.le h0.le h1 (by linarith) hy).1
  rw [table_budget_origin_value remaining multiplier firstCost secondCost exponent (by linarith)]
  exact hb.trans_le (hc.trans hlower)

example : 0 < (1 : ℝ) ∧ 0 < (1 : ℝ) ∧ 0 ≤ (2 : ℝ) ∧ 1 ≤ (2 : ℝ) ∧
    1 ≤ (2 : ℝ) ∧ 0 ≤ (0 : ℝ) ∧ Real.log 2 < (1 : ℝ) * 1 * 2 := by
  have hl := Real.log_le_sub_one_of_pos (x := (2 : ℝ)) (by norm_num)
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  linarith

/-- The same exclusion applies when the second weight is large.
Source: arXiv:2309.02390v1, appendix D, symmetric cost sum, with the
cost hypotheses interchanged rather than silently discarding one. -/
theorem table_budget_second_large (remaining : ℕ)
    (multiplier firstCost secondCost exponent x y : ℝ)
    (hl : 0 < multiplier) (h0 : 0 ≤ firstCost) (h1 : 0 < secondCost)
    (hr : 1 ≤ exponent) (hx : 0 ≤ x) (hy : 1 ≤ y)
    (hb : Real.log ((remaining : ℝ) + 2) < multiplier * secondCost * y) :
    tableBudgetLoss remaining multiplier firstCost secondCost exponent 0 0 <
      tableBudgetLoss remaining multiplier firstCost secondCost exponent x y := by
  rw [table_budget_swap remaining multiplier firstCost secondCost exponent 0 0,
    table_budget_swap remaining multiplier firstCost secondCost exponent x y]
  exact table_budget_first_large remaining multiplier secondCost firstCost exponent y x hl h1 h0 hr hy hx hb

example : 0 < (1 : ℝ) ∧ 0 ≤ (1 : ℝ) ∧ 0 < (2 : ℝ) ∧ 1 ≤ (2 : ℝ) ∧
    0 ≤ (0 : ℝ) ∧ 1 ≤ (2 : ℝ) ∧ Real.log 2 < (1 : ℝ) * 2 * 2 := by
  have hl := Real.log_le_sub_one_of_pos (x := (2 : ℝ)) (by norm_num)
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
  linarith

/-- Actual CE and growing positive power costs attain a minimum on the
nonnegative quadrant. Source: arXiv:2309.02390v1, appendix D's global
allocation premise, established here for appendix C's finite-class CE.
The source's arbitrary reduced loss is specialized explicitly. -/
theorem exists_table_budget_minimizer (remaining : ℕ)
    (multiplier firstCost secondCost exponent : ℝ)
    (hl : 0 < multiplier) (h0 : 0 < firstCost) (h1 : 0 < secondCost) (hr : 1 ≤ exponent) :
    ∃ x y : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ ∀ u v : ℝ, 0 ≤ u → 0 ≤ v →
      tableBudgetLoss remaining multiplier firstCost secondCost exponent x y ≤
        tableBudgetLoss remaining multiplier firstCost secondCost exponent u v := by
  let radius := max 1 (max (Real.log ((remaining : ℝ) + 2) / (multiplier * firstCost))
    (Real.log ((remaining : ℝ) + 2) / (multiplier * secondCost))) + 1
  have hrad : 1 < radius := by
    have hm := le_max_left (1 : ℝ) (max
      (Real.log ((remaining : ℝ) + 2) / (multiplier * firstCost))
      (Real.log ((remaining : ℝ) + 2) / (multiplier * secondCost)))
    dsimp [radius]
    linarith
  have hfirst : Real.log ((remaining : ℝ) + 2) / (multiplier * firstCost) < radius := by
    have hinner := le_max_left (Real.log ((remaining : ℝ) + 2) / (multiplier * firstCost))
      (Real.log ((remaining : ℝ) + 2) / (multiplier * secondCost))
    have houter := le_max_right (1 : ℝ) (max
      (Real.log ((remaining : ℝ) + 2) / (multiplier * firstCost))
      (Real.log ((remaining : ℝ) + 2) / (multiplier * secondCost)))
    dsimp [radius]
    linarith
  have hsecond : Real.log ((remaining : ℝ) + 2) / (multiplier * secondCost) < radius := by
    have hinner := le_max_right (Real.log ((remaining : ℝ) + 2) / (multiplier * firstCost))
      (Real.log ((remaining : ℝ) + 2) / (multiplier * secondCost))
    have houter := le_max_right (1 : ℝ) (max
      (Real.log ((remaining : ℝ) + 2) / (multiplier * firstCost))
      (Real.log ((remaining : ℝ) + 2) / (multiplier * secondCost)))
    dsimp [radius]
    linarith
  let box : Set (ℝ × ℝ) := Set.Icc 0 radius ×ˢ Set.Icc 0 radius
  have hcompact : IsCompact box := isCompact_Icc.prod isCompact_Icc
  have horigin : (0, 0) ∈ box := by
    change (0 ≤ (0 : ℝ) ∧ (0 : ℝ) ≤ radius) ∧ (0 ≤ (0 : ℝ) ∧ (0 : ℝ) ≤ radius)
    constructor <;> constructor <;> linarith
  have hc := continuous_table_budget remaining multiplier firstCost secondCost exponent (by linarith)
  obtain ⟨point, hp, hm⟩ := hcompact.exists_isMinOn ⟨(0, 0), horigin⟩ hc.continuousOn
  have hpoint : (0 ≤ point.1 ∧ point.1 ≤ radius) ∧ (0 ≤ point.2 ∧ point.2 ≤ radius) := hp
  have hm0 := (isMinOn_iff.mp hm) (0, 0) horigin
  refine ⟨point.1, point.2, hpoint.1.1, hpoint.2.1, ?_⟩
  intro u v hu hv
  by_cases hin : (u, v) ∈ box
  · exact (isMinOn_iff.mp hm) (u, v) hin
  · have hout : radius < u ∨ radius < v := by
      by_cases huf : u ≤ radius
      · right
        by_contra hvf
        apply hin
        change (0 ≤ u ∧ u ≤ radius) ∧ (0 ≤ v ∧ v ≤ radius)
        exact ⟨⟨hu, huf⟩, ⟨hv, by linarith⟩⟩
      · left
        linarith
    rcases hout with huf | hvf
    · have hb := (div_lt_iff₀ (mul_pos hl h0)).mp (hfirst.trans huf)
      have he := table_budget_first_large remaining multiplier firstCost secondCost exponent u v
        hl h0 h1.le hr (by linarith) hv (by nlinarith [hb])
      exact hm0.trans he.le
    · have hb := (div_lt_iff₀ (mul_pos hl h1)).mp (hsecond.trans hvf)
      have he := table_budget_second_large remaining multiplier firstCost secondCost exponent u v
        hl h0.le h1 hr hu (by linarith) (by nlinarith [hb])
      exact hm0.trans he.le

example : 0 < (1 / 400 : ℝ) ∧ 0 < (1 : ℝ) ∧ 0 < (4 : ℝ) ∧ 1 ≤ (5 / 3 : ℝ) := by
  norm_num

/-- The source's 113-class, scalingExp=1.2, p=2 configuration has an
attained effective minimum. Source: arXiv:2309.02390v1, appendix C,
Table simulation hyperparameters, and appendix D's alpha/p normalization;
normalized norm_m=2 gives cost four and alpha=0.005 gives multiplier 1/400. -/
theorem source_113_class_effective_minimum_exists :
    ∃ x y : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ ∀ u v : ℝ, 0 ≤ u → 0 ≤ v →
      tableBudgetLoss 111 (1 / 400) 1 4 (5 / 3) x y ≤
        tableBudgetLoss 111 (1 / 400) 1 4 (5 / 3) u v := by
  exact exists_table_budget_minimizer 111 (1 / 400) 1 4 (5 / 3)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Attainment supplies an actual feasible point no worse than the
source's tied initialization, with its multiclass loss rather than MSE. -/
example : ∃ x y : ℝ, 0 ≤ x ∧ 0 ≤ y ∧
    tableBudgetLoss 111 (1 / 400) 1 4 (5 / 3) x y ≤ Real.log 113 := by
  obtain ⟨x, y, hx, hy, hm⟩ := source_113_class_effective_minimum_exists
  refine ⟨x, y, hx, hy, ?_⟩
  have hz := hm 0 0 (by norm_num) (by norm_num)
  rw [table_budget_origin_value 111 (1 / 400) 1 4 (5 / 3) (by norm_num)] at hz
  norm_num at hz
  exact hz

end Transformer.Grokking.CircuitEfficiency
