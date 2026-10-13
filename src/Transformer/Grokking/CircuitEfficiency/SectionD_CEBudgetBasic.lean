import Transformer.Grokking.CircuitEfficiency.SectionC_TableLoss
import Transformer.Grokking.CircuitEfficiency.SectionD_TransferDirections
import Mathlib.Analysis.SpecialFunctions.Pow.Continuity

/-!
# The actual multiclass CE budget: continuity and boundary descent

Source: Varma et al., arXiv:2309.02390v1, appendix C's multiclass CE
and appendix D's effective-loss sum. Normalized circuit norms to power
p give firstCost and secondCost; exponent is p/scalingExp. The positive
penalty is coupled to the objective. Native AdamW's decoupled decay is
not silently identified with this penalty.

At exponent above one, introducing a small positive circuit weight at
the origin has zero first-order penalty and strictly improves actual
CE. This removes the zero-sum equilibrium obstruction for this loss,
without assuming a learned table or a visited optimizer trajectory.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- The source's fixed-table CE plus its effective power penalty.
Source: arXiv:2309.02390v1, appendices C and D, effective-loss formulas. -/
noncomputable def tableBudgetLoss (remaining : ℕ)
    (multiplier firstCost secondCost exponent x y : ℝ) : ℝ :=
  powerBudgetLoss (tableTrainCE remaining) multiplier firstCost secondCost exponent x y

/-- Actual CE is nonnegative for every finite total score. Source:
arXiv:2309.02390v1, appendix C; the denominator includes the true class. -/
theorem table_train_ce_nonneg (remaining : ℕ) (score : ℝ) :
    0 ≤ tableTrainCE remaining score := by
  have hle : Real.exp score ≤ Real.exp score + (remaining : ℝ) + 1 := by
    linarith [Nat.cast_nonneg (α := ℝ) remaining]
  have hl := Real.log_le_log (Real.exp_pos score) hle
  rw [Real.log_exp] at hl
  rw [table_train_ce_formula]
  linarith

/-- Actual finite-class CE is continuous on the entire real score line.
Source: arXiv:2309.02390v1, appendix C's train-loss formula. -/
theorem continuous_table_train_ce (remaining : ℕ) : Continuous (tableTrainCE remaining) := by
  have hp : ∀ s : ℝ, Real.exp s + (remaining : ℝ) + 1 ≠ 0 := by
    intro s
    have hpos : 0 < Real.exp s + (remaining : ℝ) + 1 := by positivity
    exact hpos.ne'
  have hc := ((Real.continuous_exp.add continuous_const).add continuous_const).log hp
  have he : tableTrainCE remaining =
      fun s : ℝ => Real.log (Real.exp s + (remaining : ℝ) + 1) - s := by
    funext s
    exact table_train_ce_formula remaining s
  rw [he]
  exact hc.sub continuous_id

/-- Positive real powers make the whole CE budget continuous, including
the axes. Source: arXiv:2309.02390v1, appendix D, power-budget objective. -/
theorem continuous_table_budget (remaining : ℕ)
    (multiplier firstCost secondCost exponent : ℝ) (hr : 0 ≤ exponent) :
    Continuous (fun z : ℝ × ℝ =>
      tableBudgetLoss remaining multiplier firstCost secondCost exponent z.1 z.2) := by
  have hp := Real.continuous_rpow_const hr
  have ht := (continuous_table_train_ce remaining).comp (continuous_fst.add continuous_snd)
  have hc : Continuous (fun z : ℝ × ℝ => firstCost * z.1 ^ exponent + secondCost * z.2 ^ exponent) :=
    (continuous_const.mul (hp.comp continuous_fst)).add
      (continuous_const.mul (hp.comp continuous_snd))
  exact ht.add (continuous_const.mul hc)

example : 0 ≤ (5 / 3 : ℝ) := by norm_num

/-- Each positive circuit cost separately bounds the nonnegative-domain
objective from below. Source: arXiv:2309.02390v1, appendix D. Retain
nonnegative costs and multiplier explicitly rather than presuming them. -/
theorem table_budget_cost_lower_bounds (remaining : ℕ)
    (multiplier firstCost secondCost exponent x y : ℝ)
    (hl : 0 ≤ multiplier) (h0 : 0 ≤ firstCost) (h1 : 0 ≤ secondCost)
    (hx : 0 ≤ x) (hy : 0 ≤ y) :
    multiplier * firstCost * x ^ exponent ≤
      tableBudgetLoss remaining multiplier firstCost secondCost exponent x y ∧
    multiplier * secondCost * y ^ exponent ≤
      tableBudgetLoss remaining multiplier firstCost secondCost exponent x y := by
  have ht := table_train_ce_nonneg remaining (x + y)
  have hp := Real.rpow_nonneg hx exponent
  have hq := Real.rpow_nonneg hy exponent
  unfold tableBudgetLoss powerBudgetLoss
  constructor <;> nlinarith [mul_nonneg hl (mul_nonneg h0 hp), mul_nonneg hl (mul_nonneg h1 hq)]

example : 0 ≤ (1 : ℝ) ∧ 0 ≤ (1 : ℝ) ∧ 0 ≤ (2 : ℝ) ∧ 0 ≤ (0 : ℝ) ∧ 0 ≤ (1 : ℝ) := by
  norm_num

/-- Swapping both costs and weights preserves the actual CE budget.
Source: arXiv:2309.02390v1, appendix D's sum of identical training logits. -/
theorem table_budget_swap (remaining : ℕ)
    (multiplier firstCost secondCost exponent x y : ℝ) :
    tableBudgetLoss remaining multiplier firstCost secondCost exponent x y =
      tableBudgetLoss remaining multiplier secondCost firstCost exponent y x := by
  exact powerBudgetLoss_swap (tableTrainCE remaining) multiplier firstCost secondCost exponent x y

/-- At the origin every nonzero power has zero penalty; actual CE is log q.
Source: arXiv:2309.02390v1, appendices C and D; excluding exponent zero
is necessary because the real-power convention gives zero^zero=one. -/
theorem table_budget_origin_value (remaining : ℕ)
    (multiplier firstCost secondCost exponent : ℝ) (hr : exponent ≠ 0) :
    tableBudgetLoss remaining multiplier firstCost secondCost exponent 0 0 =
      Real.log ((remaining : ℝ) + 2) := by
  simp only [tableBudgetLoss, powerBudgetLoss, Real.zero_rpow hr,
    mul_zero, add_zero]
  exact table_train_ce_initial remaining

example : (5 / 3 : ℝ) ≠ 0 := by norm_num

/-- Introducing the first circuit at the origin has the actual negative
CE derivative, independently of finite penalty coefficients when r>1.
Source: arXiv:2309.02390v1, appendix C's train CE and appendix D case 2. -/
theorem table_budget_origin_deriv (remaining : ℕ)
    (multiplier firstCost secondCost exponent : ℝ) (hr : 1 < exponent) :
    HasDerivAt (fun t : ℝ => tableBudgetLoss remaining multiplier firstCost secondCost exponent t 0)
      (-((remaining : ℝ) + 1) / ((remaining : ℝ) + 2)) 0 := by
  have he : exponent - 1 ≠ 0 := by linarith
  have hp := Real.hasDerivAt_rpow_const (x := (0 : ℝ)) (p := exponent) (Or.inr hr.le)
  rw [Real.zero_rpow he, mul_zero] at hp
  have hc := ((hp.const_mul firstCost).add_const (secondCost * (0 : ℝ) ^ exponent)).const_mul multiplier
  have hd := (table_train_ce_deriv remaining 0).add hc
  convert hd using 1
  · funext t
    simp only [tableBudgetLoss, powerBudgetLoss, add_zero]
    rfl
  · rw [Real.exp_zero]
    ring

example : (1 : ℝ) < 5 / 3 := by norm_num

/-- Actual CE admits a feasible finite improving allocation from zero.
Source: arXiv:2309.02390v1, appendix D case 2, strengthened using
appendix C's actual CE. No specified finite optimizer rate is claimed. -/
theorem table_budget_origin_improvable (remaining : ℕ)
    (multiplier firstCost secondCost exponent : ℝ) (hr : 1 < exponent) :
    ∃ epsilon : ℝ, 0 < epsilon ∧
      tableBudgetLoss remaining multiplier firstCost secondCost exponent epsilon 0 <
        tableBudgetLoss remaining multiplier firstCost secondCost exponent 0 0 := by
  have hd := table_budget_origin_deriv remaining multiplier firstCost secondCost exponent hr
  have hs : -((remaining : ℝ) + 1) / ((remaining : ℝ) + 2) < 0 := by
    have hn : 0 < (remaining : ℝ) + 1 := by positivity
    have hp : 0 < (remaining : ℝ) + 2 := by positivity
    exact div_neg_of_neg_of_pos (neg_neg_of_pos hn) hp
  obtain ⟨delta, hp, hsmall⟩ :=
    Transformer.Grokking.AdamW.locally_decreases_of_negative_derivative _ _ hd hs
  refine ⟨delta / 2, by positivity, ?_⟩
  exact hsmall _ (by positivity) (by linarith)

example : (1 : ℝ) < 5 / 3 := by norm_num

end Transformer.Grokking.CircuitEfficiency
