import Mathlib.Analysis.MeanInequalitiesPow
import Mathlib.Tactic

/-!
# Efficient allocation when circuits agree on training logits

Source: Varma et al., arXiv:2309.02390v1, appendix D, subsection
Weight decay favours efficient circuits, Theorem efficiency-circuit-
weights-logits, case 1. Explicit specialization: two fixed circuits,
with exponent `r = p / scalingExp` in `(0, 1]`, cost coefficients equal
to their normalized parameter norms to power p, and multiplier equal
to the source's weightDecayHyper / p. The training loss is evaluated
at the sum because the source assumes identical training logits.

Correction: strict exclusion requires a positive penalty multiplier;
the theorem's wording does not state that hypothesis. A bounded smooth
training loss below supplies a counterexample when it is zero. Global
optimality is an input, not an assertion that an optimizer reaches it.
No existence, subweight dynamics, AdamW L2 equivalence, learned GPTMini
circuit decomposition or held-out success is assumed or proved.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- The source's reduced objective, retaining its training-loss and cost
arguments. Source: arXiv:2309.02390v1, appendix D, generalized effective
loss equation; two-circuit specialization with exponent p/scalingExp. -/
noncomputable def powerBudgetLoss (trainLoss : ℝ → ℝ)
    (multiplier cheapCost expensiveCost exponent x y : ℝ) : ℝ :=
  trainLoss (x + y) + multiplier *
    (cheapCost * x ^ exponent + expensiveCost * y ^ exponent)

/-- The identical-logit premise really permits transferring circuit
weight while keeping every current training logit. Source:
arXiv:2309.02390v1, appendix D, Theorem case 1; the premise is pointwise,
not equality of accuracy or of mean logits alone. -/
theorem identical_circuits_reallocation {ι : Type*} (gen mem : ι → ℝ)
    (h : ∀ i, gen i = mem i) (x y : ℝ) :
    ∀ i, x * gen i + y * mem i = (x + y) * gen i := by
  intro i
  rw [← h i]
  ring

example : ∀ i : Bool, (if i then (1 : ℝ) else 0) = (if i then (1 : ℝ) else 0) := by
  intro i
  rfl

/-- Quantify the penalty saved by moving the second weight into the first.
Source: arXiv:2309.02390v1, appendix D, case 1's transfer argument and
sum-of-powers lemma; its exponent is p/scalingExp, not a layer count. -/
theorem power_budget_reallocation_bound (trainLoss : ℝ → ℝ)
    (multiplier cheapCost expensiveCost exponent x y : ℝ)
    (hl : 0 ≤ multiplier) (hc : 0 ≤ cheapCost) (hr : 0 < exponent)
    (hr1 : exponent ≤ 1) (hx : 0 ≤ x) (hy : 0 ≤ y) :
    powerBudgetLoss trainLoss multiplier cheapCost expensiveCost exponent (x + y) 0 +
      multiplier * (expensiveCost - cheapCost) * y ^ exponent ≤
    powerBudgetLoss trainLoss multiplier cheapCost expensiveCost exponent x y := by
  have hp := Real.rpow_add_le_add_rpow hx hy hr.le hr1
  have hm := mul_le_mul_of_nonneg_left hp (mul_nonneg hl hc)
  unfold powerBudgetLoss
  rw [Real.zero_rpow hr.ne', add_zero]
  nlinarith

example : 0 ≤ (1 : ℝ) ∧ 0 ≤ (1 : ℝ) ∧ 0 < (1 : ℝ) ∧ (1 : ℝ) ≤ 1 ∧
    0 ≤ (2 : ℝ) ∧ 0 ≤ (3 : ℝ) := by norm_num

/-- A positive inefficient weight admits an actual strict improvement.
Source: arXiv:2309.02390v1, appendix D, Theorem case 1, corrected to
include a positive penalty multiplier. The training term cancels exactly. -/
theorem inefficient_weight_strictly_improvable (trainLoss : ℝ → ℝ)
    (multiplier cheapCost expensiveCost exponent x y : ℝ)
    (hl : 0 < multiplier) (hc : 0 ≤ cheapCost) (hcost : cheapCost < expensiveCost)
    (hr : 0 < exponent) (hr1 : exponent ≤ 1) (hx : 0 ≤ x) (hy : 0 < y) :
    powerBudgetLoss trainLoss multiplier cheapCost expensiveCost exponent (x + y) 0 <
      powerBudgetLoss trainLoss multiplier cheapCost expensiveCost exponent x y := by
  have hb := power_budget_reallocation_bound trainLoss multiplier cheapCost expensiveCost
    exponent x y hl.le hc hr hr1 hx hy.le
  have hp := Real.rpow_pos_of_pos hy exponent
  have hs : 0 < multiplier * (expensiveCost - cheapCost) * y ^ exponent := by positivity
  linarith

example : 0 < (1 : ℝ) ∧ 0 ≤ (1 : ℝ) ∧ (1 : ℝ) < 2 ∧
    0 < (1 : ℝ) ∧ (1 : ℝ) ≤ 1 ∧ 0 ≤ (2 : ℝ) ∧ 0 < (3 : ℝ) := by norm_num

/-- Corrected two-circuit instance of the source's case 1: every global
minimizer excludes the strictly more costly circuit. Source:
arXiv:2309.02390v1, appendix D, Theorem efficiency-circuit-weights-logits;
this does not assert that native AdamW reaches a global minimizer. -/
theorem minimizer_excludes_inefficient_circuit (trainLoss : ℝ → ℝ)
    (multiplier cheapCost expensiveCost exponent x y : ℝ)
    (hl : 0 < multiplier) (hc : 0 ≤ cheapCost) (hcost : cheapCost < expensiveCost)
    (hr : 0 < exponent) (hr1 : exponent ≤ 1) (hx : 0 ≤ x) (hy : 0 ≤ y)
    (hmin : ∀ u v : ℝ, 0 ≤ u → 0 ≤ v →
      powerBudgetLoss trainLoss multiplier cheapCost expensiveCost exponent x y ≤
      powerBudgetLoss trainLoss multiplier cheapCost expensiveCost exponent u v) : y = 0 := by
  by_contra hn
  have hp : 0 < y := lt_of_le_of_ne hy (Ne.symm hn)
  have hi := inefficient_weight_strictly_improvable trainLoss multiplier cheapCost expensiveCost
    exponent x y hl hc hcost hr hr1 hx hp
  have hm := hmin (x + y) 0 (add_nonneg hx hy) le_rfl
  linarith

example : 0 < (1 : ℝ) ∧ 0 ≤ (1 : ℝ) ∧ (1 : ℝ) < 2 ∧
    0 < (1 : ℝ) ∧ (1 : ℝ) ≤ 1 ∧ 0 ≤ (1 / 2 : ℝ) ∧ 0 ≤ (0 : ℝ) ∧
    (∀ u v : ℝ, 0 ≤ u → 0 ≤ v →
      powerBudgetLoss (fun t => (t - 1) ^ 2) 1 1 2 1 (1 / 2) 0 ≤
      powerBudgetLoss (fun t => (t - 1) ^ 2) 1 1 2 1 u v) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, ?_⟩
  intro u v _ hv
  simp only [powerBudgetLoss, Real.rpow_one]
  nlinarith [sq_nonneg (u + v - 1 / 2)]

/-- Without a penalty the source's exclusion conclusion fails, even for
a bounded differentiable MSE training loss and unequal positive costs.
Source: arXiv:2309.02390v1, appendix D, Theorem case 1; counterexample
to omitting positivity of weightDecayHyper, not to its positive case. -/
theorem zero_penalty_keeps_inefficient_minimizer :
    (∀ u v : ℝ, 0 ≤ u → 0 ≤ v →
      powerBudgetLoss (fun t => (t - 1) ^ 2) 0 1 2 1 0 1 ≤
      powerBudgetLoss (fun t => (t - 1) ^ 2) 0 1 2 1 u v) ∧ (1 : ℝ) ≠ 0 := by
  constructor
  · intro u v _ _
    simp only [powerBudgetLoss, zero_mul, add_zero, Real.rpow_one]
    norm_num
    positivity
  · norm_num

/-- At equal cost and the linear-penalty boundary, allocation itself is
unidentified by the objective. Source: arXiv:2309.02390v1, appendix D,
case 1 with scalingExp = p; the theorem excludes only nonminimal costs. -/
theorem equal_cost_linear_allocation (trainLoss : ℝ → ℝ)
    (multiplier cost x y : ℝ) :
    powerBudgetLoss trainLoss multiplier cost cost 1 x y =
      powerBudgetLoss trainLoss multiplier cost cost 1 (x + y) 0 := by
  unfold powerBudgetLoss
  simp only [Real.rpow_one, add_zero, mul_zero]
  ring

/-- Equal efficiency permits the same training objective with opposite
held-out margins. Source: arXiv:2309.02390v1, section 3's Gen/Mem lookup
tables, specialized to two classes, and appendix D's linear-penalty
boundary. The held-out logits are (x,y), since the two tables support
opposite classes there. Objective equality is not a claim of optimality. -/
theorem equal_cost_opposite_heldout_margins (trainLoss : ℝ → ℝ)
    (multiplier cost weight : ℝ) (hw : 0 < weight) :
    powerBudgetLoss trainLoss multiplier cost cost 1 weight 0 =
      powerBudgetLoss trainLoss multiplier cost cost 1 0 weight ∧
    0 < weight - 0 ∧ 0 - weight < 0 := by
  have hl := equal_cost_linear_allocation trainLoss multiplier cost 0 weight
  rw [zero_add] at hl
  refine ⟨hl.symm, ?_, ?_⟩ <;> linarith

example : 0 < (1 : ℝ) := by norm_num

end Transformer.Grokking.CircuitEfficiency
