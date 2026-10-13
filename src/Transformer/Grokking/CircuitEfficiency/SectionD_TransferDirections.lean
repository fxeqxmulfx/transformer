import Transformer.Grokking.CircuitEfficiency.SectionD_PowerDerivative
import Transformer.Grokking.CircuitEfficiency.SectionD_EfficientAllocation

/-!
# Actual loss derivatives along constant-logit reallocations

Source: Varma et al., arXiv:2309.02390v1, appendix D, Theorem
efficiency-circuit-weights-logits, case 2 and its proof. Two fixed
circuits have identical training logits. The reduced training loss is
therefore unchanged by weights (x+epsilon,y-epsilon). Derive the
derivative of the stated power-budget objective without requiring an
unstated derivative of the reduced training loss.

When the budget exponent is above one, an absent circuit can receive a
small positive weight at negligible first-order cost. Removing that
weight from an active positive-cost circuit gives a negative derivative.
An actual feasible improving allocation follows from the derivative
limit, with a state-dependent increment. The source's incorrect
delta-multiplied estimate is not used. These are possible objective
improvements, not visited gradient or AdamW trajectories.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- Interchanging circuit names and their costs leaves the objective
unchanged. Source: arXiv:2309.02390v1, appendix D, effective-loss sum;
the identity is used to treat both boundary allocations. -/
theorem powerBudgetLoss_swap (trainLoss : ℝ → ℝ)
    (multiplier firstCost secondCost exponent x y : ℝ) :
    powerBudgetLoss trainLoss multiplier firstCost secondCost exponent x y =
      powerBudgetLoss trainLoss multiplier secondCost firstCost exponent y x := by
  unfold powerBudgetLoss
  rw [add_comm x y]
  ring

/-- Differentiate the actual objective on a transfer preserving the
sum of circuit weights. Source: arXiv:2309.02390v1, appendix D, case 2;
no differentiability of trainLoss is needed along its constant argument. -/
theorem power_budget_transfer_deriv (trainLoss : ℝ → ℝ)
    (multiplier firstCost secondCost exponent x y : ℝ) (hr : 1 ≤ exponent) :
    HasDerivAt
      (fun epsilon : ℝ => powerBudgetLoss trainLoss multiplier firstCost secondCost
        exponent (x + epsilon) (y - epsilon))
      (multiplier * exponent *
        (firstCost * x ^ (exponent - 1) - secondCost * y ^ (exponent - 1))) 0 := by
  have hx : HasDerivAt (fun epsilon : ℝ => x + epsilon) 1 0 :=
    (hasDerivAt_id (0 : ℝ)).const_add x
  have hy : HasDerivAt (fun epsilon : ℝ => y - epsilon) (-1) 0 :=
    (hasDerivAt_id (0 : ℝ)).const_sub y
  have hp : HasDerivAt (fun t : ℝ => t ^ exponent)
      (exponent * x ^ (exponent - 1)) (x + 0) := by
    simpa using Real.hasDerivAt_rpow_const (x := x) (p := exponent) (Or.inr hr)
  have hq : HasDerivAt (fun t : ℝ => t ^ exponent)
      (exponent * y ^ (exponent - 1)) (y - 0) := by
    simpa using Real.hasDerivAt_rpow_const (x := y) (p := exponent) (Or.inr hr)
  have h := (((hp.comp 0 hx).const_mul firstCost).add
    ((hq.comp 0 hy).const_mul secondCost)).const_mul multiplier
  convert h.const_add (trainLoss (x + y)) using 1
  · funext epsilon
    unfold powerBudgetLoss
    rw [show (x + epsilon) + (y - epsilon) = x + y by ring]
    rfl
  · ring

example : (1 : ℝ) ≤ 2 := by norm_num

/-- The absent first circuit has zero first-order budget cost above
exponent one, while the active second circuit supplies a strict gain.
Source: arXiv:2309.02390v1, appendix D, Theorem case 2; this holds for
any finite firstCost, with a positive multiplier and active cost. -/
theorem absent_first_weight_negative_deriv (trainLoss : ℝ → ℝ)
    (multiplier firstCost secondCost exponent active : ℝ)
    (hl : 0 < multiplier) (hc : 0 < secondCost) (hr : 1 < exponent) (ha : 0 < active) :
    HasDerivAt
      (fun epsilon : ℝ => powerBudgetLoss trainLoss multiplier firstCost secondCost
        exponent epsilon (active - epsilon))
      (-(multiplier * exponent * secondCost * active ^ (exponent - 1))) 0 ∧
      -(multiplier * exponent * secondCost * active ^ (exponent - 1)) < 0 := by
  have he : 0 < exponent - 1 := by linarith
  have hp := power_budget_transfer_deriv trainLoss multiplier firstCost secondCost exponent 0 active hr.le
  rw [Real.zero_rpow he.ne'] at hp
  constructor
  · convert hp using 1
    · simp only [zero_add]
    · ring
  · have hn : 0 < exponent := by linarith
    have hpower := Real.rpow_pos_of_pos ha (exponent - 1)
    have hg : 0 < multiplier * exponent * secondCost * active ^ (exponent - 1) := by positivity
    linarith

example : 0 < (1 : ℝ) ∧ 0 < (2 : ℝ) ∧ (1 : ℝ) < 2 ∧ 0 < (1 : ℝ) := by norm_num

/-- Construct an actual feasible finite improvement at the absent-
first-circuit boundary. Source: arXiv:2309.02390v1, appendix D, case 2;
the increment is strictly smaller than the active weight, so both new
weights are positive. This does not prescribe an optimizer learning rate. -/
theorem absent_first_weight_improvable (trainLoss : ℝ → ℝ)
    (multiplier firstCost secondCost exponent active : ℝ)
    (hl : 0 < multiplier) (hc : 0 < secondCost) (hr : 1 < exponent) (ha : 0 < active) :
    ∃ epsilon : ℝ, 0 < epsilon ∧ epsilon < active ∧
      powerBudgetLoss trainLoss multiplier firstCost secondCost exponent epsilon (active - epsilon) <
        powerBudgetLoss trainLoss multiplier firstCost secondCost exponent 0 active := by
  obtain ⟨hd, hn⟩ := absent_first_weight_negative_deriv
    trainLoss multiplier firstCost secondCost exponent active hl hc hr ha
  obtain ⟨delta, hp, hs⟩ :=
    Transformer.Grokking.AdamW.locally_decreases_of_negative_derivative _ _ hd hn
  have hm : 0 < min delta active := lt_min hp ha
  have he : 0 < min delta active / 2 := by positivity
  have hdelta : min delta active / 2 < delta := by
    have hb := min_le_left delta active
    linarith
  have hactive : min delta active / 2 < active := by
    have hb := min_le_right delta active
    linarith
  refine ⟨min delta active / 2, he, hactive, ?_⟩
  simpa only [sub_zero] using hs _ he hdelta

example : 0 < (1 : ℝ) ∧ 0 < (2 : ℝ) ∧ (1 : ℝ) < 2 ∧ 0 < (1 : ℝ) := by norm_num

/-- The other boundary also admits a feasible finite improvement.
Source: arXiv:2309.02390v1, appendix D, case 2; derive it by swapping
the two actual costs and weights, retaining the same training objective. -/
theorem absent_second_weight_improvable (trainLoss : ℝ → ℝ)
    (multiplier firstCost secondCost exponent active : ℝ)
    (hl : 0 < multiplier) (hc : 0 < firstCost) (hr : 1 < exponent) (ha : 0 < active) :
    ∃ epsilon : ℝ, 0 < epsilon ∧ epsilon < active ∧
      powerBudgetLoss trainLoss multiplier firstCost secondCost exponent (active - epsilon) epsilon <
        powerBudgetLoss trainLoss multiplier firstCost secondCost exponent active 0 := by
  obtain ⟨epsilon, he, hs, hi⟩ := absent_first_weight_improvable
    trainLoss multiplier secondCost firstCost exponent active hl hc hr ha
  rw [powerBudgetLoss_swap trainLoss multiplier secondCost firstCost exponent epsilon (active - epsilon),
    powerBudgetLoss_swap trainLoss multiplier secondCost firstCost exponent 0 active] at hi
  exact ⟨epsilon, he, hs, hi⟩

example : 0 < (1 : ℝ) ∧ 0 < (1 : ℝ) ∧ (1 : ℝ) < 2 ∧ 0 < (1 : ℝ) := by norm_num

/-- A concrete smooth bounded training loss has a feasible nonzero
minimum with both circuits active. Source: arXiv:2309.02390v1,
appendix D, case 2, instantiated with exponent two and costs one/two.
The training loss is (sum-1)^2; this is an explicit example of the
source's general loss class, not its softmax simulation or GPTMini.
The lower bound covers all real allocations. Its nonnegative feasible
domain therefore has this same attained minimum at (2/5,1/5). -/
theorem quadratic_mse_budget_minimum (u v : ℝ) :
    powerBudgetLoss (fun t => (t - 1) ^ 2) 1 1 2 2 (2 / 5) (1 / 5) ≤
      powerBudgetLoss (fun t => (t - 1) ^ 2) 1 1 2 2 u v := by
  simp only [powerBudgetLoss, Real.rpow_two]
  nlinarith [sq_nonneg (u + v - 3 / 5), sq_nonneg (u - 2 * v)]

end Transformer.Grokking.CircuitEfficiency
