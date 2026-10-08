import Transformer.Grokking.CircuitEfficiency.SectionD_TransferDirections
import Mathlib.Analysis.Calculus.LocalExtr.Basic

/-!
# Actual allocations at nonzero minima of the superlinear budget

Source: Varma et al., arXiv:2309.02390v1, appendix D, Theorem
efficiency-circuit-weights-logits, case 2. Two fixed circuits agree on
training logits; the exponent r=p/scalingExp exceeds one. Explicit
positive multiplier and positive normalized cost coefficients are
retained. At every nonzero nonnegative global minimum both weights
are positive, and their ratio is the inverse-cost power predicted by
the source. The zero minimum is excluded by an input sum condition.

Feasible boundary improvements and Fermat's theorem on the constant-
logit transfer derive these claims from the actual objective. Neither
the incorrect printed increment lemma nor an assigned stationary
gradient is used. The full input-output tables, minimizer existence,
subweight dynamics and arrival of native AdamW remain separate questions.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- A nonzero global minimum cannot leave either circuit absent.
Source: arXiv:2309.02390v1, appendix D, Theorem case 2, corrected with
positive penalty/cost hypotheses; actual feasible improvements give the proof. -/
theorem superlinear_minimizer_positive_weights (trainLoss : ℝ → ℝ)
    (multiplier firstCost secondCost exponent x y : ℝ)
    (hl : 0 < multiplier) (h0 : 0 < firstCost) (h1 : 0 < secondCost) (hr : 1 < exponent)
    (hx : 0 ≤ x) (hy : 0 ≤ y) (hs : 0 < x + y)
    (hmin : ∀ u v : ℝ, 0 ≤ u → 0 ≤ v →
      powerBudgetLoss trainLoss multiplier firstCost secondCost exponent x y ≤
        powerBudgetLoss trainLoss multiplier firstCost secondCost exponent u v) :
    0 < x ∧ 0 < y := by
  constructor
  · by_contra hn
    have hz : x = 0 := by linarith
    subst x
    have hp : 0 < y := by linarith
    obtain ⟨epsilon, he, hsmall, hi⟩ :=
      absent_first_weight_improvable trainLoss multiplier firstCost secondCost exponent y hl h1 hr hp
    have hm := hmin epsilon (y - epsilon) he.le (by linarith)
    linarith
  · by_contra hn
    have hz : y = 0 := by linarith
    subst y
    have hp : 0 < x := by linarith
    obtain ⟨epsilon, he, hsmall, hi⟩ :=
      absent_second_weight_improvable trainLoss multiplier firstCost secondCost exponent x hl h0 hr hp
    have hm := hmin (x - epsilon) epsilon (by linarith) he.le
    linarith

example : 0 < (2 / 5 : ℝ) ∧ 0 < (1 / 5 : ℝ) := by
  apply superlinear_minimizer_positive_weights (fun t => (t - 1) ^ 2) 1 1 2 2
  all_goals try norm_num
  intro u v _ _
  exact quadratic_mse_budget_minimum u v

/-- Positive weights put zero inside the feasible transfer interval,
so a global objective minimum is a local minimum of that actual curve.
Source: arXiv:2309.02390v1, appendix D, case 2's interior argument. -/
theorem positive_minimizer_transfer_local (trainLoss : ℝ → ℝ)
    (multiplier firstCost secondCost exponent x y : ℝ) (hx : 0 < x) (hy : 0 < y)
    (hmin : ∀ u v : ℝ, 0 ≤ u → 0 ≤ v →
      powerBudgetLoss trainLoss multiplier firstCost secondCost exponent x y ≤
        powerBudgetLoss trainLoss multiplier firstCost secondCost exponent u v) :
    IsLocalMin (fun epsilon : ℝ => powerBudgetLoss trainLoss multiplier firstCost secondCost
      exponent (x + epsilon) (y - epsilon)) 0 := by
  change ∀ᶠ epsilon in nhds (0 : ℝ),
    powerBudgetLoss trainLoss multiplier firstCost secondCost exponent (x + 0) (y - 0) ≤
      powerBudgetLoss trainLoss multiplier firstCost secondCost exponent (x + epsilon) (y - epsilon)
  apply Metric.eventually_nhds_iff.mpr
  refine ⟨min x y, lt_min hx hy, ?_⟩
  intro epsilon hd
  have ha : |epsilon| < min x y := by simpa [Real.dist_eq] using hd
  obtain ⟨hlo, hhi⟩ := abs_lt.mp ha
  have hleft := min_le_left x y
  have hright := min_le_right x y
  have hm := hmin (x + epsilon) (y - epsilon) (by linarith) (by linarith)
  simpa only [add_zero, sub_zero] using hm

example : IsLocalMin (fun epsilon : ℝ =>
    powerBudgetLoss (fun t => (t - 1) ^ 2) 1 1 2 2 (2 / 5 + epsilon) (1 / 5 - epsilon)) 0 := by
  apply positive_minimizer_transfer_local <;> try norm_num
  intro u v _ _
  exact quadratic_mse_budget_minimum u v

/-- Fermat's theorem balances the actual marginal power costs at a
nonzero minimum. Source: arXiv:2309.02390v1, appendix D, case 2;
the equality follows from the objective, not an imposed gradient equation. -/
theorem superlinear_minimizer_cost_balance (trainLoss : ℝ → ℝ)
    (multiplier firstCost secondCost exponent x y : ℝ)
    (hl : 0 < multiplier) (h0 : 0 < firstCost) (h1 : 0 < secondCost) (hr : 1 < exponent)
    (hx : 0 ≤ x) (hy : 0 ≤ y) (hs : 0 < x + y)
    (hmin : ∀ u v : ℝ, 0 ≤ u → 0 ≤ v →
      powerBudgetLoss trainLoss multiplier firstCost secondCost exponent x y ≤
        powerBudgetLoss trainLoss multiplier firstCost secondCost exponent u v) :
    firstCost * x ^ (exponent - 1) = secondCost * y ^ (exponent - 1) := by
  obtain ⟨hp, hq⟩ := superlinear_minimizer_positive_weights
    trainLoss multiplier firstCost secondCost exponent x y hl h0 h1 hr hx hy hs hmin
  have hm := positive_minimizer_transfer_local trainLoss multiplier firstCost secondCost exponent x y hp hq hmin
  have hd := power_budget_transfer_deriv trainLoss multiplier firstCost secondCost exponent x y hr.le
  have hz := hm.hasDerivAt_eq_zero hd
  have he : 0 < exponent := by linarith
  have hc : multiplier * exponent ≠ 0 := by positivity
  have hb := (mul_eq_zero.mp hz).resolve_left hc
  linarith

example : (1 : ℝ) * (2 / 5 : ℝ) ^ ((2 : ℝ) - 1) =
    2 * (1 / 5 : ℝ) ^ ((2 : ℝ) - 1) := by
  apply superlinear_minimizer_cost_balance (fun t => (t - 1) ^ 2) 1
  all_goals try norm_num
  intro u v _ _
  exact quadratic_mse_budget_minimum u v

/-- Solve the verified marginal-cost equation for the weight ratio.
Source: arXiv:2309.02390v1, appendix D, final case-2 algebra; cost
coefficients are normalized norms to power p, and exponent is p/scalingExp. -/
theorem cost_balance_weight_ratio (firstCost secondCost exponent x y : ℝ)
    (hc : firstCost ≠ 0) (hx : 0 < x) (hy : 0 < y) (hr : exponent ≠ 1)
    (hb : firstCost * x ^ (exponent - 1) = secondCost * y ^ (exponent - 1)) :
    x / y = (secondCost / firstCost) ^ ((exponent - 1)⁻¹) := by
  have hp := Real.rpow_pos_of_pos hy (exponent - 1)
  have he : exponent - 1 ≠ 0 := by intro hz; apply hr; linarith
  have hratio : (x / y) ^ (exponent - 1) = secondCost / firstCost := by
    rw [Real.div_rpow hx.le hy.le]
    apply (div_eq_div_iff hp.ne' hc).mpr
    nlinarith [hb]
  have hroot := congrArg (fun t : ℝ => t ^ ((exponent - 1)⁻¹)) hratio
  rw [Real.rpow_rpow_inv (div_nonneg hx.le hy.le) he] at hroot
  exact hroot

example : (1 : ℝ) ≠ 0 ∧ 0 < (2 / 5 : ℝ) ∧ 0 < (1 / 5 : ℝ) ∧ (2 : ℝ) ≠ 1 ∧
    (1 : ℝ) * (2 / 5 : ℝ) ^ ((2 : ℝ) - 1) = 2 * (1 / 5 : ℝ) ^ ((2 : ℝ) - 1) := by
  norm_num

/-- Corrected two-circuit case 2, with the actual ratio derived at
every nonzero feasible global minimum. Source: arXiv:2309.02390v1,
appendix D, Theorem efficiency-circuit-weights-logits. This is an
equilibrium allocation result, not repeated-step AdamW convergence. -/
theorem superlinear_minimizer_weight_ratio (trainLoss : ℝ → ℝ)
    (multiplier firstCost secondCost exponent x y : ℝ)
    (hl : 0 < multiplier) (h0 : 0 < firstCost) (h1 : 0 < secondCost) (hr : 1 < exponent)
    (hx : 0 ≤ x) (hy : 0 ≤ y) (hs : 0 < x + y)
    (hmin : ∀ u v : ℝ, 0 ≤ u → 0 ≤ v →
      powerBudgetLoss trainLoss multiplier firstCost secondCost exponent x y ≤
        powerBudgetLoss trainLoss multiplier firstCost secondCost exponent u v) :
    x / y = (secondCost / firstCost) ^ ((exponent - 1)⁻¹) := by
  obtain ⟨hp, hq⟩ := superlinear_minimizer_positive_weights
    trainLoss multiplier firstCost secondCost exponent x y hl h0 h1 hr hx hy hs hmin
  have hb := superlinear_minimizer_cost_balance
    trainLoss multiplier firstCost secondCost exponent x y hl h0 h1 hr hx hy hs hmin
  exact cost_balance_weight_ratio firstCost secondCost exponent x y h0.ne' hp hq (ne_of_gt hr) hb

example : (2 / 5 : ℝ) / (1 / 5) = ((2 : ℝ) / 1) ^ (((2 : ℝ) - 1)⁻¹) := by
  apply superlinear_minimizer_weight_ratio (fun t => (t - 1) ^ 2) 1
  all_goals try norm_num
  intro u v _ _
  exact quadratic_mse_budget_minimum u v

/-- Cost balance gives a positive held-out margin when the generalizing
table is more efficient. Source: arXiv:2309.02390v1, section 3's Gen/Mem
tables, specialized to two classes, and appendix D, case 2. The logits
there are (x,y); the minimum has just been proved to supply this balance.
This is a conditional fixed-table decision, not learning those tables. -/
theorem cost_balance_positive_heldout_margin (firstCost secondCost exponent x y : ℝ)
    (hc : 0 < firstCost) (heff : firstCost < secondCost) (hr : 1 < exponent)
    (hx : 0 ≤ x) (hy : 0 < y)
    (hb : firstCost * x ^ (exponent - 1) = secondCost * y ^ (exponent - 1)) :
    0 < x - y := by
  have he : 0 < exponent - 1 := by linarith
  have hp := Real.rpow_pos_of_pos hy (exponent - 1)
  by_contra hn
  have hxy : x ≤ y := by linarith
  have hpow := Real.rpow_le_rpow hx hxy he.le
  have hm := mul_le_mul_of_nonneg_left hpow hc.le
  have hg : 0 < (secondCost - firstCost) * y ^ (exponent - 1) := by positivity
  nlinarith [hb]

example : 0 < (1 : ℝ) ∧ (1 : ℝ) < 2 ∧ (1 : ℝ) < 2 ∧ 0 ≤ (2 : ℝ) ∧ 0 < (1 : ℝ) ∧
    (1 : ℝ) * (2 : ℝ) ^ ((2 : ℝ) - 1) = 2 * (1 : ℝ) ^ ((2 : ℝ) - 1) := by norm_num

end Transformer.Grokking.CircuitEfficiency
