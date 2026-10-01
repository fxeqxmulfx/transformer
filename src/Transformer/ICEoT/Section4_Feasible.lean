/-
# IC-EoT: convexity of the MPC feasible set

arXiv:2603.22095v2, §4.1, Corollary 2, Eqs. (41), (43).
The temperature upper bound is softened by an affine slack. Both control
bounds and both slack bounds are retained, including the finite slack cap.
-/

import Transformer.ICEoT.Section4_Objective

noncomputable section

namespace Transformer.ICEoT

/-- All feasible-set constraints in Eq. (41) define a convex set;
§4.1, Corollary 2. The set is allowed to be empty. -/
theorem mpcFeasible_convex {np n y u z : ℕ} (p : MPCProblem np n y u z)
    (hp : PredictorConditions p.predictor) : Convex ℝ (mpcFeasible p) := by
  intro X hX Y hY a b ha hb hab
  refine ⟨?_, ?_, ?_⟩
  · intro k r
    exact (convex_Icc (p.lowerControl r) (p.upperControl r))
      (hX.1 k r) (hY.1 k r) ha hb hab
  · intro k r
    have ht := (mpc_prediction_convex p.predictor p.history p.current hp
      (k.val + 1) (p.temperature r) (np := np)).2
        (Set.mem_univ X.1) (Set.mem_univ Y.1) ha hb hab
    simp only [smul_eq_mul] at ht
    change predictedTemperature p (a • X.1 + b • Y.1) k r -
      (a * X.2 (k, r) + b * Y.2 (k, r)) ≤ p.upperTemperature (k, r)
    calc
      _ ≤ (a * predictedTemperature p X.1 k r + b * predictedTemperature p Y.1 k r) -
          (a * X.2 (k, r) + b * Y.2 (k, r)) := sub_le_sub_right ht _
      _ = a * (predictedTemperature p X.1 k r - X.2 (k, r)) +
          b * (predictedTemperature p Y.1 k r - Y.2 (k, r)) := by ring
      _ ≤ a * p.upperTemperature (k, r) + b * p.upperTemperature (k, r) :=
        add_le_add (mul_le_mul_of_nonneg_left (hX.2.1 k r) ha)
          (mul_le_mul_of_nonneg_left (hY.2.1 k r) hb)
      _ = _ := by rw [← add_mul, hab, one_mul]
  · intro k r
    exact (convex_Icc (0 : ℝ) p.maxSlack) (hX.2.2 k r) (hY.2.2 k r) ha hb hab

example : PredictorConditions witnessProblem.predictor := witnessPredictor_conditions

/-- Corollary 2 in full: the reduced problem has a convex objective and
a convex feasible set in exactly `(U, E)`; arXiv:2603.22095v2, §4.1. -/
theorem soft_constrained_mpc_convex {np n y u z : ℕ} (p : MPCProblem np n y u z)
    (hp : MPCConditions p) :
    ConvexOn ℝ Set.univ (mpcObjective p) ∧ Convex ℝ (mpcFeasible p) :=
  ⟨mpcObjective_convex p hp, mpcFeasible_convex p hp.1⟩

example : MPCConditions witnessProblem := witnessProblem_conditions

end Transformer.ICEoT
