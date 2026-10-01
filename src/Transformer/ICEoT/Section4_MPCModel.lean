/-
# IC-EoT: the reduced soft-constrained MPC problem

arXiv:2603.22095v2, §4.1, Eq. (41). The decision variables are exactly
the finite controls and zone-wise slacks. Neural predictions are obtained
by substitution, with no nonlinear equality constraints. The squared
Euclidean slack norm is written as the sum of coordinate squares.
-/

import Transformer.ICEoT.Section4_Rollout

noncomputable section

namespace Transformer.ICEoT

/-- The joint decision space `(U, E)` in Eqs. (32), (40), §4.1. -/
abbrev Decision (np u z : ℕ) := Controls np u × Sequence np z

/-- Fixed data of Eq. (41), §4.1. Every model-generated variable is in
`Fin y`; temperature and energy are selected output coordinates. -/
structure MPCProblem (np n y u z : ℕ) where
  predictor : History n y u → Fin y → ℝ
  history : History n y u
  current : Fin y → ℝ
  energy : Fin y
  temperature : Fin z → Fin y
  price : Fin np → ℝ
  penalty : ℝ
  lowerControl : Fin u → ℝ
  upperControl : Fin u → ℝ
  upperTemperature : Sequence np z
  maxSlack : ℝ

/-- The hypotheses of §4.1, Corollary 2; fixed history is a parameter,
not a function of the decisions. No feasibility assumption is needed. -/
def MPCConditions {np n y u z : ℕ} (p : MPCProblem np n y u z) : Prop :=
  PredictorConditions p.predictor ∧ (∀ k, 0 ≤ p.price k) ∧ 0 ≤ p.penalty

/-- Predicted electricity at stage `k + 1`, Eqs. (38), (39), §4.1. -/
def predictedEnergy {np n y u z : ℕ} (p : MPCProblem np n y u z)
    (U : Controls np u) (k : Fin np) : ℝ :=
  (mpcRun p.predictor p.history p.current U (k.val + 1)).prediction p.energy

/-- Predicted zone temperature at stage `k + 1`, Eqs. (37), (39), §4.1. -/
def predictedTemperature {np n y u z : ℕ} (p : MPCProblem np n y u z)
    (U : Controls np u) (k : Fin np) (r : Fin z) : ℝ :=
  (mpcRun p.predictor p.history p.current U (k.val + 1)).prediction (p.temperature r)

/-- The electricity cost plus quadratic zone-wise slack penalty;
§4.1, Eq. (41). -/
def mpcObjective {np n y u z : ℕ} (p : MPCProblem np n y u z)
    (D : Decision np u z) : ℝ :=
  ∑ k, (p.price k * predictedEnergy p D.1 k + p.penalty * ∑ r, D.2 (k, r) ^ 2)

/-- Every control, upper-comfort and bounded-slack constraint in
§4.1, Eq. (41). There is no hard lower-temperature bound. -/
def mpcFeasible {np n y u z : ℕ} (p : MPCProblem np n y u z) : Set (Decision np u z) :=
  {D | (∀ k r, p.lowerControl r ≤ D.1 (k, r) ∧ D.1 (k, r) ≤ p.upperControl r) ∧
    (∀ k r, predictedTemperature p D.1 k r - D.2 (k, r) ≤ p.upperTemperature (k, r)) ∧
    (∀ k r, 0 ≤ D.2 (k, r) ∧ D.2 (k, r) ≤ p.maxSlack)}

/-- A concrete two-stage, one-zone problem for hypothesis witnesses;
§4.1, Corollary 2. Prices and slack penalty are positive. -/
def witnessProblem : MPCProblem 2 0 1 1 1 where
  predictor := witnessPredictor
  history := (0, 0)
  current := 0
  energy := 0
  temperature := fun _ => 0
  price := fun _ => 1
  penalty := 1
  lowerControl := fun _ => 0
  upperControl := fun _ => 1
  upperTemperature := fun _ => 5
  maxSlack := 2

/-- The MPC hypotheses are satisfiable; §4.1, Corollary 2. -/
theorem witnessProblem_conditions : MPCConditions witnessProblem :=
  ⟨witnessPredictor_conditions, fun _ => zero_le_one, zero_le_one⟩

end Transformer.ICEoT
