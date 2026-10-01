/-
# IC-EoT: local optima are global optima of the reduced MPC

arXiv:2603.22095v2, §4.1, paragraph after Corollary 2. A feasible local
minimum is globally minimal for the learned reduced formulation. This says
nothing about the true plant, numerical solver termination or existence of
a minimizer, none of which follow from convexity alone.
-/

import Transformer.ICEoT.Section4_Feasible
import Mathlib.Analysis.Convex.Extrema

noncomputable section

namespace Transformer.ICEoT

/-- The global-optimality consequence stated immediately after Corollary 2;
§4.1. The local minimum is restricted to the actual feasible set. -/
theorem mpc_local_minimum_global {np n y u z : ℕ} (p : MPCProblem np n y u z)
    (hp : MPCConditions p) (D : Decision np u z) (hD : D ∈ mpcFeasible p)
    (hlocal : IsLocalMinOn (mpcObjective p) (mpcFeasible p) D) :
    IsMinOn (mpcObjective p) (mpcFeasible p) D :=
  IsMinOn.of_isLocalMinOn_of_convexOn hD hlocal ((mpcObjective_convex p hp).subset (Set.subset_univ _)
    (mpcFeasible_convex p hp.1))

/-- A zero-tariff instance of the same two-stage problem, used only as an
explicit local-optimum witness; §4.1, Eq. (41), permits zero prices. -/
def zeroPriceProblem : MPCProblem 2 0 1 1 1 :=
  { witnessProblem with price := fun _ => 0 }

/-- Feasibility, all hypotheses and an actual local minimizer coexist;
§4.1, Corollary 2 and its global-optimality consequence. -/
example : MPCConditions zeroPriceProblem ∧
    (0 : Decision 2 1 1) ∈ mpcFeasible zeroPriceProblem ∧
    IsLocalMinOn (mpcObjective zeroPriceProblem) (mpcFeasible zeroPriceProblem) 0 := by
  have hf : (0 : Decision 2 1 1) ∈ mpcFeasible zeroPriceProblem := by
    refine ⟨by intro k r; norm_num [zeroPriceProblem, witnessProblem], ?_,
      by intro k r; norm_num [zeroPriceProblem, witnessProblem]⟩
    intro k r
    fin_cases k <;> norm_num [predictedTemperature, zeroPriceProblem, witnessProblem,
      mpcRun, shiftHistory, extendControl, stageControl, witnessPredictor]
  refine ⟨⟨witnessPredictor_conditions, fun _ => le_rfl, zero_le_one⟩, hf, ?_⟩
  apply IsMinOn.isLocalMinOn
  intro D _
  have hn : 0 ≤ ∑ k : Fin 2, ∑ r : Fin 1, D.2 (k, r) ^ 2 :=
    Finset.sum_nonneg fun k _ => Finset.sum_nonneg fun r _ => sq_nonneg _
  simpa [mpcObjective, zeroPriceProblem, witnessProblem] using hn

end Transformer.ICEoT
