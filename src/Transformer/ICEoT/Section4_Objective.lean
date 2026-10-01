/-
# IC-EoT: convexity of the MPC objective

arXiv:2603.22095v2, §4.1, Corollary 2 and Eq. (41). Prices and the
slack coefficient are explicitly assumed non-negative.
-/

import Transformer.ICEoT.Section4_MPCModel
import Mathlib.Analysis.Convex.Mul

noncomputable section

namespace Transformer.ICEoT

/-- The joint electricity-plus-slack objective is convex after actual
recursive neural substitution; §4.1, Corollary 2, Eq. (41). -/
theorem mpcObjective_convex {np n y u z : ℕ} (p : MPCProblem np n y u z)
    (hp : MPCConditions p) : ConvexOn ℝ Set.univ (mpcObjective p) := by
  have he := fun k : Fin np => mpc_prediction_convex p.predictor p.history p.current
    hp.1 (k.val + 1) p.energy (np := np)
  have hsq : ConvexOn ℝ Set.univ (fun x : ℝ => x ^ 2) :=
    (by decide : Even (2 : ℕ)).convexOn_pow
  refine ⟨convex_univ, ?_⟩
  intro X hx Y hy a b ha hb hab
  simp only [mpcObjective, Prod.smul_snd, Prod.snd_add, Pi.add_apply,
    Pi.smul_apply, smul_eq_mul]
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro k hk
  have henergy := mul_le_mul_of_nonneg_left
    ((he k).2 (Set.mem_univ X.1) (Set.mem_univ Y.1) ha hb hab) (hp.2.1 k)
  have hslack := Finset.sum_le_sum fun r (_ : r ∈ Finset.univ) =>
    hsq.2 (Set.mem_univ (X.2 (k, r))) (Set.mem_univ (Y.2 (k, r))) ha hb hab
  simp only [smul_eq_mul] at henergy hslack
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum] at hslack
  have hpenalty := mul_le_mul_of_nonneg_left hslack hp.2.2
  change p.price k * predictedEnergy p (a • X.1 + b • Y.1) k + _ ≤ _
  dsimp only [predictedEnergy]
  nlinarith only [henergy, hpenalty]

example : MPCConditions witnessProblem := witnessProblem_conditions

end Transformer.ICEoT
