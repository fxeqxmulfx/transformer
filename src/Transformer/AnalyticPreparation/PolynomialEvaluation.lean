/-
# Real analytic preparation: PolynomialEvaluation

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d:
https://github.com/BochaoKong/classical-complex-wpt
Only the existence proof and its coefficient infrastructure are copied.
The scalar field is real; no complex root or complex differentiation result
is used. Apache-2.0 license: third_party/classical-complex-wpt/LICENSE.
-/

import Transformer.AnalyticPreparation.WeightedEvaluation

open Filter Finset
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
set_option maxHeartbeats 5000000

namespace Transformer.AnalyticPreparation

variable {A I J : Type*} [AddCommMonoid A] [Finset.HasAntidiagonal A]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem evalL1PowerSeries_smul (c : ℝ) (a : L1Sequence) (w : ℝ) :
    evalL1PowerSeries (c • a) w = c * evalL1PowerSeries a w := by
  change l1EvalOperator w (c • a) = _
  rw [map_smul]
  rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem evalL1PowerSeries_monomialSeq (d : ℕ) {w : ℝ} (hw : ‖w‖ < 1) :
    evalL1PowerSeries (monomialSeq d) w = w ^ d := by
  rw [evalL1PowerSeries_eq_tsum _ hw, tsum_eq_single d]
  · simp [monomialSeq_apply_same]
  · intro k hkd
    simp [monomialSeq_apply_ne hkd]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem evalL1PowerSeries_eq_sum_fin_of_highShift_eq_zero
    (d : ℕ) (a : L1Sequence) (ha : seqHighShift d a = 0)
    {w : ℝ} (hw : ‖w‖ < 1) :
    evalL1PowerSeries a w = ∑ i : Fin d, a i * w ^ (i : ℕ) := by
  rw [evalL1PowerSeries_eq_tsum _ hw]
  let F : ℕ → ℝ := fun k ↦ a k * w ^ k
  change (∑' k : ℕ, F k) = ∑ i : Fin d, F i
  rw [Fin.sum_univ_eq_sum_range F d]
  apply tsum_eq_sum
  intro k hk
  simp only [Finset.mem_range, not_lt] at hk
  have hcoord := congrArg (fun b : L1Sequence ↦ b (k - d)) ha
  change a ((k - d) + d) = 0 at hcoord
  rw [Nat.sub_add_cancel hk] at hcoord
  dsimp only [F]
  rw [hcoord, zero_mul]

end Transformer.AnalyticPreparation
