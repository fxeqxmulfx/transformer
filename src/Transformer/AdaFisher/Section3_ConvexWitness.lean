/-
# AdaFisher: a complete block-Kronecker convex counterexample

arXiv:2405.16397v3, Proposition 3.3, Appendix A.2.
The scalar divergent eigendirection is embedded in an actual efficient
four-parameter block with positive nonconstant original factors `(1,2)`.
-/

import Transformer.AdaFisher.Section3_ConvexCorrected
import Transformer.AdaFisher.Section3_ConvexFalse
import Transformer.AdaFisher.Section3_NonconvexWitness

open scoped BigOperators
open Filter Topology

noncomputable section

namespace Transformer.AdaFisher

/-- Initial parameter vector for the block witness of Proposition 3.3. -/
def convexWitnessInitial (i : Fin 4) : ℝ := if i = 0 then 1 else 0

/-- The exact unmomented update of Proposition 3.3, with η=L=1 and the
nonconstant-factor efficient Fisher block with damping 0.001. -/
def convexWitnessRun : ℕ → Fin 4 → ℝ :=
  gradientRun 1 (fun _ => witnessFisher) id convexWitnessInitial

/-- The same block witness is an actual allowed β=0 AdaFisher run,
Algorithm 1 and Table 1, using the quadratic gradient at every iterate. -/
theorem convex_block_adaptive_run (t : ℕ) :
    adaFisherRun (fun _ => 1) 0 0 (fun _ => witnessFisher)
      (fun k => convexWitnessRun k) convexWitnessInitial t = convexWitnessRun t :=
  adaFisherRun_zero_gradient 1 (fun _ => witnessFisher) id convexWitnessInitial t

/-- The full four-dimensional run stays in the divergent eigendirection,
Proposition 3.3. Other coordinates remain zero under the actual update. -/
theorem convexWitnessRun_eq (t : ℕ) :
    convexWitnessRun t = fun i => if i = 0 then (-999 : ℝ) ^ t else 0 := by
  unfold convexWitnessRun
  induction t with
  | zero =>
    funext i
    simp [gradientRun, convexWitnessInitial]
  | succ t ih =>
    rw [gradientRun, ih]
    funext i
    fin_cases i <;>
      simp [gradientStep, witnessFisher, witnessCoordinate, fisherDiagonal, pow_succ]
    ring

/-- The genuine block witness has exponentially growing squared norm,
Proposition 3.3. This does not use a supplied scalar denominator detached
from the normalized Kronecker construction. -/
theorem convexWitnessRun_squaredNorm (t : ℕ) :
    squaredNorm (convexWitnessRun t) = (998001 : ℝ) ^ t := by
  rw [convexWitnessRun_eq]
  simp only [squaredNorm, ite_pow, zero_pow (by decide : 2 ≠ 0),
    Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  rw [← pow_mul, Nat.mul_comm t 2, pow_mul]
  norm_num

/-- All model and Fisher assumptions hold, yet the exact printed k=1
rate fails, Proposition 3.3, Appendix A.2. The objective is the ordinary
four-dimensional strongly convex Euclidean quadratic with gradient id;
its minimizer is zero, and η=1 satisfies the printed η≤1/L condition. -/
theorem convex_block_rate_counterexample :
    SmoothUpperModel (vectorQuadratic (d := 4)) id 1 ∧
    StrongLowerModel (vectorQuadratic (d := 4)) id 1 ∧
    (∀ i, (1 / 1000 : ℝ) ≤ witnessFisher i ∧ witnessFisher i ≤ 1 + 1 / 1000) ∧
    (1 : ℝ) ≤ 1 / 1 ∧
    vectorQuadratic (convexWitnessRun 1) - vectorQuadratic (0 : Fin 4 → ℝ) >
      squaredNorm (convexWitnessInitial - 0) / (2 * 1 * 1) := by
  refine ⟨vectorQuadratic_models.1, vectorQuadratic_models.2, ?_, by norm_num, ?_⟩
  · intro i
    exact fisherDiagonal_bounds (1 / 1000) _ _
      (fun j => ⟨Nat.cast_nonneg _, by fin_cases j <;> norm_num⟩)
      (fun j => ⟨Nat.cast_nonneg _, by fin_cases j <;> norm_num⟩) (witnessCoordinate i)
  · rw [vectorQuadratic, convexWitnessRun_squaredNorm]
    norm_num [vectorQuadratic, squaredNorm, convexWitnessInitial]

/-- The full block run diverges in squared Euclidean distance from the
true minimizer, Proposition 3.3, even with positive damping and a permitted
fixed step. -/
theorem convex_block_iterates_diverge :
    Tendsto (fun t => squaredNorm (convexWitnessRun t - 0)) atTop atTop := by
  simpa only [sub_zero, convexWitnessRun_squaredNorm] using
    (tendsto_pow_atTop_atTop_of_one_lt (by norm_num : (1 : ℝ) < 998001))

end Transformer.AdaFisher
