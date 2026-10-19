import Transformer.ScheduleFree.AMSGrad
import Transformer.MagnitudeDirection.SectionA_TrainingQuadratic

/-!
# Kernel-checked checkpoints for the two new Schedule-Free AMSGrad variants

These use the actual recurrences, with explicit expected values also frozen
in `optimizers/tests/test_schedule_free_amsgrad.py`. They check the local
composition of arXiv:2405.15682v4 §4.4, arXiv:1904.03590v4 Algorithm 1 and
arXiv:2606.25971v2 Appendix A Algorithm 2. They are finite recurrence checks;
the general invariants and query chain rule are proved in the source modules.
-/

noncomputable section

namespace Transformer.ScheduleFree.AMSGrad.Checkpoints

open MagnitudeDirection

private def wState : State (Fin 1) :=
  run ⟨1, 0, 0, 1/2, 0, 1⟩ (fun t _ _ => if t = 0 then 2 else 1) (fun _ => 2) 2

/-- Reachable two-step raw variance decreases, while the running maximum stays 4. -/
theorem w_reachable : wState.second 0 = 1 ∧ wState.maximum 0 = 4 ∧
    wState.base 0 = 1 ∧ wState.evaluation 0 = 7/6 ∧
    query ⟨1, 0, 0, 1/2, 0, 1⟩ wState 0 = 13/12 := by
  norm_num [wState, run, step, stepWith, initial, proposal, query, nextSecond,
    nextMaximum, coefficient, adamLearningRate, warmupFactor]

private def sphereParameters : MDParameters := ⟨4, 0, 0, 1/2, 0, 1⟩

private def sphereInitial : Matrix (Fin 1) (Fin 1) ℝ := fun _ _ => 2

private theorem scalar_projection (radius : ℝ) (direction : Fin 1 × Fin 1 → ℝ)
    (ij : Fin 1 × Fin 1) : projectDirection radius direction ij =
    radius / |direction (0, 0)| * direction (0, 0) := by
  rcases ij with ⟨i, j⟩
  fin_cases i
  fin_cases j
  let W : Matrix (Fin 1) (Fin 1) ℝ := fun i j => direction (i, j)
  have hW : frobeniusNorm W = |direction (0, 0)| := single_frobeniusNorm W
  change radius / frobeniusNorm W * direction (0, 0) = _
  rw [hW]

private def sphereState : MDState 1 1 :=
  mdRun sphereParameters (fun t _ _ _ => if t = 0 then 2 else -2) sphereInitial 2

/-- Two sign reversals leave the base on radius 2, its factor average at zero,
and the actual fused next training query at 1. Gains are frozen at unit scale. -/
theorem md_sphere_average : mdBaseDirection sphereState 0 0 = 2 ∧
    mdEvaluationWeight sphereState 0 0 = 0 ∧
    mdQueryWeight sphereParameters sphereState 0 0 = 1 := by
  have half (x : ℝ) : 1/2 * x + 1/2 * x = x := by ring
  have hnorm : frobeniusNorm sphereInitial = 2 := by
    rw [single_frobeniusNorm]
    norm_num [sphereInitial]
  norm_num [sphereState, sphereParameters, mdRun, mdStep, mdGradients, mdQuery,
    mdInitial, mdBaseDirection, mdEvaluationWeight, mdEvaluation, factorWeight,
    mdQueryWeight, mdDirectionParameters, mdGainParameters, step, stepWith,
    initial, proposal, query, nextSecond, nextMaximum, coefficient, adamLearningRate,
    warmupFactor, scalar_projection, hnorm, sphereInitial,
    directionGradient, rowGradient, colGradient, fuse, softplus_unit_initialization,
    half]

private def gainParameters : MDParameters := ⟨0, 1, 0, 3/5, 0, 1⟩

private def gainInitial : MDState 1 1 :=
  ⟨initial (fun _ => 2), initial (fun _ => 0), initial (fun _ => 0), 2⟩

private def gainState : MDState 1 1 :=
  mdStep gainParameters 0 gainInitial (fun _ _ => 1 / Real.log 2)

/-- With raw gains zero, G=1/log(2) gives both true raw-gain gradients 1.
The direction is frozen and both updated raw gains are exactly -1/2. -/
theorem md_gain_update : gainState.row.base 0 = -1/2 ∧
    gainState.column.base 0 = -1/2 ∧ gainState.row.second 0 = 1 ∧
    gainState.column.second 0 = 1 ∧
    mdQueryWeight gainParameters gainState 0 0 = 2 * softplus (-1/2) ^ 2 := by
  have hlog : Real.log 2 ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
  have hgradient : Real.log 2 * 2 * (Real.log 2)⁻¹ * (1/2) = 1 := by
    field_simp
  norm_num [gainState, gainInitial, gainParameters, mdStep, mdGradients, mdQuery,
    mdQueryWeight, factorWeight, mdDirectionParameters, mdGainParameters,
    step, stepWith, initial, proposal, query, nextSecond, nextMaximum,
    coefficient, adamLearningRate, warmupFactor, directionGradient,
    rowGradient, colGradient, fuse, softplus, softplusDerivative, hlog, hgradient]
  ring

#print axioms w_reachable
#print axioms md_sphere_average
#print axioms md_gain_update
#print axioms maximum_mono
#print axioms coefficient_bounds
#print axioms md_base_norm
#print axioms md_query_hasDerivAt

end Transformer.ScheduleFree.AMSGrad.Checkpoints
