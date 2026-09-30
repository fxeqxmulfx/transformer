/-
# DASH — finite inverse-root convergence does not prove learning convergence

arXiv:2602.02016v2, §2–3. This actual sequential NDB/grafting run has
nonzero PI starts, a positive regularizer, source EMA coefficient `1/2`,
zero initial states and learning rate one. It cycles on the strongly
convex `L=1` quadratic. Numerical scaling has already been corrected;
the additional training guard is what eliminates this learning failure.
The manuscript states no general training convergence theorem to refute.
-/

import Transformer.DASH.Section2_TrainingCycleStep
import Transformer.DASH.Section2_TrainingConvergence

open scoped Topology
open Filter

noncomputable section

namespace Transformer.DASH

open Optimization

/-- Actual quadratic training instance with fixed finite root budgets.
`ν=0` uses the current squared gradient for the Adam reference.
Source: arXiv:2602.02016v2, §2.3 and §3.3–3.5, training counterexample. -/
def dashCycleRun : ℕ → TrainingState 1 1 × MatrixSpace 1 1 :=
  dashTrainingRun (1 / 2) 0 0 (1 / 4) cycleStarts cycleStarts 0 1 (fun _ => 1)
    energy (fromMatrix ((1 / 4 : ℝ) • (1 : Matrix (Fin 1) (Fin 1) ℝ)))

/-- The actual state keeps both EMA matrices nonnegative and its current
weight has amplitude `1/4` forever. The optimizer computes each new
gradient from these weights. Source: arXiv:2602.02016v2, §2–3,
unconditional-training counterexample. -/
theorem dash_training_cycle (t : ℕ) :
    0 ≤ (dashCycleRun t).1.left 0 0 ∧ 0 ≤ (dashCycleRun t).1.right 0 0 ∧
      ∃ s : ℝ, s ^ 2 = 1 / 16 ∧
        (dashCycleRun t).2 = fromMatrix (s • (1 : Matrix (Fin 1) (Fin 1) ℝ)) := by
  induction t with
  | zero =>
    refine ⟨by norm_num [dashCycleRun, dashTrainingRun, initialTrainingState],
      by norm_num [dashCycleRun, dashTrainingRun, initialTrainingState], 1 / 4,
      by norm_num, rfl⟩
  | succ t ih =>
    obtain ⟨hl, hr, s, hs, hweight⟩ := ih
    have hc := dash_cycle_candidate (dashCycleRun t).1 s hl hr hs
    change 0 ≤ (dashTrainingCandidate (1 / 2) 0 0 (1 / 4) cycleStarts cycleStarts 0 1
        (dashCycleRun t).1 (gradient energy (dashCycleRun t).2)).1.left 0 0 ∧
      0 ≤ (dashTrainingCandidate (1 / 2) 0 0 (1 / 4) cycleStarts cycleStarts 0 1
        (dashCycleRun t).1 (gradient energy (dashCycleRun t).2)).1.right 0 0 ∧
      ∃ u : ℝ, u ^ 2 = 1 / 16 ∧
        (dashCycleRun t).2 - (1 : ℝ) •
          (dashTrainingCandidate (1 / 2) 0 0 (1 / 4) cycleStarts cycleStarts 0 1
            (dashCycleRun t).1 (gradient energy (dashCycleRun t).2)).2 =
              fromMatrix (u • (1 : Matrix (Fin 1) (Fin 1) ℝ))
    rw [energy_gradient]
    simp only [id_eq, one_smul, hweight]
    refine ⟨hc.2.1, hc.2.2, -s, by simpa only [neg_sq] using hs, ?_⟩
    rw [hc.1]
    ext ij
    simp [fromMatrix]
    ring

/-- The true full-gradient norm is constantly `1/4`, independently of
the evolving left and right histories. Source: arXiv:2602.02016v2,
§2–3, training counterexample. -/
theorem dash_cycle_gradient_norm (t : ℕ) :
    frobeniusNorm (toMatrix (gradient energy (dashCycleRun t).2)) = 1 / 4 := by
  obtain ⟨_, _, s, hs, hweight⟩ := dash_training_cycle t
  rw [energy_gradient]
  simp only [id_eq, hweight, toMatrix_fromMatrix]
  have henergy : Muon.squaredFrobenius (s • (1 : Matrix (Fin 1) (Fin 1) ℝ)) = s ^ 2 := by
    norm_num [Muon.squaredFrobenius]
  rw [frobeniusNorm, henergy, hs]
  norm_num

/-- Refutation of unconditional training convergence for the unguarded
finite NDB/grafting algorithm, even after its numerical scaling repair.
The loss is smooth, strongly convex and lower bounded, and `η=1/L`.
Source: arXiv:2602.02016v2, §2–3, training counterexample. -/
theorem dash_unconditional_training_counterexample :
    SmoothObjective (energy : MatrixSpace 1 1 → ℝ) 1 ∧
      StrongLowerModel (energy : MatrixSpace 1 1 → ℝ) 1 ∧
      (∀ W : MatrixSpace 1 1, 0 ≤ energy W) ∧
      ¬ Tendsto (fun t : ℕ => frobeniusNorm (toMatrix (gradient energy (dashCycleRun t).2)))
        atTop (𝓝 0) := by
  refine ⟨energy_models.1, energy_models.2,
    fun W => div_nonneg (sq_nonneg _) (by norm_num), ?_⟩
  intro h
  simp only [dash_cycle_gradient_norm, tendsto_const_nhds_iff] at h
  norm_num at h

/-- The corrected full DASH recurrence converges to the zero global
minimum on this same example, keeping its actual finite NDB/grafting
candidate computations. Source: training correction to
arXiv:2602.02016v2, §2–4. -/
theorem safeguardedDash_repairs_cycle :
    Tendsto (fun t : ℕ => (safeguardedDashRun (1 / 2) 1 (1 / 2) 0 0 (1 / 4)
      cycleStarts cycleStarts 0 1 energy
      (fromMatrix ((1 / 4 : ℝ) • (1 : Matrix (Fin 1) (Fin 1) ℝ))) t).2)
      atTop (𝓝 0) := by
  exact (safeguardedDash_minimum_convergence (1 / 2) 1 (1 / 2) 0 0 (1 / 4) 1
    cycleStarts cycleStarts 0 1 energy
    (fromMatrix ((1 / 4 : ℝ) • (1 : Matrix (Fin 1) (Fin 1) ℝ))) 0
    energy_models.1 energy_models.2 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by rw [energy_gradient]; rfl)).2.1

end Transformer.DASH
