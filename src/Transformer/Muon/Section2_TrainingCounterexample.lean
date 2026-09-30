/-
# Muon — original learning can fail on a strongly convex quadratic

arXiv:2502.16982, §2.1–2.2. With zero initial momentum, coefficient zero,
zero decay and constant learning rate one, the actual five-step printed
Muon algorithm cycles on `‖W‖²/2`. This refutes unconditional training
convergence even when `η ≤ 1/L`; it is not a refutation of a theorem in
the paper, which states no general learning convergence theorem.
-/

import Transformer.Muon.Section2_TrainingCycleStep
import Transformer.Muon.Section2_TrainingConvergence

open scoped Topology
open Filter

noncomputable section

namespace Transformer.Muon

open Optimization

/-- Every actual training weight is one of the two nonzero cycle values.
The current derivative is recomputed at that weight at every step.
Source: arXiv:2502.16982, §2.1–2.2, unconditional-training counterexample. -/
theorem muon_training_cycle (t : ℕ) :
    toMatrix (muonTrainingRun 0 0 (fun _ => 1) energy (fromMatrix cycleMatrix) t).2 =
      cycleMatrix ∨
    toMatrix (muonTrainingRun 0 0 (fun _ => 1) energy (fromMatrix cycleMatrix) t).2 =
      -cycleMatrix := by
  induction t with
  | zero => exact Or.inl rfl
  | succ t ih =>
    simp only [muonTrainingRun, toMatrix_fromMatrix, energy_gradient, id_eq]
    rcases ih with h | h
    · rw [h, muon_cycle_step_unit]
      exact Or.inr rfl
    · rw [h, muon_cycle_step_neg_unit]
      exact Or.inl rfl

/-- The current full-gradient norm never approaches zero on the cycle.
Source: arXiv:2502.16982, §2.1–2.2, training counterexample. -/
theorem muon_cycle_gradient_norm (t : ℕ) :
    DASH.frobeniusNorm (toMatrix (gradient energy
      (muonTrainingRun 0 0 (fun _ => 1) energy (fromMatrix cycleMatrix) t).2)) =
      unitResponse / 10 := by
  rw [energy_gradient]
  simp only [id_eq]
  have hnorm : DASH.frobeniusNorm cycleMatrix = unitResponse / 10 := by
    rw [cycleMatrix, DASH.frobeniusNorm_smul,
      abs_of_pos (div_pos unitResponse_pos (by norm_num))]
    norm_num [DASH.frobeniusNorm, squaredFrobenius]
  rcases muon_training_cycle t with h | h
  · rw [h, hnorm]
  · rw [h]
    have hneg : -cycleMatrix = (-1 : ℝ) • cycleMatrix := by simp
    rw [hneg, DASH.frobeniusNorm_smul, hnorm]
    norm_num

/-- Refutation of unconditional learning convergence of original Muon:
the objective is genuinely smooth, strongly convex and lower bounded,
the step is positive and satisfies `η=1/L`, yet the true gradients do
not tend to zero. All printed five matrix iterations are included.
Source: arXiv:2502.16982, §2.1–2.2, training counterexample. -/
theorem muon_unconditional_training_counterexample :
    SmoothObjective (energy : MatrixSpace 1 1 → ℝ) 1 ∧
      StrongLowerModel (energy : MatrixSpace 1 1 → ℝ) 1 ∧
      (∀ W : MatrixSpace 1 1, 0 ≤ energy W) ∧
      ¬ Tendsto (fun t : ℕ => DASH.frobeniusNorm (toMatrix (gradient energy
        (muonTrainingRun 0 0 (fun _ => 1) energy (fromMatrix cycleMatrix) t).2)))
        atTop (𝓝 0) := by
  refine ⟨energy_models.1, energy_models.2,
    fun W => div_nonneg (sq_nonneg _) (by norm_num), ?_⟩
  intro h
  simp only [muon_cycle_gradient_norm] at h
  have hc := tendsto_nhds_unique h (tendsto_const_nhds (x := unitResponse / 10))
  have hpos : 0 < unitResponse / 10 := div_pos unitResponse_pos (by norm_num)
  linarith

/-- The new safeguard repairs this same quadratic example: actual Muon
candidate computation is retained and the corrected weights converge
to the zero global minimizer. Source: training correction to
arXiv:2502.16982, §2.1–2.2. -/
theorem safeguardedMuon_repairs_cycle :
    Tendsto (fun t : ℕ => (safeguardedMuonRun (1 / 2) 1 0 0 energy
      (fromMatrix cycleMatrix) t).2) atTop (𝓝 0) := by
  exact (safeguardedMuon_minimum_convergence (1 / 2) 1 0 0 1 energy
    (fromMatrix cycleMatrix) 0 energy_models.1 energy_models.2
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by rw [energy_gradient]; rfl)).2.1

end Transformer.Muon
