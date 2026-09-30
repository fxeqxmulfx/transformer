/-
# AdaFisher — a complete nonconstant-factor convergence witness

arXiv:2405.16397v3, Algorithm 1, §3.4 and Appendix A.3.
The four-parameter quadratic uses nonzero momentum and bias correction.
Both measured factor diagonals are one-site FC Gram diagonals, computed
from current weights and true gradients. This is a concrete factor-input
witness, not an identification of every objective's Hessian with Fisher.
-/

import Transformer.AdaFisher.Section3_TrainingMinimum
import Transformer.AdaFisher.SectionA_Factors

open scoped Topology
open Filter

noncomputable section

namespace Transformer.AdaFisher

open Optimization

/-- A nonoptimal four-parameter initialization, arXiv:2405.16397v3,
§3.4, deterministic training witness. -/
def trainingWitnessInitial : TrainingSpace 2 2 :=
  WithLp.toLp 2 (fun i => (i : ℝ) + 1)

/-- One-site squared activation and sensitivity measurements using the
source's FC factor formula. Source: arXiv:2405.16397v3, Appendix A.3.
They depend on actual weights and gradients and are processed by both
EMA buffers before the min-max/damped Fisher solve. -/
def trainingWitnessFactors : TrainingFactors 2 2 := fun x g =>
  (fullyConnectedFactor (fun _ : Fin 1 => fun i => x (finProdFinEquiv (i, 0))),
    fullyConnectedFactor (fun _ : Fin 1 => fun j => g (finProdFinEquiv (0, j))))

/-- The concrete source-form factor inputs have nonzero ranges in both
dimensions, so the witness does not use the constant-factor convention.
Source: arXiv:2405.16397v3, Proposition 3.2 and Appendix A.3. -/
theorem trainingWitness_factors_nonconstant :
    let fresh := trainingWitnessFactors trainingWitnessInitial (gradient energy trainingWitnessInitial)
    fresh.1 0 < fresh.1 1 ∧ fresh.2 0 < fresh.2 1 := by
  norm_num [trainingWitnessFactors, trainingWitnessInitial, energy_gradient,
    fullyConnectedFactor, finProdFinEquiv]

/-- The actual first EMA buffers and Fisher entries are nonconstant.
Source: arXiv:2405.16397v3, Algorithm 1 and Proposition 3.2.
This checks fresh measurements, both EMAs, min-max scaling and damping
inside the complete state transition with beta=0.9. -/
theorem trainingWitness_first_buffers :
    let s := trainingRun (1 / 40000) (9 / 10) (4 / 5) (1 / 1000) energy
      trainingWitnessFactors trainingWitnessInitial 1
    s.factorH 0 = 4 / 5 ∧ s.factorH 1 = 36 / 5 ∧
      s.factorS 0 = 4 / 5 ∧ s.factorS 1 = 16 / 5 ∧
      trainingMetric (1 / 1000) s 0 = 1 / 1000 ∧
      trainingMetric (1 / 1000) s 3 = 1001 / 1000 := by
  norm_num [trainingRun, trainingStep, optimizerStep, initializeOptimizer,
    trainingPosition, trainingWitnessInitial, trainingWitnessFactors,
    fullyConnectedFactor, energy_gradient, factorEMA, trainingMetric,
    normalizedFisher, fisherDiagonal, minMaxDiagonal, diagonalMin,
    diagonalMax, Finset.univ_fin2, finProdFinEquiv]

/-- The first true parameter moves, and therefore so does its next
measured activation factor. Source: arXiv:2405.16397v3, Algorithm 1 and
Appendix A.3. The theorem concerns actual stateful training rather than a
prescribed gradient stream or a permanently frozen initialization. -/
theorem trainingWitness_first_motion :
    let s := trainingRun (1 / 40000) (9 / 10) (4 / 5) (1 / 1000) energy
      trainingWitnessFactors trainingWitnessInitial 1
    trainingPosition s 0 = 39 / 40 ∧
      (trainingWitnessFactors (trainingPosition s) (gradient energy (trainingPosition s))).1 0 =
        (39 / 40 : ℝ) ^ 2 := by
  norm_num [trainingRun, trainingStep, optimizerStep, initializeOptimizer,
    trainingPosition, trainingWitnessInitial, trainingWitnessFactors,
    fullyConnectedFactor, energy_gradient, factorEMA, parameterStep,
    normalizedFisher, fisherDiagonal, minMaxDiagonal, diagonalMin,
    diagonalMax, Finset.univ_fin2, finProdFinEquiv]

/-- The full deterministic source update, with genuinely computed
nonconstant KF factors, converges in its actual last weights and loss.
Source: arXiv:2405.16397v3, §3.4, explicit constant-step/momentum correction.
The source's defaults beta=0.9, gamma=0.8 and damping=0.001 are used.
The step is 1/40000, satisfying the explicit sufficient step condition. -/
theorem trainingWitness_converges :
    Tendsto (fun t : ℕ => trainingPosition
      (trainingRun (1 / 40000) (9 / 10) (4 / 5) (1 / 1000) energy trainingWitnessFactors trainingWitnessInitial t))
      atTop (𝓝 0) ∧
    Tendsto (fun t : ℕ => energy (trainingPosition
      (trainingRun (1 / 40000) (9 / 10) (4 / 5) (1 / 1000) energy trainingWitnessFactors trainingWitnessInitial t)))
      atTop (𝓝 0) := by
  have h := trainingRun_strong_convergence (1 / 40000) (9 / 10) (4 / 5) (1 / 1000) 1 1
    energy trainingWitnessFactors trainingWitnessInitial 0 energy_models.1 energy_lipschitz
    energy_models.2 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by rw [energy_gradient]; rfl)
  simpa only [energy, norm_zero, zero_pow (by decide : 2 ≠ 0), zero_div] using h.2

end Transformer.AdaFisher
