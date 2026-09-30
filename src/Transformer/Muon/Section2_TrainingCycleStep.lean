/-
# Muon — a quadratic training cycle with the printed five-step update

arXiv:2502.16982, §2.1–2.2. The manuscript gives no general learning
convergence theorem. This counterexample refutes an unconditional
interpretation of the original optimizer: normalized finite updates with
a fixed positive learning rate can cycle even on a strongly convex loss.
It does not refute the manuscript's empirical training observations.
-/

import Transformer.Muon.Section2_TrainingModels
import Transformer.Muon.Section2_NewtonSchulz
import Transformer.Optimization.Quadratic

open scoped Matrix

noncomputable section

namespace Transformer.Muon

open Optimization

/-- Exact positive response of the printed five-step scalar iteration.
Source: arXiv:2502.16982, §2.1, `eq:iteration`, and §2.2, iteration budget. -/
def unitResponse : ℝ :=
  scalarSchulz (34445 / 10000) (-47750 / 10000) (20315 / 10000) 1 5

/-- Positivity of the actual finite response is checked with exact
rational arithmetic. Source: arXiv:2502.16982, §2.1–2.2. -/
theorem unitResponse_pos : 0 < unitResponse := by
  norm_num [unitResponse, scalarSchulz, schulzPolynomial]

/-- The identity's actual five-step matrix output has the scalar response.
Source: arXiv:2502.16982, §2.1, `eq:iteration`, and §2.2. -/
theorem approximatePolar_unit :
    approximatePolar (1 : Matrix (Fin 1) (Fin 1) ℝ) = unitResponse • 1 := by
  have hU : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
    simp [OrthonormalColumns]
  have h := schulzIterate_spectrum (34445 / 10000) (-47750 / 10000) (20315 / 10000)
    (1 : Matrix (Fin 1) (Fin 1) ℝ) (fun _ => 1) 1 hU hU 5
  norm_num [singularMatrix, squaredFrobenius] at h
  change schulzIterate (34445 / 10000) (-47750 / 10000) (20315 / 10000) 1 5 = _
  norm_num only
  rw [h]
  ext i j
  fin_cases i
  fin_cases j
  norm_num only [unitResponse, Matrix.smul_apply, Matrix.one_apply, Matrix.diagonal_apply,
    ite_true, smul_eq_mul, mul_one]

/-- Oddness of every finite printed update, including Frobenius
normalization, is essential to the training cycle.
Source: arXiv:2502.16982, §2.1, `eq:iteration`. -/
theorem schulzIterate_neg {a b : ℕ} (α β γ : ℝ)
    (X : Matrix (Fin a) (Fin b) ℝ) (k : ℕ) :
    schulzIterate α β γ (-X) k = -schulzIterate α β γ X k := by
  have henergy : squaredFrobenius (-X) = squaredFrobenius X := by
    simp [squaredFrobenius]
  induction k with
  | zero => simp [schulzIterate, schulzInitial, henergy]
  | succ k ih =>
    rw [schulzIterate, ih, schulzIterate]
    simp [schulzStep, Matrix.transpose_neg, Matrix.neg_mul, Matrix.mul_neg,
      add_comm, add_assoc]

/-- Positive rescaling is removed by the actual initial Frobenius
normalization, including the zero-input case. Source: arXiv:2502.16982,
§2.1, before `eq:iteration`. -/
theorem schulzIterate_pos_smul {a b : ℕ} (α β γ c : ℝ)
    (X : Matrix (Fin a) (Fin b) ℝ) (hc : 0 < c) (k : ℕ) :
    schulzIterate α β γ (c • X) k = schulzIterate α β γ X k := by
  have hinit : schulzInitial (c • X) = schulzInitial X := by
    rw [schulzInitial, squaredFrobenius_smul, Real.sqrt_mul (sq_nonneg c),
      Real.sqrt_sq_eq_abs, abs_of_pos hc, mul_inv_rev, smul_smul]
    rw [mul_assoc, inv_mul_cancel₀ hc.ne', mul_one]
    rfl
  induction k with
  | zero => exact hinit
  | succ k ih => simp only [schulzIterate, ih]

/-- Positive-rescaling hypotheses hold for a nonzero input,
arXiv:2502.16982, §2.1, training counterexample. -/
example : (0 : ℝ) < 1 := by norm_num

/-- Nonzero cycle weights of amplitude `unitResponse/10`. A learning
rate of one, within the quadratic's usual smoothness bound `1/L`,
still produces a step twice this amplitude because Muon normalizes its
input. Source: arXiv:2502.16982, §2.1–2.2, training counterexample. -/
def cycleMatrix : Matrix (Fin 1) (Fin 1) ℝ := (unitResponse / 10) • 1

/-- The actual five-step update flips the positive cycle weight, using
its true quadratic gradient, zero decay and zero momentum coefficient.
The learning rate is one, not an oversized step for the loss's `L=1`.
Source: arXiv:2502.16982, §2.1–2.2, unconditional-training counterexample. -/
theorem muon_cycle_step_unit (M : Matrix (Fin 1) (Fin 1) ℝ) :
    (muonStep 0 1 0 M cycleMatrix cycleMatrix).2 = -cycleMatrix := by
  have hc : 0 < unitResponse / 10 := div_pos unitResponse_pos (by norm_num)
  have hpolar : approximatePolar cycleMatrix = unitResponse • 1 := by
    rw [approximatePolar, cycleMatrix, schulzIterate_pos_smul _ _ _ _ _ hc]
    exact approximatePolar_unit
  simp only [muonStep, nesterovInput, zero_smul, zero_add, hpolar]
  ext i j
  simp [decayStep, adjustedUpdate, shapeScale, cycleMatrix]
  ring

/-- The same paper update flips the negative cycle weight to the positive
one, with its actual quadratic gradient. Source: arXiv:2502.16982,
§2.1–2.2, unconditional-training counterexample. -/
theorem muon_cycle_step_neg_unit (M : Matrix (Fin 1) (Fin 1) ℝ) :
    (muonStep 0 1 0 M (-cycleMatrix) (-cycleMatrix)).2 = cycleMatrix := by
  have hc : 0 < unitResponse / 10 := div_pos unitResponse_pos (by norm_num)
  have hpolar : approximatePolar (-cycleMatrix) = -(unitResponse • 1) := by
    rw [approximatePolar, schulzIterate_neg, cycleMatrix,
      schulzIterate_pos_smul _ _ _ _ _ hc]
    exact congrArg Neg.neg approximatePolar_unit
  simp only [muonStep, nesterovInput, zero_smul, zero_add, hpolar]
  ext i j
  simp [decayStep, adjustedUpdate, shapeScale, cycleMatrix]
  ring

end Transformer.Muon
