/-
# DASH — matrix Clenshaw and the corrected approximation target

arXiv:2602.02016v2, Appendix A. With fitting interval `[ε,1+ε]`
and `S=2A/c-I`, the approximated function is `(A/c+εI)^(-1/p)`.
Recovering a power of the original-scale regularized matrix also
requires multiplying the result by `c^(-1/p)`.
-/

import Transformer.DASH.SectionA_Coefficients
import Transformer.DASH.Section3_Scaling

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- A constant eigenvalue corresponds to a scalar identity matrix,
arXiv:2602.02016v2, §3.1 and Appendix A. -/
theorem spectralMatrix_const (Q : Matrix (Fin n) (Fin n) ℝ) (c : ℝ) (hQ : Orthogonal Q) :
    spectralMatrix Q (fun _ => c) = c • (1 : Matrix (Fin n) (Fin n) ℝ) := by
  rw [← spectralMatrix_one Q hQ, spectralMatrix_smul]
  simp only [mul_one]

/-- Constant-spectrum hypotheses are satisfiable, arXiv:2602.02016v2, Appendix A. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- Matrix coefficients `c_k I` retain the eigenbasis during the entire
Clenshaw backward loop, arXiv:2602.02016v2, Appendix A, matrix algorithm. -/
theorem clenshawState_spectrum (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ)
    (cs : List ℝ) (hQ : Orthogonal Q) :
    clenshawState (spectralMatrix Q s) (cs.map (fun c => c • (1 : Matrix (Fin n) (Fin n) ℝ))) =
      (spectralMatrix Q (fun i => (clenshawState (s i) cs).1),
        spectralMatrix Q (fun i => (clenshawState (s i) cs).2)) := by
  have htwo : (2 : Matrix (Fin n) (Fin n) ℝ) = spectralMatrix Q (fun _ => 2) := by
    calc
      (2 : Matrix (Fin n) (Fin n) ℝ) = 1 + 1 := by norm_num
      _ = spectralMatrix Q (fun _ => 2) := by
        rw [← spectralMatrix_one Q hQ, spectralMatrix_add]
        norm_num
  induction cs with
  | nil => simp [clenshawState, spectralMatrix_zero]
  | cons c cs ih =>
    simp only [List.map_cons, clenshawState, ih]
    rw [htwo, spectralMatrix_mul Q _ _ hQ, spectralMatrix_mul Q _ _ hQ,
      spectralMatrix_sub, ← spectralMatrix_const Q c hQ, spectralMatrix_add]

/-- Backward-loop spectral assumptions are satisfiable,
arXiv:2602.02016v2, Appendix A. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- Matrix Clenshaw applies the scalar polynomial independently to every
eigenvalue, arXiv:2602.02016v2, Appendix A, matrix algorithm. -/
theorem clenshaw_spectrum (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ)
    (cs : List ℝ) (hQ : Orthogonal Q) :
    clenshaw (spectralMatrix Q s) (cs.map (fun c => c • (1 : Matrix (Fin n) (Fin n) ℝ))) =
      spectralMatrix Q (fun i => clenshaw (s i) cs) := by
  simp only [clenshaw, clenshawState_spectrum Q s cs hQ]
  rw [spectralMatrix_mul Q _ _ hQ, spectralMatrix_sub]

/-- Matrix-evaluation assumptions are satisfiable, arXiv:2602.02016v2, Appendix A. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- The matrix argument used by both matrix Clenshaw algorithms,
arXiv:2602.02016v2, Appendix A, `S=2A/‖A‖_F-I`. -/
def chebMatrixArgument (A : Matrix (Fin n) (Fin n) ℝ) (c : ℝ) :
    Matrix (Fin n) (Fin n) ℝ := (2 / c) • A - 1

/-- Exact spectrum of the normalized Clenshaw argument,
arXiv:2602.02016v2, Appendix A, both matrix algorithms. -/
theorem chebMatrixArgument_spectrum (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (c : ℝ) (hQ : Orthogonal Q) :
    chebMatrixArgument (spectralMatrix Q s) c =
      spectralMatrix Q (fun i => 2 * (s i / c) - 1) := by
  rw [chebMatrixArgument, spectralMatrix_smul, ← spectralMatrix_one Q hQ, spectralMatrix_sub]
  congr 1
  funext i
  ring

/-- Argument-spectrum assumptions are satisfiable, arXiv:2602.02016v2, Appendix A. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- Correct target coordinate for coefficients fitted on `[ε,1+ε]`.
The printed output `A^(-1/p)` omits normalization and regularization.
Source: arXiv:2602.02016v2, Appendix A, matrix Clenshaw input/output. -/
theorem cheb_regularized_target_coordinate (a c ε : ℝ) :
    chebFromCoordinate ε (1 + ε) (2 * (a / c) - 1) = a / c + ε := by
  unfold chebFromCoordinate
  ring

/-- Even an exact evaluation of the normalized regularized target differs
from the original inverse square root. Here `A=16I`, `c=16`, `ε=3`, and `p=2`.
Source: arXiv:2602.02016v2, Appendix A, the printed matrix output claim. -/
theorem cheb_normalized_target_counterexample :
    ((16 : ℝ) / 16 + 3) ^ (-(1 / (2 : ℝ))) = 1 / 2 ∧
      (16 : ℝ) ^ (-(1 / (2 : ℝ))) = 1 / 4 ∧ (1 / 2 : ℝ) ≠ 1 / 4 := by
  norm_num

end Transformer.DASH
