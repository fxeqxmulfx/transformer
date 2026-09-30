/-
# DASH — existence of the EVD data used by the solvers

arXiv:2602.02016v2, §3.1. The spectral representation is
available for every real symmetric preconditioner, and positive
definiteness supplies the positive spectrum required for inverse roots.
-/

import Transformer.DASH.Section3_MatrixConvergence
import Mathlib.Analysis.Matrix.PosDef

open scoped Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- The eigenvector matrix provided by symmetric eigendecomposition,
arXiv:2602.02016v2, §3.1, `A=QΛQᵀ`. -/
def evdFrame (A : Matrix (Fin n) (Fin n) ℝ) (hA : A.IsHermitian) :
    Matrix (Fin n) (Fin n) ℝ := hA.eigenvectorUnitary

/-- The real EVD frame is orthogonal, arXiv:2602.02016v2, §3.1. -/
theorem evdFrame_orthogonal (A : Matrix (Fin n) (Fin n) ℝ) (hA : A.IsHermitian) :
    Orthogonal (evdFrame A hA) := by
  constructor
  · simpa only [evdFrame, Muon.OrthonormalColumns, Matrix.star_eq_conjTranspose,
      Matrix.conjTranspose_eq_transpose_of_trivial] using
      (Matrix.mem_unitaryGroup_iff').mp hA.eigenvectorUnitary.property
  · simpa only [evdFrame, Matrix.star_eq_conjTranspose,
      Matrix.conjTranspose_eq_transpose_of_trivial] using
      (Matrix.mem_unitaryGroup_iff).mp hA.eigenvectorUnitary.property

/-- Symmetric inputs exist, arXiv:2602.02016v2, §3.1. -/
example : (1 : Matrix (Fin 1) (Fin 1) ℝ).IsHermitian := by simp

/-- Every real symmetric input has the spectral representation used in
the CN/NDB proofs, arXiv:2602.02016v2, §3.1, the eigendecomposition formula. -/
theorem evd_reconstruction (A : Matrix (Fin n) (Fin n) ℝ) (hA : A.IsHermitian) :
    A = spectralMatrix (evdFrame A hA) hA.eigenvalues := by
  simpa [spectralMatrix, Muon.singularMatrix, evdFrame, Unitary.conjStarAlgAut_apply,
    Matrix.star_eq_conjTranspose, Matrix.conjTranspose_eq_transpose_of_trivial,
    Function.comp_def] using hA.spectral_theorem

/-- Reconstruction hypotheses are satisfiable, arXiv:2602.02016v2, §3.1. -/
example : (1 : Matrix (Fin 1) (Fin 1) ℝ).IsHermitian := by simp

/-- Positive-definite preconditioners have a strictly positive EVD spectrum.
This makes the inverse-root solver domain a property of the original matrix.
Source: arXiv:2602.02016v2, §3.1–3.3, regularized inverse roots. -/
theorem posDef_spectral_representation (A : Matrix (Fin n) (Fin n) ℝ) (hA : A.PosDef) :
    ∃ (Q : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ),
      Orthogonal Q ∧ (∀ i, 0 < s i) ∧ A = spectralMatrix Q s :=
  ⟨evdFrame A hA.1, hA.1.eigenvalues, evdFrame_orthogonal A hA.1,
    hA.eigenvalues_pos, evd_reconstruction A hA.1⟩

/-- Positive-definite inputs exist, arXiv:2602.02016v2, §3.1–3.3. -/
example : (1 : Matrix (Fin 1) (Fin 1) ℝ).PosDef := by
  simpa using Matrix.PosDef.diagonal (fun _ : Fin 1 => (by norm_num : (0 : ℝ) < 1))

end Transformer.DASH
