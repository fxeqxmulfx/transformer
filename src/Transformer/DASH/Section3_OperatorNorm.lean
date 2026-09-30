/-
# DASH — Euclidean operator norms in the eigenframe

arXiv:2602.02016v2, §3.3–3.4. The matrix norm here is the actual
Euclidean operator norm, not the entrywise matrix norm. Orthogonal
conjugation identifies it with the largest absolute eigenvalue.
-/

import Transformer.DASH.Section3_Scaling
import Mathlib.Analysis.CStarAlgebra.Matrix

open scoped Matrix Matrix.Norms.L2Operator

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- The Euclidean operator norm of a real symmetric spectral matrix is
the supremum norm of its eigenvalue vector, including their absolute values.
The source calls it the largest eigenvalue; that wording is correct on the
positive-semidefinite Shampoo domain, not for all symmetric matrices.
Source: arXiv:2602.02016v2, §3.4, the definition of `‖A‖₂`. -/
theorem spectralMatrix_opNorm (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (hQ : Orthogonal Q) : ‖spectralMatrix Q s‖ = ‖s‖ := by
  have hU : Q ∈ unitary (Matrix (Fin n) (Fin n) ℝ) := by
    apply Matrix.mem_unitaryGroup_iff.2
    simpa only [Matrix.star_eq_conjTranspose, Matrix.conjTranspose_eq_transpose_of_trivial]
      using hQ.2
  have hUt : Q.transpose ∈ unitary (Matrix (Fin n) (Fin n) ℝ) := by
    apply Matrix.mem_unitaryGroup_iff.2
    simpa only [Matrix.star_eq_conjTranspose, Matrix.conjTranspose_eq_transpose_of_trivial,
      Matrix.transpose_transpose, Muon.OrthonormalColumns] using hQ.1
  rw [spectralMatrix, Muon.singularMatrix, CStarRing.norm_mul_mem_unitary _ hUt,
    CStarRing.norm_mem_unitary_mul _ hU, Matrix.l2_opNorm_diagonal]

/-- An orthogonal eigenframe exists, arXiv:2602.02016v2, §3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- An operator-norm bound is exactly a bound on every absolute
eigenvalue. Source: arXiv:2602.02016v2, §3.4, spectral scaling. -/
theorem spectralMatrix_opNorm_le_iff (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (δ : ℝ) (hQ : Orthogonal Q) (hδ : 0 ≤ δ) :
    ‖spectralMatrix Q s‖ ≤ δ ↔ ∀ i, |s i| ≤ δ := by
  rw [spectralMatrix_opNorm Q s hQ, pi_norm_le_iff_of_nonneg hδ]
  simp only [Real.norm_eq_abs]

/-- Nonnegative operator bounds have a nonempty instance,
arXiv:2602.02016v2, §3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧ (0 : ℝ) ≤ 1 := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

/-- A strict operator-norm bound is exactly a strict bound on every
absolute eigenvalue. Source: arXiv:2602.02016v2, §3.3–3.4, solver domains. -/
theorem spectralMatrix_opNorm_lt_iff (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (δ : ℝ) (hQ : Orthogonal Q) (hδ : 0 < δ) :
    ‖spectralMatrix Q s‖ < δ ↔ ∀ i, |s i| < δ := by
  rw [spectralMatrix_opNorm Q s hQ, pi_norm_lt_iff hδ]
  simp only [Real.norm_eq_abs]

/-- Positive strict bounds coexist with an orthogonal frame,
arXiv:2602.02016v2, §3.3–3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧ (0 : ℝ) < 2 := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

/-- On the nonnegative Shampoo spectrum, the operator norm equals an
attained largest eigenvalue. This supplies the positivity qualification
missing from the source's parenthetical definition.
Source: arXiv:2602.02016v2, §3.4, operator norm and largest eigenvalue. -/
theorem spectralMatrix_opNorm_eq_largest (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (μ : ℝ) (hQ : Orthogonal Q) (hs : ∀ i, 0 ≤ s i)
    (hupper : ∀ i, s i ≤ μ) (htop : ∃ i, s i = μ) :
    ‖spectralMatrix Q s‖ = μ := by
  obtain ⟨j, hj⟩ := htop
  have hμ : 0 ≤ μ := hj ▸ hs j
  apply le_antisymm
  · apply (spectralMatrix_opNorm_le_iff Q s μ hQ hμ).2
    intro i
    simpa only [abs_of_nonneg (hs i)] using hupper i
  · rw [spectralMatrix_opNorm Q s hQ]
    have h := norm_le_pi_norm s j
    simpa only [Real.norm_eq_abs, hj, abs_of_nonneg hμ] using h

/-- A nonnegative spectrum attains its largest value,
arXiv:2602.02016v2, §3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (∀ i : Fin 1, (0 : ℝ) ≤ (fun _ => (1 : ℝ)) i) ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i ≤ 1) ∧
    (∃ i : Fin 1, (fun _ => (1 : ℝ)) i = 1) := by
  refine ⟨by simp [Orthogonal, Muon.OrthonormalColumns], by norm_num,
    by norm_num, 0, rfl⟩

/-- Frobenius norm bounds the actual Euclidean operator norm, not merely
individual eigenvalues. Source: arXiv:2602.02016v2, §3.4, Frobenius scaling. -/
theorem spectralMatrix_opNorm_le_frobenius (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (hQ : Orthogonal Q) :
    ‖spectralMatrix Q s‖ ≤ frobeniusNorm (spectralMatrix Q s) := by
  apply (spectralMatrix_opNorm_le_iff Q s _ hQ (by unfold frobeniusNorm; positivity)).2
  exact eigenvalue_le_frobenius Q s hQ

/-- The operator/Frobenius comparison has an orthogonal frame,
arXiv:2602.02016v2, §3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

/-- The symmetric one-dimensional matrix `[-1]` has operator norm one,
whereas its only, hence largest, eigenvalue is minus one. This refutes
the source's unrestricted identification of operator norm with largest
eigenvalue. Source: arXiv:2602.02016v2, §3.4, operator-norm parenthesis. -/
theorem symmetric_operator_norm_counterexample :
    ‖(-1 : Matrix (Fin 1) (Fin 1) ℝ)‖ = 1 ∧
      ‖(-1 : Matrix (Fin 1) (Fin 1) ℝ)‖ ≠ (-1 : ℝ) := by
  constructor <;> norm_num

end Transformer.DASH
