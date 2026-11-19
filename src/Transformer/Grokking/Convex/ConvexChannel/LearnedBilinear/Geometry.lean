module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear.Basic
public import Transformer.Grokking.FlatSpectrum.Roots
public import Mathlib.Analysis.Matrix.Order

/-!
# The actual recovered representations have trainable internal geometry

Source: `review/convex/moments.md`, CM86. The two factors are balanced at every
singular direction, not only in total norm. Their row Gram matrices are the
positive square roots of the two joint Gram matrices. On a positive square
joint matrix, each normalized representation has covariance M / ‖M‖_S1.
These identities concern the actual spectral recovery, independently of the
arbitrary eigenbasis at repeated eigenvalues; no alternative factors are used.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear

open scoped BigOperators Matrix ComplexOrder MatrixOrder Matrix.Norms.Elementwise
open FlatSpectrum

variable {Q D : Type*} [Fintype Q] [Fintype D] [DecidableEq Q] [DecidableEq D]

omit [DecidableEq D] in
/-- The query factor has exactly one singular value in each hidden Gram
coordinate, CM86. Zero singular directions are retained. -/
theorem left_hidden_gram (M : Matrix Q D ℂ) :
    (leftFactor M).conjTranspose * leftFactor M =
      Matrix.diagonal (fun q => (values M q : ℂ)) := by
  have hu : (basis M).conjTranspose * basis M = 1 := eigenvectorUnitary_conjTranspose_mul _
  unfold leftFactor
  rw [Matrix.conjTranspose_mul, conjTranspose_diagonal_ofReal]
  calc
    _ = Matrix.diagonal (fun q => (Real.sqrt (values M q) : ℂ)) *
        ((basis M).conjTranspose * basis M) *
        Matrix.diagonal (fun q => (Real.sqrt (values M q) : ℂ)) := by
      simp only [Matrix.mul_assoc]
    _ = _ := by
      rw [hu, Matrix.mul_one, Matrix.diagonal_mul_diagonal]
      congr 1
      funext q
      have hv : 0 ≤ values M q := Real.sqrt_nonneg _
      rw [← Complex.ofReal_mul, Real.mul_self_sqrt hv]

omit [DecidableEq D] in
/-- The context factor has the same full hidden Gram as the query factor,
CM86. The identity includes rank-deficient matrices. -/
theorem right_hidden_gram (M : Matrix Q D ℂ) :
    (rightFactor M).conjTranspose * rightFactor M =
      Matrix.diagonal (fun q => (values M q : ℂ)) := by
  have hg := gram_diagonalization M.conjTranspose
  change (basis M).conjTranspose * (M.conjTranspose.conjTranspose * M.conjTranspose) * basis M =
    Matrix.diagonal (fun q => ((values M q ^ 2 : ℝ) : ℂ)) at hg
  unfold rightFactor
  rw [Matrix.conjTranspose_mul, conjTranspose_diagonal_ofReal, Matrix.conjTranspose_mul]
  calc
    _ = Matrix.diagonal (fun q => (((Real.sqrt (values M q))⁻¹ : ℝ) : ℂ)) *
        ((basis M).conjTranspose * (M.conjTranspose.conjTranspose * M.conjTranspose) * basis M) *
        Matrix.diagonal (fun q => (((Real.sqrt (values M q))⁻¹ : ℝ) : ℂ)) := by
      simp only [Matrix.mul_assoc]
    _ = _ := by
      rw [hg, Matrix.diagonal_mul_diagonal, Matrix.diagonal_mul_diagonal]
      congr 1
      funext q
      have hv : 0 ≤ values M q := Real.sqrt_nonneg _
      rw [← Complex.ofReal_mul, ← Complex.ofReal_mul,
        inv_sqrt_mul_sq_mul_inv_sqrt hv]

omit [DecidableEq D] in
/-- The actual query factor's object geometry is the positive square root
of M M*, CM86. It is not determined only by the scalar readout norm. -/
theorem left_covariance (M : Matrix Q D ℂ) :
    leftFactor M * (leftFactor M).conjTranspose = CFC.sqrt (M * M.conjTranspose) := by
  apply Eq.symm
  apply CFC.sqrt_unique _ (Matrix.posSemidef_self_mul_conjTranspose _).nonneg
  conv_rhs => rw [← factorization M]
  rw [Matrix.conjTranspose_mul, Matrix.conjTranspose_conjTranspose]
  calc
    _ = leftFactor M * ((leftFactor M).conjTranspose * leftFactor M) *
        (leftFactor M).conjTranspose := by simp only [Matrix.mul_assoc]
    _ = _ := by rw [left_hidden_gram, ← right_hidden_gram]; simp only [Matrix.mul_assoc]

/-- The actual context factor's geometry is the positive square root of
M* M, CM86, retaining every original context coordinate. -/
theorem right_covariance (M : Matrix Q D ℂ) :
    rightFactor M * (rightFactor M).conjTranspose = CFC.sqrt (M.conjTranspose * M) := by
  apply Eq.symm
  apply CFC.sqrt_unique _ (Matrix.posSemidef_self_mul_conjTranspose _).nonneg
  conv_rhs => rw [← factorization M]
  rw [Matrix.conjTranspose_mul, Matrix.conjTranspose_conjTranspose]
  calc
    _ = rightFactor M * ((rightFactor M).conjTranspose * rightFactor M) *
        (rightFactor M).conjTranspose := by simp only [Matrix.mul_assoc]
    _ = _ := by rw [right_hidden_gram, ← left_hidden_gram]; simp only [Matrix.mul_assoc]

omit [DecidableEq D] in
/-- At a positive square joint matrix, both actual encoders have the same
object covariance M, CM86. Eigenbasis choices cancel in these observables. -/
theorem positive_covariance {M : Matrix Q Q ℂ} (hM : M.PosSemidef) :
    leftFactor M * (leftFactor M).conjTranspose = M ∧
      rightFactor M * (rightFactor M).conjTranspose = M := by
  rw [left_covariance, right_covariance, hM.isHermitian.eq]
  have h : CFC.sqrt (M * M) = M := CFC.sqrt_unique rfl hM.nonneg
  exact ⟨h, h⟩

example := positive_covariance (Matrix.PosSemidef.one :
  (1 : Matrix (Fin 2) (Fin 2) ℂ).PosSemidef)

omit [DecidableEq D] in
/-- Both normalized learned encoders have covariance M / ‖M‖_S1 at a
positive joint matrix, CM86. This includes the zero matrix. -/
theorem positive_representation_covariance {M : Matrix Q Q ℂ}
    (hM : M.PosSemidef) :
    leftRepresentation M * (leftRepresentation M).conjTranspose = (readout M)⁻¹ • M ∧
      rightRepresentation M * (rightRepresentation M).conjTranspose = (readout M)⁻¹ • M := by
  have hs : (Real.sqrt (readout M))⁻¹ * (Real.sqrt (readout M))⁻¹ = (readout M)⁻¹ := by
    have hn : 0 ≤ readout M := nuclearNorm_nonneg M
    rw [← mul_inv, Real.mul_self_sqrt hn]
  have hc := positive_covariance hM
  simp only [leftRepresentation, rightRepresentation, Matrix.conjTranspose_smul,
    star_trivial, Matrix.smul_mul, Matrix.mul_smul, smul_smul, hs, hc.1, hc.2, and_self]

example := positive_representation_covariance (Matrix.PosSemidef.one :
  (1 : Matrix (Fin 2) (Fin 2) ℂ).PosSemidef)

end Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear
