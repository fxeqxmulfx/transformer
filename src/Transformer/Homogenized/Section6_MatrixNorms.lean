module

public import Transformer.Homogenized.Section6_DiagonalQuadraticMGF
public import Mathlib.Analysis.CStarAlgebra.Matrix

/-!
# Exact Euclidean matrix norms for Hanson–Wright

Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`.
The operator norm is the norm of the original matrix acting on
Euclidean space. It bounds individual entries, and multiplication
by coordinate projections contracts it. The corresponding masked
Frobenius sum also decreases. These exact matrix estimates retain
the original norms in the eventual quadratic-form tail bound.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal Matrix.Norms.L2Operator

namespace Transformer.Homogenized

/-- The `ℓ²` operator norm of a real square matrix, as a continuous
linear endomorphism of Euclidean space. Source: arXiv:2604.01978v1,
§6.3, `lem:Hanson_wright` (`‖B‖_op`). -/
noncomputable def matOpNorm {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) : ℝ :=
  ‖(Matrix.toEuclideanCLM (𝕜 := ℝ) B : EucSpace n →L[ℝ] EucSpace n)‖

/-- The actual Euclidean operator norm is nonnegative. Source:
arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`. -/
theorem matOpNorm_nonneg {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) : 0 ≤ matOpNorm B :=
  norm_nonneg _

/-- The actual zero-matrix norm vanishes. Source: arXiv:2604.01978v1,
§6.3, the degenerate matrix case of `lem:Hanson_wright`. -/
@[simp] theorem matOpNorm_zero {n : ℕ} : matOpNorm (0 : Matrix (Fin n) (Fin n) ℝ) = 0 := by
  simp [matOpNorm]

/-- The original norm agrees with Mathlib's Euclidean operator norm,
whose scoped instance is selected here. Source: arXiv:2604.01978v1,
§6.3, the operator norm in `lem:Hanson_wright`. -/
theorem matOpNorm_eq_l2Norm {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) : matOpNorm B = ‖B‖ :=
  Matrix.l2_opNorm_toEuclideanCLM B

/-- The original operator norm of a diagonal is the maximum absolute
diagonal entry. Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`. -/
theorem matOpNorm_diagonal {n : ℕ} (a : Fin n → ℝ) : matOpNorm (Matrix.diagonal a) = ‖a‖ := by
  rw [matOpNorm_eq_l2Norm, Matrix.l2_opNorm_diagonal]

/-- Transposition preserves the actual Euclidean operator norm.
Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`. -/
theorem matOpNorm_transpose {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) :
    matOpNorm B.transpose = matOpNorm B := by
  rw [matOpNorm_eq_l2Norm, matOpNorm_eq_l2Norm, ← Matrix.conjTranspose_eq_transpose_of_trivial]
  exact Matrix.l2_opNorm_conjTranspose B

/-- Matrix multiplication has the actual operator-norm product bound.
Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`. -/
theorem matOpNorm_mul {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℝ) :
    matOpNorm (A * B) ≤ matOpNorm A * matOpNorm B := by
  simp only [matOpNorm_eq_l2Norm]
  exact Matrix.l2_opNorm_mul A B

/-- Each original entry is bounded by the actual Euclidean operator
norm. Source: arXiv:2604.01978v1, §6.3, the diagonal coefficient
control for `lem:Hanson_wright`. -/
theorem abs_matrix_entry_le_matOpNorm {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) (i j : Fin n) :
    |B i j| ≤ matOpNorm B := by
  let e : EucSpace n := PiLp.single 2 j (1 : ℝ)
  let T : EucSpace n →L[ℝ] EucSpace n := Matrix.toEuclideanCLM (𝕜 := ℝ) B
  have he : (T e) i = B i j := by
    dsimp only [T, e]
    rw [Matrix.ofLp_toEuclideanCLM, PiLp.ofLp_single, Matrix.mulVec_single]
    simp
  calc
    |B i j| = ‖(T e) i‖ := by rw [he, Real.norm_eq_abs]
    _ ≤ ‖T e‖ := PiLp.norm_apply_le _ i
    _ ≤ ‖T‖ * ‖e‖ := T.le_opNorm e
    _ = matOpNorm B := by simp [e, T, matOpNorm, PiLp.norm_single]

/-- Extracting the original diagonal contracts the actual operator norm.
Source: arXiv:2604.01978v1, §6.3, the diagonal coefficient interval
in the proof of `lem:Hanson_wright`. -/
theorem matOpNorm_diagonal_entries_le {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) :
    matOpNorm (Matrix.diagonal (fun i => B i i)) ≤ matOpNorm B := by
  rw [matOpNorm_diagonal]
  apply (pi_norm_le_iff_of_nonneg (matOpNorm_nonneg B)).mpr
  intro i
  simpa only [Real.norm_eq_abs] using abs_matrix_entry_le_matOpNorm B i i

/-- The original diagonal coefficient-square sum is bounded by the
original Frobenius square. Source: arXiv:2604.01978v1, §6.3,
the diagonal contribution to `lem:Hanson_wright`. -/
theorem diagonal_sq_sum_le_frobenius_sq {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) :
    (∑ i : Fin n, B i i ^ 2) ≤ ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2 := by
  apply Finset.sum_le_sum
  intro i _
  exact Finset.single_le_sum (fun j _ => sq_nonneg (B i j)) (Finset.mem_univ i)

/-- The actual coordinate projection matrix for a subset. Source:
arXiv:2604.01978v1, §6.3, the coordinate mask in the decoupling
argument for `lem:Hanson_wright`. -/
noncomputable def coordinateMask {n : ℕ} (s : Finset (Fin n)) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.diagonal (fun i => if i ∈ s then 1 else 0)

/-- Coordinate projections contract the actual Euclidean norm.
Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`. -/
theorem matOpNorm_coordinateMask_le_one {n : ℕ} (s : Finset (Fin n)) :
    matOpNorm (coordinateMask s) ≤ 1 := by
  rw [coordinateMask, matOpNorm_diagonal]
  apply (pi_norm_le_iff_of_nonneg (by norm_num)).mpr
  intro i
  split_ifs <;> norm_num

/-- The actual masked cross matrix keeps the original operator-norm
bound. Source: arXiv:2604.01978v1, §6.3, the off-diagonal
decoupling estimate for `lem:Hanson_wright`. -/
theorem matOpNorm_cross_mask_le {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) (s : Finset (Fin n)) :
    matOpNorm (coordinateMask s * B * coordinateMask sᶜ) ≤ matOpNorm B := by
  calc
    _ ≤ matOpNorm (coordinateMask s * B) * matOpNorm (coordinateMask sᶜ) := matOpNorm_mul _ _
    _ ≤ (matOpNorm (coordinateMask s) * matOpNorm B) * matOpNorm (coordinateMask sᶜ) :=
      mul_le_mul_of_nonneg_right (matOpNorm_mul _ _) (matOpNorm_nonneg _)
    _ ≤ (1 * matOpNorm B) * 1 :=
      mul_le_mul (mul_le_mul_of_nonneg_right (matOpNorm_coordinateMask_le_one s) (matOpNorm_nonneg B))
        (matOpNorm_coordinateMask_le_one sᶜ) (matOpNorm_nonneg _) (by simpa using matOpNorm_nonneg B)
    _ = _ := by ring

/-- The actual cross mask retains precisely entries from the subset to
its complement. Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`. -/
theorem cross_mask_apply {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) (s : Finset (Fin n)) (i j : Fin n) :
    (coordinateMask s * B * coordinateMask sᶜ) i j = if i ∈ s ∧ j ∉ s then B i j else 0 := by
  simp only [coordinateMask, Matrix.mul_diagonal, Matrix.diagonal_mul, Finset.mem_compl]
  split_ifs <;> simp_all

/-- The actual cross matrix also retains the original Frobenius-square
bound. Source: arXiv:2604.01978v1, §6.3, the decoupling matrix
estimate for `lem:Hanson_wright`. -/
theorem cross_mask_frobenius_sq_le {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) (s : Finset (Fin n)) :
    (∑ i : Fin n, ∑ j : Fin n, (coordinateMask s * B * coordinateMask sᶜ) i j ^ 2) ≤
      ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2 := by
  apply Finset.sum_le_sum
  intro i _
  apply Finset.sum_le_sum
  intro j _
  rw [cross_mask_apply]
  split_ifs
  · exact le_rfl
  · simpa only [zero_pow (by decide : 2 ≠ 0)] using sq_nonneg (B i j)

/-- The positive Gram matrix has the exact squared original operator
norm. Source: arXiv:2604.01978v1, §6.3, the auxiliary Gaussian
quadratic estimate for `lem:Hanson_wright`. -/
theorem matOpNorm_transpose_mul_self {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) :
    matOpNorm (B.transpose * B) = matOpNorm B ^ 2 := by
  simp only [matOpNorm_eq_l2Norm]
  rw [← Matrix.conjTranspose_eq_transpose_of_trivial, Matrix.l2_opNorm_conjTranspose_mul_self, pow_two]

end Transformer.Homogenized
