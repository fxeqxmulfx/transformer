/-
# Gaussian positive quadratic-form exponentials

Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`.
For the auxiliary Gaussian in the off-diagonal proof, a positive
semidefinite quadratic form has a finite exponential moment controlled
by its exact trace when the coefficient times its actual operator norm
is small. The original form is reduced to the genuine orthonormal
eigenbasis, and the standard Gaussian law is preserved by that isometry.
The eigenvalue bound uses normalized eigenvectors, including empty
dimensions without a nonemptiness restriction on the main estimate.
-/

import Transformer.Homogenized.Section6_PositiveQuadraticMGF
import Mathlib.Analysis.Matrix.PosDef

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The original entrywise quadratic polynomial equals its actual
Euclidean inner product. Source: arXiv:2604.01978v1, §6.3,
the original quadratic form in `lem:Hanson_wright`. -/
theorem hansonWrightQuadraticForm_eq_inner {n : ℕ}
    (C : Matrix (Fin n) (Fin n) ℝ) (z : EucSpace n) :
    hansonWrightQuadraticForm C z = inner ℝ z (Matrix.toEuclideanCLM (𝕜 := ℝ) C z) := by
  rw [Matrix.inner_toEuclideanCLM]
  simp only [hansonWrightQuadraticForm, dotProduct, Matrix.mulVec, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- The original Gram trace is the exact original Frobenius square.
Source: arXiv:2604.01978v1, §6.3, the Gaussian-comparison norm
in the proof of `lem:Hanson_wright`. -/
theorem trace_gram_eq_frobenius_sq {n : ℕ} (B : Matrix (Fin n) (Fin n) ℝ) :
    Matrix.trace (B.transpose * B) = ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2 := by
  simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply, Matrix.transpose_apply, pow_two]
  rw [Finset.sum_comm]

/-- The original Gram quadratic polynomial is exactly the squared
Euclidean norm of the original matrix image. Source: arXiv:2604.01978v1,
§6.3, the auxiliary Gaussian norm square in `lem:Hanson_wright`. -/
theorem hansonWrightQuadraticForm_gram_eq_norm_sq {n : ℕ}
    (B : Matrix (Fin n) (Fin n) ℝ) (z : EucSpace n) :
    hansonWrightQuadraticForm (B.transpose * B) z =
      ‖Matrix.toEuclideanCLM (𝕜 := ℝ) B z‖ ^ 2 := by
  rw [hansonWrightQuadraticForm_eq_inner, ← Matrix.conjTranspose_eq_transpose_of_trivial B,
    map_mul, ← Matrix.star_eq_conjTranspose, StarHomClass.map_star, ContinuousLinearMap.star_eq_adjoint,
    mul_apply_eq_comp, ContinuousLinearMap.adjoint_inner_right,
    real_inner_self_eq_norm_sq]

/-- The actual matrix operator acts by its eigenvalue on each original
orthonormal eigenvector. Source: arXiv:2604.01978v1, §6.3,
the spectral Gaussian-comparison step for `lem:Hanson_wright`. -/
theorem toEuclideanCLM_eigenvectorBasis {n : ℕ} {C : Matrix (Fin n) (Fin n) ℝ}
    (hC : C.IsHermitian) (i : Fin n) :
    Matrix.toEuclideanCLM (𝕜 := ℝ) C (hC.eigenvectorBasis i) =
      hC.eigenvalues i • hC.eigenvectorBasis i := by
  apply WithLp.ofLp_injective 2
  rw [Matrix.ofLp_toEuclideanCLM, WithLp.ofLp_smul]
  exact hC.mulVec_eigenvectorBasis i

/-- Every genuine eigenvalue is bounded by the original Euclidean
operator norm. Source: arXiv:2604.01978v1, §6.3, the auxiliary
quadratic-exponential coefficient interval for `lem:Hanson_wright`. -/
theorem abs_eigenvalue_le_matOpNorm {n : ℕ} {C : Matrix (Fin n) (Fin n) ℝ}
    (hC : C.IsHermitian) (i : Fin n) : |hC.eigenvalues i| ≤ matOpNorm C := by
  have h := (Matrix.toEuclideanCLM (𝕜 := ℝ) C).le_opNorm (hC.eigenvectorBasis i)
  rw [toEuclideanCLM_eigenvectorBasis hC i, norm_smul, hC.eigenvectorBasis.norm_eq_one, mul_one, mul_one] at h
  simpa only [Real.norm_eq_abs, matOpNorm] using h

/-- The actual original quadratic form diagonalizes in its orthonormal
eigenbasis. Source: arXiv:2604.01978v1, §6.3, the Gaussian spectral
reduction in the proof of `lem:Hanson_wright`. -/
theorem hansonWrightQuadraticForm_eigenbasis {n : ℕ} {C : Matrix (Fin n) (Fin n) ℝ}
    (hC : C.IsHermitian) (z : EucSpace n) :
    hansonWrightQuadraticForm C z =
      ∑ i : Fin n, hC.eigenvalues i * (hC.eigenvectorBasis.repr z i) ^ 2 := by
  let b := hC.eigenvectorBasis
  let T : EucSpace n →L[ℝ] EucSpace n := Matrix.toEuclideanCLM (𝕜 := ℝ) C
  have ht : T z = ∑ i : Fin n, b.repr z i • T (b i) := by
    calc
      T z = T (∑ i : Fin n, b.repr z i • b i) := by rw [b.sum_repr z]
      _ = _ := by rw [map_sum]; simp_rw [map_smul]
  rw [hansonWrightQuadraticForm_eq_inner, ht, inner_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [inner_smul_right, toEuclideanCLM_eigenvectorBasis hC i, inner_smul_right,
    real_inner_comm, ← hC.eigenvectorBasis.repr_apply_apply]
  dsimp only [b]
  ring

/-- The actual positive semidefinite quadratic form is nonnegative.
Source: arXiv:2604.01978v1, §6.3, the auxiliary positive form
in `lem:Hanson_wright`. -/
theorem hansonWrightQuadraticForm_posSemidef_nonneg {n : ℕ} {C : Matrix (Fin n) (Fin n) ℝ}
    (hC : C.PosSemidef) (z : EucSpace n) : 0 ≤ hansonWrightQuadraticForm C z := by
  rw [hansonWrightQuadraticForm_eigenbasis hC.isHermitian]
  exact Finset.sum_nonneg (fun i _ => mul_nonneg (hC.eigenvalues_nonneg i) (sq_nonneg _))

/-- The original standard Gaussian positive quadratic exponential is
integrable and bounded by the exact trace under the actual operator
norm condition. Source: arXiv:2604.01978v1, §6.3, the auxiliary
Gaussian estimate in the proof of `lem:Hanson_wright`. -/
theorem stdGaussian_posSemidef_quadratic_exp {n : ℕ}
    (C : Matrix (Fin n) (Fin n) ℝ) (hC : C.PosSemidef) (r : ℝ) (hr : 0 ≤ r)
    (hsmall : r * matOpNorm C ≤ 1 / 4) :
    Integrable (fun z : EucSpace n => Real.exp (r * hansonWrightQuadraticForm C z))
      (stdGaussian (EucSpace n)) ∧
    (∫ z : EucSpace n, Real.exp (r * hansonWrightQuadraticForm C z) ∂stdGaussian (EucSpace n)) ≤
      Real.exp (8 * r * Matrix.trace C) := by
  let b := hC.isHermitian.eigenvectorBasis
  let a : Fin n → ℝ := hC.isHermitian.eigenvalues
  have ha : ∀ i, 0 ≤ a i := hC.eigenvalues_nonneg
  have hs : ∀ i, r * a i ≤ 1 / 4 := by
    intro i
    apply le_trans _ hsmall
    apply mul_le_mul_of_nonneg_left _ hr
    exact (le_abs_self _).trans (abs_eigenvalue_le_matOpNorm hC.isHermitian i)
  obtain ⟨hi, hb⟩ := stdGaussian_positive_diagonal_exp a ha r hr hs
  let f : EucSpace n → ℝ := fun z => Real.exp (r * ∑ i : Fin n, a i * (z i) ^ 2)
  have hm : StronglyMeasurable f := by
    apply Measurable.stronglyMeasurable
    dsimp only [f]
    fun_prop
  have hme : AEMeasurable b.repr (stdGaussian (EucSpace n)) := b.repr.continuous.measurable.aemeasurable
  have hmap : (stdGaussian (EucSpace n)).map b.repr = stdGaussian (EucSpace n) := stdGaussian_map b.repr
  have hj : Integrable f ((stdGaussian (EucSpace n)).map b.repr) := by rw [hmap]; exact hi
  have hp : ∀ z : EucSpace n, Real.exp (r * hansonWrightQuadraticForm C z) = f (b.repr z) := by
    intro z
    rw [hansonWrightQuadraticForm_eigenbasis hC.isHermitian]
  have htrace : (∑ i : Fin n, a i) = Matrix.trace C := hC.isHermitian.trace_eq_sum_eigenvalues.symm
  refine ⟨?_, ?_⟩
  · simpa only [hp, Function.comp_def] using (integrable_map_measure hj.aestronglyMeasurable hme).mp hj
  · calc
      _ = ∫ z : EucSpace n, f (b.repr z) ∂stdGaussian (EucSpace n) := by simp only [hp]
      _ = ∫ z : EucSpace n, f z ∂(stdGaussian (EucSpace n)).map b.repr :=
        (integral_map hme hm.aestronglyMeasurable).symm
      _ = ∫ z : EucSpace n, f z ∂stdGaussian (EucSpace n) := by rw [hmap]
      _ ≤ Real.exp (8 * r * Matrix.trace C) := by simpa only [htrace] using hb

/-- The actual auxiliary Gaussian matrix-image square has a finite
exponential moment controlled by the original Frobenius square and
operator norm. Source: arXiv:2604.01978v1, §6.3, the Gaussian
comparison estimate for `lem:Hanson_wright`. -/
theorem stdGaussian_matrix_norm_sq_exp {n : ℕ}
    (B : Matrix (Fin n) (Fin n) ℝ) (r : ℝ) (hr : 0 ≤ r)
    (hsmall : r * matOpNorm B ^ 2 ≤ 1 / 4) :
    Integrable (fun z : EucSpace n => Real.exp (r * ‖Matrix.toEuclideanCLM (𝕜 := ℝ) B z‖ ^ 2))
      (stdGaussian (EucSpace n)) ∧
    (∫ z : EucSpace n, Real.exp (r * ‖Matrix.toEuclideanCLM (𝕜 := ℝ) B z‖ ^ 2)
      ∂stdGaussian (EucSpace n)) ≤ Real.exp (8 * r * ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2) := by
  have hp : (B.transpose * B).PosSemidef := by
    simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using Matrix.posSemidef_conjTranspose_mul_self B
  simpa only [hansonWrightQuadraticForm_gram_eq_norm_sq, trace_gram_eq_frobenius_sq] using
    stdGaussian_posSemidef_quadratic_exp (B.transpose * B) hp r hr
      (by simpa only [matOpNorm_transpose_mul_self] using hsmall)

/-- The genuine identity form and a positive admissible coefficient
jointly witness positivity, Hermitian symmetry and the actual norm
condition in a nonempty dimension. -/
example : (1 : Matrix (Fin 2) (Fin 2) ℝ).PosSemidef ∧
    (1 : Matrix (Fin 2) (Fin 2) ℝ).IsHermitian ∧ (0 : ℝ) ≤ 1 / 8 ∧
    (1 / 8 : ℝ) * matOpNorm (1 : Matrix (Fin 2) (Fin 2) ℝ) ≤ 1 / 4 ∧
    (1 / 8 : ℝ) * matOpNorm (1 : Matrix (Fin 2) (Fin 2) ℝ) ^ 2 ≤ 1 / 4 := by
  have hp : (1 : Matrix (Fin 2) (Fin 2) ℝ).PosSemidef := Matrix.PosSemidef.one
  refine ⟨hp, hp.isHermitian, by norm_num, ?_⟩
  have hn : matOpNorm (1 : Matrix (Fin 2) (Fin 2) ℝ) = 1 := by
    rw [← Matrix.diagonal_one, matOpNorm_diagonal]
    simp
  rw [hn]
  norm_num

end Transformer.Homogenized
