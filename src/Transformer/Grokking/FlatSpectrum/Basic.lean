module

public import Transformer.Grokking.FlatSpectrum.Defs

/-!
# The nuclear norm bounds every contraction pairing

Source: `review/flat_spectrum.md`, Lemma 1 and Theorem A. In an orthonormal eigenbasis
`U` of `Xᴴ X` the columns of `X U` are orthogonal, with squared lengths the eigenvalues.
Hence `|tr (Zᴴ X)| ≤ ‖X‖_S1` for every contraction `Z`. Conversely a unitary `W` with
`Wᴴ Xᴴ X W = diag (d²)`, `d ≥ 0`, yields a contraction `Z` with `tr (Zᴴ X) = ∑ d`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]

omit [Fintype n] [DecidableEq n] in
/-- Diagonal Gram entries are squared column lengths. -/
lemma conjTranspose_mul_self_apply_self (A : Matrix m n ℂ) (i : n) :
    (Aᴴ * A) i i = ((∑ k, ‖A k i‖ ^ 2 : ℝ) : ℂ) := by
  simp only [mul_apply, conjTranspose_apply, Complex.ofReal_sum, Complex.ofReal_pow]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [Complex.star_def, Complex.conj_mul']

/-- The trace pairing `tr (Aᴴ B) = ∑ conj (A i j) B i j`. -/
lemma trace_conjTranspose_mul {k : Type*} [Fintype k] (A B : Matrix m k ℂ) :
    (Aᴴ * B).trace = ∑ i, ∑ j, star (A i j) * B i j := by
  simp only [trace, diag_apply, mul_apply, conjTranspose_apply]
  exact Finset.sum_comm

/-- Cauchy–Schwarz for finite complex sums. -/
lemma norm_sum_star_mul_le {ι : Type*} (s : Finset ι) (a b : ι → ℂ) :
    ‖∑ i ∈ s, star (a i) * b i‖ ≤ √(∑ i ∈ s, ‖a i‖ ^ 2) * √(∑ i ∈ s, ‖b i‖ ^ 2) := by
  refine (norm_sum_le _ _).trans ((Finset.sum_le_sum fun i _ => ?_).trans
    (Real.sum_mul_le_sqrt_mul_sqrt s (fun i => ‖a i‖) fun i => ‖b i‖))
  rw [norm_mul, norm_star]

/-- Unitary diagonalization of a Hermitian matrix by Mathlib's eigenvector unitary. -/
lemma eigenvectorUnitary_conj {A : Matrix n n ℂ} (hA : A.IsHermitian) :
    (hA.eigenvectorUnitary : Matrix n n ℂ)ᴴ * A * hA.eigenvectorUnitary =
      diagonal fun i => ((hA.eigenvalues i : ℝ) : ℂ) := by
  have h := hA.conjStarAlgAut_star_eigenvectorUnitary
  simp only [Unitary.conjStarAlgAut_star_apply, Matrix.star_eq_conjTranspose] at h
  exact h

example := eigenvectorUnitary_conj (isHermitian_conjTranspose_mul_self (1 : Matrix (Fin 2) (Fin 2) ℂ))

/-- The eigenvector unitary is unitary. -/
lemma eigenvectorUnitary_conjTranspose_mul {A : Matrix n n ℂ} (hA : A.IsHermitian) :
    (hA.eigenvectorUnitary : Matrix n n ℂ)ᴴ * hA.eigenvectorUnitary = 1 :=
  Unitary.coe_star_mul_self _

example := eigenvectorUnitary_conjTranspose_mul (isHermitian_conjTranspose_mul_self (1 : Matrix (Fin 2) (Fin 2) ℂ))

/-- The eigenvector unitary is unitary. -/
lemma eigenvectorUnitary_mul_conjTranspose {A : Matrix n n ℂ} (hA : A.IsHermitian) :
    (hA.eigenvectorUnitary : Matrix n n ℂ) * (hA.eigenvectorUnitary : Matrix n n ℂ)ᴴ = 1 :=
  Unitary.coe_mul_star_self _

example := eigenvectorUnitary_mul_conjTranspose (isHermitian_conjTranspose_mul_self (1 : Matrix (Fin 2) (Fin 2) ℂ))

/-- The identity is a contraction. -/
lemma isContraction_one : IsContraction (1 : Matrix n n ℂ) := by
  rw [IsContraction, conjTranspose_one, Matrix.mul_one, sub_self]
  exact PosSemidef.zero

/-- Columns of `Z U` have length at most one when `Z` is a contraction, `U` unitary. -/
lemma IsContraction.sum_norm_sq_le {Z : Matrix m n ℂ} (hZ : IsContraction Z)
    {U : Matrix n n ℂ} (hU : Uᴴ * U = 1) (i : n) : ∑ k, ‖(Z * U) k i‖ ^ 2 ≤ 1 := by
  have h := (hZ.conjTranspose_mul_mul_same U).diag_nonneg (i := i)
  have hE : Uᴴ * (1 - Zᴴ * Z) * U = 1 - (Z * U)ᴴ * (Z * U) := by
    simp only [Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_one, hU, conjTranspose_mul,
      Matrix.mul_assoc]
  rw [hE, Matrix.sub_apply, one_apply_eq, conjTranspose_mul_self_apply_self,
    ← Complex.ofReal_one, ← Complex.ofReal_sub, Complex.zero_le_real] at h
  linarith

example (i : Fin 2) := (isContraction_one (n := Fin 2)).sum_norm_sq_le (U := 1) (by simp) i

/-- `|tr (Zᴴ X)| ≤ ‖X‖_S1` for every contraction `Z`: the nuclear norm dominates its
dual pairing with the operator-norm unit ball, `review/flat_spectrum.md`, Theorem A. -/
theorem norm_trace_le_nuclearNorm {Z : Matrix m n ℂ} (hZ : IsContraction Z)
    (X : Matrix m n ℂ) : ‖(Zᴴ * X).trace‖ ≤ nuclearNorm X := by
  set hX := isHermitian_conjTranspose_mul_self X
  set U : Matrix n n ℂ := (hX.eigenvectorUnitary : Matrix n n ℂ)
  have hU : Uᴴ * U = 1 := eigenvectorUnitary_conjTranspose_mul hX
  have hU' : U * Uᴴ = 1 := eigenvectorUnitary_mul_conjTranspose hX
  have htr : (Zᴴ * X).trace = ∑ i, ∑ k, star ((Z * U) k i) * (X * U) k i := by
    have : ((Z * U)ᴴ * (X * U)).trace = (Zᴴ * X).trace := by
      rw [conjTranspose_mul, Matrix.mul_assoc, Matrix.trace_mul_comm, Matrix.mul_assoc,
        Matrix.mul_assoc, hU', Matrix.mul_one]
    rw [← this]
    simp only [trace, diag_apply, mul_apply, conjTranspose_apply]
  have hcol (i : n) : ∑ k, ‖(X * U) k i‖ ^ 2 = hX.eigenvalues i := by
    have h := congrFun (congrFun (eigenvectorUnitary_conj hX) i) i
    rw [diagonal_apply_eq, show Uᴴ * (Xᴴ * X) * U = (X * U)ᴴ * (X * U) by
      simp only [conjTranspose_mul, Matrix.mul_assoc], conjTranspose_mul_self_apply_self] at h
    exact_mod_cast h
  rw [htr, nuclearNorm]
  refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun i _ => ?_)
  refine (norm_sum_star_mul_le _ _ _).trans ?_
  rw [hcol i, singularValues]
  have h1 := hZ.sum_norm_sq_le hU i
  have h2 : √(∑ k, ‖(Z * U) k i‖ ^ 2) ≤ 1 := Real.sqrt_le_one.mpr h1
  exact mul_le_of_le_one_left (Real.sqrt_nonneg _) h2

example (X : Matrix (Fin 2) (Fin 2) ℂ) := norm_trace_le_nuclearNorm (isContraction_one (n := Fin 2)) X

/-- Diagonal entries `d⁻¹ d² d⁻¹` lie in `{0, 1}`. -/
lemma inv_mul_sq_mul_inv_le_one (x : ℝ) : x⁻¹ * x ^ 2 * x⁻¹ ≤ 1 := by
  rcases eq_or_ne x 0 with h | h
  · simp [h]
  · rw [show x⁻¹ * x ^ 2 * x⁻¹ = 1 by field_simp]

/-- `(√x)⁻¹ x² (√x)⁻¹ = x` for `x ≥ 0`. -/
lemma inv_sqrt_mul_sq_mul_inv_sqrt {x : ℝ} (hx : 0 ≤ x) : (√x)⁻¹ * x ^ 2 * (√x)⁻¹ = x := by
  rw [show (√x)⁻¹ * x ^ 2 * (√x)⁻¹ = x ^ 2 / √x ^ 2 by ring, Real.sq_sqrt hx]
  rcases eq_or_ne x 0 with h | h
  · simp [h]
  · field_simp

example := inv_sqrt_mul_sq_mul_inv_sqrt (x := 2) zero_le_two

omit [Fintype n] in
/-- A real diagonal matrix is Hermitian. -/
lemma conjTranspose_diagonal_ofReal (d : n → ℝ) :
    (diagonal fun i => ((d i : ℝ) : ℂ))ᴴ = diagonal fun i => ((d i : ℝ) : ℂ) := by
  rw [diagonal_conjTranspose]
  congr 1
  funext i
  simp

/-- A unitary `W` with `Wᴴ (Xᴴ X) W = diag (d²)`, `d ≥ 0`, yields the contraction
`Z = X W diag (d⁻¹) Wᴴ` with `tr (Zᴴ X) = ∑ d`: the polar factor attains the nuclear
norm, `review/flat_spectrum.md`, Theorem A. -/
theorem polar_of_diagonalization {X : Matrix m n ℂ} {W : Matrix n n ℂ} {d : n → ℝ}
    (hW : Wᴴ * W = 1) (hX : Wᴴ * (Xᴴ * X) * W = diagonal fun i => ((d i ^ 2 : ℝ) : ℂ)) :
    IsContraction (X * W * diagonal (fun i => (((d i)⁻¹ : ℝ) : ℂ)) * Wᴴ) ∧
      ((X * W * diagonal (fun i => (((d i)⁻¹ : ℝ) : ℂ)) * Wᴴ)ᴴ * X).trace =
        ∑ i, (d i : ℂ) := by
  have hW' : W * Wᴴ = 1 := mul_eq_one_comm.mp hW
  set D : Matrix n n ℂ := diagonal fun i => (((d i)⁻¹ : ℝ) : ℂ) with hDdef
  have hD : Dᴴ = D := conjTranspose_diagonal_ofReal _
  have hZ : (X * W * D * Wᴴ)ᴴ = W * D * Wᴴ * Xᴴ := by
    simp only [conjTranspose_mul, conjTranspose_conjTranspose, hD, Matrix.mul_assoc]
  have hDD : D * (diagonal fun i => ((d i ^ 2 : ℝ) : ℂ)) * D =
      diagonal fun i => (((d i)⁻¹ * d i ^ 2 * (d i)⁻¹ : ℝ) : ℂ) := by
    simp only [hDdef, diagonal_mul_diagonal]
    congr 1
    funext i
    push_cast
    ring
  have hG : (X * W * D * Wᴴ)ᴴ * (X * W * D * Wᴴ) =
      W * (D * (Wᴴ * (Xᴴ * X) * W) * D) * Wᴴ := by
    rw [hZ]
    simp only [Matrix.mul_assoc]
  rw [hX, hDD] at hG
  refine ⟨?_, ?_⟩
  · have h1 : 1 - (X * W * D * Wᴴ)ᴴ * (X * W * D * Wᴴ) =
        W * diagonal (fun i => ((1 - (d i)⁻¹ * d i ^ 2 * (d i)⁻¹ : ℝ) : ℂ)) * Wᴴ := by
      rw [hG]
      have : (diagonal fun i => ((1 - (d i)⁻¹ * d i ^ 2 * (d i)⁻¹ : ℝ) : ℂ)) =
          1 - diagonal fun i => (((d i)⁻¹ * d i ^ 2 * (d i)⁻¹ : ℝ) : ℂ) := by
        rw [← diagonal_one, diagonal_sub]
        congr 1
        funext i
        push_cast
        ring
      rw [this, Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_one, hW']
    unfold IsContraction
    rw [h1]
    refine (posSemidef_diagonal_iff.mpr fun i => ?_).mul_mul_conjTranspose_same W
    rw [Complex.zero_le_real]
    linarith [inv_mul_sq_mul_inv_le_one (d i)]
  · rw [hZ]
    have : (W * D * Wᴴ * Xᴴ * X).trace = (D * (Wᴴ * (Xᴴ * X) * W)).trace := by
      rw [show W * D * Wᴴ * Xᴴ * X = W * (D * (Wᴴ * (Xᴴ * X))) by
        simp only [Matrix.mul_assoc], Matrix.trace_mul_comm]
      simp only [Matrix.mul_assoc]
    rw [this, hX, hDdef, diagonal_mul_diagonal, trace_diagonal]
    refine Finset.sum_congr rfl fun i _ => ?_
    rcases eq_or_ne (d i) 0 with h | h
    · simp [h]
    · push_cast
      field_simp

example := polar_of_diagonalization (X := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (W := 1) (d := fun _ => 1) (by simp) (by simp)

end Transformer.Grokking.FlatSpectrum
