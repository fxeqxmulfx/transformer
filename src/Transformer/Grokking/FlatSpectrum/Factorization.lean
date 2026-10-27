module

public import Transformer.Grokking.FlatSpectrum.Basic

/-!
# Lemma 1: the nuclear norm is the least balanced factorization cost

Source: `review/flat_spectrum.md`, Lemma 1: `‖K‖_S1 = min {½ (‖U‖² + ‖V‖²) : K = U V*}`.
Every factorization costs at least `‖K‖_S1` (pair the polar contraction with `U V*`), and
the factorization through the eigenbasis of `Kᴴ K` with weights `σ^{±1/2}` attains it.
A diagonalization `Wᴴ Xᴴ X W = diag (d²)` by any unitary computes `‖X‖_S1 = ∑ d`; in
particular `‖X‖_S1 = tr P` for every positive semidefinite `P` with `P² = Xᴴ X`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq n]

omit [DecidableEq n] in
lemma frobeniusSq_nonneg (A : Matrix m n ℂ) : 0 ≤ frobeniusSq A :=
  Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _

omit [DecidableEq n] in
/-- `‖A‖_S2² = Re tr (Aᴴ A)`. -/
lemma frobeniusSq_eq_re_trace (A : Matrix m n ℂ) : frobeniusSq A = (Aᴴ * A).trace.re := by
  simp only [trace, diag_apply, conjTranspose_mul_self_apply_self, Complex.re_sum,
    Complex.ofReal_re, frobeniusSq]
  exact Finset.sum_comm

omit [DecidableEq n] in
/-- Cauchy–Schwarz for the trace pairing: `|tr (Aᴴ B)| ≤ ‖A‖_S2 ‖B‖_S2`. -/
lemma norm_trace_conjTranspose_mul_le (A B : Matrix m n ℂ) :
    ‖(Aᴴ * B).trace‖ ≤ √(frobeniusSq A) * √(frobeniusSq B) := by
  have h := norm_sum_star_mul_le (Finset.univ : Finset (m × n)) (fun x => A x.1 x.2)
    fun x => B x.1 x.2
  simp only [Finset.univ_product_univ.symm, Finset.sum_product] at h
  rwa [trace_conjTranspose_mul]

/-- A contraction does not increase the Frobenius norm. -/
lemma IsContraction.frobeniusSq_mul_le {Z : Matrix m n ℂ} (hZ : IsContraction Z)
    (V : Matrix n k ℂ) : frobeniusSq (Z * V) ≤ frobeniusSq V := by
  have h := (hZ.conjTranspose_mul_mul_same V).trace_nonneg
  have hE : Vᴴ * (1 - Zᴴ * Z) * V = Vᴴ * V - (Z * V)ᴴ * (Z * V) := by
    simp only [Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_one, conjTranspose_mul,
      Matrix.mul_assoc]
  rw [hE, trace_sub] at h
  have := (Complex.nonneg_iff.mp h).1
  rw [Complex.sub_re] at this
  rw [frobeniusSq_eq_re_trace, frobeniusSq_eq_re_trace]
  linarith

example (V : Matrix (Fin 2) (Fin 2) ℂ) := (isContraction_one (n := Fin 2)).frobeniusSq_mul_le V

/-- Mathlib's eigenvalues of `Xᴴ X` are nonnegative. -/
lemma eigenvalues_gram_nonneg (X : Matrix m n ℂ) (i : n) :
    0 ≤ (isHermitian_conjTranspose_mul_self X).eigenvalues i :=
  (posSemidef_conjTranspose_mul_self X).eigenvalues_nonneg i

/-- The eigenbasis of `Xᴴ X` diagonalizes it with the singular values. -/
lemma gram_diagonalization (X : Matrix m n ℂ) :
    ((isHermitian_conjTranspose_mul_self X).eigenvectorUnitary : Matrix n n ℂ)ᴴ *
        (Xᴴ * X) * ((isHermitian_conjTranspose_mul_self X).eigenvectorUnitary : Matrix n n ℂ) =
      diagonal fun i => ((singularValues X i ^ 2 : ℝ) : ℂ) := by
  rw [eigenvectorUnitary_conj]
  congr 1
  funext i
  rw [singularValues, Real.sq_sqrt (eigenvalues_gram_nonneg X i)]

/-- Some contraction attains `tr (Zᴴ X) = ‖X‖_S1`. -/
theorem exists_isContraction_trace_eq (X : Matrix m n ℂ) :
    ∃ Z : Matrix m n ℂ, IsContraction Z ∧ (Zᴴ * X).trace = nuclearNorm X := by
  obtain ⟨hZ, htr⟩ := polar_of_diagonalization
    (eigenvectorUnitary_conjTranspose_mul (isHermitian_conjTranspose_mul_self X))
    (gram_diagonalization X)
  exact ⟨_, hZ, by rw [htr, nuclearNorm]; push_cast; rfl⟩

/-- Every factorization `X = U Vᴴ` costs at least the nuclear norm,
`review/flat_spectrum.md`, Lemma 1 (lower bound, any inner dimension). -/
theorem nuclearNorm_le_of_eq_mul {X : Matrix m n ℂ} (U : Matrix m k ℂ) (V : Matrix n k ℂ)
    (h : X = U * Vᴴ) : nuclearNorm X ≤ (frobeniusSq U + frobeniusSq V) / 2 := by
  obtain ⟨Z, hZ, htr⟩ := exists_isContraction_trace_eq X
  have hre : nuclearNorm X = ((Z * V)ᴴ * U).trace.re := by
    have : ((Z * V)ᴴ * U).trace = (Zᴴ * X).trace := by
      rw [h, conjTranspose_mul, Matrix.mul_assoc, Matrix.trace_mul_comm, Matrix.mul_assoc]
    rw [this, htr, Complex.ofReal_re]
  have hU := frobeniusSq_nonneg U
  have hV := frobeniusSq_nonneg V
  calc nuclearNorm X ≤ ‖((Z * V)ᴴ * U).trace‖ := hre ▸ Complex.re_le_norm _
    _ ≤ √(frobeniusSq (Z * V)) * √(frobeniusSq U) := norm_trace_conjTranspose_mul_le _ _
    _ ≤ √(frobeniusSq V) * √(frobeniusSq U) := by
        gcongr
        exact hZ.frobeniusSq_mul_le V
    _ ≤ (frobeniusSq U + frobeniusSq V) / 2 := by
        nlinarith [sq_nonneg (√(frobeniusSq V) - √(frobeniusSq U)), Real.sq_sqrt hU,
          Real.sq_sqrt hV]

/-- The balanced factorization through a unitary diagonalization `Wᴴ (Xᴴ X) W = diag (d²)`,
`d ≥ 0`: `X = (X W diag (d^{-1/2})) (W diag (d^{1/2}))ᴴ`, each factor of squared Frobenius
norm `∑ d`, `review/flat_spectrum.md`, Lemma 1 (the minimum is attained). -/
theorem factor_of_diagonalization {X : Matrix m n ℂ} {W : Matrix n n ℂ} {d : n → ℝ}
    (hd : ∀ i, 0 ≤ d i) (hW : Wᴴ * W = 1)
    (hX : Wᴴ * (Xᴴ * X) * W = diagonal fun i => ((d i ^ 2 : ℝ) : ℂ)) :
    X = X * W * diagonal (fun i => (((√(d i))⁻¹ : ℝ) : ℂ)) *
        (W * diagonal fun i => ((√(d i) : ℝ) : ℂ))ᴴ ∧
      frobeniusSq (X * W * diagonal fun i => (((√(d i))⁻¹ : ℝ) : ℂ)) = ∑ i, d i ∧
      frobeniusSq (W * diagonal fun i => ((√(d i) : ℝ) : ℂ)) = ∑ i, d i := by
  have hW' : W * Wᴴ = 1 := mul_eq_one_comm.mp hW
  have hG : (X * W)ᴴ * (X * W) = diagonal fun i => ((d i ^ 2 : ℝ) : ℂ) := by
    rw [← hX]
    simp only [conjTranspose_mul, Matrix.mul_assoc]
  have hcol (k : m) (i : n) (hi : d i = 0) : (X * W) k i = 0 := by
    have h := congrFun (congrFun hG i) i
    rw [conjTranspose_mul_self_apply_self, diagonal_apply_eq, hi] at h
    have h' : ∑ k, ‖(X * W) k i‖ ^ 2 = 0 := by
      rw [zero_pow two_ne_zero, Complex.ofReal_zero] at h
      exact_mod_cast h
    have := (Finset.sum_eq_zero_iff_of_nonneg fun k _ => sq_nonneg ‖(X * W) k i‖).mp h' k
      (Finset.mem_univ _)
    simpa using this
  refine ⟨?_, ?_, ?_⟩
  · have hE : X * W * diagonal (fun i => (((√(d i))⁻¹ : ℝ) : ℂ)) *
        diagonal (fun i => ((√(d i) : ℝ) : ℂ)) = X * W := by
      rw [Matrix.mul_assoc, diagonal_mul_diagonal]
      ext k i
      rw [mul_diagonal]
      rcases eq_or_ne (d i) 0 with h | h
      · rw [hcol k i h, zero_mul]
      · have hs : √(d i) ≠ 0 := Real.sqrt_ne_zero'.mpr ((hd i).lt_of_ne' h)
        rw [← Complex.ofReal_mul, inv_mul_cancel₀ hs, Complex.ofReal_one, mul_one]
    rw [conjTranspose_mul, conjTranspose_diagonal_ofReal, ← Matrix.mul_assoc, hE,
      Matrix.mul_assoc, hW', Matrix.mul_one]
  · rw [frobeniusSq_eq_re_trace, conjTranspose_mul, conjTranspose_diagonal_ofReal,
      show ∀ A B : Matrix n n ℂ, A * (X * W)ᴴ * (X * W * B) = A * ((X * W)ᴴ * (X * W)) * B by
        intro A B; simp only [Matrix.mul_assoc], hG, diagonal_mul_diagonal,
      diagonal_mul_diagonal, trace_diagonal, Complex.re_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Complex.ofReal_mul, ← Complex.ofReal_mul, Complex.ofReal_re,
      inv_sqrt_mul_sq_mul_inv_sqrt (hd i)]
  · rw [frobeniusSq_eq_re_trace, conjTranspose_mul, conjTranspose_diagonal_ofReal,
      show ∀ A B : Matrix n n ℂ, A * Wᴴ * (W * B) = A * (Wᴴ * W) * B by
        intro A B; simp only [Matrix.mul_assoc], hW, Matrix.mul_one, diagonal_mul_diagonal,
      trace_diagonal, Complex.re_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Complex.ofReal_mul, Complex.ofReal_re, Real.mul_self_sqrt (hd i)]

example := factor_of_diagonalization (X := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (W := 1) (d := fun _ => 1)
  (fun _ => zero_le_one) (by simp) (by simp)

omit [DecidableEq n] in
/-- Relabeling the columns keeps the Frobenius norm. -/
lemma frobeniusSq_submatrix_equiv {k' : Type*} [Fintype k'] (A : Matrix m k ℂ) (e : k' ≃ k) :
    frobeniusSq (A.submatrix id e) = frobeniusSq A := by
  simp only [frobeniusSq, submatrix_apply, id]
  exact Finset.sum_congr rfl fun i _ => e.sum_comp fun j => ‖A i j‖ ^ 2

/-- **Lemma 1** of `review/flat_spectrum.md`: `‖X‖_S1 = min {½ (‖U‖² + ‖V‖²) : X = U V*}`,
the minimum over factorizations through every inner dimension `r`. -/
theorem nuclearNorm_isLeast (X : Matrix m n ℂ) :
    IsLeast {c | ∃ (r : ℕ) (U : Matrix m (Fin r) ℂ) (V : Matrix n (Fin r) ℂ),
      X = U * Vᴴ ∧ c = (frobeniusSq U + frobeniusSq V) / 2} (nuclearNorm X) := by
  refine ⟨?_, ?_⟩
  · obtain ⟨hfac, hU, hV⟩ := factor_of_diagonalization (d := singularValues X)
      (fun i => Real.sqrt_nonneg _)
      (eigenvectorUnitary_conjTranspose_mul (isHermitian_conjTranspose_mul_self X))
      (gram_diagonalization X)
    set e := (Fintype.equivFin n).symm
    refine ⟨Fintype.card n,
      (X * ((isHermitian_conjTranspose_mul_self X).eigenvectorUnitary : Matrix n n ℂ) *
        diagonal fun i =>
        (((√(singularValues X i))⁻¹ : ℝ) : ℂ)).submatrix id e,
      (((isHermitian_conjTranspose_mul_self X).eigenvectorUnitary : Matrix n n ℂ) *
        diagonal fun i => ((√(singularValues X i) : ℝ) : ℂ)).submatrix id e, ?_, ?_⟩
    · rw [conjTranspose_submatrix, ← submatrix_mul _ _ _ _ _ e.bijective, submatrix_id_id]
      exact hfac
    · rw [frobeniusSq_submatrix_equiv, frobeniusSq_submatrix_equiv, hU, hV, nuclearNorm]
      ring
  · rintro c ⟨r, U, V, h, rfl⟩
    exact nuclearNorm_le_of_eq_mul U V h

example : nuclearNorm (1 : Matrix (Fin 2) (Fin 2) ℂ) ≤
    (frobeniusSq (1 : Matrix (Fin 2) (Fin 2) ℂ) + frobeniusSq (1 : Matrix (Fin 2) (Fin 2) ℂ)) / 2 :=
  nuclearNorm_le_of_eq_mul 1 1 (by simp)

end Transformer.Grokking.FlatSpectrum
