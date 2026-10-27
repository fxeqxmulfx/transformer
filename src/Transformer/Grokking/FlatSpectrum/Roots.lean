module

public import Transformer.Grokking.FlatSpectrum.Factorization

/-!
# Computing nuclear norms exactly

Source: `review/flat_spectrum.md`, Proposition 2 and the Corollary. A unitary
diagonalization `Wᴴ Xᴴ X W = diag (d²)` gives `‖X‖_S1 = ∑ d`, and so does every positive
semidefinite square root `P` of `Xᴴ X`: `‖X‖_S1 = tr P`. Hence the nuclear norm depends
only on `Xᴴ X`, scales by `|c|`, ignores isometries on both sides, splits off an identity
block, and on `2 × 2` matrices equals `√(‖B‖_S2² + 2 |det B|)`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]

/-- A unitary diagonalization `Wᴴ (Xᴴ X) W = diag (d²)`, `d ≥ 0`, computes
`‖X‖_S1 = ∑ d`: the factorization bounds it above, the polar contraction below. -/
theorem nuclearNorm_eq_of_diagonalization {X : Matrix m n ℂ} {W : Matrix n n ℂ} {d : n → ℝ}
    (hd : ∀ i, 0 ≤ d i) (hW : Wᴴ * W = 1)
    (hX : Wᴴ * (Xᴴ * X) * W = diagonal fun i => ((d i ^ 2 : ℝ) : ℂ)) :
    nuclearNorm X = ∑ i, d i := by
  obtain ⟨hfac, hU, hV⟩ := factor_of_diagonalization hd hW hX
  obtain ⟨hZ, htr⟩ := polar_of_diagonalization hW hX
  refine le_antisymm ?_ ?_
  · have := nuclearNorm_le_of_eq_mul _ _ hfac
    rw [hU, hV] at this
    linarith
  · have h := norm_trace_le_nuclearNorm hZ X
    rwa [htr, ← Complex.ofReal_sum, Complex.norm_real,
      Real.norm_of_nonneg (Finset.sum_nonneg fun i _ => hd i)] at h


example : nuclearNorm (1 : Matrix (Fin 2) (Fin 2) ℂ) = ∑ _i : Fin 2, (1 : ℝ) :=
  nuclearNorm_eq_of_diagonalization (W := 1) (d := fun _ => 1) (fun _ => zero_le_one) (by simp)
    (by simp)

omit [DecidableEq n] in
/-- `tr (Aᴴ A) = ‖A‖_S2²`. -/
lemma trace_conjTranspose_mul_self (A : Matrix m n ℂ) : (Aᴴ * A).trace = (frobeniusSq A : ℂ) := by
  simp only [trace, diag_apply, conjTranspose_mul_self_apply_self, frobeniusSq]
  push_cast
  exact Finset.sum_comm

omit [DecidableEq n] in
lemma eq_zero_of_frobeniusSq_eq_zero {A : Matrix m n ℂ} (h : frobeniusSq A = 0) : A = 0 := by
  ext i j
  have h1 := (Finset.sum_eq_zero_iff_of_nonneg fun i _ =>
    Finset.sum_nonneg fun j _ => sq_nonneg ‖A i j‖).mp h i (Finset.mem_univ _)
  have h2 := (Finset.sum_eq_zero_iff_of_nonneg fun j _ => sq_nonneg ‖A i j‖).mp h1 j
    (Finset.mem_univ _)
  simpa using h2

example := eq_zero_of_frobeniusSq_eq_zero (A := (0 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp [frobeniusSq])

lemma nuclearNorm_zero : nuclearNorm (0 : Matrix m n ℂ) = 0 := by
  rw [nuclearNorm_eq_of_diagonalization (W := 1) (d := 0) (fun _ => le_rfl) (by simp)
    (by simp)]
  simp

/-- `‖X‖_S1 = tr P` for every positive semidefinite `P` with `P P = Xᴴ X`. -/
theorem nuclearNorm_eq_trace_of_mul_self {X : Matrix m n ℂ} {P : Matrix n n ℂ}
    (hP : P.PosSemidef) (hPX : P * P = Xᴴ * X) : (nuclearNorm X : ℂ) = P.trace := by
  set Q : Matrix n n ℂ := (hP.1.eigenvectorUnitary : Matrix n n ℂ)
  have hQ : Qᴴ * Q = 1 := eigenvectorUnitary_conjTranspose_mul hP.1
  have hQ' : Q * Qᴴ = 1 := eigenvectorUnitary_mul_conjTranspose hP.1
  have hD : Qᴴ * P * Q = diagonal fun i => ((hP.1.eigenvalues i : ℝ) : ℂ) :=
    eigenvectorUnitary_conj hP.1
  have hX : Qᴴ * (Xᴴ * X) * Q = diagonal fun i => ((hP.1.eigenvalues i ^ 2 : ℝ) : ℂ) := by
    have : Qᴴ * (P * P) * Q = Qᴴ * P * Q * (Qᴴ * P * Q) := by
      rw [show Qᴴ * P * Q * (Qᴴ * P * Q) = Qᴴ * P * (Q * Qᴴ) * P * Q by
        simp only [Matrix.mul_assoc], hQ', Matrix.mul_one]
      simp only [Matrix.mul_assoc]
    rw [← hPX, this, hD, diagonal_mul_diagonal]
    congr 1
    funext i
    push_cast
    ring
  rw [nuclearNorm_eq_of_diagonalization (fun i => hP.eigenvalues_nonneg i) hQ hX,
    hP.1.trace_eq_sum_eigenvalues]
  push_cast
  rfl

example := nuclearNorm_eq_trace_of_mul_self (X := (1 : Matrix (Fin 2) (Fin 2) ℂ)) PosSemidef.one (by simp)

/-- Every Gram matrix `Xᴴ X` has a positive semidefinite square root. -/
theorem exists_posSemidef_mul_self_eq (X : Matrix m n ℂ) :
    ∃ P : Matrix n n ℂ, P.PosSemidef ∧ P * P = Xᴴ * X := by
  set hX := isHermitian_conjTranspose_mul_self X
  set W : Matrix n n ℂ := (hX.eigenvectorUnitary : Matrix n n ℂ)
  set D : Matrix n n ℂ := diagonal fun i => ((singularValues X i : ℝ) : ℂ)
  have hW : Wᴴ * W = 1 := eigenvectorUnitary_conjTranspose_mul hX
  have hW' : W * Wᴴ = 1 := eigenvectorUnitary_mul_conjTranspose hX
  have hWa : ∀ A : Matrix n n ℂ, Wᴴ * (W * A) = A := fun A => by
    rw [← Matrix.mul_assoc, hW, Matrix.one_mul]
  have hWb : ∀ A : Matrix n n ℂ, W * (Wᴴ * A) = A := fun A => by
    rw [← Matrix.mul_assoc, hW', Matrix.one_mul]
  have hDD : D * D = Wᴴ * (Xᴴ * X) * W := by
    rw [gram_diagonalization X, diagonal_mul_diagonal]
    congr 1
    funext i
    push_cast
    ring
  refine ⟨W * D * Wᴴ, (posSemidef_diagonal_iff.mpr fun i => ?_).mul_mul_conjTranspose_same W,
    ?_⟩
  · rw [Complex.zero_le_real]
    exact Real.sqrt_nonneg _
  · calc W * D * Wᴴ * (W * D * Wᴴ) = W * (D * D) * Wᴴ := by simp only [Matrix.mul_assoc, hWa]
      _ = Xᴴ * X := by rw [hDD]; simp only [Matrix.mul_assoc, hWb, hW', Matrix.mul_one]

/-- The nuclear norm depends only on the Gram matrix `Xᴴ X`. -/
theorem nuclearNorm_eq_of_gram_eq {m' : Type*} [Fintype m'] {X : Matrix m n ℂ}
    {Y : Matrix m' n ℂ} (h : Xᴴ * X = Yᴴ * Y) : nuclearNorm X = nuclearNorm Y := by
  obtain ⟨P, hP, hPX⟩ := exists_posSemidef_mul_self_eq X
  exact_mod_cast (nuclearNorm_eq_trace_of_mul_self hP hPX).trans
    (nuclearNorm_eq_trace_of_mul_self hP (hPX.trans h)).symm

example := nuclearNorm_eq_of_gram_eq (X := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (Y := 1) rfl

/-- `‖c X‖_S1 = |c| ‖X‖_S1`. -/
theorem nuclearNorm_smul (c : ℂ) (X : Matrix m n ℂ) :
    nuclearNorm (c • X) = ‖c‖ * nuclearNorm X := by
  obtain ⟨P, hP, hPX⟩ := exists_posSemidef_mul_self_eq X
  have h := nuclearNorm_eq_trace_of_mul_self (X := c • X)
    (hP.smul (Complex.zero_le_real.mpr (norm_nonneg c))) (by
      rw [Matrix.smul_mul, Matrix.mul_smul, smul_smul, hPX, conjTranspose_smul, Matrix.smul_mul,
        Matrix.mul_smul, smul_smul, Complex.star_def, Complex.conj_mul', sq])
  rw [trace_smul, smul_eq_mul, ← nuclearNorm_eq_trace_of_mul_self hP hPX] at h
  exact_mod_cast h

/-- Isometries on both sides keep the nuclear norm: `‖J₁ B J₂ᴴ‖_S1 = ‖B‖_S1` when
`J₁ᴴ J₁ = 1` and `J₂ᴴ J₂ = 1`. -/
theorem nuclearNorm_isometry {k₁ k₂ : Type*} [Fintype k₁] [Fintype k₂] [DecidableEq k₁]
    [DecidableEq k₂] {J₁ : Matrix m k₁ ℂ} {J₂ : Matrix n k₂ ℂ} (h₁ : J₁ᴴ * J₁ = 1)
    (h₂ : J₂ᴴ * J₂ = 1) (B : Matrix k₁ k₂ ℂ) : nuclearNorm (J₁ * B * J₂ᴴ) = nuclearNorm B := by
  obtain ⟨P, hP, hPB⟩ := exists_posSemidef_mul_self_eq B
  have h₂a : ∀ A : Matrix k₂ n ℂ, J₂ᴴ * (J₂ * A) = A := fun A => by
    rw [← Matrix.mul_assoc, h₂, Matrix.one_mul]
  have h₁a : ∀ A : Matrix k₁ n ℂ, J₁ᴴ * (J₁ * A) = A := fun A => by
    rw [← Matrix.mul_assoc, h₁, Matrix.one_mul]
  have h := nuclearNorm_eq_trace_of_mul_self (X := J₁ * B * J₂ᴴ)
    (hP.mul_mul_conjTranspose_same J₂) (by
      rw [conjTranspose_mul, conjTranspose_mul, conjTranspose_conjTranspose]
      simp only [Matrix.mul_assoc, h₂a, h₁a]
      rw [← Matrix.mul_assoc P, hPB, Matrix.mul_assoc])
  rw [Matrix.trace_mul_comm, ← Matrix.mul_assoc, h₂, Matrix.one_mul,
    ← nuclearNorm_eq_trace_of_mul_self hP hPB] at h
  exact_mod_cast h

example (B : Matrix (Fin 2) (Fin 2) ℂ) :=
  nuclearNorm_isometry (J₁ := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (J₂ := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) (by simp) B

end Transformer.Grokking.FlatSpectrum
