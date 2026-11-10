module

public import Transformer.Grokking.FlatSpectrum.Training.Defs

/-!
# The trace pairing

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD1, the
algebra its proofs use. `pairing A B = Re tr (Aᴴ B)` is a real symmetric bilinear form with
`pairing A A = ‖A‖_S2²` (`pairing_self`), left and right multiplications have their adjoints
(`pairing_mul_left`, `pairing_mul_right`), Cauchy–Schwarz reads
`pairing A B ² ≤ ‖A‖_S2² ‖B‖_S2²` (`pairing_sq_le`), and the Frobenius norm is submultiplicative
(`frobeniusSq_mul_le`). A positive semidefinite `M` has `pairing W (M W) ≥ 0`
(`pairing_nonneg_of_posSemidef`), and a Hermitian `M` with `Re xᴴ M x ≥ 0` for every vector `x`
is positive semidefinite (`posSemidef_of_re_nonneg`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix
open scoped ComplexOrder

variable {l m n k : Type*} [Fintype l] [Fintype m] [Fintype n] [Fintype k]

lemma pairing_self (A : Matrix m n ℂ) : pairing A A = frobeniusSq A :=
  (frobeniusSq_eq_re_trace A).symm

lemma pairing_comm (A B : Matrix m n ℂ) : pairing A B = pairing B A := by
  rw [pairing, pairing, show Bᴴ * A = (Aᴴ * B)ᴴ by
    rw [conjTranspose_mul, conjTranspose_conjTranspose], trace_conjTranspose, Complex.star_def,
    Complex.conj_re]

lemma pairing_add_right (A B C : Matrix m n ℂ) :
    pairing A (B + C) = pairing A B + pairing A C := by
  simp only [pairing, Matrix.mul_add, trace_add, Complex.add_re]

lemma pairing_add_left (A B C : Matrix m n ℂ) :
    pairing (A + B) C = pairing A C + pairing B C := by
  simp only [pairing, conjTranspose_add, Matrix.add_mul, trace_add, Complex.add_re]

lemma pairing_sub_right (A B C : Matrix m n ℂ) :
    pairing A (B - C) = pairing A B - pairing A C := by
  simp only [pairing, Matrix.mul_sub, trace_sub, Complex.sub_re]

lemma pairing_sub_left (A B C : Matrix m n ℂ) :
    pairing (A - B) C = pairing A C - pairing B C := by
  simp only [pairing, conjTranspose_sub, Matrix.sub_mul, trace_sub, Complex.sub_re]

lemma pairing_smul_right (t : ℝ) (A B : Matrix m n ℂ) :
    pairing A (t • B) = t * pairing A B := by
  simp only [pairing, Matrix.mul_smul, trace_smul, Complex.smul_re, smul_eq_mul]

lemma pairing_smul_left (t : ℝ) (A B : Matrix m n ℂ) :
    pairing (t • A) B = t * pairing A B := by
  rw [pairing_comm, pairing_smul_right, pairing_comm]

lemma pairing_zero_right (A : Matrix m n ℂ) : pairing A 0 = 0 := by
  simp [pairing]

lemma pairing_zero_left (A : Matrix m n ℂ) : pairing 0 A = 0 := by
  simp [pairing]

/-- The adjoint of a left multiplication: `⟨C A, B⟩ = ⟨A, Cᴴ B⟩`. -/
lemma pairing_mul_left (C : Matrix l m ℂ) (A : Matrix m n ℂ) (B : Matrix l n ℂ) :
    pairing (C * A) B = pairing A (Cᴴ * B) := by
  rw [pairing, pairing, conjTranspose_mul, Matrix.mul_assoc]

/-- The adjoint of a right multiplication: `⟨A C, B⟩ = ⟨A, B Cᴴ⟩`. -/
lemma pairing_mul_right (A : Matrix m k ℂ) (C : Matrix k n ℂ) (B : Matrix m n ℂ) :
    pairing (A * C) B = pairing A (B * Cᴴ) := by
  rw [pairing, pairing, conjTranspose_mul, Matrix.mul_assoc, Matrix.trace_mul_comm,
    Matrix.mul_assoc]

/-- Cauchy–Schwarz: `⟨A, B⟩² ≤ ‖A‖_S2² ‖B‖_S2²`. -/
lemma pairing_sq_le (A B : Matrix m n ℂ) :
    pairing A B ^ 2 ≤ frobeniusSq A * frobeniusSq B := by
  have h : |pairing A B| ≤ √(frobeniusSq A) * √(frobeniusSq B) :=
    (Complex.abs_re_le_norm _).trans (norm_trace_conjTranspose_mul_le A B)
  calc pairing A B ^ 2 = |pairing A B| ^ 2 := (sq_abs _).symm
    _ ≤ (√(frobeniusSq A) * √(frobeniusSq B)) ^ 2 := by gcongr
    _ = frobeniusSq A * frobeniusSq B := by
        rw [mul_pow, Real.sq_sqrt (frobeniusSq_nonneg A), Real.sq_sqrt (frobeniusSq_nonneg B)]

/-- From `x² ≤ a b ≤ c²` with `c ≥ 0`: `|x| ≤ c`. -/
lemma abs_pairing_le {A B : Matrix m n ℂ} {c : ℝ} (hc : 0 ≤ c)
    (h : frobeniusSq A * frobeniusSq B ≤ c ^ 2) : |pairing A B| ≤ c :=
  abs_le_of_sq_le_sq ((pairing_sq_le A B).trans h) hc

example := abs_pairing_le (A := (1 : Matrix (Fin 1) (Fin 1) ℂ)) (B := 1) (c := 1) zero_le_one
  (by simp [frobeniusSq])

lemma frobeniusSq_add (A B : Matrix m n ℂ) :
    frobeniusSq (A + B) = frobeniusSq A + 2 * pairing A B + frobeniusSq B := by
  simp only [← pairing_self, pairing_add_left, pairing_add_right, pairing_comm B A]
  ring

lemma frobeniusSq_sub (A B : Matrix m n ℂ) :
    frobeniusSq (A - B) = frobeniusSq A - 2 * pairing A B + frobeniusSq B := by
  simp only [← pairing_self, pairing_sub_left, pairing_sub_right, pairing_comm B A]
  ring

lemma frobeniusSq_add_le (A B : Matrix m n ℂ) :
    frobeniusSq (A + B) ≤ 2 * frobeniusSq A + 2 * frobeniusSq B := by
  have h1 := frobeniusSq_add A B
  have h2 := frobeniusSq_sub A B
  have h3 := frobeniusSq_nonneg (A - B)
  linarith

lemma frobeniusSq_smul (t : ℝ) (A : Matrix m n ℂ) :
    frobeniusSq (t • A) = t ^ 2 * frobeniusSq A := by
  simp only [← pairing_self, pairing_smul_left, pairing_smul_right]
  ring

lemma frobeniusSq_conjTranspose (A : Matrix m n ℂ) : frobeniusSq Aᴴ = frobeniusSq A := by
  simp only [frobeniusSq, conjTranspose_apply, norm_star]
  exact Finset.sum_comm

/-- The Frobenius norm is submultiplicative: `‖A B‖_S2² ≤ ‖A‖_S2² ‖B‖_S2²`. -/
lemma frobeniusSq_mul_le (A : Matrix l m ℂ) (B : Matrix m n ℂ) :
    frobeniusSq (A * B) ≤ frobeniusSq A * frobeniusSq B := by
  have hent (i : l) (j : n) :
      ‖(A * B) i j‖ ^ 2 ≤ (∑ k, ‖A i k‖ ^ 2) * ∑ k, ‖B k j‖ ^ 2 := by
    have h := norm_sum_star_mul_le Finset.univ (fun k => star (A i k)) fun k => B k j
    simp only [star_star, norm_star] at h
    calc ‖(A * B) i j‖ ^ 2 ≤ (√(∑ k, ‖A i k‖ ^ 2) * √(∑ k, ‖B k j‖ ^ 2)) ^ 2 := by
          rw [mul_apply]
          gcongr
      _ = _ := by
          rw [mul_pow, Real.sq_sqrt (Finset.sum_nonneg fun _ _ => sq_nonneg _),
            Real.sq_sqrt (Finset.sum_nonneg fun _ _ => sq_nonneg _)]
  calc frobeniusSq (A * B) ≤ ∑ i, ∑ j, (∑ k, ‖A i k‖ ^ 2) * ∑ k, ‖B k j‖ ^ 2 :=
        Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => hent i j
    _ = frobeniusSq A * frobeniusSq B := by
        rw [← Finset.sum_mul_sum, frobeniusSq, frobeniusSq]
        congr 1
        exact Finset.sum_comm

/-- A positive semidefinite `M` has `⟨W, M W⟩ ≥ 0` for every `W`. -/
lemma pairing_nonneg_of_posSemidef {M : Matrix m m ℂ} (hM : M.PosSemidef) (W : Matrix m k ℂ) :
    0 ≤ pairing W (M * W) := by
  have h := (hM.conjTranspose_mul_mul_same W).trace_nonneg
  rw [Matrix.mul_assoc] at h
  exact (Complex.nonneg_iff.mp h).1

example := pairing_nonneg_of_posSemidef (PosSemidef.one (n := Fin 2) (R := ℂ))
  (1 : Matrix (Fin 2) (Fin 2) ℂ)

/-- A single column pairs as its vector does: `⟨[x], [y]⟩ = Re xᴴ y`. -/
lemma pairing_replicateCol (x y : m → ℂ) :
    pairing (replicateCol Unit x) (replicateCol Unit y) = (star x ⬝ᵥ y).re := by
  rw [pairing, conjTranspose_replicateCol, replicateRow_mul_replicateCol]
  simp [trace]

/-- A Hermitian `M` with `Re xᴴ M x ≥ 0` for every vector `x` is positive semidefinite. -/
lemma posSemidef_of_re_nonneg [DecidableEq m] {M : Matrix m m ℂ} (hM : M.IsHermitian)
    (h : ∀ x : m → ℂ, 0 ≤ (star x ⬝ᵥ M *ᵥ x).re) : M.PosSemidef :=
  PosSemidef.of_dotProduct_mulVec_nonneg hM fun x =>
    Complex.nonneg_iff.mpr ⟨h x, by simpa using (hM.im_star_dotProduct_mulVec_self x).symm⟩

example := posSemidef_of_re_nonneg (M := (1 : Matrix (Fin 2) (Fin 2) ℂ)) isHermitian_one
  fun x => by
    rw [one_mulVec, star_dotProduct_self_eq_sum_norm_sq, Complex.ofReal_re]
    exact Finset.sum_nonneg fun _ _ => sq_nonneg _

/-- A Hermitian `M` with `⟨W, M W⟩ ≥ 0` for every one-column `W` is positive semidefinite. -/
lemma posSemidef_of_pairing_nonneg [DecidableEq m] {M : Matrix m m ℂ} (hM : M.IsHermitian)
    (h : ∀ W : Matrix m Unit ℂ, 0 ≤ pairing W (M * W)) : M.PosSemidef :=
  posSemidef_of_re_nonneg hM fun x => by
    rw [← pairing_replicateCol, replicateCol_mulVec]
    exact h _

example := posSemidef_of_pairing_nonneg (M := (1 : Matrix (Fin 2) (Fin 2) ℂ)) isHermitian_one
  fun W => by
    rw [Matrix.one_mul, pairing_self]
    exact frobeniusSq_nonneg W

/-- A Hermitian `M` with `⟨W, M W⟩ > 0` for every one-column `W ≠ 0` is positive definite. -/
lemma posDef_of_pairing_pos [DecidableEq m] {M : Matrix m m ℂ} (hM : M.IsHermitian)
    (h : ∀ W : Matrix m Unit ℂ, W ≠ 0 → 0 < pairing W (M * W)) : M.PosDef :=
  PosDef.of_dotProduct_mulVec_pos hM fun x hx => by
    have h1 := h (replicateCol Unit x) fun h0 => hx (by
      funext i
      simpa using congrFun (congrFun h0 i) ())
    rw [← replicateCol_mulVec, pairing_replicateCol] at h1
    exact Complex.pos_iff.mpr ⟨h1, by simpa using (hM.im_star_dotProduct_mulVec_self x).symm⟩

example := posDef_of_pairing_pos (M := (1 : Matrix (Fin 2) (Fin 2) ℂ)) isHermitian_one
  fun W hW => by
    rw [Matrix.one_mul, pairing_self]
    exact lt_of_le_of_ne (frobeniusSq_nonneg W) fun h0 =>
      hW (eq_zero_of_frobeniusSq_eq_zero h0.symm)

end Transformer.Grokking.FlatSpectrum.Training
