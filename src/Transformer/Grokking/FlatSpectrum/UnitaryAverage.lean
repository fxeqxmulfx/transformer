module

public import Transformer.Grokking.FlatSpectrum.Factorization

/-!
# Averaging through two orthonormal bases

Source: `review/flat_spectrum.md`, NV10. The squared moduli of a unitary form a doubly stochastic
matrix, so the diagonal of a Gram matrix in a second orthonormal basis is an average of its
eigenvalues, and the pointwise inequality `x^{1/3} ≤ x / (3 t²) + (2/3) t` averages through it.
A matrix whose columns are orthonormal or zero has a contractive adjoint.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq n]

/-- AM–GM in the form the key inequality uses: `x^{1/3} ≤ x / (3 t²) + (2/3) t`. -/
lemma rpow_one_third_le {x t : ℝ} (hx : 0 ≤ x) (ht : 0 < t) :
    x ^ (1 / 3 : ℝ) ≤ x * (t ^ 2)⁻¹ / 3 + 2 / 3 * t := by
  have hu0 : 0 ≤ x ^ (1 / 3 : ℝ) := Real.rpow_nonneg hx _
  have hu3 : (x ^ (1 / 3 : ℝ)) ^ 3 = x := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hx]
    norm_num
  generalize x ^ (1 / 3 : ℝ) = u at hu0 hu3
  subst hu3
  have hE : u ^ 3 * (t ^ 2)⁻¹ / 3 + 2 / 3 * t - u =
      (u - t) ^ 2 * (u + 2 * t) / (3 * t ^ 2) := by
    field_simp
    ring
  have : 0 ≤ (u - t) ^ 2 * (u + 2 * t) / (3 * t ^ 2) :=
    div_nonneg (mul_nonneg (sq_nonneg _) (by linarith)) (by positivity)
  linarith

example := rpow_one_third_le (x := 1) (t := 1) zero_le_one one_pos

omit [DecidableEq n] in
/-- The averaging step: doubly stochastic weights carry the pointwise bound to the sums. A zero
`t j` is allowed when column `j` of the weights puts no mass on a nonzero `x i`. -/
lemma sum_rpow_one_third_le {c : n → n → ℝ} {x t : n → ℝ} (hc : ∀ i j, 0 ≤ c i j)
    (hrow : ∀ i, ∑ j, c i j = 1) (hcol : ∀ j, ∑ i, c i j = 1) (hx : ∀ i, 0 ≤ x i)
    (ht : ∀ j, 0 ≤ t j) (hzero : ∀ j, t j = 0 → ∑ i, c i j * x i = 0) :
    ∑ i, x i ^ (1 / 3 : ℝ) ≤ (∑ j, (t j ^ 2)⁻¹ * ∑ i, c i j * x i) / 3 + 2 / 3 * ∑ j, t j := by
  have hterm (i j : n) : c i j * x i ^ (1 / 3 : ℝ) ≤
      (t j ^ 2)⁻¹ * (c i j * x i) / 3 + 2 / 3 * (c i j * t j) := by
    rcases (ht j).eq_or_lt with h | h
    · have h0 := (Finset.sum_eq_zero_iff_of_nonneg fun i _ => mul_nonneg (hc i j) (hx i)).mp
        (hzero j h.symm) i (Finset.mem_univ _)
      rcases mul_eq_zero.mp h0 with h1 | h1
      · simp [h1]
      · rw [h1, Real.zero_rpow (by norm_num), ← h]
        simp
    · calc c i j * x i ^ (1 / 3 : ℝ)
          ≤ c i j * (x i * (t j ^ 2)⁻¹ / 3 + 2 / 3 * t j) :=
            mul_le_mul_of_nonneg_left (rpow_one_third_le (hx i) h) (hc i j)
        _ = _ := by ring
  calc ∑ i, x i ^ (1 / 3 : ℝ) = ∑ i, ∑ j, c i j * x i ^ (1 / 3 : ℝ) := by
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [← Finset.sum_mul, hrow, one_mul]
    _ ≤ ∑ i, ∑ j, ((t j ^ 2)⁻¹ * (c i j * x i) / 3 + 2 / 3 * (c i j * t j)) :=
        Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => hterm i j
    _ = (∑ j, (t j ^ 2)⁻¹ * ∑ i, c i j * x i) / 3 + 2 / 3 * ∑ j, t j := by
        rw [Finset.sum_comm, Finset.sum_div, Finset.mul_sum, ← Finset.sum_add_distrib]
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [Finset.sum_add_distrib, ← Finset.sum_div, ← Finset.mul_sum, ← Finset.mul_sum,
          ← Finset.sum_mul, hcol, one_mul]

example := sum_rpow_one_third_le (n := Fin 1) (c := fun _ _ => 1) (x := fun _ => 1)
  (t := fun _ => 1) (fun _ _ => zero_le_one) (by simp) (by simp) (fun _ => zero_le_one)
  (fun _ => zero_le_one) (by simp)

/-- The columns of a unitary are unit vectors. -/
lemma sum_norm_sq_col_eq_one {Q : Matrix n n ℂ} (hQ : Qᴴ * Q = 1) (j : n) :
    ∑ i, ‖Q i j‖ ^ 2 = 1 := by
  have h := congrFun (congrFun hQ j) j
  rw [conjTranspose_mul_self_apply_self, one_apply_eq] at h
  exact_mod_cast h

/-- The rows of a unitary are unit vectors. -/
lemma sum_norm_sq_row_eq_one {Q : Matrix n n ℂ} (hQ : Qᴴ * Q = 1) (i : n) :
    ∑ j, ‖Q i j‖ ^ 2 = 1 := by
  have h := sum_norm_sq_col_eq_one (Q := Qᴴ)
    (by rw [conjTranspose_conjTranspose]; exact mul_eq_one_comm.mp hQ) i
  simpa only [conjTranspose_apply, norm_star] using h

example := sum_norm_sq_col_eq_one (Q := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) 0
example := sum_norm_sq_row_eq_one (Q := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) 0

/-- In a second orthonormal basis, the diagonal of `P` averages the eigenvalues of `P`. -/
lemma mul_diagonal_mul_apply_self {A B P : Matrix n n ℂ} {x : n → ℝ} (hA : Aᴴ * A = 1)
    (hP : Aᴴ * P * A = diagonal fun i => (x i : ℂ)) (j : n) :
    (Bᴴ * P * B) j j = ((∑ i, ‖(Aᴴ * B) i j‖ ^ 2 * x i : ℝ) : ℂ) := by
  have hA' : A * Aᴴ = 1 := mul_eq_one_comm.mp hA
  have hPA : P = A * (diagonal fun i => (x i : ℂ)) * Aᴴ := by
    rw [← hP]
    calc P = (A * Aᴴ) * P * (A * Aᴴ) := by rw [hA', Matrix.one_mul, Matrix.mul_one]
      _ = A * (Aᴴ * P * A) * Aᴴ := by simp only [Matrix.mul_assoc]
  have hE : Bᴴ * P * B = (Aᴴ * B)ᴴ * (diagonal fun i => (x i : ℂ)) * (Aᴴ * B) := by
    rw [hPA]
    simp only [conjTranspose_mul, conjTranspose_conjTranspose, Matrix.mul_assoc]
  rw [hE, Matrix.mul_assoc, Matrix.mul_apply]
  simp only [diagonal_mul, conjTranspose_apply]
  push_cast
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Complex.star_def, mul_left_comm, Complex.conj_mul']
  ring

example := mul_diagonal_mul_apply_self (A := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (B := 1) (P := 1)
  (x := fun _ => 1) (by simp) (by simp) 0

omit [Fintype n] in
/-- A column whose Gram entry vanishes is zero. -/
lemma apply_eq_zero_of_gram {H : Matrix m n ℂ} {e : n → ℂ} (hH : Hᴴ * H = diagonal e) {j : n}
    (hj : e j = 0) (a : m) : H a j = 0 := by
  have h := congrFun (congrFun hH j) j
  rw [conjTranspose_mul_self_apply_self, diagonal_apply_eq, hj] at h
  have h0 : ∑ b, ‖H b j‖ ^ 2 = 0 := by exact_mod_cast h
  have := (Finset.sum_eq_zero_iff_of_nonneg fun b _ => sq_nonneg ‖H b j‖).mp h0 a
    (Finset.mem_univ _)
  simpa using this

example := apply_eq_zero_of_gram (H := (0 : Matrix (Fin 2) (Fin 2) ℂ)) (e := fun _ => 0)
  (by simp) (j := 0) rfl 0

/-- A matrix with orthonormal or zero columns has a contractive adjoint. -/
lemma isContraction_conjTranspose_of_gram [DecidableEq k] {W : Matrix k n ℂ} {e : n → ℂ}
    (he : ∀ j, e j = 0 ∨ e j = 1) (hW : Wᴴ * W = diagonal e) : IsContraction Wᴴ := by
  have hWe : W * diagonal e = W := by
    ext a j
    rw [mul_diagonal]
    rcases he j with h | h
    · rw [h, mul_zero, apply_eq_zero_of_gram hW h a]
    · rw [h, mul_one]
  have hP : (W * Wᴴ) * (W * Wᴴ) = W * Wᴴ := by
    rw [Matrix.mul_assoc, ← Matrix.mul_assoc Wᴴ, hW, ← Matrix.mul_assoc, hWe]
  have hsq : (1 - W * Wᴴ)ᴴ * (1 - W * Wᴴ) = 1 - W * Wᴴ := by
    rw [conjTranspose_sub, conjTranspose_one, conjTranspose_mul, conjTranspose_conjTranspose]
    simp only [Matrix.mul_sub, Matrix.sub_mul, Matrix.one_mul, Matrix.mul_one, hP]
    abel
  rw [IsContraction, conjTranspose_conjTranspose, ← hsq]
  exact posSemidef_conjTranspose_mul_self _

example := isContraction_conjTranspose_of_gram (W := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (e := fun _ => 1) (fun _ => Or.inr rfl) (by simp)

end Transformer.Grokking.FlatSpectrum
