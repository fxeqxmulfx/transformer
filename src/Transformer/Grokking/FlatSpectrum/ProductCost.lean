module

public import Transformer.Grokking.FlatSpectrum.CutRange
public import Transformer.Grokking.FlatSpectrum.UnitaryAverage

/-!
# One product layer pays the Schatten-2/3 quasi-norm

Source: `review/flat_spectrum.md`, NV10. A layer whose units read a gate `g_u` from the features
of `x` and a value `w_u` from those of `y`, and write `o_u` into a fresh coordinate, writes the
table `M = G diag(o) Wᴴ`. Its squared weight is `½ (‖G‖² + ‖o‖² + ‖W‖²)`, and this file shows
that the weight is at least `productCost M = (3/2) ∑ σ_i(M)^{2/3}`.

The proof compares two orthonormal bases. Let `A` diagonalize `Mᴴ M` and `B` diagonalize `Yᴴ Y`
for `Y = diag(o) Wᴴ`. The squared moduli of `Aᴴ B` form a doubly stochastic matrix, and the
pointwise inequality `x^{1/3} ≤ x / (3 t²) + (2/3) t` averages through it.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq n]

/-- The cost of writing `X` with one product layer: `(3/2) ∑ σ_i^{2/3}`. -/
noncomputable def productCost (X : Matrix m n ℂ) : ℝ :=
  3 / 2 * ∑ i, singularValues X i ^ (2 / 3 : ℝ)

/-- **The key inequality** of NV10: `∑ σ_i(G Y)^{2/3} ≤ ‖G‖² / 3 + (2/3) ‖Y‖_S1`. -/
theorem sum_rpow_singularValues_mul_le (G : Matrix m k ℂ) (Y : Matrix k n ℂ) :
    ∑ i, singularValues (G * Y) i ^ (2 / 3 : ℝ) ≤ frobeniusSq G / 3 + 2 / 3 * nuclearNorm Y := by
  classical
  obtain ⟨A, hA, hMA⟩ : ∃ A : Matrix n n ℂ, Aᴴ * A = 1 ∧ Aᴴ * ((G * Y)ᴴ * (G * Y)) * A =
      diagonal fun i => ((singularValues (G * Y) i ^ 2 : ℝ) : ℂ) :=
    ⟨_, eigenvectorUnitary_conjTranspose_mul (isHermitian_conjTranspose_mul_self _),
      gram_diagonalization _⟩
  obtain ⟨B, hB, hYB⟩ : ∃ B : Matrix n n ℂ, Bᴴ * B = 1 ∧ Bᴴ * (Yᴴ * Y) * B =
      diagonal fun i => ((singularValues Y i ^ 2 : ℝ) : ℂ) :=
    ⟨_, eigenvectorUnitary_conjTranspose_mul (isHermitian_conjTranspose_mul_self _),
      gram_diagonalization _⟩
  set x : n → ℝ := fun i => singularValues (G * Y) i ^ 2 with hxdef
  set t : n → ℝ := singularValues Y with htdef
  have ht : ∀ j, 0 ≤ t j := fun j => Real.sqrt_nonneg _
  set Q := Aᴴ * B with hQdef
  have hQ : Qᴴ * Q = 1 := by
    calc Qᴴ * Q = Bᴴ * (A * Aᴴ) * B := by
          simp only [hQdef, conjTranspose_mul, conjTranspose_conjTranspose, Matrix.mul_assoc]
      _ = 1 := by rw [mul_eq_one_comm.mp hA, Matrix.mul_one, hB]
  have hdiag (j : n) : ∑ a, ‖((G * Y) * B) a j‖ ^ 2 = ∑ i, ‖Q i j‖ ^ 2 * x i := by
    have h := mul_diagonal_mul_apply_self (B := B) hA hMA j
    have hE : Bᴴ * ((G * Y)ᴴ * (G * Y)) * B = ((G * Y) * B)ᴴ * ((G * Y) * B) := by
      simp only [conjTranspose_mul, Matrix.mul_assoc]
    rw [hE, conjTranspose_mul_self_apply_self] at h
    exact_mod_cast h
  have hYB' : (Y * B)ᴴ * (Y * B) = diagonal fun i => ((t i ^ 2 : ℝ) : ℂ) := by
    rw [htdef, ← hYB]
    simp only [conjTranspose_mul, Matrix.mul_assoc]
  have hzero (j : n) (hj : t j = 0) : ∑ i, ‖Q i j‖ ^ 2 * x i = 0 := by
    rw [← hdiag j]
    refine Finset.sum_eq_zero fun a _ => ?_
    have hcol : ∀ b, (Y * B) b j = 0 := fun b => apply_eq_zero_of_gram hYB' (by simp [hj]) b
    have : ((G * Y) * B) a j = 0 := by
      rw [Matrix.mul_assoc, Matrix.mul_apply]
      exact Finset.sum_eq_zero fun b _ => by rw [hcol b, mul_zero]
    simp [this]
  set W₀ : Matrix k n ℂ := Y * B * diagonal fun j => (((t j)⁻¹ : ℝ) : ℂ) with hW₀def
  have hW₀ : W₀ᴴ * W₀ = diagonal fun j => (((t j)⁻¹ * t j ^ 2 * (t j)⁻¹ : ℝ) : ℂ) := by
    rw [hW₀def, conjTranspose_mul, conjTranspose_diagonal_ofReal,
      show ∀ C D : Matrix n n ℂ, C * (Y * B)ᴴ * (Y * B * D) = C * ((Y * B)ᴴ * (Y * B)) * D by
        intro C D; simp only [Matrix.mul_assoc], hYB', diagonal_mul_diagonal,
      diagonal_mul_diagonal]
    congr 1
    funext j
    push_cast
    ring
  have he (j : n) : (((t j)⁻¹ * t j ^ 2 * (t j)⁻¹ : ℝ) : ℂ) = 0 ∨
      (((t j)⁻¹ * t j ^ 2 * (t j)⁻¹ : ℝ) : ℂ) = 1 := by
    rcases eq_or_ne (t j) 0 with h | h
    · left
      simp [h]
    · right
      have : (t j)⁻¹ * t j ^ 2 * (t j)⁻¹ = 1 := by field_simp
      rw [this, Complex.ofReal_one]
  have hfrob : ∑ j, (t j ^ 2)⁻¹ * ∑ i, ‖Q i j‖ ^ 2 * x i = frobeniusSq (G * W₀) := by
    have hE : G * W₀ = (G * Y) * B * diagonal fun j => (((t j)⁻¹ : ℝ) : ℂ) := by
      simp only [hW₀def, Matrix.mul_assoc]
    rw [hE, frobeniusSq, Finset.sum_comm]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [← hdiag j, Finset.mul_sum]
    refine Finset.sum_congr rfl fun a _ => ?_
    rw [mul_diagonal, norm_mul, Complex.norm_real, Real.norm_eq_abs,
      abs_of_nonneg (inv_nonneg.mpr (ht j)), mul_pow, inv_pow]
    ring
  have hle : frobeniusSq (G * W₀) ≤ frobeniusSq G := by
    have := (isContraction_conjTranspose_of_gram he hW₀).frobeniusSq_mul_le Gᴴ
    rwa [← conjTranspose_mul, frobeniusSq_conjTranspose, frobeniusSq_conjTranspose] at this
  have hsum := sum_rpow_one_third_le (c := fun i j => ‖Q i j‖ ^ 2) (x := x) (t := t)
    (fun _ _ => sq_nonneg _) (sum_norm_sq_row_eq_one hQ) (sum_norm_sq_col_eq_one hQ)
    (fun _ => sq_nonneg _) ht hzero
  have hrw : ∑ i, singularValues (G * Y) i ^ (2 / 3 : ℝ) = ∑ i, x i ^ (1 / 3 : ℝ) := by
    refine Finset.sum_congr rfl fun i _ => ?_
    simp only [hxdef]
    have hσ : 0 ≤ singularValues (G * Y) i := Real.sqrt_nonneg _
    rw [← Real.rpow_natCast, ← Real.rpow_mul hσ]
    norm_num
  rw [hrw, nuclearNorm]
  rw [hfrob] at hsum
  linarith

/-- **One product layer pays at least `productCost`** (NV10):
`½ (‖G‖² + ‖o‖² + ‖W‖²) ≥ (3/2) ∑ σ_i(G diag(o) Wᴴ)^{2/3}`. -/
theorem productCost_le [DecidableEq k] (G : Matrix m k ℂ) (o : k → ℂ) (W : Matrix n k ℂ) :
    productCost (G * diagonal o * Wᴴ) ≤
      (frobeniusSq G + ∑ u, ‖o u‖ ^ 2 + frobeniusSq W) / 2 := by
  have hY := nuclearNorm_le_of_eq_mul (diagonal o) W (rfl : diagonal o * Wᴴ = _)
  have hkey := sum_rpow_singularValues_mul_le G (diagonal o * Wᴴ)
  have hdiag : frobeniusSq (diagonal o) = ∑ u, ‖o u‖ ^ 2 := by
    simp only [frobeniusSq]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Finset.sum_eq_single i (fun j _ hj => by rw [diagonal_apply_ne _ hj.symm]; simp)
      (by simp), diagonal_apply_eq]
  rw [productCost, Matrix.mul_assoc]
  rw [hdiag] at hY
  linarith

end Transformer.Grokking.FlatSpectrum
