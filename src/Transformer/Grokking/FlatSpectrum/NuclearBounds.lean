module

public import Transformer.Grokking.FlatSpectrum.ContractionDual
public import Transformer.Grokking.FlatSpectrum.Interpolation

/-!
# The nuclear norm against entries, contractions and linear maps

Source: `review/flat_spectrum.md`, *Rank above one* (added), item 6.1, the lemmas of its proof.
A matrix unit is a contraction (`isContraction_single_one`), so no entry exceeds the nuclear norm
(`norm_apply_le_nuclearNorm`). A product of contractions is a contraction
(`IsContraction.mul`), and so is the projection `1 − U Uᴴ`
(`isContraction_one_sub_mul_conjTranspose`).
Multiplying by contractions does not increase the nuclear norm (`nuclearNorm_mul_mul_le`): a
contraction `Z` attaining `‖P X Q‖_S1` gives the contraction `Pᴴ Z Qᴴ`, which pairs with `X` to
the same value. Every real-linear map `g` between matrix spaces is bounded in the nuclear norm
(`exists_nuclearNorm_map_le`): `X = ∑ Re X_ij e_ij + Im X_ij (i e_ij)`, so
`‖g X‖_S1 ≤ ∑ |X_ij| (‖g e_ij‖_S1 + ‖g (i e_ij)‖_S1) ≤ C ‖X‖_S1`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- A matrix unit is a contraction. -/
lemma isContraction_single_one (i : m) (j : n) : IsContraction (single i j (1 : ℂ)) := by
  rw [isContraction_iff_sum_norm_sq_le]
  intro v
  rw [single_mulVec, one_mul,
    sum_eq_single i (fun a _ ha => by rw [Function.update_of_ne ha]; simp) (by simp),
    Function.update_self]
  exact single_le_sum (f := fun b => ‖v b‖ ^ 2) (fun b _ => sq_nonneg _) (mem_univ j)

example := isContraction_single_one (m := Fin 2) (n := Fin 3) 0 1

/-- **No entry exceeds the nuclear norm**: `|X_ij| ≤ ‖X‖_S1`. -/
theorem norm_apply_le_nuclearNorm (X : Matrix m n ℂ) (i : m) (j : n) :
    ‖X i j‖ ≤ nuclearNorm X := by
  have h : ((single i j (1 : ℂ))ᴴ * X).trace = X i j := by
    rw [show (single i j (1 : ℂ))ᴴ * X = (Xᴴ * single i j (1 : ℂ))ᴴ by
        rw [conjTranspose_mul, conjTranspose_conjTranspose],
      trace_conjTranspose, trace_conjTranspose_mul_single, mul_one, star_star]
  rw [← h]
  exact norm_trace_le_nuclearNorm (isContraction_single_one i j) X

omit [DecidableEq m] in
/-- **A product of contractions is a contraction**: `‖A B v‖ ≤ ‖B v‖ ≤ ‖v‖`. -/
theorem IsContraction.mul {k : Type*} [Fintype k] [DecidableEq k] {A : Matrix m k ℂ}
    {B : Matrix k n ℂ} (hA : IsContraction A) (hB : IsContraction B) : IsContraction (A * B) := by
  rw [isContraction_iff_sum_norm_sq_le] at hA hB ⊢
  intro v
  rw [← mulVec_mulVec]
  exact (hA (B *ᵥ v)).trans (hB v)

example := (isContraction_one (n := Fin 2)).mul isContraction_one

omit [DecidableEq n] in
/-- The projection `1 − U Uᴴ` is a contraction when `U` has orthonormal columns:
`1 − (1 − U Uᴴ)ᴴ (1 − U Uᴴ) = U Uᴴ`. -/
lemma isContraction_one_sub_mul_conjTranspose {r : Type*} [Fintype r] [DecidableEq r]
    {U : Matrix m r ℂ}
    (hU : Uᴴ * U = 1) : IsContraction (1 - U * Uᴴ) := by
  have hUU : U * Uᴴ * (U * Uᴴ) = U * Uᴴ := by
    rw [Matrix.mul_assoc, ← Matrix.mul_assoc Uᴴ, hU, Matrix.one_mul]
  have h : 1 - (1 - U * Uᴴ)ᴴ * (1 - U * Uᴴ) = U * Uᴴ := by
    rw [conjTranspose_sub, conjTranspose_one, conjTranspose_mul, conjTranspose_conjTranspose]
    simp only [Matrix.sub_mul, Matrix.mul_sub, Matrix.one_mul, Matrix.mul_one, hUU]
    abel
  rw [IsContraction, h]
  exact posSemidef_self_mul_conjTranspose U

example := isContraction_one_sub_mul_conjTranspose (U := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (by simp)

/-- **Contractions do not increase the nuclear norm**: `‖P X Q‖_S1 ≤ ‖X‖_S1` for contractions
`P` and `Q`. -/
theorem nuclearNorm_mul_mul_le {m' n' : Type*} [Fintype m'] [Fintype n'] [DecidableEq m']
    [DecidableEq n']
    {P : Matrix m' m ℂ} {Q : Matrix n n' ℂ} (hP : IsContraction P) (hQ : IsContraction Q)
    (X : Matrix m n ℂ) : nuclearNorm (P * X * Q) ≤ nuclearNorm X := by
  obtain ⟨Z, hZ, htr⟩ := exists_isContraction_trace_eq (P * X * Q)
  have h : (Zᴴ * (P * X * Q)).trace = ((Pᴴ * Z * Qᴴ)ᴴ * X).trace := by
    simp only [conjTranspose_mul, conjTranspose_conjTranspose, Matrix.mul_assoc]
    rw [Matrix.trace_mul_comm Q]
    simp only [Matrix.mul_assoc]
  have h1 := norm_trace_le_nuclearNorm ((hP.conjTranspose.mul hZ).mul hQ.conjTranspose) X
  rwa [← h, htr, Complex.norm_real, Real.norm_of_nonneg (nuclearNorm_nonneg _)] at h1

example := nuclearNorm_mul_mul_le (isContraction_one (n := Fin 2)) isContraction_one
  (1 : Matrix (Fin 2) (Fin 2) ℂ)

omit [DecidableEq m] in
/-- `‖t X‖_S1 = |t| ‖X‖_S1` for real `t`. -/
lemma nuclearNorm_real_smul_eq_abs (t : ℝ) (X : Matrix m n ℂ) :
    nuclearNorm (t • X) = |t| * nuclearNorm X := by
  have h : t • X = (t : ℂ) • X := by
    ext i j
    simp [Complex.real_smul]
  rw [h, nuclearNorm_smul, Complex.norm_real, Real.norm_eq_abs]

omit [DecidableEq m] [DecidableEq n] in
/-- `X = ∑ Re X_ij e_ij + Im X_ij (i e_ij)`. -/
lemma eq_sum_re_smul_single_add [DecidableEq m] [DecidableEq n] (X : Matrix m n ℂ) :
    X = ∑ i, ∑ j, ((X i j).re • single i j (1 : ℂ) + (X i j).im • single i j Complex.I) := by
  conv_lhs => rw [matrix_eq_sum_single X]
  refine sum_congr rfl fun i _ => sum_congr rfl fun j _ => ?_
  rw [smul_single, smul_single, ← single_add, Complex.real_smul, Complex.real_smul, mul_one,
    Complex.re_add_im]

/-- **Real-linear maps are bounded in the nuclear norm**: for every real-linear map `g` between
matrix spaces, some `C` has `‖g X‖_S1 ≤ C ‖X‖_S1` for every `X`. -/
theorem exists_nuclearNorm_map_le {m' n' : Type*} [Fintype m'] [Fintype n'] [DecidableEq n']
    (g : Matrix m n ℂ →ₗ[ℝ] Matrix m' n' ℂ) :
    ∃ C : ℝ, ∀ X, nuclearNorm (g X) ≤ C * nuclearNorm X := by
  refine ⟨∑ i, ∑ j, (nuclearNorm (g (single i j 1)) + nuclearNorm (g (single i j Complex.I))),
    fun X => ?_⟩
  calc nuclearNorm (g X)
      = nuclearNorm (∑ i, ∑ j, ((X i j).re • g (single i j 1) +
          (X i j).im • g (single i j Complex.I))) := by
        conv_lhs => rw [eq_sum_re_smul_single_add X]
        simp only [map_sum, map_add, map_smul]
    _ ≤ ∑ i, ∑ j, nuclearNorm ((X i j).re • g (single i j 1) +
          (X i j).im • g (single i j Complex.I)) :=
        (le_sum_of_subadditive nuclearNorm nuclearNorm_zero.le nuclearNorm_add_le _ _).trans
          (sum_le_sum fun i _ =>
            le_sum_of_subadditive nuclearNorm nuclearNorm_zero.le nuclearNorm_add_le _ _)
    _ ≤ ∑ i, ∑ j, nuclearNorm X *
          (nuclearNorm (g (single i j 1)) + nuclearNorm (g (single i j Complex.I))) := by
        refine sum_le_sum fun i _ => sum_le_sum fun j _ => (nuclearNorm_add_le _ _).trans ?_
        rw [nuclearNorm_real_smul_eq_abs, nuclearNorm_real_smul_eq_abs, mul_add]
        exact add_le_add
          (mul_le_mul_of_nonneg_right ((Complex.abs_re_le_norm _).trans
            (norm_apply_le_nuclearNorm X i j)) (nuclearNorm_nonneg _))
          (mul_le_mul_of_nonneg_right ((Complex.abs_im_le_norm _).trans
            (norm_apply_le_nuclearNorm X i j)) (nuclearNorm_nonneg _))
    _ = _ := by
        rw [sum_mul]
        refine sum_congr rfl fun i _ => ?_
        rw [sum_mul]
        exact sum_congr rfl fun j _ => mul_comm _ _

example := exists_nuclearNorm_map_le (LinearMap.id : Matrix (Fin 2) (Fin 2) ℂ →ₗ[ℝ] _)

end Transformer.Grokking.FlatSpectrum
