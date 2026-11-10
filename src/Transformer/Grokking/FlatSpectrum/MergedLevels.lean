module

public import Transformer.Grokking.FlatSpectrum.ProjectionKink

/-!
# Merged levels: two equal rows and two equal columns

Source: `review/flat_spectrum.md`, *The wrap of a step is a comparison* (added), item CW4.
Let rows `i ≠ i'` of `K` be equal and columns `j ≠ j'` be equal. Then `K` has the least nuclear
norm among the matrices equal to it off the cell `(i, j)` (`isMinOn_of_row_eq_of_col_eq`). The
projections of `ProjectionKink` are `mergeProj i i' = 1 − ½ (e_i − e_i') (e_i − e_i')ᴴ`, onto
the vectors with equal coordinates `i` and `i'`, and the same for the columns. Both
`P e_ij Q` and `(1 − P) e_ij (1 − Q)` have nuclear norm `½`
(`nuclearNorm_mul_single_mul_of_proj`): a projection `A` has `∑_r |A_ri|² = A_ii`, and
`P_ii = Q_jj = ½`. So the kink condition of `nuclearNorm_le_add_smul_of_proj` holds with
equality. A learner that minimizes the nuclear norm therefore fills a cell whose row and
column can each be merged with another one by the value that merges them.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- The difference of two basis vectors, `e_i − e_i'`. -/
def diffVec (i i' : m) : m → ℂ := Pi.single i 1 - Pi.single i' 1

/-- The projection onto the vectors with equal coordinates `i` and `i'`,
`1 − ½ (e_i − e_i') (e_i − e_i')ᴴ`. -/
noncomputable def mergeProj (i i' : m) : Matrix m m ℂ :=
  1 - (2⁻¹ : ℂ) • vecMulVec (diffVec i i') (star (diffVec i i'))

omit [Fintype m] in
lemma star_diffVec (i i' : m) : star (diffVec i i') = diffVec i i' := by
  ext r
  simp only [diffVec, Pi.star_apply, Pi.sub_apply, star_sub, Pi.single_apply]
  split_ifs <;> simp

lemma diffVec_dotProduct_self {i i' : m} (h : i ≠ i') : diffVec i i' ⬝ᵥ diffVec i i' = 2 := by
  simp [diffVec, h, h.symm]
  norm_num

example : diffVec (0 : Fin 2) 1 ⬝ᵥ diffVec 0 1 = 2 := diffVec_dotProduct_self (by decide)

omit [Fintype m] in
lemma conjTranspose_mergeProj (i i' : m) : (mergeProj i i')ᴴ = mergeProj i i' := by
  simp [mergeProj, conjTranspose_vecMulVec, star_diffVec]

lemma mergeProj_mul_mergeProj {i i' : m} (h : i ≠ i') :
    mergeProj i i' * mergeProj i i' = mergeProj i i' := by
  rw [mergeProj, star_diffVec]
  set M := vecMulVec (diffVec i i') (diffVec i i')
  have hMM : M * M = (2 : ℂ) • M := by
    rw [vecMulVec_mul_vecMulVec, diffVec_dotProduct_self h, vecMulVec_smul]
  rw [Matrix.sub_mul, Matrix.mul_sub, Matrix.mul_sub, Matrix.one_mul, Matrix.mul_one,
    Matrix.one_mul, Matrix.smul_mul, Matrix.mul_smul, hMM, smul_smul, smul_smul]
  module

example := mergeProj_mul_mergeProj (i := (0 : Fin 2)) (i' := 1) (by decide)

omit [Fintype m] in
lemma mergeProj_apply_self {i i' : m} (h : i ≠ i') :
    mergeProj i i' i i = ((2⁻¹ : ℝ) : ℂ) := by
  simp [mergeProj, vecMulVec_apply, diffVec, h]
  norm_num

example := mergeProj_apply_self (i := (0 : Fin 2)) (i' := 1) (by decide)

omit [DecidableEq n] [Fintype n] in
/-- Equal rows `i` and `i'` lie in the range of the projection: `P K = K`. -/
lemma mergeProj_mul_of_row_eq {K : Matrix m n ℂ} {i i' : m} (hrow : ∀ c, K i c = K i' c) :
    mergeProj i i' * K = K := by
  have h0 : star (diffVec i i') ᵥ* K = 0 := by
    ext c
    simp [diffVec, sub_vecMul, single_vecMul, hrow c]
  have h1 : vecMulVec (diffVec i i') (0 : n → ℂ) = 0 := by
    ext r c
    simp [vecMulVec_apply]
  rw [mergeProj, Matrix.sub_mul, Matrix.one_mul, Matrix.smul_mul, vecMulVec_mul, h0, h1,
    smul_zero, sub_zero]

example := mergeProj_mul_of_row_eq (K := (of fun _ _ => 1 : Matrix (Fin 2) (Fin 2) ℂ)) (i := 0)
  (i' := 1) fun _ => rfl

omit [DecidableEq m] [Fintype m] in
/-- Equal columns `j` and `j'`: `K Q = K`. -/
lemma mul_mergeProj_of_col_eq {K : Matrix m n ℂ} {j j' : n} (hcol : ∀ r, K r j = K r j') :
    K * mergeProj j j' = K := by
  have h0 : K *ᵥ diffVec j j' = 0 := by
    ext r
    simp [diffVec, mulVec_sub, hcol r]
  have h1 : vecMulVec (0 : m → ℂ) (star (diffVec j j')) = 0 := by
    ext r c
    simp [vecMulVec_apply]
  rw [mergeProj, Matrix.mul_sub, Matrix.mul_one, Matrix.mul_smul, mul_vecMulVec, h0, h1,
    smul_zero, sub_zero]

example := mul_mergeProj_of_col_eq (K := (of fun _ _ => 1 : Matrix (Fin 2) (Fin 2) ℂ)) (j := 0)
  (j' := 1) fun _ => rfl

omit [DecidableEq m] in
/-- A projection has `∑_r |A_ri|² = A_ii`. -/
lemma sum_norm_sq_col_of_proj {A : Matrix m m ℂ} (hA : Aᴴ = A) (hAA : A * A = A) (i : m) :
    ((∑ r, ‖A r i‖ ^ 2 : ℝ) : ℂ) = A i i := by
  rw [← conjTranspose_mul_self_apply_self, hA, hAA]

example := sum_norm_sq_col_of_proj (A := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) (by simp) 0

omit [DecidableEq m] in
/-- A projection has `∑_c |A_ic|² = A_ii`. -/
lemma sum_norm_sq_row_of_proj {A : Matrix m m ℂ} (hA : Aᴴ = A) (hAA : A * A = A) (i : m) :
    ((∑ c, ‖A i c‖ ^ 2 : ℝ) : ℂ) = A i i := by
  have h : ∀ c, ‖A i c‖ = ‖A c i‖ := fun c => by
    conv_lhs => rw [← hA, conjTranspose_apply, norm_star]
  simp only [h]
  exact sum_norm_sq_col_of_proj hA hAA i

example := sum_norm_sq_row_of_proj (A := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) (by simp) 0

/-- For projections `A`, `B`, `‖A e_ij B‖_S1 = √(A_ii) √(B_jj)`. -/
lemma nuclearNorm_mul_single_mul_of_proj {A : Matrix m m ℂ} {B : Matrix n n ℂ} (hA : Aᴴ = A)
    (hAA : A * A = A) (hB : Bᴴ = B) (hBB : B * B = B) (i : m) (j : n) :
    nuclearNorm (A * (single i j 1 : Matrix m n ℂ) * B) = √(A i i).re * √(B j j).re := by
  have h : A * (single i j 1 : Matrix m n ℂ) * B = vecMulVec (A.col i) (B.row j) := by
    rw [single_eq_single_vecMulVec_single, mul_vecMulVec, vecMulVec_mul, mulVec_single_one,
      single_vecMul, one_smul]
  rw [h, nuclearNorm_vecMulVec]
  have h1 := congrArg Complex.re (sum_norm_sq_col_of_proj hA hAA i)
  have h2 := congrArg Complex.re (sum_norm_sq_row_of_proj hB hBB j)
  rw [Complex.ofReal_re] at h1 h2
  simp only [col_apply, row_apply, h1, h2]

example := nuclearNorm_mul_single_mul_of_proj (A := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (B := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) (by simp) (by simp) (by simp) 0 0

/-- **Merged levels**, `review/flat_spectrum.md`, item CW4: if rows `i ≠ i'` of `K` are equal
and columns `j ≠ j'` are equal, `K` has the least nuclear norm among the matrices equal to it
off the cell `(i, j)`. -/
theorem isMinOn_of_row_eq_of_col_eq {K : Matrix m n ℂ} {i i' : m} {j j' : n} (hi : i ≠ i')
    (hj : j ≠ j') (hrow : ∀ c, K i c = K i' c) (hcol : ∀ r, K r j = K r j') :
    IsMinOn nuclearNorm
      {X | ∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)), X kl.1 kl.2 = K kl.1 kl.2} K := by
  have hP := conjTranspose_mergeProj i i'
  have hPP := mergeProj_mul_mergeProj hi
  have hQ := conjTranspose_mergeProj j j'
  have hQQ := mergeProj_mul_mergeProj hj
  have hP' : (1 - mergeProj i i')ᴴ = 1 - mergeProj i i' := by
    rw [conjTranspose_sub, conjTranspose_one, hP]
  have hQ' : (1 - mergeProj j j')ᴴ = 1 - mergeProj j j' := by
    rw [conjTranspose_sub, conjTranspose_one, hQ]
  refine isMinOn_compl_singleton_of_proj hP hPP hQ hQQ (mergeProj_mul_of_row_eq hrow)
    (mul_mergeProj_of_col_eq hcol) (le_of_eq ?_)
  rw [nuclearNorm_mul_single_mul_of_proj hP hPP hQ hQQ,
    nuclearNorm_mul_single_mul_of_proj hP' (one_sub_mul_one_sub hPP) hQ'
      (one_sub_mul_one_sub hQQ),
    Matrix.sub_apply, Matrix.sub_apply, one_apply_eq, one_apply_eq, mergeProj_apply_self hi,
    mergeProj_apply_self hj, Complex.sub_re, Complex.one_re, Complex.ofReal_re]
  norm_num

/-- The table of all ones: with any one cell unseen, it is a minimum, as its rows and its
columns are equal. -/
example : IsMinOn nuclearNorm
    {X | ∀ kl ∈ ({(0, 0)}ᶜ : Set (Fin 2 × Fin 2)), X kl.1 kl.2 =
      (of fun _ _ => 1 : Matrix (Fin 2) (Fin 2) ℂ) kl.1 kl.2} (of fun _ _ => 1) :=
  isMinOn_of_row_eq_of_col_eq (i' := 1) (j' := 1) (by decide) (by decide) (fun _ => rfl)
    fun _ => rfl

end Transformer.Grokking.FlatSpectrum
