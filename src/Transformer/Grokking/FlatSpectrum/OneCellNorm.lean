module

public import Transformer.Grokking.FlatSpectrum.OneCell

/-!
# A flat kernel with one changed cell: its nuclear norm

Source: `review/flat_spectrum.md`, *One unseen cell* (added), item OC3.

Let `K Kᴴ = c 1`, `κ = K_{ij}` and `g = ∑_{p ≠ i} |K_{pj}|² > 0`. For `X = K + t e_i e_jᵀ`, the
Gram matrix of OC2 and the block formula of OC1 give, with `τ = |κ + t|² − |κ|²`,

    ‖X‖_S1 = √c (m − 2) + √(2c + τ + 2 √|c (c + τ) − g |t|²|)
                                                    (`nuclearNorm_add_smul_single`).

The column `j` of `K` has squared length at most `c`, as `K / √c` is a contraction, so
`|κ|² + g ≤ c` (`norm_sq_add_restSq_le`). For the Hadamard matrix of size two and the cell
`(0, 0)`: `c = 2`, `κ = 1`, `g = 1`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

omit [Fintype n] [DecidableEq n] in
/-- The squared length of a part of a column is not negative. -/
lemma restSq_nonneg (K : Matrix m n ℂ) (i : m) (j : n) : 0 ≤ restSq K i j :=
  Finset.sum_nonneg fun p _ => by split_ifs <;> positivity

/-- The corner of the block: `t κ̄ + t̄ κ + t t̄ = |κ + t|² − |κ|²`. -/
lemma corner_eq (κ t : ℂ) :
    t * star κ + star t * κ + t * star t = ((‖κ + t‖ ^ 2 - ‖κ‖ ^ 2 : ℝ) : ℂ) := by
  rw [Complex.ofReal_sub, Complex.ofReal_pow, Complex.ofReal_pow, ← Complex.mul_conj',
    ← Complex.mul_conj']
  simp only [map_add, Complex.star_def]
  ring

/-- **The nuclear norm of a flat kernel with one changed cell** (`review/flat_spectrum.md`,
OC3): `‖K + t e_i e_jᵀ‖_S1 = √c (m − 2) + √(2c + τ + 2 √|c (c + τ) − g |t|²|)` with
`τ = |κ + t|² − |κ|²`, for `K Kᴴ = c 1` and `g > 0`. -/
theorem nuclearNorm_add_smul_single {K : Matrix m n ℂ} {c : ℝ} (hc : 0 ≤ c)
    (hK : K * Kᴴ = (c : ℂ) • 1) {i : m} {j : n} (hg : 0 < restSq K i j) (t : ℂ) :
    nuclearNorm (K + t • single i j (1 : ℂ)) = √c * (Fintype.card m - 2) +
      √(2 * c + (‖K i j + t‖ ^ 2 - ‖K i j‖ ^ 2) +
        2 * √|c * (c + (‖K i j + t‖ ^ 2 - ‖K i j‖ ^ 2)) - restSq K i j * ‖t‖ ^ 2|) := by
  set τ : ℝ := ‖K i j + t‖ ^ 2 - ‖K i j‖ ^ 2 with hτ
  set g : ℝ := restSq K i j with hgdef
  have hX : ((K + t • single i j (1 : ℂ))ᴴ)ᴴ * (K + t • single i j (1 : ℂ))ᴴ = (c : ℂ) • 1 +
      cellFrame K i j * !![(τ : ℂ), t; star t, 0] * (cellFrame K i j)ᴴ := by
    rw [conjTranspose_conjTranspose, mul_conjTranspose_add_smul_single hK i j t, corner_eq]
  have h := nuclearNorm_of_gram_eq_smul_one_add hc (d := ![1, g])
    (fun α => by fin_cases α <;> simp [hg]) (cellFrame_gram K i j) hX
  rw [← nuclearNorm_conjTranspose, h]
  have hgne : (g : ℂ) ≠ 0 := Complex.ofReal_ne_zero.2 hg.ne'
  have htr : ((!![(τ : ℂ), t; star t, 0] + (c : ℂ) • diagonal fun α => ((![1, g] α : ℝ) : ℂ)⁻¹) *
      diagonal fun α => ((![1, g] α : ℝ) : ℂ)).trace = ((2 * c + τ : ℝ) : ℂ) := by
    simp [Matrix.trace_fin_two, Matrix.mul_apply, diagonal_apply]
    field_simp
    ring
  have hdet : (!![(τ : ℂ), t; star t, 0] +
      (c : ℂ) • diagonal fun α => ((![1, g] α : ℝ) : ℂ)⁻¹).det =
        (((c * (c + τ) - g * ‖t‖ ^ 2) / g : ℝ) : ℂ) := by
    simp [Matrix.det_fin_two]
    rw [← Complex.mul_conj']
    field_simp
    ring
  rw [htr, hdet, Complex.ofReal_re, Complex.norm_real, Real.norm_eq_abs, abs_div, abs_of_pos hg]
  congr 4
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, one_mul]
  field_simp


omit [Fintype n] [DecidableEq n] in
/-- A column is its cell and the rest. -/
lemma norm_sq_add_restSq (K : Matrix m n ℂ) (i : m) (j : n) :
    ‖K i j‖ ^ 2 + restSq K i j = ∑ p, ‖K p j‖ ^ 2 := by
  have h1 : ‖K i j‖ ^ 2 = ∑ p, if i = p then ‖K p j‖ ^ 2 else 0 := by
    rw [Finset.sum_ite_eq]
    simp
  rw [h1, restSq, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun p _ => by split_ifs <;> simp

/-- **A column of a kernel with orthogonal rows of squared length `c` has squared length at
most `c`** (`review/flat_spectrum.md`, OC3): `K / √c` is a contraction. -/
lemma sum_norm_sq_le_of_mul_conjTranspose {K : Matrix m n ℂ} {c : ℝ} (hc : 0 < c)
    (hK : K * Kᴴ = (c : ℂ) • 1) (j : n) : ∑ p, ‖K p j‖ ^ 2 ≤ c := by
  have h := (IsFlat.isContraction hc (Or.inr hK)).sum_norm_sq_le (U := 1) (by simp) j
  simp only [Matrix.mul_one, Matrix.smul_apply, smul_eq_mul, norm_mul, mul_pow, norm_inv,
    Complex.norm_real, Real.norm_of_nonneg (Real.sqrt_nonneg c), inv_pow, Real.sq_sqrt hc.le,
    ← Finset.mul_sum] at h
  rwa [inv_mul_le_iff₀ hc, mul_one] at h

/-- `|κ|² + g ≤ c` (`review/flat_spectrum.md`, OC3). -/
lemma norm_sq_add_restSq_le {K : Matrix m n ℂ} {c : ℝ} (hc : 0 < c)
    (hK : K * Kᴴ = (c : ℂ) • 1) (i : m) (j : n) : ‖K i j‖ ^ 2 + restSq K i j ≤ c :=
  (norm_sq_add_restSq K i j).le.trans (sum_norm_sq_le_of_mul_conjTranspose hc hK j)

/-- The Hadamard matrix of size two has orthogonal rows of squared length two. -/
lemma hadamard_mul_conjTranspose :
    (!![1, 1; 1, -1] : Matrix (Fin 2) (Fin 2) ℂ) * (!![1, 1; 1, -1] : Matrix (Fin 2) (Fin 2) ℂ)ᴴ =
      ((2 : ℝ) : ℂ) • 1 := by
  ext p q
  fin_cases p <;> fin_cases q <;> simp [Matrix.mul_apply, Fin.sum_univ_two] <;> norm_num

/-- Its column `0` outside the row `0` has squared length one. -/
lemma restSq_hadamard : restSq (!![1, 1; 1, -1] : Matrix (Fin 2) (Fin 2) ℂ) 0 0 = 1 := by
  simp [restSq, Fin.sum_univ_two]

/-- The Hadamard matrix with the cell `(0, 0)` changed by `t`. -/
example (t : ℂ) := nuclearNorm_add_smul_single zero_le_two hadamard_mul_conjTranspose
  (i := 0) (j := 0) (by rw [restSq_hadamard]; exact one_pos) t

example := norm_sq_add_restSq_le two_pos hadamard_mul_conjTranspose 0 0

example := sum_norm_sq_le_of_mul_conjTranspose two_pos hadamard_mul_conjTranspose 0

end Transformer.Grokking.FlatSpectrum
