module

public import Transformer.Grokking.FlatSpectrum.GramBlock
public import Transformer.Grokking.FlatSpectrum.HallWitness
public import Transformer.Grokking.FlatSpectrum.WeightedPhases

/-!
# The exact nuclear norm of the block witness

Source: `review/flat_spectrum.md`, *Patterns*, for `X = J − t 1_R 1_Cᵀ` with `k = #R` and
`l = #C`: "By hand, not formalized: `‖X‖²_S1 = N² − 2t (k l − g) + t² k l` for `t ≥ 0`, with
`g = √(k l (N − k)(N − l))`."

Generalized to `m × n` and every real `t`. The columns `1_C` and `1_{Cᶜ}` are orthogonal, of
squared norms `l` and `n − l`, and `Xᴴ X = U M Uᴴ` for `U = [1_C, 1_{Cᶜ}]` and
`M = [[m − 2tk + t²k, m − tk], [m − tk, m]]` (`gram_onesSubBlock`), with `det M = t² k (m − k)`.
So `nuclearNorm_of_gram_eq_proj_add_diagonal` gives
`‖X‖²_S1 = m n − 2t k l + t² k l + 2|t| g` with `g = √(k l (m − k)(n − l))`
(`nuclearNorm_onesSubBlock`); for `C = ∅` and `C = univ`, `X` has rank one. At `t ≥ 0` and
`m = n = N` this is the source's formula (`nuclearNorm_onesSubBlock_sq_of_nonneg`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- `J − t 1_R 1_Cᵀ`: the all-ones matrix with `1 − t` on the block `R × C`. -/
noncomputable def onesSubBlock (R : Finset m) (C : Finset n) (t : ℝ) : Matrix m n ℂ :=
  of fun i j => if i ∈ R ∧ j ∈ C then 1 - (t : ℂ) else 1

/-- The orthogonal columns `[1_C, 1_{Cᶜ}]`. -/
def splitColumns (C : Finset n) : Matrix n (Fin 2) ℂ :=
  of fun j p => if j ∈ C then ![1, 0] p else ![0, 1] p

/-- `[1_C, 1_{Cᶜ}]ᴴ [1_C, 1_{Cᶜ}] = diag(#C, n − #C)`. -/
lemma splitColumns_gram (C : Finset n) :
    (splitColumns C)ᴴ * splitColumns C =
      diagonal fun p => ((![(#C : ℝ), Fintype.card n - #C] p : ℝ) : ℂ) := by
  have h (p q : Fin 2) (j : n) : star (splitColumns C j p) * splitColumns C j q =
      if j ∈ C then ![(1 : ℂ), 0] p * ![(1 : ℂ), 0] q else ![(0 : ℂ), 1] p * ![(0 : ℂ), 1] q := by
    fin_cases p <;> fin_cases q <;> by_cases hj : j ∈ C <;> simp [splitColumns, hj]
  ext p q
  simp only [mul_apply, conjTranspose_apply]
  rw [sum_congr rfl fun j _ => h p q j, sum_ite_mem_univ]
  fin_cases p <;> fin_cases q <;> simp [diagonal]

omit [Fintype n] in
/-- `(J − t 1_R 1_Cᵀ)ᴴ (J − t 1_R 1_Cᵀ) = U M Uᴴ` for `U = [1_C, 1_{Cᶜ}]` and
`M = [[m − 2tk + t²k, m − tk], [m − tk, m]]`, `k = #R`. -/
lemma gram_onesSubBlock (R : Finset m) (C : Finset n) (t : ℝ) :
    (onesSubBlock R C t)ᴴ * onesSubBlock R C t =
      splitColumns C * !![((Fintype.card m - 2 * t * #R + t ^ 2 * #R : ℝ) : ℂ),
        ((Fintype.card m - t * #R : ℝ) : ℂ); ((Fintype.card m - t * #R : ℝ) : ℂ),
        ((Fintype.card m : ℝ) : ℂ)] * (splitColumns C)ᴴ := by
  have h (j j' : n) (i : m) : star (onesSubBlock R C t i j) * onesSubBlock R C t i j' =
      if i ∈ R then (if j ∈ C then 1 - (t : ℂ) else 1) * (if j' ∈ C then 1 - (t : ℂ) else 1)
      else 1 := by
    by_cases hi : i ∈ R <;> by_cases hj : j ∈ C <;> by_cases hj' : j' ∈ C <;>
      simp [onesSubBlock, hi, hj, hj', Complex.conj_ofReal]
  ext j j'
  simp only [mul_apply, conjTranspose_apply]
  rw [sum_congr rfl fun i _ => h j j' i, sum_ite_mem_univ]
  by_cases hj : j ∈ C <;> by_cases hj' : j' ∈ C <;>
    simp [splitColumns, hj, hj', Fin.sum_univ_two] <;> ring

/-- **The exact nuclear norm of the block witness**, `review/flat_spectrum.md`, *Patterns*,
generalized to `m × n` and every real `t`: for `k = #R` and `l = #C`,
`‖J − t 1_R 1_Cᵀ‖_S1 = √(m n − 2t k l + t² k l + 2|t| √(k l (m − k)(n − l)))`. -/
theorem nuclearNorm_onesSubBlock (R : Finset m) (C : Finset n) (t : ℝ) :
    nuclearNorm (onesSubBlock R C t) =
      √(Fintype.card m * Fintype.card n - 2 * t * #R * #C + t ^ 2 * #R * #C +
        2 * |t| * √(#R * #C * (Fintype.card m - #R) * (Fintype.card n - #C))) := by
  have hk : (#R : ℝ) ≤ Fintype.card m := by exact_mod_cast card_le_univ R
  have hl : (#C : ℝ) ≤ Fintype.card n := by exact_mod_cast card_le_univ C
  have hk0 : (0 : ℝ) ≤ #R := Nat.cast_nonneg _
  rcases eq_empty_or_nonempty C with rfl | hC
  · have hX : onesSubBlock R (∅ : Finset n) t = vecMulVec (fun _ => 1) (fun _ => 1) := by
      ext i j
      simp [onesSubBlock, vecMulVec]
    rw [hX, nuclearNorm_vecMulVec, ← Real.sqrt_mul (by simp)]
    simp
  rcases eq_or_ne C univ with rfl | hCu
  · have hX : onesSubBlock R (univ : Finset n) t =
        vecMulVec (fun i => if i ∈ R then ((1 - t : ℝ) : ℂ) else 1) (fun _ => 1) := by
      ext i j
      by_cases hi : i ∈ R <;> simp [onesSubBlock, vecMulVec, hi]
    have hs : ∑ i, ‖(if i ∈ R then ((1 - t : ℝ) : ℂ) else 1)‖ ^ 2 =
        #R * (1 - t) ^ 2 + (Fintype.card m - #R) * 1 := by
      rw [← sum_ite_mem_univ]
      refine sum_congr rfl fun i _ => ?_
      by_cases hi : i ∈ R <;>
        simp only [hi, ite_true, ite_false, Complex.norm_real, Real.norm_eq_abs, sq_abs, norm_one,
          one_pow]
    rw [hX, nuclearNorm_vecMulVec, hs, ← Real.sqrt_mul (by nlinarith [sq_nonneg (1 - t)])]
    simp only [card_univ, sub_self, mul_zero, Real.sqrt_zero, add_zero, norm_one, one_pow,
      sum_const, nsmul_eq_mul, mul_one]
    congr 1
    ring
  have hl0 : (0 : ℝ) < #C := by exact_mod_cast hC.card_pos
  have hl1 : (#C : ℝ) < Fintype.card n := by
    exact_mod_cast lt_of_le_of_ne (card_le_univ C) (mt (card_eq_iff_eq_univ C).mp hCu)
  have hd : ∀ p, 0 < ![(#C : ℝ), Fintype.card n - #C] p := by
    intro p
    fin_cases p
    · exact hl0
    · simpa using hl1
  rw [nuclearNorm_of_gram_eq_proj_add_diagonal (E := 0) (c := 0) (M := !![
      ((Fintype.card m - 2 * t * #R + t ^ 2 * #R : ℝ) : ℂ), ((Fintype.card m - t * #R : ℝ) : ℂ);
      ((Fintype.card m - t * #R : ℝ) : ℂ), ((Fintype.card m : ℝ) : ℂ)]) le_rfl hd (by simp)
    (by simp) (splitColumns_gram C) (by simp) (by rw [gram_onesSubBlock]; simp)]
  simp only [Matrix.trace_fin_two, Matrix.det_fin_two_of, mul_diagonal, of_apply, cons_val',
    cons_val_zero, cons_val_one, empty_val', cons_val_fin_one, Matrix.trace_zero,
    Complex.zero_re, mul_zero, zero_add, ← Complex.ofReal_mul, ← Complex.ofReal_add,
    ← Complex.ofReal_sub, Complex.ofReal_re, Complex.norm_real, Real.norm_eq_abs]
  have hdet : (Fintype.card m - 2 * t * #R + t ^ 2 * #R : ℝ) * Fintype.card m -
      (Fintype.card m - t * #R) * (Fintype.card m - t * #R) =
        t ^ 2 * (#R * (Fintype.card m - #R)) := by
    ring
  have hg : √(|t ^ 2 * (#R * (Fintype.card m - #R))| * (#C * (Fintype.card n - #C))) =
      |t| * √(#R * #C * (Fintype.card m - #R) * (Fintype.card n - #C)) := by
    rw [abs_of_nonneg (by have := sub_nonneg.mpr hk; positivity), ← Real.sqrt_sq_eq_abs,
      ← Real.sqrt_mul (sq_nonneg t)]
    congr 1
    ring
  rw [hdet, hg]
  congr 1
  ring

example := nuclearNorm_onesSubBlock ({0} : Finset (Fin 2)) ({1} : Finset (Fin 2)) 1

/-- **The source's form**, `review/flat_spectrum.md`, *Patterns*: for `t ≥ 0`,
`‖J − t 1_R 1_Cᵀ‖²_S1 = m n − 2t (k l − g) + t² k l` with `g = √(k l (m − k)(n − l))`; for
`m = n = N`, `N² − 2t (k l − g) + t² k l`. -/
theorem nuclearNorm_onesSubBlock_sq_of_nonneg (R : Finset m) (C : Finset n) {t : ℝ}
    (ht : 0 ≤ t) :
    nuclearNorm (onesSubBlock R C t) ^ 2 = Fintype.card m * Fintype.card n -
      2 * t * (#R * #C - √(#R * #C * (Fintype.card m - #R) * (Fintype.card n - #C))) +
        t ^ 2 * #R * #C := by
  have hk : (#R : ℝ) ≤ Fintype.card m := by exact_mod_cast card_le_univ R
  have hl : (#C : ℝ) ≤ Fintype.card n := by exact_mod_cast card_le_univ C
  have hk0 : (0 : ℝ) ≤ #R := Nat.cast_nonneg _
  have hl0 : (0 : ℝ) ≤ #C := Nat.cast_nonneg _
  rw [nuclearNorm_onesSubBlock, abs_of_nonneg ht, Real.sq_sqrt]
  · ring
  have hg := Real.sqrt_nonneg (#R * #C * (Fintype.card m - #R) * (Fintype.card n - #C))
  nlinarith [mul_nonneg hk0 hl0, mul_le_mul hk hl hl0 (hk0.trans hk), sq_nonneg (1 - t),
    mul_nonneg (mul_nonneg hk0 hl0) (sq_nonneg (1 - t)), mul_nonneg ht hg]

example := nuclearNorm_onesSubBlock_sq_of_nonneg ({0} : Finset (Fin 2)) ({1} : Finset (Fin 2))
  zero_le_one

end Transformer.Grokking.FlatSpectrum
