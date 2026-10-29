module

public import Transformer.Grokking.FlatSpectrum.OnesCertificate
public import Transformer.Grokking.FlatSpectrum.WeightedPhases

/-!
# Contractions as the dual ball of the nuclear norm

Source: `review/flat_spectrum.md`, *Rank above one* (added), item 5, the lemmas of its proof.
`Y` is a contraction exactly when `‖Y v‖ ≤ ‖v‖` for every vector `v`
(`isContraction_iff_sum_norm_sq_le`), as `vᴴ (1 − Yᴴ Y) v = ‖v‖² − ‖Y v‖²`. Then so is `Yᴴ`
(`IsContraction.conjTranspose`): `‖Yᴴ x‖² = xᴴ Y (Yᴴ x) ≤ ‖x‖ ‖Y Yᴴ x‖ ≤ ‖x‖ ‖Yᴴ x‖`. If
`Re tr(Yᴴ X) ≤ ‖X‖_S1` for every `X`, then `Y` is a contraction
(`isContraction_of_re_trace_le`): at `X = (Y v) vᴴ` the pairing is `‖Y v‖²` and the norm is
`‖Y v‖ ‖v‖`. Every real-linear functional `g` on complex matrices is `Re tr(Yᴴ ·)`, with
`Y i j = g(e_ij) + g(i e_ij) i` (`re_trace_rieszMatrix`). The nuclear norm is subadditive
(`nuclearNorm_add_le`): a contraction attaining `‖X + X'‖_S1` pairs with `X` and `X'` to at most
their norms.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

omit [DecidableEq m] in
/-- `vᴴ (1 − Yᴴ Y) v = ‖v‖² − ‖Y v‖²`. -/
lemma star_dotProduct_one_sub_mulVec (Y : Matrix m n ℂ) (v : n → ℂ) :
    star v ⬝ᵥ ((1 - Yᴴ * Y) *ᵥ v) =
      ((∑ j, ‖v j‖ ^ 2 - ∑ i, ‖(Y *ᵥ v) i‖ ^ 2 : ℝ) : ℂ) := by
  rw [sub_mulVec, one_mulVec, ← mulVec_mulVec, dotProduct_sub, dotProduct_mulVec (star v) Yᴴ,
    ← star_mulVec, star_dotProduct_self_eq_sum_norm_sq, star_dotProduct_self_eq_sum_norm_sq,
    Complex.ofReal_sub]

omit [DecidableEq m] in
/-- **Contractions shorten vectors**: `Y` is a contraction exactly when `‖Y v‖ ≤ ‖v‖` for every
`v`. -/
theorem isContraction_iff_sum_norm_sq_le {Y : Matrix m n ℂ} :
    IsContraction Y ↔ ∀ v : n → ℂ, ∑ i, ‖(Y *ᵥ v) i‖ ^ 2 ≤ ∑ j, ‖v j‖ ^ 2 := by
  rw [IsContraction, posSemidef_iff_dotProduct_mulVec]
  simp only [star_dotProduct_one_sub_mulVec, Complex.zero_le_real, sub_nonneg]
  exact ⟨fun h => h.2, fun h => ⟨isHermitian_one.sub (isHermitian_conjTranspose_mul_self Y), h⟩⟩

/-- `a ≤ √a √b` with `a, b ≥ 0` gives `a ≤ b`. -/
lemma le_of_le_sqrt_mul_sqrt {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (h : a ≤ √a * √b) : a ≤ b := by
  nlinarith [Real.sq_sqrt ha, Real.sq_sqrt hb, sq_nonneg (√a - √b)]

example := le_of_le_sqrt_mul_sqrt (a := 1) (b := 1) zero_le_one zero_le_one (by simp)

/-- **The adjoint of a contraction is a contraction**: with `w = Yᴴ x`,
`‖w‖² = xᴴ (Y w) ≤ ‖x‖ ‖Y w‖ ≤ ‖x‖ ‖w‖`. -/
theorem IsContraction.conjTranspose {Y : Matrix m n ℂ} (hY : IsContraction Y) :
    IsContraction Yᴴ := by
  rw [isContraction_iff_sum_norm_sq_le] at hY ⊢
  intro x
  have h1 := Real.sqrt_le_sqrt (hY (Yᴴ *ᵥ x))
  have h2 : ((∑ j, ‖(Yᴴ *ᵥ x) j‖ ^ 2 : ℝ) : ℂ) = ∑ i, star (x i) * (Y *ᵥ (Yᴴ *ᵥ x)) i := by
    rw [← star_dotProduct_self_eq_sum_norm_sq]
    change _ = star x ⬝ᵥ (Y *ᵥ (Yᴴ *ᵥ x))
    rw [dotProduct_mulVec (star x) Y, star_mulVec, conjTranspose_conjTranspose]
  have h3 := norm_sum_star_mul_le univ x (Y *ᵥ (Yᴴ *ᵥ x))
  rw [← h2, Complex.norm_real, Real.norm_of_nonneg (sum_nonneg fun _ _ => sq_nonneg _)] at h3
  refine le_of_le_sqrt_mul_sqrt (sum_nonneg fun _ _ => sq_nonneg _)
    (sum_nonneg fun _ _ => sq_nonneg _) ?_
  rw [mul_comm]
  exact h3.trans (mul_le_mul_of_nonneg_left h1 (Real.sqrt_nonneg _))

example := (isContraction_one (n := Fin 2)).conjTranspose

omit [DecidableEq m] in
/-- **A pairing below the nuclear norm comes from a contraction**: if `Re tr(Yᴴ X) ≤ ‖X‖_S1`
for every `X`, then `Y` is a contraction; at `X = (Y v) vᴴ` the pairing is `‖Y v‖²` and the
norm `‖Y v‖ ‖v‖`. -/
theorem isContraction_of_re_trace_le {Y : Matrix m n ℂ}
    (h : ∀ X : Matrix m n ℂ, (Yᴴ * X).trace.re ≤ nuclearNorm X) : IsContraction Y := by
  rw [isContraction_iff_sum_norm_sq_le]
  intro v
  have h1 := h (vecMulVec (Y *ᵥ v) (star v))
  have htr : (Yᴴ * vecMulVec (Y *ᵥ v) (star v)).trace =
      ((∑ i, ‖(Y *ᵥ v) i‖ ^ 2 : ℝ) : ℂ) := by
    rw [mul_vecMulVec, trace_vecMulVec, dotProduct_comm,
      ← star_dotProduct_self_eq_sum_norm_sq (Y *ᵥ v), dotProduct_mulVec (star v) Yᴴ,
      ← star_mulVec]
  rw [htr, Complex.ofReal_re, nuclearNorm_vecMulVec] at h1
  simp only [Pi.star_apply, norm_star] at h1
  exact le_of_le_sqrt_mul_sqrt (sum_nonneg fun _ _ => sq_nonneg _)
    (sum_nonneg fun _ _ => sq_nonneg _) h1

example := isContraction_of_re_trace_le (Y := (0 : Matrix (Fin 2) (Fin 2) ℂ)) fun X => by
  simpa using nuclearNorm_nonneg X

/-- The matrix of a real-linear functional `g` on complex matrices:
`Y i j = g(e_ij) + g(i e_ij) i`. -/
noncomputable def rieszMatrix (g : Matrix m n ℂ →ₗ[ℝ] ℝ) :
    Matrix m n ℂ :=
  of fun i j => ⟨g (single i j 1), g (single i j Complex.I)⟩

/-- **Every real-linear functional on complex matrices is a trace pairing**:
`g X = Re tr(Yᴴ X)` for `Y = rieszMatrix g`. -/
theorem re_trace_rieszMatrix (g : Matrix m n ℂ →ₗ[ℝ] ℝ) (X : Matrix m n ℂ) :
    ((rieszMatrix g)ᴴ * X).trace.re = g X := by
  rw [trace_conjTranspose_mul, Complex.re_sum]
  conv_rhs => rw [matrix_eq_sum_single X, map_sum]
  refine sum_congr rfl fun i _ => ?_
  rw [Complex.re_sum, map_sum]
  refine sum_congr rfl fun j _ => ?_
  have hs : single i j (X i j) =
      (X i j).re • single i j (1 : ℂ) + (X i j).im • single i j Complex.I := by
    rw [smul_single, smul_single, ← single_add, Complex.real_smul, Complex.real_smul, mul_one,
      Complex.re_add_im]
  rw [hs, map_add, map_smul, map_smul, smul_eq_mul, smul_eq_mul]
  simp only [rieszMatrix, of_apply, Complex.mul_re, Complex.star_def, Complex.conj_re,
    Complex.conj_im]
  ring

omit [DecidableEq m] in
/-- **The triangle inequality for the nuclear norm**: `‖X + X'‖_S1 ≤ ‖X‖_S1 + ‖X'‖_S1`. -/
theorem nuclearNorm_add_le (X X' : Matrix m n ℂ) :
    nuclearNorm (X + X') ≤ nuclearNorm X + nuclearNorm X' := by
  obtain ⟨Z, hZ, htr⟩ := exists_isContraction_trace_eq (X + X')
  have h := congrArg Complex.re htr
  rw [Matrix.mul_add, trace_add, Complex.add_re, Complex.ofReal_re] at h
  have h1 := (Complex.re_le_norm _).trans (norm_trace_le_nuclearNorm hZ X)
  have h2 := (Complex.re_le_norm _).trans (norm_trace_le_nuclearNorm hZ X')
  linarith

omit [DecidableEq m] in
/-- `‖t X‖_S1 = t ‖X‖_S1` for real `t ≥ 0`. -/
theorem nuclearNorm_real_smul {t : ℝ} (ht : 0 ≤ t) (X : Matrix m n ℂ) :
    nuclearNorm (t • X) = t * nuclearNorm X := by
  have h : t • X = (t : ℂ) • X := by
    ext i j
    simp [Complex.real_smul]
  rw [h, nuclearNorm_smul, Complex.norm_real, Real.norm_of_nonneg ht]

example := nuclearNorm_real_smul (t := 2) zero_le_two (1 : Matrix (Fin 2) (Fin 2) ℂ)

end Transformer.Grokking.FlatSpectrum
