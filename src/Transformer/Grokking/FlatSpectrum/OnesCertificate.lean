module

public import Transformer.Grokking.FlatSpectrum.Roots
public import Transformer.Grokking.FlatSpectrum.Unimodular

/-!
# A dual certificate for the all-ones matrix

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel* (added). Let `J` be
the all-ones `N × N` matrix, `Q = J / N` the projection onto the constant vectors and
`Π = 1 − Q`. Let `Y` vanish off `Ω`, have unit row and column sums, and let `c (Y − Q)` be a
contraction, `c > 0`. For every `H` vanishing on `Ω`,
`N + (1 − 1/c) ‖Π H Π‖_S1 ≤ ‖J + H‖_S1`: pair `J + H` with the contraction `Q + Π Z₀ Π`,
where `Z₀` attains `‖Π H Π‖_S1`, and use `tr (Yᴴ H) = 0`. For `c > 1`, a minimum of the
nuclear norm on `{X | X = J on Ω}` has `Π (X − J) Π = 0`, so `X − J` has entries `r_i + s_j`;
when such entries vanish on `Ω` only if they vanish everywhere, `J` is the unique minimum.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- The all-ones matrix `J`. -/
def onesMatrix (n : Type*) : Matrix n n ℂ := of fun _ _ => 1

/-- The projection `Q = J / N` onto the constant vectors. -/
noncomputable def meanProj (n : Type*) [Fintype n] : Matrix n n ℂ :=
  of fun _ _ => (Fintype.card n : ℂ)⁻¹

omit [DecidableEq n] in
lemma meanProj_mul_meanProj [Nonempty n] : meanProj n * meanProj n = meanProj n := by
  ext i j
  have h : (Fintype.card n : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  simp only [meanProj, mul_apply, of_apply, sum_const, card_univ, nsmul_eq_mul]
  field_simp

omit [DecidableEq n] in
lemma conjTranspose_meanProj : (meanProj n)ᴴ = meanProj n := by
  ext i j
  simp [meanProj]

omit [DecidableEq n] in
/-- `Q Y = Q` when every column of `Y` sums to one. -/
lemma meanProj_mul {Y : Matrix n n ℂ} (hY : ∀ j, ∑ i, Y i j = 1) :
    meanProj n * Y = meanProj n := by
  ext i j
  simp only [meanProj, mul_apply, of_apply, ← mul_sum, hY, mul_one]

omit [DecidableEq n] in
/-- `Y Q = Q` when every row of `Y` sums to one. -/
lemma mul_meanProj {Y : Matrix n n ℂ} (hY : ∀ i, ∑ j, Y i j = 1) :
    Y * meanProj n = meanProj n := by
  ext i j
  simp only [meanProj, mul_apply, of_apply, ← sum_mul, hY, one_mul]

omit [DecidableEq n] in
/-- Every row and every column of `Q` sums to one. -/
lemma sum_meanProj [Nonempty n] (i : n) : ∑ j, meanProj n i j = 1 := by
  have h : (Fintype.card n : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  simp only [meanProj, of_apply, sum_const, card_univ, nsmul_eq_mul]
  field_simp

example := meanProj_mul (Y := meanProj (Fin 2)) sum_meanProj

example := mul_meanProj (Y := meanProj (Fin 2)) sum_meanProj

omit [DecidableEq n] in
lemma meanProj_mul_onesMatrix [Nonempty n] : meanProj n * onesMatrix n = onesMatrix n := by
  ext i j
  have h : (Fintype.card n : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  simp only [meanProj, onesMatrix, mul_apply, of_apply, mul_one, sum_const, card_univ,
    nsmul_eq_mul]
  field_simp

omit [DecidableEq n] in
/-- The rows of `Q M` agree. -/
lemma meanProj_mul_apply (M : Matrix n n ℂ) (i i' j : n) :
    (meanProj n * M) i j = (meanProj n * M) i' j := by
  simp [meanProj, mul_apply]

omit [DecidableEq n] in
/-- The columns of `M Q` agree. -/
lemma mul_meanProj_apply (M : Matrix n n ℂ) (i j j' : n) :
    (M * meanProj n) i j = (M * meanProj n) i j' := by
  simp [meanProj, mul_apply]

/-- For a Hermitian projection `Q` and a contraction `Z₀`, `Q + (1 − Q) Z₀ (1 − Q)` is a
contraction: `1 − Zᴴ Z = (1 − Q) (1 − Z₀ᴴ Z₀ + (Q Z₀)ᴴ (Q Z₀)) (1 − Q)`. -/
lemma IsContraction.proj_add {Q Z₀ : Matrix n n ℂ} (hQ : Qᴴ = Q) (hQQ : Q * Q = Q)
    (hZ : IsContraction Z₀) : IsContraction (Q + (1 - Q) * Z₀ * (1 - Q)) := by
  have hQQ' (M : Matrix n n ℂ) : Q * (Q * M) = Q * M := by rw [← Matrix.mul_assoc, hQQ]
  have key : 1 - (Q + (1 - Q) * Z₀ * (1 - Q))ᴴ * (Q + (1 - Q) * Z₀ * (1 - Q)) =
      (1 - Q)ᴴ * ((1 - Z₀ᴴ * Z₀) + (Q * Z₀)ᴴ * (Q * Z₀)) * (1 - Q) := by
    simp only [conjTranspose_add, conjTranspose_mul, conjTranspose_sub, conjTranspose_one, hQ,
      Matrix.mul_add, Matrix.add_mul, Matrix.mul_sub, Matrix.sub_mul, Matrix.one_mul,
      Matrix.mul_one, Matrix.mul_assoc, hQQ, hQQ']
    abel
  rw [IsContraction, key]
  exact (hZ.add (posSemidef_conjTranspose_mul_self _)).conjTranspose_mul_mul_same _

example := IsContraction.proj_add (n := Fin 2) conjTranspose_meanProj meanProj_mul_meanProj
  isContraction_one

omit [Fintype n] in
/-- The zero matrix is a contraction. -/
lemma isContraction_zero {m : Type*} [Fintype m] : IsContraction (0 : Matrix m n ℂ) := by
  rw [IsContraction, conjTranspose_zero, Matrix.zero_mul, sub_zero]
  exact PosSemidef.one

/-- A matrix of nuclear norm zero is zero. -/
lemma eq_zero_of_nuclearNorm_eq_zero {m : Type*} [Fintype m] {A : Matrix m n ℂ}
    (h : nuclearNorm A = 0) : A = 0 := by
  have h0 : ∀ i ∈ univ, singularValues A i = 0 :=
    (sum_eq_zero_iff_of_nonneg fun i _ => Real.sqrt_nonneg _).mp h
  refine eq_zero_of_frobeniusSq_eq_zero ?_
  rw [← sum_singularValues_sq]
  exact sum_eq_zero fun i _ => by rw [h0 i (mem_univ i), sq, mul_zero]

example := eq_zero_of_nuclearNorm_eq_zero (A := (0 : Matrix (Fin 2) (Fin 2) ℂ)) nuclearNorm_zero

/-- `‖J‖_S1 = N`. -/
theorem nuclearNorm_onesMatrix : nuclearNorm (onesMatrix n) = Fintype.card n := by
  refine le_antisymm ?_ (card_le_nuclearNorm_le_of_unimodular (by simp [onesMatrix])).1
  have h := nuclearNorm_le_of_eq_mul (X := onesMatrix n) (of fun (_ : n) (_ : Unit) => (1 : ℂ))
    (of fun (_ : n) (_ : Unit) => (1 : ℂ)) (by ext i j; simp [onesMatrix, mul_apply])
  have hf : frobeniusSq (of fun (_ : n) (_ : Unit) => (1 : ℂ)) = Fintype.card n := by
    simp [frobeniusSq]
  rw [hf] at h
  linarith

end Transformer.Grokking.FlatSpectrum
