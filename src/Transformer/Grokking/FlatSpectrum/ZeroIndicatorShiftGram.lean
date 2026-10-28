module

public import Transformer.Grokking.FlatSpectrum.ZeroIndicatorShift

/-!
# The shifted indicator of answer `0`: the Gram matrix in blocks

Source: `review/flat_spectrum.md`, Torch check: "31.5 at p=97, 60.0 at p=199 | These are the
centered values". Let `s = (p − 1)/2 = ∑_b u_b`, `v = 𝟙 − u` and `U = [u v]`, so
`Uᴴ U = diag(s, (p + 1)/2)`. The matrix `E = diag u − u uᵀ/s` is the projection onto the vectors
supported on `{u = 1}` with sum `0`: `Eᴴ = E`, `E² = E`, `E U = 0` and `tr E = s − 1`. Writing
`α = 1 − 2μ + p μ²` and `β = −2μ`, the Gram matrix
`K_μᴴ K_μ = α 𝟙𝟙ᵀ + β (u 𝟙ᵀ + 𝟙 uᵀ) + 2 diag u` is `2 E + U M Uᴴ` with
`M = [[α + 2β + 2/s, α + β], [α + β, α]]`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {p : ℕ} [Fact p.Prime]

variable (p) in
/-- `E = diag u − u uᵀ/s`, `s = (p − 1)/2`. -/
noncomputable def negSqProjection : Matrix (ZMod p) (ZMod p) ℂ :=
  diagonal (fun b => (negSqIndicator p b : ℂ)) - ((2 / ((p : ℝ) - 1) : ℝ) : ℂ) •
    vecMulVec (fun b => (negSqIndicator p b : ℂ)) fun b => (negSqIndicator p b : ℂ)

variable (p) in
/-- `U = [u v]`, `v = 𝟙 − u`. -/
noncomputable def negSqFrame : Matrix (ZMod p) (Fin 2) ℂ :=
  Matrix.of fun b i => ((![negSqIndicator p b, 1 - negSqIndicator p b] i : ℝ) : ℂ)

variable (p) in
/-- `M = [[α + 2β + 2/s, α + β], [α + β, α]]`, `α = 1 − 2μ + p μ²`, `β = −2μ`. -/
noncomputable def shiftedGramBlock (μ : ℝ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![((1 - 6 * μ + p * μ ^ 2 + 4 / (p - 1) : ℝ) : ℂ), ((1 - 4 * μ + p * μ ^ 2 : ℝ) : ℂ);
    ((1 - 4 * μ + p * μ ^ 2 : ℝ) : ℂ), ((1 - 2 * μ + p * μ ^ 2 : ℝ) : ℂ)]

/-- `u ⬝ u = s`. -/
lemma dotProduct_negSqIndicator (hp : p ≠ 2) :
    ((fun b => (negSqIndicator p b : ℂ)) ⬝ᵥ fun b => (negSqIndicator p b : ℂ)) =
      (((p : ℝ) - 1) / 2 : ℝ) := by
  simp only [dotProduct, ← Complex.ofReal_mul, negSqIndicator_mul_self, ← Complex.ofReal_sum,
    sum_negSqIndicator hp]

example := @dotProduct_negSqIndicator 5 ⟨Nat.prime_five⟩ (by decide)

/-- `E` is Hermitian. -/
lemma negSqProjection_conjTranspose : (negSqProjection p)ᴴ = negSqProjection p := by
  have hs : (star fun b => (negSqIndicator p b : ℂ)) = fun b => (negSqIndicator p b : ℂ) :=
    funext fun b => by simp
  rw [negSqProjection, conjTranspose_sub, diagonal_conjTranspose, conjTranspose_smul,
    conjTranspose_vecMulVec, hs]
  simp

/-- `E² = E`. -/
theorem negSqProjection_mul_self (hp : p ≠ 2) :
    negSqProjection p * negSqProjection p = negSqProjection p := by
  have hp1 : (p : ℝ) - 1 ≠ 0 := sub_ne_zero.mpr (by exact_mod_cast (Fact.out : p.Prime).ne_one)
  set u : ZMod p → ℂ := fun b => (negSqIndicator p b : ℂ) with hu
  have huu : u * u = u := funext fun b => by
    simp only [hu, Pi.mul_apply, ← Complex.ofReal_mul, negSqIndicator_mul_self]
  have hdd : diagonal u * diagonal u = diagonal u := by
    rw [diagonal_mul_diagonal]
    exact congrArg diagonal huu
  have hdA : diagonal u * vecMulVec u u = vecMulVec u u := by
    ext b b'
    rw [diagonal_mul, vecMulVec_apply, ← mul_assoc, ← Pi.mul_apply u u, huu]
  have hAd : vecMulVec u u * diagonal u = vecMulVec u u := by
    ext b b'
    rw [mul_diagonal, vecMulVec_apply, mul_assoc, ← Pi.mul_apply u u, huu]
  have hAA : ((2 / ((p : ℝ) - 1) : ℝ) : ℂ) • vecMulVec u u *
      ((2 / ((p : ℝ) - 1) : ℝ) : ℂ) • vecMulVec u u =
      ((2 / ((p : ℝ) - 1) : ℝ) : ℂ) • vecMulVec u u := by
    rw [Matrix.smul_mul, Matrix.mul_smul, vecMulVec_mul_vecMulVec, dotProduct_negSqIndicator hp,
      vecMulVec_smul, smul_smul, smul_smul, ← Complex.ofReal_mul, ← Complex.ofReal_mul,
      show 2 / ((p : ℝ) - 1) * (2 / ((p : ℝ) - 1)) * (((p : ℝ) - 1) / 2) = 2 / ((p : ℝ) - 1) by
        field_simp]
  rw [negSqProjection, ← hu, Matrix.sub_mul, Matrix.mul_sub, Matrix.mul_sub, hdd, hAA,
    Matrix.mul_smul, hdA, Matrix.smul_mul, hAd]
  abel

example := @negSqProjection_mul_self 5 ⟨Nat.prime_five⟩ (by decide)

/-- `Uᴴ U = diag(s, (p + 1)/2)`. -/
theorem negSqFrame_conjTranspose_mul (hp : p ≠ 2) :
    (negSqFrame p)ᴴ * negSqFrame p =
      diagonal fun i => ((![((p : ℝ) - 1) / 2, ((p : ℝ) + 1) / 2] i : ℝ) : ℂ) := by
  have h0 : ∑ b : ZMod p, negSqIndicator p b * negSqIndicator p b = ((p : ℝ) - 1) / 2 := by
    simp only [negSqIndicator_mul_self, sum_negSqIndicator hp]
  have h1 : ∑ b : ZMod p, negSqIndicator p b * (1 - negSqIndicator p b) = 0 := by
    simp [mul_sub, negSqIndicator_mul_self]
  have h1' : ∑ b : ZMod p, (1 - negSqIndicator p b) * negSqIndicator p b = 0 := by
    simp [sub_mul, negSqIndicator_mul_self]
  have h2 : ∑ b : ZMod p, (1 - negSqIndicator p b) * (1 - negSqIndicator p b) =
      ((p : ℝ) + 1) / 2 := by
    simp only [sub_mul, mul_sub, one_mul, mul_one, negSqIndicator_mul_self, sum_sub_distrib,
      sum_const, card_univ, ZMod.card, nsmul_eq_mul, sum_negSqIndicator hp]
    ring
  have key (i j : Fin 2) : ∑ x : ZMod p, ![negSqIndicator p x, 1 - negSqIndicator p x] i *
      ![negSqIndicator p x, 1 - negSqIndicator p x] j =
      if i = j then ![((p : ℝ) - 1) / 2, ((p : ℝ) + 1) / 2] i else 0 := by
    fin_cases i <;> fin_cases j
    · simpa using h0
    · simpa using h1
    · simpa using h1'
    · simpa using h2
  ext i j
  simp only [mul_apply, conjTranspose_apply, negSqFrame, of_apply, Complex.star_def,
    Complex.conj_ofReal, ← Complex.ofReal_mul, ← Complex.ofReal_sum, key, diagonal_apply]
  split_ifs <;> simp

example := @negSqFrame_conjTranspose_mul 5 ⟨Nat.prime_five⟩ (by decide)

/-- `E U = 0`. -/
theorem negSqProjection_mul_negSqFrame (hp : p ≠ 2) : negSqProjection p * negSqFrame p = 0 := by
  have hp1 : (p : ℝ) - 1 ≠ 0 := sub_ne_zero.mpr (by exact_mod_cast (Fact.out : p.Prime).ne_one)
  have hs (b : ZMod p) : negSqIndicator p b * negSqIndicator p b -
      2 / ((p : ℝ) - 1) * (negSqIndicator p b *
        ∑ x : ZMod p, negSqIndicator p x * negSqIndicator p x) = 0 := by
    simp only [negSqIndicator_mul_self, sum_negSqIndicator hp]
    field_simp
    ring
  have hv (b : ZMod p) : negSqIndicator p b * (1 - negSqIndicator p b) -
      2 / ((p : ℝ) - 1) * (negSqIndicator p b *
        ∑ x : ZMod p, negSqIndicator p x * (1 - negSqIndicator p x)) = 0 := by
    simp [mul_sub, negSqIndicator_mul_self]
  ext b i
  rw [negSqProjection, Matrix.sub_mul, Matrix.smul_mul, vecMulVec_mul, Matrix.sub_apply,
    Matrix.smul_apply, diagonal_mul, vecMulVec_apply, Matrix.zero_apply, smul_eq_mul]
  fin_cases i
  · simp only [negSqFrame, of_apply, vecMul, dotProduct, Fin.zero_eta, Fin.isValue,
      Matrix.cons_val_zero]
    exact_mod_cast hs b
  · simp only [negSqFrame, of_apply, vecMul, dotProduct, Fin.mk_one, Fin.isValue,
      Matrix.cons_val_one, Matrix.cons_val_fin_one]
    exact_mod_cast hv b

example := @negSqProjection_mul_negSqFrame 5 ⟨Nat.prime_five⟩ (by decide)

/-- `tr E = s − 1`. -/
theorem trace_negSqProjection (hp : p ≠ 2) :
    (negSqProjection p).trace = (((p : ℝ) - 3) / 2 : ℝ) := by
  have hp1 : (p : ℝ) - 1 ≠ 0 := sub_ne_zero.mpr (by exact_mod_cast (Fact.out : p.Prime).ne_one)
  rw [negSqProjection, trace_sub, trace_diagonal, trace_smul, trace_vecMulVec,
    dotProduct_negSqIndicator hp, ← Complex.ofReal_sum,
    sum_negSqIndicator hp, smul_eq_mul, ← Complex.ofReal_mul, ← Complex.ofReal_sub]
  congr 1
  field_simp
  ring

example := @trace_negSqProjection 5 ⟨Nat.prime_five⟩ (by decide)

/-- `K_μᴴ K_μ = 2 E + U M Uᴴ`. -/
theorem conjTranspose_mul_shiftedZeroIndicatorKernel (hp : p ≠ 2) (μ : ℝ) :
    (shiftedZeroIndicatorKernel p μ)ᴴ * shiftedZeroIndicatorKernel p μ =
      (2 : ℂ) • negSqProjection p + negSqFrame p * shiftedGramBlock p μ * (negSqFrame p)ᴴ := by
  ext b b'
  rw [conjTranspose_mul_shiftedZeroIndicatorKernel_apply hp]
  simp only [Matrix.add_apply, Matrix.smul_apply, negSqProjection, Matrix.sub_apply,
    diagonal_apply, vecMulVec_apply, smul_eq_mul, mul_apply, Fin.sum_univ_two, negSqFrame,
    shiftedGramBlock, conjTranspose_apply, of_apply, Complex.star_def, Complex.conj_ofReal,
    Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val', Matrix.empty_val',
    Matrix.cons_val_fin_one, Fin.isValue]
  split_ifs with h
  · subst h
    push_cast
    ring
  · push_cast
    ring

example := @conjTranspose_mul_shiftedZeroIndicatorKernel 5 ⟨Nat.prime_five⟩ (by decide) 0

end Transformer.Grokking.FlatSpectrum
