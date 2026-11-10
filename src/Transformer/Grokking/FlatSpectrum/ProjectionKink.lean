module

public import Transformer.Grokking.FlatSpectrum.ContractionDual
public import Transformer.Grokking.FlatSpectrum.RankOneFill

/-!
# A kink of the nuclear norm between two projections

Source: `review/flat_spectrum.md`, *The wrap of a step is a comparison* (added), item CW3.
Let `P` and `Q` be orthogonal projections with `P K = K = K Q`: `P` fixes the columns of `K`
and `Q` its rows. For contractions `Z`, `Z'`, the matrix
`P Z Q + (1 − P) Z' (1 − Q)` is a contraction (`IsContraction.proj_add_proj`):
`1 − Yᴴ Y = Q (1 − Zᴴ P Z) Q + (1 − Q) (1 − Z'ᴴ (1 − P) Z') (1 − Q)`. Pairing it with `K + x E`
gives the kink (`nuclearNorm_le_add_smul_of_proj`): if
`‖P E Q‖_S1 ≤ ‖(1 − P) E (1 − Q)‖_S1`, then `‖K‖_S1 ≤ ‖K + x E‖_S1` for every `x`. So `K` has the
least norm on the line through it along `E`, although the norm need not be flat there: the
part of `E` outside both ranges costs more than the part inside can save. For one cell,
`E = e_ij`, this reads `K` is a minimum of the matrices equal to it at the other cells
(`isMinOn_compl_singleton_of_proj`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- The complement of an orthogonal projection is one: `(1 − P)² = 1 − P`. -/
lemma one_sub_mul_one_sub {P : Matrix m m ℂ} (hPP : P * P = P) : (1 - P) * (1 - P) = 1 - P := by
  rw [Matrix.sub_mul, Matrix.one_mul, Matrix.mul_sub, Matrix.mul_one, hPP, sub_self, sub_zero]

example := one_sub_mul_one_sub (P := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (Matrix.one_mul 1)

/-- `1 − Zᴴ P Z ≥ 0` for a contraction `Z` and an orthogonal projection `P`: it is
`(1 − Zᴴ Z) + ((1 − P) Z)ᴴ ((1 − P) Z)`. -/
lemma IsContraction.posSemidef_one_sub {P : Matrix m m ℂ} (hP : Pᴴ = P)
    (hPP : P * P = P) {Z : Matrix m n ℂ} (hZ : IsContraction Z) :
    (1 - Zᴴ * P * Z).PosSemidef := by
  have h : 1 - Zᴴ * P * Z = (1 - Zᴴ * Z) + ((1 - P) * Z)ᴴ * ((1 - P) * Z) := by
    rw [conjTranspose_mul, conjTranspose_sub, conjTranspose_one, hP,
      show Zᴴ * (1 - P) * ((1 - P) * Z) = Zᴴ * ((1 - P) * (1 - P)) * Z by
        simp only [Matrix.mul_assoc], one_sub_mul_one_sub hPP, Matrix.mul_sub, Matrix.mul_one,
      Matrix.sub_mul]
    abel
  rw [h]
  exact PosSemidef.add hZ (posSemidef_conjTranspose_mul_self _)

example := (isContraction_one (n := Fin 2)).posSemidef_one_sub (P := 1) (by simp) (by simp)

/-- **Two blocks of contractions**: for orthogonal projections `P`, `Q` and contractions `Z`,
`Z'`, the matrix `P Z Q + (1 − P) Z' (1 − Q)` is a contraction. -/
theorem IsContraction.proj_add_proj {P : Matrix m m ℂ} {Q : Matrix n n ℂ} (hP : Pᴴ = P)
    (hPP : P * P = P) (hQ : Qᴴ = Q) (hQQ : Q * Q = Q) {Z Z' : Matrix m n ℂ}
    (hZ : IsContraction Z) (hZ' : IsContraction Z') :
    IsContraction (P * Z * Q + (1 - P) * Z' * (1 - Q)) := by
  have hP' : (1 - P)ᴴ = 1 - P := by rw [conjTranspose_sub, conjTranspose_one, hP]
  have hQ' : (1 - Q)ᴴ = 1 - Q := by rw [conjTranspose_sub, conjTranspose_one, hQ]
  have hPP' := one_sub_mul_one_sub hPP
  have hQQ' := one_sub_mul_one_sub hQQ
  have h01 : P * (1 - P) = 0 := by rw [Matrix.mul_sub, Matrix.mul_one, hPP, sub_self]
  have h10 : (1 - P) * P = 0 := by rw [Matrix.sub_mul, Matrix.one_mul, hPP, sub_self]
  have h : 1 - (P * Z * Q + (1 - P) * Z' * (1 - Q))ᴴ * (P * Z * Q + (1 - P) * Z' * (1 - Q)) =
      Qᴴ * (1 - Zᴴ * P * Z) * Q + (1 - Q)ᴴ * (1 - Z'ᴴ * (1 - P) * Z') * (1 - Q) := by
    have e1 : (P * Z * Q)ᴴ * (P * Z * Q) = Q * (Zᴴ * P * Z) * Q := by
      rw [conjTranspose_mul, conjTranspose_mul, hP, hQ,
        show Q * (Zᴴ * P) * (P * Z * Q) = Q * (Zᴴ * (P * P) * Z) * Q by
          simp only [Matrix.mul_assoc], hPP]
    have e2 : ((1 - P) * Z' * (1 - Q))ᴴ * ((1 - P) * Z' * (1 - Q)) =
        (1 - Q) * (Z'ᴴ * (1 - P) * Z') * (1 - Q) := by
      rw [conjTranspose_mul, conjTranspose_mul, hP', hQ',
        show (1 - Q) * (Z'ᴴ * (1 - P)) * ((1 - P) * Z' * (1 - Q)) =
          (1 - Q) * (Z'ᴴ * ((1 - P) * (1 - P)) * Z') * (1 - Q) by
          simp only [Matrix.mul_assoc], hPP']
    have e3 : (P * Z * Q)ᴴ * ((1 - P) * Z' * (1 - Q)) = 0 := by
      rw [conjTranspose_mul, conjTranspose_mul, hP, hQ,
        show Q * (Zᴴ * P) * ((1 - P) * Z' * (1 - Q)) =
          Q * Zᴴ * (P * (1 - P)) * Z' * (1 - Q) by simp only [Matrix.mul_assoc], h01]
      simp
    have e4 : ((1 - P) * Z' * (1 - Q))ᴴ * (P * Z * Q) = 0 := by
      rw [conjTranspose_mul, conjTranspose_mul, hP', hQ',
        show (1 - Q) * (Z'ᴴ * (1 - P)) * (P * Z * Q) =
          (1 - Q) * Z'ᴴ * ((1 - P) * P) * Z * Q by simp only [Matrix.mul_assoc], h10]
      simp
    rw [conjTranspose_add, Matrix.add_mul, Matrix.mul_add, Matrix.mul_add, e1, e2, e3, e4, hQ,
      hQ', Matrix.mul_sub Q, Matrix.mul_one, Matrix.sub_mul Q, hQQ, Matrix.mul_sub (1 - Q),
      Matrix.mul_one, Matrix.sub_mul (1 - Q), hQQ']
    abel
  rw [IsContraction, h]
  exact ((hZ.posSemidef_one_sub hP hPP).conjTranspose_mul_mul_same Q).add
    ((hZ'.posSemidef_one_sub hP' hPP').conjTranspose_mul_mul_same (1 - Q))

example := (isContraction_one (n := Fin 2)).proj_add_proj (P := 1) (Q := 1) (by simp) (by simp)
  (by simp) (by simp) isContraction_one

omit [DecidableEq m] [DecidableEq n] in
/-- `tr ((P Z Q)ᴴ X) = tr (Zᴴ (P X Q))` for Hermitian `P`, `Q`. -/
lemma trace_conjTranspose_proj_mul {P : Matrix m m ℂ} {Q : Matrix n n ℂ} (hP : Pᴴ = P)
    (hQ : Qᴴ = Q) (Z X : Matrix m n ℂ) :
    ((P * Z * Q)ᴴ * X).trace = (Zᴴ * (P * X * Q)).trace := by
  rw [conjTranspose_mul, conjTranspose_mul, hP, hQ, Matrix.mul_assoc, Matrix.trace_mul_comm]
  simp only [Matrix.mul_assoc]

example (Z X : Matrix (Fin 2) (Fin 2) ℂ) :=
  trace_conjTranspose_proj_mul (P := 1) (Q := 1) (by simp) (by simp) Z X

/-- **A kink of the nuclear norm**, `review/flat_spectrum.md`, item CW3: let `P`, `Q` be
orthogonal projections with `P K = K = K Q`, and let the part of `E` inside both ranges be no
longer than the part outside both, `‖P E Q‖_S1 ≤ ‖(1 − P) E (1 − Q)‖_S1`. Then
`‖K‖_S1 ≤ ‖K + x E‖_S1` for every `x`. -/
theorem nuclearNorm_le_add_smul_of_proj {K E : Matrix m n ℂ} {P : Matrix m m ℂ}
    {Q : Matrix n n ℂ} (hP : Pᴴ = P) (hPP : P * P = P) (hQ : Qᴴ = Q) (hQQ : Q * Q = Q)
    (hPK : P * K = K) (hKQ : K * Q = K)
    (hE : nuclearNorm (P * E * Q) ≤ nuclearNorm ((1 - P) * E * (1 - Q))) (x : ℂ) :
    nuclearNorm K ≤ nuclearNorm (K + x • E) := by
  obtain ⟨Z, hZ, hZK⟩ := exists_isContraction_trace_eq K
  obtain ⟨Z', hZ', hZE⟩ := exists_isContraction_trace_eq (x • ((1 - P) * E * (1 - Q)))
  have hP' : (1 - P)ᴴ = 1 - P := by rw [conjTranspose_sub, conjTranspose_one, hP]
  have hQ' : (1 - Q)ᴴ = 1 - Q := by rw [conjTranspose_sub, conjTranspose_one, hQ]
  have hY := hZ.proj_add_proj hP hPP hQ hQQ hZ'
  have h0 : (1 - P) * K = 0 := by rw [Matrix.sub_mul, Matrix.one_mul, hPK, sub_self]
  have h1 : P * (K + x • E) * Q = K + x • (P * E * Q) := by
    rw [Matrix.mul_add, Matrix.add_mul, hPK, hKQ, Matrix.mul_smul, Matrix.smul_mul]
  have h2 : (1 - P) * (K + x • E) * (1 - Q) = x • ((1 - P) * E * (1 - Q)) := by
    rw [Matrix.mul_add, h0, zero_add, Matrix.mul_smul, Matrix.smul_mul]
  have htr : ((P * Z * Q + (1 - P) * Z' * (1 - Q))ᴴ * (K + x • E)).trace =
      nuclearNorm K + x * (Zᴴ * (P * E * Q)).trace +
        ((‖x‖ * nuclearNorm ((1 - P) * E * (1 - Q)) : ℝ) : ℂ) := by
    rw [conjTranspose_add, Matrix.add_mul, trace_add, trace_conjTranspose_proj_mul hP hQ,
      trace_conjTranspose_proj_mul hP' hQ', h1, h2, hZE, nuclearNorm_smul, Matrix.mul_add,
      trace_add, hZK, Matrix.mul_smul, trace_smul, smul_eq_mul]
  have h3 := (Complex.re_le_norm _).trans (norm_trace_le_nuclearNorm hY (K + x • E))
  have h4 : ‖x * (Zᴴ * (P * E * Q)).trace‖ ≤ ‖x‖ * nuclearNorm ((1 - P) * E * (1 - Q)) := by
    rw [norm_mul]
    exact mul_le_mul_of_nonneg_left ((norm_trace_le_nuclearNorm hZ _).trans hE) (norm_nonneg x)
  have h5 := (abs_le.mp (Complex.abs_re_le_norm (x * (Zᴴ * (P * E * Q)).trace))).1
  rw [htr, Complex.add_re, Complex.add_re, Complex.ofReal_re, Complex.ofReal_re] at h3
  linarith

example (x : ℂ) := nuclearNorm_le_add_smul_of_proj (K := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (E := 0) (P := 1) (Q := 1) (by simp) (by simp) (by simp) (by simp) (by simp) (by simp)
  (by simp) x

/-- **One cell at a kink**, `review/flat_spectrum.md`, item CW3: under the hypotheses of
`nuclearNorm_le_add_smul_of_proj` for `E = e_ij`, the matrix `K` has the least nuclear norm
among the matrices equal to it at the other cells. -/
theorem isMinOn_compl_singleton_of_proj {K : Matrix m n ℂ} {P : Matrix m m ℂ}
    {Q : Matrix n n ℂ} (hP : Pᴴ = P) (hPP : P * P = P) (hQ : Qᴴ = Q) (hQQ : Q * Q = Q)
    (hPK : P * K = K) (hKQ : K * Q = K) {i : m} {j : n}
    (hE : nuclearNorm (P * (single i j 1 : Matrix m n ℂ) * Q) ≤
      nuclearNorm ((1 - P) * (single i j 1 : Matrix m n ℂ) * (1 - Q))) :
    IsMinOn nuclearNorm
      {X | ∀ kl ∈ ({(i, j)}ᶜ : Set (m × n)), X kl.1 kl.2 = K kl.1 kl.2} K := by
  rw [isMinOn_iff]
  intro X hX
  rw [eq_add_single_of_mem hX]
  exact nuclearNorm_le_add_smul_of_proj hP hPP hQ hQQ hPK hKQ hE _

example := isMinOn_compl_singleton_of_proj (K := (0 : Matrix (Fin 2) (Fin 2) ℂ)) (P := 0)
  (Q := 0) (i := 0) (j := 0) (by simp) (by simp) (by simp) (by simp) (by simp) (by simp)
  (by rw [Matrix.zero_mul, Matrix.zero_mul, nuclearNorm_zero]; exact nuclearNorm_nonneg _)

end Transformer.Grokking.FlatSpectrum
