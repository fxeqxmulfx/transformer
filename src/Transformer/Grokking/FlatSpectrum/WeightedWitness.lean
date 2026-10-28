module

public import Transformer.Grokking.FlatSpectrum.HallWitness
public import Transformer.Grokking.FlatSpectrum.Transport
public import Transformer.Grokking.FlatSpectrum.WeightedPhases

/-!
# Without the weighted Hall condition the least nuclear norm misses `a bᵀ`

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*, Theorem G, item 3,
second half, which contains Theorem F's: the table `m × n` may be rectangular, and `a`, `b` may
have zero entries. Let `Ω` miss the block `R × C`, and write `A = ‖a‖²`, `B = ‖b‖²`,
`A_R = ∑_R |a_i|²` and `B_C = ∑_C |b_j|²`. For `p ≠ 0` the factors `U = p diag(a) U₀` and
`V = p⁻¹ diag(b̄) V₀`, with `U₀`, `V₀` of Theorem E.3, give `U Vᴴ = a bᵀ − s² a_R b_Cᵀ`
(`weightedWitness_apply`), which equals `a bᵀ` on `Ω`. Lemma 1 bounds its norm by
`(p² (A − s² A_R + s⁴ A_R) + p⁻² (B + s² (B − B_C))) / 2` (`nuclearNorm_weightedWitness_le`).
With `p² = √(AB) / A` both factors cost `√(AB) = ‖a bᵀ‖_S1`, and the bound is
`√(AB) − s² (δ − s² p² A_R) / 2` with `δ = p² A_R + p⁻² B_C − √(AB)`. If `A B < A_R B + A B_C`,
then `δ > 0`, and `s = δ / (p² A_R + δ)` beats `a bᵀ` (`exists_nuclearNorm_lt_of_weighted_block`).
The review's normalization `‖a‖ = ‖b‖` is `p = 1`; the real inequality is
`exists_weighted_bound_lt`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- The left factor `p a_i [1 − s² 1_R, s 1_R]` of the weighted block witness. -/
noncomputable def weightedFactorL (a : m → ℂ) (R : Finset m) (p s : ℝ) : Matrix m (Fin 2) ℂ :=
  of fun i k => (p : ℂ) * a i * blockFactorL R s i k

/-- The right factor `p⁻¹ b̄_j [1, s 1_{Cᶜ}]` of the weighted block witness. -/
noncomputable def weightedFactorR (b : n → ℂ) (C : Finset n) (p s : ℝ) : Matrix n (Fin 2) ℂ :=
  of fun j k => ((p⁻¹ : ℝ) : ℂ) * star (b j) * blockFactorR C s j k

omit [Fintype m] [Fintype n] in
/-- The weighted block witness is `a bᵀ − s² a_R b_Cᵀ`. -/
lemma weightedWitness_apply (a : m → ℂ) (b : n → ℂ) (R : Finset m) (C : Finset n) {p : ℝ}
    (hp : p ≠ 0) (s : ℝ) (i : m) (j : n) :
    (weightedFactorL a R p s * (weightedFactorR b C p s)ᴴ) i j =
      if i ∈ R ∧ j ∈ C then (1 - (s : ℂ) ^ 2) * vecMulVec a b i j else vecMulVec a b i j := by
  have h := blockWitness_apply R C s i j
  have hp' : (p : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr hp
  have hsum : (weightedFactorL a R p s * (weightedFactorR b C p s)ᴴ) i j =
      a i * b j * (blockFactorL R s * (blockFactorR C s)ᴴ) i j := by
    simp only [mul_apply, conjTranspose_apply, weightedFactorL, weightedFactorR, of_apply,
      star_mul', Complex.star_def, Complex.conj_conj, Complex.conj_ofReal, mul_sum]
    refine sum_congr rfl fun k _ => ?_
    push_cast
    field_simp
  rw [hsum, h, vecMulVec_apply]
  split_ifs <;> ring

/-- `∑_i w_i (x if i ∈ R else y) = (∑_R w) x + (∑ w − ∑_R w) y`. -/
lemma sum_mul_ite_mem (w : n → ℝ) (R : Finset n) (x y : ℝ) :
    ∑ i, w i * (if i ∈ R then x else y) =
      (∑ i ∈ R, w i) * x + (∑ i, w i - ∑ i ∈ R, w i) * y := by
  have h1 (i : n) (hi : i ∈ R) : w i * (if i ∈ R then x else y) = w i * x := by
    rw [ite_eq_left hi]
  have h2 (i : n) (hi : i ∈ Rᶜ) : w i * (if i ∈ R then x else y) = w i * y := by
    rw [ite_eq_right (mem_compl.mp hi)]
  rw [← sum_add_sum_compl R fun i => w i * (if i ∈ R then x else y), sum_congr rfl h1,
    sum_congr rfl h2, ← sum_mul, ← sum_mul, ← sum_add_sum_compl R w]
  ring

/-- `‖U‖² = p² (A − s² A_R + s⁴ A_R)`. -/
lemma frobeniusSq_weightedFactorL (a : m → ℂ) (R : Finset m) (p s : ℝ) :
    frobeniusSq (weightedFactorL a R p s) = p ^ 2 * (∑ i, ‖a i‖ ^ 2 -
      s ^ 2 * ∑ i ∈ R, ‖a i‖ ^ 2 + s ^ 4 * ∑ i ∈ R, ‖a i‖ ^ 2) := by
  have h (i : m) : ∑ k, ‖weightedFactorL a R p s i k‖ ^ 2 =
      p ^ 2 * (‖a i‖ ^ 2 * if i ∈ R then (1 - s ^ 2) ^ 2 + s ^ 2 else 1) := by
    by_cases hi : i ∈ R
    · simp [weightedFactorL, blockFactorL, hi, Fin.sum_univ_two, Complex.norm_real, mul_pow,
        sq_abs]
      ring
    · simp [weightedFactorL, blockFactorL, hi, Fin.sum_univ_two, Complex.norm_real, mul_pow]
  rw [frobeniusSq, sum_congr rfl fun i _ => h i, ← mul_sum, sum_mul_ite_mem]
  ring

/-- `‖V‖² = p⁻² (B + s² (B − B_C))`. -/
lemma frobeniusSq_weightedFactorR (b : n → ℂ) (C : Finset n) (p s : ℝ) :
    frobeniusSq (weightedFactorR b C p s) = (p⁻¹) ^ 2 * (∑ j, ‖b j‖ ^ 2 +
      s ^ 2 * (∑ j, ‖b j‖ ^ 2 - ∑ j ∈ C, ‖b j‖ ^ 2)) := by
  have h (j : n) : ∑ k, ‖weightedFactorR b C p s j k‖ ^ 2 =
      (p⁻¹) ^ 2 * (‖b j‖ ^ 2 * if j ∈ C then 1 else 1 + s ^ 2) := by
    by_cases hj : j ∈ C
    · simp [weightedFactorR, blockFactorR, hj, Fin.sum_univ_two, Complex.norm_real, mul_pow]
    · simp [weightedFactorR, blockFactorR, hj, Fin.sum_univ_two, Complex.norm_real, mul_pow,
        sq_abs]
      ring
  rw [frobeniusSq, sum_congr rfl fun j _ => h j, ← mul_sum, sum_mul_ite_mem]
  ring

/-- **The weighted block witness is short**: by Lemma 1, `‖a bᵀ − s² a_R b_Cᵀ‖_S1` is at most
`(p² (A − s² A_R + s⁴ A_R) + p⁻² (B + s² (B − B_C))) / 2`. -/
theorem nuclearNorm_weightedWitness_le (a : m → ℂ) (b : n → ℂ) (R : Finset m) (C : Finset n)
    (p s : ℝ) :
    nuclearNorm (weightedFactorL a R p s * (weightedFactorR b C p s)ᴴ) ≤
      (p ^ 2 * (∑ i, ‖a i‖ ^ 2 - s ^ 2 * ∑ i ∈ R, ‖a i‖ ^ 2 + s ^ 4 * ∑ i ∈ R, ‖a i‖ ^ 2) +
        (p⁻¹) ^ 2 * (∑ j, ‖b j‖ ^ 2 + s ^ 2 * (∑ j, ‖b j‖ ^ 2 - ∑ j ∈ C, ‖b j‖ ^ 2))) / 2 := by
  rw [← frobeniusSq_weightedFactorL, ← frobeniusSq_weightedFactorR]
  exact nuclearNorm_le_of_eq_mul _ _ rfl

/-- **The real inequality behind the witness**: if `0 ≤ A_R ≤ A`, `0 ≤ B_C ≤ B` and
`A B < A_R B + A B_C`, some `p ≠ 0` and `s` give
`(p² (A − s² A_R + s⁴ A_R) + p⁻² (B + s² (B − B_C))) / 2 < √A √B`. -/
lemma exists_weighted_bound_lt {A B AR BC : ℝ} (hAR : 0 ≤ AR) (hARA : AR ≤ A) (hBC : 0 ≤ BC)
    (hBCB : BC ≤ B) (hab : A * B < AR * B + A * BC) :
    ∃ p s : ℝ, p ≠ 0 ∧
      (p ^ 2 * (A - s ^ 2 * AR + s ^ 4 * AR) + (p⁻¹) ^ 2 * (B + s ^ 2 * (B - BC))) / 2 <
        √A * √B := by
  have hA : 0 < A := by
    rcases (hAR.trans hARA).lt_or_eq with h | h
    · exact h
    · rw [← h, le_antisymm (hARA.trans_eq h.symm) hAR] at hab
      simp at hab
  have hB : 0 < B := by
    rcases (hBC.trans hBCB).lt_or_eq with h | h
    · exact h
    · rw [← h, le_antisymm (hBCB.trans_eq h.symm) hBC] at hab
      simp at hab
  obtain ⟨x, hx, rfl⟩ : ∃ x, 0 < x ∧ x ^ 2 = A := ⟨√A, Real.sqrt_pos.mpr hA, Real.sq_sqrt hA.le⟩
  obtain ⟨y, hy, rfl⟩ : ∃ y, 0 < y ∧ y ^ 2 = B := ⟨√B, Real.sqrt_pos.mpr hB, Real.sq_sqrt hB.le⟩
  have hx0 := hx.ne'
  have hy0 := hy.ne'
  have hq : 0 < y / x := div_pos hy hx
  -- With `p² = y / x` both factors cost `x y`; `δ > 0` is the excess of the block.
  set t := y / x * AR with ht
  set δ := t + (y / x)⁻¹ * BC - x * y with hδ
  have hδ0 : 0 < δ := by
    have h : δ = (AR * y ^ 2 + x ^ 2 * BC - x ^ 2 * y ^ 2) / (x * y) := by
      rw [hδ, ht]
      field_simp
    rw [h]
    exact div_pos (by linarith) (mul_pos hx hy)
  have ht0 : 0 ≤ t := mul_nonneg hq.le hAR
  set s := δ / (t + δ) with hs
  have hs0 : 0 < s := div_pos hδ0 (by linarith)
  have hst : s ^ 2 * t < δ := by
    rw [hs, div_pow, div_mul_eq_mul_div, div_lt_iff₀ (pow_pos (by linarith) 2)]
    nlinarith [pow_pos hδ0 3, mul_nonneg hδ0.le (sq_nonneg t), mul_nonneg (sq_nonneg δ) ht0]
  refine ⟨√(y / x), s, (Real.sqrt_pos.mpr hq).ne', ?_⟩
  rw [Real.sq_sqrt hq.le, inv_pow, Real.sq_sqrt hq.le, Real.sqrt_sq hx.le, Real.sqrt_sq hy.le]
  have key : (y / x * (x ^ 2 - s ^ 2 * AR + s ^ 4 * AR) +
      (y / x)⁻¹ * (y ^ 2 + s ^ 2 * (y ^ 2 - BC))) / 2 = x * y - s ^ 2 / 2 * (δ - s ^ 2 * t) := by
    rw [hδ, ht]
    field_simp
    ring
  rw [key]
  have := mul_pos (div_pos (pow_pos hs0 2) two_pos) (sub_pos.mpr hst)
  linarith

example := exists_weighted_bound_lt (A := 5) (B := 2) (AR := 4) (BC := 1) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- **A block that breaks the weighted condition gives a shorter completion**,
`review/flat_spectrum.md`, Theorem G, item 3: if `Ω` misses `R × C` and
`A B < A_R B + A B_C`, some `X` equal to `a bᵀ` on `Ω` has `‖X‖_S1 < ‖a bᵀ‖_S1`. -/
theorem exists_nuclearNorm_lt_of_weighted_block {a : m → ℂ} {b : n → ℂ} {Ω : Set (m × n)}
    {R : Finset m} {C : Finset n} (hRC : ∀ i ∈ R, ∀ j ∈ C, (i, j) ∉ Ω)
    (hab : (∑ i, ‖a i‖ ^ 2) * ∑ j, ‖b j‖ ^ 2 <
      (∑ i ∈ R, ‖a i‖ ^ 2) * ∑ j, ‖b j‖ ^ 2 + (∑ i, ‖a i‖ ^ 2) * ∑ j ∈ C, ‖b j‖ ^ 2) :
    ∃ X : Matrix m n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec a b ij.1 ij.2) ∧
      nuclearNorm X < nuclearNorm (vecMulVec a b) := by
  obtain ⟨p, s, hp, hlt⟩ := exists_weighted_bound_lt (sum_nonneg fun i _ => sq_nonneg ‖a i‖)
    (sum_le_sum_of_subset_of_nonneg (subset_univ R) fun i _ _ => sq_nonneg ‖a i‖)
    (sum_nonneg fun j _ => sq_nonneg ‖b j‖)
    (sum_le_sum_of_subset_of_nonneg (subset_univ C) fun j _ _ => sq_nonneg ‖b j‖) hab
  refine ⟨weightedFactorL a R p s * (weightedFactorR b C p s)ᴴ, fun ij hij => ?_, ?_⟩
  · rw [weightedWitness_apply a b R C hp, ite_eq_right fun h => hRC _ h.1 _ h.2 hij]
  · rw [nuclearNorm_vecMulVec]
    exact (nuclearNorm_weightedWitness_le a b R C p s).trans_lt hlt

example := exists_nuclearNorm_lt_of_weighted_block (a := ![2, 1]) (b := ![1, 1])
  (Ω := {ij | ij.2 = ij.1}) (R := {0}) (C := {1}) (by simp)
  (by norm_num [Fin.sum_univ_two])

/-- In the `1 × 2` table, `(1)(1, 1)ᵀ` is not a minimum from its cell `(0, 0)`: the row carries
the mass `1`, the column it meets only `1/2`. -/
example := exists_nuclearNorm_lt_of_weighted_block (a := ![(1 : ℂ)]) (b := ![1, 1])
  (Ω := {(0, 0)}) (R := {0}) (C := {1}) (by simp) (by norm_num [Fin.sum_univ_two])

end Transformer.Grokking.FlatSpectrum
