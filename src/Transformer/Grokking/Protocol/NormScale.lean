module

public import Transformer.Grokking.Protocol.Model

/-!
# Layer normalization of a multiple of a row

Source: openai/grok at 3d64b1d8, `grok/transformer.py`, the `LayerNorm` of `DecoderBlock`,
with the gain `1` and the bias `0` of its initialization; `Model.layerNorm` is its formula.

With gain `1` and bias `0`, the normalization of `t (z - c)`, a multiple of a row shifted by a
constant, is `normScale t z` times the centred row (`layerNorm_mul_sub`): the constant
disappears and the factor is `t / √(t² Var z + 10⁻⁵)`. For `t > 0` the factor is positive
(`normScale_pos`) and its inverse square is `Var z + 10⁻⁵ / t²` (`inv_normScale_sq`). So a
chain of normalizations of a row keeps its direction, and the difference of two entries keeps
its sign: this is what lets the answer be read after the four normalizations of the decoder.

A row with three marked entries has an explicit mean and variance (`sum_three`, `sum_sq_three`,
`rowVar_eq`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

variable {n : ℕ}

/-- The mean of the entries of a row. -/
noncomputable def rowMean (z : Fin n → ℝ) : ℝ := (∑ j, z j) / n

/-- The variance of the entries of a row. -/
noncomputable def rowVar (z : Fin n → ℝ) : ℝ := (∑ j, (z j - rowMean z) ^ 2) / n

lemma rowVar_nonneg (z : Fin n → ℝ) : 0 ≤ rowVar z :=
  div_nonneg (Finset.sum_nonneg fun _ _ => sq_nonneg _) (Nat.cast_nonneg n)

/-- The factor by which the normalization of `t z` multiplies the centred `z`. -/
noncomputable def normScale (t : ℝ) (z : Fin n → ℝ) : ℝ := t / √(t ^ 2 * rowVar z + 1 / 10 ^ 5)

lemma normScale_pos {t : ℝ} (ht : 0 < t) (z : Fin n → ℝ) : 0 < normScale t z := by
  have := rowVar_nonneg z
  exact div_pos ht (Real.sqrt_pos.2 (by positivity))

/-- **The inverse square of the factor** is the variance and `10⁻⁵` over the square of the
multiple. -/
lemma inv_normScale_sq {t : ℝ} (ht : 0 < t) (z : Fin n → ℝ) :
    1 / normScale t z ^ 2 = rowVar z + 1 / 10 ^ 5 / t ^ 2 := by
  have hv := rowVar_nonneg z
  have hD : 0 < t ^ 2 * rowVar z + 1 / 10 ^ 5 := by positivity
  rw [normScale, div_pow, Real.sq_sqrt hD.le]
  field_simp

/-- **The normalization of a multiple of a row** is a multiple of the centred row. -/
lemma layerNorm_mul (t : ℝ) (z : Fin n → ℝ) (i : Fin n) :
    layerNorm (fun _ => 1) (fun _ => 0) (fun j => t * z j) i =
      normScale t z * (z i - rowMean z) := by
  have hm : (∑ j, t * z j) / n = t * rowMean z := by
    rw [← Finset.mul_sum, rowMean, mul_div_assoc]
  have hv : (∑ j, (t * z j - t * rowMean z) ^ 2) / n = t ^ 2 * rowVar z := by
    rw [rowVar, ← mul_div_assoc, Finset.mul_sum]
    exact congrArg (· / (n : ℝ)) (Finset.sum_congr rfl fun j _ => by ring)
  simp only [layerNorm, hm, hv, normScale]
  ring

/-- **The normalization does not see a constant**: that of a multiple of a shifted row is the
same multiple of the centred row. -/
lemma layerNorm_mul_sub (hn : n ≠ 0) (t c : ℝ) (z : Fin n → ℝ) (i : Fin n) :
    layerNorm (fun _ => 1) (fun _ => 0) (fun j => t * (z j - c)) i =
      normScale t z * (z i - rowMean z) := by
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hn
  have hm : rowMean (fun j => z j - c) = rowMean z - c := by
    rw [rowMean, rowMean, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul, sub_div, mul_div_cancel_left₀ _ hn']
  have hv : rowVar (fun j => z j - c) = rowVar z := by
    rw [rowVar, rowVar, hm]
    exact congrArg (· / (n : ℝ)) (Finset.sum_congr rfl fun j _ => by ring)
  rw [layerNorm_mul t (fun j => z j - c) i, hm, normScale, hv, normScale]
  ring

lemma layerNorm_one (z : Fin n → ℝ) (i : Fin n) :
    layerNorm (fun _ => 1) (fun _ => 0) z i = normScale 1 z * (z i - rowMean z) := by
  have h := layerNorm_mul 1 z i
  simpa only [one_mul] using h

/-- The variance is the mean of the squares less the square of the mean. -/
lemma rowVar_eq (hn : n ≠ 0) (z : Fin n → ℝ) :
    rowVar z = ((∑ j, z j ^ 2) - (∑ j, z j) ^ 2 / n) / n := by
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hn
  have h : ∑ j, (z j - rowMean z) ^ 2 =
      (∑ j, z j ^ 2) - 2 * rowMean z * (∑ j, z j) + n * rowMean z ^ 2 := by
    rw [Finset.sum_congr rfl fun j _ => show (z j - rowMean z) ^ 2 =
      z j ^ 2 - 2 * rowMean z * z j + rowMean z ^ 2 by ring]
    rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, Finset.sum_const,
      Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  rw [rowVar, h, rowMean]
  field_simp
  ring

/-- The indicator of the entry `k` of a row. -/
def ind (k : ℕ) (i : Fin n) : ℝ := if (i : ℕ) = k then 1 else 0

lemma ind_of_eq {k : ℕ} {i : Fin n} (h : (i : ℕ) = k) : ind k i = 1 := ite_eq_left h

lemma ind_of_ne {k : ℕ} {i : Fin n} (h : (i : ℕ) ≠ k) : ind k i = 0 := ite_eq_right h

/-- A sum against an indicator is the entry. -/
lemma sum_ind_mul {k : ℕ} (hk : k < n) (f : Fin n → ℝ) : ∑ i, ind k i * f i = f ⟨k, hk⟩ := by
  rw [Finset.sum_eq_single (⟨k, hk⟩ : Fin n)]
  · rw [ind_of_eq (i := (⟨k, hk⟩ : Fin n)) rfl, one_mul]
  · intro i _ hi
    rw [ind_of_ne fun h => hi (Fin.ext h), zero_mul]
  · intro h
    exact absurd (Finset.mem_univ _) h

lemma sum_ind {k : ℕ} (hk : k < n) : ∑ i : Fin n, ind k i = 1 := by
  have h := sum_ind_mul hk fun _ => (1 : ℝ)
  simpa only [mul_one] using h

/-- The sum of a row with three marked entries. -/
lemma sum_three {p q r : ℕ} (hp : p < n) (hq : q < n) (hr : r < n) (a b c : ℝ) :
    ∑ i : Fin n, (a * ind p i + b * ind q i + c * ind r i) = a + b + c := by
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
    ← Finset.mul_sum, sum_ind hp, sum_ind hq, sum_ind hr]
  ring

/-- The sum of the squares of a row with three marked entries. -/
lemma sum_sq_three {p q r : ℕ} (hp : p < n) (hq : q < n) (hr : r < n) (hpq : p ≠ q)
    (hpr : p ≠ r) (hqr : q ≠ r) (a b c : ℝ) :
    ∑ i : Fin n, (a * ind p i + b * ind q i + c * ind r i) ^ 2 = a ^ 2 + b ^ 2 + c ^ 2 := by
  rw [← sum_three hp hq hr (a ^ 2) (b ^ 2) (c ^ 2)]
  refine Finset.sum_congr rfl fun i _ => ?_
  by_cases h1 : (i : ℕ) = p
  · rw [ind_of_eq h1, ind_of_ne fun h => hpq (h1.symm.trans h),
      ind_of_ne fun h => hpr (h1.symm.trans h)]
    ring
  · by_cases h2 : (i : ℕ) = q
    · rw [ind_of_ne h1, ind_of_eq h2, ind_of_ne fun h => hqr (h2.symm.trans h)]
      ring
    · by_cases h3 : (i : ℕ) = r
      · rw [ind_of_ne h1, ind_of_ne h2, ind_of_eq h3]
        ring
      · rw [ind_of_ne h1, ind_of_ne h2, ind_of_ne h3]
        ring

/-! The witnesses: a row of two entries, and a row of four with three of them marked. -/

example : 0 < normScale 2 ![0, 1] := normScale_pos (by norm_num) _

example := inv_normScale_sq (t := 2) (by norm_num) ![0, 1]

example := layerNorm_mul_sub (n := 2) (by norm_num) 2 1 ![0, 1] 0

example := rowVar_eq (n := 2) (by norm_num) ![0, 1]

example : ind 2 (2 : Fin 4) = 1 := ind_of_eq rfl

example : ind 2 (3 : Fin 4) = 0 := ind_of_ne (by decide)

example := sum_ind_mul (n := 4) (k := 2) (by norm_num) ![1, 2, 3, 4]

example := sum_ind (n := 4) (k := 2) (by norm_num)

example := sum_three (n := 4) (p := 0) (q := 1) (r := 3) (by norm_num) (by norm_num)
  (by norm_num) 1 2 3

example := sum_sq_three (n := 4) (p := 0) (q := 1) (r := 3) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) 1 2 3

end Transformer.Grokking.Protocol
