module

public import Transformer.Grokking.FlatSpectrum.TablePart.E6FloorDigits
public import Transformer.Grokking.FlatSpectrum.TablePart.E6FloorZIm
public import Transformer.Grokking.FlatSpectrum.TablePart.E6FloorZRe

/-!
# PT6: the kernel bounds the rows of `Ẑᴴ Ẑ`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. `Ẑ` is the rounded
polar factor of the kernel of `⌊a b / 97⌋`, in units of `2⁻¹³`. A row of `Ẑ` is packed into one
integer in base `2³⁴`, so a row of `Ẑᴴ Ẑ` is one linear combination of 97 such integers, and its
entries are read back as digits (`combRe_eq`, `combIm_eq`). The kernel adds up their absolute
values over each row (`rowSum_le`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset

/-- The index of the cell `(i, j)` in the upper triangle, row by row. -/
def triCell (i j : ℕ) : ℕ :=
  if j < i then j * (195 - j) / 2 + (i - j) else i * (195 - i) / 2 + (j - i)

/-- The real part of `Ẑ`, in units of `2⁻¹³`. -/
def zre (i j : ℕ) : ℤ := (((floorZrePack >>> (13 * triCell i j)) &&& 8191 : ℕ) : ℤ) - 4096

/-- The imaginary part of `Ẑ`, in units of `2⁻¹³`. -/
def zim (i j : ℕ) : ℤ := (((floorZimPack >>> (13 * triCell i j)) &&& 8191 : ℕ) : ℤ) - 4096

/-- The row `l` of the real part, packed. -/
def rowRe (l : ℕ) : ℤ := hornerAt (zre l) 0 97

/-- The row `l` of the imaginary part, packed. -/
def rowIm (l : ℕ) : ℤ := hornerAt (zim l) 0 97

/-- The row `p` of the real part of `Ẑᴴ Ẑ`, packed. -/
def combRe (p : ℕ) : ℤ := sumFrom (fun l => zre l p * rowRe l + zim l p * rowIm l) 0 97

/-- The row `p` of the imaginary part of `Ẑᴴ Ẑ`, packed. -/
def combIm (p : ℕ) : ℤ := sumFrom (fun l => zre l p * rowIm l - zim l p * rowRe l) 0 97

/-- The sum of the absolute values of the parts over the row `p` of `Ẑᴴ Ẑ`. -/
def rowSum (p : ℕ) : ℤ :=
  sumFrom (fun q => (Int.natAbs (digit (combRe p) q) : ℤ) + Int.natAbs (digit (combIm p) q)) 0 97

/-- **The kernel bounds every row of `Ẑᴴ Ẑ`**, in units of `2⁻²⁶`. -/
theorem rowSum_le : ∀ p, p < 97 → rowSum p ≤ 67700000 := by
  decide +kernel

/-- The real part of the entry `(p, q)` of `Ẑᴴ Ẑ`, in units of `2⁻²⁶`. -/
def gramRe (p q : ℕ) : ℤ := ∑ l ∈ range 97, (zre l p * zre l q + zim l p * zim l q)

/-- The imaginary part of the entry `(p, q)` of `Ẑᴴ Ẑ`, in units of `2⁻²⁶`. -/
def gramIm (p q : ℕ) : ℤ := ∑ l ∈ range 97, (zre l p * zim l q - zim l p * zre l q)

lemma combRe_eq (p : ℕ) : combRe p = hornerAt (gramRe p) 0 97 := by
  simp only [combRe, rowRe, rowIm, sumFrom_eq, hornerAt_eq, gramRe, zero_add, Finset.mul_sum,
    ← Finset.sum_add_distrib, Finset.sum_mul]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun q _ => Finset.sum_congr rfl fun l _ => ?_
  ring

lemma combIm_eq (p : ℕ) : combIm p = hornerAt (gramIm p) 0 97 := by
  simp only [combIm, rowRe, rowIm, sumFrom_eq, hornerAt_eq, gramIm, zero_add, Finset.mul_sum,
    ← Finset.sum_sub_distrib, Finset.sum_mul]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun q _ => Finset.sum_congr rfl fun l _ => ?_
  ring

/-- A field of 13 bits, shifted down by 4096, has absolute value at most 4096. -/
lemma abs_field_le (x : ℕ) : |((x &&& 8191 : ℕ) : ℤ) - 4096| ≤ 4096 := by
  have h : x &&& 8191 ≤ 8191 := Nat.and_le_right
  rw [abs_le]
  omega

example : |((5 &&& 8191 : ℕ) : ℤ) - 4096| ≤ 4096 := abs_field_le 5

lemma abs_zre_le (i j : ℕ) : |zre i j| ≤ 4096 := abs_field_le _

lemma abs_zim_le (i j : ℕ) : |zim i j| ≤ 4096 := abs_field_le _

/-- Two products of numbers of size at most 4096. -/
lemma abs_pair_le {a b c d : ℤ} (ha : |a| ≤ 4096) (hb : |b| ≤ 4096) (hc : |c| ≤ 4096)
    (hd : |d| ≤ 4096) : |a * b + c * d| ≤ 33554432 := by
  calc |a * b + c * d| ≤ |a * b| + |c * d| := abs_add_le _ _
    _ = |a| * |b| + |c| * |d| := by rw [abs_mul, abs_mul]
    _ ≤ 4096 * 4096 + 4096 * 4096 := by gcongr
    _ = 33554432 := by norm_num

example : |(1 : ℤ) * 2 + 3 * 4| ≤ 33554432 :=
  abs_pair_le (by norm_num) (by norm_num) (by norm_num) (by norm_num)

lemma abs_gramRe_le (p q : ℕ) : |gramRe p q| ≤ 3254779904 := by
  calc |gramRe p q| ≤ ∑ l ∈ range 97, |zre l p * zre l q + zim l p * zim l q| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _l ∈ range 97, (33554432 : ℤ) := Finset.sum_le_sum fun l _ =>
        abs_pair_le (abs_zre_le _ _) (abs_zre_le _ _) (abs_zim_le _ _) (abs_zim_le _ _)
    _ = 3254779904 := by simp

lemma abs_gramIm_le (p q : ℕ) : |gramIm p q| ≤ 3254779904 := by
  calc |gramIm p q| ≤ ∑ l ∈ range 97, |zre l p * zim l q + -zim l p * zre l q| := by
        refine (Finset.abs_sum_le_sum_abs _ _).trans (le_of_eq ?_)
        refine Finset.sum_congr rfl fun l _ => ?_
        rw [neg_mul, ← sub_eq_add_neg]
    _ ≤ ∑ _l ∈ range 97, (33554432 : ℤ) := Finset.sum_le_sum fun l _ =>
        abs_pair_le (abs_zre_le _ _) (abs_zim_le _ _) (by rw [abs_neg]; exact abs_zim_le _ _)
          (abs_zre_le _ _)
    _ = 3254779904 := by simp

/-- **Every row of `Ẑᴴ Ẑ` has absolute sum at most `67700000 / 2²⁶`**, in its two parts. -/
theorem sum_abs_gram_le {p : ℕ} (hp : p < 97) :
    ∑ q ∈ range 97, (|gramRe p q| + |gramIm p q|) ≤ 67700000 := by
  have hre : ∀ k, k < 97 → -8589934592 ≤ gramRe p k ∧ gramRe p k < 8589934592 := fun k _ => by
    have h := abs_le.1 (abs_gramRe_le p k)
    omega
  have him : ∀ k, k < 97 → -8589934592 ≤ gramIm p k ∧ gramIm p k < 8589934592 := fun k _ => by
    have h := abs_le.1 (abs_gramIm_le p k)
    omega
  have h := rowSum_le p hp
  rw [rowSum, sumFrom_eq] at h
  refine le_trans (le_of_eq (Finset.sum_congr rfl fun q hq => ?_)) h
  have hq' := mem_range.1 hq
  rw [zero_add, combRe_eq, combIm_eq, digit_hornerAt hre hq', digit_hornerAt him hq',
    Int.natCast_natAbs, Int.natCast_natAbs]

example : ∑ q ∈ range 97, (|gramRe 0 q| + |gramIm 0 q|) ≤ 67700000 :=
  sum_abs_gram_le (by norm_num)

end Transformer.Grokking.FlatSpectrum.TablePart.E6
