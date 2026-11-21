module

public import Transformer.Grokking.FlatSpectrum.TablePart.E6Peel

/-!
# PT6: the tridiagonal inverse of `[ρ^{|a − b|}]`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. For any `ρ`, the
matrix `[ρ^{|a − b|}]` on `0, …, 96` times the tridiagonal `tri ρ` is `(1 − ρ²) 1`. Each column of
`tri ρ` has three entries at most, so the identity is a three-term cancellation.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset

/-- The distance `|a − b|` of two naturals. -/
def natDist (a b : ℕ) : ℕ := a - b + (b - a)

lemma natDist_comm (a b : ℕ) : natDist b a = natDist a b := by
  unfold natDist
  omega

/-- `1 − ρ²` times the inverse of `[ρ^{|a − b|}]`: a tridiagonal matrix. -/
noncomputable def tri (ρ : ℂ) (l j : ℕ) : ℂ :=
  if l = j then (if j = 0 then 1 else if j = 96 then 1 else 1 + ρ ^ 2)
  else if l + 1 = j then -ρ else if j + 1 = l then -ρ else 0

/-- The term with `x + 1 = j` of a sum over `range 97`. -/
lemma sum_ite_succ_eq (G : ℕ → ℂ) {j : ℕ} (hj : j < 97) :
    (∑ x ∈ range 97, if x + 1 = j then G x else 0) = if 0 < j then G (j - 1) else 0 := by
  rcases j with _ | j
  · simp
  · have hlt : j < 97 := by omega
    simp [hlt]

example : (∑ x ∈ range 97, if x + 1 = 3 then (x : ℂ) else 0) =
    if 0 < 3 then ((3 - 1 : ℕ) : ℂ) else 0 :=
  sum_ite_succ_eq (fun x => (x : ℂ)) (by norm_num)

/-- A column of `tri` against a vector: three terms at most. -/
lemma sum_mul_tri (g : ℕ → ℂ) (ρ : ℂ) {j : ℕ} (hj : j < 97) :
    ∑ x ∈ range 97, g x * tri ρ x j =
      g j * (if j = 0 then 1 else if j = 96 then 1 else 1 + ρ ^ 2) +
        (if 0 < j then g (j - 1) * -ρ else 0) + (if j + 1 < 97 then g (j + 1) * -ρ else 0) := by
  have h3 (x : ℕ) : g x * tri ρ x j =
      (if x = j then g x * (if j = 0 then 1 else if j = 96 then 1 else 1 + ρ ^ 2) else 0) +
        (if x + 1 = j then g x * -ρ else 0) + (if x = j + 1 then g x * -ρ else 0) := by
    unfold tri
    split_ifs <;> first | (exfalso; omega) | simp
  simp only [h3, Finset.sum_add_distrib, Finset.sum_ite_eq', mem_range]
  rw [ite_eq_left hj, sum_ite_succ_eq (fun x => g x * -ρ) hj]

example (ρ : ℂ) := sum_mul_tri (fun _ => 1) ρ (j := 3) (by norm_num)

/-- **`(1 − ρ²)` times the inverse of `[ρ^{|a − b|}]` is `tri ρ`.** -/
theorem sum_kms_tri (ρ : ℂ) {a j : ℕ} (ha : a < 97) (hj : j < 97) :
    ∑ x ∈ range 97, ρ ^ natDist a x * tri ρ x j = if a = j then 1 - ρ ^ 2 else 0 := by
  rw [sum_mul_tri (fun x => ρ ^ natDist a x) ρ hj]
  rcases lt_trichotomy a j with h | rfl | h
  · obtain ⟨k, rfl⟩ : ∃ k, j = a + k + 1 := ⟨j - a - 1, by omega⟩
    have h1 : natDist a (a + k + 1 - 1) = k := by unfold natDist; omega
    have h2 : natDist a (a + k + 1) = k + 1 := by unfold natDist; omega
    have h3 : natDist a (a + k + 1 + 1) = k + 2 := by unfold natDist; omega
    simp only [h1, h2, h3]
    split_ifs <;> first | (exfalso; omega) | ring1
  · have h2 : natDist a a = 0 := by unfold natDist; omega
    have h3 : natDist a (a + 1) = 1 := by unfold natDist; omega
    simp only [h2, h3]
    by_cases h0 : 0 < a
    · have h1 : natDist a (a - 1) = 1 := by unfold natDist; omega
      simp only [h1]
      split_ifs <;> first | (exfalso; omega) | ring1
    · split_ifs <;> first | (exfalso; omega) | ring1
  · obtain ⟨k, rfl⟩ : ∃ k, a = j + k + 1 := ⟨a - j - 1, by omega⟩
    have h2 : natDist (j + k + 1) j = k + 1 := by unfold natDist; omega
    have h3 : natDist (j + k + 1) (j + 1) = k := by unfold natDist; omega
    simp only [h2, h3]
    by_cases h0 : 0 < j
    · have h1 : natDist (j + k + 1) (j - 1) = k + 2 := by unfold natDist; omega
      simp only [h1]
      split_ifs <;> first | (exfalso; omega) | ring1
    · split_ifs <;> first | (exfalso; omega) | ring1

example (ρ : ℂ) := sum_kms_tri ρ (a := 2) (j := 3) (by norm_num) (by norm_num)

lemma tri_ne_zero {ρ : ℂ} (h0 : ρ ≠ 0) (h2 : 1 + ρ ^ 2 ≠ 0) {l j : ℕ} (hl : l < 97)
    (hj : j ∈ nb l) : tri ρ l j ≠ 0 := by
  have h := (mem_nb hl).1 hj
  unfold tri
  split_ifs <;> first | exact one_ne_zero | exact h2 | exact neg_ne_zero.2 h0 | (exfalso; omega)

example : tri 2 1 2 ≠ 0 :=
  tri_ne_zero (by norm_num) (by norm_num) (by norm_num) ((mem_nb (by norm_num)).2 (by norm_num))

lemma tri_eq_zero (ρ : ℂ) {l j : ℕ} (hl : l < 97) (hj : j < 97) (hn : j ∉ nb l) :
    tri ρ l j = 0 := by
  have h : ¬ (l ≤ j + 1 ∧ j ≤ l + 1) := fun h => hn ((mem_nb hl).2 ⟨hj, h.1, h.2⟩)
  unfold tri
  split_ifs <;> first | rfl | (exfalso; omega)

example : tri 2 1 5 = 0 :=
  tri_eq_zero 2 (by norm_num) (by norm_num) fun h => by
    have := (mem_nb (by norm_num)).1 h
    omega

attribute [local instance] fact_prime_97

/-- A power of `ψ(x)` with `x ≠ 0` is not `1`, unless the exponent is a multiple of `97`. -/
lemma stdAddChar_pow_ne_one {x : ZMod 97} (hx : x ≠ 0) {n : ℕ} (hn : (n : ZMod 97) ≠ 0) :
    (ZMod.stdAddChar x : ℂ) ^ n ≠ 1 := by
  rw [← AddChar.map_nsmul_eq_pow, nsmul_eq_mul]
  intro h
  have h0 : (n : ZMod 97) * x = 0 :=
    ZMod.injective_stdAddChar (by rw [h, AddChar.map_zero_eq_one])
  exact (mul_ne_zero hn hx) h0

example : (ZMod.stdAddChar (1 : ZMod 97) : ℂ) ^ 2 ≠ 1 :=
  stdAddChar_pow_ne_one one_ne_zero (by decide +kernel)

/-- `1 + ψ(x)² ≠ 0` for `x ≠ 0`. -/
lemma one_add_sq_ne_zero {x : ZMod 97} (hx : x ≠ 0) : 1 + (ZMod.stdAddChar x : ℂ) ^ 2 ≠ 0 := by
  intro h
  have h4 : (ZMod.stdAddChar x : ℂ) ^ 4 = 1 := by
    linear_combination ((ZMod.stdAddChar x : ℂ) ^ 2 - 1) * h
  exact stdAddChar_pow_ne_one hx (n := 4) (by decide +kernel) h4

example : 1 + (ZMod.stdAddChar (1 : ZMod 97) : ℂ) ^ 2 ≠ 0 := one_add_sq_ne_zero one_ne_zero

end Transformer.Grokking.FlatSpectrum.TablePart.E6
