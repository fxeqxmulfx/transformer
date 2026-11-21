module

public import Transformer.Grokking.FlatSpectrum.TablePart.E6DivParts

/-!
# PT6: a row of integers packed into one integer

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. The kernel of Lean
multiplies big integers fast and has no fast arrays. So a row of 97 small integers travels as one
integer in base `2³⁴`, a linear combination of rows is a linear combination of these integers, and
each entry of the result is read back as a digit. A digit in `[−2³³, 2³³)` is read exactly
(`digit_hornerAt`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset

/-- Horner in base `2 ^ 34`. -/
def hornerAt (v : ℕ → ℤ) (i : ℕ) : ℕ → ℤ
  | 0 => 0
  | n + 1 => v i + 17179869184 * hornerAt v (i + 1) n

/-- The sum `F i + ⋯ + F (i + n − 1)`. -/
def sumFrom (F : ℕ → ℤ) (i : ℕ) : ℕ → ℤ
  | 0 => 0
  | n + 1 => F i + sumFrom F (i + 1) n

/-- The digit `2³³` in each of 97 places. -/
def offset : ℤ := hornerAt (fun _ => 8589934592) 0 97

/-- The digit of `P` in place `q`, read as a signed number. -/
def digit (P : ℤ) (q : ℕ) : ℤ :=
  ((((P + offset).toNat >>> (34 * q)) &&& 17179869183 : ℕ) : ℤ) - 8589934592

lemma sumFrom_eq (F : ℕ → ℤ) (i n : ℕ) : sumFrom F i n = ∑ k ∈ range n, F (i + k) := by
  induction n generalizing i with
  | zero => simp [sumFrom]
  | succ n ih =>
    rw [sumFrom, ih, Finset.sum_range_succ', add_zero, add_comm]
    refine congrArg (· + F i) (Finset.sum_congr rfl fun k _ => ?_)
    rw [add_assoc, add_comm 1 k]

lemma hornerAt_eq (v : ℕ → ℤ) (i n : ℕ) :
    hornerAt v i n = ∑ k ∈ range n, v (i + k) * 17179869184 ^ k := by
  induction n generalizing i with
  | zero => simp [hornerAt]
  | succ n ih =>
    rw [hornerAt, ih, Finset.sum_range_succ', Finset.mul_sum, add_zero, pow_zero, mul_one, add_comm]
    refine congrArg (· + v i) (Finset.sum_congr rfl fun k _ => ?_)
    rw [show i + (k + 1) = i + 1 + k by omega, pow_succ]
    ring

lemma hornerAt_add (v w : ℕ → ℤ) (i n : ℕ) :
    hornerAt v i n + hornerAt w i n = hornerAt (fun k => v k + w k) i n := by
  simp only [hornerAt_eq, ← Finset.sum_add_distrib, add_mul]

/-- Digits in `[0, 2³⁴)` are read back from their Horner sum. -/
lemma horner_digits {w : ℕ → ℤ} : ∀ n i : ℕ,
    (∀ k, k < n → 0 ≤ w (i + k) ∧ w (i + k) < 17179869184) →
      0 ≤ hornerAt w i n ∧ ∀ q, q < n →
        (hornerAt w i n).toNat / 17179869184 ^ q % 17179869184 = (w (i + q)).toNat := by
  intro n
  induction n with
  | zero => exact fun i _ => ⟨le_rfl, fun q hq => absurd hq (Nat.not_lt_zero q)⟩
  | succ n ih =>
    intro i h
    obtain ⟨h0, h1⟩ := h 0 (by omega)
    rw [add_zero] at h0 h1
    obtain ⟨hH, hd⟩ := ih (i + 1) fun k hk => by
      have h2 := h (k + 1) (by omega)
      rwa [show i + (k + 1) = i + 1 + k by omega] at h2
    obtain ⟨x, hx⟩ := Int.eq_ofNat_of_zero_le h0
    obtain ⟨y, hy⟩ := Int.eq_ofNat_of_zero_le hH
    have hxlt : x < 17179869184 := by omega
    have hN : hornerAt w i (n + 1) = ((x + 17179869184 * y : ℕ) : ℤ) := by
      rw [hornerAt, hx, hy]
      push_cast
      ring
    refine ⟨by rw [hN]; exact Int.natCast_nonneg _, fun q hq => ?_⟩
    rw [hN, Int.toNat_natCast]
    cases q with
    | zero =>
      rw [pow_zero, Nat.div_one, add_zero, hx, Int.toNat_natCast, Nat.add_mul_mod_self_left,
        Nat.mod_eq_of_lt hxlt]
    | succ q =>
      have e : (x + 17179869184 * y) / 17179869184 = y := by
        rw [Nat.add_mul_div_left _ _ (by norm_num), Nat.div_eq_of_lt hxlt, zero_add]
      have h3 := hd q (by omega)
      rw [hy, Int.toNat_natCast] at h3
      rw [pow_succ', ← Nat.div_div_eq_div_mul, e, h3, show i + (q + 1) = i + 1 + q by omega]

example : (hornerAt (fun _ => 5) 0 2).toNat / 17179869184 ^ 1 % 17179869184 =
    ((fun _ => (5 : ℤ)) (0 + 1)).toNat :=
  (horner_digits 2 0 fun _ _ => by norm_num).2 1 (by norm_num)

/-- **A digit in `[−2³³, 2³³)` is read back exactly.** -/
theorem digit_hornerAt {v : ℕ → ℤ} (hv : ∀ k, k < 97 → -8589934592 ≤ v k ∧ v k < 8589934592)
    {q : ℕ} (hq : q < 97) : digit (hornerAt v 0 97) q = v q := by
  have hsum : hornerAt v 0 97 + offset = hornerAt (fun k => v k + 8589934592) 0 97 :=
    hornerAt_add v (fun _ => 8589934592) 0 97
  obtain ⟨-, hd⟩ := horner_digits (w := fun k => v k + 8589934592) 97 0 fun k hk => by
    have h1 := hv k hk
    simp only [zero_add]
    omega
  have h := hd q hq
  simp only [zero_add] at h
  have hmask : (17179869183 : ℕ) = 2 ^ 34 - 1 := by norm_num
  have hpow : (2 : ℕ) ^ (34 * q) = 17179869184 ^ q := by
    rw [pow_mul]
    norm_num
  have hbase : (2 : ℕ) ^ 34 = 17179869184 := by norm_num
  have hnn : 0 ≤ v q + 8589934592 := by
    have h1 := hv q hq
    omega
  rw [digit, hsum, hmask, Nat.and_two_pow_sub_one_eq_mod, Nat.shiftRight_eq_div_pow, hpow, hbase, h,
    Int.toNat_of_nonneg hnn]
  ring

example : digit (hornerAt (fun _ => -3) 0 97) 7 = -3 :=
  digit_hornerAt (fun _ _ => by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.TablePart.E6
