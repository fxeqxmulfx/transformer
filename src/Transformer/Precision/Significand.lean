/-
# Numbers with `p` significant bits

A format has `p` significant bits if every number in it is `± m · 2^k` with
`m < 2^p` (`HasSignificand`).  The exponent range is not bounded: this covers
every IEEE binary format, bfloat16, the FP8 formats, and the idealised formats
with no overflow and no underflow alike.

Beyond `2^{p-1} · 2^s` such numbers are multiples of `2^s`
(`HasSignificand.dvd`), so there consecutive numbers of the format are at least
`2^s` apart (`HasSignificand.gap`).  This spacing is what a running sum in the
format runs into (`Precision.Accumulate`).
-/

import Mathlib.Basic.Real.Basic
import Mathlib.Algebra.Order.Field.Power
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace Transformer
namespace Precision

/-- Every number of `G` has at most `p` significant bits: `|z| = m · 2^k` with
`m < 2^p`. -/
def HasSignificand (p : ℕ) (G : Set ℝ) : Prop :=
  ∀ z ∈ G, ∃ m : ℕ, m < 2 ^ p ∧ ∃ k : ℤ, |z| = m * (2 : ℝ) ^ k

/-- **Above `2^M · 2^s`, every number with `M + 1` significant bits is a
multiple of `2^s`.** -/
theorem HasSignificand.dvd {M : ℕ} {G : Set ℝ} (hG : HasSignificand (M + 1) G) {z : ℝ} (h : z ∈ G)
    {s : ℤ} (hz : (2 : ℝ) ^ M * 2 ^ s ≤ z) : ∃ j : ℤ, z = 2 ^ s * j := by
  obtain ⟨n, hn, k, hk⟩ := hG z h
  have hz0 : 0 < z := lt_of_lt_of_le (by positivity) hz
  rw [abs_of_pos hz0] at hk
  subst hk
  have hsk : s ≤ k := by
    by_contra hlt
    have h1 : (n : ℝ) + 1 ≤ 2 ^ (M + 1) := by exact_mod_cast hn
    have h2 : (2 : ℝ) ^ (k + 1) ≤ 2 ^ s := zpow_le_zpow_right₀ (by norm_num) (by omega)
    rw [zpow_add_one₀ (by norm_num)] at h2
    have h3 : (2 : ℝ) ^ (M + 1) = 2 ^ M * 2 := pow_succ _ _
    nlinarith [zpow_pos (show (0 : ℝ) < 2 by norm_num) k, pow_pos (show (0 : ℝ) < 2 by norm_num) M]
  obtain ⟨j, hj⟩ : ∃ j : ℕ, k = s + j := ⟨(k - s).toNat, by omega⟩
  refine ⟨n * 2 ^ j, ?_⟩
  rw [hj, zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0), zpow_natCast]
  push_cast; ring

/-- Past `2^M · 2^s`, the next number with `M + 1` significant bits is at least
`2^s` away. -/
theorem HasSignificand.gap {M : ℕ} {G : Set ℝ} (hG : HasSignificand (M + 1) G) {a z : ℝ}
    (ha : a ∈ G) (hz : z ∈ G) {s : ℤ} (h : (2 : ℝ) ^ M * 2 ^ s ≤ a) (haz : a < z) :
    a + 2 ^ s ≤ z := by
  obtain ⟨i, rfl⟩ := hG.dvd ha h
  obtain ⟨j, rfl⟩ := hG.dvd hz (s := s) (by linarith)
  have hs : (0 : ℝ) < 2 ^ s := zpow_pos (by norm_num) _
  have : i < j := by
    have : (i : ℝ) < j := lt_of_mul_lt_mul_left haz hs.le
    exact_mod_cast this
  have : (i : ℝ) + 1 ≤ j := by exact_mod_cast this
  nlinarith

/-- `{0, 1, 2, 4}` has one significant bit. -/
theorem hasSignificand_one_pow_two : HasSignificand 1 ({0, 1, 2, 4} : Set ℝ) := by
  rintro z (rfl | rfl | rfl | rfl)
  · exact ⟨0, by norm_num, 0, by norm_num⟩
  · exact ⟨1, by norm_num, 0, by norm_num⟩
  · exact ⟨1, by norm_num, 1, by norm_num⟩
  · exact ⟨1, by norm_num, 2, by norm_num⟩

/-- The hypotheses of `HasSignificand.dvd` and `HasSignificand.gap` are
satisfiable: `2 < 4` in `{0, 1, 2, 4}`, above `2^0 · 2^1`. -/
example : HasSignificand (0 + 1) ({0, 1, 2, 4} : Set ℝ) ∧ (2 : ℝ) ∈ ({0, 1, 2, 4} : Set ℝ) ∧
    (4 : ℝ) ∈ ({0, 1, 2, 4} : Set ℝ) ∧ (2 : ℝ) ^ 0 * 2 ^ (1 : ℤ) ≤ 2 ∧ (2 : ℝ) < 4 :=
  ⟨hasSignificand_one_pow_two, by simp, by simp, by norm_num, by norm_num⟩

end Precision
end Transformer
