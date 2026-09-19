/-
# Walsh characters and the orthogonality of the Hadamard matrix

Support for arXiv:2601.22813v2, §3.2: the randomized Hadamard transform `RHT`
of `Quartet II` is the Walsh–Hadamard matrix with random signs, and the two
statements §3.2 makes about it — that it is invertible, and that it cancels
along the inner dimension of a product — are both the orthogonality of its
rows.

`walsh n a b` is the sign `(-1)^{⟨a,b⟩}` of the Walsh character on `n` bits:
`⟨a,b⟩` counts the bit positions below `n` where `a` and `b` are both set.
The one fact everything rests on is `sum_walsh_mul_range`: summing
`walsh n a c · walsh n b c` over all `c < 2^n` gives `2^n` when `a` and `b`
agree below `n`, and `0` otherwise — the characters of `(ℤ/2)^n` are
orthogonal.  It is proved by induction on `n`, splitting `[0, 2^{n+1})` at
`2^n`, which is exactly splitting on the top bit.
-/

import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Nat.Bitwise
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Mathlib.Algebra.BigOperators.Ring.Finset

namespace Transformer
namespace Quartet

/-- The Walsh character on `n` bits: `(-1)` to the number of positions below
`n` at which both `a` and `b` are set. -/
noncomputable def walsh (n a b : ℕ) : ℝ :=
  (-1) ^ ((Finset.range n).filter fun t => a.testBit t && b.testBit t).card

/-- The character is symmetric in its two indices, so the Hadamard matrix of
§3.2 is a symmetric matrix. -/
theorem walsh_comm (n a b : ℕ) : walsh n a b = walsh n b a := by
  unfold walsh
  congr 2
  ext t
  simp [Bool.and_comm]

/-- The character reads only the bits below `n`. -/
theorem walsh_congr_right {n a b b' : ℕ} (h : ∀ t < n, b.testBit t = b'.testBit t) :
    walsh n a b = walsh n a b' := by
  unfold walsh
  congr 2
  ext t
  simp only [Finset.mem_filter, Finset.mem_range, and_congr_right_iff]
  intro ht
  rw [h t ht]

/-- The hypothesis of `walsh_congr_right` is satisfiable: two numbers that
agree below `n` are its intended arguments, and `b = b'` is the extreme
case. -/
example (n b : ℕ) : ∀ t < n, b.testBit t = b.testBit t := fun _ _ => rfl

/-- One more bit multiplies the character by the sign of that bit. -/
theorem walsh_succ (n a b : ℕ) :
    walsh (n + 1) a b = walsh n a b * (if a.testBit n && b.testBit n then -1 else 1) := by
  unfold walsh
  rw [Finset.range_add_one, Finset.filter_insert]
  by_cases h : (a.testBit n && b.testBit n) = true
  · rw [ite_eq_left h, Finset.card_insert_of_notMem (by simp), pow_succ, ite_eq_left h]
  · rw [ite_eq_right h, ite_eq_right h, mul_one]

/-- **The Walsh characters are orthogonal.**  This is the whole content of
§3.2: the rows of the Hadamard matrix are pairwise orthogonal and of equal
norm. -/
theorem sum_walsh_mul_range (n a b : ℕ) :
    ∑ c ∈ Finset.range (2 ^ n), walsh n a c * walsh n b c =
      if ∀ t < n, a.testBit t = b.testBit t then (2 ^ n : ℝ) else 0 := by
  induction n with
  | zero => simp [walsh]
  | succ n ih =>
    have hsplit : ∑ c ∈ Finset.range (2 ^ (n + 1)), walsh (n + 1) a c * walsh (n + 1) b c
        = (∑ d ∈ Finset.range (2 ^ n), walsh (n + 1) a d * walsh (n + 1) b d) +
          ∑ d ∈ Finset.range (2 ^ n),
            walsh (n + 1) a (2 ^ n + d) * walsh (n + 1) b (2 ^ n + d) := by
      have h2 : 2 ^ (n + 1) = 2 ^ n + 2 ^ n := by rw [pow_succ, Nat.mul_two]
      rw [h2, Finset.sum_range_add]
    have hlow : ∀ d ∈ Finset.range (2 ^ n),
        walsh (n + 1) a d * walsh (n + 1) b d = walsh n a d * walsh n b d := by
      intro d hd
      have hdb : d.testBit n = false :=
        Nat.testBit_lt_two_pow (Finset.mem_range.mp hd)
      rw [walsh_succ, walsh_succ, hdb]
      simp
    have hhigh : ∀ d ∈ Finset.range (2 ^ n),
        walsh (n + 1) a (2 ^ n + d) * walsh (n + 1) b (2 ^ n + d) =
          (walsh n a d * walsh n b d) *
            ((if a.testBit n then -1 else 1) * (if b.testBit n then -1 else 1)) := by
      intro d hd
      have hdb : d.testBit n = false :=
        Nat.testBit_lt_two_pow (Finset.mem_range.mp hd)
      have htop : (2 ^ n + d).testBit n = true := by
        rw [Nat.testBit_two_pow_add_eq, hdb]; rfl
      have hlow' : ∀ t < n, (2 ^ n + d).testBit t = d.testBit t :=
        fun t ht => Nat.testBit_two_pow_add_gt ht d
      rw [walsh_succ, walsh_succ, htop, walsh_congr_right hlow', walsh_congr_right hlow']
      simp only [Bool.and_true]
      ring
    rw [hsplit, Finset.sum_congr rfl hlow, Finset.sum_congr rfl hhigh, ← Finset.sum_mul, ih]
    by_cases hab : a.testBit n = b.testBit n
    · have hs : ((if a.testBit n then (-1 : ℝ) else 1) * (if b.testBit n then -1 else 1)) = 1 := by
        rw [hab]; cases b.testBit n <;> norm_num
      rw [hs, mul_one]
      by_cases hall : ∀ t < n, a.testBit t = b.testBit t
      · have hall' : ∀ t < n + 1, a.testBit t = b.testBit t := by
          intro t ht
          rcases Nat.lt_succ_iff_lt_or_eq.mp ht with h | h
          · exact hall t h
          · exact h ▸ hab
        rw [ite_eq_left hall, ite_eq_left hall']
        ring
      · have hall' : ¬ ∀ t < n + 1, a.testBit t = b.testBit t := fun h =>
          hall fun t ht => h t (Nat.lt_succ_of_lt ht)
        rw [ite_eq_right hall, ite_eq_right hall']
        ring
    · have hs : ((if a.testBit n then (-1 : ℝ) else 1) * (if b.testBit n then -1 else 1)) = -1 := by
        cases ha : a.testBit n <;> cases hb : b.testBit n <;> simp_all
      have hall' : ¬ ∀ t < n + 1, a.testBit t = b.testBit t := fun h =>
        hab (h n (Nat.lt_succ_self n))
      rw [hs, ite_eq_right hall']
      ring

/-- The orthogonality on an index type of size `2^n`: the diagonal is `2^n`
and everything else vanishes. -/
theorem sum_walsh_mul_fin {m n : ℕ} (hm : m = 2 ^ n) (a b : Fin m) :
    ∑ c : Fin m, walsh n a.val c.val * walsh n b.val c.val = if a = b then (m : ℝ) else 0 := by
  subst hm
  have hiff : (∀ t < n, a.val.testBit t = b.val.testBit t) ↔ a = b := by
    refine ⟨fun h => Fin.ext (Nat.eq_of_testBit_eq fun t => ?_), fun h t _ => by rw [h]⟩
    rcases lt_or_ge t n with ht | ht
    · exact h t ht
    · rw [Nat.testBit_lt_two_pow (a.isLt.trans_le (Nat.pow_le_pow_right (by norm_num) ht)),
        Nat.testBit_lt_two_pow (b.isLt.trans_le (Nat.pow_le_pow_right (by norm_num) ht))]
  rw [Fin.sum_univ_eq_sum_range fun c => walsh n a.val c * walsh n b.val c,
    sum_walsh_mul_range, if_congr hiff rfl rfl]
  simp

/-- The hypothesis of `sum_walsh_mul_fin` is satisfiable, at the two index
sizes §3.2 uses: the `16` entries of an NVFP4 group, and the `2^k` groups of a
tensor. -/
example (k : ℕ) : (16 : ℕ) = 2 ^ 4 ∧ (2 : ℕ) ^ k = 2 ^ k := ⟨by norm_num, rfl⟩

end Quartet
end Transformer
