/-
# The cumulative sum comes back, but not exactly

The construction recovers a running total by reading a uniform average and
multiplying it back up: the head returns `vsum · (1/count)`
(`hull2d_cht.h`, `HullMeta::resolve`) and the FFN multiplies by `position`
(`graph/core.py`, `fetch_sum`), so what the machine computes is
`fl(fl(s · fl(1/p)) · p)` and what it wants is `s`.  Those are not the same
number.  `todo3.md` §8 measures the round trip failing on 25.8 % of
byte-valued payloads, the first at `s = 3`, `p = 5`, and §8a measures the
consequence on the running machine: 2.8 % to 7.1 % of the hard-attention
queries of the six reference programs are not integers, the worst off by two
units in the last place.

Two things are proved here.  `recip_isBinary_iff`: the reciprocal the head
multiplies by is representable only when the position is a power of two, so
for three positions out of four the average is already inexact before the FFN
touches it — no rounding mode and no precision changes that.  And
`round_trip_drift`: what comes back is within `(2u + u²)|s|` of the total, and
that is all that can be claimed of it.

The division is not a mistake.  A cumulative sum by uniform attention is the
only constant-depth route to one — the recurrence `c(p) = c(p-1) + δ(p)` needs
the previous position's *computed* value, and a layer can only read the layer
below, so it costs one layer per token.  The average is forced, the division is
forced, and the inexactness is forced with it.  So exactness is not the thing
to prove on this path: `Transformer.ALM.DriftMargin` proves what is true
instead, that the retrieval survives the drift with a margin, and where it does
not.

Source: `todo3.md` §8 and §8a; `transformer_vm/wasm/interpreter.py:316-320`.
-/

import Transformer.ALM.ScoreWall

namespace Transformer
namespace ALM

/-! ### The reciprocal is a float only at powers of two -/

/-- **`1/p` is representable exactly when `p` is a power of two.**  `resolve`
divides by the count once and hands the quotient to the FFN, so at every
position that is not a power of two the average the machine multiplies back up
is already the wrong number — whatever the precision, whatever the rounding.
Source: `hull2d_cht.h`, `HullMeta::resolve`. -/
theorem recip_isBinary_iff {prec p : ℕ} (hprec : 0 < prec) (hp : 0 < p) :
    (∃ a : ℝ, IsBinary prec a ∧ a * p = 1) ↔ ∃ j : ℕ, p = 2 ^ j := by
  constructor
  · rintro ⟨a, ⟨m, e, _, rfl⟩, ha⟩
    have hpR : (0 : ℝ) < (p : ℝ) := by exact_mod_cast hp
    have hmp : ((m * p : ℤ) : ℝ) = (2 : ℝ) ^ (-e) := by
      have h2 : (2 : ℝ) ^ e ≠ 0 := (zpow_pos (by norm_num) e).ne'
      push_cast
      rw [zpow_neg]
      field_simp
      linear_combination ha
    have hne : (m * p : ℤ) ≠ 0 := by
      intro h
      rw [h] at hmp
      have hpos : (0 : ℝ) < (2 : ℝ) ^ (-e) := zpow_pos (by norm_num) (-e)
      rw [← hmp] at hpos
      simp at hpos
    have hnonneg : 0 ≤ -e := by
      by_contra hc
      have hlt : (2 : ℝ) ^ (-e) < 1 := zpow_lt_one_of_neg₀ (by norm_num) (by omega)
      have h1 : (1 : ℝ) ≤ |((m * p : ℤ) : ℝ)| := by
        rw [← Int.cast_abs]
        exact_mod_cast Int.one_le_abs hne
      rw [hmp, abs_of_pos (by positivity)] at h1
      linarith
    obtain ⟨n, hn⟩ : ∃ n : ℕ, -e = (n : ℤ) := ⟨(-e).toNat, by omega⟩
    rw [hn, zpow_natCast] at hmp
    have hmpZ : m * p = 2 ^ n := by exact_mod_cast hmp
    have hdvd : (p : ℤ) ∣ 2 ^ n := ⟨m, by linarith [hmpZ]⟩
    have hdvdN : p ∣ 2 ^ n := by
      have : ((p : ℤ)) ∣ ((2 ^ n : ℕ) : ℤ) := by push_cast; exact hdvd
      exact_mod_cast this
    obtain ⟨j, _, hj⟩ := (Nat.dvd_prime_pow Nat.prime_two).mp hdvdN
    exact ⟨j, hj⟩
  · rintro ⟨j, rfl⟩
    refine ⟨(2 : ℝ) ^ (-(j : ℤ)), ⟨1, -(j : ℤ), ?_, by norm_num⟩, ?_⟩
    · calc |(1 : ℤ)| = 2 ^ 0 := by norm_num
        _ < 2 ^ prec := by exact_mod_cast Nat.one_lt_two_pow_iff.mpr (by omega)
    · push_cast
      rw [← zpow_natCast (2 : ℝ) j, ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
      simp

/-! ### What the round trip does to the total -/

/-- **The total comes back within a relative `2u`.**  The head's division and
the FFN's multiplication each land within a relative `u` of their exact value,
so the recovered cumulative sum is within `(2u + u²)|s|` of `s` — not equal to
it, and this is the whole of what `fetch_sum` guarantees.  `todo3.md` §8. -/
theorem round_trip_drift {s p a r u : ℝ} (hp : 0 < p) (hu : 0 ≤ u)
    (ha : |a - s / p| ≤ u * |s / p|) (hr : |r - a * p| ≤ u * |a * p|) :
    |r - s| ≤ (2 * u + u ^ 2) * |s| := by
  have habs : |s / p| = |s| / p := by rw [abs_div, abs_of_pos hp]
  have hmid : |a * p - s| ≤ u * |s| := by
    have heq : a * p - s = (a - s / p) * p := by field_simp
    rw [heq, abs_mul, abs_of_pos hp]
    calc |a - s / p| * p ≤ (u * (|s| / p)) * p :=
          mul_le_mul_of_nonneg_right (habs ▸ ha) hp.le
      _ = u * |s| := by field_simp
  have hap : |a * p| ≤ (1 + u) * |s| := by
    have := abs_sub_abs_le_abs_sub (a * p) s
    nlinarith [abs_nonneg s, abs_nonneg (a * p)]
  calc |r - s| ≤ |r - a * p| + |a * p - s| := abs_sub_le _ _ _
    _ ≤ u * ((1 + u) * |s|) + u * |s| := by
        exact add_le_add (le_trans hr (mul_le_mul_of_nonneg_left hap hu)) hmid
    _ = (2 * u + u ^ 2) * |s| := by ring

/-- **And the query inherits it linearly.**  `wasm/interpreter.py:320` forms
the lookup query as `5·cursor + 1`, and the hat filter offsets it again, so a
counter recovered `ε` off its integer value produces a query `5ε` off its
integer value.  Nothing damps it. -/
lemma affine_drift (a b : ℝ) {c c₀ ε : ℝ} (h : |c - c₀| ≤ ε) :
    |(a * c + b) - (a * c₀ + b)| ≤ |a| * ε := by
  have : a * c + b - (a * c₀ + b) = a * (c - c₀) := by ring
  rw [this, abs_mul]
  exact mul_le_mul_of_nonneg_left h (abs_nonneg a)

/-! ### The hypotheses are satisfiable -/

/-- At `p = 4` the reciprocal is a float.  At `p = 5` — `todo3.md` §8's first
failing round trip, `s = 3`, `p = 5`, which comes back as
`3.0000000000000004` — it is not, at any precision. -/
example :
    (∃ a : ℝ, IsBinary 53 a ∧ a * 4 = 1) ∧ ¬ ∃ a : ℝ, IsBinary 53 a ∧ a * 5 = 1 := by
  constructor
  · exact (recip_isBinary_iff (by norm_num) (by norm_num)).mpr ⟨2, by norm_num⟩
  · intro h
    obtain ⟨j, hj⟩ := (recip_isBinary_iff (by norm_num) (by norm_num)).mp h
    have h3 : j < 3 := by
      by_contra hc
      have h8 : (2 : ℕ) ^ 3 ≤ 2 ^ j := Nat.pow_le_pow_right (by norm_num) (by omega)
      rw [← hj] at h8
      norm_num at h8
    interval_cases j <;> norm_num at hj

/-- The round trip's hypotheses are satisfiable at the working precision, and
not only by exact arithmetic: `u = 2⁻⁵³`, the payload `3` over `5` positions,
with both operations rounded to their exact values. -/
example :
    (0 : ℝ) < 5 ∧ (0 : ℝ) ≤ (2 : ℝ) ^ (-53 : ℤ) ∧
      |(3 : ℝ) / 5 - 3 / 5| ≤ (2 : ℝ) ^ (-53 : ℤ) * |(3 : ℝ) / 5| ∧
      |(3 : ℝ) - 3 / 5 * 5| ≤ (2 : ℝ) ^ (-53 : ℤ) * |(3 : ℝ) / 5 * 5| := by
  norm_num

end ALM
end Transformer
