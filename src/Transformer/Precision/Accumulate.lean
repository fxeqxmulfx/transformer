/-
# A running sum in `p` significant bits is capped

The format is any one with `p` significant bits (`HasSignificand`, in
`Precision.Significand`): every IEEE binary format, bfloat16, the FP8 formats,
and the idealised formats with no overflow and no underflow alike.

Sum increments `t_k` one at a time, rounding to nearest after each addition,
`a_{k+1} = r(a_k + t_k)`, as a dot product or an attention accumulator does.
Beyond `2^{p-1} · 2^s` consecutive numbers of the format are `2^s` apart, so an
increment in `[0, 2^{s-1})` is absorbed (`accum_stall`); strictly beyond it, in
absolute value, so is an increment of either sign with `|t| < 2^{s-1}`
(`accum_stall_abs`), though not at `2^{p-1} · 2^s` itself, where the format
towards `0` can be twice as dense (`not_accum_stall_abs_le`).  A sum of
increments at most `u = 2^s` never exceeds `2^{p+1} u` however many there are
(`accum_le`).  The exact sum of `n` increments `u` is `n u`, so the computed one
is off by a factor of at least `n / 2^{p+1}`: in binary16 (`p = 11`) past
`4096` terms, in bfloat16 (`p = 8`) past `512`.

This is a property of the order of summation, not of the format.  The same
format sums `2^K` increments `2^s` exactly in a balanced tree
(`pairwise_exact`), whatever `K`, as long as `2^{s+K}` is a number of the
format.  What is fundamental is the cap on the sequential sum.

Round-to-nearest is IEEE 754-2008, §4.3.1; the analysis of recursive summation
is Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., §4.2 and
§4.3 (pairwise summation).  The cap `accum_le` in this form is this
repository's own statement.
-/

import Transformer.Precision.Nearest
import Transformer.Precision.Significand
import Mathlib.Tactic.IntervalCases

namespace Transformer
namespace Precision

variable {M : ℕ} {G : Set ℝ} {Q : ℝ → ℝ}

/-- **The accumulator stops counting.**  `a_{k+1} = r(a_k + t_k)` with `r` a
nearest rounding to a format of `M + 1` significant bits: once `a_m ≥ 2^M · 2^s`,
increments in `[0, 2^{s-1})` leave it where it is. -/
theorem accum_stall (hG : HasSignificand (M + 1) G) (hQ : IsNearest G Q) {a t : ℕ → ℝ}
    (ha : ∀ k, a (k + 1) = Q (a k + t k)) {m : ℕ} {s : ℤ} (hm : a m ∈ G)
    (hge : (2 : ℝ) ^ M * 2 ^ s ≤ a m) (ht : ∀ k, m ≤ k → 0 ≤ t k ∧ 2 * t k < 2 ^ s) :
    ∀ k, m ≤ k → a k = a m := by
  intro k hk
  induction k, hk using Nat.le_induction with
  | base => rfl
  | succ k hk ih =>
    rw [ha, ih]
    exact hQ.add_eq hm (fun z hz => hG.gap hm hz hge) (ht k hk).1 (ht k hk).2

/-- The hypotheses of `accum_stall` are satisfiable: in `{0, 1, 2, 4}`, from
`a_0 = 4 ≥ 2^0 · 2^2`, increments `1 < 2^1`. -/
example : ∃ (Q : ℝ → ℝ) (a t : ℕ → ℝ), HasSignificand (0 + 1) ({0, 1, 2, 4} : Set ℝ) ∧
    IsNearest {0, 1, 2, 4} Q ∧ (∀ k, a (k + 1) = Q (a k + t k)) ∧ a 0 ∈ ({0, 1, 2, 4} : Set ℝ) ∧
    (2 : ℝ) ^ 0 * 2 ^ (2 : ℤ) ≤ a 0 ∧ ∀ k, 0 ≤ k → 0 ≤ t k ∧ 2 * t k < 2 ^ (2 : ℤ) := by
  obtain ⟨Q, hQ⟩ := exists_isNearest (G := ({0, 1, 2, 4} : Set ℝ)) (Set.toFinite _) ⟨0, by simp⟩
  exact ⟨Q, fun k => Nat.rec 4 (fun _ ak => Q (ak + 1)) k, fun _ => 1, hasSignificand_one_pow_two,
    hQ, fun _ => rfl, by simp, by norm_num, fun _ _ => by norm_num⟩

/-- **The accumulator stops counting, whatever the sign of the increments.**
`a_{k+1} = r(a_k + t_k)` with `r` a nearest rounding to a format of `M + 1`
significant bits: once `|a_m| > 2^M · 2^s`, increments with `|t_k| < 2^{s-1}`,
of either sign, leave it where it is.

The inequality is strict, where that of `accum_stall` is not: at
`|a_m| = 2^M · 2^s` the format towards `0` can be twice as dense, and an
increment towards `0` then escapes (`not_accum_stall_abs_le`).

Round-to-nearest is IEEE 754-2008, §4.3.1; the statement is this repository's
own. -/
theorem accum_stall_abs (hG : HasSignificand (M + 1) G) (hQ : IsNearest G Q) {a t : ℕ → ℝ}
    (ha : ∀ k, a (k + 1) = Q (a k + t k)) {m : ℕ} {s : ℤ} (hm : a m ∈ G)
    (hgt : (2 : ℝ) ^ M * 2 ^ s < |a m|) (ht : ∀ k, m ≤ k → 2 * |t k| < 2 ^ s) :
    ∀ k, m ≤ k → a k = a m := by
  intro k hk
  induction k, hk using Nat.le_induction with
  | base => rfl
  | succ k hk ih =>
    rw [ha, ih]
    exact hQ.add_eq_of_abs hm (fun z hz hza => hG.le_abs_sub hm hz hgt hza) (ht k hk)

/-- The hypotheses of `accum_stall_abs` are satisfiable: in `{0, 1, 2, 4}`, from
`a_0 = 4 > 2^0 · 2^1`, increments `-1/2`. -/
example : ∃ (Q : ℝ → ℝ) (a t : ℕ → ℝ), HasSignificand (0 + 1) ({0, 1, 2, 4} : Set ℝ) ∧
    IsNearest {0, 1, 2, 4} Q ∧ (∀ k, a (k + 1) = Q (a k + t k)) ∧ a 0 ∈ ({0, 1, 2, 4} : Set ℝ) ∧
    (2 : ℝ) ^ 0 * 2 ^ (1 : ℤ) < |a 0| ∧ ∀ k, 0 ≤ k → 2 * |t k| < 2 ^ (1 : ℤ) := by
  obtain ⟨Q, hQ⟩ := exists_isNearest (G := ({0, 1, 2, 4} : Set ℝ)) (Set.toFinite _) ⟨0, by simp⟩
  exact ⟨Q, fun k => Nat.rec 4 (fun _ ak => Q (ak + -(1 / 2))) k, fun _ => -(1 / 2),
    hasSignificand_one_pow_two, hQ, fun _ => rfl, by simp, by norm_num,
    fun _ _ => by norm_num [abs_of_pos]⟩

/-- **At `|a_m| = 2^M · 2^s` an increment towards `0` can escape**, so the strict
inequality of `accum_stall_abs` cannot be relaxed to the `≤` of `accum_stall`.
`{0, 1, 2, 3, 4}` has two significant bits (`M = 1`); from
`a_0 = 4 = 2^1 · 2^1`, the increment `-9/10`, with `2 · 9/10 < 2^1`, rounds to
`3`, since below `4` the format is `1` apart, not `2`. -/
theorem not_accum_stall_abs_le : ¬ ∀ (M : ℕ) (G : Set ℝ) (Q : ℝ → ℝ) (a t : ℕ → ℝ) (m : ℕ)
    (s : ℤ), HasSignificand (M + 1) G → IsNearest G Q → (∀ k, a (k + 1) = Q (a k + t k)) →
    a m ∈ G → (2 : ℝ) ^ M * 2 ^ s ≤ |a m| → (∀ k, m ≤ k → 2 * |t k| < 2 ^ s) →
    ∀ k, m ≤ k → a k = a m := by
  intro h
  obtain ⟨Q, hQ⟩ :=
    exists_isNearest (G := ({0, 1, 2, 3, 4} : Set ℝ)) (Set.toFinite _) ⟨0, by simp⟩
  have hG : HasSignificand (1 + 1) ({0, 1, 2, 3, 4} : Set ℝ) := by
    rintro z (rfl | rfl | rfl | rfl | rfl)
    · exact ⟨0, by norm_num, 0, by norm_num⟩
    · exact ⟨1, by norm_num, 0, by norm_num⟩
    · exact ⟨1, by norm_num, 1, by norm_num⟩
    · exact ⟨3, by norm_num, 0, by norm_num⟩
    · exact ⟨1, by norm_num, 2, by norm_num⟩
  -- `3` is the only nearest point to `4 - 9/10`.
  have h3 : Q (4 + -(9 / 10)) = 3 := by
    obtain ⟨hmem, hle⟩ := hQ (4 + -(9 / 10))
    have h1 := hle 3 (by simp)
    rw [show (3 : ℝ) - (4 + -(9 / 10)) = -(1 / 10) by norm_num, abs_neg,
      abs_of_pos (show (0 : ℝ) < 1 / 10 by norm_num)] at h1
    obtain ⟨h1l, h1r⟩ := abs_le.mp h1
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hmem
    rcases hmem with h' | h' | h' | h' | h' <;> rw [h'] at h1l h1r ⊢ <;> exfalso <;> linarith
  have h4 : Q (4 + -(9 / 10)) = 4 :=
    h 1 {0, 1, 2, 3, 4} Q (fun k => Nat.rec 4 (fun _ ak => Q (ak + -(9 / 10))) k)
      (fun _ => -(9 / 10)) 0 1 hG hQ (fun _ => rfl) (by simp) (by norm_num)
      (fun _ _ => by norm_num [abs_of_pos]) 1 zero_le_one
  rw [h3] at h4
  norm_num at h4

/-- **A sequential sum in `M + 1` significant bits is capped at `2^{M+2} u`.**
With increments at most `u = 2^s`, a start at most `T = 2^{M+2} · 2^s`, and `T`
a number of the format, the running sum stays at most `T` forever, whatever
the number of terms.  Near `T` the format is `4u` apart, and a nearest
rounding of `a + t ≤ T + u` cannot reach the next number. -/
theorem accum_le (hG : HasSignificand (M + 1) G) (hQ : IsNearest G Q) {a t : ℕ → ℝ}
    (ha : ∀ k, a (k + 1) = Q (a k + t k)) {s : ℤ} (hT : (2 : ℝ) ^ (M + 2) * 2 ^ s ∈ G)
    (h0 : a 0 ≤ 2 ^ (M + 2) * 2 ^ s) (ht : ∀ k, t k ≤ 2 ^ s) :
    ∀ k, a k ≤ 2 ^ (M + 2) * 2 ^ s := by
  set T : ℝ := 2 ^ (M + 2) * 2 ^ s
  have hs : (0 : ℝ) < 2 ^ s := zpow_pos (by norm_num) _
  have hTs : (2 : ℝ) ^ M * 2 ^ (s + 2) ≤ T := le_of_eq (by
    show (2 : ℝ) ^ M * 2 ^ (s + 2) = 2 ^ (M + 2) * 2 ^ s
    rw [zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0), pow_add]; norm_num; ring)
  have hgap : ∀ z ∈ G, T < z → T + 4 * 2 ^ s ≤ z := fun z hz h => by
    have := hG.gap hT hz hTs h
    rwa [zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0), show (2 : ℝ) ^ (2 : ℤ) = 4 by norm_num,
      mul_comm] at this
  intro k
  induction k with
  | zero => exact h0
  | succ k ih =>
    rw [ha]
    set x := a k + t k
    have hx : x ≤ T + 2 ^ s := by linarith [ht k]
    have hnear := (hQ x).2 T hT
    by_contra! hlt
    have hz := hgap _ (hQ x).1 hlt
    have h1 := le_abs_self (Q x - x)
    rcases abs_cases (T - x) with ⟨h2, _⟩ | ⟨h2, _⟩ <;> linarith

/-- The hypotheses of `accum_le` are satisfiable: in `{0, 1, 2, 4}`, from
`a_0 = 0`, increments `1`, cap `2^2 · 2^0 = 4`. -/
example : ∃ (Q : ℝ → ℝ) (a t : ℕ → ℝ), HasSignificand (0 + 1) ({0, 1, 2, 4} : Set ℝ) ∧
    IsNearest {0, 1, 2, 4} Q ∧ (∀ k, a (k + 1) = Q (a k + t k)) ∧
    (2 : ℝ) ^ (0 + 2) * 2 ^ (0 : ℤ) ∈ ({0, 1, 2, 4} : Set ℝ) ∧
    a 0 ≤ 2 ^ (0 + 2) * 2 ^ (0 : ℤ) ∧ ∀ k, t k ≤ 2 ^ (0 : ℤ) := by
  obtain ⟨Q, hQ⟩ := exists_isNearest (G := ({0, 1, 2, 4} : Set ℝ)) (Set.toFinite _) ⟨0, by simp⟩
  exact ⟨Q, fun k => Nat.rec 0 (fun _ ak => Q (ak + 1)) k, fun _ => 1, hasSignificand_one_pow_two,
    hQ, fun _ => rfl, by norm_num, by norm_num, fun _ => by norm_num⟩

/-- **Pairwise summation does not stall.**  Summing `2^K` copies of `2^s` in a
balanced tree, `b_{K+1} = r(b_K + b_K)`, is exact at every level whose value is
a number of the format. -/
theorem pairwise_exact (hQ : IsNearest G Q) {s : ℤ} {K₀ : ℕ} (hG : ∀ K ≤ K₀, (2 : ℝ) ^ (s + K) ∈ G)
    {b : ℕ → ℝ} (h0 : b 0 = 2 ^ s) (hb : ∀ K, b (K + 1) = Q (b K + b K)) :
    ∀ K ≤ K₀, b K = 2 ^ (s + K) := by
  intro K hK
  induction K with
  | zero => simpa using h0
  | succ K ih =>
    rw [hb, ih (by omega), ← two_mul, ← zpow_one_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    push_cast
    rw [show 1 + (s + K) = s + (K + 1) by ring]
    exact hQ.eq_self (by exact_mod_cast hG (K + 1) hK)

/-- The hypotheses of `pairwise_exact` are satisfiable: `1, 2, 4` in
`{0, 1, 2, 4}`. -/
example : ∃ (Q : ℝ → ℝ) (b : ℕ → ℝ), IsNearest {0, 1, 2, 4} Q ∧
    (∀ K : ℕ, K ≤ 2 → (2 : ℝ) ^ ((0 : ℤ) + K) ∈ ({0, 1, 2, 4} : Set ℝ)) ∧ b 0 = 2 ^ (0 : ℤ) ∧
    ∀ K, b (K + 1) = Q (b K + b K) := by
  obtain ⟨Q, hQ⟩ := exists_isNearest (G := ({0, 1, 2, 4} : Set ℝ)) (Set.toFinite _) ⟨0, by simp⟩
  refine ⟨Q, fun K => Nat.rec 1 (fun _ bK => Q (bK + bK)) K, hQ, fun K hK => ?_, by norm_num,
    fun _ => rfl⟩
  interval_cases K <;> norm_num

end Precision
end Transformer
