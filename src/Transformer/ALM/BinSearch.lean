/-
# What the query actually costs

`Transformer.ALM.Hull` proves that the index `lower_bound(q)` returns is an
argmax of the score, but `lower_bound` was a hypothesis there: the index `i`
was *described* by `hlt` and `hge`, not computed.  `Transformer.ALM.LookupIndex`
has the matching gap on the cost side, where `NNIndex.query` is declared data.

This file removes both.  `bsearch` is `std::lower_bound` written out —
halve, compare once, recurse — and `bcount` counts the comparisons it makes
on the same recursion.  Then

* `bsearch_lt` and `bsearch_ge_of_lt` are exactly the `hlt` and `hge` that
  `hull_isGreatest` asks for, so `hull_bsearch_isGreatest` needs no
  hypothesis about the returned index at all;
* `bcount_le_log` proves `bcount len ≤ log₂ len + 1`.

So the logarithmic query cost of the planar hull is a theorem about the
procedure that runs, not a number declared in a structure field.

The procedure that runs *in C++*, that is.  `alm-hull/src/tree.rs` keeps the
envelope in a red-black tree rather than a sorted array, and pays
`2·log₂(n + 1)` for the same query; `Transformer.ALM.TreeQuery` is that
descent and its price, and `TreeQuery.log_succ_bound` is the factor of two
between the two containers.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 203-215 (`argmax`
calls `lower_bound` once); the comparison count is the standard bound
`⌈log₂ (len + 1)⌉` for `std::lower_bound`.
-/

import Transformer.ALM.Hull

namespace Transformer
namespace ALM

/-! ### The search, and its comparison count -/

/-- **`std::lower_bound`.**  The first index of `[lo, lo + len)` at which `p`
holds, or `lo + len` if none does.  Each step tests the midpoint once and
keeps the half that can still contain the answer. -/
def bsearch (p : ℕ → Bool) (lo len : ℕ) : ℕ :=
  match len with
  | 0 => lo
  | m + 1 =>
      if p (lo + m / 2) then bsearch p lo (m / 2)
      else bsearch p (lo + m / 2 + 1) (m - m / 2)
termination_by len

/-- The number of comparisons `bsearch` makes in the worst case, on the same
recursion: one per step, down whichever half is deeper. -/
def bcount (len : ℕ) : ℕ :=
  match len with
  | 0 => 0
  | m + 1 => 1 + max (bcount (m / 2)) (bcount (m - m / 2))
termination_by len

@[simp] lemma bsearch_zero (p : ℕ → Bool) (lo : ℕ) : bsearch p lo 0 = lo := by
  rw [bsearch]

lemma bsearch_succ (p : ℕ → Bool) (lo m : ℕ) :
    bsearch p lo (m + 1) =
      if p (lo + m / 2) then bsearch p lo (m / 2)
      else bsearch p (lo + m / 2 + 1) (m - m / 2) := by
  rw [bsearch]

@[simp] lemma bcount_zero : bcount 0 = 0 := by rw [bcount]

lemma bcount_succ (m : ℕ) :
    bcount (m + 1) = 1 + max (bcount (m / 2)) (bcount (m - m / 2)) := by
  rw [bcount]

/-! ### The search returns a `lower_bound` -/

/-- The answer never leaves the range on the left. -/
theorem le_bsearch (p : ℕ → Bool) (lo len : ℕ) : lo ≤ bsearch p lo len := by
  induction lo, len using bsearch.induct p with
  | case1 lo => simp
  | case2 lo m _ ih => rw [bsearch_succ, ite_eq_left ‹_›]; exact ih
  | case3 lo m _ ih => rw [bsearch_succ, ite_eq_right ‹_›]; omega

/-- Nor on the right: `lo + len` is the `it == end` fallback. -/
theorem bsearch_le (p : ℕ → Bool) (lo len : ℕ) : bsearch p lo len ≤ lo + len := by
  induction lo, len using bsearch.induct p with
  | case1 lo => simp
  | case2 lo m _ ih => rw [bsearch_succ, ite_eq_left ‹_›]; omega
  | case3 lo m _ ih => rw [bsearch_succ, ite_eq_right ‹_›]; omega

/-- **The search only looks inside its window.**  Two predicates that agree
on `[lo, lo + len)` send the search down the same path, so the answer depends
on nothing outside the range being searched.  `Transformer.ALM.FloatIndex`
needs exactly this: the floating-point comparisons agree with the exact ones
only where the keys are, and that is enough. -/
theorem bsearch_congr (p p' : ℕ → Bool) (lo len : ℕ) :
    (∀ j, lo ≤ j → j < lo + len → p j = p' j) →
      bsearch p lo len = bsearch p' lo len := by
  induction lo, len using bsearch.induct p with
  | case1 lo => intro _; simp
  | case2 lo m htrue ih =>
      intro h
      have hm : p (lo + m / 2) = p' (lo + m / 2) := h _ (by omega) (by omega)
      rw [bsearch_succ, bsearch_succ, ite_eq_left htrue, ite_eq_left (hm ▸ htrue)]
      exact ih (fun j hj hlt => h j hj (by omega))
  | case3 lo m hfalse ih =>
      intro h
      have hm : p (lo + m / 2) = p' (lo + m / 2) := h _ (by omega) (by omega)
      rw [bsearch_succ, bsearch_succ, ite_eq_right hfalse, ite_eq_right (hm ▸ hfalse)]
      exact ih (fun j hj hlt => h j (by omega) (by omega))

/-- **Nothing before the answer satisfies `p`.**  This is the `hlt`
hypothesis of `lowerBound_isGreatest`.  Monotonicity of `p` is used exactly
once, to discard the half the search skipped. -/
theorem bsearch_lt (p : ℕ → Bool) (hp : ∀ a b, a ≤ b → p a = true → p b = true)
    (lo len : ℕ) : ∀ j, lo ≤ j → j < bsearch p lo len → p j = false := by
  induction lo, len using bsearch.induct p with
  | case1 lo => intro j hj hlt; simp at hlt; omega
  | case2 lo m _ ih =>
      intro j hj hlt
      rw [bsearch_succ, ite_eq_left ‹_›] at hlt
      exact ih j hj hlt
  | case3 lo m hfalse ih =>
      intro j hj hlt
      rw [bsearch_succ, ite_eq_right hfalse] at hlt
      rcases Nat.lt_or_ge j (lo + m / 2 + 1) with hlow | hge
      · rcases Nat.lt_or_ge j (lo + m / 2) with hj' | hj'
        · by_contra hcon
          exact hfalse (hp j (lo + m / 2) hj'.le (by simpa using hcon))
        · have : j = lo + m / 2 := by omega
          rw [this]
          simpa using hfalse
      · exact ih j hge hlt

/-- **The answer itself satisfies `p`, unless it is the fallback.**  This is
the `hge` hypothesis of `lowerBound_isGreatest`; it needs no monotonicity. -/
theorem bsearch_ge_of_lt (p : ℕ → Bool) (lo len : ℕ) :
    bsearch p lo len < lo + len → p (bsearch p lo len) = true := by
  induction lo, len using bsearch.induct p with
  | case1 lo => intro h; simp at h
  | case2 lo m htrue ih =>
      intro _
      rw [bsearch_succ, ite_eq_left htrue]
      rcases Nat.lt_or_ge (bsearch p lo (m / 2)) (lo + m / 2) with h | h
      · exact ih h
      · have : bsearch p lo (m / 2) = lo + m / 2 :=
          le_antisymm (bsearch_le p lo (m / 2)) h
        rw [this]; exact htrue
  | case3 lo m hfalse ih =>
      intro h
      rw [bsearch_succ, ite_eq_right hfalse] at h ⊢
      exact ih (by omega)

/-! ### The comparison count is logarithmic -/

/-- **`log₂ len + 1` comparisons suffice.**  Each step halves the range, and
the two halves of `len + 1` have sizes `len / 2` and `len - len / 2`, both at
most `(len + 1) / 2`. -/
theorem bcount_le_log (len : ℕ) : bcount len ≤ Nat.log 2 len + 1 := by
  induction len using Nat.strong_induction_on with
  | _ len ih =>
    match len with
    | 0 => simp
    | m + 1 =>
      have key : ∀ k : ℕ, 2 * k ≤ m + 1 → k < m + 1 → bcount k ≤ Nat.log 2 (m + 1) := by
        intro k hk hlt
        rcases Nat.eq_zero_or_pos k with rfl | hk0
        · simp
        · have hstep : Nat.log 2 k + 1 = Nat.log 2 (k * 2) :=
            (Nat.log_mul_base (by norm_num) (by omega)).symm
          calc bcount k ≤ Nat.log 2 k + 1 := ih k hlt
            _ = Nat.log 2 (k * 2) := hstep
            _ ≤ Nat.log 2 (m + 1) := Nat.log_mono_right (by omega)
      rw [bcount_succ]
      have h1 := key (m / 2) (by omega) (by omega)
      have h2 := key (m - m / 2) (by omega) (by omega)
      omega

/-! ### The hull query, with no hypothesis about the index -/

/-- **The planar lookup, computed and paid for.**  For a strictly increasing
key sequence the midpoints are increasing, so the `q ≤ midpoint` test is
monotone and `bsearch` applies.  The key it returns maximizes the paraboloid
score over the first `n` keys, and reaching it took at most `log₂ n + 1`
comparisons. -/
theorem hull_bsearch_isGreatest (K : ℕ → ℝ) (q : ℝ) (n : ℕ)
    (hstep : ∀ j, K j < K (j + 1)) :
    let p : ℕ → Bool := fun j => decide (q ≤ (K j + K (j + 1)) / 2)
    (∀ j ≤ n, lineEval (liftKey (K j)) q
        ≤ lineEval (liftKey (K (bsearch p 0 n))) q) ∧ bcount n ≤ Nat.log 2 n + 1 := by
  intro p
  have hmid : ∀ a b : ℕ, a ≤ b → (K a + K (a + 1)) / 2 ≤ (K b + K (b + 1)) / 2 := by
    intro a b hab
    have h1 : K a ≤ K b := le_of_step_lt (n := b) (fun j _ => hstep j) hab le_rfl
    have h2 : K (a + 1) ≤ K (b + 1) :=
      le_of_step_lt (n := b + 1) (fun j _ => hstep j) (by omega) le_rfl
    linarith
  have hp : ∀ a b, a ≤ b → p a = true → p b = true := by
    intro a b hab ha
    have := hmid a b hab
    simp only [p, decide_eq_true_eq] at ha ⊢
    linarith
  refine ⟨hull_isGreatest K q (fun j _ => hstep j) (bsearch p 0 n)
    (by simpa using bsearch_le p 0 n) (fun j hj => ?_) (fun hin => ?_), bcount_le_log n⟩
  · have hfalse := bsearch_lt p hp 0 n j (Nat.zero_le j) hj
    simp only [p, decide_eq_false_iff_not, not_le] at hfalse
    exact hfalse.le
  · have := bsearch_ge_of_lt p 0 n (by simpa using hin)
    simpa [p] using this

/-- The hypotheses are satisfiable and the search is not trivial: on the keys
`K j = j` with `q = 2.5` the search over `[0, 4)` lands on index `2`, having
made at most `log₂ 4 + 1 = 3` comparisons. -/
example :
    let K : ℕ → ℝ := fun j => (j : ℝ)
    let p : ℕ → Bool := fun j => decide ((2.5 : ℝ) ≤ (K j + K (j + 1)) / 2)
    bsearch p 0 4 = 2 ∧ bcount 4 ≤ Nat.log 2 4 + 1 := by
  refine ⟨?_, bcount_le_log 4⟩
  norm_num [bsearch_succ, bsearch_zero]

end ALM
end Transformer
