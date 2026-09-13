/-
# On the lattice the search needs no separation at all

`Transformer.ALM.FloatIndex` proves the floating-point search right under a
separation hypothesis: the query must clear every breakpoint by more than the
rounding error.  That hypothesis fails in exactly one situation — when the
query sits *on* a breakpoint — and that situation is not exotic: it is the tie
the two `while` loops of `HullHalf::query` exist to handle.  So the theorem
was silent precisely where the code does something interesting.

Integer data removes the hypothesis.  The `LookUp` primitive stores integer
keys (`Transformer.ALM.ScalarInt`), and the breakpoint of two of them is a
multiple of `1/2`.  An integer query is therefore either exactly on a
breakpoint or at least `1/2` away from it, and `1/2` is far more than the
rounding error of `isect` (`Transformer.ALM.IntGrid` has that arithmetic).
Where the comparison could go either way the two keys score the same, which is
what the weakened `hlt` of `Transformer.ALM.Query` now allows.

`fpSearch_isGreatest_of_int` is the statement: for integer keys and an integer
query, the search the implementation runs returns an exact argmax, with no
hypothesis on the query beyond its being an integer.
`fpProbe_mem_argmaxSet_of_int` puts that answer in the tie set
`Transformer.ALM.HullScan` walks, which is what the walk's own correctness
starts from.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 60-70 (`isect`) and
203-215 (`argmax`).
-/

import Transformer.ALM.IntGrid
import Transformer.ALM.FloatIndex
import Transformer.ALM.HullScan

namespace Transformer
namespace ALM

variable {n : ℕ}

/-! ### The search, with the separation hypothesis gone -/

/-- **The running search is exact on integer data.**  Keys and query integers,
keys sorted, the computed breakpoints accurate to better than `1/2` on the
window searched: then the index `fpSearch` returns maximizes the paraboloid
score over the whole window.  No hypothesis separates the query from the
breakpoints — at a breakpoint the comparison may round either way, and both
ways are right, because the two keys there score the same.

Source: `hull2d_cht.h`, lines 203-215. -/
theorem fpSearch_isGreatest_of_int (F : FPArith) (K : ℕ → ℝ) (q : ℤ) (N : ℕ) (M : ℝ)
    (hstep : ∀ j, K j < K (j + 1)) (hKint : ∀ j, ∃ z : ℤ, K j = (z : ℝ))
    (hbd : ∀ j ≤ N, |K j| ≤ M) (hu : F.u * M < 1 / 2) :
    ∀ j ≤ N, lineEval (liftKey (K j)) (q : ℝ)
      ≤ lineEval (liftKey (K (fpSearch F K (q : ℝ) N))) (q : ℝ) := by
  have hM : 0 ≤ M := le_trans (abs_nonneg _) (hbd 0 (Nat.zero_le N))
  have hu0 : 0 ≤ F.u * M := mul_nonneg F.u_nonneg hM
  have herr : ∀ j, j < N → |F.isect (liftKey (K j)) (liftKey (K (j + 1)))
      - (K j + K (j + 1)) / 2| ≤ F.u * M :=
    fun j hj => isect_liftKey_error F (ne_of_lt (hstep j)) (hbd j hj.le) (hbd (j + 1) hj)
  have hup : ∀ j, j < N → F.isect (liftKey (K j)) (liftKey (K (j + 1)))
      ≤ (K j + K (j + 1)) / 2 + F.u * M := by
    intro j hj
    have := (abs_le.mp (herr j hj)).2
    linarith
  have hlow : ∀ j, j < N → (K j + K (j + 1)) / 2 - F.u * M
      ≤ F.isect (liftKey (K j)) (liftKey (K (j + 1))) := by
    intro j hj
    have := (abs_le.mp (herr j hj)).1
    linarith
  set p : ℕ → Bool := fun j =>
    if j < N then decide ((q : ℝ) ≤ F.isect (liftKey (K j)) (liftKey (K (j + 1))))
    else decide ((q : ℝ) ≤ (K j + K (j + 1)) / 2) with hp
  have hpin : ∀ j, j < N →
      (p j = true ↔ (q : ℝ) ≤ F.isect (liftKey (K j)) (liftKey (K (j + 1)))) := by
    intro j hj
    simp only [hp, if_pos hj, decide_eq_true_eq]
  have hpout : ∀ j, ¬ j < N → (p j = true ↔ (q : ℝ) ≤ (K j + K (j + 1)) / 2) := by
    intro j hj
    simp only [hp, if_neg hj, decide_eq_true_eq]
  have hcongr : fpSearch F K (q : ℝ) N = bsearch p 0 N := by
    unfold fpSearch
    refine bsearch_congr _ _ 0 N (fun j _ hj => ?_)
    rw [Nat.zero_add] at hj
    simp only [hp, if_pos hj]
  have hmono : ∀ a b, a ≤ b → p a = true → p b = true := by
    intro a b hab ha
    rcases Nat.eq_or_lt_of_le hab with rfl | hlt
    · exact ha
    have hmid := mid_add_one_le hstep hKint hlt
    by_cases hbN : b < N
    · have haN : a < N := by omega
      have h1 := hup a haN
      have h2 := hlow b hbN
      have ha' := (hpin a haN).mp ha
      exact (hpin b hbN).mpr (by linarith)
    · by_cases haN : a < N
      · have h1 := hup a haN
        have ha' := (hpin a haN).mp ha
        exact (hpout b hbN).mpr (by linarith)
      · have ha' := (hpout a haN).mp ha
        exact (hpout b hbN).mpr (by linarith)
  have hile : bsearch p 0 N ≤ N := by simpa using bsearch_le p 0 N
  have hlt : ∀ j, j < bsearch p 0 N → (K j + K (j + 1)) / 2 ≤ (q : ℝ) := by
    intro j hj
    have hfalse := bsearch_lt p hmono 0 N j (Nat.zero_le j) hj
    have hnot : ¬ (p j = true) := by rw [hfalse]; simp
    by_cases hjN : j < N
    · have hgt : F.isect (liftKey (K j)) (liftKey (K (j + 1))) < (q : ℝ) :=
        not_le.mp fun h => hnot ((hpin j hjN).mpr h)
      obtain ⟨z, hz⟩ := mid_half_int hKint j
      have hb := hlow j hjN
      rw [hz] at hb ⊢
      exact half_int_le_of_lt_add_half (by linarith)
    · have hgt : (K j + K (j + 1)) / 2 < (q : ℝ) :=
        not_le.mp fun h => hnot ((hpout j hjN).mpr h)
      linarith
  have hge : bsearch p 0 N < N →
      (q : ℝ) ≤ (K (bsearch p 0 N) + K (bsearch p 0 N + 1)) / 2 := by
    intro hin
    have htrue := (hpin _ hin).mp (bsearch_ge_of_lt p 0 N (by simpa using hin))
    obtain ⟨z, hz⟩ := mid_half_int hKint (bsearch p 0 N)
    have hb := hup _ hin
    rw [hz] at hb ⊢
    exact le_half_int_of_lt_add_half (by linarith)
  intro j hj
  rw [hcongr]
  exact hull_isGreatest K (q : ℝ) (fun k _ => hstep k) (bsearch p 0 N) hile hlt hge j hj

/-- The hypotheses hold exactly where the old ones failed: the keys `0, 2` and
the query `1`, where the breakpoint *is* the query, so the separation
hypothesis of `fpSearch_eq_of_bounded` is false while these hold. -/
example :
    (∀ j : ℕ, (2 * (j : ℝ)) < 2 * ((j + 1 : ℕ) : ℝ)) ∧
      (∀ j : ℕ, ∃ z : ℤ, 2 * (j : ℝ) = (z : ℝ)) ∧
      (∀ j : ℕ, j ≤ 1 → |2 * (j : ℝ)| ≤ 2) ∧ exactArith.u * 2 < 1 / 2 ∧
      ¬ (exactArith.u * 2 < |((1 : ℤ) : ℝ) - (2 * ((0 : ℕ) : ℝ) + 2 * ((1 : ℕ) : ℝ)) / 2|) := by
  refine ⟨fun j => by push_cast; linarith, fun j => ⟨2 * (j : ℤ), by push_cast; ring⟩,
    fun j hj => ?_, by norm_num [exactArith], by norm_num [exactArith]⟩
  have hle : (j : ℝ) ≤ 1 := by exact_mod_cast hj
  rw [abs_of_nonneg (mul_nonneg (by norm_num) (Nat.cast_nonneg j))]
  linarith

/-! ### And the answer is in the tie set the walk starts from -/

/-- Every entry of the sorted key sequence of an integer family is an integer,
the unit continuation past the last distinct key included. -/
lemma sortedKey_int [Nonempty (Fin n)] (K : Fin n → ℝ) (hK : ∀ i, ∃ z : ℤ, K i = (z : ℝ))
    (j : ℕ) : ∃ z : ℤ, sortedKey K j = (z : ℝ) := by
  by_cases hj : j ≤ keyCard K - 1
  · obtain ⟨i, hi⟩ := exists_eq_sortedKey K hj
    obtain ⟨z, hz⟩ := hK i
    exact ⟨z, by rw [← hi, hz]⟩
  · have hge : keyCard K - 1 ≤ j := by omega
    obtain ⟨i, hi⟩ := exists_eq_sortedKey K (le_refl (keyCard K - 1))
    obtain ⟨z, hz⟩ := hK i
    refine ⟨z + ((j - (keyCard K - 1) : ℕ) : ℤ), ?_⟩
    rw [sortedKey_of_ge K hge, ← sortedKey_of_le K (le_refl (keyCard K - 1)), ← hi, hz]
    push_cast
    ring

/-- **The probe the running code lands on is one of the winners.**  For
integer keys and an integer query the floating-point probe is a member of the
tie set `argmaxSet`, whatever the rounding does at a tie.  This is the
hypothesis the merge walk of `Transformer.ALM.HullScan` needs, now supplied by
the arithmetic that actually runs rather than assumed. -/
theorem fpProbe_mem_argmaxSet_of_int [Nonempty (Fin n)] (F : FPArith) (K : Fin n → ℝ)
    (hK : ∀ i, ∃ z : ℤ, K i = (z : ℝ)) (q : ℤ) (M : ℝ)
    (hbd : ∀ j ≤ keyCard K - 1, |sortedKey K j| ≤ M) (hu : F.u * M < 1 / 2) :
    fpProbe F K (q : ℝ) ∈ argmaxSet (sortedKey K) (q : ℝ) (keyCard K - 1) := by
  refine (mem_argmaxSet _ _ _).mpr ⟨fpProbe_le F K (q : ℝ), fun i hi => ?_⟩
  exact fpSearch_isGreatest_of_int F (sortedKey K) q (keyCard K - 1) M
    (sortedKey_lt_succ K) (sortedKey_int K hK) hbd hu i hi

/-- The hypotheses are satisfiable for every integer family: finitely many
sorted keys are bounded, and exact arithmetic clears the margin. -/
example [Nonempty (Fin n)] (K : Fin n → ℤ) :
    (∀ i, ∃ z : ℤ, ((K i : ℝ)) = (z : ℝ)) ∧
      ∃ M : ℝ, (∀ j ≤ keyCard (fun i => (K i : ℝ)) - 1,
          |sortedKey (fun i => (K i : ℝ)) j| ≤ M) ∧ exactArith.u * M < 1 / 2 := by
  obtain ⟨B, hB⟩ := exists_bound_sortedKey (fun i => (K i : ℝ))
  exact ⟨fun i => ⟨K i, rfl⟩, B, hB, by simp [exactArith]⟩

end ALM
end Transformer
