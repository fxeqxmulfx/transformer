/-
# When tie-breaking matters

A hull line in `transformer_vm/attention/hull2d_cht.h` carries a `HullMeta`
(lines 41-84): the running sum, count, last value and largest sequence number
of every stored value that landed on that line.  `HullHalf::query` (lines
252-310) walks left and right from the argmax, merges the metas of every line
whose score equals the best one, and calls `resolve` on the result.

Two questions the file leaves open, and this module answers:

* **Is that path reachable on the lookup path?**  No, when the query is one of
  the stored keys: `argmax_unique_of_query_mem`.  It is reachable otherwise,
  and `sScore_eq_iff` says exactly where — the tie locus of two integer keys
  is the single query `2q = k₁ + k₂`.
* **Do the two tie-break modes agree?**  The file comment (lines 26-28) says
  `TieBreak::LATEST` "is accepted but behaves like AVERAGE", but `resolve`
  (lines 71-84) branches on the mode and returns `vlast` for `LATEST`.  They
  agree on an unmerged line (`resolve_eq_of_single`) and, as the final example
  records, differ as soon as a tie merges two distinct values.

The scores are `Transformer.ALM.sScore` from `Transformer.ALM.Defs`.
-/

import Transformer.ALM.Basic

namespace Transformer
namespace ALM

/-! ### Where ties live -/

/-- **The tie locus of two integer keys.**  Distinct keys score equally at
exactly one query, their midpoint. -/
theorem sScore_eq_iff (q k₁ k₂ : ℤ) (hne : k₁ ≠ k₂) :
    sScore q k₁ = sScore q k₂ ↔ 2 * q = k₁ + k₂ := by
  have hc : ((k₁ : ℝ) - (k₂ : ℝ)) ≠ 0 := by
    rw [sub_ne_zero]
    exact_mod_cast hne
  unfold sScore
  constructor
  · intro h
    have key : ((k₁ : ℝ) - k₂) * (2 * (q : ℝ)) = ((k₁ : ℝ) - k₂) * ((k₁ : ℝ) + k₂) := by
      linear_combination h
    have := mul_left_cancel₀ hc key
    exact_mod_cast this
  · intro h
    have hr : 2 * (q : ℝ) = (k₁ : ℝ) + (k₂ : ℝ) := by exact_mod_cast h
    linear_combination ((k₁ : ℝ) - (k₂ : ℝ)) * hr

/-- The scalar score is strictly maximized at an exact key match.  This is
`score_lt_of_ne` of `Transformer.ALM.Basic` in the scalar integer case that
`LookUp` compiles to. -/
theorem sScore_lt_of_ne (q k : ℤ) (hne : k ≠ q) : sScore q k < sScore q q := by
  have hgap : sScore q q - sScore q k = ((k : ℝ) - (q : ℝ)) ^ 2 := by
    unfold sScore; ring
  have hne' : ((k : ℝ) - (q : ℝ)) ≠ 0 := by
    rw [sub_ne_zero]
    exact_mod_cast hne
  have : 0 < ((k : ℝ) - (q : ℝ)) ^ 2 := by positivity
  linarith

/-- **The tie-break path is dead on the lookup path.**  If the query is one of
the stored keys, every line achieving the best score carries that same key, so
the left and right scans of `query` merge nothing new and `resolve` sees a
single value. -/
theorem argmax_unique_of_query_mem {n : ℕ} (K : Fin n → ℤ) (q : ℤ) (i₀ : Fin n)
    (h₀ : K i₀ = q) (j : Fin n) (hj : sScore q (K j) = sScore q (K i₀)) :
    K j = q := by
  by_contra hne
  have hlt : sScore q (K j) < sScore q q := sScore_lt_of_ne q (K j) hne
  rw [hj, h₀] at hlt
  exact lt_irrefl _ hlt

/-- But the path is reachable in general: with the query absent from the key
set, keys `0` and `2` tie at `q = 1`. -/
example : sScore 1 0 = sScore 1 2 ∧ (0 : ℤ) ≠ 1 ∧ (2 : ℤ) ≠ 1 := by
  refine ⟨?_, by decide, by decide⟩
  unfold sScore
  norm_num

/-! ### The aggregate a line carries -/

/-- A model of `HullMeta`: the running aggregate of every value that landed on
one hull line. -/
structure Meta where
  /-- Componentwise sum of the stored values. -/
  vsum : ℝ × ℝ
  /-- The value with the largest sequence number seen so far. -/
  vlast : ℝ × ℝ
  /-- How many values were stored. -/
  count : ℕ
  /-- The largest sequence number seen so far; `-1` before any insertion. -/
  lastSeq : ℤ

namespace Meta

/-- The aggregate of a fresh line, `HullMeta`'s default member initializers. -/
def empty : Meta := ⟨(0, 0), (0, 0), 0, -1⟩

/-- `HullMeta::add`. -/
def add (m : Meta) (v : ℝ × ℝ) (seq : ℤ) : Meta :=
  { vsum := (m.vsum.1 + v.1, m.vsum.2 + v.2)
    vlast := if m.lastSeq < seq then v else m.vlast
    count := m.count + 1
    lastSeq := max seq m.lastSeq }

/-- `HullMeta::merge`. -/
def merge (m o : Meta) : Meta :=
  { vsum := (m.vsum.1 + o.vsum.1, m.vsum.2 + o.vsum.2)
    vlast := if m.lastSeq < o.lastSeq then o.vlast else m.vlast
    count := m.count + o.count
    lastSeq := max o.lastSeq m.lastSeq }

/-- `HullMeta::resolve` under `TieBreak::AVERAGE`. -/
noncomputable def resolveAverage (m : Meta) : ℝ × ℝ :=
  if m.count = 0 then (0, 0)
  else ((m.count : ℝ)⁻¹ * m.vsum.1, (m.count : ℝ)⁻¹ * m.vsum.2)

/-- `HullMeta::resolve` under `TieBreak::LATEST`. -/
def resolveLatest (m : Meta) : ℝ × ℝ :=
  if m.count = 0 then (0, 0) else m.vlast

/-- **The two modes agree on an unmerged line.**  A single stored value is
returned unchanged by both, so the mode is invisible wherever no tie occurred
— in particular everywhere on the lookup path, by
`argmax_unique_of_query_mem`. -/
theorem resolve_eq_of_single (v : ℝ × ℝ) (seq : ℤ) (hseq : 0 ≤ seq) :
    (empty.add v seq).resolveAverage = v ∧ (empty.add v seq).resolveLatest = v := by
  have hlt : (-1 : ℤ) < seq := by linarith
  constructor
  · simp [resolveAverage, add, empty, hlt]
  · simp [resolveLatest, add, empty, hlt]

/-- **Merge order is irrelevant only when sequence numbers are distinct.**
`query` merges the left neighbours in one order and the right neighbours in
another, so the aggregate it produces is well defined only under this
hypothesis. -/
theorem merge_comm {m o : Meta} (h : m.lastSeq ≠ o.lastSeq) : m.merge o = o.merge m := by
  unfold merge
  rcases lt_or_gt_of_ne h with hlt | hgt
  · rw [ite_eq_left hlt, ite_eq_right (not_lt.mpr hlt.le)]
    congr 1 <;> simp [add_comm, max_comm]
  · rw [ite_eq_right (not_lt.mpr hgt.le), ite_eq_left hgt]
    congr 1 <;> simp [add_comm, max_comm]

/-- Without distinct sequence numbers the merge is genuinely order-dependent:
two lines carrying different values at the same sequence number give different
aggregates depending on which side `query` reaches first. -/
example :
    (Meta.mk (1, 0) (1, 0) 1 0).merge (Meta.mk (3, 0) (3, 0) 1 0)
      ≠ (Meta.mk (3, 0) (3, 0) 1 0).merge (Meta.mk (1, 0) (1, 0) 1 0) := by
  simp [merge]

/-- **The file comment is wrong.**  `hull2d_cht.h` line 27 states that
`TieBreak::LATEST` "behaves like AVERAGE"; after a tie merges the values `1`
and `3` the average is `2` and the latest is `3`. -/
example :
    ((Meta.mk (1, 0) (1, 0) 1 0).merge (Meta.mk (3, 0) (3, 0) 1 1)).resolveAverage
      ≠ ((Meta.mk (1, 0) (1, 0) 1 0).merge (Meta.mk (3, 0) (3, 0) 1 1)).resolveLatest := by
  simp [merge, resolveAverage, resolveLatest]
  norm_num

end Meta

end ALM
end Transformer
