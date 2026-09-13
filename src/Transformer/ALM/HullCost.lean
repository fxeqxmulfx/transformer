/-
# What one query costs, in full

`Transformer.ALM.HullIndex` prices the binary search: `hullIndex_query_paid`
bounds the comparisons `bsearch` makes.  `Transformer.ALM.HullScan` prices
the tie-merge walk that follows it: `scan_merge_count_le_one` bounds the
merges, but it bounds them *starting from an arbitrary element of*
`argmaxSet`, because nothing in that file knows where the search lands.

The two halves never touched, so the sentence "one query is a logarithmic
search plus at most one merge" was true of the implementation and unstated in
Lean.  `hullProbe_mem_argmaxSet` supplies the missing step — the index the
search returns *is* a maximizer, which is the defining property of
`argmaxSet` — and `hullQuery_cost_total` is then the whole price of
`HullHalf::query` (`transformer_vm/attention/hull2d_cht.h`, lines 270-303) in
one statement.

`hullQuery_collects` then says, with no hypothesis at all, which lines the
walk can ever see: the winner alone, or the winner with one neighbour on one
side — the left loop and the right loop, each running at most once.
-/

import Transformer.ALM.HullIndex
import Transformer.ALM.HullScan

namespace Transformer
namespace ALM

variable {n : ℕ}

/-- **The search lands where the walk begins.**  `bsearch` returns a
maximizer among the sorted keys, which is exactly membership in `argmaxSet`,
so the bounds of `Transformer.ALM.HullScan` apply to the index
`HullHalf::query` actually starts its two loops from. -/
theorem hullProbe_mem_argmaxSet [Nonempty (Fin n)] (K : Fin n → ℝ) (q : ℝ) :
    hullProbe K q ∈ argmaxSet (sortedKey K) q (keyCard K - 1) := by
  rw [mem_argmaxSet]
  refine ⟨hullProbe_le K q, fun i hi => ?_⟩
  obtain ⟨j, hj⟩ := exists_eq_sortedKey K hi
  rw [← hj]
  exact hullProbe_isGreatest K q j

/-- **The whole price of a query.**  The search pays what `hullIndex` charges
for it, and the walk that follows performs at most one merge on top — so the
declared query cost of the index is the cost of the procedure, not of half of
it. -/
theorem hullQuery_cost_total [Nonempty (Fin n)] (K : Fin n → ℝ) (q : ℝ) :
    ((bcount (keyCard K - 1) : ℕ) : ℝ) ≤ hullIndex.query n 1 ∧
      ((argmaxSet (sortedKey K) q (keyCard K - 1)).erase (hullProbe K q)).card ≤ 1 :=
  ⟨hullIndex_query_paid K,
    scan_merge_count_le_one (sortedKey K) q (keyCard K - 1) (sortedKey_lt_succ K)
      (hullProbe_mem_argmaxSet K q)⟩

/-- **And exactly what it collects.**  No hypothesis: whatever the keys and
the query, the tie set is the winner alone, the winner with its successor, or
the winner with its predecessor — the left loop and the right loop of
`HullHalf::query` between them, each running at most once. -/
theorem hullQuery_collects [Nonempty (Fin n)] (K : Fin n → ℝ) (q : ℝ) :
    argmaxSet (sortedKey K) q (keyCard K - 1) = {hullProbe K q} ∨
      argmaxSet (sortedKey K) q (keyCard K - 1) = {hullProbe K q, hullProbe K q + 1} ∨
      argmaxSet (sortedKey K) q (keyCard K - 1) = {hullProbe K q - 1, hullProbe K q} := by
  set S := argmaxSet (sortedKey K) q (keyCard K - 1) with hSdef
  set b := hullProbe K q with hbdef
  have hb : b ∈ S := hullProbe_mem_argmaxSet K q
  have hcard : S.card ≤ 2 :=
    argmaxSet_card_le_two (sortedKey K) q (keyCard K - 1) (sortedKey_lt_succ K)
  have hpos : 1 ≤ S.card := Finset.card_pos.mpr ⟨b, hb⟩
  have herase : (S.erase b).card = S.card - 1 := Finset.card_erase_of_mem hb
  have hins : insert b (S.erase b) = S := Finset.insert_erase hb
  rcases Nat.lt_or_ge (S.erase b).card 1 with h1 | h1
  · left
    have hempty : S.erase b = ∅ := Finset.card_eq_zero.mp (by omega)
    rw [← hins, hempty]
    rfl
  · obtain ⟨c, hc⟩ := Finset.card_eq_one.mp (show (S.erase b).card = 1 by omega)
    have hcS : c ∈ S := Finset.mem_of_mem_erase (by rw [hc]; simp)
    have hcne : c ≠ b := Finset.ne_of_mem_erase (by rw [hc]; simp)
    have hSeq : S = {b, c} := by rw [← hins, hc]
    rcases Nat.lt_or_ge c b with hlt | hge
    · right; right
      have hadj := argmaxSet_adjacent (sortedKey K) q (keyCard K - 1) (sortedKey_lt_succ K)
        hcS hb hlt
      rw [hSeq, show c = b - 1 by omega, Finset.pair_comm]
    · right; left
      have hlt' : b < c := lt_of_le_of_ne hge (Ne.symm hcne)
      have hadj := argmaxSet_adjacent (sortedKey K) q (keyCard K - 1) (sortedKey_lt_succ K)
        hb hcS hlt'
      rw [hSeq, show c = b + 1 by omega]

/-- The hypotheses are satisfiable: `Fin 3` is nonempty, so the price and the
trichotomy are statements about a family of keys that exists. -/
example :
    argmaxSet (sortedKey (fun j : Fin 3 => (j : ℝ))) 2
        (keyCard (fun j : Fin 3 => (j : ℝ)) - 1)
      = {hullProbe (fun j : Fin 3 => (j : ℝ)) 2} ∨
    argmaxSet (sortedKey (fun j : Fin 3 => (j : ℝ))) 2
        (keyCard (fun j : Fin 3 => (j : ℝ)) - 1)
      = {hullProbe (fun j : Fin 3 => (j : ℝ)) 2,
          hullProbe (fun j : Fin 3 => (j : ℝ)) 2 + 1} ∨
    argmaxSet (sortedKey (fun j : Fin 3 => (j : ℝ))) 2
        (keyCard (fun j : Fin 3 => (j : ℝ)) - 1)
      = {hullProbe (fun j : Fin 3 => (j : ℝ)) 2 - 1,
          hullProbe (fun j : Fin 3 => (j : ℝ)) 2} :=
  hullQuery_collects _ _

end ALM
end Transformer
