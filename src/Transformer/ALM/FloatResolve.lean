/-
# The value the running code returns

`Transformer.ALM.FloatWalk` ends at a set: on integer data the lines the two
merge loops collect are exactly the maximizers, and there are at most two of
them.  `Transformer.ALM.HullValue` ends at a value, but at the index the
*exact* search returns.  The last line of `HullHalf::query` is
`combined.resolve(tb, out)`, and what it writes into `out` had no statement in
the arithmetic that runs: at a tie the floating-point search need not land on
`hullProbe`, so the real-arithmetic theorem does not apply to it.

`fp_query_resolve` and `fp_query_resolveLatest` state it.  Both are
`argmaxSet_resolve` at `fpProbe` — legitimate because that index is a
maximizer (`fpProbe_mem_argmaxSet_of_int`) and because `argmaxSet_resolve` was
stated at an arbitrary maximizer — with the tie set rewritten as the set the
loops actually collect (`fp_walk_collects_of_grid`).  So the conclusion is
about the index the running search returns, the comparison the running loops
make, and the value the running `resolve` writes out, under three conditions
on the arithmetic and none on the query.  `fp_query_cost_total` carries the
price over the same way: the search costs what the index charges for it, and
the loops merge at most one line.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 268-306.
-/

import Transformer.ALM.FloatGrid
import Transformer.ALM.HullValue

namespace Transformer
namespace ALM

open Classical

variable {n : ℕ}

/-- **`combined.resolve(AVERAGE, out)`, in the arithmetic that runs.**  The
lines the loops merge are the line the floating-point search found, alone or
with one neighbour, and what is written out is that line's payload or the
componentwise mean of the two. -/
theorem fp_query_resolve (S : FPScore) (F : FPArith) [Nonempty (Fin n)] (K : Fin n → ℝ)
    (hK : ∀ i, ∃ z : ℤ, K i = (z : ℝ)) (q : ℤ) (B : ℝ)
    (hbd : ∀ j ≤ keyCard K - 1, |sortedKey K j| ≤ B) (hu : F.u * B < 1 / 2)
    (hδ : S.δ < 1)
    (hgrid : ∀ i ≤ keyCard K - 1,
      ∃ z : ℤ, S.eval (liftQuery (q : ℝ)) (liftKey (sortedKey K i)) = (z : ℝ))
    (M : ℕ → Meta) (V : ℕ → ℝ × ℝ) (s : ℕ → ℤ) (hs : ∀ j, 0 ≤ s j)
    (hM : ∀ j, M j = Meta.empty.add (V j) (s j)) :
    let b := fpProbe F K (q : ℝ)
    (fpTieSet S F K (q : ℝ) = {b} ∧ (scanBest M b).resolveAverage = V b) ∨
      (fpTieSet S F K (q : ℝ) = {b, b + 1} ∧
        (scanCombined M b (b + 1)).resolveAverage
          = (((V b).1 + (V (b + 1)).1) / 2, ((V b).2 + (V (b + 1)).2) / 2)) ∨
      (fpTieSet S F K (q : ℝ) = {b - 1, b} ∧
        (scanCombined M b (b - 1)).resolveAverage
          = (((V b).1 + (V (b - 1)).1) / 2, ((V b).2 + (V (b - 1)).2) / 2)) := by
  intro b
  have hcol := fp_walk_collects_of_grid S F K hK q B hbd hu hδ hgrid
  have h := argmaxSet_resolve (sortedKey K) (q : ℝ) (keyCard K - 1) (sortedKey_lt_succ K)
    (fpProbe_mem_argmaxSet_of_int F K hK q B hbd hu) M V s hs hM
  rw [← hcol] at h
  exact h

/-- **And `resolve(LATEST, out)`.**  Where the search's line is alone the walk
writes out its payload; where a neighbour tied, the payload of whichever of
the two was appended later. -/
theorem fp_query_resolveLatest (S : FPScore) (F : FPArith) [Nonempty (Fin n)]
    (K : Fin n → ℝ) (hK : ∀ i, ∃ z : ℤ, K i = (z : ℝ)) (q : ℤ) (B : ℝ)
    (hbd : ∀ j ≤ keyCard K - 1, |sortedKey K j| ≤ B) (hu : F.u * B < 1 / 2)
    (hδ : S.δ < 1)
    (hgrid : ∀ i ≤ keyCard K - 1,
      ∃ z : ℤ, S.eval (liftQuery (q : ℝ)) (liftKey (sortedKey K i)) = (z : ℝ))
    (M : ℕ → Meta) (V : ℕ → ℝ × ℝ) (s : ℕ → ℤ) (hs : ∀ j, 0 ≤ s j)
    (hsinj : Function.Injective s)
    (hM : ∀ j, M j = Meta.empty.add (V j) (s j)) :
    let b := fpProbe F K (q : ℝ)
    (fpTieSet S F K (q : ℝ) = {b} ∧ (scanBest M b).resolveLatest = V b) ∨
      ∃ c, c ≠ b ∧
        (fpTieSet S F K (q : ℝ) = {b, c} ∨ fpTieSet S F K (q : ℝ) = {c, b}) ∧
        (scanCombined M b c).resolveLatest = if s b < s c then V c else V b := by
  intro b
  have hcol := fp_walk_collects_of_grid S F K hK q B hbd hu hδ hgrid
  have h := argmaxSet_resolveLatest (sortedKey K) (q : ℝ) (keyCard K - 1)
    (sortedKey_lt_succ K) (fpProbe_mem_argmaxSet_of_int F K hK q B hbd hu) M V s hs hsinj hM
  rw [← hcol] at h
  exact h

/-- **And the price, in the arithmetic that runs.**  `hullQuery_cost_total`
charges the search and bounds the merges at `hullProbe`; the same two bounds
hold of the running query.  The search count is unchanged — `bcount` depends
on the length of the range and not on how the comparisons come out — and the
merge bound is the trichotomy above: the loops merge at most one line on top
of the one the search found.

Source: `hull2d_cht.h`, lines 268-306. -/
theorem fp_query_cost_total (S : FPScore) (F : FPArith) [Nonempty (Fin n)] (K : Fin n → ℝ)
    (hK : ∀ i, ∃ z : ℤ, K i = (z : ℝ)) (q : ℤ) (B : ℝ)
    (hbd : ∀ j ≤ keyCard K - 1, |sortedKey K j| ≤ B) (hu : F.u * B < 1 / 2)
    (hδ : S.δ < 1)
    (hgrid : ∀ i ≤ keyCard K - 1,
      ∃ z : ℤ, S.eval (liftQuery (q : ℝ)) (liftKey (sortedKey K i)) = (z : ℝ)) :
    ((bcount (keyCard K - 1) : ℕ) : ℝ) ≤ hullIndex.query n 1 ∧
      ((fpTieSet S F K (q : ℝ)).erase (fpProbe F K (q : ℝ))).card ≤ 1 := by
  refine ⟨hullIndex_query_paid K, ?_⟩
  rw [fp_walk_collects_of_grid S F K hK q B hbd hu hδ hgrid]
  exact scan_merge_count_le_one (sortedKey K) (q : ℝ) (keyCard K - 1) (sortedKey_lt_succ K)
    (fpProbe_mem_argmaxSet_of_int F K hK q B hbd hu)

/-- Every hypothesis holds at once, for the rounding score routine on any
integer family: the keys are integers and bounded, exact breakpoints clear the
margin, `roundScore` is grid-valued with `δ = 1/2 < 1`, and the aggregates are
those an append-only log builds. -/
example [Nonempty (Fin n)] (K : Fin n → ℤ) (q : ℤ) :
    (∀ i, ∃ z : ℤ, ((K i : ℝ)) = (z : ℝ)) ∧ roundScore.δ < 1 ∧
      (∀ i ≤ keyCard (fun i => (K i : ℝ)) - 1, ∃ z : ℤ,
        roundScore.eval (liftQuery (q : ℝ))
          (liftKey (sortedKey (fun i => (K i : ℝ)) i)) = (z : ℝ)) ∧
      (∀ j : ℕ, (0 : ℤ) ≤ (j : ℤ)) ∧ Function.Injective (fun i : ℕ => (i : ℤ)) ∧
      (∀ j : ℕ, (fun i : ℕ => Meta.empty.add ((i : ℝ), 0) (i : ℤ)) j
        = Meta.empty.add ((fun i : ℕ => ((i : ℝ), (0 : ℝ))) j) ((fun i : ℕ => (i : ℤ)) j)) ∧
      ∃ B : ℝ, (∀ j ≤ keyCard (fun i => (K i : ℝ)) - 1,
          |sortedKey (fun i => (K i : ℝ)) j| ≤ B) ∧ exactArith.u * B < 1 / 2 := by
  refine ⟨fun i => ⟨K i, rfl⟩, by norm_num [roundScore], fun i _ => roundScore_grid _ _,
    fun j => Int.natCast_nonneg j, fun a b h => by simpa using h, fun _ => rfl, ?_⟩
  obtain ⟨c, -, hc⟩ := Finset.exists_max_image (Finset.range (keyCard (fun i => (K i : ℝ))))
    (fun j => |sortedKey (fun i => (K i : ℝ)) j|)
    ⟨0, Finset.mem_range.mpr (keyCard_pos (fun i => (K i : ℝ)))⟩
  refine ⟨|sortedKey (fun i => (K i : ℝ)) c|, fun j hj => hc j (Finset.mem_range.mpr ?_), ?_⟩
  · have hpos := keyCard_pos (fun i => (K i : ℝ))
    omega
  · simp [exactArith]

end ALM
end Transformer
