/-
# What the merge walk actually returns

`Transformer.ALM.HullScan` bounds the walk: the tie set of a sorted key family
is a singleton or an adjacent pair, so `HullHalf::query` merges at most one
neighbour on each side.  What it did not say is what comes out.  The aggregate
the walk builds is

    HullMeta combined;
    combined.merge(best_it->meta);
    ... combined.merge(prev->meta) ...  ... combined.merge(itR->meta) ...
    combined.resolve(tb, out);

(`transformer_vm/attention/hull2d_cht.h`, lines 276-306), and `resolve`
divides the summed values by the count, or returns the latest one
(lines 70-83).  `scanCombined_count` only added the counts.

`merge_empty_left` is the step that makes the rest short: merging into the
fresh `HullMeta` is the identity on any line that has stored something, so the
walk's aggregate is the plain merge of the lines it visited.  From there
`scanCombined_resolveAverage` gives the value the head returns at a tie — the
midpoint of the two tied payloads, which is what the file's own header comment
(lines 26-27) promises — and `tie_resolve` attaches it to the tie set of
`Transformer.ALM.HullScan`, so the pair the walk visits is the pair the
geometry says is there.

`Meta` and `resolve` are `Transformer.ALM.TieBreak`.
-/

import Transformer.ALM.HullScan

namespace Transformer
namespace ALM

namespace Meta

/-- **The fresh aggregate is a unit.**  `HullMeta combined;` followed by
`combined.merge(x)` leaves `x` alone, provided `x` has stored a value: its
sequence number is then above the `-1` of the default member initializers, so
the merge takes `x`'s own `vlast`. -/
theorem merge_empty_left (m : Meta) (h : 0 ≤ m.lastSeq) : Meta.empty.merge m = m := by
  have hlt : Meta.empty.lastSeq < m.lastSeq := by
    show (-1 : ℤ) < m.lastSeq
    omega
  unfold merge
  rw [ite_eq_left hlt]
  simp [Meta.empty, max_eq_left (show (-1 : ℤ) ≤ m.lastSeq by omega)]

end Meta

/-- The aggregate after `combined.merge(best_it->meta)`, with no tie found. -/
def scanBest (M : ℕ → Meta) (b : ℕ) : Meta := Meta.merge Meta.empty (M b)

/-- And after one more `combined.merge` for the neighbour that tied — the whole
of the walk, by `scan_merge_count_le_one`. -/
def scanCombined (M : ℕ → Meta) (b c : ℕ) : Meta := Meta.merge (scanBest M b) (M c)

/-- **`resolve` sees both values and nothing else.**  The counts add, so the
`AVERAGE` mode of `Transformer.ALM.TieBreak` divides by exactly two when two
single-valued lines tie. -/
theorem scanCombined_count (M : ℕ → Meta) (b c : ℕ) :
    (scanCombined M b c).count = (M b).count + (M c).count := by
  simp [scanCombined, scanBest, Meta.merge, Meta.empty]

/-- The walk that found no tie returns the line it started from, unchanged. -/
theorem scanBest_eq (M : ℕ → Meta) (b : ℕ) (h : 0 ≤ (M b).lastSeq) :
    scanBest M b = M b :=
  Meta.merge_empty_left (M b) h

/-- **The tie is resolved to the midpoint.**  Two lines each carrying one
stored value, merged by the walk, resolve under `AVERAGE` to the componentwise
mean — the "return mean of tied values" of the file's header comment. -/
theorem scanCombined_resolveAverage (M : ℕ → Meta) (b c : ℕ) (v w : ℝ × ℝ) (sb sc : ℤ)
    (hsb : 0 ≤ sb) (hsc : 0 ≤ sc)
    (hb : M b = Meta.empty.add v sb) (hc : M c = Meta.empty.add w sc) :
    (scanCombined M b c).resolveAverage = ((v.1 + w.1) / 2, (v.2 + w.2) / 2) := by
  have hlb : (-1 : ℤ) < sb := by linarith
  have hlc : (-1 : ℤ) < sc := by linarith
  have hcount : (scanCombined M b c).count = 2 := by
    rw [scanCombined_count, hb, hc]; simp [Meta.add, Meta.empty]
  have hsum : (scanCombined M b c).vsum = (v.1 + w.1, v.2 + w.2) := by
    simp [scanCombined, scanBest, Meta.merge, Meta.empty, hb, hc, Meta.add]
  rw [Meta.resolveAverage, ite_eq_right (by omega), hcount, hsum]
  norm_num
  constructor <;> ring

/-- And under `LATEST` to the value with the larger sequence number, which is
the later `append` — the mode the file says does not exist. -/
theorem scanCombined_resolveLatest (M : ℕ → Meta) (b c : ℕ) (v w : ℝ × ℝ) (sb sc : ℤ)
    (hsb : 0 ≤ sb) (hlt : sb < sc)
    (hb : M b = Meta.empty.add v sb) (hc : M c = Meta.empty.add w sc) :
    (scanCombined M b c).resolveLatest = w := by
  have hcount : (scanCombined M b c).count = 2 := by
    rw [scanCombined_count, hb, hc]; simp [Meta.add, Meta.empty]
  have hvlast : (scanCombined M b c).vlast = w := by
    have h1 : (-1 : ℤ) < sb := by omega
    have h2 : (-1 : ℤ) < sc := by omega
    simp [scanCombined, scanBest, Meta.merge, Meta.empty, hb, hc, Meta.add, h1, h2, hlt]
  rw [Meta.resolveLatest, ite_eq_right (by omega), hvlast]

/-- **Which side the walk reached first does not matter.**  `query` merges the
left neighbour before the right one, so the aggregate depends on an order the
geometry does not fix; with distinct sequence numbers `Meta.merge_comm` says
the two orders agree, and the walk's answer is well defined. -/
theorem scanCombined_comm (M : ℕ → Meta) (b c : ℕ)
    (hb : 0 ≤ (M b).lastSeq) (hc : 0 ≤ (M c).lastSeq)
    (hne : (M b).lastSeq ≠ (M c).lastSeq) :
    scanCombined M b c = scanCombined M c b := by
  unfold scanCombined
  rw [scanBest_eq M b hb, scanBest_eq M c hc]
  exact Meta.merge_comm hne

/-- **LATEST, in either order.**  `scanCombined_resolveLatest` fixed the order
of the two sequence numbers; the hull fixes the order of the two *positions*
and says nothing about insertion times.  Under `TieBreak::LATEST` the walk
returns the payload of whichever tied line was appended later, whichever side
of the winner it sits on. -/
theorem scanCombined_resolveLatest_of_ne (M : ℕ → Meta) (b c : ℕ) (v w : ℝ × ℝ) (sb sc : ℤ)
    (hsb : 0 ≤ sb) (hsc : 0 ≤ sc) (hne : sb ≠ sc)
    (hb : M b = Meta.empty.add v sb) (hc : M c = Meta.empty.add w sc) :
    (scanCombined M b c).resolveLatest = if sb < sc then w else v := by
  have hlb : (M b).lastSeq = sb := by rw [hb]; simp [Meta.add, Meta.empty]; omega
  have hlc : (M c).lastSeq = sc := by rw [hc]; simp [Meta.add, Meta.empty]; omega
  rcases lt_or_gt_of_ne hne with hlt | hgt
  · rw [ite_eq_left hlt]
    exact scanCombined_resolveLatest M b c v w sb sc hsb hlt hb hc
  · rw [ite_eq_right (not_lt.mpr hgt.le),
      scanCombined_comm M b c (by omega) (by omega) (by omega)]
    exact scanCombined_resolveLatest M c b w v sc sb hsc hgt hc hb

/-- The hypotheses are satisfiable in the order the earlier lemma could not
reach: here the *first* position carries the later insertion, and LATEST
returns its payload rather than the second one's. -/
example :
    (scanCombined (fun j : ℕ => Meta.empty.add ((j : ℝ), 0) (1 - (j : ℤ))) 0 1).resolveLatest
      = (0, 0) := by
  rw [scanCombined_resolveLatest_of_ne _ 0 1 (0, 0) (1, 0) 1 0 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)]
  norm_num

/-- The hypotheses are satisfiable, and the two modes really do differ on them:
the values `(1,0)` and `(3,0)` appended in that order average to `(2,0)` and
resolve latest to `(3,0)`. -/
example :
    (scanCombined (fun j : ℕ => Meta.empty.add (2 * (j : ℝ) + 1, 0) (j : ℤ)) 0 1).resolveAverage
        = (2, 0) ∧
      (scanCombined (fun j : ℕ => Meta.empty.add (2 * (j : ℝ) + 1, 0) (j : ℤ)) 0 1).resolveLatest
        = (3, 0) := by
  constructor
  · rw [scanCombined_resolveAverage _ 0 1 (1, 0) (3, 0) 0 1 (by norm_num) (by norm_num)
      (by norm_num) (by norm_num)]
    norm_num
  · rw [scanCombined_resolveLatest _ 0 1 (1, 0) (3, 0) 0 1 (by norm_num) (by norm_num)
      (by norm_num) (by norm_num)]

/-- **The walk visits the tie set, and returns its mean.**  Joining
`argmaxSet_eq_pair` to the resolution: where the geometry of
`Transformer.ALM.HullScan` says two adjacent keys tie, the head returns the
midpoint of their two payloads, and merges nothing else. -/
theorem tie_resolve (K : ℕ → ℝ) (q : ℝ) (N : ℕ) (hstep : ∀ j, K j < K (j + 1))
    (M : ℕ → Meta) (b : ℕ) (v w : ℝ × ℝ) (sb sc : ℤ) (hsb : 0 ≤ sb) (hsc : 0 ≤ sc)
    (hb : b ∈ argmaxSet K q N) (hb1 : b + 1 ∈ argmaxSet K q N)
    (hMb : M b = Meta.empty.add v sb) (hMc : M (b + 1) = Meta.empty.add w sc) :
    argmaxSet K q N = {b, b + 1} ∧
      (scanCombined M b (b + 1)).resolveAverage = ((v.1 + w.1) / 2, (v.2 + w.2) / 2) :=
  ⟨argmaxSet_eq_pair K q N hstep hb hb1,
    scanCombined_resolveAverage M b (b + 1) v w sb sc hsb hsc hMb hMc⟩

/-- The hypotheses are satisfiable and not vacuous: the keys `j ↦ j` tie at
`q = 1/2`, where the walk really runs, and the mean of the payloads `0` and `1`
is the `1/2` no single line carries. -/
example :
    argmaxSet (fun i : ℕ => (i : ℝ)) (1 / 2) 1 = {0, 1} ∧
      (scanCombined (fun j : ℕ => Meta.empty.add ((j : ℝ), 0) (j : ℤ)) 0 1).resolveAverage
        = (1 / 2, 0) := by
  have hmem : ∀ i : ℕ, i ≤ 1 → i ∈ argmaxSet (fun i : ℕ => (i : ℝ)) (1 / 2) 1 := by
    intro i hi
    rw [mem_argmaxSet]
    refine ⟨hi, fun k hk => ?_⟩
    interval_cases i <;> interval_cases k <;> simp [lineEval_liftKey]
  have h := tie_resolve (fun i : ℕ => (i : ℝ)) (1 / 2) 1 (fun j => by push_cast; linarith)
    (fun j : ℕ => Meta.empty.add ((j : ℝ), 0) (j : ℤ)) 0 (0, 0) (1, 0) 0 1
    (by norm_num) (by norm_num) (hmem 0 (by omega)) (hmem 1 (by omega))
    (by norm_num) (by norm_num)
  refine ⟨h.1, ?_⟩
  rw [h.2]
  norm_num

end ALM
end Transformer
