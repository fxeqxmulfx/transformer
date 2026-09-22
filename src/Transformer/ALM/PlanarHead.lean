/-
# One hull query answers any planar head

`Transformer.ALM.HullBranch` proves the three branches of `query` one at a
time: the upper hull for `qy > 0`, the negated family for `qy < 0`, and the
two ends of the degenerate branch for `qy = 0`.  Each is a theorem about the
branch the code took, and nothing put them together — so the claim the
manuscript makes,

> the geometric fast path is not limited to our executor construction.  In
> principle, it can accelerate any transformer with 2D heads at decoding time,
> replacing full linear scans with efficient geometric retrieval

was three theorems and a dispatch left to the reader.  The dispatch matters:
the executor only ever queries with `qy = 1` (`liftQuery_snd_pos`), so the
other two branches exist for queries the lookup machine never makes — the
queries a *trained* planar head makes.

`planarAns` is that dispatch, written out as `query` dispatches, and
`planarAns_isGreatest` is the one theorem on top: for an arbitrary planar
query — `qy > 0`, `qy < 0`, `qy = 0` either way, and `q = 0` — the line it
returns maximizes `q · k` over the hull the branch consults, at the cost of
one binary search.  `planar_head_argmax` is the same statement for an
arbitrary family of planar keys, each dominated at the query by a line of that
hull, which is what the erase rule of `Transformer.ALM.HullErase` preserves.

The two hulls are two families of their own, of their own lengths.  An earlier
statement asked every upper line to be a negated lower one on a common range;
together with both breakpoint invariants that forces every breakpoint of the
upper hull in the range to coincide, since negation reverses the order of the
slopes and keeps `interX`.  That statement held only for concurrent lines and
is replaced.

The keys here are arbitrary: no lift, no paraboloid, no integrality.  A
trained 2D head gets the same `log₂ n + 1`.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 255-315 (`query`);
Percepta, *Can LLMs Be Computers?* (2026-03-11).
-/

import Transformer.ALM.HullBranch

namespace Transformer
namespace ALM

/-! ### The dispatch itself -/

/-- **`query`, as the code dispatches it.**  `L` is what the upper hull
stores, `nL + 1` lines, and `N` what the lower hull stores, `nN + 1` lines —
points negated, as `hull2d_cht.h` pushes them — and the answer is a *key*,
the line the branch lands on.  The `qy = 0` branches are the code's queries at
`±INF` of the upper hull, answered with no search at all. -/
noncomputable def planarAns (L N : ℕ → ℝ × ℝ) (nL nN : ℕ) (q : ℝ × ℝ) : ℝ × ℝ :=
  if 0 < q.2 then
    L (bsearch (fun j => decide (q.1 / q.2 ≤ interX (L j) (L (j + 1)))) 0 nL)
  else if q.2 < 0 then
    -N (bsearch (fun j => decide (q.1 / q.2 ≤ interX (N j) (N (j + 1)))) 0 nN)
  else if 0 < q.1 then L nL else L 0

/-! ### And it answers every query -/

/-- **One query, every branch.**  Under the two hull invariants on each stored
family, the key `planarAns` returns maximizes `q · k` over the hull the branch
consults — the upper one for `qy ≥ 0`, the degenerate `qy = 0` branches and
`q = 0` included, the negated lower one for `qy < 0` — and each search costs
at most `log₂ n + 1` comparisons.

Source: `hull2d_cht.h`, lines 255-315. -/
theorem planarAns_isGreatest (L N : ℕ → ℝ × ℝ) (nL nN : ℕ) (q : ℝ × ℝ)
    (hslope : ∀ j, (L j).1 < (L (j + 1)).1)
    (hbp : ∀ a b, a ≤ b → interX (L a) (L (a + 1)) ≤ interX (L b) (L (b + 1)))
    (hslopeN : ∀ j, (N j).1 < (N (j + 1)).1)
    (hbpN : ∀ a b, a ≤ b → interX (N a) (N (a + 1)) ≤ interX (N b) (N (b + 1))) :
    (0 ≤ q.2 → ∀ j ≤ nL, dot q (L j) ≤ dot q (planarAns L N nL nN q)) ∧
      (q.2 < 0 → ∀ j ≤ nN, dot q (-N j) ≤ dot q (planarAns L N nL nN q)) ∧
      bcount nL ≤ Nat.log 2 nL + 1 ∧ bcount nN ≤ Nat.log 2 nN + 1 := by
  refine ⟨fun hq j hj => ?_, fun hq j hj => ?_, bcount_le_log nL, bcount_le_log nN⟩
  · unfold planarAns
    split_ifs with h1 h2 h3
    · exact (planar_bsearch_of_pos L q h1 nL hslope hbp).1 j hj
    · exact absurd hq (not_le.mpr h2)
    · exact planar_argmax_of_snd_eq_zero L q (le_antisymm (not_lt.mp h1) hq) h3 nL hslope j hj
    · have hq2 : q.2 = 0 := le_antisymm (not_lt.mp h1) hq
      rcases lt_or_eq_of_le (not_lt.mp h3) with hq1 | hq1
      · exact planar_argmax_of_snd_eq_zero_neg L q hq2 hq1 nL hslope j hj
      · simp [dot, hq1, hq2]
  · unfold planarAns
    split_ifs with h1
    · exact absurd h1 (not_lt.mpr hq.le)
    · exact (planar_bsearch_of_neg N q hq nN hslopeN hbpN).1 j hj

/-- **And so it answers an arbitrary planar head.**  For any finite family of
planar keys, each dominated at the query by a line of the hull the branch
consults — the invariant the erase rule preserves — the key `planarAns`
returns maximizes the head's score over the whole family.  Nothing here is
lifted, integral, or produced by the executor: this is the fast path for a
trained 2D head.

Source: `hull2d_cht.h`, lines 255-315. -/
theorem planar_head_argmax {n' : ℕ} (L N : ℕ → ℝ × ℝ) (nL nN : ℕ) (q : ℝ × ℝ)
    (hslope : ∀ j, (L j).1 < (L (j + 1)).1)
    (hbp : ∀ a b, a ≤ b → interX (L a) (L (a + 1)) ≤ interX (L b) (L (b + 1)))
    (hslopeN : ∀ j, (N j).1 < (N (j + 1)).1)
    (hbpN : ∀ a b, a ≤ b → interX (N a) (N (a + 1)) ≤ interX (N b) (N (b + 1)))
    (K : Fin n' → ℝ × ℝ)
    (hcovL : 0 ≤ q.2 → ∀ i, ∃ j ≤ nL, dot q (K i) ≤ dot q (L j))
    (hcovN : q.2 < 0 → ∀ i, ∃ j ≤ nN, dot q (K i) ≤ dot q (-N j)) :
    (∀ i, dot q (K i) ≤ dot q (planarAns L N nL nN q)) ∧
      bcount nL ≤ Nat.log 2 nL + 1 ∧ bcount nN ≤ Nat.log 2 nN + 1 := by
  obtain ⟨hL, hN, hcL, hcN⟩ := planarAns_isGreatest L N nL nN q hslope hbp hslopeN hbpN
  refine ⟨fun i => ?_, hcL, hcN⟩
  rcases le_or_gt 0 q.2 with hq | hq
  · obtain ⟨j, hj, hle⟩ := hcovL hq i
    exact hle.trans (hL hq j hj)
  · obtain ⟨j, hj, hle⟩ := hcovN hq i
    exact hle.trans (hN hq j hj)

/-- The hypotheses are satisfiable, by the family the machine itself stores,
and on both branches at once.  The lifted keys `parabLine` satisfy both
invariants, so they serve as the upper hull and as the stored lower one.  At
`q = (0, 1)` every lifted key `j ≤ n` is dominated by itself on the upper
hull; at `q = (0, -1)` the keys `-parabLine j` are dominated by themselves on
the lower one. -/
example (n : ℕ) :
    (∀ j, (parabLine j).1 < (parabLine (j + 1)).1) ∧
      (∀ a b, a ≤ b → interX (parabLine a) (parabLine (a + 1))
        ≤ interX (parabLine b) (parabLine (b + 1))) ∧
      (0 ≤ ((0 : ℝ), (1 : ℝ)).2 → ∀ i : Fin (n + 1),
        ∃ j ≤ n, dot (0, 1) (parabLine i) ≤ dot (0, 1) (parabLine j)) ∧
      (((0 : ℝ), (-1 : ℝ)).2 < 0 → ∀ i : Fin (n + 1),
        ∃ j ≤ n, dot (0, -1) (-parabLine i) ≤ dot (0, -1) (-parabLine j)) :=
  ⟨parabLine_slope, parabLine_bp, fun _ i => ⟨i, Nat.lt_succ_iff.mp i.isLt, le_rfl⟩,
    fun _ i => ⟨i, Nat.lt_succ_iff.mp i.isLt, le_rfl⟩⟩

end ALM
end Transformer
