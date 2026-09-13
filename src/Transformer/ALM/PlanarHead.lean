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
query — `qy > 0`, `qy < 0`, `qy = 0` either way, and `q = 0` — the index it
returns maximizes `q · k` over every stored line, at the cost of one binary
search.  `planar_head_argmax` is the same statement for an arbitrary family of
planar keys, each dominated at the query by a stored line, which is what the
erase rule of `Transformer.ALM.HullErase` preserves.

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
stores and `N` what the lower hull stores — the same points negated, as
`hull2d_cht.h` pushes them — and the answer is a stored *key*, the line the
branch lands on.  The `qy = 0` branches are the code's queries at `±INF`,
answered with no search at all. -/
noncomputable def planarAns (L N : ℕ → ℝ × ℝ) (n : ℕ) (q : ℝ × ℝ) : ℝ × ℝ :=
  if 0 < q.2 then
    L (bsearch (fun j => decide (q.1 / q.2 ≤ interX (L j) (L (j + 1)))) 0 n)
  else if q.2 < 0 then
    -N (bsearch (fun j => decide (q.1 / q.2 ≤ interX (N j) (N (j + 1)))) 0 n)
  else if 0 < q.1 then L n else L 0

/-! ### And it answers every query -/

/-- **One query, every branch.**  Under the two hull invariants on both stored
families, and with the two families holding the same points (`hsame`: every
line of the upper hull is a negated line of the lower one), the key
`planarAns` returns maximizes `q · k` over the whole stored range — for an
arbitrary planar query, the degenerate `qy = 0` branches and `q = 0` included
— and the search that found it cost `log₂ n + 1` comparisons.

Source: `hull2d_cht.h`, lines 255-315. -/
theorem planarAns_isGreatest (L N : ℕ → ℝ × ℝ) (n : ℕ) (q : ℝ × ℝ)
    (hslope : ∀ j, (L j).1 < (L (j + 1)).1)
    (hbp : ∀ a b, a ≤ b → interX (L a) (L (a + 1)) ≤ interX (L b) (L (b + 1)))
    (hslopeN : ∀ j, (N j).1 < (N (j + 1)).1)
    (hbpN : ∀ a b, a ≤ b → interX (N a) (N (a + 1)) ≤ interX (N b) (N (b + 1)))
    (hsame : ∀ j ≤ n, ∃ i ≤ n, -N i = L j) :
    (∀ j ≤ n, dot q (L j) ≤ dot q (planarAns L N n q)) ∧ bcount n ≤ Nat.log 2 n + 1 := by
  refine ⟨fun j hj => ?_, bcount_le_log n⟩
  unfold planarAns
  split_ifs with h1 h2 h3
  · exact (planar_bsearch_of_pos L q h1 n hslope hbp).1 j hj
  · obtain ⟨i, hi, hiL⟩ := hsame j hj
    rw [← hiL]
    exact (planar_bsearch_of_neg N q h2 n hslopeN hbpN).1 i hi
  · exact planar_argmax_of_snd_eq_zero L q (le_antisymm (not_lt.mp h1) (not_lt.mp h2)) h3 n
      hslope j hj
  · have hq2 : q.2 = 0 := le_antisymm (not_lt.mp h1) (not_lt.mp h2)
    rcases lt_or_eq_of_le (not_lt.mp h3) with hq1 | hq1
    · exact planar_argmax_of_snd_eq_zero_neg L q hq2 hq1 n hslope j hj
    · simp [dot, hq1, hq2]

/-- **And so it answers an arbitrary planar head.**  For any finite family of
planar keys, each dominated at the query by some stored line — the invariant
the erase rule preserves — the key `planarAns` returns maximizes the head's
score over the whole family.  Nothing here is lifted, integral, or produced by
the executor: this is the fast path for a trained 2D head. -/
theorem planar_head_argmax {n' : ℕ} (L N : ℕ → ℝ × ℝ) (n : ℕ) (q : ℝ × ℝ)
    (hslope : ∀ j, (L j).1 < (L (j + 1)).1)
    (hbp : ∀ a b, a ≤ b → interX (L a) (L (a + 1)) ≤ interX (L b) (L (b + 1)))
    (hslopeN : ∀ j, (N j).1 < (N (j + 1)).1)
    (hbpN : ∀ a b, a ≤ b → interX (N a) (N (a + 1)) ≤ interX (N b) (N (b + 1)))
    (hsame : ∀ j ≤ n, ∃ i ≤ n, -N i = L j)
    (K : Fin n' → ℝ × ℝ) (hcov : ∀ i, ∃ j ≤ n, dot q (K i) ≤ dot q (L j)) :
    (∀ i, dot q (K i) ≤ dot q (planarAns L N n q)) ∧ bcount n ≤ Nat.log 2 n + 1 := by
  refine ⟨fun i => ?_, bcount_le_log n⟩
  obtain ⟨j, hj, hle⟩ := hcov i
  exact hle.trans ((planarAns_isGreatest L N n q hslope hbp hslopeN hbpN hsame).1 j hj)

/-- The hypotheses are satisfiable, and by the family the machine itself
stores: the lifted keys satisfy both invariants, their negations satisfy them
in the other order only after the code's own reordering, so `hsame` is stated
of the pair and here witnessed by taking the lower hull to be the negated
upper one.  A key is dominated at the query by itself. -/
example (q : ℝ × ℝ) (n : ℕ) :
    (∀ j, (parabLine j).1 < (parabLine (j + 1)).1) ∧
      (∀ a b, a ≤ b → interX (parabLine a) (parabLine (a + 1))
        ≤ interX (parabLine b) (parabLine (b + 1))) ∧
      (∀ j ≤ n, ∃ i ≤ n, -(-parabLine i) = parabLine j) ∧
      ∀ i : Fin (n + 1), ∃ j ≤ n, dot q (parabLine i) ≤ dot q (parabLine j) :=
  ⟨parabLine_slope, parabLine_bp, fun j hj => ⟨j, hj, neg_neg _⟩,
    fun i => ⟨i, Nat.lt_succ_iff.mp i.isLt, le_rfl⟩⟩

end ALM
end Transformer
