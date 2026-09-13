/-
# The whole query in floating point

Three separate results now describe `HullHalf::query`, each about a different
part of the arithmetic: `Transformer.ALM.FloatLattice` says the search lands
on a winner even though `isect` is rounded, `Transformer.ALM.FloatTie` says
the `==` the merge loops branch on fires on winners and on nothing else, and
`Transformer.ALM.HullCost` says a tie set is a winner alone or a winner with
one neighbour.  What the function does is their composition, and that
composition was prose.

`fpTieSet` is the set of lines the two loops accept: those whose recomputed
score compares equal to `best_score`, around the index the floating-point
search returned.  Four statements about it, the last three on the integer data
the `LookUp` primitive stores:

* `fpProbe_mem_fpTieSet` — it is never empty: the probe compares equal to
  itself, so `combined.resolve` always has a line to resolve.
* `fp_walk_sound` — it contains no line that is not an exact maximizer, with
  no exactness assumed of the score routine, only `2δ < 1`.
* `fp_walk_collects` — with the scores computed exactly it is precisely
  `argmaxSet`, the set the development resolves; `Transformer.ALM.FloatGrid`
  gets that exactness out of a grid condition a rounded routine can meet.
* `fp_walk_trichotomy` — and so it is the winner alone, or the winner with one
  neighbour: each loop runs at most once, and `combined.resolve` is handed
  exactly what `Transformer.ALM.HullResolve` says it is handed.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 268-306.
-/

import Transformer.ALM.FloatTie

namespace Transformer
namespace ALM

open Classical

variable {n : ℕ}

/-- **The lines the two loops accept.**  Everything in range whose recomputed
score compares equal to `best_score`, the comparison being the `==` of the
loops and the centre being the index the floating-point search returned.

Source: `hull2d_cht.h`, lines 277-303. -/
noncomputable def fpTieSet (S : FPScore) (F : FPArith) [Nonempty (Fin n)]
    (K : Fin n → ℝ) (q : ℝ) : Finset ℕ :=
  (Finset.range (keyCard K)).filter (fun j =>
    S.eval (liftQuery q) (liftKey (sortedKey K j))
      = S.eval (liftQuery q) (liftKey (sortedKey K (fpProbe F K q))))

/-- **The probe is always kept.**  The comparison the loops make is an
equality with the score at the probe itself, so the probe passes it, and the
set the two `while` loops hand to `combined.resolve` is never empty: whatever
the rounding does, the line the search landed on is resolved.

Source: `hull2d_cht.h`, lines 277-303. -/
theorem fpProbe_mem_fpTieSet (S : FPScore) (F : FPArith) [Nonempty (Fin n)]
    (K : Fin n → ℝ) (q : ℝ) : fpProbe F K q ∈ fpTieSet S F K q := by
  refine Finset.mem_filter.mpr ⟨Finset.mem_range.mpr ?_, rfl⟩
  have hle := fpProbe_le F K q
  have hpos := keyCard_pos K
  omega

/-- And so the tie set is nonempty, for every score routine and every
query. -/
theorem fpTieSet_nonempty (S : FPScore) (F : FPArith) [Nonempty (Fin n)]
    (K : Fin n → ℝ) (q : ℝ) : (fpTieSet S F K q).Nonempty :=
  ⟨fpProbe F K q, fpProbe_mem_fpTieSet S F K q⟩

/-- **The walk accepts no loser.**  On integer keys at an integer query, with
the breakpoints accurate to better than `1/2` and the scores to better than
`1/2` each, every line the loops merge is an exact maximizer.  Nothing is
assumed exact: this is soundness of the running code in the arithmetic it
runs in. -/
theorem fp_walk_sound (S : FPScore) (F : FPArith) [Nonempty (Fin n)] (K : Fin n → ℝ)
    (hK : ∀ i, ∃ z : ℤ, K i = (z : ℝ)) (q : ℤ) (M : ℝ)
    (hbd : ∀ j ≤ keyCard K - 1, |sortedKey K j| ≤ M) (hu : F.u * M < 1 / 2)
    (hδ : 2 * S.δ < 1) :
    fpTieSet S F K (q : ℝ) ⊆ argmaxSet (sortedKey K) (q : ℝ) (keyCard K - 1) := by
  intro j hj
  simp only [fpTieSet, Finset.mem_filter, Finset.mem_range] at hj
  have hpos := keyCard_pos K
  exact fp_tie_no_false_positive S (sortedKey K) (sortedKey_int K hK) q (keyCard K - 1) hδ
    (fpProbe_mem_argmaxSet_of_int F K hK q M hbd hu) (by omega) hj.2

/-- **And misses no winner.**  With the scores computed exactly the set the
loops collect is exactly the tie set the development resolves;
`fp_walk_collects_of_grid` weakens that hypothesis to the grid condition. -/
theorem fp_walk_collects (S : FPScore) (F : FPArith) [Nonempty (Fin n)] (K : Fin n → ℝ)
    (hK : ∀ i, ∃ z : ℤ, K i = (z : ℝ)) (q : ℤ) (M : ℝ)
    (hbd : ∀ j ≤ keyCard K - 1, |sortedKey K j| ≤ M) (hu : F.u * M < 1 / 2)
    (hexact : ∀ i ≤ keyCard K - 1, S.eval (liftQuery (q : ℝ)) (liftKey (sortedKey K i))
      = dot (liftQuery (q : ℝ)) (liftKey (sortedKey K i))) :
    fpTieSet S F K (q : ℝ) = argmaxSet (sortedKey K) (q : ℝ) (keyCard K - 1) := by
  have hpos := keyCard_pos K
  have hb := fpProbe_mem_argmaxSet_of_int F K hK q M hbd hu
  have hbN := ((mem_argmaxSet (sortedKey K) (q : ℝ) (keyCard K - 1)).mp hb).1
  refine Finset.Subset.antisymm (fun j hj => ?_) (fun j hj => ?_)
  · simp only [fpTieSet, Finset.mem_filter, Finset.mem_range] at hj
    have hjN : j ≤ keyCard K - 1 := by omega
    have heq := hj.2
    rw [hexact j hjN, hexact _ hbN, dot_liftQuery, dot_liftQuery] at heq
    refine (mem_argmaxSet (sortedKey K) (q : ℝ) (keyCard K - 1)).mpr ⟨hjN, fun i hi => ?_⟩
    rw [heq]
    exact ((mem_argmaxSet (sortedKey K) (q : ℝ) (keyCard K - 1)).mp hb).2 i hi
  · have hjN := ((mem_argmaxSet (sortedKey K) (q : ℝ) (keyCard K - 1)).mp hj).1
    simp only [fpTieSet, Finset.mem_filter, Finset.mem_range]
    exact ⟨by omega, fp_tie_no_false_negative S (sortedKey K) (q : ℝ) (keyCard K - 1)
      hexact hb hj⟩

/-- **So the loops run at most once each.**  What `combined.resolve` receives
is the winner alone, or the winner together with one neighbour — the shape
`Transformer.ALM.HullResolve` reads a value off, now established for the index
the floating-point search returns and the comparison the floating-point loops
make, rather than for their real-arithmetic idealizations. -/
theorem fp_walk_trichotomy (S : FPScore) (F : FPArith) [Nonempty (Fin n)] (K : Fin n → ℝ)
    (hK : ∀ i, ∃ z : ℤ, K i = (z : ℝ)) (q : ℤ) (M : ℝ)
    (hbd : ∀ j ≤ keyCard K - 1, |sortedKey K j| ≤ M) (hu : F.u * M < 1 / 2)
    (hexact : ∀ i ≤ keyCard K - 1, S.eval (liftQuery (q : ℝ)) (liftKey (sortedKey K i))
      = dot (liftQuery (q : ℝ)) (liftKey (sortedKey K i))) :
    fpTieSet S F K (q : ℝ) = {fpProbe F K (q : ℝ)} ∨
      fpTieSet S F K (q : ℝ) = {fpProbe F K (q : ℝ), fpProbe F K (q : ℝ) + 1} ∨
      fpTieSet S F K (q : ℝ) = {fpProbe F K (q : ℝ) - 1, fpProbe F K (q : ℝ)} := by
  rw [fp_walk_collects S F K hK q M hbd hu hexact]
  exact argmaxSet_trichotomy (sortedKey K) (q : ℝ) (keyCard K - 1) (sortedKey_lt_succ K)
    (fpProbe_mem_argmaxSet_of_int F K hK q M hbd hu)

/-- All of it is satisfiable at once, for every integer family and every
integer query: finitely many sorted keys are bounded, and exact arithmetic
meets the three error conditions with room to spare. -/
example [Nonempty (Fin n)] (K : Fin n → ℤ) (q : ℤ) :
    (∀ i, ∃ z : ℤ, ((K i : ℝ)) = (z : ℝ)) ∧ 2 * exactScore.δ < 1 ∧
      (∀ i ≤ keyCard (fun i => (K i : ℝ)) - 1,
        exactScore.eval (liftQuery (q : ℝ)) (liftKey (sortedKey (fun i => (K i : ℝ)) i))
          = dot (liftQuery (q : ℝ)) (liftKey (sortedKey (fun i => (K i : ℝ)) i))) ∧
      ∃ M : ℝ, (∀ j ≤ keyCard (fun i => (K i : ℝ)) - 1,
          |sortedKey (fun i => (K i : ℝ)) j| ≤ M) ∧ exactArith.u * M < 1 / 2 := by
  refine ⟨fun i => ⟨K i, rfl⟩, by norm_num [exactScore], fun i _ => rfl, ?_⟩
  obtain ⟨b, -, hb⟩ := Finset.exists_max_image (Finset.range (keyCard (fun i => (K i : ℝ))))
    (fun j => |sortedKey (fun i => (K i : ℝ)) j|)
    ⟨0, Finset.mem_range.mpr (keyCard_pos (fun i => (K i : ℝ)))⟩
  refine ⟨|sortedKey (fun i => (K i : ℝ)) b|, fun j hj => hb j (Finset.mem_range.mpr ?_), ?_⟩
  · have hpos := keyCard_pos (fun i => (K i : ℝ))
    omega
  · simp [exactArith]

end ALM
end Transformer
