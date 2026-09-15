/-
# The machine as an index, and why the barrier does not reach it

Two halves of this development never met.  `Transformer.ALM.BinSearch` proves
that the planar hull answers a lookup in `log₂ n + 1` comparisons;
`Transformer.ALM.Hardness` proves that an exact index cannot answer one in
fewer than `n^{1-ε}` at dimension `Θ(log n)`.  Both are theorems, and they
speak about different objects: the first about a procedure, the second about
an `NNIndex`.  Nothing in Lean compared them, so the reconciliation — "the
machine runs at dimension one, where the barrier is silent" — was prose.

This module makes it a theorem.  `hullIndex` is an `NNIndex` whose answer in
dimension one *is* the hull's binary search over the sorted keys of
`Transformer.ALM.KeyOrder`, and whose declared query price there is exactly
the comparison count `bcount` that search pays (`hullIndex_query_paid`) — the
opposite of `Transformer.ALM.Independence`'s `freeIndex`, whose prices are
invented.  Outside dimension one it falls back to a scan and is charged for
it.

Then the reconciliation, in three statements:

* `reduction_dimension_even` — the lookup the Orthogonal Vectors reduction
  performs is never one-dimensional.  `Implements.cost_le` queries the index
  at `d + d`, and that is even, so the logarithmic price is never the one the
  reduction pays.  The machine's regime and the barrier's regime are disjoint
  for a reason that needs no conjecture.
* `naiveModel_implements_hullIndex` — the reduction *is* implemented by this
  index, in a model where Orthogonal Vectors is hard.
* `hullIndex_consistent_with_OVHard` — so the two coexist: a logarithmic
  exact lookup in dimension one and the quadratic barrier at dimension
  `Θ(log n)` are simultaneously satisfiable.  The machine is not a
  counterexample to the conjecture, and the conjecture is not an obstacle to
  the machine.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 203-215.
-/

import Transformer.ALM.KeyOrder
import Transformer.ALM.Hardness

namespace Transformer
namespace ALM

variable {n : ℕ}

/-! ### The hull's answer on an arbitrary family of scalar keys -/

/-- The index `lower_bound(q)` returns, run over the sorted distinct keys. -/
noncomputable def hullProbe [Nonempty (Fin n)] (K : Fin n → ℝ) (q : ℝ) : ℕ :=
  bsearch (fun j => decide (q ≤ (sortedKey K j + sortedKey K (j + 1)) / 2)) 0
    (keyCard K - 1)

lemma hullProbe_le [Nonempty (Fin n)] (K : Fin n → ℝ) (q : ℝ) :
    hullProbe K q ≤ keyCard K - 1 := by
  simpa [hullProbe] using
    bsearch_le (fun j => decide (q ≤ (sortedKey K j + sortedKey K (j + 1)) / 2)) 0
      (keyCard K - 1)

/-- **The search still wins against the original family.**  The hull sees only
the sorted distinct keys, but every key of the family is one of them, so the
line it returns is on top at `q` against all of them. -/
theorem hullProbe_isGreatest [Nonempty (Fin n)] (K : Fin n → ℝ) (q : ℝ) (i : Fin n) :
    lineEval (liftKey (K i)) q
      ≤ lineEval (liftKey (sortedKey K (hullProbe K q))) q := by
  obtain ⟨j, hj, hji⟩ := exists_sortedKey_eq K i
  have h := (hull_bsearch_isGreatest (sortedKey K) q (keyCard K - 1)
    (sortedKey_lt_succ K)).1 j hj
  rw [hji] at h
  exact h

/-- The key of the original family that the hull's answer names. -/
noncomputable def hullIdx [Nonempty (Fin n)] (K : Fin n → ℝ) (q : ℝ) : Fin n :=
  (exists_eq_sortedKey K (hullProbe_le K q)).choose

lemma hullIdx_spec [Nonempty (Fin n)] (K : Fin n → ℝ) (q : ℝ) :
    K (hullIdx K q) = sortedKey K (hullProbe K q) :=
  (exists_eq_sortedKey K (hullProbe_le K q)).choose_spec

theorem hullIdx_isGreatest [Nonempty (Fin n)] (K : Fin n → ℝ) (q : ℝ) (i : Fin n) :
    lineEval (liftKey (K i)) q ≤ lineEval (liftKey (K (hullIdx K q))) q := by
  rw [hullIdx_spec K q]
  exact hullProbe_isGreatest K q i

/-- **What the answer costs.**  Deduplicating can only shorten the list, so
the comparison count of `bcount_le_log` is bounded by the logarithm of the
number of keys the index was built on. -/
theorem hullProbe_cost [Nonempty (Fin n)] (K : Fin n → ℝ) :
    bcount (keyCard K - 1) ≤ Nat.log 2 n + 1 :=
  le_trans (bcount_le_log _)
    (Nat.add_le_add_right
      (Nat.log_mono_right (le_trans (Nat.sub_le _ _) (keyCard_le K))) 1)

/-! ### The same, as an answer about scores -/

/-- In dimension one the hull's answer is an argmax of the attention score. -/
theorem hullIdx_isGreatest_score [Nonempty (Fin n)] (K : Fin n → EucSpace 1)
    (q : EucSpace 1) (i : Fin n) :
    score q (K i) ≤ score q (K (hullIdx (fun j => K j 0) (q 0))) := by
  have h := hullIdx_isGreatest (fun j => K j 0) (q 0) i
  rw [score_eq_lineEval, score_eq_lineEval]
  exact h

/-- The index's answer: the hull in dimension one, an exhaustive scan
elsewhere — which is what a planar hull can honestly claim. -/
noncomputable def hullAns {m n : ℕ} [Nonempty (Fin n)] (K : Fin n → EucSpace m)
    (q : EucSpace m) : Fin n :=
  match m, K, q with
  | 1, K, q => hullIdx (fun j => K j 0) (q 0)
  | _, _, _ => bfAns K q

theorem hullAns_isGreatest {m n : ℕ} [Nonempty (Fin n)] (K : Fin n → EucSpace m)
    (q : EucSpace m) (j : Fin n) : score q (K j) ≤ score q (K (hullAns K q)) := by
  match m, K, q with
  | 0, K, q => exact bfAns_isGreatest K q j
  | 1, K, q => exact hullIdx_isGreatest_score K q j
  | (_ + 2), K, q => exact bfAns_isGreatest K q j

/-! ### The index -/

/-- **The machine as an index.**  Exact in every dimension; logarithmic in the
one it is built for, where the price is the comparison count of the binary
search, and a scan elsewhere.  Building sorts the keys. -/
noncomputable def hullIndex : NNIndex where
  ans := fun K q => hullAns K q
  ans_isGreatest := fun K q j => hullAns_isGreatest K q j
  build := fun n m => if m = 1 then 3 * (n : ℝ) * ((Nat.log 2 n : ℝ) + 1) else 0
  query := fun n m => if m = 1 then (Nat.log 2 n : ℝ) + 1 else (n : ℝ) * (m : ℝ)

/-- **The declared price is the price paid.**  In dimension one the query cost
of `hullIndex` is not a number put into a field: it dominates `bcount`, the
comparisons `bsearch` actually makes on the keys it was built on. -/
theorem hullIndex_query_paid [Nonempty (Fin n)] (K : Fin n → ℝ) :
    ((bcount (keyCard K - 1) : ℕ) : ℝ) ≤ hullIndex.query n 1 := by
  have h := hullProbe_cost K
  have hq : hullIndex.query n 1 = (Nat.log 2 n : ℝ) + 1 := by
    show (if (1 : ℕ) = 1 then (Nat.log 2 n : ℝ) + 1 else (n : ℝ) * ((1 : ℕ) : ℝ)) = _
    rw [ite_eq_left rfl]
  rw [hq]
  exact_mod_cast h

/-! ### Why the barrier never charges that price -/

/-- **The reduction never makes a one-dimensional lookup.**  `Implements`
queries the index at `d + d`, so the dimension it asks about is even and
never `1`: the logarithmic branch of `hullIndex` is unreachable from the
Orthogonal Vectors side, whatever `d` is. -/
theorem reduction_dimension_even (n d : ℕ) :
    hullIndex.query n (d + d) = (n : ℝ) * ((d + d : ℕ) : ℝ) := by
  show (if d + d = 1 then (Nat.log 2 n : ℝ) + 1 else (n : ℝ) * ((d + d : ℕ) : ℝ)) = _
  rw [ite_eq_right (by omega)]

/-- The quadratic model implements the reduction through `hullIndex`, at
exactly the accounted cost — the same accounting as for `bruteForce`, because
the reduction only ever reaches the scanning branch. -/
noncomputable def naiveModel_implements_hullIndex : naiveModel.Implements hullIndex where
  alg := ()
  decides_reduce := fun A B => (hullIndex.reduce_iff A B).symm
  cost_le := fun n d => by
    rw [reduction_dimension_even n d]
    show 2 * (n : ℝ) ^ 2 * (d : ℝ)
      ≤ (if d + d = 1 then 3 * (n : ℝ) * ((Nat.log 2 n : ℝ) + 1) else 0)
        + (n : ℝ) * ((n : ℝ) * ((d + d : ℕ) : ℝ))
    rw [ite_eq_right (by omega)]
    push_cast
    ring_nf
    exact le_refl _

/-- **The machine and the barrier coexist.**  There is a model in which
Orthogonal Vectors is hard, the reduction through `hullIndex` is implemented,
and the index still answers a one-dimensional lookup in `log₂ n + 1`.  So the
logarithmic lookup of `hull2d_cht.h` neither refutes the conjecture nor is
refuted by it: the two speak about different dimensions, and
`reduction_dimension_even` says which. -/
theorem hullIndex_consistent_with_OVHard :
    ∃ M : CostModel, M.OVHard ∧ Nonempty (M.Implements hullIndex) ∧
      ∀ n : ℕ, hullIndex.query n 1 = (Nat.log 2 n : ℝ) + 1 :=
  ⟨naiveModel, naiveModel_OVHard, ⟨naiveModel_implements_hullIndex⟩, fun n => by
    show (if (1 : ℕ) = 1 then (Nat.log 2 n : ℝ) + 1 else (n : ℝ) * ((1 : ℕ) : ℝ)) = _
    rw [ite_eq_left rfl]⟩

/-- The hypotheses are satisfiable: `Fin 3` is nonempty, and on the scalar keys
`0, 1, 2` the index really returns a best-scoring one. -/
example : ∀ j : Fin 3,
    score (WithLp.toLp 2 ![(1.9 : ℝ)]) (WithLp.toLp 2 ![(j : ℝ)])
      ≤ score (WithLp.toLp 2 ![(1.9 : ℝ)])
          ((fun i : Fin 3 => WithLp.toLp 2 ![(i : ℝ)])
            (hullIndex.ans (fun i : Fin 3 => WithLp.toLp 2 ![(i : ℝ)])
              (WithLp.toLp 2 ![(1.9 : ℝ)]))) :=
  fun j => hullIndex.ans_isGreatest (fun i : Fin 3 => WithLp.toLp 2 ![(i : ℝ)])
    (WithLp.toLp 2 ![(1.9 : ℝ)]) j

end ALM
end Transformer
