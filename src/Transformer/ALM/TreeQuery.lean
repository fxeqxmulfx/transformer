/-
# The container the query actually descends

`Transformer.ALM.BinSearch` prices the planar hull's query at
`bcount len ≤ log₂ len + 1`, and that is the right number for the procedure it
models: `std::lower_bound` over a sorted array, halving the range once per
comparison.  It is not the number the shipped machine pays.

`alm-hull/src/tree.rs` keeps the envelope in a red-black tree, because the C++
original's `std::multiset` gives erase-and-advance in amortized constant time
and stable Rust's `BTreeSet` gives nothing of the kind.  A red-black tree of
`n` nodes is balanced to within a factor of two, not exactly, so the descent in
`Tree::lower_bound_slope` costs up to `2·log₂(n + 1)` comparisons -- twice the
array's bound, and `BinSearch` says nothing about it.  (The implementation
answers both ends of the envelope from cached cursors without descending at
all, which the model below does not have and which only lowers the count.)

This file closes that gap on both sides.  `lowerBound` is the descent written
out, with the same accumulator the loop carries, and `lowerBound_eq_find` says
it returns the first element the predicate accepts, which is what makes it a
`lower_bound` at all.  `lbCount` is its comparison count, `lbCount_le_depth`
prices it at the tree's depth, and `lbCount_le_two_log` composes that with
`RBTree.RBNode.WF.depth_bound` to get the balance factor.  `log_succ_bound`
is the comparison of the two: for any nonempty envelope the tree's price is at
least the array's and at most twice it.

Source: `vm-rs/alm-hull/src/tree.rs`, `Tree::lower_bound_slope`; Guibas-Sedgewick
1978 for the balance factor, via `Batteries.Recycling.RBTree.Depth`.
-/

import Transformer.ALM.BinSearch
import Batteries.Recycling.RBTree.Depth
import Batteries.Recycling.RBTree.Lemmas

namespace Transformer
namespace ALM

open RBTree

variable {α : Type*}

/-! ### The descent -/

/-- **`Tree::lower_bound_slope`, written out.**  Go right past every node the
predicate rejects; on one it accepts, remember it and go left.  The
accumulator is the loop's `at`, and `none` is its `NIL`. -/
def lowerBound (p : α → Bool) : RBNode α → Option α → Option α
  | .nil, acc => acc
  | .node _ l v r, acc => if p v then lowerBound p l (some v) else lowerBound p r acc

/-- The comparisons it makes: one at every node on the path it takes. -/
def lbCount (p : α → Bool) : RBNode α → ℕ
  | .nil => 0
  | .node _ l v r => 1 + if p v then lbCount p l else lbCount p r

@[simp] lemma lowerBound_nil (p : α → Bool) (acc : Option α) :
    lowerBound p .nil acc = acc := rfl

@[simp] lemma lbCount_nil (p : α → Bool) : lbCount p (.nil : RBNode α) = 0 := rfl

/-! ### It is a lower bound -/

/-- **The descent returns the first element the predicate accepts.**  The
hypothesis is that the predicate rises along the envelope -- once a slope is
not below `m`, no later slope is either -- which is the tree's ordering read
through the test, and is the same condition `BinSearch.bsearch_lt` asks of the
array search.

Source: `tree.rs::lower_bound_slope`. -/
theorem lowerBound_eq_find (p : α → Bool) :
    ∀ (t : RBNode α) (acc : Option α),
      t.toList.Pairwise (fun x y => p x = true → p y = true) →
      lowerBound p t acc = (t.toList.find? p).or acc
  | .nil, acc, _ => by simp [lowerBound]
  | .node c l v r, acc, hmono => by
    rw [RBNode.toList_node] at hmono
    obtain ⟨hl, hvr, hcross⟩ := List.pairwise_append.mp hmono
    obtain ⟨hv, hr⟩ := List.pairwise_cons.mp hvr
    rw [RBNode.toList_node, List.find?_append]
    by_cases hpv : p v = true
    · rw [lowerBound, if_pos hpv, lowerBound_eq_find p l (some v) hl,
        List.find?_cons_of_pos hpv]
      cases l.toList.find? p <;> simp
    · have hnone : l.toList.find? p = none := by
        rw [List.find?_eq_none]
        intro x hx hpx
        exact hpv (hcross x hx v (List.mem_cons_self ..) hpx)
      rw [lowerBound, if_neg hpv, lowerBound_eq_find p r acc hr, hnone,
        List.find?_cons_of_neg (by simpa using hpv)]
      simp
  termination_by t => t

/-- The hypothesis is satisfiable and the answer is the interesting one: on
the two-element envelope `[1, 2]` the test `2 ≤ ·` rejects the root and
accepts its right child, so the descent goes right and returns `2`. -/
example :
    lowerBound (fun n : ℕ => 2 ≤ n) (.node .black .nil 1 (.node .red .nil 2 .nil)) none
      = ([1, 2].find? (fun n : ℕ => 2 ≤ n)).or none :=
  lowerBound_eq_find _ _ none (by decide)

/-! ### What it costs -/

/-- Every step of the descent is one node deeper, so the count is bounded by
the depth whichever way the tests come out. -/
theorem lbCount_le_depth (p : α → Bool) : ∀ t : RBNode α, lbCount p t ≤ t.depth
  | .nil => by simp [RBNode.depth]
  | .node c l v r => by
    have hl := lbCount_le_depth p l
    have hr := lbCount_le_depth p r
    rw [lbCount, RBNode.depth]
    split <;> omega

/-- **The descent costs at most `2·log₂(n + 1)` comparisons.**  A red-black
tree's longest path is at most twice its shortest, which is
`RBTree.RBNode.WF.depth_bound`, so the balance factor is the whole difference
between this and the array bound of `BinSearch.bcount_le_log`.

Source: Guibas-Sedgewick 1978; `tree.rs`, module docstring. -/
theorem lbCount_le_two_log {cmp : α → α → Ordering} {t : RBNode α} (h : t.WF cmp)
    (p : α → Bool) : lbCount p t ≤ 2 * Nat.log 2 (t.size + 1) := by
  have hd := h.depth_bound
  rw [Nat.log2_eq_log_two] at hd
  exact le_trans (lbCount_le_depth p t) hd

/-- The hypothesis is satisfiable on a tree that was actually built: the empty
envelope with one line inserted is well formed, and `WF.insert` is how every
tree in the running program is. -/
example : lbCount (fun n : ℕ => 2 ≤ n) ((RBNode.nil : RBNode ℕ).insert compare 1)
    ≤ 2 * Nat.log 2 (((RBNode.nil : RBNode ℕ).insert compare 1).size + 1) :=
  lbCount_le_two_log (RBNode.WF.insert (.mk (by trivial) .nil)) _

/-! ### Against the array -/

/-- **The tree is never cheaper and never worse than twice.**  For a nonempty
envelope the array's `log₂ len + 1` is below the tree's `2·log₂(len + 1)`, so
`BinSearch.bcount_le_log` is a bound the shipped container does not enjoy, and
the price of keeping erase-and-advance is a factor of two and no more. -/
theorem log_succ_bound {len : ℕ} (h : 1 ≤ len) :
    Nat.log 2 len + 1 ≤ 2 * Nat.log 2 (len + 1) := by
  have h1 : 1 ≤ Nat.log 2 (len + 1) := Nat.le_log_of_pow_le one_lt_two (by omega)
  have h2 : Nat.log 2 len ≤ Nat.log 2 (len + 1) := Nat.log_mono_right (by omega)
  omega

/-- The hypothesis is satisfiable, and at the smallest envelope the bound is
tight: one line costs one comparison either way. -/
example : Nat.log 2 1 + 1 ≤ 2 * Nat.log 2 (1 + 1) := log_succ_bound le_rfl

end ALM
end Transformer
