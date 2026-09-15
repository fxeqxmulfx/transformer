/-
# The breakpoint the envelope caches, and the one write that repairs it

`Transformer.ALM.HullErase` says which lines `add_line` may drop and
`Transformer.ALM.HullPrune` carries those rules across a whole build, both over
a `Finset (ℝ × ℝ)`.  A `Finset` has no cursors, and the port does not run on
one: `vm-rs/alm-hull/src/envelope.rs` keeps the envelope as a list of nodes in
increasing slope order, and beside each node it caches the breakpoint at which
that node gives way to the next one.  Every comparison in both erase loops
reads that cache rather than recomputing it (`new.p >= self.t.break_of(hi)`),
so the cache is as much a part of the answer as the lines are.

What is not obvious, and is what this file states, is how little the surgery
has to write.  `add_line` erases a contiguous run of nodes, links the new line
where the run was, and then writes exactly one cached breakpoint --
`self.t.set_break(cur, p)`, at the node left of the run.  Nodes before `cur`
are not touched, nodes after the run are not touched, and none of them is even
read.  `cacheOk_splice` is that this is enough: the cache is a property of
adjacent pairs, the splice changes two adjacencies, and one write and the
value the successor walk already computed for the new line cover both.

`cacheOk_splice_dropped` is the other branch, where the predecessor test finds
the new line already hidden (`keep = false`) and the run closes over nothing.
The one write is the same write, against a different right neighbour, which is
exactly the `right` binding the port chooses before the walk.

Breakpoints live in `WithTop ℝ`: `⊤` is `Break::PosInf`, the value the
successor walk stores when it runs off the end of the envelope.  The other two
constructors `Break` carries are not reachable here -- `Unset` is a node not
yet linked and `NegInf` needs two equal slopes, which the envelope's slope
order forbids between neighbours.

Source: `vm-rs/alm-hull/src/envelope.rs`, `Envelope::add_line`, and the
`_HullCHT::add_line` it is ported from, `transformer_vm/attention/hull2d_cht.h`
lines 122-192.
-/

import Transformer.ALM.Envelope

namespace Transformer
namespace ALM

/-- **A node of the envelope, as the port stores it**: the line, and the
breakpoint cached beside it. -/
abbrev HullNode : Type := (ℝ × ℝ) × WithTop ℝ

/-- **The breakpoint a node ought to be carrying**: where its line gives way to
the line after it, and `Break::PosInf` when there is no line after it. -/
noncomputable def breakTo (l : ℝ × ℝ) : Option (ℝ × ℝ) → WithTop ℝ
  | some l' => ((interX l l' : ℝ) : WithTop ℝ)
  | none => ⊤

/-- **The cache is right at this adjacency.**  A relation on neighbours, which
is the whole point: nothing about a node's cached breakpoint depends on any
node but the one after it. -/
def Cached (a b : HullNode) : Prop := a.2 = breakTo a.1 (some b.1)

/-- **And right along the whole envelope**: at every adjacency, and `PosInf` at
the end. -/
def CacheOk (e : List HullNode) : Prop :=
  e.IsChain Cached ∧ ∀ z ∈ e.getLast?, z.2 = ⊤

/-! ### Two small facts the splice needs -/

/-- **A node's cached value is invisible from its left.**  `Cached a b` reads
`b`'s line and not `b`'s cache, so rewriting the cache at the end of a chain
leaves the chain a chain.  This is what lets `set_break(cur, p)` be a write and
not a rebuild. -/
theorem isChain_append_singleton_congr {pre : List HullNode} {a b : HullNode}
    (hab : a.1 = b.1) (h : (pre ++ [a]).IsChain Cached) : (pre ++ [b]).IsChain Cached := by
  rw [List.isChain_append] at h ⊢
  refine ⟨h.1, List.IsChain.singleton _, fun x hx y hy => ?_⟩
  have hy' : y = b := by simp at hy; exact hy.symm
  have hxa := h.2.2 x hx a (by simp)
  unfold Cached at hxa ⊢
  rw [hy', ← hab]
  exact hxa

/-- **The value the successor walk stops with is the value the cache wants.**
When the walk runs off the end it stores `PosInf`, and that is `breakTo`
against no successor. -/
theorem breakTo_none (l : ℝ × ℝ) : breakTo l none = ⊤ := rfl

/-! ### The splice -/

/-- **One write repairs the cache.**  The envelope is `pre ++ c :: (mid ++ post)`;
`add_line` erases `mid`, links `new` where it was, and writes one cached
breakpoint, at `c`.  Then the cache is right again -- given only that `new`
carries the breakpoint against whatever the run stopped at, which is what the
successor loop leaves in `new.p`.

Nothing in `pre` or `post` appears on the right-hand side except unchanged,
which is the statement's real content: the repair is local. -/
theorem cacheOk_splice {pre mid post : List HullNode} {c new : HullNode}
    (h : CacheOk (pre ++ c :: (mid ++ post)))
    (hnew : new.2 = breakTo new.1 (post.head?.map Prod.fst)) :
    CacheOk (pre ++ (c.1, breakTo c.1 (some new.1)) :: new :: post) := by
  obtain ⟨hchain, hlast⟩ := h
  rw [List.isChain_split] at hchain
  obtain ⟨hpre, htail⟩ := hchain
  have hpost : post.IsChain Cached := (List.isChain_append.mp htail.tail).2.1
  refine ⟨List.isChain_append_cons_cons.mpr
    ⟨isChain_append_singleton_congr (a := c) (b := (c.1, breakTo c.1 (some new.1)))
      rfl hpre, rfl, ?_⟩, ?_⟩
  · refine List.isChain_cons.mpr ⟨fun y hy => ?_, hpost⟩
    rw [Option.mem_def, List.head?_eq_some_iff] at hy
    obtain ⟨ys, hys⟩ := hy
    show new.2 = _
    rw [hnew, hys]
    rfl
  · intro z hz
    rw [List.getLast?_append_cons, List.getLast?_cons_cons] at hz
    cases post with
    | nil =>
      have hzn : new = z := by simpa using hz
      rw [← hzn, hnew]
      rfl
    | cons p ps =>
      rw [List.getLast?_cons_cons] at hz
      refine hlast z ?_
      have heq : pre ++ c :: (mid ++ (p :: ps)) = (pre ++ c :: mid) ++ (p :: ps) := by simp
      rw [heq]
      exact List.mem_getLast?_append_of_mem_getLast? hz

/-- **And the branch that keeps nothing.**  When the predecessor already hides
the new line the port sets `keep = false`, computes the same one breakpoint
against the node the successor walk stopped at instead of against the new line,
and links nothing.  The cache is right again for the same reason. -/
theorem cacheOk_splice_dropped {pre mid post : List HullNode} {c : HullNode}
    (h : CacheOk (pre ++ c :: (mid ++ post))) :
    CacheOk (pre ++ (c.1, breakTo c.1 (post.head?.map Prod.fst)) :: post) := by
  obtain ⟨hchain, hlast⟩ := h
  rw [List.isChain_split] at hchain
  obtain ⟨hpre, htail⟩ := hchain
  have hpost : post.IsChain Cached := (List.isChain_append.mp htail.tail).2.1
  refine ⟨List.isChain_split.mpr
    ⟨isChain_append_singleton_congr
      (a := c) (b := (c.1, breakTo c.1 (post.head?.map Prod.fst))) rfl hpre, ?_⟩, ?_⟩
  · refine List.isChain_cons.mpr ⟨fun y hy => ?_, hpost⟩
    rw [Option.mem_def, List.head?_eq_some_iff] at hy
    obtain ⟨ys, hys⟩ := hy
    show breakTo c.1 (post.head?.map Prod.fst) = _
    rw [hys]
    rfl
  · intro z hz
    rw [List.getLast?_append_cons] at hz
    cases post with
    | nil =>
      have hzn : (c.1, breakTo c.1 (([] : List HullNode).head?.map Prod.fst)) = z := by
        simpa using hz
      rw [← hzn]
      rfl
    | cons p ps =>
      rw [List.getLast?_cons_cons] at hz
      refine hlast z ?_
      have heq : pre ++ c :: (mid ++ (p :: ps)) = (pre ++ c :: mid) ++ (p :: ps) := by simp
      rw [heq]
      exact List.mem_getLast?_append_of_mem_getLast? hz

/-! ### The hypotheses are satisfiable -/

/-- **A two-line envelope with its cache filled in**: `y = 0` gives way to
`y = x` at the origin, and the second line is the last one, so it caches
`PosInf`. -/
theorem the_two_line_cache :
    CacheOk [(((0 : ℝ), (0 : ℝ)), ((0 : ℝ) : WithTop ℝ)), (((1 : ℝ), (0 : ℝ)), ⊤)] := by
  refine ⟨List.isChain_cons_cons.mpr ⟨?_, List.IsChain.singleton _⟩, by simp⟩
  show ((0 : ℝ) : WithTop ℝ) = breakTo ((0 : ℝ), (0 : ℝ)) (some ((1 : ℝ), (0 : ℝ)))
  norm_num [breakTo, interX]

/-- Both of `cacheOk_splice`'s hypotheses at once, on that envelope: the line
`y = 2x - 1` arrives, the successor walk stops at `y = x`, and nothing is
erased. -/
example :
    CacheOk [(((0 : ℝ), (0 : ℝ)), breakTo ((0 : ℝ), (0 : ℝ)) (some ((2 : ℝ), (-1 : ℝ)))),
      (((2 : ℝ), (-1 : ℝ)), breakTo ((2 : ℝ), (-1 : ℝ)) (some ((1 : ℝ), (0 : ℝ)))),
      (((1 : ℝ), (0 : ℝ)), (⊤ : WithTop ℝ))] :=
  cacheOk_splice (pre := []) (mid := []) (c := (((0 : ℝ), (0 : ℝ)), ((0 : ℝ) : WithTop ℝ)))
    (new := (((2 : ℝ), (-1 : ℝ)), breakTo ((2 : ℝ), (-1 : ℝ)) (some ((1 : ℝ), (0 : ℝ)))))
    (post := [(((1 : ℝ), (0 : ℝ)), (⊤ : WithTop ℝ))]) the_two_line_cache rfl

/-- And `cacheOk_splice_dropped`'s one hypothesis, on the same envelope: the
arriving line is hidden, the first line is rewritten against the second, and
the envelope is the one it started as. -/
example :
    CacheOk [(((0 : ℝ), (0 : ℝ)),
        breakTo ((0 : ℝ), (0 : ℝ))
          (([(((1 : ℝ), (0 : ℝ)), (⊤ : WithTop ℝ))] : List HullNode).head?.map Prod.fst)),
      (((1 : ℝ), (0 : ℝ)), (⊤ : WithTop ℝ))] :=
  cacheOk_splice_dropped (pre := []) (mid := []) (c := (((0 : ℝ), (0 : ℝ)), ((0 : ℝ) : WithTop ℝ)))
    (post := [(((1 : ℝ), (0 : ℝ)), (⊤ : WithTop ℝ))]) the_two_line_cache

/-- The congruence's hypotheses, at the same envelope's first node: the cache
it carries is replaced by the one the splice computes, and the chain before it
does not notice. -/
example :
    ([(((0 : ℝ), (0 : ℝ)), breakTo ((0 : ℝ), (0 : ℝ)) (some ((2 : ℝ), (-1 : ℝ))))]
      : List HullNode).IsChain Cached :=
  isChain_append_singleton_congr (pre := []) rfl (List.IsChain.singleton _)

end ALM
end Transformer
