/-
# The three comparisons the splice creates, and the three tests that make them

`Transformer.ALM.HullMono` is the invariant one triple at a time.  This file is
the surgery: `add_line` erases a contiguous run, links the new line where the
run was, and the result has to have increasing breakpoints again.

The list before is `pre ++ c :: (mid ++ post)` and the list after is
`pre ++ (c.1, breakTo c.1 new.1) :: new :: post`, so exactly three adjacencies
are new -- the one into the rewritten `c`, the one from `c` to `new`, and the
one from `new` into `post` -- and the rest of the order is the order that was
already there.  `breakOrd_splice` is that, and what makes it worth stating is
that all three are comparisons the port has already made:

* `hback` is the predecessor walk's exit, `self.t.break_of(back) < p`;
* `hkeep` is the decision that kept the line at all, the negation of
  `if p >= new.p { keep = false }`;
* `hhi` is the successor walk's exit, `new.p < self.t.break_of(hi)`.

The walk tests the second of those against its first left neighbour and then
moves left without retesting, which is what
`Transformer.ALM.HullMono.interX_lt_interX_widen_left` licenses.

`breakOrd_splice_dropped` is the `keep == false` branch: one node instead of
two, the same two outer comparisons, and no middle one to make.

Source: `vm-rs/alm-hull/src/envelope.rs`, `Envelope::add_line` and the
`well_formed` assertion its tests run after every insertion;
`transformer_vm/attention/hull2d_cht.h` lines 122-192.
-/

import Transformer.ALM.HullMono

namespace Transformer
namespace ALM

/-- **The envelope's second invariant.**  The cached breakpoints increase
along the node list, which is what lets one tree ordered by slope answer a
query stated as a breakpoint. -/
def BreakOrd (e : List HullNode) : Prop := e.IsChain (fun a b => a.2 < b.2)

/-! ### What the surgery keeps -/

/-- **The nodes left of the run are ordered among themselves**, and nothing the
splice does can reach them. -/
theorem breakOrd_pre {pre mid post : List HullNode} {c : HullNode}
    (h : BreakOrd (pre ++ c :: (mid ++ post))) : BreakOrd pre :=
  (List.isChain_append.mp (List.isChain_split.mp h).1).1

/-- **And so are the nodes right of it.**  Together with `breakOrd_pre` these
are the two halves the splice reuses unchanged; the run `mid` is the only part
of the hypothesis that is thrown away. -/
theorem breakOrd_post {pre mid post : List HullNode} {c : HullNode}
    (h : BreakOrd (pre ++ c :: (mid ++ post))) : BreakOrd post :=
  (List.isChain_append.mp (List.isChain_split.mp h).2.tail).2.1

/-! ### Gluing two ordered halves around one or two new nodes -/

/-- **Two new nodes between two ordered halves.**  Three comparisons and
nothing else: into the first, between them, out of the second. -/
theorem breakOrd_append_cons_cons {pre post : List HullNode} {u v : HullNode}
    (hpre : BreakOrd pre) (hback : ∀ z ∈ pre.getLast?, z.2 < u.2) (huv : u.2 < v.2)
    (hhi : ∀ z ∈ post.head?, v.2 < z.2) (hpost : BreakOrd post) :
    BreakOrd (pre ++ u :: v :: post) := by
  refine List.isChain_append.mpr ⟨hpre, ?_, fun x hx y hy => ?_⟩
  · exact List.isChain_cons_cons.mpr ⟨huv, List.isChain_cons.mpr ⟨hhi, hpost⟩⟩
  · have hyu : y = u := by simp at hy; exact hy.symm
    rw [hyu]
    exact hback x hx

/-- **One new node between two ordered halves.**  The same statement with the
middle comparison gone, which is the shape the dropped branch leaves. -/
theorem breakOrd_append_cons {pre post : List HullNode} {u : HullNode}
    (hpre : BreakOrd pre) (hback : ∀ z ∈ pre.getLast?, z.2 < u.2)
    (hhi : ∀ z ∈ post.head?, u.2 < z.2) (hpost : BreakOrd post) :
    BreakOrd (pre ++ u :: post) := by
  refine List.isChain_append.mpr ⟨hpre, ?_, fun x hx y hy => ?_⟩
  · exact List.isChain_cons.mpr ⟨hhi, hpost⟩
  · have hyu : y = u := by simp at hy; exact hy.symm
    rw [hyu]
    exact hback x hx

/-! ### The splice -/

/-- **The splice leaves the breakpoints increasing.**  Everything the erase
loops did is summarised by three comparisons, and each of them is a comparison
the port makes: the left walk stops on `hback`, the keep decision is `hkeep`,
and the right walk stops on `hhi`.  Neither `pre` nor `post` is consulted for
anything but its end. -/
theorem breakOrd_splice {pre mid post : List HullNode} {c new : HullNode}
    (h : BreakOrd (pre ++ c :: (mid ++ post)))
    (hback : ∀ z ∈ pre.getLast?, z.2 < breakTo c.1 (some new.1))
    (hkeep : breakTo c.1 (some new.1) < new.2)
    (hhi : ∀ z ∈ post.head?, new.2 < z.2) :
    BreakOrd (pre ++ (c.1, breakTo c.1 (some new.1)) :: new :: post) :=
  breakOrd_append_cons_cons (breakOrd_pre h) hback hkeep hhi (breakOrd_post h)

/-- **And the branch that links nothing.**  When the predecessor already hides
the arriving line the run closes over it, `c` is rewritten against whatever the
successor walk stopped at, and two comparisons are all that is left to make. -/
theorem breakOrd_splice_dropped {pre mid post : List HullNode} {c : HullNode}
    (h : BreakOrd (pre ++ c :: (mid ++ post)))
    (hback : ∀ z ∈ pre.getLast?, z.2 < breakTo c.1 (post.head?.map Prod.fst))
    (hhi : ∀ z ∈ post.head?, breakTo c.1 (post.head?.map Prod.fst) < z.2) :
    BreakOrd (pre ++ (c.1, breakTo c.1 (post.head?.map Prod.fst)) :: post) :=
  breakOrd_append_cons (breakOrd_pre h) hback hhi (breakOrd_post h)

/-! ### The hypotheses are satisfiable -/

/-- **The two-line envelope, ordered.**  `y = 0` gives way to `y = x` at the
origin and `y = x` runs forever, so the breakpoints are `0` and `PosInf`.  This
is `Transformer.ALM.the_two_line_cache`'s envelope, seen through the other
invariant. -/
theorem the_two_line_order :
    BreakOrd [(((0 : ℝ), (0 : ℝ)), ((0 : ℝ) : WithTop ℝ)), (((1 : ℝ), (0 : ℝ)), ⊤)] :=
  List.isChain_cons_cons.mpr ⟨WithTop.coe_lt_top _, List.IsChain.singleton _⟩

/-- Both halves the splice reuses, read off that envelope. -/
example :
    BreakOrd ([] : List HullNode) ∧ BreakOrd [(((1 : ℝ), (0 : ℝ)), (⊤ : WithTop ℝ))] :=
  ⟨breakOrd_pre (pre := []) (mid := []) the_two_line_order,
    breakOrd_post (pre := []) (mid := []) the_two_line_order⟩

/-- All three of `breakOrd_splice`'s comparisons at once: `y = 2x - 1` arrives
past the end of that envelope, `y = x` gives way to it at `1`, and the new line
runs forever. -/
example :
    BreakOrd [(((0 : ℝ), (0 : ℝ)), ((0 : ℝ) : WithTop ℝ)),
      (((1 : ℝ), (0 : ℝ)), breakTo ((1 : ℝ), (0 : ℝ)) (some ((2 : ℝ), (-1 : ℝ)))),
      (((2 : ℝ), (-1 : ℝ)), (⊤ : WithTop ℝ))] := by
  refine breakOrd_splice (pre := [(((0 : ℝ), (0 : ℝ)), ((0 : ℝ) : WithTop ℝ))]) (mid := [])
    (c := (((1 : ℝ), (0 : ℝ)), (⊤ : WithTop ℝ)))
    (new := (((2 : ℝ), (-1 : ℝ)), (⊤ : WithTop ℝ))) (post := []) the_two_line_order ?_ ?_ ?_
  · intro z hz
    have hz' : z = (((0 : ℝ), (0 : ℝ)), ((0 : ℝ) : WithTop ℝ)) := by simpa using hz.symm
    rw [hz']
    show ((0 : ℝ) : WithTop ℝ) < ((interX ((1 : ℝ), (0 : ℝ)) ((2 : ℝ), (-1 : ℝ)) : ℝ) : WithTop ℝ)
    rw [WithTop.coe_lt_coe]
    norm_num [interX]
  · exact WithTop.coe_lt_top _
  · simp

/-- And `breakOrd_splice_dropped`'s two, on the same envelope: the arriving
line is hidden, the first node is rewritten against the second, and the
envelope is the one it started as. -/
example :
    BreakOrd [(((0 : ℝ), (0 : ℝ)),
        breakTo ((0 : ℝ), (0 : ℝ))
          (([(((1 : ℝ), (0 : ℝ)), (⊤ : WithTop ℝ))] : List HullNode).head?.map Prod.fst)),
      (((1 : ℝ), (0 : ℝ)), (⊤ : WithTop ℝ))] := by
  refine breakOrd_splice_dropped (pre := []) (mid := [])
    (c := (((0 : ℝ), (0 : ℝ)), ((0 : ℝ) : WithTop ℝ)))
    (post := [(((1 : ℝ), (0 : ℝ)), (⊤ : WithTop ℝ))]) the_two_line_order (by simp) ?_
  intro z hz
  have hz' : z = (((1 : ℝ), (0 : ℝ)), (⊤ : WithTop ℝ)) := by simpa using hz.symm
  rw [hz']
  exact WithTop.coe_lt_top _

end ALM
end Transformer
