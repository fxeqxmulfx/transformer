module

public import Transformer.GainMatroid.Section2_PathSurgery

/-!
# A cycle through two edges at a vertex

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5. A cycle less one of its edges is a path
between the ends of that edge (`Walk.IsCycleWalk.exists_isPath_insert`), so a cycle through two
edges `e ≠ f` at `u` consists of `e`, `f` and a path between their ends away from `u` that misses
`u` (`IsCycle.exists_isPath_of_mem`). Such a path is openly disjoint from the path through `u`
along `e` and `f` (`Walk.openly_of_not_mem`), and moving the first edge of a path onto the other
path keeps the two openly disjoint (`Walk.Openly.of_cons`).
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE}

namespace OrientedGraph

variable {G : OrientedGraph V E}

/-- `u` is an end of an edge joining `u` and `v`. -/
theorem Joins.tail_or_head {e : E} {u v : V} (h : G.Joins e u v) : G.tail e = u ∨ G.head e = u :=
  h.imp And.left And.right

/-- An end of an edge joining `s` and `t` is `s` or `t`. -/
theorem Joins.eq_or_eq {e : E} {s t u : V} (h : G.Joins e s t)
    (hu : G.tail e = u ∨ G.head e = u) : u = s ∨ u = t := by
  rcases h with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩ <;> rcases hu with hu | hu
  · exact Or.inl (hu.symm.trans h₁)
  · exact Or.inr (hu.symm.trans h₂)
  · exact Or.inr (hu.symm.trans h₁)
  · exact Or.inl (hu.symm.trans h₂)

/-- An edge joining `x` to `b` and to `q` has `b = q`. -/
theorem Joins.eq_of_joins {e : E} {x b q : V} (hb : G.Joins e x b) (hq : G.Joins e x q) :
    b = q := by
  rcases hb with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩ <;> rcases hq with ⟨h₃, h₄⟩ | ⟨h₃, h₄⟩
  · exact h₂.symm.trans h₄
  · exact h₂.symm.trans (h₄.trans (h₁.symm.trans h₃))
  · exact h₁.symm.trans (h₃.trans (h₂.symm.trans h₄))
  · exact h₁.symm.trans h₃

/-- The end of a link away from `u` is not `u`. -/
theorem otherEnd_ne {u : V} {e : E} (hl : G.IsLink e) : G.otherEnd u e ≠ u := by
  unfold otherEnd
  split_ifs with ht
  · exact fun hh => hl (ht.trans hh.symm)
  · exact ht

/-- An edge joining `y` to `u` has `y` as its end away from `u`. -/
theorem Joins.otherEnd_eq {e : E} {y u : V} (h : G.Joins e y u) : G.otherEnd u e = y := by
  unfold otherEnd
  rcases h with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
  · split_ifs with ht
    · exact h₂.trans (ht.symm.trans h₁)
    · exact h₁
  · exact (ite_eq_left h₁).trans h₂

/-- An edge at `u` joins its other end to `u`. -/
theorem joins_otherEnd {e : E} {u : V} (h : G.tail e = u ∨ G.head e = u) :
    G.Joins e (G.otherEnd u e) u := by
  unfold otherEnd
  split_ifs with ht
  · exact Or.inr ⟨ht, rfl⟩
  · exact Or.inl ⟨rfl, h.resolve_left ht⟩

namespace Walk

/-- A path from `u` through an edge `f` at `u` starts with `f`. -/
theorem IsPath.exists_of_mem_start {u v : V} {R : G.Walk u v} (hR : R.IsPath) {f : E}
    (hf : f ∈ R.edges) (hfu : G.tail f = u ∨ G.head f = u) :
    ∃ (y : V) (_ : G.Joins f y u) (R' : G.Walk y v), R'.IsPath ∧ u ∉ R'.support ∧
      R.edgeSet = insert f R'.edgeSet := by
  cases R with
  | nil => simp [edges] at hf
  | cons g hg R' =>
    obtain rfl := hR.eq_of_mem_start hf hfu
    obtain ⟨hu, hR'⟩ := isPath_cons.1 hR
    exact ⟨_, hg.symm, R', hR', hu, edgeSet_cons _ _ _⟩

/-- A cycle walk less one of its edges `e` is a path between the ends of `e`. -/
theorem IsCycleWalk.exists_isPath_insert {a : V} {W : G.Walk a a} (hW : W.IsCycleWalk) {e : E}
    (he : e ∈ W.edges) :
    ∃ (s t : V) (_ : G.Joins e s t) (R : G.Walk t s), R.IsPath ∧ e ∉ R.edges ∧
      W.edgeSet = insert e R.edgeSet := by
  obtain ⟨s, t, hj, A, B, rfl⟩ := W.exists_append_cons_of_mem he
  have hsup := hW.support_tail_nodup
  have hed := List.nodup_append.1 (edges_append A (cons e hj B) ▸ hW.edges_nodup)
  rw [support_append, A.support_eq_cons, List.cons_append, List.tail_cons] at hsup
  refine ⟨s, t, hj, B.append A, ?_, fun h => ?_, ?_⟩
  · unfold IsPath
    rw [support_append]
    exact List.nodup_append_comm.1 hsup
  · rcases mem_edges_append.1 h with h | h
    · exact (List.nodup_cons.1 hed.2.1).1 h
    · exact hed.2.2 e h e List.mem_cons_self rfl
  · ext g
    show g ∈ (A.append (cons e hj B)).edges ↔ g = e ∨ g ∈ (B.append A).edges
    rw [mem_edges_append, mem_edges_append, show (cons e hj B).edges = e :: B.edges from rfl,
      List.mem_cons]
    tauto

/-- Moving the first edge of `R` onto `S` keeps the two openly disjoint. -/
theorem Openly.of_cons {x q y : V} {e : E} {h : G.Joins e x q} {R : G.Walk q y}
    {S : G.Walk x y} (hR : (cons e h R).IsPath) (hRS : Openly (cons e h R) S) :
    Openly R (cons e h.symm S) := by
  obtain ⟨hx, -⟩ := isPath_cons.1 hR
  refine ⟨fun g hg hg' => ?_, fun z hz hz' => ?_⟩
  · rcases List.mem_cons.1 hg' with rfl | hg'
    · exact (List.nodup_cons.1 hR.edges_nodup).1 hg
    · exact hRS.edges g (List.mem_cons_of_mem _ hg) hg'
  · rcases List.mem_cons.1 hz' with rfl | hz'
    · exact Or.inl rfl
    · rcases hRS.verts z (List.mem_cons_of_mem _ hz) hz' with rfl | rfl
      · exact (hx hz).elim
      · exact Or.inr rfl

/-- A walk that misses `u` is openly disjoint from the path through `u` along two edges at `u`. -/
theorem openly_of_not_mem {x u y : V} {e f : E} (he : G.Joins e x u) (hf : G.Joins f u y)
    {P : G.Walk x y} (hu : u ∉ P.support) : Openly P (cons e he (cons f hf (nil y))) := by
  refine ⟨fun g hg hg' => ?_, fun z hz hz' => ?_⟩
  · simp only [edges, List.mem_cons, List.not_mem_nil, or_false] at hg'
    rcases hg' with rfl | rfl
    · exact hu (mem_support_of_mem_edges hg he.symm.tail_or_head)
    · exact hu (mem_support_of_mem_edges hg hf.tail_or_head)
  · simp only [support, List.mem_cons, List.not_mem_nil, or_false] at hz'
    rcases hz' with rfl | rfl | rfl
    · exact Or.inl rfl
    · exact (hu hz).elim
    · exact Or.inr rfl

end Walk

/-- A cycle through two edges `e ≠ f` at `u` consists of `e`, `f` and a path between their ends
away from `u` that misses `u`. -/
theorem IsCycle.exists_isPath_of_mem {C : Set E} (hC : G.IsCycle C) {u : V} {e f : E}
    (hne : e ≠ f) (heC : e ∈ C) (hfC : f ∈ C) (heu : G.tail e = u ∨ G.head e = u)
    (hfu : G.tail f = u ∨ G.head f = u) :
    ∃ P : G.Walk (G.otherEnd u e) (G.otherEnd u f), P.IsPath ∧ u ∉ P.support ∧
      C = insert e (insert f P.edgeSet) := by
  obtain ⟨a, W, hW, rfl⟩ := hC
  obtain ⟨s, t, hj, R, hR, -, hWR⟩ := hW.exists_isPath_insert heC
  rw [hWR] at hfC ⊢
  have hfR : f ∈ R.edges := (Set.mem_insert_iff.1 hfC).resolve_left hne.symm
  rcases hj.eq_or_eq heu with rfl | rfl
  · obtain ⟨y, hy, R', hR', huR', hRR'⟩ :=
      hR.reverse.exists_of_mem_start (Walk.mem_edges_reverse.2 hfR) hfu
    obtain rfl := hj.symm.otherEnd_eq
    obtain rfl := hy.otherEnd_eq
    refine ⟨R'.reverse, hR'.reverse, fun h => huR' (Walk.mem_support_reverse.1 h), ?_⟩
    rw [← R.edgeSet_reverse, hRR', R'.edgeSet_reverse]
  · obtain ⟨y, hy, R', hR', huR', hRR'⟩ := hR.exists_of_mem_start hfR hfu
    obtain rfl := hj.otherEnd_eq
    obtain rfl := hy.otherEnd_eq
    exact ⟨R'.reverse, hR'.reverse, fun h => huR' (Walk.mem_support_reverse.1 h),
      by rw [hRR', R'.edgeSet_reverse]⟩

/-- The path `2 → 0` along the last edge of `triangle`, for the examples. -/
def triangleLast : triangle.Walk 0 2 := .cons 2 (Or.inr ⟨rfl, rfl⟩) (.nil 2)

example := pathThree_isPath.exists_of_mem_start (f := 0) (by decide) (Or.inl rfl)

example := (pumpkin_isTheta.openPQ.isCycleWalk pumpkin_isTheta.ne pumpkin_isTheta.pathP
  pumpkin_isTheta.pathQ).exists_isPath_insert (e := 0) (by decide)

example := (pumpkinWalk_openly (i := 0) (j := 1) (by decide)).of_cons (pumpkinWalk_isPath 0)

example := Walk.openly_of_not_mem (G := triangle) (e := 0) (f := 1) (Or.inl ⟨rfl, rfl⟩)
  (Or.inl ⟨rfl, rfl⟩) (P := triangleLast) (by decide)

example := (isCycle_pair (G := pumpkin) (e := 0) (f := 1) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)
  (by decide) (by decide)).exists_isPath_of_mem (u := 0) (e := 0) (f := 1) (by decide) (by simp)
  (by simp) (Or.inl rfl) (Or.inl rfl)

example := Joins.tail_or_head (G := pathThree) (e := 0) (u := 0) (v := 1) (Or.inl ⟨rfl, rfl⟩)

example := Joins.eq_or_eq (G := pathThree) (e := 0) (s := 0) (t := 1) (u := 1)
  (Or.inl ⟨rfl, rfl⟩) (Or.inr rfl)

example := Joins.eq_of_joins (G := pathThree) (e := 0) (x := 0) (b := 1) (q := 1)
  (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)

example := otherEnd_ne (G := pathThree) (u := 1) (e := 0) (by decide)

example := Joins.otherEnd_eq (G := pathThree) (e := 0) (y := 1) (u := 0) (Or.inr ⟨rfl, rfl⟩)

example := joins_otherEnd (G := pathThree) (e := 1) (u := 2) (Or.inr rfl)

end OrientedGraph

end Transformer.GainMatroid
