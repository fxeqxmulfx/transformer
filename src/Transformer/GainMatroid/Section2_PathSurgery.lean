module

public import Transformer.GainMatroid.Section2_UnbalancingClasses

/-!
# Splitting and joining paths

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.1. The proofs of §2.5 cut paths at a vertex and
join the pieces again. Two paths that share only their junction join to a path
(`Walk.IsPath.append`), and the pieces of a path are paths (`Walk.IsPath.of_append_left`,
`Walk.IsPath.of_append_right`) that meet only at the cut (`Walk.IsPath.eq_of_mem_of_mem`) and share
no edge (`Walk.IsPath.not_mem_of_mem`). A walk splits at a vertex on it
(`Walk.exists_eq_append_of_mem_support`) and at its first vertex in a given set
(`Walk.exists_eq_append_first`).
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE}

namespace OrientedGraph

variable {G : OrientedGraph V E}

namespace Walk

theorem isPath_cons {u v w : V} {e : E} {h : G.Joins e u v} {p : G.Walk v w} :
    (cons e h p).IsPath ↔ u ∉ p.support ∧ p.IsPath :=
  List.nodup_cons

theorem mem_support_append {u v w : V} {p : G.Walk u v} {q : G.Walk v w} {z : V} :
    z ∈ (p.append q).support ↔ z ∈ p.support ∨ z ∈ q.support := by
  rw [support_append, List.mem_append]
  refine ⟨fun h => h.imp_right List.mem_of_mem_tail, fun h => h.elim Or.inl fun h => ?_⟩
  rw [q.support_eq_cons, List.mem_cons] at h
  rcases h with rfl | h
  · exact Or.inl p.end_mem_support
  · exact Or.inr h

theorem mem_edges_append {u v w : V} {p : G.Walk u v} {q : G.Walk v w} {e : E} :
    e ∈ (p.append q).edges ↔ e ∈ p.edges ∨ e ∈ q.edges := by
  rw [edges_append, List.mem_append]

theorem mem_support_reverse {u v : V} {p : G.Walk u v} {z : V} :
    z ∈ p.reverse.support ↔ z ∈ p.support := by
  rw [support_reverse, List.mem_reverse]

theorem mem_edges_reverse {u v : V} {p : G.Walk u v} {e : E} :
    e ∈ p.reverse.edges ↔ e ∈ p.edges := by
  rw [edges_reverse, List.mem_reverse]

theorem edgeSet_reverse {u v : V} (p : G.Walk u v) : p.reverse.edgeSet = p.edgeSet :=
  Set.ext fun _ => mem_edges_reverse

/-- An end of an edge of a walk lies on the walk. -/
theorem mem_support_of_mem_edges {u v w : V} {p : G.Walk u v} {e : E} (he : e ∈ p.edges)
    (hw : G.tail e = w ∨ G.head e = w) : w ∈ p.support := by
  obtain ⟨h₁, h₂⟩ := p.ends_mem_support he
  rcases hw with rfl | rfl
  · exact h₁
  · exact h₂

/-- Two paths that share only the junction `v` join to a path. -/
theorem IsPath.append {u v w : V} {p : G.Walk u v} {q : G.Walk v w} (hp : p.IsPath)
    (hq : q.IsPath) (h : ∀ z ∈ p.support, z ∈ q.support → z = v) : (p.append q).IsPath := by
  unfold IsPath at hp hq ⊢
  rw [support_append]
  rw [q.support_eq_cons] at hq
  refine List.nodup_append.2 ⟨hp, (List.nodup_cons.1 hq).2, fun a ha b hb hab => ?_⟩
  subst hab
  obtain rfl : a = v := h a ha (by rw [q.support_eq_cons]; exact List.mem_cons_of_mem _ hb)
  exact (List.nodup_cons.1 hq).1 hb

theorem IsPath.of_append_left {u v w : V} {p : G.Walk u v} {q : G.Walk v w}
    (h : (p.append q).IsPath) : p.IsPath := by
  unfold IsPath at h ⊢
  rw [support_append] at h
  exact h.of_append_left

theorem IsPath.of_append_right {u v w : V} {p : G.Walk u v} {q : G.Walk v w}
    (h : (p.append q).IsPath) : q.IsPath := by
  unfold IsPath at h ⊢
  rw [support_append] at h
  rw [q.support_eq_cons]
  exact List.nodup_cons.2
    ⟨fun hv => (List.nodup_append.1 h).2.2 v p.end_mem_support v hv rfl, h.of_append_right⟩

/-- The two pieces of a path meet only at the cut. -/
theorem IsPath.eq_of_mem_of_mem {u v w : V} {p : G.Walk u v} {q : G.Walk v w}
    (h : (p.append q).IsPath) {z : V} (hp : z ∈ p.support) (hq : z ∈ q.support) : z = v := by
  unfold IsPath at h
  rw [support_append] at h
  rw [q.support_eq_cons, List.mem_cons] at hq
  exact hq.resolve_right fun hq => (List.nodup_append.1 h).2.2 z hp z hq rfl

/-- The two pieces of a path share no edge. -/
theorem IsPath.not_mem_of_mem {u v w : V} {p : G.Walk u v} {q : G.Walk v w}
    (h : (p.append q).IsPath) {e : E} (hp : e ∈ p.edges) : e ∉ q.edges := by
  have hnd := h.edges_nodup
  rw [edges_append] at hnd
  exact fun hq => (List.nodup_append.1 hnd).2.2 e hp e hq rfl

/-- A path between equal vertices is empty. -/
theorem IsPath.eq_nil {u : V} {p : G.Walk u u} (hp : p.IsPath) : p = nil u := by
  cases p with
  | nil => rfl
  | cons e h p => exact ((List.nodup_cons.1 hp).1 p.end_mem_support).elim

/-- The edges of a path are links. -/
theorem IsPath.isLink_of_mem {u v : V} {p : G.Walk u v} (hp : p.IsPath) {e : E}
    (he : e ∈ p.edges) : G.IsLink e := by
  induction p with
  | nil => simp [edges] at he
  | @cons a b c f h p ih =>
    obtain ⟨hnot, hp'⟩ := isPath_cons.1 hp
    rcases List.mem_cons.1 he with rfl | he
    · have hab : a ≠ b := fun hab => hnot (hab ▸ p.start_mem_support)
      rcases h with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
      · rw [IsLink, h₁, h₂]
        exact hab
      · rw [IsLink, h₁, h₂]
        exact hab.symm
    · exact ih hp' he

/-- In a path, an edge at the start is the first edge. -/
theorem IsPath.eq_of_mem_start {u v w : V} {g e : E} {h : G.Joins g u v} {p : G.Walk v w}
    (hp : (cons g h p).IsPath) (he : e ∈ (cons g h p).edges)
    (hu : G.tail e = u ∨ G.head e = u) : e = g :=
  (List.mem_cons.1 he).resolve_right fun he => (isPath_cons.1 hp).1 (mem_support_of_mem_edges he hu)

/-- A walk splits at any vertex on it. -/
theorem exists_eq_append_of_mem_support {u v w : V} (p : G.Walk u v) (hw : w ∈ p.support) :
    ∃ (A : G.Walk u w) (B : G.Walk w v), p = A.append B := by
  induction p with
  | nil u =>
    obtain rfl : w = u := by simpa [support] using hw
    exact ⟨nil w, nil w, rfl⟩
  | @cons a b c f h p ih =>
    rcases List.mem_cons.1 hw with rfl | hw
    · exact ⟨nil w, cons f h p, rfl⟩
    · obtain ⟨A, B, rfl⟩ := ih hw
      exact ⟨cons f h A, B, rfl⟩

/-- A walk that ends in `S` splits at its first vertex in `S`. -/
theorem exists_eq_append_first {S : Set V} {u v : V} (p : G.Walk u v) (hv : v ∈ S) :
    ∃ (w : V) (A : G.Walk u w) (B : G.Walk w v), w ∈ S ∧ p = A.append B ∧
      ∀ z ∈ A.support, z ∈ S → z = w := by
  induction p with
  | nil u => exact ⟨u, nil u, nil u, hv, rfl, fun z hz _ => by simpa [support] using hz⟩
  | @cons a b c f h p ih =>
    by_cases ha : a ∈ S
    · exact ⟨a, nil a, cons f h p, ha, rfl, fun z hz _ => by simpa [support] using hz⟩
    · obtain ⟨w, A, B, hw, rfl, hA⟩ := ih hv
      refine ⟨w, cons f h A, B, hw, rfl, fun z hz hzS => ?_⟩
      rcases List.mem_cons.1 hz with rfl | hz
      · exact (ha hzS).elim
      · exact hA z hz hzS

end Walk

/-- The first edge of `pathThree` as a walk, for the examples. -/
def pathThreeFirst : pathThree.Walk 0 1 := .cons 0 (Or.inl ⟨rfl, rfl⟩) (.nil 1)

/-- The second edge of `pathThree` as a walk, for the examples. -/
def pathThreeSecond : pathThree.Walk 1 2 := .cons 1 (Or.inl ⟨rfl, rfl⟩) (.nil 2)

theorem pathThree_isPath : (pathThreeFirst.append pathThreeSecond).IsPath := by
  unfold Walk.IsPath
  decide

example : pathThreeSecond.IsPath := Walk.isPath_cons.2 ⟨by decide, List.nodup_singleton 2⟩

example := Walk.mem_support_of_mem_edges (p := pathThreeFirst) (e := 0) (w := 1)
  List.mem_cons_self (Or.inr rfl)

example := pathThree_isPath.of_append_left.append pathThree_isPath.of_append_right (by decide)

example := pathThree_isPath.eq_of_mem_of_mem (z := 1) (by decide) (by decide)

example := pathThree_isPath.not_mem_of_mem (e := 0) (by decide)

example := Walk.IsPath.eq_nil (p := Walk.nil (G := pathThree) 0) (List.nodup_singleton 0)

example := pathThree_isPath.isLink_of_mem (e := 1) (by decide)

example := pathThree_isPath.eq_of_mem_start (e := 0) (by decide) (Or.inl rfl)

example := (pathThreeFirst.append pathThreeSecond).exists_eq_append_of_mem_support (w := 1)
  (by decide)

example := (pathThreeFirst.append pathThreeSecond).exists_eq_append_first (S := {1, 2}) (by simp)

end OrientedGraph

end Transformer.GainMatroid
