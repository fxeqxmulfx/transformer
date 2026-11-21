module

public import Transformer.GainMatroid.Section2_Walks

/-!
# Cycles and thetas

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.1. A cycle is the edge set of a closed walk with
at least one edge that repeats no edge and no vertex (`IsCycleWalk`), so a loop is a cycle of length
one and two parallel edges a cycle of length two. A theta is three internally disjoint paths
linking a pair of distinct vertices; `Openly.isCycleWalk` shows that two of them traverse a cycle.
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE}

namespace OrientedGraph

namespace Walk

variable {G : OrientedGraph V E}

/-- A closed walk traverses a cycle when it has an edge and repeats no edge and no vertex. -/
structure IsCycleWalk {u : V} (W : G.Walk u u) : Prop where
  edges_ne_nil : W.edges ≠ []
  edges_nodup : W.edges.Nodup
  support_tail_nodup : W.support.tail.Nodup

/-- Two walks are **openly disjoint**: they share no edge, and only their ends as vertices. -/
structure Openly {x y : V} (P Q : G.Walk x y) : Prop where
  edges : ∀ e, e ∈ P.edges → e ∉ Q.edges
  verts : ∀ z, z ∈ P.support → z ∈ Q.support → z = x ∨ z = y

theorem Openly.symm {x y : V} {P Q : G.Walk x y} (h : Openly P Q) : Openly Q P :=
  ⟨fun e he hP => h.edges e hP he, fun z hQ hP => h.verts z hP hQ⟩

/-- A path between distinct vertices has an edge. -/
theorem edges_ne_nil_of_ne {x y : V} (hxy : x ≠ y) (P : G.Walk x y) : P.edges ≠ [] := by
  cases P with
  | nil => exact absurd rfl hxy
  | cons e h p => simp [edges]

/-- Two openly disjoint paths between distinct vertices traverse a cycle. -/
theorem Openly.isCycleWalk {x y : V} {P Q : G.Walk x y} (hxy : x ≠ y) (hP : P.IsPath)
    (hQ : Q.IsPath) (h : Openly P Q) : (P.append Q.reverse).IsCycleWalk := by
  refine ⟨?_, ?_, ?_⟩
  · rw [edges_append]
    simpa using fun hnil => absurd hnil (edges_ne_nil_of_ne hxy P)
  · rw [edges_append, edges_reverse]
    refine List.nodup_append.2 ⟨hP.edges_nodup, List.nodup_reverse.2 hQ.edges_nodup, ?_⟩
    intro a ha b hb hab
    subst hab
    exact h.edges a ha (List.mem_reverse.1 hb)
  · rw [support_append, P.support_eq_cons, List.cons_append, List.tail_cons]
    have hP' : P.support.tail.Nodup := by
      have := hP
      rw [IsPath, P.support_eq_cons] at this
      exact (List.nodup_cons.1 this).2
    have hR : Q.reverse.IsPath := hQ.reverse
    have hR' : (Q.reverse).support.tail.Nodup := by
      have := hR
      rw [IsPath, Q.reverse.support_eq_cons] at this
      exact (List.nodup_cons.1 this).2
    refine List.nodup_append.2 ⟨hP', hR', ?_⟩
    intro a ha b hb hab
    subst hab
    have hax : a ≠ x := by
      have := hP
      rw [IsPath, P.support_eq_cons] at this
      exact fun hax => (List.nodup_cons.1 this).1 (hax ▸ ha)
    have hay : a ≠ y := by
      have := hR
      rw [IsPath, Q.reverse.support_eq_cons] at this
      exact fun hay => (List.nodup_cons.1 this).1 (hay ▸ hb)
    have haP : a ∈ P.support := by
      rw [P.support_eq_cons]
      exact List.mem_cons_of_mem _ ha
    have haQ : a ∈ Q.support := by
      have : a ∈ Q.reverse.support := by
        rw [Q.reverse.support_eq_cons]
        exact List.mem_cons_of_mem _ hb
      rwa [support_reverse, List.mem_reverse] at this
    rcases h.verts a haP haQ with h₁ | h₁
    · exact hax h₁
    · exact hay h₁

theorem edgeSet_append_reverse {x y : V} (P Q : G.Walk x y) :
    (P.append Q.reverse).edgeSet = P.edgeSet ∪ Q.edgeSet := by
  ext e
  simp [edgeSet, edges_append, edges_reverse]

end Walk

/-- `C` is a **cycle** of `G`: the edge set of a closed walk traversing a cycle. -/
def IsCycle (G : OrientedGraph V E) (C : Set E) : Prop :=
  ∃ (u : V) (W : G.Walk u u), W.IsCycleWalk ∧ W.edgeSet = C

/-- Three paths form a **theta** (§2.1): they are internally disjoint and link distinct vertices. -/
structure IsTheta {G : OrientedGraph V E} {x y : V} (P Q R : G.Walk x y) : Prop where
  ne : x ≠ y
  pathP : P.IsPath
  pathQ : Q.IsPath
  pathR : R.IsPath
  openPQ : Walk.Openly P Q
  openPR : Walk.Openly P R
  openQR : Walk.Openly Q R

/-- Each pair of paths of a theta forms a cycle, with edge set their union. -/
theorem IsTheta.isCycle {G : OrientedGraph V E} {x y : V} {P Q R : G.Walk x y}
    (h : IsTheta P Q R) :
    G.IsCycle (P.edgeSet ∪ Q.edgeSet) ∧ G.IsCycle (P.edgeSet ∪ R.edgeSet) ∧
      G.IsCycle (Q.edgeSet ∪ R.edgeSet) :=
  ⟨⟨x, _, h.openPQ.isCycleWalk h.ne h.pathP h.pathQ, Walk.edgeSet_append_reverse P Q⟩,
    ⟨x, _, h.openPR.isCycleWalk h.ne h.pathP h.pathR, Walk.edgeSet_append_reverse P R⟩,
    ⟨x, _, h.openQR.isCycleWalk h.ne h.pathQ h.pathR, Walk.edgeSet_append_reverse Q R⟩⟩

/-- Three parallel links from `0` to `1`, for the examples. -/
def pumpkin : OrientedGraph (Fin 2) (Fin 3) := ⟨fun _ => 0, fun _ => 1⟩

/-- The one-edge walk along edge `i` of `pumpkin`. -/
def pumpkinWalk (i : Fin 3) : pumpkin.Walk 0 1 := .cons i (Or.inl ⟨rfl, rfl⟩) (.nil 1)

theorem pumpkinWalk_isPath (i : Fin 3) : (pumpkinWalk i).IsPath := by
  simp [Walk.IsPath, pumpkinWalk, Walk.support, pumpkin]

theorem pumpkinWalk_openly {i j : Fin 3} (hij : i ≠ j) :
    Walk.Openly (pumpkinWalk i) (pumpkinWalk j) := by
  refine ⟨fun e he he' => ?_, fun z hz _ => ?_⟩
  · simp only [pumpkinWalk, Walk.edges, List.mem_singleton] at he he'
    exact hij (he.symm.trans he')
  · simpa [pumpkinWalk, Walk.support, pumpkin] using hz

theorem pumpkin_isTheta : IsTheta (pumpkinWalk 0) (pumpkinWalk 1) (pumpkinWalk 2) :=
  ⟨by decide, pumpkinWalk_isPath 0, pumpkinWalk_isPath 1, pumpkinWalk_isPath 2,
    pumpkinWalk_openly (by decide), pumpkinWalk_openly (by decide),
    pumpkinWalk_openly (by decide)⟩

example : Walk.Openly (pumpkinWalk 1) (pumpkinWalk 0) := pumpkin_isTheta.openPQ.symm

example : (pumpkinWalk 0).edges ≠ [] := Walk.edges_ne_nil_of_ne (by decide) _

example : ((pumpkinWalk 0).append (pumpkinWalk 1).reverse).IsCycleWalk :=
  pumpkin_isTheta.openPQ.isCycleWalk pumpkin_isTheta.ne pumpkin_isTheta.pathP pumpkin_isTheta.pathQ

example : pumpkin.IsCycle ((pumpkinWalk 0).edgeSet ∪ (pumpkinWalk 2).edgeSet) :=
  pumpkin_isTheta.isCycle.2.1

end OrientedGraph

end Transformer.GainMatroid
