module

public import Transformer.GainMatroid.Section2_NablaTripods

/-!
# Thetas of `∇_Y G` through two edges at a corner of `Y`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, the proof of Proposition 2.7. Let `T`, `S` be
an exchange pair on `Y = {a, b, c}` (`IsExchange`), with the triangle `Y` in `T` and the star `Y` at
`v` in `S`, and `Ω` a biased graph on `S`. If two paths of a theta of `T` have edges of `Y`, and
their cycle is not `Y`, then they leave an end of the theta, a corner of `Y`, along them, and the
third path misses `Y`. Without these two edges the three paths form a tripod off `Y` from the
corners of `Y`, which gives a theta of `S` between `v` and the other end; the theta of `T`
matches it under `∇_Y B` (`IsExchange.matchesTheta_of_mem_of_mem`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open scoped symmDiff

namespace OrientedGraph

variable {V W E : Type*}

/-- In a path from `u`, an edge from `u` to `w` is the first edge. -/
theorem Walk.IsPath.exists_eq_cons {G : OrientedGraph V E} {u w t : V} {e : E} {P : G.Walk u t}
    (hP : P.IsPath) (he : e ∈ P.edges) (hj : G.Joins e u w) :
    ∃ (hj' : G.Joins e u w) (P₁ : G.Walk w t), P = .cons e hj' P₁ := by
  cases P with
  | nil => simp [Walk.edges] at he
  | cons g hg P₁ =>
    obtain rfl := hP.eq_of_mem_start he hj.tail_or_head
    obtain rfl := hg.eq_of_joins hj
    exact ⟨hg, P₁, rfl⟩

/-- `insert b X ∆ {a, b, c} = insert a (insert c X)` for `X` off `{a, b, c}`. -/
theorem insert_symmDiff_triple_b {a b c : E} {X : Set E} (hX : Disjoint X {a, b, c})
    (hab : a ≠ b) (hbc : b ≠ c) : insert b X ∆ {a, b, c} = insert a (insert c X) := by
  have hY : ({a, b, c} : Set E) = {b, a, c} := Set.insert_comm a b {c}
  rw [hY, insert_symmDiff_triple (fun g hg => Set.disjoint_left.1 (hY ▸ hX) hg) hab.symm hbc,
    Set.insert_union, Set.singleton_union]

/-- `insert c X ∆ {a, b, c} = insert a (insert b X)` for `X` off `{a, b, c}`. -/
theorem insert_symmDiff_triple_c {a b c : E} {X : Set E} (hX : Disjoint X {a, b, c})
    (hac : a ≠ c) (hbc : b ≠ c) : insert c X ∆ {a, b, c} = insert a (insert b X) := by
  have hY : ({a, b, c} : Set E) = {c, a, b} := by rw [Set.pair_comm b c, Set.insert_comm a c]
  rw [hY, insert_symmDiff_triple (fun g hg => Set.disjoint_left.1 (hY ▸ hX) hg) hac.symm
    hbc.symm, Set.insert_union, Set.singleton_union]

namespace IsExchange

variable {T : OrientedGraph V E} {ι : V → W} {v : W} {a b c : E} {x y z : V}
  {Ω : BiasedGraph W E}

/-- A cycle of `T` made of an edge `e` of `Y` and edges off `Y` is in `∇_Y B` exactly when its
symmetric difference with `Y` is in `B`. -/
theorem mem_nablaBalanced_iff_insert (h : IsExchange T Ω.graph ι v a b c x y z) {X : Set E}
    (hX : Disjoint X {a, b, c}) {e : E} (he : e ∈ ({a, b, c} : Set E))
    (hD : T.IsCycle (insert e X)) :
    insert e X ∈ nablaBalanced T Ω.balanced {a, b, c} ↔ insert e X ∆ {a, b, c} ∈ Ω.balanced :=
  h.mem_nablaBalanced_iff_of_singleton hD he (by
    rw [Set.insert_inter_of_mem he, Set.disjoint_iff_inter_eq_empty.1 hX, insert_empty_eq])

/-- A cycle of `T` made of two edges of `Y` and edges off `Y` is in `∇_Y B` exactly when it is in
`B`. -/
theorem mem_nablaBalanced_iff_insert_insert (h : IsExchange T Ω.graph ι v a b c x y z)
    {X : Set E} (hX : Disjoint X {a, b, c}) {e₁ e₂ : E} (he₁ : e₁ ∈ ({a, b, c} : Set E))
    (he₂ : e₂ ∈ ({a, b, c} : Set E)) (hne : e₁ ≠ e₂) (hD : T.IsCycle (insert e₁ (insert e₂ X))) :
    insert e₁ (insert e₂ X) ∈ nablaBalanced T Ω.balanced {a, b, c} ↔
      insert e₁ (insert e₂ X) ∈ Ω.balanced :=
  h.mem_nablaBalanced_iff_of_pair hD he₁ he₂ hne (by
    rw [Set.insert_inter_of_mem he₁, Set.insert_inter_of_mem he₂,
      Set.disjoint_iff_inter_eq_empty.1 hX, insert_empty_eq])

/-- §2.4: a theta of `T` from the corner `x` whose first two paths start along `b` and `c`, and
whose cycle through them is not `Y`, matches a theta of `Ω` under `∇_Y B`. -/
theorem matchesTheta_of_corner (h : IsExchange T Ω.graph ι v a b c x y z) {t : V}
    {P Q R : T.Walk x t} (hθ : IsTheta P Q R) (hb : b ∈ P.edges) (hc : c ∈ Q.edges)
    (hPQ : P.edgeSet ∪ Q.edgeSet ≠ {a, b, c}) :
    MatchesTheta (nablaBalanced T Ω.balanced {a, b, c}) Ω P Q R := by
  have hC := hθ.isCycle
  obtain ⟨hb', P₁, rfl⟩ := hθ.pathP.exists_eq_cons hb h.triangle.joins_b
  obtain ⟨hc', Q₁, rfl⟩ := hθ.pathQ.exists_eq_cons hc h.triangle.joins_c
  have hxP := (Walk.isPath_cons.1 hθ.pathP).1
  have hxQ := (Walk.isPath_cons.1 hθ.pathQ).1
  have haP : a ∉ P₁.edges := fun ha => hPQ (h.triangle.eq_of_subset hC.1 (by
    rintro g (rfl | rfl | rfl)
    · exact Or.inl (List.mem_cons_of_mem _ ha)
    · exact Or.inl List.mem_cons_self
    · exact Or.inr List.mem_cons_self))
  have haQ : a ∉ Q₁.edges := fun ha => hPQ (h.triangle.eq_of_subset hC.1 (by
    rintro g (rfl | rfl | rfl)
    · exact Or.inr (List.mem_cons_of_mem _ ha)
    · exact Or.inl List.mem_cons_self
    · exact Or.inr List.mem_cons_self))
  have haR : a ∉ R.edges := fun ha => by
    rcases hθ.openQR.verts y (List.mem_cons_of_mem _ Q₁.start_mem_support)
      (Walk.mem_support_of_mem_edges ha h.triangle.joins_a.tail_or_head) with hy | hy
    · exact h.triangle.ne_xy hy.symm
    rcases hθ.openPR.verts z (List.mem_cons_of_mem _ P₁.start_mem_support)
      (Walk.mem_support_of_mem_edges ha h.triangle.joins_a.symm.tail_or_head) with hz | hz
    · exact h.triangle.ne_xz hz.symm
    exact h.triangle.ne_yz (hy.trans hz.symm)
  have hT : IsTripod {a, b, c} R Q₁ P₁ :=
    ⟨hθ.pathR, (Walk.isPath_cons.1 hθ.pathQ).2, (Walk.isPath_cons.1 hθ.pathP).2,
      fun g hg hgY => by
        rcases hgY with rfl | rfl | rfl
        · exact haR hg
        · exact hθ.openPR.edges _ List.mem_cons_self hg
        · exact hθ.openQR.edges _ List.mem_cons_self hg,
      fun g hg hgY => by
        rcases hgY with rfl | rfl | rfl
        · exact haQ hg
        · exact hθ.openPQ.edges _ List.mem_cons_self (List.mem_cons_of_mem _ hg)
        · exact hxQ (Walk.mem_support_of_mem_edges hg h.triangle.joins_c.tail_or_head),
      fun g hg hgY => by
        rcases hgY with rfl | rfl | rfl
        · exact haP hg
        · exact hxP (Walk.mem_support_of_mem_edges hg h.triangle.joins_b.tail_or_head)
        · exact hθ.openPQ.edges _ (List.mem_cons_of_mem _ hg) List.mem_cons_self,
      fun g hg hg' => hθ.openQR.edges g (List.mem_cons_of_mem _ hg') hg,
      fun g hg hg' => hθ.openPR.edges g (List.mem_cons_of_mem _ hg') hg,
      fun g hg hg' => hθ.openPQ.edges g (List.mem_cons_of_mem _ hg') (List.mem_cons_of_mem _ hg),
      fun w hw hw' => (hθ.openQR.verts w (List.mem_cons_of_mem _ hw') hw).resolve_left
        fun hwx => hxQ (hwx ▸ hw'),
      fun w hw hw' => (hθ.openPR.verts w (List.mem_cons_of_mem _ hw') hw).resolve_left
        fun hwx => hxP (hwx ▸ hw'),
      fun w hw hw' => (hθ.openPQ.verts w (List.mem_cons_of_mem _ hw')
        (List.mem_cons_of_mem _ hw)).resolve_left fun hwx => hxP (hwx ▸ hw')⟩
  have hP₁ : Disjoint P₁.edgeSet {a, b, c} := Set.disjoint_left.2 fun g hg => hT.offR g hg
  have hQ₁ : Disjoint Q₁.edgeSet {a, b, c} := Set.disjoint_left.2 fun g hg => hT.offQ g hg
  have hR : Disjoint R.edgeSet {a, b, c} := Set.disjoint_left.2 fun g hg => hT.offP g hg
  obtain ⟨A', B', C', hθ', hA', hB', hC'⟩ := h.exists_isTheta_of_isTripod hT
  simp only [Walk.edgeSet_cons] at hC
  refine ⟨v, ι t, C', B', A', hθ'.swap₁₂.swap₂₃.swap₁₂, ?_, ?_, ?_⟩
  · have e₁ : insert b P₁.edgeSet ∪ insert c Q₁.edgeSet =
        insert b (insert c (P₁.edgeSet ∪ Q₁.edgeSet)) := by
      ext g; simp only [Set.mem_union, Set.mem_insert_iff]; tauto
    have e₂ : insert c P₁.edgeSet ∪ insert b Q₁.edgeSet =
        insert b (insert c (P₁.edgeSet ∪ Q₁.edgeSet)) := by
      ext g; simp only [Set.mem_union, Set.mem_insert_iff]; tauto
    rw [Walk.edgeSet_cons, Walk.edgeSet_cons, hC', hB', e₁, e₂]
    exact h.mem_nablaBalanced_iff_insert_insert (Set.disjoint_union_left.2 ⟨hP₁, hQ₁⟩)
      (by simp) (by simp) h.triangle.ne_bc (e₁ ▸ hC.1)
  · have hD := hC.2.1
    rw [Set.insert_union] at hD
    have e₃ : insert a (insert c (P₁.edgeSet ∪ R.edgeSet)) =
        insert c P₁.edgeSet ∪ insert a R.edgeSet := by
      ext g; simp only [Set.mem_union, Set.mem_insert_iff]; tauto
    rw [Walk.edgeSet_cons, Set.insert_union,
      h.mem_nablaBalanced_iff_insert (Set.disjoint_union_left.2 ⟨hP₁, hR⟩) (by simp) hD,
      insert_symmDiff_triple_b (Set.disjoint_union_left.2 ⟨hP₁, hR⟩) h.triangle.ne_ab
      h.triangle.ne_bc, hC', hA', e₃]
  · have hD := hC.2.2
    rw [Set.insert_union] at hD
    have e₄ : insert a (insert b (Q₁.edgeSet ∪ R.edgeSet)) =
        insert b Q₁.edgeSet ∪ insert a R.edgeSet := by
      ext g; simp only [Set.mem_union, Set.mem_insert_iff]; tauto
    rw [Walk.edgeSet_cons, Set.insert_union,
      h.mem_nablaBalanced_iff_insert (Set.disjoint_union_left.2 ⟨hQ₁, hR⟩) (by simp) hD,
      insert_symmDiff_triple_c (Set.disjoint_union_left.2 ⟨hQ₁, hR⟩) h.triangle.ne_ac
      h.triangle.ne_bc, hB', hA', e₄]

/-- §2.4: a theta of `T` whose first two paths have edges of `Y`, and whose cycle through them is
not `Y`, matches a theta of `Ω` under `∇_Y B`. -/
theorem matchesTheta_of_mem_of_mem (h : IsExchange T Ω.graph ι v a b c x y z) {s t : V}
    {P Q R : T.Walk s t} (hθ : IsTheta P Q R) {e₁ e₂ : E} (he₁ : e₁ ∈ P.edges)
    (he₁Y : e₁ ∈ ({a, b, c} : Set E)) (he₂ : e₂ ∈ Q.edges) (he₂Y : e₂ ∈ ({a, b, c} : Set E))
    (hPQ : P.edgeSet ∪ Q.edgeSet ≠ {a, b, c}) :
    MatchesTheta (nablaBalanced T Ω.balanced {a, b, c}) Ω P Q R := by
  obtain ⟨e₀, x', y', z', h', hX⟩ :=
    h.exists_relabel he₁Y he₂Y fun he => hθ.openPQ.edges e₁ he₁ (he ▸ he₂)
  rw [← hX] at hPQ ⊢
  rcases hθ.openPQ.verts x' (Walk.mem_support_of_mem_edges he₁ h'.triangle.joins_b.tail_or_head)
    (Walk.mem_support_of_mem_edges he₂ h'.triangle.joins_c.tail_or_head) with rfl | rfl
  · exact h'.matchesTheta_of_corner hθ he₁ he₂ hPQ
  · refine MatchesTheta.of_reverse (h'.matchesTheta_of_corner hθ.reverse
      (Walk.mem_edges_reverse.2 he₁) (Walk.mem_edges_reverse.2 he₂) ?_)
    rwa [Walk.edgeSet_reverse, Walk.edgeSet_reverse]

end IsExchange

end OrientedGraph

end Transformer.GainMatroid
