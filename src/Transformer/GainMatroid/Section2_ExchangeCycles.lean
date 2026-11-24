module

public import Transformer.GainMatroid.Section2_ExchangePairs

/-!
# The cycles of an exchange pair

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. For an exchange pair `T`, `S` on
`X = {a, b, c}` (`IsExchange`), with the triangle `X` in `T` and the star `X` at `v` in `S`:

* a set of edges off `X` is a cycle of `T` exactly when it is one of `S`
  (`IsExchange.isCycle_iff_of_disjoint`);
* for a cycle `C` of `T` meeting `X` in `a` alone, `C ∆ X` is a cycle of `S`
  (`IsExchange.isCycle_symmDiff_of_inter_eq_singleton`), and a cycle of `T` meeting `X` in `b` and
  `c` alone is a cycle of `S` (`IsExchange.isCycle_of_inter_eq_pair`);
* for a cycle `D` of `S` meeting `X` in `b` and `c` alone, `D ∆ X` is a cycle of `T`
  (`IsExchange.isCycle_symmDiff`);
* the only cycle of `T` containing `X` is `X` (`IsLabelledTriangle.eq_of_subset`), as a cycle
  contains no other cycle (`IsCycle.eq_of_subset`).

Up to the labels of the triangle, every set of edges meets `X` in one of these ways
(`IsExchange.inter_cases`, in `Section2_ExchangeLabels`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open scoped symmDiff

namespace OrientedGraph

variable {V W E : Type*}

section General

variable {G : OrientedGraph V E}

/-- A cycle through an edge `e` joining `s` and `t` is `e` and a path from `s` to `t`. -/
theorem IsCycle.exists_isPath_insert_of_joins {C : Set E} (hC : G.IsCycle C) {e : E} {s t : V}
    (he : G.Joins e s t) (heC : e ∈ C) :
    ∃ R : G.Walk s t, R.IsPath ∧ e ∉ R.edges ∧ C = insert e R.edgeSet := by
  obtain rfl := he.otherEnd_eq
  exact hC.exists_isPath_insert_of_link heC he.symm.tail_or_head

/-- A cycle through edges `e ≠ f` joining `s` and `t` to `u` is `e`, `f` and a path from `s` to
`t` avoiding `u`. -/
theorem IsCycle.exists_isPath_of_joins {C : Set E} (hC : G.IsCycle C) {e f : E} {s t u : V}
    (he : G.Joins e s u) (hf : G.Joins f t u) (hne : e ≠ f) (heC : e ∈ C) (hfC : f ∈ C) :
    ∃ P : G.Walk s t, P.IsPath ∧ u ∉ P.support ∧ C = insert e (insert f P.edgeSet) := by
  obtain rfl := he.otherEnd_eq
  obtain rfl := hf.otherEnd_eq
  exact hC.exists_isPath_of_mem hne heC hfC he.symm.tail_or_head hf.symm.tail_or_head

/-- A path inside the edges of a path with the same ends has all of its edges. -/
theorem Walk.IsPath.mem_edges_of_subset {s t : V} {P Q : G.Walk s t} (hP : P.IsPath)
    (hQ : Q.IsPath) (hQP : ∀ g ∈ Q.edges, g ∈ P.edges) {g : E} (hg : g ∈ P.edges) :
    g ∈ Q.edges := by
  have hlen : P.support.length ≤ Q.support.length :=
    List.Nodup.length_le_of_subset hP fun u hu => hP.mem_support_of_subset hQ hQP hu
  rw [Walk.length_support, Walk.length_support] at hlen
  exact ((List.Nodup.subperm hQ.edges_nodup hQP).perm_of_length_le (by omega)).mem_iff.2 hg

/-- A cycle contains no other cycle. -/
theorem IsCycle.eq_of_subset {C D : Set E} (hC : G.IsCycle C) (hD : G.IsCycle D) (hDC : D ⊆ C) :
    D = C := by
  obtain ⟨e, heD⟩ := hD.nonempty
  have hj : G.Joins e (G.tail e) (G.head e) := Or.inl ⟨rfl, rfl⟩
  obtain ⟨R, hR, -, rfl⟩ := hC.exists_isPath_insert_of_joins hj (hDC heD)
  obtain ⟨R', hR', heR', rfl⟩ := hD.exists_isPath_insert_of_joins hj heD
  refine hDC.antisymm (Set.insert_subset_insert fun g hg => hR.mem_edges_of_subset hR'
    (fun g' hg' => ?_) hg)
  rcases hDC (Or.inr hg' : g' ∈ insert e R'.edgeSet) with rfl | h
  · exact absurd hg' heR'
  · exact h

/-- The only cycle containing a triangle is the triangle. -/
theorem IsLabelledTriangle.eq_of_subset {a b c : E} {x y z : V}
    (h : G.IsLabelledTriangle a b c x y z) {C : Set E} (hC : G.IsCycle C)
    (hXC : ({a, b, c} : Set E) ⊆ C) : C = {a, b, c} :=
  (hC.eq_of_subset h.isCycle hXC).symm

/-- A cycle meeting a triangle `X` in `b` and `c` alone is `b`, `c` and a path from `z` to `y` off
`X` avoiding `x`. -/
theorem IsLabelledTriangle.exists_isPath_of_inter_eq_pair {a b c : E} {x y z : V}
    (h : G.IsLabelledTriangle a b c x y z) {C : Set E} (hC : G.IsCycle C)
    (hCX : C ∩ {a, b, c} = {b, c}) :
    ∃ P : G.Walk z y, P.IsPath ∧ x ∉ P.support ∧ (∀ g ∈ P.edges, g ∉ ({a, b, c} : Set E)) ∧
      C = insert b (insert c P.edgeSet) := by
  have hbC : b ∈ C := ((Set.ext_iff.1 hCX b).2 (Or.inl rfl)).1
  have hcC : c ∈ C := ((Set.ext_iff.1 hCX c).2 (Or.inr rfl)).1
  obtain ⟨P, hP, hxP, rfl⟩ :=
    hC.exists_isPath_of_joins h.joins_b.symm h.joins_c.symm h.ne_bc hbC hcC
  refine ⟨P, hP, hxP, fun g hg hgX => ?_, rfl⟩
  rcases (Set.ext_iff.1 hCX g).1 ⟨Or.inr (Or.inr hg), hgX⟩ with rfl | rfl
  · exact hxP (Walk.mem_support_of_mem_edges hg h.joins_b.tail_or_head)
  · exact hxP (Walk.mem_support_of_mem_edges hg h.joins_c.tail_or_head)

end General

namespace IsExchange

variable {T : OrientedGraph V E} {S : OrientedGraph W E} {ι : V → W} {v : W} {a b c : E}
  {x y z : V}

/-- A set of edges off `X` is a cycle of `T` exactly when it is one of `S`. -/
theorem isCycle_iff_of_disjoint (h : IsExchange T S ι v a b c x y z) {C : Set E}
    (hC : Disjoint C {a, b, c}) : T.IsCycle C ↔ S.IsCycle C :=
  ⟨fun hT => hT.map_vertex h.inj fun g hg => h.agree g (Set.disjoint_left.1 hC hg),
    fun hS => hS.of_map_vertex h.inj fun g hg => h.agree g (Set.disjoint_left.1 hC hg)⟩

/-- A path of `T` off `X` from a vertex not mapped to `v` is a path of `S` avoiding `v`. -/
theorem exists_map_path (h : IsExchange T S ι v a b c x y z) {s t : V} {P : T.Walk s t}
    (hP : P.IsPath) (hPX : ∀ g ∈ P.edges, g ∉ ({a, b, c} : Set E)) (hs : ι s ≠ v) :
    ∃ q : S.Walk (ι s) (ι t), q.IsPath ∧ v ∉ q.support ∧ q.edgeSet = P.edgeSet := by
  obtain ⟨q, hq₁, hq₂⟩ := P.exists_map_vertex ι fun g hg => h.agree g (hPX g hg)
  refine ⟨q, ?_, ?_, ?_⟩
  · unfold Walk.IsPath
    rw [hq₂]
    exact List.Nodup.map h.inj hP
  · rw [hq₂]
    intro hv
    obtain ⟨u, hu, hu'⟩ := List.mem_map.1 hv
    exact h.ne_of_mem_support P hs hu hu'
  · unfold Walk.edgeSet
    rw [hq₁]

/-- A path of `S` off `X` between images of vertices of `T` is a path of `T`. -/
theorem exists_lift_path (h : IsExchange T S ι v a b c x y z) {s t : V} {q : S.Walk (ι s) (ι t)}
    (hq : q.IsPath) (hqX : ∀ g ∈ q.edges, g ∉ ({a, b, c} : Set E)) :
    ∃ P : T.Walk s t, P.IsPath ∧ P.edgeSet = q.edgeSet := by
  obtain ⟨t', P, ht', hP₁, hP₂⟩ := q.exists_lift h.inj (fun g hg => h.agree g (hqX g hg)) rfl
  obtain rfl := h.inj ht'
  refine ⟨P, List.Nodup.of_map ι ?_, ?_⟩
  · rw [hP₂]
    exact hq
  · unfold Walk.edgeSet
    rw [hP₁]

end IsExchange

/-- The triangle `{3, 4, 5}` of `K4` with the labels of §2.4: `a = 5`, `b = 4`, `c = 3` on `x = 1`,
`y = 2`, `z = 3`. -/
theorem K4_isLabelledTriangle : K4.IsLabelledTriangle 5 4 3 1 2 3 :=
  ⟨Or.inl ⟨rfl, rfl⟩, Or.inl ⟨rfl, rfl⟩, Or.inl ⟨rfl, rfl⟩, by decide, by decide, by decide,
    by decide, by decide, by decide⟩

/-- A triangle of `K4` meeting `{3, 4, 5}` in `5` alone. -/
theorem K4_isCycle_125 : K4.IsCycle {1, 2, 5} := by
  simpa using (K4_isCycle_iff {1, 2, 5}).2 (by decide)

/-- A quadrilateral of `K4` meeting `{3, 4, 5}` in `3` and `4` alone. -/
theorem K4_isCycle_1243 : K4.IsCycle {1, 2, 4, 3} := by
  simpa using (K4_isCycle_iff {1, 2, 4, 3}).2 (by decide)

theorem K4_inter_125 : ({1, 2, 5} : Set (Fin 6)) ∩ {5, 4, 3} = {5} := by
  ext g
  fin_cases g <;> simp

theorem K4_inter_1243 : ({1, 2, 4, 3} : Set (Fin 6)) ∩ {5, 4, 3} = {4, 3} := by
  ext g
  fin_cases g <;> simp

example := K4_isCycle_125.exists_isPath_insert_of_joins (e := 5) (s := 2) (t := 3)
  (Or.inl ⟨rfl, rfl⟩) (by simp)

example := K4_isCycle_125.exists_isPath_of_joins (e := 5) (f := 2) (s := 2) (t := 0) (u := 3)
  (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by simp) (by simp)

example := (pumpkinWalk_isPath 0).mem_edges_of_subset (pumpkinWalk_isPath 0) (fun _ h => h)
  (g := 0) (by simp [pumpkinWalk, Walk.edges])

example := K4_isCycle_125.eq_of_subset K4_isCycle_125 Set.Subset.rfl

example := K4_isLabelledTriangle.eq_of_subset K4_isLabelledTriangle.isCycle Set.Subset.rfl

example := K4_isLabelledTriangle.exists_isPath_of_inter_eq_pair K4_isCycle_1243 K4_inter_1243

example := K4_isLabelledTriangle.isExchange.isCycle_iff_of_disjoint (Set.empty_disjoint _)

example := K4_isLabelledTriangle.isExchange.exists_map_path (P := Walk.nil 1)
  (by simp [Walk.IsPath, Walk.support]) (by simp [Walk.edges]) (Option.some_ne_none 1)

example := K4_isLabelledTriangle.isExchange.exists_lift_path (s := 1) (q := Walk.nil (some 1))
  (by simp [Walk.IsPath, Walk.support]) (by simp [Walk.edges])

end OrientedGraph

end Transformer.GainMatroid
