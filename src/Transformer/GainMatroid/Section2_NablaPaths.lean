module

public import Transformer.GainMatroid.Section2_NablaCycles

/-!
# Paths of `∇_Y G` as paths of `G`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, the proof of Proposition 2.7. Let `T`, `S` be
an exchange pair on `Y = {a, b, c}` (`IsExchange`), with the triangle `Y` in `T` and the star `Y` at
`v` in `S`. A path `P'` of `S` is a **detour** of a path `P` of `T` (`IsDetour`) if its vertices
other than `v` are images of vertices of `P` and its edges are edges of `P` or of `Y`. A path of `T`
off `Y` is a detour of itself (`IsExchange.exists_isDetour_of_off`), and a detour of one path of a
theta of `T` whose other two paths are off `Y` makes a theta of `S` with them
(`IsExchange.exists_isTheta_of_isDetour`). Paths of `T` off `Y` from `s` to an end of `a` and from
its other end to `t` join in `S` through `b`, `v` and `c` (`IsExchange.exists_isPath_of_joins`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open scoped symmDiff

namespace OrientedGraph

variable {V W E : Type*} {T : OrientedGraph V E} {S : OrientedGraph W E}

/-- A path `P'` of `S` from `ι s` to `ι t` is a **detour** of a path `P` of `T` from `s` to `t`
through `v` (§2.4): its vertices other than `v` are images of vertices of `P`, and its edges are
edges of `P` or of `Y`. -/
structure IsDetour (Y : Set E) (ι : V → W) (v : W) {s t : V} (P : T.Walk s t)
    (P' : S.Walk (ι s) (ι t)) : Prop where
  isPath : P'.IsPath
  support : ∀ w ∈ P'.support, w = v ∨ ∃ u ∈ P.support, ι u = w
  edges : ∀ g ∈ P'.edges, g ∈ P.edges ∨ g ∈ Y

namespace IsExchange

variable {ι : V → W} {v : W} {a b c : E} {x y z : V}

/-- A path of `T` off `Y` is a path of `S` with the same vertices and edges: a detour of itself. -/
theorem exists_isDetour_of_off (h : IsExchange T S ι v a b c x y z) {s t : V} {P : T.Walk s t}
    (hP : P.IsPath) (hPY : ∀ g ∈ P.edges, g ∉ ({a, b, c} : Set E)) :
    ∃ P' : S.Walk (ι s) (ι t), IsDetour {a, b, c} ι v P P' ∧ P'.edges = P.edges ∧
      P'.support = P.support.map ι := by
  obtain ⟨P', hP'e, hP's⟩ := P.exists_map_vertex ι fun g hg => h.agree g (hPY g hg)
  refine ⟨P', ⟨?_, fun w hw => ?_, fun g hg => Or.inl (hP'e ▸ hg)⟩, hP'e, hP's⟩
  · unfold Walk.IsPath
    rw [hP's]
    exact List.Nodup.map h.inj hP
  · rw [hP's] at hw
    obtain ⟨u, hu, rfl⟩ := List.mem_map.1 hw
    exact Or.inr ⟨u, hu, rfl⟩

/-- A detour of a path `P` and a path of `S` with the vertices and edges of a path `Q` off `Y`,
openly disjoint from `P`, are openly disjoint. -/
theorem openly_of_isDetour (h : IsExchange T S ι v a b c x y z) {s t : V} (hs : ι s ≠ v)
    {P Q : T.Walk s t} (hPQ : Walk.Openly P Q) {P' Q' : S.Walk (ι s) (ι t)}
    (hP : IsDetour {a, b, c} ι v P P') (hQY : ∀ g ∈ Q.edges, g ∉ ({a, b, c} : Set E))
    (hQe : Q'.edges = Q.edges) (hQs : Q'.support = Q.support.map ι) : Walk.Openly P' Q' := by
  refine ⟨fun g hg hg' => ?_, fun w hw hw' => ?_⟩
  · rw [hQe] at hg'
    rcases hP.edges g hg with hg | hg
    · exact hPQ.edges g hg hg'
    · exact hQY g hg' hg
  · rw [hQs] at hw'
    obtain ⟨u, hu, rfl⟩ := List.mem_map.1 hw'
    rcases hP.support _ hw with hv | ⟨u', hu', he⟩
    · exact absurd hv (h.ne_of_mem_support Q hs hu)
    · obtain rfl := h.inj he
      exact (hPQ.verts u' hu' hu).imp (congrArg ι) (congrArg ι)

/-- A theta of `T` with two paths off `Y`, and a detour of its third path, give a theta of `S`. -/
theorem exists_isTheta_of_isDetour (h : IsExchange T S ι v a b c x y z) {s t : V}
    {P Q R : T.Walk s t} (hθ : IsTheta P Q R) (hQY : ∀ g ∈ Q.edges, g ∉ ({a, b, c} : Set E))
    (hRY : ∀ g ∈ R.edges, g ∉ ({a, b, c} : Set E)) {P' : S.Walk (ι s) (ι t)}
    (hP : IsDetour {a, b, c} ι v P P') :
    ∃ Q' R' : S.Walk (ι s) (ι t), IsTheta P' Q' R' ∧ Q'.edgeSet = Q.edgeSet ∧
      R'.edgeSet = R.edgeSet := by
  obtain ⟨u, e, he, -, -⟩ := P.exists_eq_cons_of_ne hθ.ne
  have hs := h.ne_of_joins he
  obtain ⟨Q', hQ', hQe, hQs⟩ := h.exists_isDetour_of_off hθ.pathQ hQY
  obtain ⟨R', -, hRe, hRs⟩ := h.exists_isDetour_of_off hθ.pathR hRY
  refine ⟨Q', R', ⟨fun he => hθ.ne (h.inj he), hP.isPath, hQ'.isPath, ?_,
    h.openly_of_isDetour hs hθ.openPQ hP hQY hQe hQs,
    h.openly_of_isDetour hs hθ.openPR hP hRY hRe hRs,
    h.openly_of_isDetour hs hθ.openQR hQ' hRY hRe hRs⟩, ?_, ?_⟩
  · unfold Walk.IsPath
    rw [hRs]
    exact List.Nodup.map h.inj hθ.pathR
  · unfold Walk.edgeSet
    rw [hQe]
  · unfold Walk.edgeSet
    rw [hRe]

/-- Paths of `T` off `Y`, from `s` to `y` and from `z` to `t`, with no common vertex, join in `S`
through `b`, `v` and `c`. -/
theorem exists_isPath_through (h : IsExchange T S ι v a b c x y z) {s t : V} {A : T.Walk s y}
    {B : T.Walk z t} (hAB : (A.support ++ B.support).Nodup)
    (hA : ∀ g ∈ A.edges, g ∉ ({a, b, c} : Set E)) (hB : ∀ g ∈ B.edges, g ∉ ({a, b, c} : Set E)) :
    ∃ P' : S.Walk (ι s) (ι t), P'.IsPath ∧ P'.support = A.support.map ι ++ v :: B.support.map ι ∧
      P'.edges = A.edges ++ b :: c :: B.edges := by
  obtain ⟨A', hA'e, hA's⟩ := A.exists_map_vertex ι fun g hg => h.agree g (hA g hg)
  obtain ⟨B', hB'e, hB's⟩ := B.exists_map_vertex ι fun g hg => h.agree g (hB g hg)
  have hsup : (A'.append (.cons b h.joins_b.symm (.cons c h.joins_c B'))).support =
      A.support.map ι ++ v :: B.support.map ι := by
    simp [Walk.support_append, Walk.support, hA's, hB's]
  refine ⟨A'.append (.cons b h.joins_b.symm (.cons c h.joins_c B')), ?_, hsup, ?_⟩
  · unfold Walk.IsPath
    rw [hsup, List.nodup_middle, List.nodup_cons, ← List.map_append]
    refine ⟨fun hv => ?_, List.Nodup.map h.inj hAB⟩
    obtain ⟨u, hu, hvu⟩ := List.mem_map.1 hv
    rcases List.mem_append.1 hu with hu | hu
    · exact h.ne_of_mem_support A.reverse (h.ne_of_joins h.triangle.joins_a)
        (Walk.mem_support_reverse.2 hu) hvu
    · exact h.ne_of_mem_support B (h.ne_of_joins h.triangle.joins_a.symm) hu hvu
  · simp [Walk.edges_append, Walk.edges, hA'e, hB'e]

/-- Paths of `T` off `Y` from `s` to one end of `a` and from the other end to `t`, with no common
vertex, join in `S` through `v`, with their edges and `b` and `c`. -/
theorem exists_isPath_of_joins (h : IsExchange T S ι v a b c x y z) {s t p q : V}
    (hpq : T.Joins a p q) {A : T.Walk s p} {B : T.Walk q t} (hAB : (A.support ++ B.support).Nodup)
    (hA : ∀ g ∈ A.edges, g ∉ ({a, b, c} : Set E)) (hB : ∀ g ∈ B.edges, g ∉ ({a, b, c} : Set E)) :
    ∃ P' : S.Walk (ι s) (ι t), P'.IsPath ∧ P'.support = A.support.map ι ++ v :: B.support.map ι ∧
      P'.edgeSet = insert b (insert c (A.edgeSet ∪ B.edgeSet)) := by
  have hY : ({a, c, b} : Set E) = {a, b, c} := by rw [Set.pair_comm]
  rcases h.triangle.joins_a.eq_or_eq hpq.tail_or_head with rfl | rfl
  · obtain rfl := hpq.eq_of_joins h.triangle.joins_a
    obtain ⟨P', hP', hs, he⟩ := h.exists_isPath_through hAB hA hB
    refine ⟨P', hP', hs, ?_⟩
    ext g
    simp only [Walk.edgeSet, he, Set.mem_ofPred_eq, List.mem_append, List.mem_cons,
      Set.mem_insert_iff, Set.mem_union]
    tauto
  · obtain rfl := hpq.eq_of_joins h.triangle.joins_a.symm
    obtain ⟨P', hP', hs, he⟩ := h.swap₂₃.exists_isPath_through hAB (hY ▸ hA) (hY ▸ hB)
    refine ⟨P', hP', hs, ?_⟩
    ext g
    simp only [Walk.edgeSet, he, Set.mem_ofPred_eq, List.mem_append, List.mem_cons,
      Set.mem_insert_iff, Set.mem_union]
    tauto

end IsExchange

end OrientedGraph

end Transformer.GainMatroid
