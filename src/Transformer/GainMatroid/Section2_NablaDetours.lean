module

public import Transformer.GainMatroid.Section2_NablaPaths

/-!
# Detours of paths of `∇_Y G`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, the proof of Proposition 2.7. Let `T`, `S` be
an exchange pair on `Y = {a, b, c}` (`IsExchange`), with the triangle `Y` in `T` and the star `Y` at
`v` in `S`. A path of `T` meets the triangle `Y` in at most two edges. If it meets `Y` in `a` alone,
its detour through `v` replaces `a` by `b` and `c`, with edges `P ∆ Y`
(`IsExchange.exists_isDetour_of_inter_eq_singleton`). If it meets `Y` in `b` and `c`, it runs
through their common corner `x` along both, and its detour runs through `v` instead, with the same
edges (`IsExchange.exists_isDetour_of_inter_eq_pair`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open scoped symmDiff

namespace OrientedGraph

variable {V W E : Type*} {T : OrientedGraph V E} {S : OrientedGraph W E}

/-- Appending walks is associative. -/
theorem Walk.append_assoc {G : OrientedGraph V E} {u v w z : V} (p : G.Walk u v) (q : G.Walk v w)
    (r : G.Walk w z) : (p.append q).append r = p.append (q.append r) := by
  induction p with
  | nil => rfl
  | cons e h p ih => exact congrArg (Walk.cons e h) (ih q)


namespace IsExchange

variable {ι : V → W} {v : W} {a b c : E} {x y z : V}

/-- A path of `T` whose only edge of `Y` is `a` has a detour through `v` with edges `P ∆ Y`. -/
theorem exists_isDetour_of_inter_eq_singleton (h : IsExchange T S ι v a b c x y z) {s t : V}
    {P : T.Walk s t} (hP : P.IsPath) (hPY : P.edgeSet ∩ {a, b, c} = {a}) :
    ∃ P' : S.Walk (ι s) (ι t), IsDetour {a, b, c} ι v P P' ∧
      P'.edgeSet = P.edgeSet ∆ {a, b, c} := by
  obtain ⟨p, q, hpq, A, B, rfl⟩ := P.exists_append_cons_of_mem ((Set.ext_iff.1 hPY a).2 rfl).1
  have hs : (A.append (.cons a hpq B)).support = A.support ++ B.support := by
    simp [Walk.support_append, Walk.support]
  have he : (A.append (.cons a hpq B)).edges = A.edges ++ a :: B.edges := by
    simp [Walk.edges_append, Walk.edges]
  have hnd := hP.edges_nodup
  rw [he, List.nodup_middle, List.nodup_cons] at hnd
  have hoff : ∀ g ∈ A.edges ++ B.edges, g ∉ ({a, b, c} : Set E) := fun g hg hgY => by
    have hgP : g ∈ (A.append (.cons a hpq B)).edges := by
      rw [he]
      rcases List.mem_append.1 hg with hg | hg
      · exact List.mem_append_left _ hg
      · exact List.mem_append_right _ (List.mem_cons_of_mem _ hg)
    obtain rfl : g = a := (Set.ext_iff.1 hPY g).1 ⟨hgP, hgY⟩
    exact hnd.1 hg
  have hAB : (A.support ++ B.support).Nodup := by
    rw [← hs]
    exact hP
  obtain ⟨P', hP', hP's, hP'e⟩ := h.exists_isPath_of_joins hpq hAB
    (fun g hg => hoff g (List.mem_append_left _ hg))
    (fun g hg => hoff g (List.mem_append_right _ hg))
  refine ⟨P', ⟨hP', fun w hw => ?_, fun g hg => ?_⟩, ?_⟩
  · rw [hP's, List.mem_append, List.mem_cons] at hw
    rw [hs]
    rcases hw with hw | rfl | hw
    · obtain ⟨u, hu, rfl⟩ := List.mem_map.1 hw
      exact Or.inr ⟨u, List.mem_append_left _ hu, rfl⟩
    · exact Or.inl rfl
    · obtain ⟨u, hu, rfl⟩ := List.mem_map.1 hw
      exact Or.inr ⟨u, List.mem_append_right _ hu, rfl⟩
  · have hg' : g ∈ P'.edgeSet := hg
    rw [hP'e] at hg'
    rw [he]
    rcases hg' with rfl | rfl | hg | hg
    · exact Or.inr (by simp)
    · exact Or.inr (by simp)
    · exact Or.inl (List.mem_append_left _ hg)
    · exact Or.inl (List.mem_append_right _ (List.mem_cons_of_mem _ hg))
  · have hPe : (A.append (.cons a hpq B)).edgeSet = insert a (A.edgeSet ∪ B.edgeSet) := by
      ext g
      simp only [Walk.edgeSet, he, Set.mem_ofPred_eq, List.mem_append, List.mem_cons,
        Set.mem_insert_iff, Set.mem_union]
      tauto
    rw [hP'e, hPe, insert_symmDiff_triple (fun g hg => hoff g (List.mem_append.2 hg))
      h.triangle.ne_ab h.triangle.ne_ac, Set.insert_union, Set.singleton_union]


/-- A path `A, e₁, e₂, B` of `T` meeting `Y` in `e₁` and `e₂` alone has a detour through `v` with
the same edges: `e₁` and `e₂` meet at a corner of the triangle, and the detour runs from the end of
`A` to `v` along `e₂` and on to the start of `B` along `e₁`. -/
theorem exists_isDetour_of_cons_cons (h : IsExchange T S ι v a b c x y z) {s t p w q : V}
    {e₁ e₂ : E} {A : T.Walk s p} {h₁ : T.Joins e₁ p w} {h₂ : T.Joins e₂ w q} {B : T.Walk q t}
    (hP : (A.append (.cons e₁ h₁ (.cons e₂ h₂ B))).IsPath)
    (hPY : (A.append (.cons e₁ h₁ (.cons e₂ h₂ B))).edgeSet ∩ {a, b, c} = {e₁, e₂}) :
    ∃ P' : S.Walk (ι s) (ι t), IsDetour {a, b, c} ι v (A.append (.cons e₁ h₁ (.cons e₂ h₂ B))) P' ∧
      P'.edgeSet = (A.append (.cons e₁ h₁ (.cons e₂ h₂ B))).edgeSet := by
  have he : (A.append (.cons e₁ h₁ (.cons e₂ h₂ B))).edges = A.edges ++ e₁ :: e₂ :: B.edges := by
    simp [Walk.edges_append, Walk.edges]
  have hs : (A.append (.cons e₁ h₁ (.cons e₂ h₂ B))).support = A.support ++ w :: B.support := by
    simp [Walk.support_append, Walk.support]
  have hY : ∀ g, g ∈ A.edges ∨ g ∈ B.edges → g ∈ ({a, b, c} : Set E) → g = e₁ ∨ g = e₂ :=
    fun g hg hgY => (Set.ext_iff.1 hPY g).1 ⟨show g ∈ (A.append _).edges by
      rw [he]; rcases hg with hg | hg <;> simp [hg], hgY⟩
  have he₁ := ((Set.ext_iff.1 hPY e₁).2 (Or.inl rfl)).2
  have he₂ := ((Set.ext_iff.1 hPY e₂).2 (Or.inr rfl)).2
  have hnd := hP.edges_nodup
  rw [he, List.nodup_middle, List.nodup_cons, List.nodup_middle, List.nodup_cons] at hnd
  have hne : e₁ ≠ e₂ := fun h' => hnd.1 (by simp [h'])
  have hoff : ∀ g ∈ A.edges ++ B.edges, g ∉ ({a, b, c} : Set E) := fun g hg hgY => by
    rcases hY g (List.mem_append.1 hg) hgY with rfl | rfl
    · exact hnd.1 (by rcases List.mem_append.1 hg with hg | hg <;> simp [hg])
    · exact hnd.2.1 hg
  have hAB : (A.support ++ B.support).Nodup := by
    have hnd' : (A.append (.cons e₁ h₁ (.cons e₂ h₂ B))).support.Nodup := hP
    rw [hs, List.nodup_middle, List.nodup_cons] at hnd'
    exact hnd'.2
  obtain ⟨e₀, x', y', z', h', hX⟩ := h.exists_relabel he₂ he₁ hne.symm
  have hw : w = x' := by
    rcases h'.triangle.joins_c.eq_or_eq h₁.symm.tail_or_head with hw | hw
    · exact hw
    · rcases h'.triangle.joins_b.eq_or_eq h₂.tail_or_head with hw' | hw'
      · exact hw'
      · exact absurd (hw.symm.trans hw') h'.triangle.ne_yz
  subst hw
  obtain rfl := h₁.symm.eq_of_joins h'.triangle.joins_c
  obtain rfl := h₂.eq_of_joins h'.triangle.joins_b
  obtain ⟨P', hP', hP's, hP'e⟩ := h'.exists_isPath_through hAB
    (fun g hg => hX ▸ hoff g (List.mem_append_left _ hg))
    (fun g hg => hX ▸ hoff g (List.mem_append_right _ hg))
  refine ⟨P', ⟨hP', fun r hr => ?_, fun g hg => Or.inl ?_⟩, ?_⟩
  · rw [hP's, List.mem_append, List.mem_cons] at hr
    rw [hs]
    rcases hr with hr | rfl | hr
    · obtain ⟨u, hu, rfl⟩ := List.mem_map.1 hr
      exact Or.inr ⟨u, List.mem_append_left _ hu, rfl⟩
    · exact Or.inl rfl
    · obtain ⟨u, hu, rfl⟩ := List.mem_map.1 hr
      exact Or.inr ⟨u, by simp [hu], rfl⟩
  · rw [hP'e] at hg
    rw [he]
    simp only [List.mem_append, List.mem_cons] at hg ⊢
    tauto
  · ext g
    simp only [Walk.edgeSet, hP'e, he, Set.mem_ofPred_eq, List.mem_append, List.mem_cons]
    tauto

/-- A path of `T` whose edges in `Y` are `b` and `c` has a detour through `v` with the same
edges. -/
theorem exists_isDetour_of_inter_eq_pair (h : IsExchange T S ι v a b c x y z) {s t : V}
    {P : T.Walk s t} (hP : P.IsPath) (hPY : P.edgeSet ∩ {a, b, c} = {b, c}) :
    ∃ P' : S.Walk (ι s) (ι t), IsDetour {a, b, c} ι v P P' ∧ P'.edgeSet = P.edgeSet := by
  have hb : b ∈ P.edges := ((Set.ext_iff.1 hPY b).2 (Or.inl rfl)).1
  obtain ⟨s₁, t₁, hj, A, B₀, rfl⟩ := P.exists_append_cons_of_mem ((Set.ext_iff.1 hPY c).2
    (Or.inr rfl)).1
  have hnd : (A.support ++ B₀.support).Nodup := by
    have hnd : (A.append (.cons c hj B₀)).support.Nodup := hP
    simpa [Walk.support_append, Walk.support] using hnd
  have hbAB : b ∈ A.edges ∨ b ∈ B₀.edges := by
    have hb' : b ∈ A.edges ++ c :: B₀.edges := by simpa [Walk.edges_append, Walk.edges] using hb
    rcases List.mem_append.1 hb' with hb' | hb'
    · exact Or.inl hb'
    · exact Or.inr ((List.mem_cons.1 hb').resolve_left h.triangle.ne_bc)
  have hxb := h.triangle.joins_b.tail_or_head
  rcases h.triangle.joins_c.eq_or_eq hj.tail_or_head with rfl | rfl
  · obtain rfl := hj.eq_of_joins h.triangle.joins_c
    have hbA : b ∈ A.edges := hbAB.resolve_right fun hbB =>
      List.disjoint_of_nodup_append hnd A.end_mem_support (Walk.mem_support_of_mem_edges hbB hxb)
    obtain ⟨s₂, t₂, hj', A₁, A₂, rfl⟩ := A.exists_append_cons_of_mem hbA
    have hA : (A₁.append (.cons b hj' A₂)).IsPath := hP.of_append_left
    rcases h.triangle.joins_b.eq_or_eq hj'.tail_or_head with rfl | rfl
    · have hnd' : (A₁.support ++ A₂.support).Nodup := by
        have hnd' : (A₁.append (.cons b hj' A₂)).support.Nodup := hA
        simpa [Walk.support_append, Walk.support] using hnd'
      exact (List.disjoint_of_nodup_append hnd' A₁.end_mem_support A₂.end_mem_support).elim
    · obtain rfl := hj'.eq_of_joins h.triangle.joins_b.symm
      obtain rfl := (Walk.isPath_cons.1 hA.of_append_right).2.eq_nil
      rw [Walk.append_assoc] at hP hPY ⊢
      exact h.exists_isDetour_of_cons_cons hP hPY
  · obtain rfl := hj.eq_of_joins h.triangle.joins_c.symm
    have hbB : b ∈ B₀.edges := hbAB.resolve_left fun hbA =>
      List.disjoint_of_nodup_append hnd (Walk.mem_support_of_mem_edges hbA hxb) B₀.start_mem_support
    cases B₀ with
    | nil => simp [Walk.edges] at hbB
    | cons g hg B =>
      obtain rfl := (Walk.isPath_cons.1 hP.of_append_right).2.eq_of_mem_start hbB hxb
      exact h.exists_isDetour_of_cons_cons hP (hPY.trans (Set.pair_comm b c))

end IsExchange

end OrientedGraph

end Transformer.GainMatroid
