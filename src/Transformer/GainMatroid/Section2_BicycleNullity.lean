module

public import Transformer.GainMatroid.Section2_CycleBicycles
public import Transformer.GainMatroid.Section2_CycleNullity

/-!
# The nullity of thetas, handcuffs and pairs of cycles

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2. The circuits of `F(G, B)` and `L(G, B)` that
are not balanced cycles have nullity two: they contain two distinct cycles (`two_le_nullity`), and
they are built from one cycle by adding paths whose inner vertices are new.

* A walk whose inner vertices are new raises the nullity by at most one
  (`nullity_union_le_of_ear`), and a path whose vertices other than its start are new keeps it
  (`nullity_union_le_of_hanging`).
* Hence a theta, a tight or loose handcuff and two vertex-disjoint cycles have nullity at most two
  (`IsThetaSet.nullity_le`, `IsTightHandcuff.nullity_le`, `IsLooseHandcuff.nullity_le`,
  `IsDisjointCycles.nullity_le`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E : Type*} {G : OrientedGraph V E} [Finite V] [Finite E]

/-- **An ear raises the nullity by at most one.** Adding the edges of a walk whose inner vertices
are distinct, differ from its ends and are not on `S` raises the nullity by at most one. -/
theorem nullity_union_le_of_ear {S : Set E} {a b : V} (p : G.Walk a b)
    (hnd : p.support.tail.Nodup) (hab : a ∉ p.support.tail ∨ a = b)
    (hfresh : ∀ z ∈ p.support, z ≠ a → z ≠ b → z ∉ G.verts S) :
    G.nullity (S ∪ p.edgeSet) ≤ G.nullity S + 1 := by
  induction p generalizing S with
  | nil a => simp [Walk.edgeSet, Walk.edges]
  | @cons a w b e hj q ih =>
    have hq : q.IsPath := hnd
    rw [Walk.edgeSet_cons, Set.union_insert, ← Set.insert_union]
    by_cases hwb : w = b
    · subst hwb
      obtain rfl := hq.eq_nil
      simpa [Walk.edgeSet, Walk.edges] using nullity_insert_le S e
    have hwq : w ∈ q.support := q.start_mem_support
    have hwa : w ≠ a := by
      rintro rfl
      rcases hab with h | rfl
      · exact h hwq
      · exact hwb rfl
    have hwS : w ∉ G.verts S := hfresh w (List.mem_cons_of_mem _ hwq) hwa hwb
    rw [← nullity_insert_of_not_reachableIn (not_reachableIn_of_joins hj hwa hwS)]
    refine ih hq.tail_nodup (Or.inl hq.start_notMem_tail) fun z hz hzw hzb => ?_
    have hza : z ≠ a := by
      rintro rfl
      rcases hab with h | rfl
      · exact h hz
      · exact hzb rfl
    exact notMem_verts_insert hj hza hzw (hfresh z (List.mem_cons_of_mem _ hz) hza hzb)

/-- **A hanging path keeps the nullity.** Adding the edges of a path whose vertices other than its
start are not on `S` keeps the nullity. -/
theorem nullity_union_le_of_hanging {S : Set E} {a b : V} (p : G.Walk a b) (hp : p.IsPath)
    (hfresh : ∀ z ∈ p.support, z ≠ a → z ∉ G.verts S) :
    G.nullity (S ∪ p.edgeSet) ≤ G.nullity S := by
  induction p generalizing S with
  | nil a => simp [Walk.edgeSet, Walk.edges]
  | @cons a w b e hj q ih =>
    obtain ⟨haq, hq⟩ := Walk.isPath_cons.1 hp
    have hwa : w ≠ a := fun h => haq (h ▸ q.start_mem_support)
    have hwS : w ∉ G.verts S := hfresh w (List.mem_cons_of_mem _ q.start_mem_support) hwa
    rw [Walk.edgeSet_cons, Set.union_insert, ← Set.insert_union,
      ← nullity_insert_of_not_reachableIn (not_reachableIn_of_joins hj hwa hwS)]
    refine ih hq fun z hz hzw => ?_
    have hza : z ≠ a := fun h => haq (h ▸ hz)
    exact notMem_verts_insert hj hza hzw (hfresh z (List.mem_cons_of_mem _ hz) hza)

/-- A theta has nullity at most two. -/
theorem IsThetaSet.nullity_le {X : Set E} (h : G.IsThetaSet X) : G.nullity X ≤ 2 := by
  obtain ⟨x, y, P, Q, R, hT, rfl⟩ := h
  have hle := nullity_union_le_of_ear (S := P.edgeSet ∪ Q.edgeSet) R hT.pathR.tail_nodup
    (Or.inl hT.pathR.start_notMem_tail) fun z hz hzx hzy hzS => by
      obtain ⟨f, hf, hfz⟩ := mem_verts.1 hzS
      rcases hf with hf | hf
      · exact (hT.openPR.verts z (Walk.mem_support_of_mem_edges hf hfz) hz).elim hzx hzy
      · exact (hT.openQR.verts z (Walk.mem_support_of_mem_edges hf hfz) hz).elim hzx hzy
  rwa [hT.isCycle.1.nullity_eq_one] at hle

/-- Adding a cycle that meets `S` in at most one vertex `v` raises the nullity by at most one. -/
theorem IsCycle.nullity_union_le {S D : Set E} (hD : G.IsCycle D) {v : V} (hv : v ∈ G.verts D)
    (hDS : ∀ z ∈ G.verts D, z ∈ G.verts S → z = v) : G.nullity (S ∪ D) ≤ G.nullity S + 1 := by
  obtain ⟨W, hW, rfl⟩ := hD.exists_walk_at hv
  exact nullity_union_le_of_ear W hW.support_tail_nodup (Or.inr rfl) fun z hz hzv _ hzS =>
    hzv (hDS z (W.mem_verts_of_mem_support hW.edges_ne_nil hz) hzS)

/-- A tight handcuff has nullity at most two. -/
theorem IsTightHandcuff.nullity_le {X : Set E} (h : G.IsTightHandcuff X) : G.nullity X ≤ 2 := by
  obtain ⟨C, D, v, hC, hD, -, hv, rfl⟩ := h
  have hvD : v ∈ G.verts D := (hv.symm ▸ Set.mem_singleton v : v ∈ G.verts C ∩ G.verts D).2
  have := hD.nullity_union_le hvD fun z hzD hzC =>
    Set.mem_singleton_iff.1 (hv ▸ (⟨hzC, hzD⟩ : z ∈ G.verts C ∩ G.verts D))
  rwa [hC.nullity_eq_one] at this

/-- Two vertex-disjoint cycles have nullity at most two. -/
theorem IsDisjointCycles.nullity_le {X : Set E} (h : G.IsDisjointCycles X) :
    G.nullity X ≤ 2 := by
  obtain ⟨C, D, hC, hD, hCD, rfl⟩ := h
  obtain ⟨e, he⟩ := hD.nonempty
  have hv : G.tail e ∈ G.verts D := mem_verts.2 ⟨e, he, Or.inl rfl⟩
  have := hD.nullity_union_le hv fun z hzD hzC => (Set.disjoint_left.1 hCD hzC hzD).elim
  rwa [hC.nullity_eq_one] at this

/-- A loose handcuff has nullity at most two. -/
theorem IsLooseHandcuff.nullity_le {X : Set E} (h : G.IsLooseHandcuff X) : G.nullity X ≤ 2 := by
  obtain ⟨C, D, x, y, P, hC, hD, hCD, hP, hx, hy, hPC, hPD, rfl⟩ := h
  have h₁ : G.nullity (C ∪ P.edgeSet) ≤ 1 := by
    rw [← hC.nullity_eq_one]
    exact nullity_union_le_of_hanging P hP fun z hz hzx hzC => hzx (hPC z hz hzC)
  have hyD : y ∈ G.verts D := hy
  have h₂ := hD.nullity_union_le (S := C ∪ P.edgeSet) hyD fun z hzD hzS => by
    obtain ⟨f, hf, hfz⟩ := mem_verts.1 hzS
    rcases hf with hf | hf
    · exact (Set.disjoint_left.1 hCD (mem_verts.2 ⟨f, hf, hfz⟩) hzD).elim
    · exact hPD z (Walk.mem_support_of_mem_edges hf hfz) hzD
  rw [show C ∪ D ∪ P.edgeSet = C ∪ P.edgeSet ∪ D by
    rw [Set.union_assoc, Set.union_comm D, ← Set.union_assoc]]
  omega

/-- A loop at `0`, a loop at `1` and a link from `0` to `1`, for the examples. -/
def loopLink : OrientedGraph (Fin 2) (Fin 3) := ⟨![0, 1, 0], ![0, 1, 1]⟩

/-- The link of `loopLink` as a walk. -/
def loopLinkWalk : loopLink.Walk 0 1 := .cons 2 (Or.inl ⟨rfl, rfl⟩) (.nil 1)

theorem loopLink_disjoint : Disjoint (loopLink.verts {0}) (loopLink.verts {1}) := by
  rw [Set.disjoint_left]
  intro v hv hv'
  obtain ⟨e, rfl, he⟩ := mem_verts.1 hv
  obtain ⟨f, rfl, hf⟩ := mem_verts.1 hv'
  rcases he with rfl | rfl <;> rcases hf with h | h <;> exact absurd h (by decide)

/-- The two loops of `loopLink` and its link form a loose handcuff. -/
theorem loopLink_isLooseHandcuff :
    loopLink.IsLooseHandcuff ({0} ∪ {1} ∪ loopLinkWalk.edgeSet) := by
  refine ⟨{0}, {1}, 0, 1, loopLinkWalk, isCycle_singleton_of_loop rfl,
    isCycle_singleton_of_loop rfl, loopLink_disjoint, by unfold Walk.IsPath; decide,
    mem_verts.2 ⟨0, rfl, Or.inl rfl⟩, mem_verts.2 ⟨1, rfl, Or.inl rfl⟩, ?_, ?_, rfl⟩ <;>
  · intro z _ hz
    obtain ⟨e, rfl, h⟩ := mem_verts.1 hz
    rcases h with rfl | rfl <;> rfl

/-- The two loops of `twoLoops` form a tight handcuff. -/
theorem twoLoops_isTightHandcuff : twoLoops.IsTightHandcuff ({true} ∪ {false}) :=
  ⟨{true}, {false}, (), isCycle_singleton_of_loop rfl, isCycle_singleton_of_loop rfl,
    Set.disjoint_singleton.2 (by decide),
    Set.eq_singleton_iff_unique_mem.2 ⟨⟨mem_verts.2 ⟨true, rfl, Or.inl rfl⟩,
      mem_verts.2 ⟨false, rfl, Or.inl rfl⟩⟩, fun _ _ => rfl⟩, rfl⟩

example := nullity_union_le_of_ear (S := ∅) pathThreeFirst (by decide) (Or.inl (by decide))
  fun _ _ _ _ => by simp [verts]

example := nullity_union_le_of_hanging (S := ∅) pathThreeFirst (by unfold Walk.IsPath; decide)
  fun _ _ _ => by simp [verts]

example := IsThetaSet.nullity_le (G := pumpkin)
  ⟨0, 1, pumpkinWalk 0, pumpkinWalk 1, pumpkinWalk 2, pumpkin_isTheta, rfl⟩

example := pumpkin_pair_isCycle.nullity_union_le (S := ∅)
  (mem_verts.2 ⟨0, Set.mem_insert 0 {1}, Or.inl rfl⟩) fun _ _ hz => by simp [verts] at hz

example := IsTightHandcuff.nullity_le twoLoops_isTightHandcuff

example := IsDisjointCycles.nullity_le (G := loopLink) ⟨{0}, {1}, isCycle_singleton_of_loop rfl,
  isCycle_singleton_of_loop rfl, loopLink_disjoint, rfl⟩

example := IsLooseHandcuff.nullity_le loopLink_isLooseHandcuff

end OrientedGraph

end Transformer.GainMatroid
