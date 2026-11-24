module

public import Transformer.GainMatroid.Section2_AgreeingGraphs

/-!
# The cycles of a roll-up

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5, obtain the roll-up `(G_U, B_U)` from `(Ĝ, B̂)`
"by replacing each link `e = uv ∈ U` with a joint incident to `v`", and compare the circuits of the
two. `BiasedGraph.rollUp` keeps the ends of every edge off `U` (`agreeOn_rollUp`), so the cycles of
the roll-up that avoid `U` are those of `Ω`, and an edge of `U` becomes a joint at its end away
from `u` (`isJoint_rollUp`, `verts_rollUp_singleton`), the only cycle of the roll-up through it
(`eq_singleton_of_mem_rollUp`), so every other cycle of the roll-up avoids `U`
(`eq_singleton_or_disjoint_rollUp`). A link of the roll-up is not in `U`
(`notMem_of_isLink_rollUp`), so no path of the roll-up meets `U` (`disjoint_rollUp_of_isPath`) and
each is a path of `Ω` (`exists_isPath_of_rollUp`). The balanced cycles of the roll-up are balanced
in `Ω`, so a set without balanced cycles of `Ω` has none of the roll-up (`notMem_balanced_rollUp`).
Two edges of `U` and a path of the roll-up between their joints form a frame circuit of the roll-up
(`isFrameCircuit_rollUp_of_joints`), and an edge of `U`, a cycle off `U` and a path of the roll-up
from its joint to the cycle form a handcuff of the roll-up (`isHandcuff_rollUp`).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

open OrientedGraph

namespace BiasedGraph

variable {Ω : BiasedGraph V E} {u : V} {U : Set E}

theorem rollUp_tail_of_mem {e : E} (he : e ∈ U) :
    (Ω.rollUp u U).graph.tail e = Ω.graph.otherEnd u e :=
  rollUpGraph_tail_of_mem he

theorem rollUp_head_of_mem {e : E} (he : e ∈ U) :
    (Ω.rollUp u U).graph.head e = Ω.graph.otherEnd u e :=
  rollUpGraph_head_of_mem he

/-- The roll-up gives the edges off `U` their ends in `Ω`. -/
theorem agreeOn_rollUp {X : Set E} (hX : Disjoint X U) :
    (Ω.rollUp u U).graph.AgreeOn Ω.graph X := fun _ hg =>
  ⟨rollUpGraph_tail_of_notMem (Set.disjoint_left.1 hX hg),
    rollUpGraph_head_of_notMem (Set.disjoint_left.1 hX hg)⟩

/-- Each edge of `U` is a joint of the roll-up. -/
theorem isJoint_rollUp {e : E} (he : e ∈ U) : (Ω.rollUp u U).IsJoint e :=
  ⟨(rollUp_tail_of_mem he).trans (rollUp_head_of_mem he).symm,
    fun hb => Set.disjoint_left.1 hb.2 (Set.mem_singleton e) he⟩

/-- A link of the roll-up is not in `U`. -/
theorem notMem_of_isLink_rollUp {e : E} (he : (Ω.rollUp u U).graph.IsLink e) : e ∉ U :=
  fun heU => he (isJoint_rollUp heU).1

/-- In the roll-up an edge of `U` meets only its end away from `u` in `Ω`. -/
theorem verts_rollUp_singleton {e : E} (he : e ∈ U) :
    (Ω.rollUp u U).graph.verts {e} = {Ω.graph.otherEnd u e} := by
  ext v
  rw [mem_verts, Set.mem_singleton_iff]
  constructor
  · rintro ⟨g, hg, hv⟩
    rw [Set.mem_singleton_iff.1 hg, rollUp_tail_of_mem he, rollUp_head_of_mem he, or_self] at hv
    exact hv.symm
  · rintro rfl
    exact ⟨e, Set.mem_singleton e, Or.inl (rollUp_tail_of_mem he)⟩

/-- A cycle of the roll-up through an edge of `U` is that edge. -/
theorem eq_singleton_of_mem_rollUp {C : Set E} (hC : (Ω.rollUp u U).graph.IsCycle C) {e : E}
    (heC : e ∈ C) (he : e ∈ U) : C = {e} :=
  hC.eq_singleton_of_loop heC (isJoint_rollUp he).1

/-- A cycle of the roll-up is an edge of `U` or avoids `U`. -/
theorem eq_singleton_or_disjoint_rollUp {C : Set E} (hC : (Ω.rollUp u U).graph.IsCycle C) :
    (∃ e ∈ U, C = {e}) ∨ Disjoint C U := by
  by_cases hd : Disjoint C U
  · exact Or.inr hd
  obtain ⟨e, heC, he⟩ := Set.not_disjoint_iff.1 hd
  exact Or.inl ⟨e, he, eq_singleton_of_mem_rollUp hC heC he⟩

/-- A path of the roll-up avoids `U`: its edges are links. -/
theorem disjoint_rollUp_of_isPath {x y : V} {P : (Ω.rollUp u U).graph.Walk x y}
    (hP : P.IsPath) : Disjoint P.edgeSet U :=
  Set.disjoint_left.2 fun _ hg => notMem_of_isLink_rollUp (hP.isLink_of_mem hg)

/-- A path of the roll-up is a path of `Ω` through the same edges and vertices. -/
theorem exists_isPath_of_rollUp {x y : V} {P : (Ω.rollUp u U).graph.Walk x y} (hP : P.IsPath) :
    ∃ Q : Ω.graph.Walk x y, Q.IsPath ∧ Q.edges = P.edges ∧ Q.support = P.support := by
  obtain ⟨Q, hQ₁, hQ₂⟩ := P.exists_of_agreeOn rfl rfl
    (agreeOn_rollUp (disjoint_rollUp_of_isPath hP))
  refine ⟨Q, ?_, hQ₁, hQ₂⟩
  unfold Walk.IsPath
  rw [hQ₂]
  exact hP

/-- A set with no balanced cycle of `Ω` has no balanced cycle of the roll-up. -/
theorem notMem_balanced_rollUp {X : Set E} (hX : ∀ C ⊆ X, Ω.graph.IsCycle C → C ∉ Ω.balanced) :
    ∀ C ⊆ X, (Ω.rollUp u U).graph.IsCycle C → C ∉ (Ω.rollUp u U).balanced :=
  fun C hCX _ hCB => hX C hCX (Ω.isCycle_of_mem C hCB.1) hCB.1

/-- Two edges of `U` and a path of the roll-up between their ends away from `u` form a frame
circuit of the roll-up: two joints and a path joining them. -/
theorem isFrameCircuit_rollUp_of_joints {e f : E} (he : e ∈ U) (hf : f ∈ U) (hef : e ≠ f)
    {P : (Ω.rollUp u U).graph.Walk (Ω.graph.otherEnd u e) (Ω.graph.otherEnd u f)}
    (hP : P.IsPath) : (Ω.rollUp u U).IsFrameCircuit (insert e (insert f P.edgeSet)) := by
  obtain ⟨q, hq₁, hq₂⟩ := P.exists_of_agreeOn (H := (Ω.rollUp u U).graph)
    (rollUp_tail_of_mem he).symm (rollUp_tail_of_mem hf).symm fun _ _ => ⟨rfl, rfl⟩
  have hq : q.IsPath := by
    unfold Walk.IsPath
    rw [hq₂]
    exact hP
  have hqP : q.edgeSet = P.edgeSet := by
    unfold Walk.edgeSet
    rw [hq₁]
  have h := isFrameCircuit_of_joints (isJoint_rollUp he) (isJoint_rollUp hf) hef hq
  rwa [hqP, Set.insert_union, Set.singleton_union] at h

/-- An edge `e` of `U`, a cycle `D` of `Ω` off `U`, and a path of the roll-up from the end of `e`
away from `u` to a vertex `y` of `D` that meets `D` only at `y` form a tight handcuff of the roll-up
if the path is empty, and a loose one otherwise. -/
theorem isHandcuff_rollUp {e : E} (he : e ∈ U) {D : Set E} (hD : Ω.graph.IsCycle D)
    (hDU : Disjoint D U) {y : V} {P : (Ω.rollUp u U).graph.Walk (Ω.graph.otherEnd u e) y}
    (hP : P.IsPath) (hyD : y ∈ Ω.graph.verts D)
    (hPD : ∀ z ∈ P.support, z ∈ Ω.graph.verts D → z = y) :
    (Ω.rollUp u U).graph.IsTightHandcuff ({e} ∪ D ∪ P.edgeSet) ∨
      (Ω.rollUp u U).graph.IsLooseHandcuff ({e} ∪ D ∪ P.edgeSet) := by
  have hA := agreeOn_rollUp (Ω := Ω) (u := u) hDU
  rw [← hA.verts_eq] at hyD hPD
  have hJ := isCycle_singleton_of_loop (isJoint_rollUp (Ω := Ω) (u := u) he).1
  rcases eq_or_ne (Ω.graph.otherEnd u e) y with rfl | hy
  · obtain rfl := hP.eq_nil
    refine Or.inl ⟨{e}, D, Ω.graph.otherEnd u e, hJ, hD.of_agreeOn hA.symm,
      Set.disjoint_singleton_left.2 fun h => Set.disjoint_left.1 hDU h he, ?_, ?_⟩
    · rw [verts_rollUp_singleton he]
      exact Set.eq_singleton_iff_unique_mem.2 ⟨⟨Set.mem_singleton _, hyD⟩, fun _ hz => hz.1⟩
    · ext g
      simp [Walk.edgeSet, Walk.edges]
  refine Or.inr ⟨{e}, D, Ω.graph.otherEnd u e, y, P, hJ, hD.of_agreeOn hA.symm, ?_, hP, ?_, hyD,
    fun z _ hz => ?_, hPD, rfl⟩
  · rw [verts_rollUp_singleton he, Set.disjoint_singleton_left]
    exact fun h => hy (hPD _ P.start_mem_support h)
  · rw [verts_rollUp_singleton he]
    exact Set.mem_singleton _
  · rw [verts_rollUp_singleton he] at hz
    exact hz

end BiasedGraph

open BiasedGraph

/-- Each edge of `pumpkin` is a link at `0`. -/
theorem pumpkin_mem_linkStar (i : Fin 3) :
    i ∈ (pumpkin.gainBiased (1 : Fin 3 → ℚˣ)).linkStar 0 :=
  ⟨(by decide : (0 : Fin 2) ≠ 1), Or.inl rfl⟩

example := rollUp_tail_of_mem (Ω := pumpkin.gainBiased (1 : Fin 3 → ℚˣ)) (u := 0)
  (pumpkin_mem_linkStar 0)
example := rollUp_head_of_mem (Ω := pumpkin.gainBiased (1 : Fin 3 → ℚˣ)) (u := 0)
  (pumpkin_mem_linkStar 0)
example := agreeOn_rollUp (Ω := pumpkin.gainBiased (1 : Fin 3 → ℚˣ)) (u := 0) (X := Set.univ)
  (Set.disjoint_empty _)
example := isJoint_rollUp (Ω := pumpkin.gainBiased (1 : Fin 3 → ℚˣ)) (u := 0)
  (pumpkin_mem_linkStar 0)
example := notMem_of_isLink_rollUp (Ω := pumpkin.gainBiased (1 : Fin 3 → ℚˣ)) (u := 0) (U := ∅)
  (e := 0) fun h => absurd ((rollUpGraph_tail_of_notMem (Set.notMem_empty 0)).symm.trans
    (h.trans (rollUpGraph_head_of_notMem (Set.notMem_empty 0)))) (by decide)
example := verts_rollUp_singleton (Ω := pumpkin.gainBiased (1 : Fin 3 → ℚˣ)) (u := 0)
  (pumpkin_mem_linkStar 0)
example := eq_singleton_of_mem_rollUp (isCycle_singleton_of_loop
  (isJoint_rollUp (Ω := pumpkin.gainBiased (1 : Fin 3 → ℚˣ)) (u := 0) (pumpkin_mem_linkStar 0)).1)
  (Set.mem_singleton 0) (pumpkin_mem_linkStar 0)
example := eq_singleton_or_disjoint_rollUp (isCycle_singleton_of_loop
  (isJoint_rollUp (Ω := pumpkin.gainBiased (1 : Fin 3 → ℚˣ)) (u := 0) (pumpkin_mem_linkStar 0)).1)
example := disjoint_rollUp_of_isPath (Ω := pumpkin.gainBiased (1 : Fin 3 → ℚˣ)) (u := 0) (U := ∅)
  (P := .nil 0) (by unfold Walk.IsPath Walk.support; exact List.nodup_singleton _)
example := exists_isPath_of_rollUp (Ω := pumpkin.gainBiased (1 : Fin 3 → ℚˣ)) (u := 0) (U := ∅)
  (P := .nil 0) (by unfold Walk.IsPath Walk.support; exact List.nodup_singleton _)
example := notMem_balanced_rollUp (Ω := twoJointsHat.contrabalanced) (u := 0) (U := ∅)
  (X := Set.univ) fun C _ _ hb => Set.notMem_empty C hb
example := isFrameCircuit_rollUp_of_joints (Ω := pumpkin.gainBiased (1 : Fin 3 → ℚˣ)) (u := 0)
  (pumpkin_mem_linkStar 0) (pumpkin_mem_linkStar 1) (by decide) (P := .nil _)
  (by unfold Walk.IsPath; exact List.nodup_singleton _)
example := isHandcuff_rollUp (Ω := pumpkin.contrabalanced) (u := 0) (U := {0}) (e := 0) rfl
  (isCycle_pair (G := pumpkin) (e := 1) (f := 2) (u := 0) (v := 1) (Or.inl ⟨rfl, rfl⟩)
    (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)) (Set.disjoint_singleton_right.2 (by simp))
  (P := .nil _) (by unfold Walk.IsPath; exact List.nodup_singleton _)
  (by rw [Joins.otherEnd_eq (G := pumpkin.contrabalanced.graph) (e := 0) (y := 1) (u := 0)
        (Or.inr ⟨rfl, rfl⟩)]
      exact mem_verts.2 ⟨1, Or.inl rfl, Or.inr rfl⟩)
  fun _ hz _ => List.mem_singleton.1 hz

end Transformer.GainMatroid
