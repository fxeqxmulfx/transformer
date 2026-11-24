module

public import Transformer.GainMatroid.Section2_RollUpJoints

/-!
# Frame circuits of a roll-up are frame circuits

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5:

> "For each unbalancing class `U` of `Σ_Ĝ(u)` there is a biased graph `(G_U, B_U)` for which
> `F(G_U, B_U) = F(G, B) = F(Ĝ, B̂)` […] (this fact is straightforward to check by comparing
> circuits […])."

This file compares the circuits in one direction: for a set `U` with the properties collected in
`IsRollUpClass`, every frame circuit of the roll-up `Ω.rollUp u U` is a frame circuit of `Ω`
(`IsRollUpClass.isFrameCircuit_of_rollUp`). A circuit off `U` keeps its shape, and a theta of the
roll-up has only links, so it avoids `U`. A circuit through an edge `e` of `U` uses `e` as a joint
at the end of `e` away from `u`:

* two joints at the same vertex unroll to two links at `u`, a balanced cycle;
* a joint and a cycle `D` off `U`, joined by a path, unroll to a tight handcuff at `u` or a theta,
  as `D` is unbalanced in `Ω` and so passes through `u`;
* two joints joined by a path `P` unroll to a balanced cycle if `P` avoids `u`, and to a tight
  handcuff at `u` otherwise (`IsRollUpClass.isFrameCircuit_of_joints_rollUp`, in
  `Section2_RollUpJoints`).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

open OrientedGraph

namespace BiasedGraph

variable {Ω : BiasedGraph V E} {u : V} {U : Set E}

namespace IsRollUpClass

/-- **§2.5**, one half of "`F(G_U, B_U) = F(Ĝ, B̂)`": a frame circuit of the roll-up is a frame
circuit of `Ω`. -/
theorem isFrameCircuit_of_rollUp (hU : Ω.IsRollUpClass u U) {X : Set E}
    (hX : (Ω.rollUp u U).IsFrameCircuit X) : Ω.IsFrameCircuit X := by
  rcases hX with hb | ⟨hs, hno⟩
  · exact Or.inl hb.1
  by_cases hd : Disjoint X U
  · have h := agreeOn_rollUp (Ω := Ω) (u := u) hd
    refine Or.inr ⟨?_, hU.notMem_balanced_of_rollUp hno fun g hg _ _ hgU _ =>
      absurd hgU (Set.disjoint_left.1 hd hg)⟩
    rcases hs with hs | hs | hs
    · exact Or.inl (hs.of_agreeOn h)
    · exact Or.inr (Or.inl (hs.of_agreeOn h))
    · exact Or.inr (Or.inr (hs.of_agreeOn h))
  -- A joint `e` of the roll-up, a cycle `D` off `U` and a path of the roll-up between them.
  have hJC {e : E} {D : Set E} {y : V} {P : (Ω.rollUp u U).graph.Walk (Ω.graph.otherEnd u e) y}
      (he : e ∈ U) (hD : (Ω.rollUp u U).graph.IsCycle D) (hDU : Disjoint D U) (hP : P.IsPath)
      (hyD : y ∈ (Ω.rollUp u U).graph.verts D)
      (hPD : ∀ z ∈ P.support, z ∈ (Ω.rollUp u U).graph.verts D → z = y)
      (hXe : X = {e} ∪ D ∪ P.edgeSet) : Ω.IsFrameCircuit X := by
    subst hXe
    have hA := agreeOn_rollUp (Ω := Ω) (u := u) hDU
    have hDG := hD.of_agreeOn hA
    have hDB : D ∉ Ω.balanced := fun h =>
      hno D (Set.subset_union_right.trans Set.subset_union_left) hD ⟨h, hDU⟩
    have hPU := disjoint_rollUp_of_isPath hP
    obtain ⟨Q, hQ, hQ₁, hQ₂⟩ := exists_isPath_of_rollUp hP
    rw [hA.verts_eq] at hyD hPD
    rw [← hQ₂] at hPD
    have hXQ : {e} ∪ D ∪ P.edgeSet = insert e Q.edgeSet ∪ D := by
      have hQe : Q.edgeSet = P.edgeSet := by
        unfold Walk.edgeSet
        rw [hQ₁]
      rw [hQe]
      ext g
      simp only [Set.mem_union, Set.mem_insert_iff, Set.mem_singleton_iff]
      tauto
    have hXe : ∀ g ∈ {e} ∪ D ∪ P.edgeSet, g ∈ U → g = e := fun g hg hgU => by
      rcases hg with (rfl | hg) | hg
      · rfl
      · exact absurd hgU (Set.disjoint_left.1 hDU hg)
      · exact absurd hgU (Set.disjoint_left.1 hPU hg)
    refine Or.inr ⟨?_, hU.notMem_balanced_of_rollUp hno fun g hg g' hg' hgU hg'U =>
      (hXe g hg hgU).trans (hXe g' hg' hg'U).symm⟩
    rw [hXQ]
    rcases hDG.isTightHandcuff_or_isThetaSet_insert (joins_otherEnd (hU.link he).2).symm
      (hU.vertex D ⟨hDG, hDB⟩) (fun h => Set.disjoint_left.1 hDU h he) hQ
      (by rw [hQ₁]; exact fun h => Set.disjoint_left.1 hPU h he) hyD hPD with h | h
    · exact Or.inl h
    · exact Or.inr (Or.inr h)
  rcases hs with ⟨C, D, v, hC, hD, hCD, hv, rfl⟩ |
    ⟨C, D, x, y, P, hC, hD, hCD, hP, hx, hy, hPC, hPD, rfl⟩ | ⟨x, y, P, Q, R, hT, rfl⟩
  · have hvCD : v ∈ (Ω.rollUp u U).graph.verts C ∩ (Ω.rollUp u U).graph.verts D := by
      rw [hv]
      exact Set.mem_singleton v
    obtain ⟨hvC, hvD⟩ := hvCD
    rcases eq_singleton_or_disjoint_rollUp hC with ⟨e, he, rfl⟩ | hCU <;>
      rcases eq_singleton_or_disjoint_rollUp hD with ⟨f, hf, rfl⟩ | hDU
    · rw [verts_rollUp_singleton he, Set.mem_singleton_iff] at hvC
      rw [verts_rollUp_singleton hf, Set.mem_singleton_iff] at hvD
      obtain ⟨hl, heu⟩ := hU.link he
      have hef : e ≠ f := Set.disjoint_singleton.1 hCD
      have hfj : Ω.graph.Joins f u (Ω.graph.otherEnd u e) := by
        rw [← hvC, hvD]
        exact (joins_otherEnd (hU.link hf).2).symm
      exact Or.inl (hU.pair _ (isCycle_pair (joins_otherEnd heu).symm hfj hef (otherEnd_ne hl).symm)
        e (Set.mem_insert _ _) f (Set.mem_insert_of_mem _ rfl) hef he hf)
    · rw [verts_rollUp_singleton he, Set.mem_singleton_iff] at hvC
      subst hvC
      refine hJC (P := .nil _) he hD hDU (by unfold Walk.IsPath Walk.support; simp) hvD
        (fun z hz _ => List.mem_singleton.1 hz) ?_
      ext g
      simp [Walk.edgeSet, Walk.edges]
    · rw [verts_rollUp_singleton hf, Set.mem_singleton_iff] at hvD
      subst hvD
      refine hJC (P := .nil _) hf hC hCU (by unfold Walk.IsPath Walk.support; simp) hvC
        (fun z hz _ => List.mem_singleton.1 hz) ?_
      ext g
      simp only [Set.mem_union, Set.mem_singleton_iff, Walk.edgeSet, Set.mem_ofPred_eq,
        Walk.edges, List.not_mem_nil, or_false]
      tauto
    · exact absurd (Set.disjoint_union_left.2 ⟨hCU, hDU⟩) hd
  · have hPU := disjoint_rollUp_of_isPath hP
    rcases eq_singleton_or_disjoint_rollUp hC with ⟨e, he, rfl⟩ | hCU <;>
      rcases eq_singleton_or_disjoint_rollUp hD with ⟨f, hf, rfl⟩ | hDU
    · rw [verts_rollUp_singleton he, Set.mem_singleton_iff] at hx
      rw [verts_rollUp_singleton hf, Set.mem_singleton_iff] at hy
      rw [verts_rollUp_singleton he, verts_rollUp_singleton hf, Set.disjoint_singleton] at hCD
      exact hU.isFrameCircuit_of_joints_rollUp he hf
        (fun h => hCD (congrArg (Ω.graph.otherEnd u) h)) hx.symm hy.symm hP hno
    · rw [verts_rollUp_singleton he, Set.mem_singleton_iff] at hx
      subst hx
      exact hJC he hD hDU hP hy hPD rfl
    · rw [verts_rollUp_singleton hf, Set.mem_singleton_iff] at hy
      subst hy
      refine hJC hf hC hCU hP.reverse hx
        (fun z hz hzC => hPC z (Walk.mem_support_reverse.1 hz) hzC) ?_
      rw [Walk.edgeSet_reverse, Set.union_comm C]
    · exact absurd (Set.disjoint_union_left.2 ⟨Set.disjoint_union_left.2 ⟨hCU, hDU⟩, hPU⟩) hd
  · exact absurd (Set.disjoint_union_left.2 ⟨Set.disjoint_union_left.2
      ⟨disjoint_rollUp_of_isPath hT.pathP, disjoint_rollUp_of_isPath hT.pathQ⟩,
      disjoint_rollUp_of_isPath hT.pathR⟩) hd

end IsRollUpClass

end BiasedGraph

open BiasedGraph

example := IsRollUpClass.isFrameCircuit_of_rollUp (Ω := pumpkin.gainBiased (1 : Fin 3 → ℚˣ))
  (u := 0) (U := ∅) ⟨fun C hC => (hC.2 (gainBiased_one_isBalanced pumpkin C hC.1)).elim,
    Set.empty_subset _, fun _ _ _ _ _ _ _ he _ => (Set.notMem_empty _ he).elim,
    fun _ _ _ _ he _ _ _ => (Set.notMem_empty _ he).elim⟩
  (Or.inl ⟨gainBiased_one_isBalanced pumpkin _ pumpkin_isTheta.isCycle.1, Set.disjoint_empty _⟩)

end Transformer.GainMatroid
