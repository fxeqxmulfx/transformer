module

public import Transformer.GainMatroid.Section2_RollUpThetas

/-!
# Frame circuits are frame circuits of a roll-up

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5:

> "For each unbalancing class `U` of `Σ_Ĝ(u)` there is a biased graph `(G_U, B_U)` for which
> `F(G_U, B_U) = F(G, B) = F(Ĝ, B̂)` […] (this fact is straightforward to check by comparing
> circuits […])."

This file compares the circuits in the direction opposite to `Section2_RollUpForward`: for a set
`U` with the properties collected in `IsRollUpClass`, every frame circuit of `Ω` is a frame circuit
of the roll-up `Ω.rollUp u U` (`IsRollUpClass.isFrameCircuit_rollUp`). A circuit off `U` keeps its
shape. Each edge of `U` in a circuit becomes a joint at its end away from `u`:

* a balanced cycle through an edge of `U` is two edges of `U` and a path off `u`, which roll up to
  two joints joined by a path;
* the two cycles of a tight handcuff are unbalanced, so they meet at `u`, and each has one edge of
  `U` at most: one such edge rolls up to a loose handcuff, two to two joints joined by a path;
* the two cycles of a loose handcuff are unbalanced and vertex-disjoint, so they cannot both pass
  through `u`;
* a theta through an edge of `U` rolls up to a handcuff (`IsRollUpClass.exists_of_isTheta`, in
  `Section2_RollUpThetas`).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

open OrientedGraph

namespace BiasedGraph

variable {Ω : BiasedGraph V E} {u : V} {U : Set E}

namespace IsRollUpClass

/-- **§2.5**, the other half of "`F(G_U, B_U) = F(Ĝ, B̂)`": a frame circuit of `Ω` is a frame
circuit of the roll-up. -/
theorem isFrameCircuit_rollUp (hU : Ω.IsRollUpClass u U) {X : Set E} (hX : Ω.IsFrameCircuit X) :
    (Ω.rollUp u U).IsFrameCircuit X := by
  rcases hX with hb | ⟨hs, hno⟩
  · by_cases hd : Disjoint X U
    · exact Or.inl ⟨hb, hd⟩
    obtain ⟨e, heX, he⟩ := Set.not_disjoint_iff.1 hd
    obtain ⟨f, -, hf, hfe, P, hP, -, rfl⟩ := hU.exists_of_mem_balanced hb heX he
    exact isFrameCircuit_rollUp_of_joints he hf hfe.symm hP
  have hno' := notMem_balanced_rollUp (u := u) (U := U) hno
  by_cases hd : Disjoint X U
  · have h := (agreeOn_rollUp (Ω := Ω) (u := u) hd).symm
    refine Or.inr ⟨?_, hno'⟩
    rcases hs with hs | hs | hs
    · exact Or.inl (hs.of_agreeOn h)
    · exact Or.inr (Or.inl (hs.of_agreeOn h))
    · exact Or.inr (Or.inr (hs.of_agreeOn h))
  -- An edge `e` of `U`, a cycle `D` off `U` and a path of the roll-up from `e` to `D`.
  have hJD {e : E} {D : Set E} {y : V} {P : (Ω.rollUp u U).graph.Walk (Ω.graph.otherEnd u e) y}
      (he : e ∈ U) (hD : Ω.graph.IsCycle D) (hDU : Disjoint D U) (hP : P.IsPath)
      (hyD : y ∈ Ω.graph.verts D) (hPD : ∀ z ∈ P.support, z ∈ Ω.graph.verts D → z = y)
      (hXe : X = {e} ∪ D ∪ P.edgeSet) : (Ω.rollUp u U).IsFrameCircuit X := by
    subst hXe
    refine Or.inr ⟨?_, hno'⟩
    rcases isHandcuff_rollUp he hD hDU hP hyD hPD with h | h
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
  rcases hs with ⟨C, D, v, hC, hD, hCD, hv, rfl⟩ |
    ⟨C, D, x, y, P, hC, hD, hCD, hP, hx, hy, hPC, hPD, rfl⟩ | ⟨x, y, P, Q, R, hT, hXT⟩
  · have hCB : C ∉ Ω.balanced := hno C Set.subset_union_left hC
    have hDB : D ∉ Ω.balanced := hno D Set.subset_union_right hD
    have hu : u ∈ Ω.graph.verts C ∩ Ω.graph.verts D :=
      ⟨hU.vertex C ⟨hC, hCB⟩, hU.vertex D ⟨hD, hDB⟩⟩
    rw [hv, Set.mem_singleton_iff] at hu
    rw [← hu] at hv
    have hCD' (z : V) (hzC : z ∈ Ω.graph.verts C) (hzD : z ∈ Ω.graph.verts D) : z = u := by
      have h : z ∈ Ω.graph.verts C ∩ Ω.graph.verts D := ⟨hzC, hzD⟩
      rw [hv] at h
      exact h
    by_cases hCU : Disjoint C U <;> by_cases hDU : Disjoint D U
    · exact absurd (Set.disjoint_union_left.2 ⟨hCU, hDU⟩) hd
    · obtain ⟨f, hfD, hf⟩ := Set.not_disjoint_iff.1 hDU
      obtain ⟨S, hS, -, hDS, hSD⟩ := hU.exists_of_notMem_balanced hD hDB hfD hf
      refine hJD hf hC hCU hS (hU.vertex C ⟨hC, hCB⟩) (fun z hz hzC => hCD' z hzC (hSD z hz)) ?_
      rw [hDS]
      ext g
      simp only [Set.mem_union, Set.mem_insert_iff, Set.mem_singleton_iff]
      tauto
    · obtain ⟨e, heC, he⟩ := Set.not_disjoint_iff.1 hCU
      obtain ⟨R, hR, -, hCR, hRC⟩ := hU.exists_of_notMem_balanced hC hCB heC he
      refine hJD he hD hDU hR (hU.vertex D ⟨hD, hDB⟩) (fun z hz hzD => hCD' z (hRC z hz) hzD) ?_
      rw [hCR]
      ext g
      simp only [Set.mem_union, Set.mem_insert_iff, Set.mem_singleton_iff]
      tauto
    · obtain ⟨e, heC, he⟩ := Set.not_disjoint_iff.1 hCU
      obtain ⟨f, hfD, hf⟩ := Set.not_disjoint_iff.1 hDU
      obtain ⟨R, hR, -, hCR, hRC⟩ := hU.exists_of_notMem_balanced hC hCB heC he
      obtain ⟨S, hS, -, hDS, hSD⟩ := hU.exists_of_notMem_balanced hD hDB hfD hf
      have hRS : (R.append S.reverse).IsPath := hR.append hS.reverse fun z hz hz' =>
        hCD' z (hRC z hz) (hSD z (Walk.mem_support_reverse.1 hz'))
      have hX : C ∪ D = insert e (insert f (R.append S.reverse).edgeSet) := by
        rw [hCR, hDS]
        ext g
        simp only [Set.mem_union, Set.mem_insert_iff, Walk.edgeSet, Set.mem_ofPred_eq,
          Walk.mem_edges_append, Walk.mem_edges_reverse]
        tauto
      rw [hX]
      exact isFrameCircuit_rollUp_of_joints he hf
        (fun hef => Set.disjoint_left.1 hCD heC (hef ▸ hfD)) hRS
  · exact absurd (hU.vertex D ⟨hD, hno D (Set.subset_union_right.trans Set.subset_union_left) hD⟩)
      (Set.disjoint_left.1 hCD
        (hU.vertex C ⟨hC, hno C (Set.subset_union_left.trans Set.subset_union_left) hC⟩))
  · obtain ⟨e, heX, he⟩ := Set.not_disjoint_iff.1 hd
    -- The theta is `e`, a cycle off `U` and a path of the roll-up from `e` to the cycle.
    have hθ {P Q R : Ω.graph.Walk x y} (hT : IsTheta P Q R)
        (hXT : X = P.edgeSet ∪ Q.edgeSet ∪ R.edgeSet) (heP : e ∈ P.edges) :
        (Ω.rollUp u U).IsFrameCircuit X := by
      subst hXT
      obtain ⟨D, y', P₀, hD, hDU, hP₀, hyD, hP₀D, hX⟩ := hU.exists_of_isTheta hT hno he heP
      exact hJD he hD hDU hP₀ hyD hP₀D hX
    rw [hXT] at heX
    rcases heX with (heP | heQ) | heR
    · exact hθ hT hXT heP
    · refine hθ hT.swap₁₂ (hXT.trans ?_) heQ
      rw [Set.union_comm P.edgeSet]
    · refine hθ hT.swap₂₃.swap₁₂ (hXT.trans ?_) heR
      ext g
      simp only [Set.mem_union]
      tauto

end IsRollUpClass

end BiasedGraph

open BiasedGraph

example := pumpkin_isRollUpClass.isFrameCircuit_rollUp
  (Or.inr ⟨Or.inr (Or.inr ⟨0, 1, _, _, _, pumpkin_isTheta, rfl⟩), fun C _ _ hC =>
    Set.notMem_empty C hC⟩)

end Transformer.GainMatroid
