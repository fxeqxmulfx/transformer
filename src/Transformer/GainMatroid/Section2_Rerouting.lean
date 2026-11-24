module

public import Transformer.GainMatroid.Section2_CyclePaths

/-!
# Rerouting a cycle through thetas

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5. Let every cycle that misses a vertex `u` be
balanced, and let `P` and `Q` be paths between the same ends that miss `u`, each openly disjoint
from a path `S`. If `Q` starts with an edge `e₁`, then `P` can be replaced by a path that starts
with `e₁` and closes a cycle of the same bias with `S` (`BiasedGraph.exists_reroute`): the old and
the new cycle are two cycles of a theta whose third cycle misses `u` and is balanced. In
`rerouteBiased`, two parallel links and a path through a third vertex, every cycle is balanced
(`rerouteBiased_balanced`).
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE}

open OrientedGraph OrientedGraph.Walk

namespace BiasedGraph

variable {Ω : BiasedGraph V E}

/-- One step of rerouting: if `Q` starts with `e₁`, the path `P` with the same ends can be replaced
by a path starting with `e₁` whose cycle with `S` has the same bias. If `e₁` is not on `P`, the
part of `Q` up to its first vertex `w` on `P` and the two parts of `P` cut at `w` form a theta with
`S`; its cycle on `Q` and `P` misses `u`. -/
theorem exists_reroute {u x q y : V} {e₁ : E} {h₁ : Ω.graph.Joins e₁ x q}
    {Q : Ω.graph.Walk q y} {P S : Ω.graph.Walk x y}
    (hbal : ∀ {a b : V} (A B : Ω.graph.Walk a b), a ≠ b → A.IsPath → B.IsPath → Openly A B →
      u ∉ A.support → u ∉ B.support → A.edgeSet ∪ B.edgeSet ∈ Ω.balanced)
    (hP : P.IsPath) (hQ : (cons e₁ h₁ Q).IsPath) (hS : S.IsPath) (hPS : Openly P S)
    (hQS : Openly (cons e₁ h₁ Q) S) (huP : u ∉ P.support) (huQ : u ∉ (cons e₁ h₁ Q).support) :
    ∃ P' : Ω.graph.Walk q y, (cons e₁ h₁ P').IsPath ∧ u ∉ (cons e₁ h₁ P').support ∧
      Openly (cons e₁ h₁ P') S ∧
      (P.edgeSet ∪ S.edgeSet ∈ Ω.balanced ↔ (cons e₁ h₁ P').edgeSet ∪ S.edgeSet ∈ Ω.balanced) := by
  by_cases he₁ : e₁ ∈ P.edges
  · cases P with
    | nil => simp [edges] at he₁
    | cons g hg P₁ =>
      obtain rfl := hP.eq_of_mem_start he₁ h₁.tail_or_head
      obtain rfl := hg.eq_of_joins h₁
      exact ⟨P₁, hP, huP, hPS, Iff.rfl⟩
  obtain ⟨w, A, B', hwP, rfl, hA⟩ :=
    Q.exists_eq_append_first (S := {z | z ∈ P.support}) P.end_mem_support
  obtain ⟨B, C, rfl⟩ := P.exists_eq_append_of_mem_support hwP
  have hAP : ∀ z ∈ A.support, z ∈ (B.append C).support → z = w := hA
  change ((cons e₁ h₁ A).append B').IsPath at hQ
  change Openly ((cons e₁ h₁ A).append B') S at hQS
  change u ∉ ((cons e₁ h₁ A).append B').support at huQ
  have hAp : (cons e₁ h₁ A).IsPath := hQ.of_append_left
  have hxA : x ∉ A.support := (isPath_cons.1 hAp).1
  have hwx : x ≠ w := by
    rintro rfl
    exact hxA A.end_mem_support
  have hxC : x ∉ C.support := fun h => hwx (hP.eq_of_mem_of_mem B.start_mem_support h)
  have hAe : ∀ g ∈ A.edges, g ∉ (B.append C).edges := fun g hgA hgP =>
    hP.isLink_of_mem hgP
      ((hAP _ (A.ends_mem_support hgA).1 ((B.append C).ends_mem_support hgP).1).trans
        (hAP _ (A.ends_mem_support hgA).2 ((B.append C).ends_mem_support hgP).2).symm)
  have hθ : IsTheta (cons e₁ h₁ A) B (S.append C.reverse) := by
    refine ⟨hwx, hAp, hP.of_append_left, hS.append hP.of_append_right.reverse fun z hzS hzC => ?_,
      ⟨fun g hg hgB => ?_, fun z hz hzB => ?_⟩, ⟨fun g hg hgT => ?_, fun z hz hzT => ?_⟩,
      ⟨fun g hg hgT => ?_, fun z hz hzT => ?_⟩⟩
    · rw [mem_support_reverse] at hzC
      rcases hPS.verts z (mem_support_append.2 (Or.inr hzC)) hzS with rfl | h
      · exact (hxC hzC).elim
      · exact h
    · rcases List.mem_cons.1 hg with rfl | hg
      · exact he₁ (mem_edges_append.2 (Or.inl hgB))
      · exact hAe g hg (mem_edges_append.2 (Or.inl hgB))
    · rcases List.mem_cons.1 hz with rfl | hz
      · exact Or.inl rfl
      · exact Or.inr (hAP z hz (mem_support_append.2 (Or.inl hzB)))
    · rcases mem_edges_append.1 hgT with hgS | hgC
      · exact hQS.edges g (mem_edges_append.2 (Or.inl hg)) hgS
      · rw [mem_edges_reverse] at hgC
        rcases List.mem_cons.1 hg with rfl | hg
        · exact he₁ (mem_edges_append.2 (Or.inr hgC))
        · exact hAe g hg (mem_edges_append.2 (Or.inr hgC))
    · rcases List.mem_cons.1 hz with rfl | hzA
      · exact Or.inl rfl
      right
      rcases mem_support_append.1 hzT with hzS | hzC
      · rcases hQS.verts z (mem_support_append.2 (Or.inl (List.mem_cons_of_mem _ hzA))) hzS with
          rfl | rfl
        · exact (hxA hzA).elim
        · exact hQ.eq_of_mem_of_mem (List.mem_cons_of_mem _ hzA) B'.end_mem_support
      · rw [mem_support_reverse] at hzC
        exact hAP z hzA (mem_support_append.2 (Or.inr hzC))
    · rcases mem_edges_append.1 hgT with hgS | hgC
      · exact hPS.edges g (mem_edges_append.2 (Or.inl hg)) hgS
      · rw [mem_edges_reverse] at hgC
        exact hP.not_mem_of_mem hg hgC
    · rcases mem_support_append.1 hzT with hzS | hzC
      · rcases hPS.verts z (mem_support_append.2 (Or.inl hz)) hzS with h | h
        · exact Or.inl h
        · subst h
          exact Or.inr (hP.eq_of_mem_of_mem hz C.end_mem_support)
      · rw [mem_support_reverse] at hzC
        exact Or.inr (hP.eq_of_mem_of_mem hz hzC)
  have hAB : (cons e₁ h₁ A).edgeSet ∪ B.edgeSet ∈ Ω.balanced :=
    hbal (cons e₁ h₁ A) B hwx hAp hθ.pathQ hθ.openPQ
      (fun h => huQ (mem_support_append.2 (Or.inl h)))
      (fun h => huP (mem_support_append.2 (Or.inl h)))
  refine ⟨A.append C, ?_, ?_, ?_, ?_⟩
  · change ((cons e₁ h₁ A).append C).IsPath
    refine hAp.append hP.of_append_right fun z hzA hzC => ?_
    rcases List.mem_cons.1 hzA with rfl | hzA
    · exact (hxC hzC).elim
    · exact hAP z hzA (mem_support_append.2 (Or.inr hzC))
  · change u ∉ ((cons e₁ h₁ A).append C).support
    intro h
    rcases mem_support_append.1 h with h | h
    · exact huQ (mem_support_append.2 (Or.inl h))
    · exact huP (mem_support_append.2 (Or.inr h))
  · change Openly ((cons e₁ h₁ A).append C) S
    refine ⟨fun g hg hgS => ?_, fun z hz hzS => ?_⟩
    · rcases mem_edges_append.1 hg with hg | hg
      · exact hQS.edges g (mem_edges_append.2 (Or.inl hg)) hgS
      · exact hPS.edges g (mem_edges_append.2 (Or.inr hg)) hgS
    · rcases mem_support_append.1 hz with hz | hz
      · exact hQS.verts z (mem_support_append.2 (Or.inl hz)) hzS
      · exact hPS.verts z (mem_support_append.2 (Or.inr hz)) hzS
  · have hT : (S.append C.reverse).edgeSet = S.edgeSet ∪ C.edgeSet := by
      rw [edgeSet_append, edgeSet_reverse]
    have hold : (B.append C).edgeSet ∪ S.edgeSet = B.edgeSet ∪ (S.append C.reverse).edgeSet := by
      rw [hT, edgeSet_append]
      ext g
      simp only [Set.mem_union]
      tauto
    have hnew : (cons e₁ h₁ (A.append C)).edgeSet ∪ S.edgeSet =
        (cons e₁ h₁ A).edgeSet ∪ (S.append C.reverse).edgeSet := by
      rw [hT, edgeSet_cons, edgeSet_cons, edgeSet_append]
      ext g
      simp only [Set.mem_union, Set.mem_insert_iff]
      tauto
    rw [hold, hnew]
    exact ⟨fun h => Ω.theta hθ.swap₁₂ (by rwa [Set.union_comm]) h, fun h => Ω.theta hθ hAB h⟩

end BiasedGraph

/-- Two parallel links `0 → 1` and the path `0 → 2 → 1`, for the examples. -/
def rerouteGraph : OrientedGraph (Fin 3) (Fin 4) := ⟨![0, 0, 0, 2], ![1, 1, 2, 1]⟩

/-- `rerouteGraph` with trivial gains, so that every cycle is balanced. -/
def rerouteBiased : BiasedGraph (Fin 3) (Fin 4) := rerouteGraph.gainBiased (1 : Fin 4 → ℚˣ)

/-- In `rerouteBiased` two openly disjoint paths form a balanced cycle. -/
theorem rerouteBiased_balanced {a b : Fin 3} (A B : rerouteBiased.graph.Walk a b) (hab : a ≠ b)
    (hA : A.IsPath) (hB : B.IsPath) (hAB : Openly A B) :
    A.edgeSet ∪ B.edgeSet ∈ rerouteBiased.balanced :=
  mem_balancedCycles_iff.2 ⟨⟨a, _, hAB.isCycleWalk hab hA hB, edgeSet_append_reverse A B⟩, 1,
    fun _ _ => by simp [switch]⟩

/-- The link `0` from `0` to `1`. -/
def rerouteP : rerouteBiased.graph.Walk 0 1 := .cons 0 (Or.inl ⟨rfl, rfl⟩) (.nil 1)

/-- The link `1` from `0` to `1`. -/
def rerouteQ : rerouteBiased.graph.Walk 0 1 := .cons 1 (Or.inl ⟨rfl, rfl⟩) (.nil 1)

/-- The path `0 → 2 → 1`. -/
def rerouteS : rerouteBiased.graph.Walk 0 1 :=
  .cons 2 (Or.inl ⟨rfl, rfl⟩) (.cons 3 (Or.inl ⟨rfl, rfl⟩) (.nil 1))

theorem rerouteP_isPath : rerouteP.IsPath := by
  unfold Walk.IsPath
  decide

theorem rerouteQ_isPath : rerouteQ.IsPath := by
  unfold Walk.IsPath
  decide

theorem rerouteS_isPath : rerouteS.IsPath := by
  unfold Walk.IsPath
  decide

theorem rerouteP_openly : Openly rerouteP rerouteS := ⟨by decide, by decide⟩

theorem rerouteQ_openly : Openly rerouteQ rerouteS := ⟨by decide, by decide⟩

example := rerouteBiased_balanced rerouteP rerouteQ (by decide) rerouteP_isPath rerouteQ_isPath
  ⟨by decide, by decide⟩

example := BiasedGraph.exists_reroute (Ω := rerouteBiased) (u := 2) (x := 0) (q := 1) (y := 1)
  (e₁ := 1) (h₁ := Or.inl ⟨rfl, rfl⟩) (Q := .nil 1) (P := rerouteP) (S := rerouteS)
  (fun A B hab hA hB hAB _ _ => rerouteBiased_balanced A B hab hA hB hAB) rerouteP_isPath
  rerouteQ_isPath rerouteS_isPath rerouteP_openly rerouteQ_openly (by decide) (by decide)

end Transformer.GainMatroid
