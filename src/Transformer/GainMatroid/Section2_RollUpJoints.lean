module

public import Transformer.GainMatroid.Section2_RollUpClass
public import Transformer.GainMatroid.Section2_TightHandcuffs

/-!
# Two joints of a roll-up

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5, compare the frame circuits of a biased graph
`Ω` and of its roll-up from `u` along a set `U` with the properties collected in `IsRollUpClass`.
In the roll-up each edge of `U` is a joint at its end away from `u`. Two such joints `e ≠ f`
joined by a path `P` of the roll-up, with no balanced cycle of the roll-up among them, unroll to a
frame circuit of `Ω` (`IsRollUpClass.isFrameCircuit_of_joints_rollUp`): a cycle through `e` and
`f`, balanced by `IsRollUpClass.pair`, if `P` avoids `u`, and a tight handcuff at `u` otherwise.
A balanced cycle of that handcuff would be `e`, `f` and a path of the roll-up off `u` between their
ends (`IsRollUpClass.exists_of_mem_balanced`), inside `P`, so through `u` as `P` is
(`Walk.IsPath.mem_support_of_subset`).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

open OrientedGraph

namespace BiasedGraph

variable {Ω : BiasedGraph V E} {u : V} {U : Set E}

namespace IsRollUpClass

/-- Two edges `e ≠ f` of `U`, joints of the roll-up, and a path of the roll-up between their ends
away from `u` form a frame circuit of `Ω` if they contain no balanced cycle of the roll-up: a
balanced cycle if the path avoids `u`, and a tight handcuff at `u` otherwise. -/
theorem isFrameCircuit_of_joints_rollUp (hU : Ω.IsRollUpClass u U) {e f : E} (he : e ∈ U)
    (hf : f ∈ U) (hef : e ≠ f) {a b : V} (ha : Ω.graph.otherEnd u e = a)
    (hb : Ω.graph.otherEnd u f = b) {P : (Ω.rollUp u U).graph.Walk a b} (hP : P.IsPath)
    (hno : ∀ C ⊆ {e} ∪ {f} ∪ P.edgeSet, (Ω.rollUp u U).graph.IsCycle C →
      C ∉ (Ω.rollUp u U).balanced) :
    Ω.IsFrameCircuit ({e} ∪ {f} ∪ P.edgeSet) := by
  subst ha hb
  obtain ⟨hl, heu⟩ := hU.link he
  obtain ⟨-, hfu⟩ := hU.link hf
  have hPU := disjoint_rollUp_of_isPath hP
  obtain ⟨Q, hQ, hQ₁, hQ₂⟩ := exists_isPath_of_rollUp hP
  have hQe : Q.edgeSet = P.edgeSet := by
    unfold Walk.edgeSet
    rw [hQ₁]
  by_cases huP : u ∈ P.support
  swap
  · have h := isCycle_pair_union (joins_otherEnd heu) (joins_otherEnd hfu) hef hQ
      (by rw [hQ₂]; exact huP)
    rw [hQe] at h
    exact Or.inl (hU.pair _ h e (Or.inl (Or.inl rfl)) f (Or.inl (Or.inr rfl)) hef he hf)
  refine Or.inr ⟨Or.inl ?_, fun C hCX hC hCB => ?_⟩
  · have h := isTightHandcuff_of_mem_support (joins_otherEnd heu).symm (joins_otherEnd hfu).symm
      hef hQ (by rw [hQ₂]; exact huP) (by rw [hQ₁]; exact fun h => Set.disjoint_left.1 hPU h he)
      (by rw [hQ₁]; exact fun h => Set.disjoint_left.1 hPU h hf)
    rw [hQe] at h
    rw [Set.singleton_union, Set.insert_union, Set.singleton_union]
    exact h
  by_cases hd : Disjoint C U
  · exact hno C hCX (hC.of_agreeOn (agreeOn_rollUp hd).symm) ⟨hCB, hd⟩
  have hCU : ∀ g ∈ C, g ∈ U → g = e ∨ g = f := fun g hg hgU => by
    rcases hCX hg with (h | h) | h
    · exact Or.inl h
    · exact Or.inr h
    · exact absurd hgU (Set.disjoint_left.1 hPU h)
  obtain ⟨g, hgC, hgU⟩ := Set.not_disjoint_iff.1 hd
  obtain ⟨g', hg'C, hg'U, hg'g, R, hR, huR, hCR⟩ := hU.exists_of_mem_balanced hCB hgC hgU
  have hRU := disjoint_rollUp_of_isPath hR
  have hRP : ∀ k ∈ R.edges, k ∈ P.edges := fun k hk => by
    have hkC : k ∈ C := by
      rw [hCR]
      exact Or.inr (Or.inr hk)
    rcases hCX hkC with (hke | hkf) | hkP
    · exact absurd (Set.mem_singleton_iff.1 hke ▸ he) (Set.disjoint_left.1 hRU hk)
    · exact absurd (Set.mem_singleton_iff.1 hkf ▸ hf) (Set.disjoint_left.1 hRU hk)
    · exact hkP
  rcases hCU g hgC hgU with rfl | rfl <;> rcases hCU g' hg'C hg'U with rfl | rfl
  · exact hg'g rfl
  · exact huR (hP.mem_support_of_subset hR hRP huP)
  · exact huR (Walk.mem_support_reverse.1 (hP.mem_support_of_subset hR.reverse
      (fun k hk => hRP k (Walk.mem_edges_reverse.1 hk)) huP))
  · exact hg'g rfl

end IsRollUpClass

end BiasedGraph

open BiasedGraph

example := (IsBalanced.isRollUpClass (gainBiased_one_isBalanced (Γ := ℚˣ) pumpkin) 0)
  |>.isFrameCircuit_of_joints_rollUp (e := 0) (f := 1) (a := 1) (b := 1)
    (pumpkin_mem_linkStar 0) (pumpkin_mem_linkStar 1) (by decide)
    (Joins.otherEnd_eq (G := pumpkin) (Or.inr ⟨rfl, rfl⟩))
    (Joins.otherEnd_eq (G := pumpkin) (Or.inr ⟨rfl, rfl⟩)) (P := .nil 1)
    (by unfold Walk.IsPath Walk.support; simp) fun C _ hC hb => by
      obtain ⟨a, W, hW, rfl⟩ := hC
      obtain ⟨g, hg⟩ := List.exists_mem_of_ne_nil _ hW.edges_ne_nil
      exact Set.disjoint_left.1 hb.2 hg (pumpkin_mem_linkStar g)

end Transformer.GainMatroid
