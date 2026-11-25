module

public import Transformer.GainMatroid.Section1_Fan
public import Transformer.GainMatroid.Section2_UnbalancingClasses

/-!
# A contrabalanced theta with a unique balancing vertex

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §3.2. The contrabalanced `fanGraph` of §1, a
triangle `x y u` with the links `xu` and `yu` doubled, meets the hypotheses of Proposition 3.5: it
is 2-connected, the two links `xu` and the path `x y u` form a contrabalanced theta
(`fanGraph_isTheta`), and `u = 2` lies on every cycle of at least two edges while each other
vertex misses a doubled link (`fan_existsUnique_isLinkBalancingVertex`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph BiasedGraph

/-- `fanGraph` is 2-connected: a complete graph on three vertices. -/
theorem fanGraph_isKConnected : fanGraph.IsKConnected 2 := by
  refine Or.inr ⟨by simp [Set.encard_univ]; norm_num, ?_⟩
  simp only [Joins, fanGraph]
  decide

/-- The links `1`, `2` and the path `0, 3` form a theta between `0` and `2` in `fanGraph`. -/
theorem fanGraph_isTheta :
    IsTheta (G := fanGraph) (.cons 1 (Or.inl ⟨rfl, rfl⟩) (.nil 2))
      (.cons 2 (Or.inl ⟨rfl, rfl⟩) (.nil 2))
      (.cons 0 (Or.inl ⟨rfl, rfl⟩) (.cons 3 (Or.inl ⟨rfl, rfl⟩) (.nil 2))) :=
  ⟨by decide, by unfold Walk.IsPath; decide, by unfold Walk.IsPath; decide,
    by unfold Walk.IsPath; decide, ⟨by decide, by decide⟩, ⟨by decide, by decide⟩,
    ⟨by decide, by decide⟩⟩

/-- In the contrabalanced `fanGraph`, every cycle of two or more edges passes through `2`, and `2`
is the only vertex with this property. -/
theorem fan_existsUnique_isLinkBalancingVertex :
    ∃! u, fanGraph.contrabalanced.IsLinkBalancingVertex u := by
  have hhead : ∀ e : Fin 5, e ≠ 0 → fanGraph.head e = 2 := by
    simp only [fanGraph]
    decide
  refine ⟨2, fun C _ h2 => ?_, fun v hv => ?_⟩
  · by_contra hv
    have hsub : C ⊆ {0} := fun e he => by
      by_contra hne
      exact hv (mem_verts.2 ⟨e, he, Or.inr (hhead e hne)⟩)
    have hle := Set.encard_le_encard hsub
    rw [Set.encard_singleton] at hle
    exact absurd (h2.trans hle) (by norm_num)
  · have h₁ := hv {1, 2} ⟨isCycle_pair (u := 0) (v := 2) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)
      (by decide) (by decide), Set.notMem_empty _⟩ (by rw [Set.encard_pair (by decide)])
    have h₂ := hv {3, 4} ⟨isCycle_pair (u := 1) (v := 2) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)
      (by decide) (by decide), Set.notMem_empty _⟩ (by rw [Set.encard_pair (by decide)])
    clear hv
    revert h₁ h₂
    simp only [contrabalanced, mem_verts, Set.mem_insert_iff, Set.mem_singleton_iff]
    revert v
    decide

end Transformer.GainMatroid
