module

public import Transformer.GainMatroid.Section2_ExchangeIndep
public import Transformer.GainMatroid.Section2_ExchangeWalks
public import Transformer.GainMatroid.Section2_NablaCycles

/-!
# Balanced cycles agree across `Δ_X` and `∇_Y`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. `Ω` and `Δ_X Ω`, and `∇_Y Ω` and `Ω`, are
biased graphs of exchange pairs (`BiasedGraph.IsDeltaExchange.isExchange`,
`BiasedGraph.IsNablaExchange.isExchange`), and their balanced cycles agree across the exchange
(`BiasedGraph.IsDeltaExchange.balanceAgree`, `BiasedGraph.IsNablaExchange.balanceAgree`). For
`Δ_X B` this is its definition, with the theta property at the balanced triangle `X` for a cycle
meeting `X` in two edges; for `∇_Y B` it is the case analysis of
`mem_nablaBalanced_iff_of_disjoint`, `mem_nablaBalanced_iff_of_singleton` and
`mem_nablaBalanced_iff_of_pair`. The triangle `Y` is
balanced in `∇_Y Ω` (`BiasedGraph.IsNablaExchange.mem_balanced`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph
open scoped symmDiff

namespace BiasedGraph

variable {V E : Type*} {Ω : BiasedGraph V E} {a b c : E} {x y z : V}

/-- `Ω` and `Δ_X Ω` are the biased graphs of an exchange pair. -/
theorem IsDeltaExchange.isExchange {Ω' : BiasedGraph (Option V) E}
    (h : Ω.IsDeltaExchange a b c x y z Ω') : IsExchange Ω.graph Ω'.graph some none a b c x y z := by
  rw [h.graph_eq]
  exact h.triangle.isExchange

/-- The balanced cycles of `Ω` and `Δ_X Ω` agree across the exchange. -/
theorem IsDeltaExchange.balanceAgree {Ω' : BiasedGraph (Option V) E}
    (h : Ω.IsDeltaExchange a b c x y z Ω') : Ω.BalanceAgree Ω' {a, b, c} := by
  have hex := h.isExchange
  refine ⟨fun D _ hDX => ?_, fun D e _ hDe => ?_, fun D e f hD hef hDX => ?_⟩
  · rw [h.balanced_eq, mem_deltaBalanced_iff_of_inter_eq_empty h.triangle.ne_ab hDX]
  · have he : e ∈ ({a, b, c} : Set E) := ((Set.ext_iff.1 hDe e).2 (Set.mem_singleton e)).2
    obtain ⟨b', c', x', y', z', h', hX⟩ : ∃ (b' c' : E) (x' y' z' : V),
        IsExchange Ω.graph Ω'.graph some none e b' c' x' y' z' ∧
          ({e, b', c'} : Set E) = {a, b, c} := by
      rcases he with rfl | rfl | rfl
      · exact ⟨_, _, _, _, _, hex, rfl⟩
      · exact ⟨_, _, _, _, _, hex.swap₁₂, Set.insert_comm _ _ _⟩
      · refine ⟨_, _, _, _, _, hex.swap₂₃.swap₁₂, ?_⟩
        ext g
        simp only [Set.mem_insert_iff, Set.mem_singleton_iff]
        tauto
    have hbal := h.balanced
    rw [← hX] at hDe hbal
    rw [h.balanced_eq, ← hX, mem_deltaBalanced_iff_of_inter_eq_pair h'.triangle hbal
      (symmDiff_inter_of_inter_eq_singleton h'.triangle.ne_ab h'.triangle.ne_ac hDe),
      symmDiff_symmDiff_cancel_right]
  · obtain ⟨e₀, x', y', z', h', hX⟩ := hex.exists_relabel
      ((Set.ext_iff.1 hDX e).2 (Set.mem_insert e _)).2
      ((Set.ext_iff.1 hDX f).2 (Set.mem_insert_of_mem e (Set.mem_singleton f))).2 hef
    have hbal := h.balanced
    rw [← hX] at hDX hbal
    rw [h.balanced_eq, ← hX, mem_deltaBalanced_iff_of_inter_eq_pair h'.triangle hbal hDX]
    exact mem_balanced_iff_symmDiff h'.triangle hbal hD hDX

variable {v : V} {Ω' : BiasedGraph V E}

/-- `∇_Y Ω` and `Ω` are the biased graphs of an exchange pair. -/
theorem IsNablaExchange.isExchange (h : Ω.IsNablaExchange v a b c x y z Ω') :
    IsExchange Ω'.graph Ω.graph id v a b c x y z := by
  rw [h.graph_eq]
  exact h.claw.isExchange

theorem IsNablaExchange.balanced_eq_nablaBalanced (h : Ω.IsNablaExchange v a b c x y z Ω') :
    Ω'.balanced = nablaBalanced Ω'.graph Ω.balanced {a, b, c} := by
  rw [h.balanced_eq, h.graph_eq]

/-- The triangle `Y` is balanced in `∇_Y Ω`. -/
theorem IsNablaExchange.mem_balanced (h : Ω.IsNablaExchange v a b c x y z Ω') :
    ({a, b, c} : Set E) ∈ Ω'.balanced := by
  rw [h.balanced_eq_nablaBalanced]
  exact ⟨h.isExchange.triangle.isCycle, Or.inl rfl⟩

/-- The balanced cycles of `∇_Y Ω` and `Ω` agree across the exchange. -/
theorem IsNablaExchange.balanceAgree (h : Ω.IsNablaExchange v a b c x y z Ω') :
    Ω'.BalanceAgree Ω {a, b, c} := by
  have hex := h.isExchange
  refine ⟨fun D hD hDX => ?_, fun D e hD hDe => ?_, fun D e f hD hef hDX => ?_⟩ <;>
    rw [h.balanced_eq_nablaBalanced]
  · exact hex.mem_nablaBalanced_iff_of_disjoint hD hDX
  · exact hex.mem_nablaBalanced_iff_of_singleton hD
      ((Set.ext_iff.1 hDe e).2 (Set.mem_singleton e)).2 hDe
  · exact hex.mem_nablaBalanced_iff_of_pair hD ((Set.ext_iff.1 hDX e).2 (Set.mem_insert e _)).2
      ((Set.ext_iff.1 hDX f).2 (Set.mem_insert_of_mem e (Set.mem_singleton f))).2 hef hDX

end BiasedGraph

end Transformer.GainMatroid
