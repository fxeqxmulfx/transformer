module

public import Transformer.GainMatroid.Section2_ExchangeBalance
public import Transformer.GainMatroid.Section2_ExchangeSwitching

/-!
# Gain functions on an exchange pair

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, Proposition 2.10 (i) for an exchange pair
`T`, `S` on `X = {a, b, c}` (`IsExchange`). Let `Ω` be a biased graph on `T` in which `X` is
balanced, and `Ω'` one on `S` whose balanced cycles are `Δ_X B`. A gain function with value one on
`X` realizes `Ω` exactly when it realizes `Ω'` (`IsExchange.realizes_iff`), and an additive one with
value zero on `X` likewise (`IsExchange.addRealizes_iff`). `G` and `Δ_X G` are such a pair
(`BiasedGraph.IsDeltaExchange.isExchange`).

The source's proof compares the balanced cycles of both sides "assigning identity gains to each edge
in `X`". Up to the labels of the triangle (`IsExchange.inter_cases`,
`IsExchange.inter_cases_of_isCycle`), a cycle of `T` or `S` misses `X`, is `X`, or meets `X` in
`a` alone in `T`, or in `b` and `c` alone. A cycle off `X` is balanced in both graphs or in neither
(`IsExchange.mem_balancedCycles_iff_of_disjoint`), and is in `Δ_X B` exactly when it is in `B`
(`mem_deltaBalanced_iff_of_inter_eq_empty`). A cycle `D` through `b` and `c` is balanced in `S`
exactly when `D ∆ X` is balanced in `T` (`IsExchange.mem_balancedCycles_iff_symmDiff`), and is in
`Δ_X B` exactly when `D ∆ X` is in `B` (`BiasedGraph.mem_deltaBalanced_iff_of_inter_eq_pair`). In
`T`, it is balanced exactly when `D ∆ X` is, both for `B` and for the gains, as `X` is balanced for
both (`BiasedGraph.mem_balanced_iff_symmDiff`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open scoped symmDiff

namespace OrientedGraph.IsExchange

variable {V W E Γ : Type*} [Group Γ] [DecidableEq V] [DecidableEq W] {ι : V → W} {v : W}
  {a b c : E} {x y z : V}

/-- §2.4, Proposition 2.10 (i) for an exchange pair, with the source proof's identity gains on `X`:
if `X` is balanced in `Ω` and `Ω'` has the balanced cycles `Δ_X B`, a gain function with value one
on `X` realizes `Ω` exactly when it realizes `Ω'`. -/
@[to_additive addRealizes_iff /-- §2.4, Proposition 2.10 (i) for an exchange pair, additive: if `X`
is balanced in `Ω` and `Ω'` has the balanced cycles `Δ_X B`, an additive gain function with value
zero on `X` realizes `Ω` exactly when it realizes `Ω'`. -/]
theorem realizes_iff {Ω : BiasedGraph V E} {Ω' : BiasedGraph W E}
    (h : IsExchange Ω.graph Ω'.graph ι v a b c x y z) (hbal : ({a, b, c} : Set E) ∈ Ω.balanced)
    (hΩ' : Ω'.balanced = deltaBalanced Ω.balanced {a, b, c}) {γ : E → Γ}
    (hγ : ∀ e ∈ ({a, b, c} : Set E), γ e = 1) : Ω.Realizes γ ↔ Ω'.Realizes γ := by
  have hone : ∀ {a' b' c' : E} {x' y' z' : V}, Ω.graph.IsLabelledTriangle a' b' c' x' y' z' →
      (∀ e ∈ ({a', b', c'} : Set E), γ e = 1) →
        ({a', b', c'} : Set E) ∈ Ω.graph.balancedCycles γ := fun hT hγ' =>
    mem_balancedCycles_iff.2 ⟨hT.isCycle, 1, fun e he => by
      rw [switch_one]
      exact hγ' e he⟩
  unfold BiasedGraph.Realizes
  constructor
  · intro hR
    ext D
    by_cases hD : Ω'.graph.IsCycle D
    · rw [hΩ']
      rcases h.inter_cases_of_isCycle hD with h0 | ⟨a', b', c', x', y', z', h', hX, h2⟩
      · rw [← h.mem_balancedCycles_iff_of_disjoint γ (Set.disjoint_iff_inter_eq_empty.2 h0), hR,
          mem_deltaBalanced_iff_of_inter_eq_empty h.triangle.ne_ab h0]
      · rw [← hX] at hγ hbal h2 ⊢
        rw [h'.mem_balancedCycles_iff_symmDiff hγ hD h2, hR,
          Ω.mem_deltaBalanced_iff_of_inter_eq_pair h'.triangle hbal h2]
    · exact iff_of_false (fun hD' => hD (mem_balancedCycles_iff.1 hD').1)
        fun hD' => hD (Ω'.isCycle_of_mem D hD')
  · intro hR'
    ext C
    by_cases hC : Ω.graph.IsCycle C
    · rcases h.inter_cases C with h0 | hXC | ⟨a', b', c', x', y', z', h', hX, h1 | h2⟩
      · rw [h.mem_balancedCycles_iff_of_disjoint γ (Set.disjoint_iff_inter_eq_empty.2 h0), hR',
          hΩ', mem_deltaBalanced_iff_of_inter_eq_empty h.triangle.ne_ab h0]
      · rw [h.triangle.eq_of_subset hC hXC]
        exact iff_of_true (hone h.triangle hγ) hbal
      · rw [← hX] at hγ hbal hΩ' h1
        have hDX := symmDiff_inter_of_inter_eq_singleton h'.triangle.ne_ab h'.triangle.ne_ac h1
        have hD := h'.mem_balancedCycles_iff_symmDiff hγ
          (h'.isCycle_symmDiff_of_inter_eq_singleton hC h1) hDX
        rw [symmDiff_symmDiff_cancel_right, hR', hΩ',
          Ω.mem_deltaBalanced_iff_of_inter_eq_pair h'.triangle hbal hDX,
          symmDiff_symmDiff_cancel_right] at hD
        exact hD.symm
      · rw [← hX] at hγ hbal hΩ' h2
        have hΘ : C ∈ Ω.graph.balancedCycles γ ↔ C ∆ {a', b', c'} ∈ Ω.graph.balancedCycles γ :=
          (Ω.graph.gainBiased γ).mem_balanced_iff_symmDiff h'.triangle (hone h'.triangle hγ) hC h2
        rw [hΘ, ← h'.mem_balancedCycles_iff_symmDiff hγ (h'.isCycle_of_inter_eq_pair hC h2) h2,
          hR', hΩ', Ω.mem_deltaBalanced_iff_of_inter_eq_pair h'.triangle hbal h2,
          ← Ω.mem_balanced_iff_symmDiff h'.triangle hbal hC h2]
    · exact iff_of_false (fun hC' => hC (mem_balancedCycles_iff.1 hC').1)
        fun hC' => hC (Ω.isCycle_of_mem C hC')

end OrientedGraph.IsExchange

open OrientedGraph

/-- `Ω` and `Δ_X Ω` live on an exchange pair (§2.4). -/
theorem BiasedGraph.IsDeltaExchange.isExchange {V E : Type*} {Ω : BiasedGraph V E} {a b c : E}
    {x y z : V} {Ω' : BiasedGraph (Option V) E} (h : Ω.IsDeltaExchange a b c x y z Ω') :
    IsExchange Ω.graph Ω'.graph some none a b c x y z :=
  h.graph_eq ▸ h.triangle.isExchange

example := triangle_isDeltaExchange.isExchange

example := triangle_isDeltaExchange.isExchange.realizes_iff triangle_isDeltaExchange.balanced
  triangle_isDeltaExchange.balanced_eq (γ := (1 : Fin 3 → ℚˣ)) fun _ _ => rfl

example := triangle_isDeltaExchange.isExchange.addRealizes_iff triangle_isDeltaExchange.balanced
  triangle_isDeltaExchange.balanced_eq (γ := (0 : Fin 3 → ℚ)) fun _ _ => rfl

end Transformer.GainMatroid
