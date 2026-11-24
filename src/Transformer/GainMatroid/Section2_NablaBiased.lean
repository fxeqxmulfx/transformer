module

public import Transformer.GainMatroid.Section2_NablaCovers

/-!
# Proposition 2.7: `∇_Y` of a biased graph

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, Proposition 2.7: if `Y` is a
`K_{1,3}`-subgraph of a biased graph `(G, B)`, then `(∇_Y G, ∇_Y B)` is a biased graph. The source
leaves the proof as "a straightforward check". We check it for any exchange pair `T`, `S` on
`Y = {a, b, c}` (`IsExchange`), with the triangle `Y` in `T` and the star `Y` at `v` in `S`, and a
biased graph `Ω` on `S` (`IsExchange.exists_nablaBiasedGraph`); `∇_Y G` and `G` are one
(`IsLabelledClaw.isExchange`).

* The members of `∇_Y B` are cycles of `T` by definition (`nablaBalanced`).
* `∇_Y B` has the theta property (`IsExchange.theta_nablaBalanced`). If a cycle of the theta is `Y`,
  the other two are `D` and `D ∆ Y` for a cycle `D ≠ Y` (`IsTheta.symmDiff_union`), which `∇_Y B`
  contains together or not at all (`IsExchange.mem_nablaBalanced_iff_symmDiff`). Otherwise the
  theta matches a theta of `S` (`IsExchange.matchesTheta`): through two edges of `Y` at a corner
  (`IsExchange.matchesTheta_of_mem_of_mem`), or with edges of `Y` on one path at most
  (`IsExchange.matchesTheta_of_off`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open scoped symmDiff

namespace OrientedGraph

variable {V W E : Type*}

/-- In a theta, the symmetric difference of the cycle through `P` and `R` and the cycle through `P`
and `Q` is the cycle through `Q` and `R`. -/
theorem IsTheta.symmDiff_union {G : OrientedGraph V E} {s t : V} {P Q R : G.Walk s t}
    (hθ : IsTheta P Q R) :
    (P.edgeSet ∪ R.edgeSet) ∆ (P.edgeSet ∪ Q.edgeSet) = Q.edgeSet ∪ R.edgeSet := by
  ext g
  have h₁ := hθ.openPQ.edges g
  have h₂ := hθ.openPR.edges g
  have h₃ := hθ.openQR.edges g
  simp only [Walk.edgeSet, Set.mem_symmDiff, Set.mem_union, Set.mem_ofPred_eq]
  tauto

namespace IsExchange

variable {T : OrientedGraph V E} {ι : V → W} {v : W} {a b c : E} {x y z : V}
  {Ω : BiasedGraph W E}

/-- §2.4, Proposition 2.7: a theta of `T` none of whose cycles is `Y` matches a theta of `Ω` under
`∇_Y B`. -/
theorem matchesTheta (h : IsExchange T Ω.graph ι v a b c x y z) {s t : V} {P Q R : T.Walk s t}
    (hθ : IsTheta P Q R) (hPQ : P.edgeSet ∪ Q.edgeSet ≠ {a, b, c})
    (hPR : P.edgeSet ∪ R.edgeSet ≠ {a, b, c}) (hQR : Q.edgeSet ∪ R.edgeSet ≠ {a, b, c}) :
    MatchesTheta (nablaBalanced T Ω.balanced {a, b, c}) Ω P Q R := by
  by_cases hP : ∃ g ∈ P.edges, g ∈ ({a, b, c} : Set E)
  · obtain ⟨e₁, he₁, he₁Y⟩ := hP
    by_cases hQ : ∃ g ∈ Q.edges, g ∈ ({a, b, c} : Set E)
    · obtain ⟨e₂, he₂, he₂Y⟩ := hQ
      exact h.matchesTheta_of_mem_of_mem hθ he₁ he₁Y he₂ he₂Y hPQ
    by_cases hR : ∃ g ∈ R.edges, g ∈ ({a, b, c} : Set E)
    · obtain ⟨e₂, he₂, he₂Y⟩ := hR
      exact (h.matchesTheta_of_mem_of_mem hθ.swap₂₃ he₁ he₁Y he₂ he₂Y hPR).swap₂₃
    push Not at hQ hR
    exact h.matchesTheta_of_off hθ hQ hR
  push Not at hP
  by_cases hQ : ∃ g ∈ Q.edges, g ∈ ({a, b, c} : Set E)
  · obtain ⟨e₁, he₁, he₁Y⟩ := hQ
    by_cases hR : ∃ g ∈ R.edges, g ∈ ({a, b, c} : Set E)
    · obtain ⟨e₂, he₂, he₂Y⟩ := hR
      exact (h.matchesTheta_of_mem_of_mem hθ.swap₁₂.swap₂₃ he₁ he₁Y he₂ he₂Y hQR).swap₂₃.swap₁₂
    push Not at hR
    exact (h.matchesTheta_of_off hθ.swap₁₂ hP hR).swap₁₂
  push Not at hQ
  exact (h.matchesTheta_of_off hθ.swap₂₃.swap₁₂ hP hQ).swap₁₂.swap₂₃

/-- §2.4, Proposition 2.7: `∇_Y B` has the theta property on `T`. -/
theorem theta_nablaBalanced (h : IsExchange T Ω.graph ι v a b c x y z) {s t : V}
    {P Q R : T.Walk s t} (hθ : IsTheta P Q R)
    (hPQ : P.edgeSet ∪ Q.edgeSet ∈ nablaBalanced T Ω.balanced {a, b, c})
    (hPR : P.edgeSet ∪ R.edgeSet ∈ nablaBalanced T Ω.balanced {a, b, c}) :
    Q.edgeSet ∪ R.edgeSet ∈ nablaBalanced T Ω.balanced {a, b, c} := by
  have hC := hθ.isCycle
  by_cases hQR : Q.edgeSet ∪ R.edgeSet = {a, b, c}
  · rw [hQR]
    exact h.triangle.mem_nablaBalanced Ω.balanced
  by_cases hPQY : P.edgeSet ∪ Q.edgeSet = {a, b, c}
  · have e : (P.edgeSet ∪ R.edgeSet) ∆ {a, b, c} = Q.edgeSet ∪ R.edgeSet := by
      rw [← hPQY, hθ.symmDiff_union]
    have hPRY : P.edgeSet ∪ R.edgeSet ≠ {a, b, c} := fun hPRY => by
      have hQR' := hθ.symmDiff_union
      rw [hPRY, hPQY, symmDiff_self] at hQR'
      obtain ⟨g, hg⟩ := hC.2.2.nonempty
      rw [← hQR'] at hg
      simp at hg
    have hiff := h.mem_nablaBalanced_iff_symmDiff hC.2.1 (by rw [e]; exact hC.2.2) hPRY
    rw [e] at hiff
    exact hiff.1 hPR
  by_cases hPRY : P.edgeSet ∪ R.edgeSet = {a, b, c}
  · have e : (P.edgeSet ∪ Q.edgeSet) ∆ {a, b, c} = Q.edgeSet ∪ R.edgeSet := by
      rw [← hPRY, hθ.swap₂₃.symmDiff_union, Set.union_comm R.edgeSet]
    have hiff := h.mem_nablaBalanced_iff_symmDiff hC.1 (by rw [e]; exact hC.2.2) hPQY
    rw [e] at hiff
    exact hiff.1 hPQ
  exact (h.matchesTheta hθ hPQY hPRY hQR).theta hPQ hPR

/-- §2.4, Proposition 2.7 for an exchange pair: for a biased graph `Ω` on `S`, `(T, ∇_Y B)` is a
biased graph. -/
theorem exists_nablaBiasedGraph (h : IsExchange T Ω.graph ι v a b c x y z) :
    ∃ Ω' : BiasedGraph V E, Ω'.graph = T ∧
      Ω'.balanced = nablaBalanced T Ω.balanced {a, b, c} :=
  ⟨⟨T, nablaBalanced T Ω.balanced {a, b, c}, fun _ hD => hD.1,
    fun hθ hPQ hPR => h.theta_nablaBalanced hθ hPQ hPR⟩, rfl, rfl⟩

end IsExchange

end OrientedGraph

namespace BiasedGraph

variable {V E : Type*}

/-- **Proposition 2.7** (§2.4). If `Y` is a `K_{1,3}`-subgraph of `(G, B)`, then
`(∇_Y G, ∇_Y B)` is a biased graph. `∇_Y B` keeps the cycles of `∇_Y G` among the minimal members
of the source's family (`nablaBalanced`; `nablaFamily_minimal_not_isCycle` shows a minimal member
that is not a cycle). -/
theorem exists_isNablaExchange (Ω : BiasedGraph V E) {v : V} {a b c : E} {x y z : V}
    (hY : Ω.graph.IsLabelledClaw v a b c x y z) :
    ∃ Ω' : BiasedGraph V E, Ω.IsNablaExchange v a b c x y z Ω' := by
  obtain ⟨Ω', hg, hB⟩ := hY.isExchange.exists_nablaBiasedGraph
  exact ⟨Ω', hY, hg, hB⟩

end BiasedGraph

open OrientedGraph

example := BiasedGraph.exists_isNablaExchange clawGraph.contrabalanced clawGraph_isLabelledClaw

end Transformer.GainMatroid
