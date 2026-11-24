module

public import Transformer.GainMatroid.Section2_NablaCorners
public import Transformer.GainMatroid.Section2_TransferCycles

/-!
# Thetas of `∇_Y G` with one path through `Y`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, the proof of Proposition 2.7. Let `T`, `S` be
an exchange pair on `Y = {a, b, c}` (`IsExchange`), with the triangle `Y` in `T` and the star `Y` at
`v` in `S`, and `Ω` a biased graph on `S`. A path of `T` that does not contain `Y` has a detour
through `v` that keeps its cycles: with edges off `Y` it closes a cycle in `∇_Y B` exactly when the
detour closes one in `B` with them (`IsExchange.exists_isDetour`). So a theta of `T` whose other two
paths miss `Y` matches the theta of `S` made of the detour and the other two paths
(`IsExchange.matchesTheta_of_off`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open scoped symmDiff

/-- `(P ∪ D) ∩ Y = P ∩ Y` for `D` off `Y`. -/
theorem union_inter_of_disjoint {E : Type*} {P D Y : Set E} (hD : Disjoint D Y) :
    (P ∪ D) ∩ Y = P ∩ Y := by
  rw [Set.union_inter_distrib_right, Set.disjoint_iff_inter_eq_empty.1 hD, Set.union_empty]

namespace OrientedGraph.IsExchange

variable {V W E : Type*} {T : OrientedGraph V E} {ι : V → W} {v : W} {a b c : E} {x y z : V}
  {Ω : BiasedGraph W E}

/-- §2.4: a path of `T` that does not contain `Y` has a detour through `v` that keeps its cycles:
with a set `D` of edges off `Y`, it makes a cycle in `∇_Y B` exactly when the detour makes one in
`B`. -/
theorem exists_isDetour (h : IsExchange T Ω.graph ι v a b c x y z) {s t : V} {P : T.Walk s t}
    (hP : P.IsPath) (hPY : ¬ {a, b, c} ⊆ P.edgeSet) :
    ∃ P' : Ω.graph.Walk (ι s) (ι t), IsDetour {a, b, c} ι v P P' ∧
      ∀ D : Set E, Disjoint D {a, b, c} → T.IsCycle (P.edgeSet ∪ D) →
        (P.edgeSet ∪ D ∈ nablaBalanced T Ω.balanced {a, b, c} ↔ P'.edgeSet ∪ D ∈ Ω.balanced) := by
  rcases h.inter_cases P.edgeSet with h0 | hsub | ⟨a', b', c', x', y', z', h', hX, h1 | h2⟩
  · obtain ⟨P', hP', hP'e, -⟩ := h.exists_isDetour_of_off hP fun g hg hgY =>
      (Set.ext_iff.1 h0 g).1 ⟨hg, hgY⟩
    have hP'P : P'.edgeSet = P.edgeSet := by
      unfold Walk.edgeSet
      rw [hP'e]
    refine ⟨P', hP', fun D hD hC => ?_⟩
    rw [hP'P]
    exact h.mem_nablaBalanced_iff_of_disjoint hC (by rw [union_inter_of_disjoint hD, h0])
  · exact absurd hsub hPY
  · rw [← hX] at h1 ⊢
    obtain ⟨P', hP', hP'e⟩ := h'.exists_isDetour_of_inter_eq_singleton hP h1
    refine ⟨P', hP', fun D hD hC => ?_⟩
    rw [h'.mem_nablaBalanced_iff_of_singleton hC (e := a') (by simp)
      (by rw [union_inter_of_disjoint hD, h1]), hP'e, union_symmDiff_of_disjoint hD]
  · rw [← hX] at h2 ⊢
    obtain ⟨P', hP', hP'e⟩ := h'.exists_isDetour_of_inter_eq_pair hP h2
    refine ⟨P', hP', fun D hD hC => ?_⟩
    rw [h'.mem_nablaBalanced_iff_of_pair hC (by simp) (by simp) h'.triangle.ne_bc
      (by rw [union_inter_of_disjoint hD, h2]), hP'e]

/-- §2.4: a theta of `T` whose last two paths miss `Y` matches a theta of `Ω` under `∇_Y B`. -/
theorem matchesTheta_of_off (h : IsExchange T Ω.graph ι v a b c x y z) {s t : V}
    {P Q R : T.Walk s t} (hθ : IsTheta P Q R) (hQ : ∀ g ∈ Q.edges, g ∉ ({a, b, c} : Set E))
    (hR : ∀ g ∈ R.edges, g ∉ ({a, b, c} : Set E)) :
    MatchesTheta (nablaBalanced T Ω.balanced {a, b, c}) Ω P Q R := by
  have hC := hθ.isCycle
  have hQY : Disjoint Q.edgeSet {a, b, c} := Set.disjoint_left.2 fun g hg => hQ g hg
  have hRY : Disjoint R.edgeSet {a, b, c} := Set.disjoint_left.2 fun g hg => hR g hg
  obtain ⟨_, e, _, _, hQe⟩ := Q.exists_eq_cons_of_ne hθ.ne
  have heQ : e ∈ Q.edges := by
    rw [hQe]
    exact List.mem_cons_self
  obtain ⟨P', hP', hcyc⟩ := h.exists_isDetour hθ.pathP fun hsub => hQ e heQ (by
    rw [← h.triangle.eq_of_subset hC.1 (hsub.trans Set.subset_union_left)]
    exact Or.inr heQ)
  obtain ⟨Q', R', hθ', hQ', hR'⟩ := h.exists_isTheta_of_isDetour hθ hQ hR hP'
  refine ⟨_, _, P', Q', R', hθ', ?_, ?_, ?_⟩
  · rw [hQ']
    exact hcyc _ hQY hC.1
  · rw [hR']
    exact hcyc _ hRY hC.2.1
  · rw [hQ', hR']
    exact h.mem_nablaBalanced_iff_of_disjoint hC.2.2
      (Set.disjoint_iff_inter_eq_empty.1 (Set.disjoint_union_left.2 ⟨hQY, hRY⟩))

end OrientedGraph.IsExchange

end Transformer.GainMatroid
