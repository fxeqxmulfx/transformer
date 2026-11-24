module

public import Transformer.GainMatroid.Section2_StarThetas

/-!
# Proposition 2.6: `Δ_X` of a biased graph

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, Proposition 2.6: if `X` is a balanced
triangle of a biased graph `(G, B)`, then `(Δ_X G, Δ_X B)` is a biased graph. The source leaves the
proof as "a straightforward check". We check it for any exchange pair `T`, `S` on
`X = {a, b, c}` (`IsExchange`), with the triangle `X` in `T` and the star `X` at `v` in `S`
(`IsExchange.exists_biasedGraph`); `G` and `Δ_X G` are one (`IsLabelledTriangle.isExchange`).

* The members of `Δ_X B` are cycles of `S` (`IsExchange.isCycle_of_mem_deltaBalanced`): a cycle of
  `B` off `X` or through two edges of `X` is a cycle of `S`, and so is `C ∆ X` for a cycle `C` of
  `B` through one edge of `X`.
* `Δ_X B` has the theta property (`IsExchange.theta_deltaBalanced`). A theta of `S` with an end at
  `v` is the case of `IsExchange.theta_of_eq`. A theta of `S` with ends other than `v` becomes a
  theta of `T` (`IsExchange.exists_isTransfer`, `IsExchange.openly_of_isTransfer`), whose cycles
  are in `B` exactly when those of the theta of `S` are in `Δ_X B`
  (`IsExchange.mem_deltaBalanced_iff_of_isTransfer`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open scoped symmDiff

namespace OrientedGraph.IsExchange

variable {V W E : Type*} {S : OrientedGraph W E} {ι : V → W} {v : W} {a b c : E} {x y z : V}

/-- §2.4, Proposition 2.6: the members of `Δ_X B` are cycles of `S`. -/
theorem isCycle_of_mem_deltaBalanced {Ω : BiasedGraph V E}
    (h : IsExchange Ω.graph S ι v a b c x y z) {D : Set E}
    (hD : D ∈ deltaBalanced Ω.balanced {a, b, c}) : S.IsCycle D := by
  have h3 : ({a, b, c} : Set E).ncard = 3 := Set.ncard_eq_three.2
    ⟨a, b, c, h.triangle.ne_ab, h.triangle.ne_ac, h.triangle.ne_bc, rfl⟩
  rcases hD with ⟨hDB, h02⟩ | ⟨C, hCB, h1, rfl⟩
  · have hC := Ω.isCycle_of_mem D hDB
    rcases h.inter_cases D with h0 | hXD | ⟨a', b', c', x', y', z', h', hX, hD1 | hD2⟩
    · exact (h.isCycle_iff_of_disjoint (Set.disjoint_iff_inter_eq_empty.2 h0)).1 hC
    · rw [Set.inter_eq_right.2 hXD, h3] at h02
      omega
    · rw [hD1, Set.ncard_singleton] at h02
      omega
    · rw [← hX] at hD2
      exact h'.isCycle_of_inter_eq_pair hC hD2
  · have hC := Ω.isCycle_of_mem C hCB
    rcases h.inter_cases C with h0 | hXC | ⟨a', b', c', x', y', z', h', hX, hC1 | hC2⟩
    · rw [h0, Set.ncard_empty] at h1
      omega
    · rw [Set.inter_eq_right.2 hXC, h3] at h1
      omega
    · rw [← hX] at hC1 ⊢
      exact h'.isCycle_symmDiff_of_inter_eq_singleton hC hC1
    · rw [hC2, Set.ncard_pair h'.triangle.ne_bc] at h1
      omega

/-- §2.4, Proposition 2.6: for `X` balanced in `Ω`, `Δ_X B` has the theta property on `S`. -/
theorem theta_deltaBalanced {Ω : BiasedGraph V E} (h : IsExchange Ω.graph S ι v a b c x y z)
    (hbal : ({a, b, c} : Set E) ∈ Ω.balanced) {s t : W} {P Q R : S.Walk s t}
    (hθ : IsTheta P Q R) (hPQ : P.edgeSet ∪ Q.edgeSet ∈ deltaBalanced Ω.balanced {a, b, c})
    (hPR : P.edgeSet ∪ R.edgeSet ∈ deltaBalanced Ω.balanced {a, b, c}) :
    Q.edgeSet ∪ R.edgeSet ∈ deltaBalanced Ω.balanced {a, b, c} := by
  by_cases hs : s = v
  · subst hs
    exact h.theta_of_eq hbal hθ hPQ hPR
  by_cases ht : t = v
  · subst ht
    have := h.theta_of_eq hbal hθ.reverse (by rwa [Walk.edgeSet_reverse, Walk.edgeSet_reverse])
      (by rwa [Walk.edgeSet_reverse, Walk.edgeSet_reverse])
    rwa [Walk.edgeSet_reverse, Walk.edgeSet_reverse] at this
  obtain ⟨s', rfl⟩ := h.exists_eq_of_walk P hθ.ne hs
  obtain ⟨t', rfl⟩ := h.exists_eq_of_walk P.reverse hθ.ne.symm ht
  obtain ⟨P', hP'⟩ := h.exists_isTransfer hθ.pathP hs ht
  obtain ⟨Q', hQ'⟩ := h.exists_isTransfer hθ.pathQ hs ht
  obtain ⟨R', hR'⟩ := h.exists_isTransfer hθ.pathR hs ht
  have hθ' : IsTheta P' Q' R' :=
    ⟨fun he => hθ.ne (congrArg ι he), hP'.isPath, hQ'.isPath, hR'.isPath,
      h.openly_of_isTransfer hs ht hθ.openPQ hP' hQ',
      h.openly_of_isTransfer hs ht hθ.openPR hP' hR',
      h.openly_of_isTransfer hs ht hθ.openQR hQ' hR'⟩
  have hC := hθ.isCycle
  rw [h.mem_deltaBalanced_iff_of_isTransfer hbal hs ht hθ.openPQ hC.1 hP' hQ'] at hPQ
  rw [h.mem_deltaBalanced_iff_of_isTransfer hbal hs ht hθ.openPR hC.2.1 hP' hR'] at hPR
  rw [h.mem_deltaBalanced_iff_of_isTransfer hbal hs ht hθ.openQR hC.2.2 hQ' hR']
  exact Ω.theta hθ' hPQ hPR

/-- §2.4, Proposition 2.6 for an exchange pair: if `X` is a balanced triangle of `Ω` on `T`, then
`(S, Δ_X B)` is a biased graph. -/
theorem exists_biasedGraph {Ω : BiasedGraph V E} (h : IsExchange Ω.graph S ι v a b c x y z)
    (hbal : ({a, b, c} : Set E) ∈ Ω.balanced) :
    ∃ Ω' : BiasedGraph W E, Ω'.graph = S ∧ Ω'.balanced = deltaBalanced Ω.balanced {a, b, c} :=
  ⟨⟨S, deltaBalanced Ω.balanced {a, b, c}, fun _ hD => h.isCycle_of_mem_deltaBalanced hD,
    fun hθ hPQ hPR => h.theta_deltaBalanced hbal hθ hPQ hPR⟩, rfl, rfl⟩

end OrientedGraph.IsExchange

namespace BiasedGraph

variable {V E : Type*}

/-- **Proposition 2.6** (§2.4). If `X` is a balanced triangle of `(G, B)`, then
`(Δ_X G, Δ_X B)` is a biased graph. -/
theorem exists_isDeltaExchange (Ω : BiasedGraph V E) {a b c : E} {x y z : V}
    (hX : Ω.graph.IsLabelledTriangle a b c x y z) (hb : ({a, b, c} : Set E) ∈ Ω.balanced) :
    ∃ Ω' : BiasedGraph (Option V) E, Ω.IsDeltaExchange a b c x y z Ω' := by
  obtain ⟨Ω', hg, hB⟩ := hX.isExchange.exists_biasedGraph hb
  exact ⟨Ω', hX, hb, hg, hB⟩

end BiasedGraph

open OrientedGraph

/-- The edge `0` of `clawGraph`, from the centre `0` to `1`. -/
def clawSpoke1 : clawGraph.Walk 0 1 := .cons 0 (Or.inl ⟨rfl, rfl⟩) (.nil 1)

/-- The path `0, 2, 1` of `clawGraph`. -/
def clawSpoke2 : clawGraph.Walk 0 1 :=
  .cons 1 (Or.inl ⟨rfl, rfl⟩) (.cons 3 (Or.inl ⟨rfl, rfl⟩) (.nil 1))

/-- The path `0, 3, 1` of `clawGraph`. -/
def clawSpoke3 : clawGraph.Walk 0 1 :=
  .cons 2 (Or.inl ⟨rfl, rfl⟩) (.cons 4 (Or.inr ⟨rfl, rfl⟩) (.nil 1))

/-- The theta of `clawGraph` between the centre `0` and `1`. -/
theorem claw_isTheta : IsTheta clawSpoke1 clawSpoke2 clawSpoke3 := by
  refine ⟨by decide, ?_, ?_, ?_, ⟨by decide, by decide⟩, ⟨by decide, by decide⟩,
    ⟨by decide, by decide⟩⟩ <;> (unfold Walk.IsPath; decide)

/-- The cycle `{0, 1, 3}` of the theta is in `Δ` of the balanced cycles of `clawNabla`: its
symmetric difference with `{0, 1, 2}` is the digon `{2, 3}`. -/
theorem claw_mem_deltaBalanced₁₂ : clawSpoke1.edgeSet ∪ clawSpoke2.edgeSet ∈
    deltaBalanced (clawNabla.gainBiased (1 : Fin 5 → ℚˣ)).balanced {0, 1, 2} := by
  rw [clawNabla_isExchange.mem_deltaBalanced_iff_of_isCycle (Ω := clawNabla.gainBiased 1)
    clawNabla_mem_balanced claw_isTheta.isCycle.1 (g := 0)
    (by simp [clawSpoke1, Walk.edgeSet, Walk.edges]) (by simp)]
  have : (clawSpoke1.edgeSet ∪ clawSpoke2.edgeSet) ∆ {0, 1, 2} =
      ((({2, 3} : Finset (Fin 5))) : Set (Fin 5)) := by
    ext g
    fin_cases g <;> simp [clawSpoke1, clawSpoke2, Walk.edgeSet, Walk.edges, Set.mem_symmDiff]
  rw [this]
  exact gainBiased_one_isBalanced clawNabla _ ((clawNabla_isCycle_iff _).2 (by decide))

/-- The cycle `{0, 2, 4}` of the theta is in `Δ` of the balanced cycles of `clawNabla`: its
symmetric difference with `{0, 1, 2}` is the digon `{1, 4}`. -/
theorem claw_mem_deltaBalanced₁₃ : clawSpoke1.edgeSet ∪ clawSpoke3.edgeSet ∈
    deltaBalanced (clawNabla.gainBiased (1 : Fin 5 → ℚˣ)).balanced {0, 1, 2} := by
  rw [clawNabla_isExchange.mem_deltaBalanced_iff_of_isCycle (Ω := clawNabla.gainBiased 1)
    clawNabla_mem_balanced claw_isTheta.isCycle.2.1 (g := 0)
    (by simp [clawSpoke1, Walk.edgeSet, Walk.edges]) (by simp)]
  have : (clawSpoke1.edgeSet ∪ clawSpoke3.edgeSet) ∆ {0, 1, 2} =
      ((({1, 4} : Finset (Fin 5))) : Set (Fin 5)) := by
    ext g
    fin_cases g <;> simp [clawSpoke1, clawSpoke3, Walk.edgeSet, Walk.edges, Set.mem_symmDiff]
  rw [this]
  exact gainBiased_one_isBalanced clawNabla _ ((clawNabla_isCycle_iff _).2 (by decide))

example := clawNabla_isExchange.theta_of_eq (Ω := clawNabla.gainBiased (1 : Fin 5 → ℚˣ))
  clawNabla_mem_balanced claw_isTheta claw_mem_deltaBalanced₁₂ claw_mem_deltaBalanced₁₃

example := clawNabla_isExchange.isCycle_of_mem_deltaBalanced
  (Ω := clawNabla.gainBiased (1 : Fin 5 → ℚˣ)) claw_mem_deltaBalanced₁₂

example := clawNabla_isExchange.theta_deltaBalanced (Ω := clawNabla.gainBiased (1 : Fin 5 → ℚˣ))
  clawNabla_mem_balanced claw_isTheta claw_mem_deltaBalanced₁₂ claw_mem_deltaBalanced₁₃

example := clawNabla_isExchange.exists_biasedGraph (Ω := clawNabla.gainBiased (1 : Fin 5 → ℚˣ))
  clawNabla_mem_balanced

example := BiasedGraph.exists_isDeltaExchange (triangle.gainBiased (1 : Fin 3 → ℚˣ))
  triangle_isLabelledTriangle triangle_mem_gainBiased

end Transformer.GainMatroid
