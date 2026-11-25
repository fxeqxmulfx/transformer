module

public import Transformer.GainMatroid.Section3_Examples
public import Transformer.GainMatroid.Section3_FanTheta

/-!
# Unavoidable minors and subdivisions

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §1.2 and §3.1–§3.2. Every 2-connected properly
unbalanced biased graph has a member of `𝒢₀` as a link minor (Theorem 4, `hasG0LinkMinor`) and a
subdivision of a member of `𝒯₀` as a biased subgraph (Theorem 5, `hasT0Subdivision`). The source
proves these from Proposition 3.2 (`hasSubdivision_tube`, two vertex-disjoint unbalanced cycles),
Lemma 3.4 (`IsTangled.exists_twoC3_linkMinor`) and Theorem 3.3
(`IsTangled.hasLinkMinor_twoC3_or_K4`, the tangled case). Proposition 3.5
(`hasSubdivision_D10_or_BPrime`) is the analogue for a contrabalanced theta with a unique balancing
vertex.

The source calls `𝒢₀` "the set of biased graphs that are minor-minimal amongst all 2-connected,
properly unbalanced biased graphs". Its members are 2-connected and properly unbalanced
(`IsG0TwoC3.isKConnected`, …), every 2-connected properly unbalanced biased graph has at least the
six edges of a member (`six_le_card_of_isProperlyUnbalanced`), so no proper minor of a member is
one, and by Theorem 4 every minor-minimal one has a member as a link minor.

Graphs are finite (§2.1).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

namespace BiasedGraph

universe uV uE

variable {V : Type uV} {E : Type uE} {Ω : BiasedGraph V E}

/-- **Proposition 3.2.** "Let `(G, B)` be a 2-connected biased graph. If `(G, B)` contains a
vertex-disjoint pair of unbalanced cycles, neither of which is a loop, then `(G, B)` contains a
subdivision of `𝖡₀`, `𝖡₁`, or `𝖡₂`." -/
theorem hasSubdivision_tube [Finite V] [Finite E] (h2 : Ω.graph.IsKConnected 2) {C D : Set E}
    (hC : Ω.IsUnbalancedCycle C) (hD : Ω.IsUnbalancedCycle D)
    (hCD : Disjoint (Ω.graph.verts C) (Ω.graph.verts D)) (hC2 : 2 ≤ C.encard)
    (hD2 : 2 ≤ D.encard) : ∃ S, IsG0Tube S ∧ Ω.HasSubdivision S := by
  sorry

/-- **Lemma 3.4.** "Let `Ω` be a tangled biased graph. Assume `Ω` contains an unbalanced cycle `C`
and a pair of unbalanced cycles `C_x` and `C_y` such that `V(C) ∩ V(C_x) = {x}`,
`V(C) ∩ V(C_y) = {y}`, and `x ≠ y`. Then `C ∪ C_x ∪ C_y` contains a biased `2C₃` with no balanced
2-cycle as a link minor": a link minor of `Ω` deleting every edge off `C ∪ C_x ∪ C_y`. -/
theorem IsTangled.exists_twoC3_linkMinor (hΩ : Ω.IsTangled) {C Cx Cy : Set E} {x y : V}
    (hC : Ω.IsUnbalancedCycle C) (hCx : Ω.IsUnbalancedCycle Cx) (hCy : Ω.IsUnbalancedCycle Cy)
    (hx : Ω.graph.verts C ∩ Ω.graph.verts Cx = {x})
    (hy : Ω.graph.verts C ∩ Ω.graph.verts Cy = {y}) (hxy : x ≠ y) :
    ∃ S, IsG0TwoC3 S ∧ ∃ m : Ω.LinkMinorMap S, (C ∪ Cx ∪ Cy)ᶜ ⊆ m.deleteSet := by
  sorry

/-- **Theorem 3.3.** "Every tangled biased graph contains as a link minor either a biased `2C₃`
with no balanced 2-cycle or a biased `K₄` with no balanced triangle." -/
theorem IsTangled.hasLinkMinor_twoC3_or_K4 [Finite V] [Finite E] (hΩ : Ω.IsTangled) :
    (∃ S, IsG0TwoC3 S ∧ Ω.HasLinkMinor S) ∨ ∃ S, IsG0K4 S ∧ Ω.HasLinkMinor S := by
  sorry

/-- **Theorem 4.** "Every 2-connected properly unbalanced biased graph contains a biased graph in
`𝒢₀` as a link minor." -/
theorem hasG0LinkMinor [Finite V] [Finite E] (h2 : Ω.graph.IsKConnected 2)
    (hΩ : Ω.IsProperlyUnbalanced) : Ω.HasG0LinkMinor := by
  sorry

/-- **§1.2, §3.1.** No 2-connected properly unbalanced biased graph has fewer edges than the members
of `𝒢₀`: with Theorem 4 this makes `𝒢₀` "the set of biased graphs that are minor-minimal amongst
all 2-connected, properly unbalanced biased graphs", since a proper minor has fewer edges. -/
theorem six_le_card_of_isProperlyUnbalanced [Finite V] [Finite E] (h2 : Ω.graph.IsKConnected 2)
    (hΩ : Ω.IsProperlyUnbalanced) : 6 ≤ Nat.card E := by
  sorry

/-- **Theorem 5.** "Every 2-connected properly unbalanced biased graph contains as a biased
subgraph a subdivision of a biased graph in `𝒯₀`." -/
theorem hasT0Subdivision [Finite V] [Finite E] (h2 : Ω.graph.IsKConnected 2)
    (hΩ : Ω.IsProperlyUnbalanced) : Ω.HasT0Subdivision := by
  sorry

/-- **Proposition 3.5.** "Let `(G, B)` be a 2-connected biased graph that contains a contrabalanced
theta subgraph and a unique balancing vertex after removing joints. Then `(G, B)` contains a
subdivision of `𝖣_{1,0}`, `𝖡₀'`, `𝖡₁'`, or `𝖡₂'`." -/
theorem hasSubdivision_D10_or_BPrime [Finite V] [Finite E] (h2 : Ω.graph.IsKConnected 2)
    (hθ : ∃ (x y : V) (P Q R : Ω.graph.Walk x y), IsTheta P Q R ∧
      P.edgeSet ∪ Q.edgeSet ∉ Ω.balanced ∧ P.edgeSet ∪ R.edgeSet ∉ Ω.balanced ∧
      Q.edgeSet ∪ R.edgeSet ∉ Ω.balanced)
    (hu : ∃! u, Ω.IsLinkBalancingVertex u) :
    (∃ S, IsD10 S ∧ Ω.HasSubdivision S) ∨ ∃ S, IsBPrime S ∧ Ω.HasSubdivision S := by
  sorry

end BiasedGraph

open BiasedGraph

/-- A reorientation keeps the ends of every edge. -/
theorem OrientedGraph.IsReorientationOf.mem_verts_univ {V E : Type*} {G H : OrientedGraph V E}
    (h : G.IsReorientationOf H) {v : V} (hv : v ∈ H.verts Set.univ) : v ∈ G.verts Set.univ := by
  obtain ⟨e, -, he⟩ := mem_verts.1 hv
  rcases h e with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
  · rcases he with rfl | rfl
    exacts [mem_verts.2 ⟨e, trivial, Or.inl h₁⟩, mem_verts.2 ⟨e, trivial, Or.inr h₂⟩]
  · rcases he with rfl | rfl
    exacts [mem_verts.2 ⟨e, trivial, Or.inr h₂⟩, mem_verts.2 ⟨e, trivial, Or.inl h₁⟩]

/-- **§3.1.** The biased `2C₃` of `𝒢₀` are 2-connected and properly unbalanced. -/
theorem IsG0TwoC3.isKConnected {S : BiasedGraph (Fin 3) (Fin 6)} (h : IsG0TwoC3 S) :
    S.graph.IsKConnected 2 ∧ S.IsProperlyUnbalanced := by
  refine ⟨Or.inr ⟨by simp [Set.encard_univ]; norm_num, fun u v huv => ?_⟩,
    (isProperlyUnbalanced_iff_of_twoC3 h.1).2 h.2⟩
  obtain ⟨e, he⟩ : ∃ e, twoC3.Joins e u v := by
    revert u v
    simp only [Joins, twoC3]
    decide
  exact ⟨e, h.1.joins he⟩

/-- **§3.1.** The biased tubes of `𝒢₀` are 2-connected and properly unbalanced. -/
theorem IsG0Tube.isKConnected {S : BiasedGraph (Fin 4) (Fin 6)} (h : IsG0Tube S) :
    S.graph.IsKConnected 2 ∧ S.IsProperlyUnbalanced :=
  ⟨isKConnected_two_of_connectedOff (by simp [Set.encard_univ])
      (fun v => h.1.mem_verts_univ (tube_mem_verts v)) (h.1.connectedOff tube_star),
    (isProperlyUnbalanced_iff_of_tube h.1).2 h.2⟩

/-- **§3.1.** The biased `K₄` of `𝒢₀` are 2-connected and properly unbalanced. -/
theorem IsG0K4.isKConnected {S : BiasedGraph (Fin 4) (Fin 6)} (h : IsG0K4 S) :
    S.graph.IsKConnected 2 ∧ S.IsProperlyUnbalanced :=
  ⟨isKConnected_two_of_connectedOff (by simp [Set.encard_univ])
      (fun v => h.1.mem_verts_univ (K4_mem_verts v)) (h.1.connectedOff K4_star),
    (isProperlyUnbalanced_iff_of_K4 h.1).2 h.2⟩

example := IsG0TwoC3.isKConnected twoC3Gain_isG0TwoC3

example := IsG0Tube.isKConnected tubeGain_isG0Tube

example := IsG0K4.isKConnected K4Gain_isG0K4

/-- The hypotheses of Proposition 3.2 hold on the tube with the gains `tubeGain`. -/
example : tube.IsKConnected 2 ∧ (tube.gainBiased tubeGain).IsUnbalancedCycle {0, 1} ∧
    (tube.gainBiased tubeGain).IsUnbalancedCycle {4, 5} ∧
    Disjoint (tube.verts {0, 1}) (tube.verts {4, 5}) ∧ 2 ≤ ({0, 1} : Set (Fin 6)).encard ∧
    2 ≤ ({4, 5} : Set (Fin 6)).encard :=
  ⟨tube_isKConnected, ⟨tube_isCycle₀₁, fun h => tubeGain_noPair _ h (Set.encard_pair (by decide))⟩,
    ⟨tube_isCycle₄₅, fun h => tubeGain_noPair _ h (Set.encard_pair (by decide))⟩,
    tube_disjoint₀₁₄₅, (Set.encard_pair (by decide)).ge, (Set.encard_pair (by decide)).ge⟩

/-- The hypotheses of Lemma 3.4 hold on the tangled `2C₃` of `twoC3Gain`, with its three
2-cycles. -/
example : (twoC3.gainBiased twoC3Gain).IsTangled ∧
    (twoC3.gainBiased twoC3Gain).IsUnbalancedCycle {0, 1} ∧
    (twoC3.gainBiased twoC3Gain).IsUnbalancedCycle {2, 3} ∧
    (twoC3.gainBiased twoC3Gain).IsUnbalancedCycle {4, 5} ∧
    twoC3.verts {0, 1} ∩ twoC3.verts {2, 3} = {1} ∧
    twoC3.verts {0, 1} ∩ twoC3.verts {4, 5} = {0} ∧ (1 : Fin 3) ≠ 0 :=
  ⟨twoC3Gain_isTangled,
    ⟨twoC3_isCycle₀₁, fun h => twoC3Gain_noPair _ h (Set.encard_pair (by decide))⟩,
    ⟨twoC3_isCycle₂₃, fun h => twoC3Gain_noPair _ h (Set.encard_pair (by decide))⟩,
    ⟨twoC3_isCycle₄₅, fun h => twoC3Gain_noPair _ h (Set.encard_pair (by decide))⟩,
    twoC3_verts₀₁_inter₂₃, twoC3_verts₀₁_inter₄₅, by decide⟩

/-- The hypotheses of Theorems 4 and 5 hold on the `2C₃` of `twoC3Gain`. -/
example : twoC3.IsKConnected 2 ∧ (twoC3.gainBiased twoC3Gain).IsProperlyUnbalanced :=
  ⟨twoC3_isKConnected, twoC3Gain_isProperlyUnbalanced⟩

/-- The hypotheses of Proposition 3.5 hold on the contrabalanced `fanGraph`. -/
example : fanGraph.contrabalanced.graph.IsKConnected 2 ∧
    (∃ (x y : Fin 3) (P Q R : fanGraph.Walk x y), IsTheta P Q R ∧
      P.edgeSet ∪ Q.edgeSet ∉ fanGraph.contrabalanced.balanced ∧
      P.edgeSet ∪ R.edgeSet ∉ fanGraph.contrabalanced.balanced ∧
      Q.edgeSet ∪ R.edgeSet ∉ fanGraph.contrabalanced.balanced) ∧
    ∃! u, fanGraph.contrabalanced.IsLinkBalancingVertex u :=
  ⟨fanGraph_isKConnected, ⟨0, 2, _, _, _, fanGraph_isTheta, Set.notMem_empty _,
    Set.notMem_empty _, Set.notMem_empty _⟩, fan_existsUnique_isLinkBalancingVertex⟩

end Transformer.GainMatroid
