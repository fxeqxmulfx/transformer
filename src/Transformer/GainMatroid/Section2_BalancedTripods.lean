module

public import Transformer.GainMatroid.Section2_ExchangeBalance

/-!
# Three paths from a balanced triangle

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, for the proof of Proposition 2.6, which the
source leaves as a "straightforward check". Let `X = {a, b, c}` be a balanced triangle of a biased
graph, with `a = yz`, `b = xz` and `c = xy`, and let `P`, `Q`, `R` be paths off `X` from `x`, `y`,
`z` to a vertex `u` that meet only at `u` (`IsTripod`). With `X` they form a subdivision of `K₄`
when `u` is not a corner of `X`.

For `u ≠ z`, if the cycles `c ∪ P ∪ Q` and `b ∪ P ∪ R` are balanced, so is `a ∪ Q ∪ R`
(`BiasedGraph.mem_balanced_of_tripod_of_ne`): `X` and `c ∪ P ∪ Q` are two cycles of a theta between
`x` and `y` whose third cycle `a ∪ b ∪ P ∪ Q` is then balanced, and that cycle and `b ∪ P ∪ R` are
two cycles of a theta between `z` and `u` whose third cycle is `a ∪ Q ∪ R`. Exchanging the last two
paths of a tripod gives a tripod (`IsTripod.swap₂₃`).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

namespace OrientedGraph

variable {G : OrientedGraph V E}

/-- Paths `P`, `Q`, `R` off `X` from `x`, `y`, `z` to `u` that meet only at `u` (§2.4). -/
structure IsTripod (X : Set E) {x y z u : V} (P : G.Walk x u) (Q : G.Walk y u) (R : G.Walk z u) :
    Prop where
  pathP : P.IsPath
  pathQ : Q.IsPath
  pathR : R.IsPath
  offP : ∀ g ∈ P.edges, g ∉ X
  offQ : ∀ g ∈ Q.edges, g ∉ X
  offR : ∀ g ∈ R.edges, g ∉ X
  edgesPQ : ∀ g ∈ P.edges, g ∉ Q.edges
  edgesPR : ∀ g ∈ P.edges, g ∉ R.edges
  edgesQR : ∀ g ∈ Q.edges, g ∉ R.edges
  vertsPQ : ∀ w ∈ P.support, w ∈ Q.support → w = u
  vertsPR : ∀ w ∈ P.support, w ∈ R.support → w = u
  vertsQR : ∀ w ∈ Q.support, w ∈ R.support → w = u

/-- Exchanging the last two paths of a tripod gives a tripod. -/
theorem IsTripod.swap₂₃ {X : Set E} {x y z u : V} {P : G.Walk x u} {Q : G.Walk y u}
    {R : G.Walk z u} (h : IsTripod X P Q R) : IsTripod X P R Q :=
  ⟨h.pathP, h.pathR, h.pathQ, h.offP, h.offR, h.offQ, h.edgesPR, h.edgesPQ,
    fun g hR hQ => h.edgesQR g hQ hR, h.vertsPR, h.vertsPQ, fun w hR hQ => h.vertsQR w hQ hR⟩

end OrientedGraph

open OrientedGraph

namespace BiasedGraph

variable {Ω : BiasedGraph V E} {a b c : E} {x y z : V}

/-- §2.4: if `X = {a, b, c}` is a balanced triangle and `P`, `Q`, `R` are paths off `X` from `x`,
`y`, `z` to `u ≠ z` that meet only at `u`, and `c ∪ P ∪ Q` and `b ∪ P ∪ R` are balanced, so is
`a ∪ Q ∪ R`: two thetas, between `x` and `y` and between `z` and `u`. -/
theorem mem_balanced_of_tripod_of_ne (hX : Ω.graph.IsLabelledTriangle a b c x y z)
    (hbal : ({a, b, c} : Set E) ∈ Ω.balanced) {u : V} {P : Ω.graph.Walk x u}
    {Q : Ω.graph.Walk y u} {R : Ω.graph.Walk z u} (hT : IsTripod {a, b, c} P Q R) (huz : u ≠ z)
    (hc : insert c (P.edgeSet ∪ Q.edgeSet) ∈ Ω.balanced)
    (hb : insert b (P.edgeSet ∪ R.edgeSet) ∈ Ω.balanced) :
    insert a (Q.edgeSet ∪ R.edgeSet) ∈ Ω.balanced := by
  have hzP : z ∉ P.support := fun hz => huz (hT.vertsPR z hz R.start_mem_support).symm
  have hzQ : z ∉ Q.support := fun hz => huz (hT.vertsQR z hz R.start_mem_support).symm
  set C₁ : Ω.graph.Walk x y := .cons c hX.joins_c (.nil y)
  set C₂ : Ω.graph.Walk x y := P.append Q.reverse
  set C₃ : Ω.graph.Walk x y := .cons b hX.joins_b (.cons a hX.joins_a.symm (.nil y))
  have hθ₁ : IsTheta C₁ C₂ C₃ := by
    refine ⟨hX.ne_xy, ?_, hT.pathP.append hT.pathQ.reverse fun w hw hw' =>
      hT.vertsPQ w hw (Walk.mem_support_reverse.1 hw'), ?_, Walk.openly_single hX.joins_c ?_,
      Walk.openly_single hX.joins_c ?_, Walk.openly_of_not_mem hX.joins_b hX.joins_a.symm ?_⟩
    · simp [C₁, Walk.IsPath, Walk.support, hX.ne_xy]
    · simp [C₃, Walk.IsPath, Walk.support, hX.ne_xz, hX.ne_xy, hX.ne_yz.symm]
    · rw [Walk.mem_edges_append, Walk.mem_edges_reverse]
      rintro (h | h)
      · exact hT.offP c h (Or.inr (Or.inr rfl))
      · exact hT.offQ c h (Or.inr (Or.inr rfl))
    · simp [C₃, Walk.edges, hX.ne_bc.symm, hX.ne_ac.symm]
    · rw [Walk.mem_support_append, Walk.mem_support_reverse]
      exact fun h => h.elim hzP hzQ
  set D₁ : Ω.graph.Walk z u := .cons b hX.joins_b.symm P
  set D₃ : Ω.graph.Walk z u := .cons a hX.joins_a.symm Q
  have hθ₂ : IsTheta D₁ R D₃ := by
    refine ⟨huz.symm, Walk.isPath_cons.2 ⟨hzP, hT.pathP⟩, hT.pathR,
      Walk.isPath_cons.2 ⟨hzQ, hT.pathQ⟩, ⟨fun g hg hg' => ?_, fun w hw hw' => ?_⟩,
      ⟨fun g hg hg' => ?_, fun w hw hw' => ?_⟩, ⟨fun g hg hg' => ?_, fun w hw hw' => ?_⟩⟩
    · rcases List.mem_cons.1 hg with rfl | hg
      · exact hT.offR _ hg' (Or.inr (Or.inl rfl))
      · exact hT.edgesPR g hg hg'
    · rcases List.mem_cons.1 hw with rfl | hw
      · exact Or.inl rfl
      · exact Or.inr (hT.vertsPR w hw hw')
    · rcases List.mem_cons.1 hg with rfl | hg <;> rcases List.mem_cons.1 hg' with h' | hg'
      · exact hX.ne_ab h'.symm
      · exact hT.offQ _ hg' (Or.inr (Or.inl rfl))
      · exact hT.offP _ (h' ▸ hg) (Or.inl rfl)
      · exact hT.edgesPQ g hg hg'
    · rcases List.mem_cons.1 hw with rfl | hw
      · exact Or.inl rfl
      rcases List.mem_cons.1 hw' with rfl | hw'
      · exact Or.inl rfl
      · exact Or.inr (hT.vertsPQ w hw hw')
    · rcases List.mem_cons.1 hg' with rfl | hg'
      · exact hT.offR _ hg (Or.inl rfl)
      · exact hT.edgesQR g hg' hg
    · rcases List.mem_cons.1 hw' with rfl | hw'
      · exact Or.inl rfl
      · exact Or.inr (hT.vertsQR w hw' hw)
  have e₁ : C₁.edgeSet ∪ C₂.edgeSet = insert c (P.edgeSet ∪ Q.edgeSet) := by
    ext g
    simp only [C₁, C₂, Walk.edgeSet, Walk.edges, Walk.edges_append, Walk.edges_reverse,
      Set.mem_union, Set.mem_insert_iff, Set.mem_ofPred_eq, List.mem_cons, List.mem_append,
      List.mem_reverse, List.not_mem_nil, or_false]
  have e₂ : C₁.edgeSet ∪ C₃.edgeSet = {a, b, c} := by
    ext g
    simp only [C₁, C₃, Walk.edgeSet, Walk.edges, Set.mem_union, Set.mem_insert_iff,
      Set.mem_singleton_iff, Set.mem_ofPred_eq, List.mem_cons, List.not_mem_nil, or_false]
    tauto
  have e₃ : D₁.edgeSet ∪ D₃.edgeSet = C₂.edgeSet ∪ C₃.edgeSet := by
    ext g
    simp only [C₂, C₃, D₁, D₃, Walk.edgeSet, Walk.edges, Walk.edges_append, Walk.edges_reverse,
      Set.mem_union, Set.mem_ofPred_eq, List.mem_cons, List.mem_append, List.mem_reverse,
      List.not_mem_nil, or_false]
    tauto
  have e₄ : D₁.edgeSet ∪ R.edgeSet = insert b (P.edgeSet ∪ R.edgeSet) := by
    ext g
    simp only [D₁, Walk.edgeSet, Walk.edges, Set.mem_union, Set.mem_insert_iff, Set.mem_ofPred_eq,
      List.mem_cons]
    tauto
  have e₅ : R.edgeSet ∪ D₃.edgeSet = insert a (Q.edgeSet ∪ R.edgeSet) := by
    ext g
    simp only [D₃, Walk.edgeSet, Walk.edges, Set.mem_union, Set.mem_insert_iff, Set.mem_ofPred_eq,
      List.mem_cons]
    tauto
  have hD := Ω.theta hθ₁ (by rwa [e₁]) (by rwa [e₂])
  have := Ω.theta hθ₂ (by rwa [e₄]) (by rwa [e₃])
  rwa [e₅] at this

end BiasedGraph

/-- The edge `0` of `K₄`, from the corner `1` of the triangle `{3, 4, 5}` to `0`. -/
def K4SpokeX : K4.Walk 1 0 := .cons 0 (Or.inr ⟨rfl, rfl⟩) (.nil 0)

/-- The edge `1` of `K₄`, from the corner `2` of the triangle `{3, 4, 5}` to `0`. -/
def K4SpokeY : K4.Walk 2 0 := .cons 1 (Or.inr ⟨rfl, rfl⟩) (.nil 0)

/-- The edge `2` of `K₄`, from the corner `3` of the triangle `{3, 4, 5}` to `0`. -/
def K4SpokeZ : K4.Walk 3 0 := .cons 2 (Or.inr ⟨rfl, rfl⟩) (.nil 0)

/-- The three spokes of `K₄` at `0` form a tripod off the triangle `{3, 4, 5}`. -/
theorem K4_isTripod : IsTripod {5, 4, 3} K4SpokeX K4SpokeY K4SpokeZ := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, by decide, by decide, by decide, by decide, by decide,
    by decide⟩ <;> simp [K4SpokeX, K4SpokeY, K4SpokeZ, K4, Walk.IsPath, Walk.support, Walk.edges]

/-- The triangle `0, 1, 2` of `K₄`. -/
theorem K4_isCycle_spokesXY : K4.IsCycle (insert 3 (K4SpokeX.edgeSet ∪ K4SpokeY.edgeSet)) := by
  have h : K4.IsLabelledTriangle 3 1 0 0 1 2 := ⟨Or.inl ⟨rfl, rfl⟩, Or.inl ⟨rfl, rfl⟩,
    Or.inl ⟨rfl, rfl⟩, by decide, by decide, by decide, by decide, by decide, by decide⟩
  convert h.isCycle using 1
  ext g
  simp only [K4SpokeX, K4SpokeY, Walk.edgeSet, Walk.edges, Set.mem_insert_iff, Set.mem_union,
    Set.mem_ofPred_eq, Set.mem_singleton_iff, List.mem_cons, List.not_mem_nil, or_false]
  tauto

/-- The triangle `0, 1, 3` of `K₄`. -/
theorem K4_isCycle_spokesXZ : K4.IsCycle (insert 4 (K4SpokeX.edgeSet ∪ K4SpokeZ.edgeSet)) := by
  have h : K4.IsLabelledTriangle 4 2 0 0 1 3 := ⟨Or.inl ⟨rfl, rfl⟩, Or.inl ⟨rfl, rfl⟩,
    Or.inl ⟨rfl, rfl⟩, by decide, by decide, by decide, by decide, by decide, by decide⟩
  convert h.isCycle using 1
  ext g
  simp only [K4SpokeX, K4SpokeZ, Walk.edgeSet, Walk.edges, Set.mem_insert_iff, Set.mem_union,
    Set.mem_ofPred_eq, Set.mem_singleton_iff, List.mem_cons, List.not_mem_nil, or_false]
  tauto

example := K4_isTripod.swap₂₃

example := (K4.gainBiased (1 : Fin 6 → ℚˣ)).mem_balanced_of_tripod_of_ne K4_isLabelledTriangle
  (gainBiased_one_isBalanced K4 _ K4_isLabelledTriangle.isCycle) K4_isTripod (by decide)
  (gainBiased_one_isBalanced K4 _ K4_isCycle_spokesXY)
  (gainBiased_one_isBalanced K4 _ K4_isCycle_spokesXZ)

end Transformer.GainMatroid
