module

public import Transformer.GainMatroid.Section2_RerouteBias

/-!
# The relation of §2.5 on the edges at a balancing vertex

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5. The source cites its reference
[AlmostBalancedReps], which is not among the local sources, for two claims about a biased graph
with a unique balancing vertex `u` after deleting its joints: all cycles through two edges of `δ(u)`
have the same bias (`IsRollable.balanced_iff`), and `∼` is an equivalence relation
(`IsRollable.unbalancingRel_trans`). The first is proved here: every cycle that misses `u` is
balanced, so by `BiasedGraph.insert_mem_balanced_iff` two cycles through the same two edges at `u`
have one bias. The source's next sentence gives the dichotomy `IsRollable.isCircuit_or_indep`. The
source says "each pair" of edges; these claims are for two distinct edges, since a single edge lies
on balanced and on unbalanced cycles alike.

`isRollable_of_digons` gives the hypothesis on any graph with two digons at a vertex whose links all
meet that vertex, and `digonJoint` is the instance used for the examples of §2.5.
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

open OrientedGraph

namespace BiasedGraph

variable {Ω : BiasedGraph V E} {u : V}

/-- §2.5: "for each pair `e, e′ ∈ δ(u)`, either all cycles containing both `e` and `e′` are
balanced or all cycles containing both `e` and `e′` are unbalanced". The source cites
[AlmostBalancedReps, Section 1]. Proved here: a cycle of two openly disjoint paths that miss `u` is
balanced, since `u` is on every unbalanced cycle of at least two edges, and the two cycles are `e`,
`f` and two paths between their other ends that miss `u` (`insert_mem_balanced_iff`). -/
theorem IsRollable.balanced_iff (h : Ω.IsRollable u) {e f : E} (he : e ∈ Ω.linkStar u)
    (hf : f ∈ Ω.linkStar u) (hne : e ≠ f) {C D : Set E} (hC : Ω.graph.IsCycle C)
    (hD : Ω.graph.IsCycle D) (heC : e ∈ C) (hfC : f ∈ C) (heD : e ∈ D) (hfD : f ∈ D) :
    C ∈ Ω.balanced ↔ D ∈ Ω.balanced := by
  have hbal : ∀ {a b : V} (A B : Ω.graph.Walk a b), a ≠ b → A.IsPath → B.IsPath →
      Walk.Openly A B → u ∉ A.support → u ∉ B.support → A.edgeSet ∪ B.edgeSet ∈ Ω.balanced := by
    intro a b A B hab hA hB hAB huA huB
    by_contra hnot
    obtain ⟨g, hg⟩ := List.exists_mem_of_ne_nil _ (Walk.edges_ne_nil_of_ne hab A)
    obtain ⟨g', hg'⟩ := List.exists_mem_of_ne_nil _ (Walk.edges_ne_nil_of_ne hab B)
    have hsub : ({g, g'} : Set E) ⊆ A.edgeSet ∪ B.edgeSet :=
      Set.insert_subset_iff.2 ⟨Or.inl hg, Set.singleton_subset_iff.2 (Or.inr hg')⟩
    have hgg : g ≠ g' := fun h => hAB.edges g hg (h ▸ hg')
    have h2 : 2 ≤ (A.edgeSet ∪ B.edgeSet).encard :=
      (Set.encard_pair hgg).symm.le.trans (Set.encard_le_encard hsub)
    obtain ⟨c, hc, hcu⟩ := mem_verts.1 ((h.unique u).2 rfl _
      ⟨⟨a, _, hAB.isCycleWalk hab hA hB, Walk.edgeSet_append_reverse A B⟩, hnot⟩ h2)
    rcases hc with hc | hc
    · exact huA (Walk.mem_support_of_mem_edges (show c ∈ A.edges from hc) hcu)
    · exact huB (Walk.mem_support_of_mem_edges (show c ∈ B.edges from hc) hcu)
  obtain ⟨P, hP, huP, rfl⟩ := hC.exists_isPath_of_mem hne heC hfC he.2 hf.2
  obtain ⟨Q, hQ, huQ, rfl⟩ := hD.exists_isPath_of_mem hne heD hfD he.2 hf.2
  exact insert_mem_balanced_iff hbal (joins_otherEnd he.2) (joins_otherEnd hf.2) hP hQ huP huQ

/-- §2.5: for each pair of edges `e₁, e₂` of `Σ(u)`, "either every minimal path linking the
endpoints of `e₁` and `e₂` in `G − u` together with `{e₁, e₂}` forms a circuit in `F(G, B)`, or each
such path together with `{e₁, e₂}` is independent in `F(G, B)`". -/
theorem IsRollable.isCircuit_or_indep {M : Matroid E} (h : Ω.IsRollable u)
    (hM : Ω.IsFrameMatroid M) {e f : E} (he : e ∈ Ω.sigmaStar u) (hf : f ∈ Ω.sigmaStar u)
    (hne : e ≠ f) :
    (∀ P : Ω.graph.Walk (Ω.graph.otherEnd u e) (Ω.graph.otherEnd u f), P.IsPath → u ∉ P.support →
        M.IsCircuit ({e, f} ∪ P.edgeSet)) ∨
      ∀ P : Ω.graph.Walk (Ω.graph.otherEnd u e) (Ω.graph.otherEnd u f), P.IsPath →
        u ∉ P.support → M.Indep ({e, f} ∪ P.edgeSet) := by
  sorry

/-- §2.5: "`∼` is an equivalence relation", citing [AlmostBalancedReps, Lemma 1.4]. Reflexivity is
the first clause of `UnbalancingRel` and symmetry is `UnbalancingRel.symm`. -/
theorem IsRollable.unbalancingRel_trans (h : Ω.IsRollable u) {e f g : E}
    (he : e ∈ Ω.sigmaStar u) (hf : f ∈ Ω.sigmaStar u) (hg : g ∈ Ω.sigmaStar u)
    (hef : Ω.UnbalancingRel u e f) (hfg : Ω.UnbalancingRel u f g) : Ω.UnbalancingRel u e g := by
  sorry

end BiasedGraph

/-- An edge between `x` and `y` misses every other vertex. -/
theorem OrientedGraph.Joins.ends_ne {G : OrientedGraph V E} {c : E} {x y v : V}
    (hc : G.Joins c x y) (hx : v ≠ x) (hy : v ≠ y) : G.tail c ≠ v ∧ G.head c ≠ v := by
  rcases hc with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩ <;> rw [h₁, h₂]
  · exact ⟨hx.symm, hy.symm⟩
  · exact ⟨hy.symm, hx.symm⟩

/-- Two edges between `x` and `y` miss every other vertex. -/
theorem OrientedGraph.not_mem_verts_pair {G : OrientedGraph V E} {c d : E} {x y v : V}
    (hc : G.Joins c x y) (hd : G.Joins d x y) (hx : v ≠ x) (hy : v ≠ y) : v ∉ G.verts {c, d} := by
  rw [mem_verts]
  rintro ⟨f, (rfl | rfl), hv | hv⟩
  · exact (hc.ends_ne hx hy).1 hv
  · exact (hc.ends_ne hx hy).2 hv
  · exact (hd.ends_ne hx hy).1 hv
  · exact (hd.ends_ne hx hy).2 hv

/-- A contrabalanced graph on three vertices with a digon from `0` to each of the others, whose
links all meet `0`, has `0` as its unique balancing vertex after deleting its joints. -/
theorem isRollable_of_digons {G : OrientedGraph (Fin 3) E} {a a' b b' : E}
    (ha : G.Joins a 0 1) (ha' : G.Joins a' 0 1) (haa : a ≠ a')
    (hb : G.Joins b 0 2) (hb' : G.Joins b' 0 2) (hbb : b ≠ b')
    (hlink : ∀ e, G.IsLink e → G.tail e = 0 ∨ G.head e = 0) : G.contrabalanced.IsRollable 0 := by
  have hA := isCycle_pair ha ha' haa (by decide)
  have hB := isCycle_pair hb hb' hbb (by decide)
  have hzero : ∀ C, G.contrabalanced.IsUnbalancedCycle C → 2 ≤ C.encard →
      (0 : Fin 3) ∈ G.verts C := fun C hC h2 => by
    obtain ⟨e, heC, he⟩ := IsCycle.exists_isLink (G := G) hC.1 h2
    exact mem_verts.2 ⟨e, heC, hlink e he⟩
  refine ⟨⟨fun h => h _ hA, 0, hzero⟩, fun v => ⟨fun h => ?_, ?_⟩⟩
  · by_contra hv
    have h12 : ∀ w : Fin 3, w ≠ 0 → w = 1 ∨ w = 2 := by decide
    rcases h12 v hv with rfl | rfl
    · exact not_mem_verts_pair hb hb' (by decide) (by decide)
        (h _ ⟨hB, Set.notMem_empty _⟩ (Set.encard_pair hbb).ge)
    · exact not_mem_verts_pair ha ha' (by decide) (by decide)
        (h _ ⟨hA, Set.notMem_empty _⟩ (Set.encard_pair haa).ge)
  · rintro rfl
    exact hzero

/-- Two digons at `0` with a joint at `1`, for the examples of §2.5. -/
def digonJoint : OrientedGraph (Fin 3) (Fin 5) := ⟨![0, 0, 0, 0, 1], ![1, 1, 2, 2, 1]⟩

theorem digonJoint_isRollable : digonJoint.contrabalanced.IsRollable 0 :=
  isRollable_of_digons (a := 0) (a' := 1) (b := 2) (b' := 3) (Or.inl ⟨rfl, rfl⟩)
    (Or.inl ⟨rfl, rfl⟩) (by decide) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide)
    (by decide)

/-- The edge `0` is a link at `0`. -/
theorem digonJoint_link₀ : (0 : Fin 5) ∈ digonJoint.contrabalanced.linkStar 0 :=
  ⟨by decide, Or.inl rfl⟩

/-- The edge `1` is a link at `0`. -/
theorem digonJoint_link₁ : (1 : Fin 5) ∈ digonJoint.contrabalanced.linkStar 0 :=
  ⟨by decide, Or.inl rfl⟩

/-- The digon on the edges `0` and `1`. -/
theorem digonJoint_isCycle : digonJoint.contrabalanced.graph.IsCycle ({0, 1} : Set (Fin 5)) :=
  isCycle_pair (G := digonJoint) (u := 0) (v := 1) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)
    (by decide) (by decide)

example := Walk.IsPath.isForest (G := pumpkin) (pumpkinWalk_isPath 0)

example := Walk.IsPath.edges_eq_nil_of_eq (P := Walk.nil (G := pumpkin) 0) (List.nodup_singleton 0)
  rfl

example := IsCycle.eq_singleton_of_loop (G := twoLoops) (isCycle_singleton_of_loop (e := true) rfl)
  (Set.mem_singleton true) rfl

example := IsCycle.exists_isLink digonJoint_isCycle (Set.encard_pair (by decide)).ge

example := BiasedGraph.UnbalancingRel.symm (Ω := digonJoint.contrabalanced) (u := 0) (e := 0)
  (f := 0) (Or.inl rfl)

example := BiasedGraph.IsJoint.not_mem_of_balanced
  (Ω := twoLoops.singleCycleBiased {true} (isCycle_singleton_of_loop rfl)) (e := false)
  ⟨rfl, fun h => Bool.false_ne_true (Set.singleton_eq_singleton_iff.1 h)⟩ (Set.mem_singleton _)

example := BiasedGraph.isFrameCircuit_of_joints (Ω := twoLoops.contrabalanced) (f := true)
  (g := false) ⟨rfl, Set.notMem_empty _⟩ ⟨rfl, Set.notMem_empty _⟩ (by decide)
  (P := Walk.nil ()) (List.nodup_singleton ())

example := digonJoint_isRollable.balanced_iff digonJoint_link₀ digonJoint_link₁ (by decide)
  digonJoint_isCycle digonJoint_isCycle (by simp) (by simp) (by simp) (by simp)

example := digonJoint_isRollable.isCircuit_or_indep
  (Classical.choose_spec digonJoint.contrabalanced.exists_isFrameMatroid)
  (Or.inl digonJoint_link₀) (Or.inl digonJoint_link₁) (by decide)

example := digonJoint_isRollable.unbalancingRel_trans (Or.inl digonJoint_link₀)
  (Or.inl digonJoint_link₀) (Or.inl digonJoint_link₀) (Or.inl rfl) (Or.inl rfl)

end Transformer.GainMatroid
