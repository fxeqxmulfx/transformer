module

public import Transformer.GainMatroid.Section2_PairCircuits

/-!
# The relation of §2.5 on the edges at a balancing vertex

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5. The source cites its reference
[AlmostBalancedReps], which is not among the local sources, for two claims about a biased graph
with a unique balancing vertex `u` after deleting its joints: all cycles through two edges of `δ(u)`
have the same bias (`IsRollable.balanced_iff`), and `∼` is an equivalence relation
(`IsRollable.unbalancingRel_trans`). Both are proved here: every cycle that misses `u` is balanced,
so by `BiasedGraph.insert_mem_balanced_iff` two cycles through the same two edges at `u` have one
bias, and a theta through three links at `u` (`BiasedGraph.exists_balanced_of_links`) makes `∼`
transitive. The dichotomy of the source's next sentence (`IsRollable.isCircuit_or_indep`) follows
from the first claim (`Section2_PairCircuits`). The source says "each pair" of edges; these claims
are for two distinct edges, since a single edge lies on balanced and on unbalanced cycles alike.

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
such path together with `{e₁, e₂}` is independent in `F(G, B)`". Proved here: for two links by
`IsRollable.balanced_iff`, for a link and a joint the set is independent, and for two joints a
circuit (`isCircuit_or_indep_of_links`, `indep_of_link_joint`, `isFrameCircuit_of_joints`). -/
theorem IsRollable.isCircuit_or_indep {M : Matroid E} (h : Ω.IsRollable u)
    (hM : Ω.IsFrameMatroid M) {e f : E} (he : e ∈ Ω.sigmaStar u) (hf : f ∈ Ω.sigmaStar u)
    (hne : e ≠ f) :
    (∀ P : Ω.graph.Walk (Ω.graph.otherEnd u e) (Ω.graph.otherEnd u f), P.IsPath → u ∉ P.support →
        M.IsCircuit ({e, f} ∪ P.edgeSet)) ∨
      ∀ P : Ω.graph.Walk (Ω.graph.otherEnd u e) (Ω.graph.otherEnd u f), P.IsPath →
        u ∉ P.support → M.Indep ({e, f} ∪ P.edgeSet) := by
  rcases he with heL | heJ <;> rcases hf with hfL | hfJ
  · exact isCircuit_or_indep_of_links hM (joins_otherEnd heL.2) (joins_otherEnd hfL.2) hne
      fun _ _ hC hD heC hfC heD hfD => h.balanced_iff heL hfL hne hC hD heC hfC heD hfD
  · exact Or.inr fun _ hP huP => indep_of_link_joint hM (joins_otherEnd heL.2) hfJ.1 hP huP
  · exact Or.inr fun _ hP huP => indep_of_joint_link hM heJ.1 (joins_otherEnd hfL.2) hP huP
  · exact Or.inl fun _ hP _ => (hM.2 _).2 (isFrameCircuit_of_joints_otherEnd heJ.1 hfJ.1 hne hP)

/-- §2.5: "`∼` is an equivalence relation", citing [AlmostBalancedReps, Lemma 1.4]. Reflexivity is
the first clause of `UnbalancingRel` and symmetry is `UnbalancingRel.symm`. Transitivity is proved
here: no balanced cycle contains a joint, and balanced cycles through `e, f` and through `f, g`
give one through `e` and `g` (`exists_balanced_of_links`, with `IsRollable.balanced_iff`). -/
theorem IsRollable.unbalancingRel_trans (h : Ω.IsRollable u) {e f g : E}
    (he : e ∈ Ω.sigmaStar u) (hf : f ∈ Ω.sigmaStar u) (hg : g ∈ Ω.sigmaStar u)
    (hef : Ω.UnbalancingRel u e f) (hfg : Ω.UnbalancingRel u f g) : Ω.UnbalancingRel u e g := by
  have hlink : ∀ {c : E} {B : Set E}, c ∈ Ω.sigmaStar u → B ∈ Ω.balanced → c ∈ B →
      c ∈ Ω.linkStar u := fun hc hB hcB =>
    Or.resolve_right hc fun hJ => hJ.1.not_mem_of_balanced hB hcB
  rcases hef with rfl | ⟨C, hC, heC, hfC⟩ | ⟨heJ, hfJ⟩
  · exact hfg
  · rcases hfg with rfl | ⟨D, hD, hfD, hgD⟩ | ⟨hfJ, -⟩
    · exact Or.inr (Or.inl ⟨C, hC, heC, hfC⟩)
    · by_cases hef : e = f
      · subst hef
        exact Or.inr (Or.inl ⟨D, hD, hfD, hgD⟩)
      by_cases hfg : f = g
      · subst hfg
        exact Or.inr (Or.inl ⟨C, hC, heC, hfC⟩)
      by_cases heg : e = g
      · exact Or.inl heg
      have heL := hlink he hC heC
      have hfL := hlink hf hC hfC
      have hgL := hlink hg hD hgD
      have hC' := Ω.isCycle_of_mem _ hC
      have hD' := Ω.isCycle_of_mem _ hD
      exact Or.inr (Or.inl (exists_balanced_of_links hef hfg heg heL.2 hfL.2 hgL.2 hC' hD' heC hfC
        hfD hgD (fun _ hB heB hfB => (h.balanced_iff heL hfL hef hB hC' heB hfB heC hfC).2 hC)
        fun _ hB hfB hgB => (h.balanced_iff hfL hgL hfg hB hD' hfB hgB hfD hgD).2 hD))
    · exact (hfJ.1.not_mem_of_balanced hC hfC).elim
  · rcases hfg with rfl | ⟨D, hD, hfD, -⟩ | ⟨-, hgJ⟩
    · exact Or.inr (Or.inr ⟨heJ, hfJ⟩)
    · exact (hfJ.1.not_mem_of_balanced hD hfD).elim
    · exact Or.inr (Or.inr ⟨heJ, hgJ⟩)

end BiasedGraph

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
