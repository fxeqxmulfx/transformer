module

public import Transformer.GainMatroid.Section2_LinkTheta
public import Transformer.GainMatroid.Section4_U24Matroid

/-!
# Two edges at a balancing vertex with a path: circuit or independent

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5. Let `e` and `f` be edges of `Σ(u)` and `P` a
path between their ends away from `u` that misses `u`. A set in which every cycle contains one edge
`g` and no cycle is balanced is independent in the frame matroid
(`BiasedGraph.IsFrameMatroid.indep_of_forall_mem`). If `e` joins the start of `P` to `u`, every
cycle among `e`, `f` and `P` contains `f` (`OrientedGraph.Walk.IsPath.mem_of_isCycle_subset`). So
`{e, f} ∪ P` is independent when one of `e`, `f` is a link at `u` and the other a joint
(`BiasedGraph.indep_of_link_joint`, `BiasedGraph.indep_of_joint_link`) and a circuit when both are
joints (`BiasedGraph.isFrameCircuit_of_joints_otherEnd`). When both are links it is a cycle
(`OrientedGraph.isCycle_pair_union`), and if all cycles through `e` and `f` have one bias, it is a
circuit for every `P` or independent for every `P` (`BiasedGraph.isCircuit_or_indep_of_links`).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

namespace OrientedGraph

variable {G : OrientedGraph V E}

open Walk

/-- The end of a loop away from any vertex is its one end. -/
theorem otherEnd_of_loop {u : V} {e : E} (h : G.tail e = G.head e) :
    G.otherEnd u e = G.tail e := by
  unfold otherEnd
  split_ifs
  exacts [h.symm, rfl]

/-- If `e` joins the start of a path `P` to a vertex `u` off `P`, every cycle among `e`, `f` and
the edges of `P` contains `f`. -/
theorem Walk.IsPath.mem_of_isCycle_subset {u x y : V} {e f : E} {P : G.Walk x y} (hP : P.IsPath)
    (he : G.Joins e x u) (huP : u ∉ P.support) {C : Set E} (hC : G.IsCycle C)
    (hCX : C ⊆ {e, f} ∪ P.edgeSet) : f ∈ C := by
  by_contra hfC
  refine (isPath_cons.2 ⟨huP, hP⟩ : (cons e he.symm P).IsPath).isForest C hC fun c hc => ?_
  rcases hCX hc with (rfl | rfl) | hcP
  · exact List.mem_cons_self
  · exact absurd hc hfC
  · exact List.mem_cons_of_mem _ hcP

/-- Two edges `e ≠ f` joining the ends of a path `P` to a vertex `u` off `P` close a cycle with
`P`. -/
theorem isCycle_pair_union {u x y : V} {e f : E} (he : G.Joins e x u) (hf : G.Joins f y u)
    (hne : e ≠ f) {P : G.Walk x y} (hP : P.IsPath) (huP : u ∉ P.support) :
    G.IsCycle ({e, f} ∪ P.edgeSet) := by
  have hxu : x ≠ u := fun h => huP (by subst h; exact P.start_mem_support)
  have hyu : y ≠ u := fun h => huP (by subst h; exact P.end_mem_support)
  by_cases hxy : x = y
  · subst hxy
    rw [hP.eq_nil]
    convert isCycle_pair he hf hne hxu using 1
    ext g
    simp [Walk.edgeSet, Walk.edges]
  have hS : (cons e he (cons f hf.symm (nil y))).IsPath := by
    unfold Walk.IsPath
    simp [Walk.support, hxu, hxy, hyu.symm]
  refine ⟨x, _, (openly_of_not_mem he hf.symm huP).isCycleWalk hxy hP hS, ?_⟩
  rw [edgeSet_append_reverse]
  ext g
  simp [Walk.edgeSet, Walk.edges]
  tauto

end OrientedGraph

open OrientedGraph

namespace BiasedGraph

variable {Ω : BiasedGraph V E} {M : Matroid E}

/-- A set of edges in which every cycle contains `g` and no cycle is balanced is independent in the
frame matroid: a tight handcuff contains two cycles without a common edge, a loose handcuff two
vertex-disjoint cycles, and a theta three cycles of which no edge lies on all three. -/
theorem IsFrameMatroid.indep_of_forall_mem (hM : Ω.IsFrameMatroid M) {X : Set E} {g : E}
    (hg : ∀ C ⊆ X, Ω.graph.IsCycle C → g ∈ C)
    (hbal : ∀ C ⊆ X, Ω.graph.IsCycle C → C ∉ Ω.balanced) : M.Indep X := by
  refine Matroid.indep_iff_forall_subset_not_isCircuit'.2 ⟨fun Y hY hYc => ?_, ?_⟩
  · rcases (hM.2 Y).1 hYc with hb | ⟨hY', -⟩
    · exact hbal Y hY (Ω.isCycle_of_mem Y hb) hb
    rcases hY' with ⟨C, D, _, hC, hD, hCD, -, rfl⟩ |
        ⟨C, D, _, _, _, hC, hD, hCD, -, -, -, -, -, rfl⟩ | ⟨_, _, P, Q, R, hθ, rfl⟩
    · exact Set.disjoint_left.1 hCD (hg C (Set.subset_union_left.trans hY) hC)
        (hg D (Set.subset_union_right.trans hY) hD)
    · exact Set.disjoint_left.1 hCD
        (mem_verts.2 ⟨g, hg C (fun c hc => hY (Or.inl (Or.inl hc))) hC, Or.inl rfl⟩)
        (mem_verts.2 ⟨g, hg D (fun c hc => hY (Or.inl (Or.inr hc))) hD, Or.inl rfl⟩)
    · obtain ⟨hPQ, hPR, hQR⟩ := hθ.isCycle
      have h₁ := hg _ (Set.subset_union_left.trans hY) hPQ
      have h₂ := hg (P.edgeSet ∪ R.edgeSet)
        (fun c hc => hY (Or.elim hc (fun h => Or.inl (Or.inl h)) Or.inr)) hPR
      rcases hg (Q.edgeSet ∪ R.edgeSet)
        (fun c hc => hY (Or.elim hc (fun h => Or.inl (Or.inr h)) Or.inr)) hQR with h₃ | h₃
      · rcases h₂ with h₂ | h₂
        · exact hθ.openPQ.edges g h₂ h₃
        · exact hθ.openQR.edges g h₃ h₂
      · rcases h₁ with h₁ | h₁
        · exact hθ.openPR.edges g h₁ h₃
        · exact hθ.openQR.edges g h₁ h₃
  · rw [hM.1]
    exact fun c _ => Set.mem_univ c

/-- An edge `e` joining the start of a path `P` to a vertex `u` off `P` and a joint `f` are
independent with `P` in the frame matroid: every cycle among them contains `f`, which no balanced
cycle contains. -/
theorem indep_of_link_joint (hM : Ω.IsFrameMatroid M) {u x y : V} {e f : E}
    (he : Ω.graph.Joins e x u) (hf : Ω.IsJoint f) {P : Ω.graph.Walk x y} (hP : P.IsPath)
    (huP : u ∉ P.support) : M.Indep ({e, f} ∪ P.edgeSet) :=
  hM.indep_of_forall_mem (fun _ hCX hC => hP.mem_of_isCycle_subset he huP hC hCX)
    fun _ hCX hC hb => hf.not_mem_of_balanced hb (hP.mem_of_isCycle_subset he huP hC hCX)

/-- A joint `e` and an edge `f` joining the end of a path `P` to a vertex `u` off `P` are
independent with `P` in the frame matroid. -/
theorem indep_of_joint_link (hM : Ω.IsFrameMatroid M) {u x y : V} {e f : E} (he : Ω.IsJoint e)
    (hf : Ω.graph.Joins f y u) {P : Ω.graph.Walk x y} (hP : P.IsPath) (huP : u ∉ P.support) :
    M.Indep ({e, f} ∪ P.edgeSet) := by
  rw [Set.pair_comm, ← P.edgeSet_reverse]
  exact indep_of_link_joint hM hf he hP.reverse fun h => huP (Walk.mem_support_reverse.1 h)

/-- Two joints and a path between their ends form a circuit of `F(G, B)`
(`isFrameCircuit_of_joints`), with the ends taken away from any vertex `u`. -/
theorem isFrameCircuit_of_joints_otherEnd {u : V} {e f : E} (he : Ω.IsJoint e)
    (hf : Ω.IsJoint f) (hne : e ≠ f)
    {P : Ω.graph.Walk (Ω.graph.otherEnd u e) (Ω.graph.otherEnd u f)} (hP : P.IsPath) :
    Ω.IsFrameCircuit ({e, f} ∪ P.edgeSet) := by
  have key : ∀ {x y : V} (Q : Ω.graph.Walk x y), x = Ω.graph.tail e → y = Ω.graph.tail f →
      Q.IsPath → Ω.IsFrameCircuit ({e, f} ∪ Q.edgeSet) := by
    rintro x y Q rfl rfl hQ
    exact isFrameCircuit_of_joints he hf hne hQ
  exact key P (otherEnd_of_loop he.1) (otherEnd_of_loop hf.1) hP

/-- Let edges `e ≠ f` join `x` and `y` to `u`. If all cycles through `e` and `f` have one bias,
then `{e, f}` with a path from `x` to `y` that misses `u` is a circuit of the frame matroid for
every such path, or independent for every such path. -/
theorem isCircuit_or_indep_of_links (hM : Ω.IsFrameMatroid M) {u x y : V} {e f : E}
    (he : Ω.graph.Joins e x u) (hf : Ω.graph.Joins f y u) (hne : e ≠ f)
    (hbias : ∀ C D, Ω.graph.IsCycle C → Ω.graph.IsCycle D → e ∈ C → f ∈ C → e ∈ D → f ∈ D →
      (C ∈ Ω.balanced ↔ D ∈ Ω.balanced)) :
    (∀ P : Ω.graph.Walk x y, P.IsPath → u ∉ P.support → M.IsCircuit ({e, f} ∪ P.edgeSet)) ∨
      ∀ P : Ω.graph.Walk x y, P.IsPath → u ∉ P.support → M.Indep ({e, f} ∪ P.edgeSet) := by
  have hmem : ∀ P : Ω.graph.Walk x y,
      e ∈ ({e, f} ∪ P.edgeSet : Set E) ∧ f ∈ ({e, f} ∪ P.edgeSet : Set E) :=
    fun _ => ⟨Or.inl (Or.inl rfl), Or.inl (Or.inr rfl)⟩
  by_cases hb : ∃ P : Ω.graph.Walk x y, P.IsPath ∧ u ∉ P.support ∧ {e, f} ∪ P.edgeSet ∈ Ω.balanced
  · obtain ⟨P₀, hP₀, huP₀, hb₀⟩ := hb
    refine Or.inl fun P hP huP => (hM.2 _).2 (Or.inl ((hbias _ _ (isCycle_pair_union he hf hne hP₀
      huP₀) (isCycle_pair_union he hf hne hP huP) (hmem P₀).1 (hmem P₀).2 (hmem P).1
      (hmem P).2).1 hb₀))
  refine Or.inr fun P hP huP => ?_
  have heC : ∀ C ⊆ ({e, f} ∪ P.edgeSet : Set E), Ω.graph.IsCycle C → e ∈ C := fun C hCX hC =>
    hP.reverse.mem_of_isCycle_subset hf (fun h => huP (Walk.mem_support_reverse.1 h)) hC
      (by rwa [Set.pair_comm, P.edgeSet_reverse])
  exact hM.indep_of_forall_mem heC fun C hCX hC hCb => hb ⟨P, hP, huP, (hbias _ _ hC
    (isCycle_pair_union he hf hne hP huP) (heC C hCX hC) (hP.mem_of_isCycle_subset he huP hC hCX)
    (hmem P).1 (hmem P).2).1 hCb⟩

end BiasedGraph

example := otherEnd_of_loop (G := U2) (u := 0) (e := 1) rfl

example := Walk.IsPath.mem_of_isCycle_subset (G := U2) (u := 0) (e := 2) (f := 1)
  (P := Walk.nil 1) (by unfold Walk.IsPath; decide) (Or.inr ⟨rfl, rfl⟩) (by decide)
  (isCycle_singleton_of_loop (e := 1) rfl) (by simp)

example := isCycle_pair_union (G := rerouteBiased.graph) (u := 2) (e := 2) (f := 3)
  (Or.inl ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) (by decide) rerouteP_isPath (by decide)

example := U2_isFrameMatroid.indep_of_forall_mem (X := {2, 1} ∪ (Walk.nil 1 : U2.Walk 1 1).edgeSet)
  (fun _ hCX hC => Walk.IsPath.mem_of_isCycle_subset (u := 0) (P := (Walk.nil 1 : U2.Walk 1 1))
    (by unfold Walk.IsPath; decide) (Or.inr ⟨rfl, rfl⟩) (by decide) hC hCX)
  fun _ _ _ => Set.notMem_empty _

example := BiasedGraph.indep_of_link_joint U2_isFrameMatroid (u := 0) (e := 2) (f := 1)
  (Or.inr ⟨rfl, rfl⟩) ⟨rfl, Set.notMem_empty _⟩ (P := Walk.nil 1)
  (by unfold Walk.IsPath; decide) (by decide)

example := BiasedGraph.indep_of_joint_link U2_isFrameMatroid (u := 0) (e := 1) (f := 2)
  ⟨rfl, Set.notMem_empty _⟩ (Or.inr ⟨rfl, rfl⟩) (P := Walk.nil 1)
  (by unfold Walk.IsPath; decide) (by decide)

example := BiasedGraph.isFrameCircuit_of_joints_otherEnd (Ω := twoLoops.contrabalanced) (u := ())
  (e := true) (f := false) ⟨rfl, Set.notMem_empty _⟩ ⟨rfl, Set.notMem_empty _⟩ (by decide)
  (P := Walk.nil ()) (List.nodup_singleton ())

example := BiasedGraph.isCircuit_or_indep_of_links triangleBiased_isFrameMatroid (u := 0) (e := 0)
  (f := 2) (Or.inr ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) fun C D hC hD _ _ _ _ =>
    iff_of_true (triangleBiased_isBalanced C hC) (triangleBiased_isBalanced D hD)

end Transformer.GainMatroid
