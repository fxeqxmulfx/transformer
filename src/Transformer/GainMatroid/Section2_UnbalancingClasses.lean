module

public import Transformer.GainMatroid.Section2_BiasedMatroids
public import Transformer.GainMatroid.Section4_TwoVertexCircuits

/-!
# Unbalancing classes of an almost-balanced biased graph

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5. Let `(G, B)` be almost balanced with a unique
balancing vertex `u` after deleting its joints (`BiasedGraph.IsRollable`). The source lets `J` be
the joints not incident to `u`, `δ(u)` the links incident to `u`, and `Σ(u) = δ(u) ∪ J`, and
defines:

> "a relation `∼` on `Σ_G(u)` in which `e_i ∼ e_j` if and only if `i = j`, or there is a balanced
> cycle containing `e_i` and `e_j`, or `e_i` and `e_j` are both in `J`."

The source's observation that two joints and a path between them form a frame circuit is proved
here for any two joints (`isFrameCircuit_of_joints`). The claims about `∼` that the source cites
from [AlmostBalancedReps] are stated in `Section2_UnbalancingRelation`.
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

namespace OrientedGraph

variable {G : OrientedGraph V E}

/-- A path contains no cycle. -/
theorem Walk.IsPath.isForest {x y : V} {P : G.Walk x y} (hP : P.IsPath) :
    G.IsForest P.edgeSet := by
  intro C hC hsub
  induction P with
  | nil x =>
    obtain ⟨c, hc⟩ := hC.nonempty
    exact List.not_mem_nil (hsub hc)
  | cons e he p ih =>
    rename_i a b c
    have hnd : (a :: p.support).Nodup := hP
    obtain ⟨hnot, hp⟩ := List.nodup_cons.1 hnd
    rw [Walk.edgeSet_cons] at hsub
    by_cases heC : e ∈ C
    · refine hC.not_mem_of_pendant (p := a) ?_ ?_ (fun f hf hne => ?_) heC
      · rcases he with ⟨h, -⟩ | ⟨-, h⟩
        · exact Or.inl h
        · exact Or.inr h
      · have hab : a ≠ b := fun h => hnot (h ▸ p.start_mem_support)
        rcases he with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
        · rwa [h₁, h₂]
        · rw [h₁, h₂]
          exact hab.symm
      · have hmem : f ∈ p.edges := (hsub hf).resolve_left hne
        have h := p.ends_mem_support hmem
        exact ⟨fun h' => hnot (h' ▸ h.1), fun h' => hnot (h' ▸ h.2)⟩
    · exact ih hp fun d hd => (hsub hd).resolve_left fun h => heC (h ▸ hd)

/-- A path between equal vertices has no edge. -/
theorem Walk.IsPath.edges_eq_nil_of_eq {x y : V} {P : G.Walk x y} (hP : P.IsPath) (h : x = y) :
    P.edges = [] := by
  subst h
  cases P with
  | nil => rfl
  | cons e he p =>
    have hnd : (x :: p.support).Nodup := hP
    exact ((List.nodup_cons.1 hnd).1 p.end_mem_support).elim

/-- A cycle through a loop is that loop. -/
theorem IsCycle.eq_singleton_of_loop {C : Set E} (hC : G.IsCycle C) {l : E} (hl : l ∈ C)
    (hloop : G.tail l = G.head l) : C = {l} := by
  obtain ⟨w, W, hW, rfl⟩ := hC
  exact hW.edgeSet_eq_singleton_of_loop hl hloop

/-- A cycle of at least two edges contains a link. -/
theorem IsCycle.exists_isLink {C : Set E} (hC : G.IsCycle C) (h2 : 2 ≤ C.encard) :
    ∃ e ∈ C, G.IsLink e := by
  obtain ⟨c, hc⟩ := hC.nonempty
  by_cases hl : G.tail c = G.head c
  · rw [hC.eq_singleton_of_loop hc hl, Set.encard_singleton] at h2
    exact absurd h2 (by norm_num)
  · exact ⟨c, hc, hl⟩

open Classical in
/-- The end of `e` other than `u`: the end of a link at `u` away from `u`, or the end of a loop
away from `u`. -/
noncomputable def otherEnd (G : OrientedGraph V E) (u : V) (e : E) : V :=
  if G.tail e = u then G.head e else G.tail e

end OrientedGraph

open OrientedGraph

namespace BiasedGraph

section Definitions

variable (Ω : BiasedGraph V E) (u : V)

/-- `J` (§2.5): the joints of `Ω` not incident to `u`. -/
def offJoints : Set E := {e | Ω.IsJoint e ∧ Ω.graph.tail e ≠ u}

/-- `δ(u)` (§2.5): the links incident to `u`. -/
def linkStar : Set E := {e | Ω.graph.IsLink e ∧ (Ω.graph.tail e = u ∨ Ω.graph.head e = u)}

/-- `Σ(u) = δ(u) ∪ J` (§2.5). -/
def sigmaStar : Set E := Ω.linkStar u ∪ Ω.offJoints u

/-- The relation `∼` of §2.5: `e = f`, or a balanced cycle contains both, or both lie in `J`. -/
def UnbalancingRel (e f : E) : Prop :=
  e = f ∨ (∃ C ∈ Ω.balanced, e ∈ C ∧ f ∈ C) ∨ (e ∈ Ω.offJoints u ∧ f ∈ Ω.offJoints u)

/-- The **unbalancing class** of `e` in `Σ(u)` (§2.5). -/
def unbalancingClass (e : E) : Set E := {f | f ∈ Ω.sigmaStar u ∧ Ω.UnbalancingRel u e f}

/-- `Ω` is almost balanced and, after deleting its joints, `u` is its unique balancing vertex
(§2.5). The cycles left by deleting the joints that can be unbalanced are those of at least two
edges. -/
structure IsRollable : Prop where
  /-- `Ω` is almost balanced. -/
  almost : Ω.IsAlmostBalanced
  /-- `u` is the only vertex on every unbalanced cycle of at least two edges. -/
  unique : ∀ v, (∀ C, Ω.IsUnbalancedCycle C → 2 ≤ C.encard → v ∈ Ω.graph.verts C) ↔ v = u

end Definitions

variable {Ω : BiasedGraph V E} {u : V}

/-- `∼` is symmetric. -/
theorem UnbalancingRel.symm {e f : E} (h : Ω.UnbalancingRel u e f) : Ω.UnbalancingRel u f e := by
  rcases h with rfl | ⟨C, hC, he, hf⟩ | ⟨he, hf⟩
  · exact Or.inl rfl
  · exact Or.inr (Or.inl ⟨C, hC, hf, he⟩)
  · exact Or.inr (Or.inr ⟨hf, he⟩)

/-- No balanced cycle contains a joint. -/
theorem IsJoint.not_mem_of_balanced {e : E} (he : Ω.IsJoint e) {C : Set E} (hC : C ∈ Ω.balanced) :
    e ∉ C := fun h =>
  he.2 ((Ω.isCycle_of_mem _ hC).eq_singleton_of_loop h he.1 ▸ hC)

/-- §2.5: "for every pair `f, f′ ∈ J`, every path linking the endpoint of `f` with the endpoint of
`f′` together with `f` and `f′` is a circuit of `F(G, B)`". This holds for any two joints. -/
theorem isFrameCircuit_of_joints {f g : E} (hf : Ω.IsJoint f) (hg : Ω.IsJoint g) (hne : f ≠ g)
    {P : Ω.graph.Walk (Ω.graph.tail f) (Ω.graph.tail g)} (hP : P.IsPath) :
    Ω.IsFrameCircuit ({f, g} ∪ P.edgeSet) := by
  have hnone : ∀ C ⊆ ({f, g} ∪ P.edgeSet : Set E), Ω.graph.IsCycle C → C ∉ Ω.balanced := by
    intro C hsub hC hb
    by_cases hfC : f ∈ C
    · exact hf.not_mem_of_balanced hb hfC
    by_cases hgC : g ∈ C
    · exact hg.not_mem_of_balanced hb hgC
    refine hP.isForest C hC fun c hc => ?_
    rcases hsub hc with (rfl | rfl) | h
    · exact (hfC hc).elim
    · exact (hgC hc).elim
    · exact h
  refine Or.inr ⟨?_, hnone⟩
  have hCf := isCycle_singleton_of_loop hf.1
  have hCg := isCycle_singleton_of_loop hg.1
  have hvf : Ω.graph.verts {f} = {Ω.graph.tail f} := verts_singleton_of_loop rfl hf.1.symm
  have hvg : Ω.graph.verts {g} = {Ω.graph.tail g} := verts_singleton_of_loop rfl hg.1.symm
  by_cases hxy : Ω.graph.tail f = Ω.graph.tail g
  · have hnil := hP.edges_eq_nil_of_eq hxy
    refine Or.inl ⟨{f}, {g}, Ω.graph.tail f, hCf, hCg, Set.disjoint_singleton.2 hne,
      by rw [hvf, hvg, ← hxy, Set.inter_self], ?_⟩
    have hempty : P.edgeSet = ∅ := by
      ext c
      simp [Walk.edgeSet, hnil]
    rw [hempty, Set.union_empty]
    exact Set.insert_eq f {g}
  · refine Or.inr (Or.inl ⟨{f}, {g}, _, _, P, hCf, hCg, ?_, hP, by rw [hvf]; rfl, by rw [hvg]; rfl,
      fun z _ hz => by rwa [hvf] at hz, fun z _ hz => by rwa [hvg] at hz, ?_⟩)
    · rw [hvf, hvg]
      exact Set.disjoint_singleton.2 hxy
    · rw [Set.insert_eq]

end BiasedGraph

end Transformer.GainMatroid
