module

public import Transformer.GainMatroid.Section1_Conjectures
public import Transformer.GainMatroid.Section2_BiasedExamples
public import Transformer.GainMatroid.Section2_Matroids

/-!
# Link minors and the frame-type contraction of a joint

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.1 and §2.3. A **link minor** of a biased graph
is obtained by deleting edges and contracting links, and "may always be obtained by contraction of
an acyclic set" (§2.3). `LinkMinorMap Ω Ω'` presents `Ω'`, up to isomorphism, as `Ω / K \ D` for a
forest `K`. A cycle `C` of `Ω / K` is balanced when `C ∪ K₀ ∈ B` for some `K₀ ⊆ K`, which is what
the source's rule "`C ∈ B` or `C ∪ e ∈ B`" gives when applied to the edges of `K` one at a time.

A presentation records the data rather than building `Ω / K \ D`, so the theta property of a minor
is not assumed anywhere. `LinkMinorMap.IsInducedGain` is the source's induced gain function: a
switching of `γ` that is the identity on `K`, read along the orientation of `Ω'`.

The contraction of a joint comes in two types (§2.1). `IsLiftContraction` deletes it and declares
every cycle balanced. `IsFrameContraction` turns each link at its vertex `v` into a joint at the
other end and each other loop at `v` into a balanced loop. `FrameMinorMap` is the one kind of minor
beyond link minors that the source needs: a link minor, one frame-type contraction of a joint and
another link minor (proof of Theorem 6).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

universe uV uE uV' uE'

variable {V : Type uV} {E : Type uE} {V' : Type uV'} {E' : Type uE'} {Γ : Type*}

/-- The graph of a frame-type contraction at `v` (§2.1): a link at `v` becomes a loop at its other
end, and every other edge keeps its ends. -/
noncomputable def OrientedGraph.frameContract (G : OrientedGraph V E) (v : V) : OrientedGraph V E :=
  open Classical in
  ⟨fun f => if G.tail f = v then G.head f else G.tail f,
    fun f => if G.head f = v then G.tail f else G.head f⟩

namespace BiasedGraph

/-- A presentation of `Ω'` as a **link minor** `Ω / K \ D` of `Ω`, up to isomorphism (§2.1, §2.3).
`K` is a forest, and a cycle of `Ω'` is balanced when its edges and some `K₀ ⊆ K` form a balanced
cycle of `Ω`. -/
structure LinkMinorMap (Ω : BiasedGraph V E) (Ω' : BiasedGraph V' E') where
  /-- The contracted forest `K`. -/
  contractSet : Set E
  /-- The deleted edges `D`. -/
  deleteSet : Set E
  /-- The edge of `G / K \ D` behind each edge of `Ω'`. -/
  edge : E' → ↥contractSetᶜ
  /-- The vertex of `G / K` behind each vertex of `Ω'`. -/
  vertex : V' → Quotient (Ω.graph.reachableSetoid contractSet)
  isForest : Ω.graph.IsForest contractSet
  disjoint : Disjoint contractSet deleteSet
  edge_injective : Function.Injective edge
  range_edge : Set.range (fun e' => (edge e' : E)) = (contractSet ∪ deleteSet)ᶜ
  vertex_injective : Function.Injective vertex
  joins : ∀ e', (Ω.graph.contract contractSet).Joins (edge e') (vertex (Ω'.graph.tail e'))
    (vertex (Ω'.graph.head e'))
  balanced_iff : ∀ C', Ω'.graph.IsCycle C' → (C' ∈ Ω'.balanced ↔
    ∃ K₀ ⊆ contractSet, (fun e' => (edge e' : E)) '' C' ∪ K₀ ∈ Ω.balanced)

/-- `Ω` has an `Ω'`-link minor (§2.3). -/
def HasLinkMinor (Ω : BiasedGraph V E) (Ω' : BiasedGraph V' E') : Prop :=
  Nonempty (Ω.LinkMinorMap Ω')

variable {Ω : BiasedGraph V E} {Ω' : BiasedGraph V' E'}

/-- `ι` is a gain function **induced** by `γ` on the link minor (§2.3): a switching of `γ` that is
the identity on `K`, read from the tail to the head of each edge of `Ω'`. -/
@[to_additive IsAddInducedGain
  /-- `ι` is an additive gain function induced by `γ` on the link minor (§2.3). -/]
def LinkMinorMap.IsInducedGain [Group Γ] (m : Ω.LinkMinorMap Ω') (γ : E → Γ) (ι : E' → Γ) : Prop :=
  ∃ η : V → Γ, (∀ f ∈ m.contractSet, switch Ω.graph η γ f = 1) ∧ ∀ e',
    ((Ω.graph.contract m.contractSet).tail (m.edge e') = m.vertex (Ω'.graph.tail e') ∧
        ι e' = switch Ω.graph η γ (m.edge e')) ∨
      ((Ω.graph.contract m.contractSet).head (m.edge e') = m.vertex (Ω'.graph.tail e') ∧
        ι e' = (switch Ω.graph η γ (m.edge e'))⁻¹)

/-- Every biased graph is a link minor of itself. -/
def LinkMinorMap.refl (Ω : BiasedGraph V E) : Ω.LinkMinorMap Ω where
  contractSet := ∅
  deleteSet := ∅
  edge e := ⟨e, by simp⟩
  vertex v := Quotient.mk _ v
  isForest := fun C hC hsub => by
    obtain ⟨u, W, hW, rfl⟩ := hC
    obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hW.edges_ne_nil
    exact Set.notMem_empty e (hsub he)
  disjoint := Set.empty_disjoint _
  edge_injective _ _ h := congrArg Subtype.val h
  range_edge := by
    ext e
    simp
  vertex_injective u v h := by
    obtain ⟨p, hp⟩ := Quotient.exact h
    exact Walk.eq_of_forall_mem_edges id p fun e he => absurd (hp e he) (Set.notMem_empty e)
  joins _ := Or.inl ⟨rfl, rfl⟩
  balanced_iff C _ := by
    simp [Set.subset_empty_iff]

/-- `Ω'` comes from `Ω` by a **lift-type contraction** of the joint `e` (§2.1): delete `e` and make
every cycle balanced. -/
structure IsLiftContraction (Ω : BiasedGraph V E) (e : E) (Ω' : BiasedGraph V ↥({e}ᶜ : Set E)) :
    Prop where
  isJoint : Ω.IsJoint e
  graph_eq : Ω'.graph = Ω.graph.delete {e}
  balanced : ∀ C, Ω'.graph.IsCycle C → C ∈ Ω'.balanced

/-- `Ω'` comes from `Ω` by a **frame-type contraction** of the joint `e` at `v` (§2.1): each link at
`v` becomes a joint at its other end, each other loop at `v` a balanced loop, and the remaining
balanced cycles are those of `Ω` avoiding `v`. -/
structure IsFrameContraction (Ω : BiasedGraph V E) (e : E) (v : V)
    (Ω' : BiasedGraph V ↥({e}ᶜ : Set E)) : Prop where
  isJoint : Ω.IsJoint e
  tail_eq : Ω.graph.tail e = v
  graph_eq : Ω'.graph = (Ω.graph.frameContract v).delete {e}
  balanced_iff : ∀ C, Ω'.graph.IsCycle C → (C ∈ Ω'.balanced ↔
    (∃ f : ↥({e}ᶜ : Set E), Ω.graph.tail f = v ∧ Ω.graph.head f = v ∧ C = {f}) ∨
      (Subtype.val '' C ∈ Ω.balanced ∧ v ∉ Ω.graph.verts (Subtype.val '' C)))

/-- The gain function induced by a **frame-type contraction** of `e` at `v` (§2.3): `γ` off `v`, the
identity on each loop at `v`, and `γ e` on each new joint. -/
@[to_additive IsAddFrameInducedGain
  /-- The additive gain function induced by a frame-type contraction of `e` at `v` (§2.3). -/]
def IsFrameInducedGain [Group Γ] (Ω : BiasedGraph V E) (e : E) (v : V) (γ : E → Γ)
    (ι : ↥({e}ᶜ : Set E) → Γ) : Prop :=
  ∀ f : ↥({e}ᶜ : Set E),
    (Ω.graph.tail f ≠ v ∧ Ω.graph.head f ≠ v ∧ ι f = γ f) ∨
      (Ω.graph.tail f = v ∧ Ω.graph.head f = v ∧ ι f = 1) ∨
        (Ω.graph.IsLink f ∧ (Ω.graph.tail f = v ∨ Ω.graph.head f = v) ∧ ι f = γ e)

/-- A presentation of `Ω'` as a minor using **one frame-type contraction** of a joint: a link minor
`Ω₁` of `Ω`, a frame-type contraction `Ω₂` of a joint of `Ω₁`, and a link minor of `Ω₂` (§3.3). -/
structure FrameMinorMap (Ω : BiasedGraph V E) (Ω' : BiasedGraph V' E') where
  /-- The vertices of the first link minor. -/
  V₁ : Type uV
  /-- The edges of the first link minor. -/
  E₁ : Type uE
  /-- The first link minor. -/
  Ω₁ : BiasedGraph V₁ E₁
  first : Ω.LinkMinorMap Ω₁
  /-- The contracted joint. -/
  joint : E₁
  /-- The vertex of the contracted joint. -/
  vertex : V₁
  /-- The frame-type contraction. -/
  Ω₂ : BiasedGraph V₁ ↥({joint}ᶜ : Set E₁)
  contraction : Ω₁.IsFrameContraction joint vertex Ω₂
  second : Ω₂.LinkMinorMap Ω'

/-- `ι` is induced by `γ` through the three steps of a `FrameMinorMap` (§2.3). -/
@[to_additive IsAddInducedGain
  /-- `ι` is additively induced by `γ` through the three steps of a `FrameMinorMap` (§2.3). -/]
def FrameMinorMap.IsInducedGain [Group Γ] (m : Ω.FrameMinorMap Ω') (γ : E → Γ) (ι : E' → Γ) :
    Prop :=
  ∃ (γ₁ : m.E₁ → Γ) (γ₂ : ↥({m.joint}ᶜ : Set m.E₁) → Γ), m.first.IsInducedGain γ γ₁ ∧
    m.Ω₁.IsFrameInducedGain m.joint m.vertex γ₁ γ₂ ∧ m.second.IsInducedGain γ₂ ι

/-- Contracting one of three parallel links leaves two loops. -/
theorem pumpkin_hasLinkMinor_twoLoops :
    pumpkin.contrabalanced.HasLinkMinor twoLoops.contrabalanced := by
  refine ⟨{ contractSet := {0}, deleteSet := ∅, edge := fun b => ⟨if b then 1 else 2, ?_⟩,
            vertex := fun _ => Quotient.mk _ 0, isForest := pumpkin_isForest,
            disjoint := by simp, edge_injective := ?_, range_edge := ?_, vertex_injective := ?_,
            joins := ?_, balanced_iff := ?_ }⟩
  · cases b <;> simp
  · intro a b h
    cases a <;> cases b <;> simp_all
  · ext a
    fin_cases a <;> simp
  · intro _ _ _
    rfl
  · exact fun _ => Or.inl ⟨rfl, (contract_class_eq (G := pumpkin) (Set.mem_singleton 0)).symm⟩
  · exact fun _ _ => ⟨fun h => absurd h (Set.notMem_empty _),
      fun ⟨_, _, h⟩ => absurd h (Set.notMem_empty _)⟩

example [Group Γ] (γ : E → Γ) : (LinkMinorMap.refl Ω).IsInducedGain γ γ :=
  ⟨1, fun f hf => absurd hf (Set.notMem_empty f), fun e => Or.inl ⟨rfl, by rw [switch_one]; rfl⟩⟩

end BiasedGraph

end Transformer.GainMatroid
