module

public import Transformer.GainMatroid.Section2_LinkMinorIndep

/-!
# Frame and lift independence through a graph carried into a contraction

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2, §2.4. A graph `H` is carried into `G / K`,
for a forest `K` of a biased graph `Ω`, by injective maps of its vertices and edges under which each
edge joins the images of its ends (`BiasedGraph.ContractEmbedding`). A set `S` of edges of `H` gives
the set `ι(S) ∪ K` of edges of `G` (`BiasedGraph.ContractEmbedding.carry`), and a cycle of `H` is
balanced through the embedding when its image with some edges of `K` is balanced in `Ω`
(`BiasedGraph.ContractEmbedding.Balanced`). A link minor gives such an embedding
(`BiasedGraph.LinkMinorMap.toContractEmbedding`).

The set `ι(S) ∪ K` is frame-independent exactly when no cycle of `H` in `S` is balanced through the
embedding and no two distinct cycles of `H` in `S` are linked along `S`
(`BiasedGraph.ContractEmbedding.isFrameIndep_iff`), and lift-independent exactly when no cycle in
`S` is balanced through it and `S` holds at most one cycle
(`BiasedGraph.ContractEmbedding.isLiftIndep_iff`). These conditions involve only `H` and its cycles
balanced through the embedding, so two embeddings of one graph with the same balanced cycles carry
the same sets to independent sets (`BiasedGraph.ContractEmbedding.isFrameIndep_congr`,
`BiasedGraph.ContractEmbedding.isLiftIndep_congr`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace BiasedGraph

open OrientedGraph

variable {V E V' E' : Type*}

/-- A graph `H` carried into `G / K` for a forest `K` of `Ω`, by injective maps of its vertices and
edges under which each edge joins the images of its ends. -/
structure ContractEmbedding (Ω : BiasedGraph V E) (H : OrientedGraph V' E') where
  /-- The contracted forest `K`. -/
  contractSet : Set E
  /-- The vertex of `G / K` behind each vertex of `H`. -/
  vertex : V' → Quotient (Ω.graph.reachableSetoid contractSet)
  /-- The edge of `G / K` behind each edge of `H`. -/
  edge : E' → ↥contractSetᶜ
  isForest : Ω.graph.IsForest contractSet
  vertex_injective : Function.Injective vertex
  edge_injective : Function.Injective edge
  joins : ∀ e, (Ω.graph.contract contractSet).Joins (edge e) (vertex (H.tail e))
    (vertex (H.head e))

variable {Ω : BiasedGraph V E} {H : OrientedGraph V' E'}

/-- The graph of a link minor, carried into the contraction. -/
def LinkMinorMap.toContractEmbedding {Ω' : BiasedGraph V' E'} (m : Ω.LinkMinorMap Ω') :
    Ω.ContractEmbedding Ω'.graph :=
  ⟨m.contractSet, m.vertex, m.edge, m.isForest, m.vertex_injective, m.edge_injective, m.joins⟩

namespace ContractEmbedding

variable (m : Ω.ContractEmbedding H)

/-- The edges of `G` carrying a set of edges of `H`, together with `K`. -/
def carry (S : Set E') : Set E := Subtype.val '' (m.edge '' S) ∪ m.contractSet

/-- A set of edges of `H` is balanced through `m` when its image with some edges of `K` is
balanced in `Ω`. -/
def Balanced (C : Set E') : Prop :=
  ∃ K₀ ⊆ m.contractSet, Subtype.val '' (m.edge '' C) ∪ K₀ ∈ Ω.balanced

/-- A set of edges of `G / K` among the images of `S` is the image of its preimage, which lies in
`S`. -/
theorem image_preimage_of_subset {S : Set E'} {C : Set ↥m.contractSetᶜ} (hC : C ⊆ m.edge '' S) :
    m.edge '' (m.edge ⁻¹' C) = C ∧ m.edge ⁻¹' C ⊆ S :=
  ⟨Set.image_preimage_eq_of_subset (hC.trans (Set.image_subset_range _ _)), fun _ he =>
    m.edge_injective.mem_set_image.1 (hC he)⟩

/-- The cycles of `H` are carried to the cycles of `G / K`. -/
theorem isCycle_iff {C : Set E'} :
    H.IsCycle C ↔ (Ω.graph.contract m.contractSet).IsCycle (m.edge '' C) :=
  ⟨fun h => h.map_both m.vertex_injective m.edge_injective m.joins,
    fun h => h.of_map_both m.vertex_injective m.edge_injective m.joins⟩

/-- The cycles of `H` in `S` balanced through `m` correspond to the cycles of `G / K` among the
images of `S` whose union with some edges of `K` is balanced in `Ω`. -/
theorem forall_not_balanced_iff {S : Set E'} :
    (∀ C ⊆ S, H.IsCycle C → ¬m.Balanced C) ↔
      ∀ C ⊆ m.edge '' S, (Ω.graph.contract m.contractSet).IsCycle C →
        ¬∃ K₀ ⊆ m.contractSet, Subtype.val '' C ∪ K₀ ∈ Ω.balanced := by
  refine ⟨fun hB C hCS hC hK => ?_, fun hB C hCS hC hCB =>
    hB _ (Set.image_mono hCS) ((m.isCycle_iff).1 hC) hCB⟩
  obtain ⟨hCe, hCS'⟩ := m.image_preimage_of_subset hCS
  rw [← hCe] at hC hK
  exact hB _ hCS' ((m.isCycle_iff).2 hC) hK

variable [Finite V] [Finite E]

/-- **Frame independence through an embedding.** `ι(S) ∪ K` is frame-independent exactly when no
cycle of `H` in `S` is balanced through `m` and no two distinct cycles in `S` are linked along
`S`. -/
theorem isFrameIndep_iff (S : Set E') :
    Ω.IsFrameIndep (m.carry S) ↔ (∀ C ⊆ S, H.IsCycle C → ¬m.Balanced C) ∧
      ∀ C D, H.IsCycle C → H.IsCycle D → C ⊆ S → D ⊆ S →
        ∀ u ∈ H.verts C, ∀ w ∈ H.verts D, H.ReachableIn S u w → C = D := by
  rw [carry, isFrameIndep_union_iff m.isForest, m.forall_not_balanced_iff]
  refine and_congr_right fun _ => ⟨fun hl C D hC hD hCS hDS u hu w hw huw =>
    m.edge_injective.image_injective (hl _ _ ((m.isCycle_iff).1 hC) ((m.isCycle_iff).1 hD)
      (Set.image_mono hCS) (Set.image_mono hDS) _
      (by rw [verts_map_both m.joins]; exact Set.mem_image_of_mem _ hu) _
      (by rw [verts_map_both m.joins]; exact Set.mem_image_of_mem _ hw)
      ((reachableIn_map_both_iff m.vertex_injective m.edge_injective m.joins).1 huw)),
    fun hl C D hC hD hCS hDS u hu w hw huw => ?_⟩
  obtain ⟨hCe, hCS'⟩ := m.image_preimage_of_subset hCS
  obtain ⟨hDe, hDS'⟩ := m.image_preimage_of_subset hDS
  rw [← hCe] at hC hu ⊢
  rw [← hDe] at hD hw ⊢
  rw [verts_map_both m.joins] at hu hw
  obtain ⟨u', hu', rfl⟩ := hu
  obtain ⟨w', hw', rfl⟩ := hw
  rw [hl _ _ ((m.isCycle_iff).2 hC) ((m.isCycle_iff).2 hD) hCS' hDS' u' hu' w' hw'
    ((reachableIn_map_both_iff m.vertex_injective m.edge_injective m.joins).2 huw)]

/-- **Lift independence through an embedding.** `ι(S) ∪ K` is lift-independent exactly when no
cycle of `H` in `S` is balanced through `m` and `S` holds at most one cycle. -/
theorem isLiftIndep_iff (S : Set E') :
    Ω.IsLiftIndep (m.carry S) ↔ (∀ C ⊆ S, H.IsCycle C → ¬m.Balanced C) ∧
      ∀ C D, H.IsCycle C → H.IsCycle D → C ⊆ S → D ⊆ S → C = D := by
  rw [carry, isLiftIndep_union_iff m.isForest, m.forall_not_balanced_iff, nullity_le_one_iff]
  refine and_congr_right fun _ => ⟨fun h C D hC hD hCS hDS => m.edge_injective.image_injective
    (h _ _ ((m.isCycle_iff).1 hC) ((m.isCycle_iff).1 hD) (Set.image_mono hCS)
      (Set.image_mono hDS)), fun h C D hC hD hCS hDS => ?_⟩
  obtain ⟨hCe, hCS'⟩ := m.image_preimage_of_subset hCS
  obtain ⟨hDe, hDS'⟩ := m.image_preimage_of_subset hDS
  rw [← hCe, ← hDe] at ⊢
  rw [← hCe] at hC
  rw [← hDe] at hD
  rw [h _ _ ((m.isCycle_iff).2 hC) ((m.isCycle_iff).2 hD) hCS' hDS']

variable {V₂ E₂ : Type*} [Finite V₂] [Finite E₂] {Ω₂ : BiasedGraph V₂ E₂}

/-- Two embeddings of one graph with the same balanced cycles carry the same sets to
frame-independent sets. -/
theorem isFrameIndep_congr (m₂ : Ω₂.ContractEmbedding H)
    (hB : ∀ C, H.IsCycle C → (m.Balanced C ↔ m₂.Balanced C)) (S : Set E') :
    Ω.IsFrameIndep (m.carry S) ↔ Ω₂.IsFrameIndep (m₂.carry S) := by
  rw [m.isFrameIndep_iff, m₂.isFrameIndep_iff]
  exact and_congr_left fun _ => forall₂_congr fun C _ => imp_congr_right fun hC =>
    not_congr (hB C hC)

/-- Two embeddings of one graph with the same balanced cycles carry the same sets to
lift-independent sets. -/
theorem isLiftIndep_congr (m₂ : Ω₂.ContractEmbedding H)
    (hB : ∀ C, H.IsCycle C → (m.Balanced C ↔ m₂.Balanced C)) (S : Set E') :
    Ω.IsLiftIndep (m.carry S) ↔ Ω₂.IsLiftIndep (m₂.carry S) := by
  rw [m.isLiftIndep_iff, m₂.isLiftIndep_iff]
  exact and_congr_left fun _ => forall₂_congr fun C _ => imp_congr_right fun hC =>
    not_congr (hB C hC)

end ContractEmbedding

end BiasedGraph

open BiasedGraph

example := pumpkin_hasLinkMinor_twoLoops.some.toContractEmbedding.image_preimage_of_subset
  (S := ∅) subset_rfl

example := pumpkin_hasLinkMinor_twoLoops.some.toContractEmbedding.isCycle_iff (C := {true})

example := pumpkin_hasLinkMinor_twoLoops.some.toContractEmbedding.forall_not_balanced_iff
  (S := {true})

example := pumpkin_hasLinkMinor_twoLoops.some.toContractEmbedding.isFrameIndep_iff {true}

example := pumpkin_hasLinkMinor_twoLoops.some.toContractEmbedding.isLiftIndep_iff {true}

example := pumpkin_hasLinkMinor_twoLoops.some.toContractEmbedding.isFrameIndep_congr
  pumpkin_hasLinkMinor_twoLoops.some.toContractEmbedding (fun _ _ => Iff.rfl) {true}

example := pumpkin_hasLinkMinor_twoLoops.some.toContractEmbedding.isLiftIndep_congr
  pumpkin_hasLinkMinor_twoLoops.some.toContractEmbedding (fun _ _ => Iff.rfl) {true}

end Transformer.GainMatroid
