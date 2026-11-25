module

public import Transformer.GainMatroid.Section2_ContractIndep

/-!
# Frame and lift independence in a link minor

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.1, §2.3, present a link minor `Ω'` of `Ω` as
`Ω / K \ D` up to isomorphism (`BiasedGraph.LinkMinorMap`). Its maps carry the cycles of `Ω'` to
those of `G / K` (`BiasedGraph.LinkMinorMap.isCycle_iff`), and a cycle of `Ω'` is balanced when its
image and some edges of `K` form a balanced cycle of `Ω`
(`BiasedGraph.LinkMinorMap.mem_balanced_iff`).

Hence a set `I` of edges of `Ω'` is frame-independent, or lift-independent, exactly when its image
together with `K` is so in `Ω` (`BiasedGraph.LinkMinorMap.isFrameIndep_iff`,
`BiasedGraph.LinkMinorMap.isLiftIndep_iff`). For the lift case, a set has nullity at most one
exactly when any two of its cycles are equal (`OrientedGraph.nullity_le_one_iff`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E : Type*} {G : OrientedGraph V E} [Finite V] [Finite E]

/-- A set has nullity at most one exactly when any two of its cycles are equal. -/
theorem nullity_le_one_iff {S : Set E} :
    G.nullity S ≤ 1 ↔ ∀ C D, G.IsCycle C → G.IsCycle D → C ⊆ S → D ⊆ S → C = D :=
  ⟨fun h _ _ hC hD hCS hDS => eq_of_nullity_le_one h hC hD hCS hDS, fun h => not_lt.1 fun hlt => by
    obtain ⟨C, D, hC, hD, hCD, hCS, hDS⟩ := exists_two_cycles hlt
    exact hCD (h C D hC hD hCS hDS)⟩

end OrientedGraph

namespace BiasedGraph

open OrientedGraph

variable {V E V' E' : Type*} {Ω : BiasedGraph V E} {Ω' : BiasedGraph V' E'}

namespace LinkMinorMap

variable (m : Ω.LinkMinorMap Ω')

/-- The cycles of a link minor are carried to the cycles of `G / K`. -/
theorem isCycle_iff {C : Set E'} :
    Ω'.graph.IsCycle C ↔ (Ω.graph.contract m.contractSet).IsCycle (m.edge '' C) :=
  ⟨fun h => h.map_both m.vertex_injective m.edge_injective m.joins,
    fun h => h.of_map_both m.vertex_injective m.edge_injective m.joins⟩

/-- A cycle of a link minor is balanced when its image in `G / K` and some edges of `K` form a
balanced cycle of `Ω`. -/
theorem mem_balanced_iff {C : Set E'} (hC : Ω'.graph.IsCycle C) :
    C ∈ Ω'.balanced ↔ ∃ K₀ ⊆ m.contractSet, Subtype.val '' (m.edge '' C) ∪ K₀ ∈ Ω.balanced := by
  rw [m.balanced_iff C hC, Set.image_image]

/-- A set of edges of `G / K` among the images of `I` is the image of its preimage, which lies in
`I`. -/
theorem image_preimage_of_subset {I : Set E'} {C : Set ↥m.contractSetᶜ} (hC : C ⊆ m.edge '' I) :
    m.edge '' (m.edge ⁻¹' C) = C ∧ m.edge ⁻¹' C ⊆ I :=
  ⟨Set.image_preimage_eq_of_subset (hC.trans (Set.image_subset_range _ _)), fun _ he =>
    m.edge_injective.mem_set_image.1 (hC he)⟩

/-- The balanced cycles of `Ω'` in `I` correspond to the cycles of `G / K` among the images of `I`
whose union with some edges of `K` is balanced in `Ω`. -/
theorem forall_notMem_balanced_iff {I : Set E'} :
    (∀ C ⊆ I, Ω'.graph.IsCycle C → C ∉ Ω'.balanced) ↔
      ∀ C ⊆ m.edge '' I, (Ω.graph.contract m.contractSet).IsCycle C →
        ¬∃ K₀ ⊆ m.contractSet, Subtype.val '' C ∪ K₀ ∈ Ω.balanced := by
  refine ⟨fun hB C hCI hC hK => ?_, fun hB C hCI hC hCB =>
    hB _ (Set.image_mono hCI) ((m.isCycle_iff).1 hC) ((m.mem_balanced_iff hC).1 hCB)⟩
  obtain ⟨hCe, hCI'⟩ := m.image_preimage_of_subset hCI
  rw [← hCe] at hC hK
  exact hB _ hCI' ((m.isCycle_iff).2 hC) ((m.mem_balanced_iff ((m.isCycle_iff).2 hC)).2 hK)

/-- **Frame independence in a link minor.** A set of edges of `Ω'` is frame-independent exactly
when its image together with `K` is frame-independent in `Ω`. -/
theorem isFrameIndep_iff [Finite V] [Finite E] {I : Set E'} :
    Ω'.IsFrameIndep I ↔ Ω.IsFrameIndep ((fun e' => (m.edge e' : E)) '' I ∪ m.contractSet) := by
  rw [← Set.image_image Subtype.val m.edge I, isFrameIndep_union_iff m.isForest,
    BiasedGraph.isFrameIndep_iff, m.forall_notMem_balanced_iff]
  refine and_congr_right fun _ => ⟨fun hl C D hC hD hCI hDI u hu w hw huw => ?_,
    fun hl C D hC hD hCI hDI u hu w hw huw => m.edge_injective.image_injective
      (hl _ _ ((m.isCycle_iff).1 hC) ((m.isCycle_iff).1 hD) (Set.image_mono hCI)
        (Set.image_mono hDI) _ (by rw [verts_map_both m.joins]; exact Set.mem_image_of_mem _ hu)
        _ (by rw [verts_map_both m.joins]; exact Set.mem_image_of_mem _ hw)
        ((reachableIn_map_both_iff m.vertex_injective m.edge_injective m.joins).1 huw))⟩
  obtain ⟨hCe, hCI'⟩ := m.image_preimage_of_subset hCI
  obtain ⟨hDe, hDI'⟩ := m.image_preimage_of_subset hDI
  rw [← hCe] at hC hu ⊢
  rw [← hDe] at hD hw ⊢
  rw [verts_map_both m.joins] at hu hw
  obtain ⟨u', hu', rfl⟩ := hu
  obtain ⟨w', hw', rfl⟩ := hw
  rw [hl _ _ ((m.isCycle_iff).2 hC) ((m.isCycle_iff).2 hD) hCI' hDI' u' hu' w' hw'
    ((reachableIn_map_both_iff m.vertex_injective m.edge_injective m.joins).2 huw)]

/-- **Lift independence in a link minor.** A set of edges of `Ω'` is lift-independent exactly when
its image together with `K` is lift-independent in `Ω`. -/
theorem isLiftIndep_iff [Finite V] [Finite E] {I : Set E'} :
    Ω'.IsLiftIndep I ↔ Ω.IsLiftIndep ((fun e' => (m.edge e' : E)) '' I ∪ m.contractSet) := by
  have : Finite V' := Finite.of_injective _ m.vertex_injective
  have : Finite E' := Finite.of_injective _ m.edge_injective
  rw [← Set.image_image Subtype.val m.edge I, isLiftIndep_union_iff m.isForest,
    BiasedGraph.isLiftIndep_iff, m.forall_notMem_balanced_iff, nullity_le_one_iff,
    nullity_le_one_iff]
  refine and_congr_right fun _ => ⟨fun h C D hC hD hCI hDI => ?_,
    fun h C D hC hD hCI hDI => m.edge_injective.image_injective (h _ _ ((m.isCycle_iff).1 hC)
      ((m.isCycle_iff).1 hD) (Set.image_mono hCI) (Set.image_mono hDI))⟩
  obtain ⟨hCe, hCI'⟩ := m.image_preimage_of_subset hCI
  obtain ⟨hDe, hDI'⟩ := m.image_preimage_of_subset hDI
  rw [← hCe, ← hDe] at ⊢
  rw [← hCe] at hC
  rw [← hDe] at hD
  rw [h _ _ ((m.isCycle_iff).2 hC) ((m.isCycle_iff).2 hD) hCI' hDI']

end LinkMinorMap

end BiasedGraph

open OrientedGraph BiasedGraph

example := nullity_le_one_iff (G := pumpkin) (S := {0})

example := (LinkMinorMap.refl triangleBiased).isCycle_iff (C := {0})

example := pumpkin_hasLinkMinor_twoLoops.some.mem_balanced_iff (C := {true})
  (isCycle_singleton_of_loop rfl)

example := pumpkin_hasLinkMinor_twoLoops.some.image_preimage_of_subset (I := ∅) subset_rfl

example := pumpkin_hasLinkMinor_twoLoops.some.forall_notMem_balanced_iff (I := {true})

example := pumpkin_hasLinkMinor_twoLoops.some.isFrameIndep_iff (I := {true})

example := pumpkin_hasLinkMinor_twoLoops.some.isLiftIndep_iff (I := {true})

end Transformer.GainMatroid
