module

public import Transformer.GainMatroid.Section2_MinorMaps

/-!
# Frame and lift independence after contracting a forest

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.1, contract a link `e` of `(G, B)`: the
balanced cycles of `G / e` are the cycles `C` with `C ∈ B` or `C ∪ e ∈ B`. Contracting a forest `K`
at once, a cycle `C` of `G / K` is balanced when `C ∪ K₀ ∈ B` for some `K₀ ⊆ K`.

A set `S` of edges outside `K` with `S ∪ K` frame-independent in `(G, B)` contains no such balanced
cycle of `G / K` and no two distinct cycles of `G / K` linked along `S`, and conversely
(`BiasedGraph.isFrameIndep_union_iff`); likewise `S ∪ K` is lift-independent exactly when `S` has no
such balanced cycle and nullity at most one in `G / K` (`BiasedGraph.isLiftIndep_union_iff`). Both
rest on the lift of a cycle of `G / K` (`OrientedGraph.IsCycle.exists_lift_contract`): in a set of
nullity at most one, a cycle of `G` is the lift of any cycle of `G / K` among its edges
(`OrientedGraph.eq_image_union_of_nullity_le_one`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E : Type*} {G : OrientedGraph V E} {K : Set E}

/-- The edges outside `K` of a subset of `S ∪ K` lie in `S`. -/
theorem preimage_val_subset {S : Set ↥Kᶜ} {C : Set E} (hC : C ⊆ Subtype.val '' S ∪ K) :
    Subtype.val ⁻¹' C ⊆ S := fun f hf =>
  (hC hf).elim (fun ⟨_, hg, hgf⟩ => Subtype.val_injective hgf ▸ hg) fun hK => absurd hK f.2

/-- A set of edges lies in its edges outside `K` and `K`. -/
theorem subset_image_preimage_val_union (C : Set E) :
    C ⊆ Subtype.val '' (Subtype.val ⁻¹' C : Set ↥Kᶜ) ∪ K := fun f hf => by
  by_cases hfK : f ∈ K
  · exact Or.inr hfK
  · exact Or.inl ⟨⟨f, hfK⟩, hf, rfl⟩

/-- A set of edges between `C` and `C ∪ K` has `C` as its edges outside `K`. -/
theorem eq_preimage_of_subset_union {C : Set ↥Kᶜ} {L : Set E} (h₁ : Subtype.val '' C ⊆ L)
    (h₂ : L ⊆ Subtype.val '' C ∪ K) : C = Subtype.val ⁻¹' L :=
  Set.Subset.antisymm (Set.image_subset_iff.1 h₁) (preimage_val_subset h₂)

/-- If the vertices of `Y ⊆ S ∪ K` are linked along `S ∪ K`, then the vertices of `G / K` met by
edges of `Y` outside `K` are linked along `S`. -/
theorem reachableIn_contract_of_forall {S : Set ↥Kᶜ} {Y : Set E}
    (hY : ∀ a ∈ G.verts Y, ∀ b ∈ G.verts Y, G.ReachableIn (Subtype.val '' S ∪ K) a b)
    {C D : Set ↥Kᶜ} (hC : Subtype.val '' C ⊆ Y) (hD : Subtype.val '' D ⊆ Y) {u w : _}
    (hu : u ∈ (G.contract K).verts C) (hw : w ∈ (G.contract K).verts D) :
    (G.contract K).ReachableIn S u w := by
  rw [verts_contract] at hu hw
  obtain ⟨a, ha, rfl⟩ := hu
  obtain ⟨b, hb, rfl⟩ := hw
  exact ReachableIn.contract (hY a (verts_mono hC ha) b (verts_mono hD hb))

variable [Finite V] [Finite E]

/-- In a set `X ∪ K` of nullity at most one, a cycle `C` of `G` is the lift of any cycle `C'` of
`G / K` in `X`: its edges are those of `C'` and some of `K`. -/
theorem eq_image_union_of_nullity_le_one (hK : G.IsForest K) {X C' : Set ↥Kᶜ} {C : Set E}
    (h : G.nullity (Subtype.val '' X ∪ K) ≤ 1) (hC : G.IsCycle C)
    (hCX : C ⊆ Subtype.val '' X ∪ K) (hC' : (G.contract K).IsCycle C') (hC'X : C' ⊆ X) :
    Subtype.val '' C' ∪ (C ∩ K) = C := by
  obtain ⟨L, hL, hC'L, hLC'⟩ := hC'.exists_lift_contract hK
  have hLX : L ⊆ Subtype.val '' X ∪ K :=
    hLC'.trans (Set.union_subset_union (Set.image_mono hC'X) subset_rfl)
  have hLC := eq_of_nullity_le_one h hL hC hLX hCX
  rw [hLC] at hC'L hLC'
  exact Set.Subset.antisymm (Set.union_subset hC'L Set.inter_subset_left) fun f hf =>
    (hLC' hf).elim Or.inl fun hfK => Or.inr ⟨hf, hfK⟩

/-- The edges outside `K` of a set of positive nullity in `G` have positive nullity in `G / K`. -/
theorem nullity_preimage_pos (hK : G.IsForest K) {C : Set E} (h : 0 < G.nullity C) :
    0 < (G.contract K).nullity (Subtype.val ⁻¹' C) := by
  rw [nullity_contract hK]
  exact h.trans_le (nullity_mono (subset_image_preimage_val_union C))

end OrientedGraph

namespace BiasedGraph

open OrientedGraph

variable {V E : Type*} {Ω : BiasedGraph V E} {K : Set E} [Finite V] [Finite E]

/-- **Lift independence after contracting a forest `K`.** `S ∪ K` is lift-independent exactly when
`S` contains no cycle `C` of `G / K` with `C ∪ K₀ ∈ B` for some `K₀ ⊆ K`, and has nullity at most
one in `G / K`. -/
theorem isLiftIndep_union_iff (hK : Ω.graph.IsForest K) {S : Set ↥Kᶜ} :
    Ω.IsLiftIndep (Subtype.val '' S ∪ K) ↔
      (∀ C ⊆ S, (Ω.graph.contract K).IsCycle C →
        ¬∃ K₀ ⊆ K, Subtype.val '' C ∪ K₀ ∈ Ω.balanced) ∧
      (Ω.graph.contract K).nullity S ≤ 1 := by
  rw [isLiftIndep_iff, nullity_contract hK]
  refine and_congr_left fun hle => ⟨fun hB C hCS hC ⟨K₀, hK₀, hCB⟩ =>
    hB _ (Set.union_subset_union (Set.image_mono hCS) hK₀) (Ω.isCycle_of_mem _ hCB) hCB,
    fun hB C hCT hC hCB => ?_⟩
  obtain ⟨C', hC'X, hC'⟩ := exists_isCycle_of_nullity_pos
    (nullity_preimage_pos (C := C) hK (by rw [hC.nullity_eq_one]; exact zero_lt_one))
  have hX := preimage_val_subset hCT
  refine hB C' (hC'X.trans hX) hC' ⟨C ∩ K, Set.inter_subset_right, ?_⟩
  rwa [eq_image_union_of_nullity_le_one hK hle hC hCT hC' (hC'X.trans hX)]

/-- **Frame independence after contracting a forest `K`.** `S ∪ K` is frame-independent exactly
when `S` contains no cycle `C` of `G / K` with `C ∪ K₀ ∈ B` for some `K₀ ⊆ K`, and no two distinct
cycles of `G / K` linked along `S`. -/
theorem isFrameIndep_union_iff (hK : Ω.graph.IsForest K) {S : Set ↥Kᶜ} :
    Ω.IsFrameIndep (Subtype.val '' S ∪ K) ↔
      (∀ C ⊆ S, (Ω.graph.contract K).IsCycle C →
        ¬∃ K₀ ⊆ K, Subtype.val '' C ∪ K₀ ∈ Ω.balanced) ∧
      ∀ C D, (Ω.graph.contract K).IsCycle C → (Ω.graph.contract K).IsCycle D → C ⊆ S → D ⊆ S →
        ∀ u ∈ (Ω.graph.contract K).verts C, ∀ w ∈ (Ω.graph.contract K).verts D,
          (Ω.graph.contract K).ReachableIn S u w → C = D := by
  rw [isFrameIndep_iff]
  constructor
  · rintro ⟨hB, hl⟩
    refine ⟨fun C hCS hC ⟨K₀, hK₀, hCB⟩ =>
      hB _ (Set.union_subset_union (Set.image_mono hCS) hK₀) (Ω.isCycle_of_mem _ hCB) hCB,
      fun C D hC hD hCS hDS u hu w hw huw => ?_⟩
    obtain ⟨LC, hLC, hCLC, hLCC⟩ := hC.exists_lift_contract hK
    obtain ⟨LD, hLD, hDLD, hLDD⟩ := hD.exists_lift_contract hK
    rw [verts_contract] at hu hw
    obtain ⟨x, hx, rfl⟩ := hu
    obtain ⟨y, hy, rfl⟩ := hw
    have hLCD := hl LC LD hLC hLD
      (hLCC.trans (Set.union_subset_union (Set.image_mono hCS) subset_rfl))
      (hLDD.trans (Set.union_subset_union (Set.image_mono hDS) subset_rfl))
      x (verts_mono hCLC hx) y (verts_mono hDLD hy) (ReachableIn.of_contract huw)
    rw [eq_preimage_of_subset_union hCLC hLCC, eq_preimage_of_subset_union hDLD hLDD, hLCD]
  · rintro ⟨hB, hl⟩
    -- the edges outside `K` of a set with linked vertices hold no two distinct cycles of `G / K`
    have htwo : ∀ Y ⊆ Subtype.val '' S ∪ K, (∀ a ∈ Ω.graph.verts Y, ∀ b ∈ Ω.graph.verts Y,
        Ω.graph.ReachableIn (Subtype.val '' S ∪ K) a b) →
        ¬2 ≤ (Ω.graph.contract K).nullity (Subtype.val ⁻¹' Y) := fun Y hYT hY h2 => by
      obtain ⟨C₁, C₂, hC₁, hC₂, hne, hC₁Y, hC₂Y⟩ := exists_two_cycles h2
      obtain ⟨f₁, hf₁⟩ := hC₁.nonempty
      obtain ⟨f₂, hf₂⟩ := hC₂.nonempty
      have hYS := preimage_val_subset hYT
      have hu : _ ∈ (Ω.graph.contract K).verts C₁ := mem_verts.2 ⟨f₁, hf₁, Or.inl rfl⟩
      have hw : _ ∈ (Ω.graph.contract K).verts C₂ := mem_verts.2 ⟨f₂, hf₂, Or.inl rfl⟩
      exact hne (hl C₁ C₂ hC₁ hC₂ (hC₁Y.trans hYS) (hC₂Y.trans hYS) _ hu _ hw
        (reachableIn_contract_of_forall hY (Set.image_subset_iff.2 hC₁Y)
          (Set.image_subset_iff.2 hC₂Y) hu hw))
    refine ⟨fun C hCT hC hCB => ?_, fun C D hC hD hCT hDT x hx y hy hxy => ?_⟩
    · have hle : Ω.graph.nullity (Subtype.val '' (Subtype.val ⁻¹' C : Set ↥Kᶜ) ∪ K) ≤ 1 := by
        have := htwo C hCT fun a ha b hb => (hC.reachableIn ha hb).mono hCT
        rw [← nullity_contract hK]
        omega
      obtain ⟨C', hC'X, hC'⟩ := exists_isCycle_of_nullity_pos
        (nullity_preimage_pos (C := C) hK (by rw [hC.nullity_eq_one]; exact zero_lt_one))
      refine hB C' (hC'X.trans (preimage_val_subset hCT)) hC' ⟨C ∩ K, Set.inter_subset_right, ?_⟩
      rwa [eq_image_union_of_nullity_le_one hK hle hC (subset_image_preimage_val_union C) hC' hC'X]
    · by_contra hCD
      have hto : ∀ z ∈ Ω.graph.verts (C ∪ D),
          Ω.graph.ReachableIn (Subtype.val '' S ∪ K) z x := fun z hz => by
        obtain ⟨f, hf, hfz⟩ := mem_verts.1 hz
        rcases hf with hfC | hfD
        · exact (hC.reachableIn (mem_verts.2 ⟨f, hfC, hfz⟩) hx).mono hCT
        · exact ((hD.reachableIn (mem_verts.2 ⟨f, hfD, hfz⟩) hy).mono hDT).trans hxy.symm
      refine htwo (C ∪ D) (Set.union_subset hCT hDT)
        (fun a ha b hb => (hto a ha).trans (hto b hb).symm) ?_
      rw [nullity_contract hK]
      exact (two_le_nullity hC hD hCD Set.subset_union_left Set.subset_union_right).trans
        (nullity_mono (subset_image_preimage_val_union (C ∪ D)))

end BiasedGraph

open OrientedGraph BiasedGraph

example := preimage_val_subset (S := ∅) (K := ({0} : Set (Fin 3))) (Set.empty_subset _)

example : ∃ L : Set (Fin 3), {pumpkinContractOne} = Subtype.val ⁻¹' L := by
  obtain ⟨L, -, h₁, h₂⟩ := pumpkin_contract_isCycle.exists_lift_contract pumpkin_isForest
  exact ⟨L, eq_preimage_of_subset_union h₁ h₂⟩

example : (pumpkin.contract {0}).ReachableIn {pumpkinContractOne}
    ((pumpkin.contract {0}).tail pumpkinContractOne)
    ((pumpkin.contract {0}).tail pumpkinContractOne) := by
  obtain ⟨L, hL, h₁, h₂⟩ := pumpkin_contract_isCycle.exists_lift_contract pumpkin_isForest
  exact reachableIn_contract_of_forall (fun a ha b hb => (hL.reachableIn ha hb).mono h₂) h₁ h₁
    (mem_verts.2 ⟨_, rfl, Or.inl rfl⟩) (mem_verts.2 ⟨_, rfl, Or.inl rfl⟩)

example : ∃ L : Set (Fin 3), Subtype.val '' {pumpkinContractOne} ∪ (L ∩ {0}) = L := by
  obtain ⟨L, hL, -, h₂⟩ := pumpkin_contract_isCycle.exists_lift_contract pumpkin_isForest
  refine ⟨L, eq_image_union_of_nullity_le_one pumpkin_isForest ?_ hL h₂ pumpkin_contract_isCycle
    subset_rfl⟩
  rw [← nullity_contract pumpkin_isForest, pumpkin_contract_isCycle.nullity_eq_one]

example := nullity_preimage_pos (G := pumpkin) pumpkin_isForest (C := {0, 1})
  (by rw [pumpkin_pair_isCycle.nullity_eq_one]; exact zero_lt_one)

example := isLiftIndep_union_iff (Ω := pumpkin.gainBiased (1 : Fin 3 → ℚˣ)) pumpkin_isForest
  (S := ∅)

example := isFrameIndep_union_iff (Ω := pumpkin.gainBiased (1 : Fin 3 → ℚˣ)) pumpkin_isForest
  (S := ∅)

end Transformer.GainMatroid
