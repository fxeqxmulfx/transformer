module

public import Transformer.GainMatroid.Section2_FrameContractGraph
public import Transformer.GainMatroid.Section2_FrameMatroid
public import Transformer.GainMatroid.Section2_LiftContraction

/-!
# Frame independence next to a joint and in a deletion

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2, contract a joint `e` of `(G, B)` in the frame
matroid `F(G, B)`. The independent sets of a frame matroid are the sets containing no frame circuit
(`BiasedGraph.IsFrameMatroid.indep_iff`), and a joint is one of them
(`BiasedGraph.IsJoint.isFrameIndep_singleton`). A loop adds no reachability
(`OrientedGraph.reachableIn_insert_of_loop`), so adding a joint `e` to a set `S` avoiding it keeps
`S` independent exactly when `S` is independent and none of its cycles is linked along `S` to the
vertex of `e` (`BiasedGraph.IsJoint.isFrameIndep_insert_iff`).

Frame independence in a biased graph on `H \ D` reads off the cycles, vertices and reachability of
`H` (`BiasedGraph.isFrameIndep_iff_of_graph_eq_delete`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

namespace OrientedGraph

variable {V E : Type*} {G : OrientedGraph V E}

/-- A loop adds no reachability. -/
theorem reachableIn_insert_of_loop {S : Set E} {e : E} (he : G.tail e = G.head e) {u w : V} :
    G.ReachableIn (insert e S) u w ↔ G.ReachableIn S u w := by
  refine ⟨ReachableIn.of_forall fun f hf => ?_, ReachableIn.mono (Set.subset_insert e S)⟩
  rcases hf with rfl | hf
  · rw [he]
    exact reachableIn_refl _ _
  · exact reachableIn_of_mem hf

end OrientedGraph

namespace BiasedGraph

variable {V E : Type*} {Ω : BiasedGraph V E} {M : Matroid E}

/-- The independent sets of a frame matroid are the sets containing no frame circuit. -/
theorem IsFrameMatroid.indep_iff (hM : Ω.IsFrameMatroid M) {I : Set E} :
    M.Indep I ↔ Ω.IsFrameIndep I := by
  rw [Matroid.indep_iff_forall_subset_not_isCircuit (by rw [hM.1]; exact Set.subset_univ I)]
  exact forall₂_congr fun X _ => not_congr (hM.2 X)

/-- A cycle through a joint is that joint. -/
theorem IsJoint.eq_singleton {e : E} (he : Ω.IsJoint e) {C : Set E} (hC : Ω.graph.IsCycle C)
    (heC : e ∈ C) : C = {e} :=
  hC.eq_singleton_of_loop heC he.1

/-- A joint is independent in the frame matroid. -/
theorem IsJoint.isFrameIndep_singleton {e : E} (he : Ω.IsJoint e) : Ω.IsFrameIndep {e} := by
  have hsub : ∀ C ⊆ ({e} : Set E), Ω.graph.IsCycle C → C = {e} := fun C hCe hC => by
    obtain ⟨f, hf⟩ := hC.nonempty
    have hfe : f = e := hCe hf
    subst hfe
    exact he.eq_singleton hC hf
  exact isFrameIndep_iff.2 ⟨fun C hCe hC hCB => he.2 (hsub C hCe hC ▸ hCB),
    fun C D hC hD hCe hDe _ _ _ _ _ => (hsub C hCe hC).trans (hsub D hDe hD).symm⟩

/-- Adding a joint `e` to a set `S` avoiding it keeps `S` frame-independent exactly when `S` is and
no cycle of `S` is linked along `S` to the vertex of `e`. -/
theorem IsJoint.isFrameIndep_insert_iff {e : E} {S : Set E} (he : Ω.IsJoint e) (heS : e ∉ S) :
    Ω.IsFrameIndep (insert e S) ↔
      Ω.IsFrameIndep S ∧ ¬Ω.graph.ReachesCycle S (Ω.graph.tail e) := by
  have hCe : ∀ C ⊆ insert e S, e ∉ C → C ⊆ S := fun C hCI heC f hf =>
    (hCI hf).resolve_left fun h => heC (h ▸ hf)
  have hve : ∀ x ∈ Ω.graph.verts {e}, x = Ω.graph.tail e := fun x hx => by
    obtain ⟨f, hf, h⟩ := mem_verts.1 hx
    rw [show f = e from hf] at h
    exact h.elim Eq.symm fun h => h.symm.trans he.1.symm
  constructor
  · intro hI
    refine ⟨fun X hX => hI X (hX.trans (Set.subset_insert e S)), ?_⟩
    rintro ⟨C, hCS, hC, u, hu, hue⟩
    have hCe' := (isFrameIndep_iff.1 hI).2 C {e} hC (isCycle_singleton_of_loop he.1)
      (hCS.trans (Set.subset_insert e S)) (Set.singleton_subset_iff.2 (Set.mem_insert e S)) u hu
      _ (mem_verts.2 ⟨e, rfl, Or.inl rfl⟩) (hue.mono (Set.subset_insert e S))
    exact heS (hCS (hCe' ▸ Set.mem_singleton e))
  · rintro ⟨hS, hRC⟩
    obtain ⟨hSB, hSl⟩ := isFrameIndep_iff.1 hS
    refine isFrameIndep_iff.2 ⟨fun C hCI hC hCB => ?_, fun C D hC hD hCI hDI u hu w hw huw => ?_⟩
    · by_cases heC : e ∈ C
      · exact he.2 (he.eq_singleton hC heC ▸ hCB)
      · exact hSB C (hCe C hCI heC) hC hCB
    · rw [reachableIn_insert_of_loop he.1] at huw
      by_cases heC : e ∈ C <;> by_cases heD : e ∈ D
      · rw [he.eq_singleton hC heC, he.eq_singleton hD heD]
      · rw [he.eq_singleton hC heC] at hu
        obtain rfl := hve u hu
        exact absurd ((hD.reachesCycle (hCe D hDI heD) hw).of_reachableIn huw.symm) hRC
      · rw [he.eq_singleton hD heD] at hw
        obtain rfl := hve w hw
        exact absurd ((hC.reachesCycle (hCe C hCI heC) hu).of_reachableIn huw) hRC
      · exact hSl C D hC hD (hCe C hCI heC) (hCe D hDI heD) u hu w hw huw

/-- Frame independence in a biased graph on `H \ D`, read off `H`. -/
theorem isFrameIndep_iff_of_graph_eq_delete {H : OrientedGraph V E} {D : Set E}
    {Ω₂ : BiasedGraph V ↥Dᶜ} (hH : Ω₂.graph = H.delete D) {I : Set ↥Dᶜ} :
    Ω₂.IsFrameIndep I ↔
      (∀ C ⊆ Subtype.val '' I, H.IsCycle C → Subtype.val ⁻¹' C ∉ Ω₂.balanced) ∧
      ∀ C C', H.IsCycle C → H.IsCycle C' → C ⊆ Subtype.val '' I → C' ⊆ Subtype.val '' I →
        ∀ u ∈ H.verts C, ∀ w ∈ H.verts C', H.ReachableIn (Subtype.val '' I) u w → C = C' := by
  have hr : ∀ C ⊆ Subtype.val '' I, Subtype.val '' (Subtype.val ⁻¹' C) = C := fun C hC =>
    Set.image_preimage_eq_of_subset (hC.trans (Set.image_subset_range _ _))
  have hpre : ∀ C ⊆ Subtype.val '' I, Subtype.val ⁻¹' C ⊆ I := fun C hC f hf =>
    Subtype.val_injective.mem_set_image.1 (hC hf)
  have hcyc : ∀ C ⊆ Subtype.val '' I, H.IsCycle C → (H.delete D).IsCycle (Subtype.val ⁻¹' C) :=
    fun C hC h => by
      rw [isCycle_delete_iff, hr C hC]
      exact h
  rw [isFrameIndep_iff, hH]
  constructor
  · rintro ⟨hB, hl⟩
    refine ⟨fun C hCI hC => hB _ (hpre C hCI) (hcyc C hCI hC),
      fun C C' hC hC' hCI hC'I u hu w hw huw => ?_⟩
    rw [← hr C hCI, ← hr C' hC'I]
    refine congrArg _ (hl _ _ (hcyc C hCI hC) (hcyc C' hC'I hC') (hpre C hCI) (hpre C' hC'I) u
      ?_ w ?_ (reachableIn_delete_iff.2 huw))
    · rwa [verts_delete, hr C hCI]
    · rwa [verts_delete, hr C' hC'I]
  · rintro ⟨hB, hl⟩
    refine ⟨fun C hCI hC hCB => hB _ (Set.image_mono hCI) (isCycle_delete_iff.1 hC) ?_,
      fun C C' hC hC' hCI hC'I u hu w hw huw => ?_⟩
    · rwa [Set.preimage_image_eq _ Subtype.val_injective]
    rw [verts_delete] at hu hw
    exact Subtype.val_injective.image_injective (hl _ _ (isCycle_delete_iff.1 hC)
      (isCycle_delete_iff.1 hC') (Set.image_mono hCI) (Set.image_mono hC'I) u hu w hw
      (reachableIn_delete_iff.1 huw))

end BiasedGraph

open BiasedGraph

example := reachableIn_insert_of_loop (G := twoLoops) (S := ∅) (e := true) rfl (u := ()) (w := ())

example := (Classical.choose_spec triangleBiased.exists_isFrameMatroid).indep_iff (I := ∅)

example := twoLoops_isJoint_true.eq_singleton (isCycle_singleton_of_loop rfl)
  (Set.mem_singleton true)

example := twoLoops_isJoint_true.isFrameIndep_singleton

example := twoLoops_isJoint_true.isFrameIndep_insert_iff (S := {false}) (by simp)

example := isFrameIndep_iff_of_graph_eq_delete (Ω₂ := twoLoopsContract)
  (H := twoLoops.frameContract ()) (D := {true}) rfl (I := ∅)

end Transformer.GainMatroid
