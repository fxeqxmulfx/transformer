module

public import Transformer.GainMatroid.Section2_SpanningForests

/-!
# The exchange axiom of the frame matroid

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2, cite Zaslavsky for the frame matroid
`F(G, B)`. Its independent sets, the sets `I` with no balanced cycle in which no two distinct cycles
are linked (`BiasedGraph.isFrameIndep_iff`), satisfy the exchange axiom
(`BiasedGraph.IsFrameIndep.exists_insert`). Suppose no edge of a larger such set `J` can be added
to `I`. Split `I` into the edges `I_U` of its components without a cycle, a forest
(`OrientedGraph.isForest_acyclic`), and the edges `I_W` on the vertices `W` linked to a cycle.

* An edge of `J` with an end outside `W` has its ends linked along `I_U`, and every cycle it closes
  with `I` is balanced (`BiasedGraph.reachableIn_of_not_isFrameIndep_insert`,
  `BiasedGraph.mem_balanced_of_not_isFrameIndep_insert`). So these edges form a forest
  (`BiasedGraph.mem_balanced_of_fundamental`) with at least as many components as `I_U`, hence with
  at most as many edges.
* The other edges of `J` have both ends in `W`, so there are at most `|W|` of them
  (`OrientedGraph.ncard_le_ncard_verts`), and `|W| ≤ |I_W|` (`OrientedGraph.ncard_reachesCycle_le`).

Hence `|J| ≤ |I|`.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

variable {V E : Type*}

namespace OrientedGraph

variable {G : OrientedGraph V E}

/-- The edges of `I` whose tails are not linked to a cycle of `I` form a forest. -/
theorem isForest_acyclic (I : Set E) : G.IsForest {f ∈ I | ¬G.ReachesCycle I (G.tail f)} := by
  intro C hC hCI
  obtain ⟨g, hg⟩ := hC.nonempty
  exact (hCI hg).2 (hC.reachesCycle (fun f hf => (hCI hf).1) (mem_verts.2 ⟨g, hg, Or.inl rfl⟩))

end OrientedGraph

namespace BiasedGraph

variable {Ω : BiasedGraph V E}

/-- If adding an edge `e` to a frame-independent set `I` breaks independence and an end of `e` is
not linked to a cycle of `I`, then the ends of `e` are linked along `I`. -/
theorem reachableIn_of_not_isFrameIndep_insert {I : Set E} {e : E} (hI : Ω.IsFrameIndep I)
    (hblock : ¬Ω.IsFrameIndep (insert e I))
    (he : ¬Ω.graph.ReachesCycle I (Ω.graph.tail e) ∨ ¬Ω.graph.ReachesCycle I (Ω.graph.head e)) :
    Ω.graph.ReachableIn I (Ω.graph.tail e) (Ω.graph.head e) := by
  obtain ⟨hIB, hIl⟩ := isFrameIndep_iff.1 hI
  by_contra hr
  have hsub : ∀ C ⊆ insert e I, Ω.graph.IsCycle C → C ⊆ I := fun C hCI hC f hf =>
    (hCI hf).resolve_left fun h => hr (hC.reachableIn_of_subset_insert (h ▸ hf) hCI)
  refine hblock (isFrameIndep_iff.2 ⟨fun C hCI hC => hIB C (hsub C hCI hC) hC,
    fun C D hC hD hCI hDI u hu v hv huv => ?_⟩)
  have hCI' := hsub C hCI hC
  have hDI' := hsub D hDI hD
  rcases reachableIn_insert_iff.1 huv with h | ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
  · exact hIl C D hC hD hCI' hDI' u hu v hv h
  · exact he.elim (fun h => absurd ((hC.reachesCycle hCI' hu).of_reachableIn h₁) h)
      fun h => absurd ((hD.reachesCycle hDI' hv).of_reachableIn h₂.symm) h
  · exact he.elim (fun h => absurd ((hD.reachesCycle hDI' hv).of_reachableIn h₂.symm) h)
      fun h => absurd ((hC.reachesCycle hCI' hu).of_reachableIn h₁) h

variable [Finite V] [Finite E]

/-- If adding an edge `e` to a frame-independent set `I` breaks independence and an end of `e` is
not linked to a cycle of `I`, then every cycle through `e` inside `insert e I` is balanced. -/
theorem mem_balanced_of_not_isFrameIndep_insert {I : Set E} {e : E} (hI : Ω.IsFrameIndep I)
    (hblock : ¬Ω.IsFrameIndep (insert e I))
    (he : ¬Ω.graph.ReachesCycle I (Ω.graph.tail e) ∨ ¬Ω.graph.ReachesCycle I (Ω.graph.head e))
    {D : Set E} (hD : Ω.graph.IsCycle D) (heD : e ∈ D) (hDI : D ⊆ insert e I) :
    D ∈ Ω.balanced := by
  obtain ⟨hIB, hIl⟩ := isFrameIndep_iff.1 hI
  have hr := reachableIn_of_not_isFrameIndep_insert hI hblock he
  have ht : ¬Ω.graph.ReachesCycle I (Ω.graph.tail e) := fun h =>
    he.elim (fun h' => h' h) fun h' => h' (h.of_reachableIn hr)
  have hle : Ω.graph.nullity (insert e {f ∈ I | ¬Ω.graph.ReachesCycle I (Ω.graph.tail f)}) ≤ 1 := by
    have h₁ := nullity_insert_le (G := Ω.graph)
      {f ∈ I | ¬Ω.graph.ReachesCycle I (Ω.graph.tail f)} e
    have h₂ := nullity_eq_zero_iff.2 (isForest_acyclic (G := Ω.graph) I)
    omega
  -- a cycle through `e` inside `insert e I` avoids the components of `I` with a cycle
  have huniq : ∀ D', Ω.graph.IsCycle D' → e ∈ D' → D' ⊆ insert e I → D' = D := by
    have hsub : ∀ D', Ω.graph.IsCycle D' → e ∈ D' → D' ⊆ insert e I →
        D' ⊆ insert e {f ∈ I | ¬Ω.graph.ReachesCycle I (Ω.graph.tail f)} := by
      intro D' hD' heD' hD'I f hf
      rcases hD'I hf with h | hfI
      · exact Or.inl h
      · exact Or.inr ⟨hfI, fun h => ht (h.of_reachableIn
          (hD'.reachableIn_of_mem_insert heD' hD'I (mem_verts.2 ⟨f, hf, Or.inl rfl⟩)))⟩
    exact fun D' hD' heD' hD'I => eq_of_nullity_le_one hle hD' hD (hsub D' hD' heD' hD'I)
      (hsub D hD heD hDI)
  have hlink : ∀ C ⊆ I, Ω.graph.IsCycle C → ∀ D', Ω.graph.IsCycle D' → e ∈ D' →
      D' ⊆ insert e I → ∀ u ∈ Ω.graph.verts C, ∀ v ∈ Ω.graph.verts D',
        ¬Ω.graph.ReachableIn I u v :=
    fun C hCI hC D' hD' heD' hD'I u hu v hv huv => ht ((hC.reachesCycle hCI hu).of_reachableIn
      (huv.trans (hD'.reachableIn_of_mem_insert heD' hD'I hv)))
  have hoff : ∀ C ⊆ insert e I, e ∉ C → C ⊆ I := fun C hCI heC f hf =>
    (hCI hf).resolve_left fun h => heC (h ▸ hf)
  by_contra hDB
  refine hblock (isFrameIndep_iff.2 ⟨fun C hCI hC hCB => ?_,
    fun C₁ C₂ hC₁ hC₂ hC₁I hC₂I u hu v hv huv => ?_⟩)
  · by_cases heC : e ∈ C
    · exact hDB (huniq C hC heC hCI ▸ hCB)
    · exact hIB C (hoff C hCI heC) hC hCB
  · have huv' : Ω.graph.ReachableIn I u v := huv.of_forall fun f hf => by
      rcases hf with rfl | hf
      · exact hr
      · exact reachableIn_of_mem hf
    by_cases he₁ : e ∈ C₁ <;> by_cases he₂ : e ∈ C₂
    · exact (huniq C₁ hC₁ he₁ hC₁I).trans (huniq C₂ hC₂ he₂ hC₂I).symm
    · exact absurd huv'.symm
        (hlink C₂ (hoff C₂ hC₂I he₂) hC₂ C₁ hC₁ he₁ hC₁I v hv u hu)
    · exact absurd huv' (hlink C₁ (hoff C₁ hC₁I he₁) hC₁ C₂ hC₂ he₂ hC₂I u hu v hv)
    · exact hIl C₁ C₂ hC₁ hC₂ (hoff C₁ hC₁I he₁) (hoff C₂ hC₂I he₂) u hu v hv huv'

/-- **The exchange axiom** for the sets containing no frame circuit. -/
theorem IsFrameIndep.exists_insert {I J : Set E} (hI : Ω.IsFrameIndep I) (hJ : Ω.IsFrameIndep J)
    (hlt : I.ncard < J.ncard) : ∃ e ∈ J, e ∉ I ∧ Ω.IsFrameIndep (insert e I) := by
  by_contra! hno
  obtain ⟨hJB, hJl⟩ := isFrameIndep_iff.1 hJ
  let IU : Set E := {f ∈ I | ¬Ω.graph.ReachesCycle I (Ω.graph.tail f)}
  let JU : Set E := {e ∈ J | ¬Ω.graph.ReachesCycle I (Ω.graph.tail e) ∨
    ¬Ω.graph.ReachesCycle I (Ω.graph.head e)}
  have hIU : Ω.graph.IsForest IU := isForest_acyclic I
  have hmem : ∀ e ∈ JU, e ∈ I → e ∈ IU := fun e he heI => ⟨heI, fun h =>
    he.2.elim (fun h' => h' h) fun h' => h' (h.of_reachableIn (reachableIn_of_mem heI))⟩
  have hK1 : ∀ e ∈ JU, Ω.graph.ReachableIn IU (Ω.graph.tail e) (Ω.graph.head e) := by
    intro e he
    by_cases heI : e ∈ I
    · exact reachableIn_of_mem (hmem e he heI)
    have hr := reachableIn_of_not_isFrameIndep_insert hI (hno e he.1 heI) he.2
    have ht : ¬Ω.graph.ReachesCycle I (Ω.graph.tail e) := fun h =>
      he.2.elim (fun h' => h' h) fun h' => h' (h.of_reachableIn hr)
    exact hr.restrict (P := fun v => ¬Ω.graph.ReachesCycle I v)
      (fun f hf => not_congr (reachesCycle_tail_iff hf)) ht
  have hK2 : ∀ e ∈ JU, ∀ D, Ω.graph.IsCycle D → D ⊆ insert e IU → D ∈ Ω.balanced := by
    intro e he D hD hDI
    by_cases heI : e ∈ I
    · rw [Set.insert_eq_of_mem (hmem e he heI)] at hDI
      exact absurd hDI (hIU D hD)
    have heD : e ∈ D := by
      by_contra heD
      exact hIU D hD fun f hf => (hDI hf).resolve_left fun h => heD (h ▸ hf)
    exact mem_balanced_of_not_isFrameIndep_insert hI (hno e he.1 heI) he.2 hD heD
      (hDI.trans (Set.insert_subset_insert fun f hf => hf.1))
  have hJU : Ω.graph.IsForest JU := fun C hC hCJ => hJB C (fun f hf => (hCJ hf).1) hC
    (Ω.mem_balanced_of_fundamental hIU hK1 hK2 hC (hCJ.trans Set.subset_union_right))
  have hc := compCount_le_of_forall hK1
  have h₁ := (card_le_ncard_add_compCount (G := Ω.graph) IU).2.2 hIU
  have h₂ := (card_le_ncard_add_compCount (G := Ω.graph) JU).2.2 hJU
  have h₃ : (J \ JU).ncard ≤ {v | Ω.graph.ReachesCycle I v}.ncard := by
    refine (ncard_le_ncard_verts fun C D hC hD hCJ hDJ u hu v hv huv => hJl C D hC hD
      (hCJ.trans Set.sdiff_subset) (hDJ.trans Set.sdiff_subset) u hu v hv
      (huv.mono Set.sdiff_subset)).trans (Set.ncard_le_ncard fun w hw => ?_)
    obtain ⟨f, hf, hfw⟩ := mem_verts.1 hw
    have hRC : Ω.graph.ReachesCycle I (Ω.graph.tail f) ∧
        Ω.graph.ReachesCycle I (Ω.graph.head f) := by
      by_contra h
      exact hf.2 ⟨hf.1, not_and_or.1 h⟩
    rcases hfw with rfl | rfl
    · exact hRC.1
    · exact hRC.2
  have h₄ : {v | Ω.graph.ReachesCycle I v}.ncard ≤ (I \ IU).ncard := by
    have hIW : I \ IU = {f ∈ I | Ω.graph.ReachesCycle I (Ω.graph.tail f)} :=
      Set.ext fun f => ⟨fun h => ⟨h.1, not_not.1 fun hn => h.2 ⟨h.1, hn⟩⟩,
        fun h => ⟨h.1, fun h' => h'.2 h.2⟩⟩
    rw [hIW]
    refine (Set.ncard_le_ncard fun v hv => ?_).trans (ncard_reachesCycle_le (G := Ω.graph) _)
    obtain ⟨C, hCI, hC, u, hu, huv⟩ := hv
    exact ⟨C, fun f hf => ⟨hCI hf, hC.reachesCycle hCI (mem_verts.2 ⟨f, hf, Or.inl rfl⟩)⟩, hC, u,
      hu, huv.restrict (P := Ω.graph.ReachesCycle I) (fun f hf => reachesCycle_tail_iff hf)
        (hC.reachesCycle hCI hu)⟩
  have h₅ := Set.ncard_sdiff_add_ncard_of_subset (show JU ⊆ J from fun f hf => hf.1)
  have h₆ := Set.ncard_sdiff_add_ncard_of_subset (show IU ⊆ I from fun f hf => hf.1)
  omega

end BiasedGraph

end Transformer.GainMatroid
