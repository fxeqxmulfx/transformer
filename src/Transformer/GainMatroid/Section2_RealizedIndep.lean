module

public import Transformer.GainMatroid.Section2_RealizedCycles

/-!
# Frame and lift independence in a realization

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.3: a gain function `γ` realizes `(G, B)` when the
balanced cycles are those of gain one. Let `T ⊆ I` be a forest linking the ends of every edge of
`I`, with gain one on `T`. Contracting `T` makes the edges of `I \ T` loops, and such a loop forms a
balanced cycle with edges of `T` exactly when its gain is one
(`BiasedGraph.Realizes.forall_not_balanced_iff`). Hence `I` is frame-independent exactly when no
edge of `I \ T` has gain one and no two edges of `I \ T` meet one component of `T`
(`BiasedGraph.Realizes.isFrameIndep_iff_of_isForest`). For an additive gain function with gain zero
on `T`, `I` is lift-independent exactly when no edge of `I \ T` has gain zero and `I \ T` has at
most one edge (`BiasedGraph.AddRealizes.isLiftIndep_iff_of_isForest`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace BiasedGraph

open OrientedGraph

variable {V E Γ : Type*} [DecidableEq V] [Finite V] [Finite E] {Ω : BiasedGraph V E} {I T : Set E}

/-- With gain one on a forest `T` linking the ends of every edge of `I`, no edge of `I \ T`, a loop
of `G / T`, forms a balanced cycle with edges of `T` exactly when none has gain one. -/
@[to_additive
  /-- With additive gain zero on a forest `T` linking the ends of every edge of `I`, no edge of
  `I \ T` forms a balanced cycle with edges of `T` exactly when none has gain zero. -/]
theorem Realizes.forall_not_balanced_iff [Group Γ] {γ : E → Γ} (h : Ω.Realizes γ)
    (hT : Ω.graph.IsForest T)
    (hr : ∀ e ∈ I, Ω.graph.ReachableIn T (Ω.graph.tail e) (Ω.graph.head e))
    (hγ : ∀ e ∈ T, γ e = 1) :
    (∀ C ⊆ (Subtype.val ⁻¹' I : Set ↥Tᶜ), (Ω.graph.contract T).IsCycle C →
        ¬∃ K₀ ⊆ T, Subtype.val '' C ∪ K₀ ∈ Ω.balanced) ↔ ∀ e ∈ I \ T, γ e ≠ 1 := by
  have hloop : ∀ e : ↥Tᶜ, (e : E) ∈ I →
      (Ω.graph.contract T).tail e = (Ω.graph.contract T).head e := fun e he =>
    Quotient.sound (hr e he)
  rw [← h]
  constructor
  · intro hB e he he1
    have hC := isCycle_singleton_of_loop (hloop ⟨e, he.2⟩ he.1)
    obtain ⟨L, hL, hCL, hLC⟩ := hC.exists_lift_contract hT
    refine hB _ (Set.singleton_subset_iff.2 he.1) hC ⟨L ∩ T, Set.inter_subset_right, ?_⟩
    rw [Set.Subset.antisymm (Set.union_subset hCL Set.inter_subset_left) fun f hf =>
      (hLC hf).elim Or.inl fun hfT => Or.inr ⟨hf, hfT⟩]
    refine mem_balancedCycles_of_forall_eq_one hL fun f hf => ?_
    rcases hLC hf with ⟨g, hg, rfl⟩ | hfT
    · rw [Set.mem_singleton_iff.1 hg]
      exact he1
    · exact hγ f hfT
  · rintro hB C hCI hC ⟨K₀, hK₀, hCB⟩
    obtain ⟨e, he⟩ := hC.nonempty
    rw [hC.eq_singleton_of_loop he (hloop e (hCI he)), Set.image_singleton] at hCB
    exact hB e ⟨hCI he, e.2⟩ (eq_one_of_mem_balancedCycles hCB (Or.inl rfl) fun f hf hfe =>
      hγ f (hK₀ (hf.resolve_left hfe)))

/-- **Frame independence in a realization.** Let `T ⊆ I` be a forest linking the ends of every
edge of `I`, with gain one on `T`. Then `I` is frame-independent exactly when no edge of `I \ T`
has gain one and no two edges of `I \ T` meet one component of `T`. -/
theorem Realizes.isFrameIndep_iff_of_isForest [Group Γ] {γ : E → Γ} (h : Ω.Realizes γ)
    (hTI : T ⊆ I) (hT : Ω.graph.IsForest T)
    (hr : ∀ e ∈ I, Ω.graph.ReachableIn T (Ω.graph.tail e) (Ω.graph.head e))
    (hγ : ∀ e ∈ T, γ e = 1) :
    Ω.IsFrameIndep I ↔ (∀ e ∈ I \ T, γ e ≠ 1) ∧
      ∀ e ∈ I \ T, ∀ f ∈ I \ T,
        Ω.graph.ReachableIn T (Ω.graph.tail e) (Ω.graph.tail f) → e = f := by
  have key := isFrameIndep_union_iff hT (S := Subtype.val ⁻¹' I)
  rw [image_preimage_val_union_eq hTI] at key
  rw [key, h.forall_not_balanced_iff hT hr hγ]
  refine and_congr_right fun _ => ?_
  have hloop : ∀ e : ↥Tᶜ, (e : E) ∈ I →
      (Ω.graph.contract T).tail e = (Ω.graph.contract T).head e := fun e he =>
    Quotient.sound (hr e he)
  constructor
  · intro hl e he f hf hef
    have hu : (Ω.graph.contract T).tail ⟨e, he.2⟩ = (Ω.graph.contract T).tail ⟨f, hf.2⟩ :=
      Quotient.sound hef
    refine congrArg Subtype.val (Set.singleton_injective (hl _ _
      (isCycle_singleton_of_loop (hloop ⟨e, he.2⟩ he.1))
      (isCycle_singleton_of_loop (hloop ⟨f, hf.2⟩ hf.1)) (Set.singleton_subset_iff.2 he.1)
      (Set.singleton_subset_iff.2 hf.1) _ (mem_verts.2 ⟨_, rfl, Or.inl rfl⟩) _
      (mem_verts.2 ⟨_, rfl, Or.inl rfl⟩) ?_))
    rw [hu]
    exact reachableIn_refl _ _
  · intro hl C D hC hD hCI hDI u hu w hw huw
    obtain ⟨e, he⟩ := hC.nonempty
    obtain ⟨f, hf⟩ := hD.nonempty
    rw [hC.eq_singleton_of_loop he (hloop e (hCI he))] at hu ⊢
    rw [hD.eq_singleton_of_loop hf (hloop f (hDI hf))] at hw ⊢
    have hef : Quotient.mk (Ω.graph.reachableSetoid T) (Ω.graph.tail e) =
        Quotient.mk _ (Ω.graph.tail f) :=
      (eq_tail_of_mem_verts_loop (hloop e (hCI he)) hu).symm.trans
        ((huw.eq_of_forall_loop fun g hg => hloop g hg).trans
          (eq_tail_of_mem_verts_loop (hloop f (hDI hf)) hw))
    rw [Subtype.ext (hl e ⟨hCI he, e.2⟩ f ⟨hDI hf, f.2⟩ (Quotient.exact hef))]

/-- **Lift independence in a realization.** Let `T ⊆ I` be a forest linking the ends of every edge
of `I`, with additive gain zero on `T`. Then `I` is lift-independent exactly when no edge of
`I \ T` has gain zero and `I \ T` has at most one edge. -/
theorem AddRealizes.isLiftIndep_iff_of_isForest [AddGroup Γ] {γ : E → Γ} (h : Ω.AddRealizes γ)
    (hTI : T ⊆ I) (hT : Ω.graph.IsForest T)
    (hr : ∀ e ∈ I, Ω.graph.ReachableIn T (Ω.graph.tail e) (Ω.graph.head e))
    (hγ : ∀ e ∈ T, γ e = 0) :
    Ω.IsLiftIndep I ↔ (∀ e ∈ I \ T, γ e ≠ 0) ∧ ∀ e ∈ I \ T, ∀ f ∈ I \ T, e = f := by
  have key := isLiftIndep_union_iff hT (S := Subtype.val ⁻¹' I)
  rw [image_preimage_val_union_eq hTI] at key
  rw [key, h.forall_not_balanced_iff hT hr hγ, nullity_le_one_iff]
  have hloop : ∀ e : ↥Tᶜ, (e : E) ∈ I →
      (Ω.graph.contract T).tail e = (Ω.graph.contract T).head e := fun e he =>
    Quotient.sound (hr e he)
  refine and_congr_right fun _ => ⟨fun hl e he f hf => congrArg Subtype.val
    (Set.singleton_injective (hl _ _ (isCycle_singleton_of_loop (hloop ⟨e, he.2⟩ he.1))
      (isCycle_singleton_of_loop (hloop ⟨f, hf.2⟩ hf.1)) (Set.singleton_subset_iff.2 he.1)
      (Set.singleton_subset_iff.2 hf.1))), fun hl C D hC hD hCI hDI => ?_⟩
  obtain ⟨e, he⟩ := hC.nonempty
  obtain ⟨f, hf⟩ := hD.nonempty
  rw [hC.eq_singleton_of_loop he (hloop e (hCI he)), hD.eq_singleton_of_loop hf (hloop f (hDI hf)),
    Subtype.ext (hl e ⟨hCI he, e.2⟩ f ⟨hDI hf, f.2⟩)]

end BiasedGraph

open OrientedGraph BiasedGraph

/-- The forest `{0}` of `pumpkin` links the ends of every edge. -/
theorem pumpkin_reachableIn (e : Fin 3) :
    pumpkin.ReachableIn {0} (pumpkin.tail e) (pumpkin.head e) :=
  show pumpkin.ReachableIn {0} 0 1 from reachableIn_of_mem (Set.mem_singleton 0)

example := Realizes.forall_not_balanced_iff (Ω := pumpkin.gainBiased (1 : Fin 3 → ℚˣ))
  (I := {0, 1}) rfl pumpkin_isForest (fun e _ => pumpkin_reachableIn e) fun _ _ => rfl

example := AddRealizes.forall_not_balanced_iff (Ω := pumpkin.addGainBiased (0 : Fin 3 → ℚ))
  (I := {0, 1}) rfl pumpkin_isForest (fun e _ => pumpkin_reachableIn e) fun _ _ => rfl

/-- With gain one on every edge of `pumpkin`, two parallel links are not frame-independent. -/
example : ¬(pumpkin.gainBiased (1 : Fin 3 → ℚˣ)).IsFrameIndep {0, 1} := fun h =>
  (((Realizes.isFrameIndep_iff_of_isForest (Ω := pumpkin.gainBiased (1 : Fin 3 → ℚˣ)) rfl
    (by simp) pumpkin_isForest (fun e _ => pumpkin_reachableIn e) fun _ _ => rfl).1 h).1 1
    ⟨Or.inr rfl, by simp⟩) rfl

/-- With gain zero on every edge of `pumpkin`, two parallel links are not lift-independent. -/
example : ¬(pumpkin.addGainBiased (0 : Fin 3 → ℚ)).IsLiftIndep {0, 1} := fun h =>
  (((AddRealizes.isLiftIndep_iff_of_isForest (Ω := pumpkin.addGainBiased (0 : Fin 3 → ℚ)) rfl
    (by simp) pumpkin_isForest (fun e _ => pumpkin_reachableIn e) fun _ _ => rfl).1 h).1 1
    ⟨Or.inr rfl, by simp⟩) rfl

end Transformer.GainMatroid
