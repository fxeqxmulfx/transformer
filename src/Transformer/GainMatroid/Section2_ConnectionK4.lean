module

public import Transformer.GainMatroid.Section2_DeltaConnection
public import Transformer.GainMatroid.Section2_K4Flats

/-!
# The parallel connection restricts to `M(K₄)`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. In a triangle `{a, b, c}` of `M` with
distinct elements, a set of `k` elements has rank `min k 2` (`natRk_image_triple`). So on the
triangle and the three new elements, the rank `connRk` of `deltaConnection` is the rank in
`M(K₄)`: the restriction of the generalized parallel connection along `deltaEmbed` is `M(K₄)` with
the labels of §2.4 (`deltaConnection_comap_deltaEmbed`), by comparing `k4Indep` with this rank on
all `64` sets.

Flats pull back along any map (`isFlat_comap`), so a flat of `deltaConnection` meets the elements of
`M` in a flat of `M` and the elements of `M(K₄)` in a flat of `M(K₄)`.
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {E : Type*} {M : Matroid E} {a b c : E}

/-- Flats pull back to flats along a map. -/
theorem isFlat_comap {α β : Type*} {P : Matroid β} {S : Set β} (hS : P.IsFlat S) (f : α → β) :
    (P.comap f).IsFlat (f ⁻¹' S) := by
  rw [Matroid.isFlat_iff_closure_eq, Matroid.comap_closure_eq]
  refine Set.Subset.antisymm (Set.preimage_mono ?_) fun x hx => ?_
  · exact (P.closure_subset_closure (Set.image_preimage_subset f S)).trans hS.closure.subset
  · exact P.subset_closure _ ((Set.image_preimage_subset f S).trans hS.subset_ground) ⟨x, hx, rfl⟩

/-- The labels of a triangle with distinct elements are injective. -/
theorem triple_injective (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    Function.Injective ![a, b, c] := by
  intro i j h
  fin_cases i <;> fin_cases j <;> simp_all [eq_comm]

/-- All three labels give the triangle. -/
theorem image_univ_triple :
    ![a, b, c] '' ↑(Finset.univ : Finset (Fin 3)) = ({a, b, c} : Set E) := by
  rw [Finset.coe_univ, Set.image_univ]
  ext e
  exact ⟨fun ⟨k, hk⟩ => hk ▸ triple_mem a b c k, exists_triple_eq⟩

variable [Finite E]

/-- In a triangle with distinct elements, a set of `k` of them has rank `min k 2`. -/
theorem natRk_image_triple (hT : M.IsCircuit {a, b, c}) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c)
    (A : Finset (Fin 3)) : natRk M (![a, b, c] '' ↑A) = min A.card 2 := by
  have hinj := triple_injective hab hac hbc
  by_cases hA : A = Finset.univ
  · subst hA
    have h3 : ({a, b, c} : Set E).ncard = 3 := Set.ncard_eq_three.2 ⟨a, b, c, hab, hac, hbc, rfl⟩
    have := natRk_add_one_of_isCircuit hT
    rw [image_univ_triple, Finset.card_univ, Fintype.card_fin]
    omega
  · have hsub : ![a, b, c] '' ↑A ⊂ ({a, b, c} : Set E) := by
      refine ⟨fun _ ⟨k, _, hk⟩ => hk ▸ triple_mem a b c k, fun h => hA ?_⟩
      ext k
      simpa using hinj.mem_set_image.1 (h (triple_mem a b c k))
    have hcard : A.card < 3 := by
      simpa using Finset.card_lt_card (Finset.ssubset_univ_iff.2 hA)
    rw [indep_iff_natRk.1 (hT.ssubset_indep hsub), Set.ncard_image_of_injective _ hinj,
      Set.ncard_coe_finset]
    omega

set_option maxRecDepth 10000 in
private theorem k4Indep_iff_rank : ∀ Y : Finset (Fin 3 ⊕ Fin 3), k4Indep Y ↔
    min ((Finset.univ.filter fun i => Sum.inl i ∈ Y) ∪
      spanIdx (Finset.univ.filter fun i => Sum.inr i ∈ Y)).card 2 +
        nonemptyInd (Finset.univ.filter fun i => Sum.inr i ∈ Y) = Y.card := by
  decide +kernel

/-- **The generalized parallel connection restricts to `M(K₄)`** on the triangle and the new
elements, with the labels of §2.4. -/
theorem deltaConnection_comap_deltaEmbed (hT : M.IsCircuit {a, b, c}) (hab : a ≠ b) (hac : a ≠ c)
    (hbc : b ≠ c) : (deltaConnection M a b c hT).comap (deltaEmbed a b c) = k4Matroid := by
  refine Matroid.ext_indep ?_ fun Y _ => ?_
  · rw [Matroid.comap_ground_eq, deltaConnection_E, k4Matroid, matroidOfFinset_E]
    ext (i | i)
    · simpa [deltaEmbed, connGround] using hT.subset_ground (triple_mem a b c i)
    · simp [deltaEmbed, connGround]
  obtain ⟨Y, rfl⟩ := (Set.toFinite Y).exists_finset_coe
  have hinl : Sum.inl ⁻¹' (deltaEmbed a b c '' ↑Y) =
      ![a, b, c] '' ↑(Finset.univ.filter fun i => Sum.inl i ∈ Y) := by
    ext e
    simp [deltaEmbed]
  have hinr : newIdx (deltaEmbed a b c '' ↑Y) = Finset.univ.filter fun i => Sum.inr i ∈ Y := by
    ext i
    simp [mem_newIdx, deltaEmbed]
  have hsub : deltaEmbed a b c '' ↑Y ⊆ connGround M := by
    rintro _ ⟨i | i, -, rfl⟩
    · exact Or.inl ⟨_, hT.subset_ground (triple_mem a b c i), rfl⟩
    · exact Or.inr ⟨i, rfl⟩
  have hinj := deltaEmbed_injective hab hac hbc
  rw [Matroid.comap_indep_iff, deltaConnection_indep_iff, k4Matroid, matroidOfFinset_indep,
    k4Indep_iff_rank, connRk, hinl, hinr, ← Set.image_union, ← Finset.coe_union,
    natRk_image_triple hT hab hac hbc, Set.ncard_image_of_injective _ hinj, Set.ncard_coe_finset]
  exact ⟨fun h => h.1.2, fun h => ⟨⟨hsub, h⟩, hinj.injOn⟩⟩

end Transformer.GainMatroid
