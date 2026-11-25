module

public import Transformer.GainMatroid.Section2_DeltaConnectionFlats

/-!
# The rank of a Δ-Y exchange

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4: `Δ_X M` is the generalized parallel connection
of `M` and `M(K₄)` across a triangle `X = {a, b, c}`, with `X` deleted and the new elements named
`a, b, c` (`deltaRelabel`). Its rank does not depend on the labels: it is the rank in `M` of the
elements of a set off `X` together with the elements of `X` whose two others lie in the set
(`starSpan`), plus one if the set meets `X` (`deltaRk`, `connRk_image_deltaRelabel`).

Contracting `c` in `Δ_X M` leaves `M \ c` with `a` and `b` exchanged: for `I` avoiding `c`, the set
`I` with `a`, `b` exchanged and `c` added has rank `deltaRk` one more than the rank of `I` in `M`
(`deltaRk_insert_swap`). A set off `X` has the same rank in both (`deltaRk_of_disjoint`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open Set

variable {E : Type*} [DecidableEq E] {M : Matroid E} {a b c : E} {I X : Set E}

/-- The elements of `X` whose other elements of `X` lie in `I`. -/
def starSpan (X I : Set E) : Set E := {t | t ∈ X ∧ X \ {t} ⊆ I}

open Classical in
/-- **The rank of `Δ_X M`** (§2.4): the rank in `M` of the elements of `I` off `X` together with
`starSpan X I`, plus one if `I` meets `X`. -/
noncomputable def deltaRk (M : Matroid E) (X I : Set E) : ℕ :=
  natRk M (I \ X ∪ starSpan X I) + if (I ∩ X).Nonempty then 1 else 0

theorem deltaRelabel_eq_inl_iff {e f : E} :
    deltaRelabel a b c f = Sum.inl e ↔ f ∉ ({a, b, c} : Set E) ∧ f = e := by
  unfold deltaRelabel
  split_ifs <;> simp_all

theorem deltaRelabel_eq_inr_iff (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) {f : E} {k : Fin 3} :
    deltaRelabel a b c f = Sum.inr k ↔ f = ![a, b, c] k := by
  unfold deltaRelabel
  fin_cases k <;> split_ifs <;> simp_all [eq_comm]

/-- The rank of `deltaConnection` on a relabelled set is `deltaRk`. -/
theorem connRk_image_deltaRelabel (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) (I : Set E) :
    connRk M a b c (deltaRelabel a b c '' I) = deltaRk M {a, b, c} I := by
  have hinj := triple_injective hab hac hbc
  have hl : Sum.inl ⁻¹' (deltaRelabel a b c '' I) = I \ {a, b, c} := by
    ext e
    simp only [mem_preimage, mem_image, deltaRelabel_eq_inl_iff, mem_sdiff]
    exact ⟨fun ⟨f, hf, hfX, hfe⟩ => hfe ▸ ⟨hf, hfX⟩, fun ⟨he, heX⟩ => ⟨e, he, heX, rfl⟩⟩
  have hr (k : Fin 3) : k ∈ newIdx (deltaRelabel a b c '' I) ↔ ![a, b, c] k ∈ I := by
    rw [mem_newIdx]
    simp only [mem_image, deltaRelabel_eq_inr_iff hab hac hbc]
    exact ⟨fun ⟨f, hf, hfk⟩ => hfk ▸ hf, fun h => ⟨_, h, rfl⟩⟩
  have hspan : ![a, b, c] '' ↑(spanIdx (newIdx (deltaRelabel a b c '' I))) =
      starSpan {a, b, c} I := by
    ext t
    simp only [mem_image, Finset.mem_coe, spanIdx, Finset.mem_filter, Finset.mem_univ, true_and,
      hr]
    constructor
    · rintro ⟨k, hk, rfl⟩
      refine ⟨triple_mem a b c k, fun e ⟨he, hek⟩ => ?_⟩
      obtain ⟨i, rfl⟩ := exists_triple_eq he
      exact hk i fun hik => hek (by rw [hik]; rfl)
    · rintro ⟨ht, hsub⟩
      obtain ⟨k, rfl⟩ := exists_triple_eq ht
      exact ⟨k, fun i hik => hsub ⟨triple_mem a b c i, fun h => hik (hinj h)⟩, rfl⟩
  have hne : (newIdx (deltaRelabel a b c '' I)).Nonempty ↔ (I ∩ {a, b, c}).Nonempty := by
    constructor
    · rintro ⟨k, hk⟩
      exact ⟨_, (hr k).1 hk, triple_mem a b c k⟩
    · rintro ⟨e, he, heX⟩
      obtain ⟨k, rfl⟩ := exists_triple_eq heX
      exact ⟨k, (hr k).2 he⟩
  rw [connRk, deltaRk, hl, hspan, nonemptyInd]
  by_cases h : (I ∩ {a, b, c}).Nonempty
  · rw [ite_eq_left (hne.2 h), ite_eq_left h]
  · rw [ite_eq_right (mt hne.1 h), ite_eq_right h]

/-- Off the triangle, `deltaRk` is the rank in `M`. -/
theorem deltaRk_of_disjoint (hab : a ≠ b) (hI : Disjoint I {a, b, c}) :
    deltaRk M {a, b, c} I = natRk M I := by
  have hspan : starSpan {a, b, c} I = ∅ := by
    refine eq_empty_of_forall_notMem fun t ⟨ht, hsub⟩ => ?_
    by_cases hta : t = a
    · exact disjoint_left.1 hI (hsub ⟨Or.inr (Or.inl rfl), fun h => hab (hta ▸ h).symm⟩)
        (Or.inr (Or.inl rfl))
    · exact disjoint_left.1 hI (hsub ⟨Or.inl rfl, fun h => hta h.symm⟩) (Or.inl rfl)
  have hne : ¬(I ∩ {a, b, c}).Nonempty := by
    rw [not_nonempty_iff_eq_empty, ← disjoint_iff_inter_eq_empty]
    exact hI
  rw [deltaRk, hspan, union_empty, hI.sdiff_eq_left, ite_eq_right hne, add_zero]

variable [Finite E]

/-- **Contracting `c` in `Δ_X M`.** For `I` avoiding `c`, exchanging `a` and `b` in `I` and adding
`c` gives a set of rank `deltaRk` one more than the rank of `I` in `M`. -/
theorem deltaRk_insert_swap (hT : M.IsCircuit {a, b, c}) (hab : a ≠ b) (hac : a ≠ c)
    (hbc : b ≠ c) (hcI : c ∉ I) :
    deltaRk M {a, b, c} (insert c (Equiv.swap a b '' I)) = natRk M I + 1 := by
  set I' := insert c (Equiv.swap a b '' I)
  have hmem (e : E) : e ∈ Equiv.swap a b '' I ↔ Equiv.swap a b e ∈ I := by
    rw [mem_image_equiv, Equiv.symm_swap]
  have hoff : I' \ {a, b, c} = I \ {a, b, c} := by
    ext e
    simp only [I', mem_sdiff, mem_insert_iff, hmem, mem_singleton_iff, not_or]
    constructor
    · rintro ⟨rfl | he, hea, heb, hec⟩
      · exact absurd rfl hec
      · exact ⟨by rwa [Equiv.swap_apply_of_ne_of_ne hea heb] at he, hea, heb, hec⟩
    · rintro ⟨he, hea, heb, hec⟩
      exact ⟨Or.inr (by rwa [Equiv.swap_apply_of_ne_of_ne hea heb]), hea, heb, hec⟩
  have hsub : I ⊆ I' \ {a, b, c} ∪ starSpan {a, b, c} I' := by
    intro e he
    by_cases heX : e ∈ ({a, b, c} : Set E)
    · refine Or.inr ⟨heX, fun f ⟨hf, hfe⟩ => ?_⟩
      rcases heX with rfl | rfl | rfl
      · rcases hf with rfl | rfl | rfl
        · exact absurd rfl hfe
        · exact Or.inr ((hmem _).2 (by rwa [Equiv.swap_apply_right]))
        · exact Or.inl rfl
      · rcases hf with rfl | rfl | rfl
        · exact Or.inr ((hmem _).2 (by rwa [Equiv.swap_apply_left]))
        · exact absurd rfl hfe
        · exact Or.inl rfl
      · exact absurd he hcI
    · rw [hoff]
      exact Or.inl ⟨he, heX⟩
  have hcl : I' \ {a, b, c} ∪ starSpan {a, b, c} I' ⊆ I ∪ M.closure I := by
    rintro e (he | ⟨heX, hX⟩)
    · rw [hoff] at he
      exact Or.inl he.1
    · have ha : b ∈ I' → a ∈ I := fun h => by
        rcases h with h | h
        · exact absurd h hbc
        · rwa [hmem, Equiv.swap_apply_right] at h
      have hb : a ∈ I' → b ∈ I := fun h => by
        rcases h with h | h
        · exact absurd h hac
        · rwa [hmem, Equiv.swap_apply_left] at h
      rcases heX with rfl | rfl | rfl
      · exact Or.inl (ha (hX ⟨Or.inr (Or.inl rfl), hab.symm⟩))
      · exact Or.inl (hb (hX ⟨Or.inl rfl, hab⟩))
      · refine Or.inr (M.closure_subset_closure ?_ (hT.subset_closure_sdiff_singleton e
          (Or.inr (Or.inr rfl))))
        rintro f ⟨hf, hfe⟩
        rcases hf with rfl | rfl | rfl
        · exact ha (hX ⟨Or.inr (Or.inl rfl), hbc⟩)
        · exact hb (hX ⟨Or.inl rfl, hac⟩)
        · exact absurd rfl hfe
  have hrk : natRk M (I' \ {a, b, c} ∪ starSpan {a, b, c} I') = natRk M I :=
    le_antisymm ((natRk_mono M hcl).trans (natRk_union_of_subset_closure M subset_rfl).le)
      (natRk_mono M hsub)
  have hne : (I' ∩ {a, b, c}).Nonempty := ⟨c, mem_insert c _, Or.inr (Or.inr rfl)⟩
  rw [deltaRk, hrk, ite_eq_left hne]

end Transformer.GainMatroid
