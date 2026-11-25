module

public import Transformer.GainMatroid.Section2_ConnectionK4

/-!
# The flats of the generalized parallel connection

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, after Oxley, §11.4: the flats of the
generalized parallel connection of `M` and `M(K₄)` across a triangle `X = {a, b, c}` are the sets
whose elements of `M` form a flat of `M` and whose elements of `M(K₄)` form a flat of `M(K₄)`
(`isFlat_deltaConnection_iff`), as `IsDeltaMatroid` asks.

A flat pulls back along the restrictions to `M` and to `M(K₄)` (`isFlat_comap`). Conversely, let `S`
meet `M` in a flat and `M(K₄)` in a flat. Then the elements of `X` spanned by the new elements of
`S` lie in `S`, so `S` has rank that of its elements of `M`, plus one if it has a new element. An
old element outside `S` raises the rank in `M`. A new element outside `S` either is the first new
element, or spans with those of `S` an element of `X` outside `S`
(`k4_flat_facts`, decided on all `64` sets); either way it raises the rank.
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {E : Type*} {M : Matroid E} {a b c : E}

set_option maxRecDepth 10000 in
set_option synthInstance.maxSize 4096 in
private theorem k4_flat_facts : ∀ F : Finset (Fin 3 ⊕ Fin 3),
    (∀ C ∈ k4Triangles, ∀ e ∈ C, (∀ f ∈ C, f ≠ e → f ∈ F) → e ∈ F) →
      (∀ k ∈ spanIdx (Finset.univ.filter fun i => Sum.inr i ∈ F), Sum.inl k ∈ F) ∧
      ((Finset.univ.filter fun i => Sum.inr i ∈ F).Nonempty → ∀ i, Sum.inr i ∉ F →
        ∃ k ∈ spanIdx (insert i (Finset.univ.filter fun i => Sum.inr i ∈ F)), Sum.inl k ∉ F) := by
  decide +kernel

/-- The elements of the triangle spanned by the new elements of a flat of `M(K₄)` lie in it, and a
new element outside it spans with them an element of the triangle outside it. -/
theorem k4_isFlat_spanIdx {S : Set (E ⊕ Fin 3)} (hK : k4Matroid.IsFlat (deltaEmbed a b c ⁻¹' S)) :
    (∀ k ∈ spanIdx (newIdx S), ![a, b, c] k ∈ Sum.inl ⁻¹' S) ∧
      ((newIdx S).Nonempty → ∀ i ∉ newIdx S,
        ∃ k ∈ spanIdx (insert i (newIdx S)), ![a, b, c] k ∉ Sum.inl ⁻¹' S) := by
  obtain ⟨F, hF⟩ := (Set.toFinite (deltaEmbed a b c ⁻¹' S)).exists_finset_coe
  have hflat := (k4Matroid_isFlat_iff _).1 hK
  rw [← hF] at hflat
  simp only [Finset.mem_coe] at hflat
  have hl (k : Fin 3) : Sum.inl k ∈ F ↔ ![a, b, c] k ∈ Sum.inl ⁻¹' S := by
    rw [← Finset.mem_coe, hF]
    rfl
  have hr : (Finset.univ.filter fun i => Sum.inr i ∈ F) = newIdx S := by
    ext i
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    rw [← Finset.mem_coe, hF, mem_newIdx]
    rfl
  obtain ⟨h1, h2⟩ := k4_flat_facts F hflat
  rw [hr] at h1 h2
  refine ⟨fun k hk => (hl k).1 (h1 k hk), fun hne i hi => ?_⟩
  have hiF : Sum.inr i ∉ F := fun h => hi (by rw [← hr]; simpa using h)
  obtain ⟨k, hk, hkF⟩ := h2 hne i hiF
  exact ⟨k, hk, fun h => hkF ((hl k).2 h)⟩

variable [Finite E]

/-- **The flats of the generalized parallel connection** of `M` and `M(K₄)` across a triangle
(§2.4, Oxley §11.4): the sets meeting `M` in a flat of `M` and `M(K₄)` in a flat of `M(K₄)`. -/
theorem isFlat_deltaConnection_iff (hT : M.IsCircuit {a, b, c}) (hab : a ≠ b) (hac : a ≠ c)
    (hbc : b ≠ c) (S : Set (E ⊕ Fin 3)) :
    (deltaConnection M a b c hT).IsFlat S ↔
      M.IsFlat (Sum.inl ⁻¹' S) ∧ k4Matroid.IsFlat (deltaEmbed a b c ⁻¹' S) := by
  refine ⟨fun hS => ?_, fun ⟨hM, hK⟩ => ?_⟩
  · rw [← deltaConnection_comap_inl hT, ← deltaConnection_comap_deltaEmbed hT hab hac hbc]
    exact ⟨isFlat_comap hS _, isFlat_comap hS _⟩
  have hSE : S ⊆ connGround M := by
    rintro (e | i) he
    · exact Or.inl ⟨e, hM.subset_ground he, rfl⟩
    · exact Or.inr ⟨i, rfl⟩
  obtain ⟨hK1, hK2⟩ := k4_isFlat_spanIdx hK
  have hσ : ![a, b, c] '' ↑(spanIdx (newIdx S)) ⊆ Sum.inl ⁻¹' S := by
    rintro _ ⟨k, hk, rfl⟩
    exact hK1 k hk
  have hS : connRk M a b c S = natRk M (Sum.inl ⁻¹' S) + nonemptyInd (newIdx S) := by
    rw [connRk, Set.union_eq_self_of_subset_right hσ]
  rw [isFlat_iff_natRk]
  refine ⟨hSE, fun e he heS => ?_⟩
  rw [natRk_deltaConnection hT hSE, natRk_deltaConnection hT (Set.insert_subset he hSE), hS]
  rcases e with e | i
  · have heM : e ∈ M.E := by
      rw [deltaConnection_E] at he
      simpa [connGround] using he
    have hnew : newIdx (insert (Sum.inl e) S) = newIdx S := by
      ext i
      simp [mem_newIdx]
    have hpre : Sum.inl ⁻¹' (insert (Sum.inl e) S) = insert e (Sum.inl ⁻¹' S) := by
      ext f
      simp
    rw [connRk, hnew, hpre, Set.insert_union, Set.union_eq_self_of_subset_right hσ,
      natRk_insert_of_notMem_closure ⟨heM, by rw [hM.closure]; exact heS⟩]
    omega
  · have hi : i ∉ newIdx S := fun h => heS (mem_newIdx.1 h)
    have hnew : newIdx (insert (Sum.inr i) S) = insert i (newIdx S) := by
      ext j
      simp [mem_newIdx]
    have hpre : Sum.inl ⁻¹' (insert (Sum.inr i) S) = Sum.inl ⁻¹' S := by
      ext f
      simp
    have hind : nonemptyInd (insert i (newIdx S)) = 1 := by
      simp [nonemptyInd]
    rw [connRk, hnew, hpre, hind]
    by_cases hne : (newIdx S).Nonempty
    · obtain ⟨k, hk, hkS⟩ := hK2 hne i hi
      have hk' := natRk_insert_of_notMem_closure (M := M)
        ⟨hT.subset_ground (triple_mem a b c k), by rw [hM.closure]; exact hkS⟩
      have hle := natRk_mono M (Set.insert_subset (Or.inr ⟨k, hk, rfl⟩) Set.subset_union_left :
        insert (![a, b, c] k) (Sum.inl ⁻¹' S) ⊆
          Sum.inl ⁻¹' S ∪ ![a, b, c] '' ↑(spanIdx (insert i (newIdx S))))
      have : nonemptyInd (newIdx S) = 1 := by simp [nonemptyInd, hne]
      omega
    · have : nonemptyInd (newIdx S) = 0 := by simp [nonemptyInd, hne]
      have := natRk_mono M (Set.subset_union_left (s := Sum.inl ⁻¹' S)
        (t := ![a, b, c] '' ↑(spanIdx (insert i (newIdx S)))))
      omega

end Transformer.GainMatroid
