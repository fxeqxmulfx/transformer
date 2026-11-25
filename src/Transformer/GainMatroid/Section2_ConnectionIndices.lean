module

public import Transformer.GainMatroid.Section2_MatroidExchange

/-!
# Labels of a triangle and indices of new elements

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. The generalized parallel connection of `M`
and `M(K₄)` across a triangle `X = {a, b, c}` lives on `E ⊕ Fin 3`: the elements of `M` and three
new elements `inr 0, 1, 2`. The labels `![a, b, c]` name the elements of `X` (`triple_mem`,
`exists_triple_eq`), and `newIdx Z` collects the indices of the new elements of a set `Z`, which
with its old elements make up its size (`ncard_eq_inl_add_newIdx`).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {E : Type*} {a b c : E}

/-- The labels `![a, b, c]` take values in the triangle. -/
theorem triple_mem (a b c : E) (k : Fin 3) : ![a, b, c] k ∈ ({a, b, c} : Set E) := by
  fin_cases k <;> simp

/-- Every element of the triangle has a label. -/
theorem exists_triple_eq {e : E} (he : e ∈ ({a, b, c} : Set E)) : ∃ k, ![a, b, c] k = e := by
  rcases he with rfl | rfl | rfl
  exacts [⟨0, rfl⟩, ⟨1, rfl⟩, ⟨2, rfl⟩]

/-- The indices of the new elements of `Z`. -/
noncomputable def newIdx (Z : Set (E ⊕ Fin 3)) : Finset (Fin 3) := by
  classical exact Finset.univ.filter fun i => Sum.inr i ∈ Z

theorem mem_newIdx {Z : Set (E ⊕ Fin 3)} {i : Fin 3} : i ∈ newIdx Z ↔ Sum.inr i ∈ Z := by
  unfold newIdx
  simp

theorem newIdx_union (Z W : Set (E ⊕ Fin 3)) : newIdx (Z ∪ W) = newIdx Z ∪ newIdx W := by
  ext i
  simp [mem_newIdx]

theorem newIdx_inter (Z W : Set (E ⊕ Fin 3)) : newIdx (Z ∩ W) = newIdx Z ∩ newIdx W := by
  ext i
  simp [mem_newIdx]

theorem newIdx_mono {Z W : Set (E ⊕ Fin 3)} (h : Z ⊆ W) : newIdx Z ⊆ newIdx W := fun _ hi =>
  mem_newIdx.2 (h (mem_newIdx.1 hi))

/-- A set of old and new elements has as many elements as its old part and its new indices. -/
theorem ncard_eq_inl_add_newIdx [Finite E] (Z : Set (E ⊕ Fin 3)) :
    Z.ncard = (Sum.inl ⁻¹' Z).ncard + (newIdx Z).card := by
  have hZ : Z = Sum.inl '' (Sum.inl ⁻¹' Z) ∪ Sum.inr '' ↑(newIdx Z) := by
    ext (e | i) <;> simp [mem_newIdx]
  have hdisj :
      Disjoint (Sum.inl '' (Sum.inl ⁻¹' Z) : Set (E ⊕ Fin 3)) (Sum.inr '' ↑(newIdx Z)) := by
    rw [Set.disjoint_left]
    rintro _ ⟨e, -, rfl⟩ ⟨i, -, h⟩
    exact Sum.inr_ne_inl h
  conv_lhs => rw [hZ]
  rw [Set.ncard_union_eq hdisj, Set.ncard_image_of_injective _ Sum.inl_injective,
    Set.ncard_image_of_injective _ Sum.inr_injective, Set.ncard_coe_finset]

end Transformer.GainMatroid
