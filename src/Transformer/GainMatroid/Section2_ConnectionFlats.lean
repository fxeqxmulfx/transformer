module

public import Transformer.GainMatroid.Section2_TriangleConnection

/-!
# The flats of the parallel connection across a triangle

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, citing Brylawski. Let `A` represent `M`,
with the columns of the triangle `X = {a, b, c}` the last three columns of `I(K₄)`
(`IsTriangleForm`). The matroid of `connectionMatrix A x y z` is the generalized parallel
connection of `M` and `M(K₄)` across `X`: a set `S` is a flat of it exactly when its elements of `M`
form a flat of `M` and its elements of `K₄` a flat of `M(K₄)`
(`IsTriangleForm.isFlat_connection_iff`).

* The columns of `A` in `T`, extended by a zero entry at `none`, span the extension of the span of
  their columns in `A` (`span_image_inl`). So a flat of `M` stays closed among them
  (`mem_span_inl_iff`), and their span vanishes at `none` (`none_eq_zero_of_mem_span`).
* In a triangle of `M(K₄)`, the column of each edge lies in the span of the other two
  (`IsTriangleForm.mem_span_of_mem_k4Triangles`): a flat meets `K₄` in a flat of `M(K₄)`.
* Conversely, fix a new column `i` in `S`. `shiftNone i` adds to a vector its entry at `none` times
  the column of `i`. It keeps the columns of `A`, and sends a new column `j ≠ i` to the column of
  the edge of `X` in the triangle through `i` and `j`, up to sign. When `S` is closed under the
  triangles of `M(K₄)`, it sends the span of `S` into the span of its columns of `A`
  (`IsTriangleForm.shiftNone_mem_span`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open Submodule

variable {E R F : Type*} [DecidableEq R] [Field F] {A : Matrix R E F} {x y z : R}

omit [DecidableEq R] in
theorem extendZero_injective : Function.Injective (extendZero (R := R) (F := F)) :=
  fun _ _ h => funext fun w => congrFun h (some w)

/-- The columns of `A` in `T`, extended by a zero entry, span the extension of the span of the
columns of `T` in `A`. -/
theorem span_image_inl (T : Set E) :
    span F (connectionCol A x y z '' (Sum.inl '' T)) =
      (span F ((fun e r => A r e) '' T)).map extendZero := by
  rw [map_span, Set.image_image, Set.image_image]
  exact congrArg _ (Set.image_congr fun e _ => connectionCol_inl e)

/-- An extended vector lies in the span of the extended columns of `T` exactly when the vector
lies in the span of the columns of `T`. -/
theorem mem_span_inl_iff (T : Set E) (g : R → F) :
    extendZero g ∈ span F (connectionCol A x y z '' (Sum.inl '' T)) ↔
      g ∈ span F ((fun e r => A r e) '' T) := by
  rw [span_image_inl]
  refine ⟨fun h => ?_, mem_map_of_mem⟩
  obtain ⟨g', hg', hgg'⟩ := mem_map.1 h
  exact extendZero_injective hgg' ▸ hg'

/-- The span of the columns of `A` vanishes at `none`. -/
theorem none_eq_zero_of_mem_span {T : Set E} {w : Option R → F}
    (hw : w ∈ span F (connectionCol A x y z '' (Sum.inl '' T))) : w none = 0 := by
  rw [span_image_inl] at hw
  obtain ⟨g, -, rfl⟩ := mem_map.1 hw
  rfl

omit [DecidableEq R] in
/-- If `p = q - r`, a submodule holding two of `p`, `q`, `r` holds the third. -/
theorem mem_of_eq_sub {V : Type*} [AddCommGroup V] [Module F V] {s : Submodule F V} {p q r : V}
    (h : p = q - r) :
    (q ∈ s → r ∈ s → p ∈ s) ∧ (p ∈ s → r ∈ s → q ∈ s) ∧ (p ∈ s → q ∈ s → r ∈ s) := by
  subst h
  exact ⟨sub_mem, fun hp hr => by simpa using add_mem hp hr,
    fun hp hq => by simpa using sub_mem hq hp⟩

/-- Adding to a vector its entry at `none` times the new column `i`. -/
def shiftNone (A : Matrix R E F) (x y z : R) (i : Fin 3) : (Option R → F) →ₗ[F] Option R → F :=
  LinearMap.id + (LinearMap.proj none).smulRight (connectionCol A x y z (.inr i))

theorem shiftNone_apply (i : Fin 3) (w : Option R → F) :
    shiftNone A x y z i w = w + w none • connectionCol A x y z (.inr i) := rfl

theorem shiftNone_inl (i : Fin 3) (e : E) :
    shiftNone A x y z i (connectionCol A x y z (.inl e)) = connectionCol A x y z (.inl e) := by
  rw [shiftNone_apply, connectionCol_inl_none, zero_smul, add_zero]

theorem shiftNone_inr (i j : Fin 3) :
    shiftNone A x y z i (connectionCol A x y z (.inr j)) =
      connectionCol A x y z (.inr j) - connectionCol A x y z (.inr i) := by
  rw [shiftNone_apply, connectionCol_inr_none, neg_one_smul, sub_eq_add_neg]

namespace IsTriangleForm

variable {a b c : E}

/-- In a triangle `C` of `M(K₄)`, the column of each edge `e` lies in the span of the columns of a
set `S` holding the other two edges. -/
theorem mem_span_of_mem_k4Triangles (hform : IsTriangleForm A a b c x y z)
    {C : Finset (Fin 3 ⊕ Fin 3)} (hC : C ∈ k4Triangles) {e : Fin 3 ⊕ Fin 3} (he : e ∈ C)
    {S : Set (E ⊕ Fin 3)} (hS : ∀ f ∈ C, f ≠ e → f ∈ deltaEmbed a b c ⁻¹' S) :
    connectionCol A x y z (deltaEmbed a b c e) ∈ span F (connectionCol A x y z '' S) := by
  have h f (hf : f ∈ C) (hfe : f ≠ e) :
      connectionCol A x y z (deltaEmbed a b c f) ∈ span F (connectionCol A x y z '' S) :=
    subset_span ⟨deltaEmbed a b c f, hS f hf hfe, rfl⟩
  have h0 : connectionCol A x y z (deltaEmbed a b c (.inl 0)) =
      connectionCol A x y z (deltaEmbed a b c (.inr 1)) -
        connectionCol A x y z (deltaEmbed a b c (.inr 2)) := hform.connectionCol_a
  have h1 : connectionCol A x y z (deltaEmbed a b c (.inl 1)) =
      connectionCol A x y z (deltaEmbed a b c (.inr 0)) -
        connectionCol A x y z (deltaEmbed a b c (.inr 2)) := hform.connectionCol_b
  have h2 : connectionCol A x y z (deltaEmbed a b c (.inl 2)) =
      connectionCol A x y z (deltaEmbed a b c (.inr 0)) -
        connectionCol A x y z (deltaEmbed a b c (.inr 1)) := hform.connectionCol_c
  have h3 : connectionCol A x y z (deltaEmbed a b c (.inl 0)) =
      connectionCol A x y z (deltaEmbed a b c (.inl 1)) -
        connectionCol A x y z (deltaEmbed a b c (.inl 2)) := by
    rw [h0, h1, h2]
    abel
  simp only [k4Triangles, Finset.mem_insert, Finset.mem_singleton] at hC
  rcases hC with rfl | rfl | rfl | rfl <;>
    simp only [Finset.mem_insert, Finset.mem_singleton] at he <;> rcases he with rfl | rfl | rfl
  · exact (mem_of_eq_sub h3).1 (h _ (by decide) (by decide)) (h _ (by decide) (by decide))
  · exact (mem_of_eq_sub h3).2.1 (h _ (by decide) (by decide)) (h _ (by decide) (by decide))
  · exact (mem_of_eq_sub h3).2.2 (h _ (by decide) (by decide)) (h _ (by decide) (by decide))
  · exact (mem_of_eq_sub h0).1 (h _ (by decide) (by decide)) (h _ (by decide) (by decide))
  · exact (mem_of_eq_sub h0).2.1 (h _ (by decide) (by decide)) (h _ (by decide) (by decide))
  · exact (mem_of_eq_sub h0).2.2 (h _ (by decide) (by decide)) (h _ (by decide) (by decide))
  · exact (mem_of_eq_sub h1).2.1 (h _ (by decide) (by decide)) (h _ (by decide) (by decide))
  · exact (mem_of_eq_sub h1).1 (h _ (by decide) (by decide)) (h _ (by decide) (by decide))
  · exact (mem_of_eq_sub h1).2.2 (h _ (by decide) (by decide)) (h _ (by decide) (by decide))
  · exact (mem_of_eq_sub h2).2.1 (h _ (by decide) (by decide)) (h _ (by decide) (by decide))
  · exact (mem_of_eq_sub h2).2.2 (h _ (by decide) (by decide)) (h _ (by decide) (by decide))
  · exact (mem_of_eq_sub h2).1 (h _ (by decide) (by decide)) (h _ (by decide) (by decide))

/-- Let `S` hold the new column `i` and be closed under the triangles of `M(K₄)`. Then
`shiftNone i` sends the span of the columns of `S` into the span of its columns of `A`. -/
theorem shiftNone_mem_span (hform : IsTriangleForm A a b c x y z) {S : Set (E ⊕ Fin 3)}
    {i : Fin 3} (hi : .inr i ∈ S)
    (hS : ∀ C ∈ k4Triangles, ∀ e ∈ C, (∀ f ∈ C, f ≠ e → f ∈ deltaEmbed a b c ⁻¹' S) →
      e ∈ deltaEmbed a b c ⁻¹' S)
    {w : Option R → F} (hw : w ∈ span F (connectionCol A x y z '' S)) :
    shiftNone A x y z i w ∈ span F (connectionCol A x y z '' (Sum.inl '' (Sum.inl ⁻¹' S))) := by
  refine span_le.2 ?_ (image_span_subset_span (shiftNone A x y z i) _ ⟨w, hw, rfl⟩)
  rintro _ ⟨_, ⟨e | j, hq, rfl⟩, rfl⟩
  · rw [shiftNone_inl]
    exact subset_span ⟨.inl e, ⟨e, hq, rfl⟩, rfl⟩
  rw [shiftNone_inr]
  obtain rfl | hji := eq_or_ne j i
  · rw [sub_self]
    exact zero_mem _
  obtain ⟨k, hC, hk⟩ := hform.exists_triangle j i hji
  have hkS : Sum.inl (![a, b, c] k) ∈ S := hS _ hC (.inl k) (by simp) fun f hf hfe => by
    simp only [Finset.mem_insert, Finset.mem_singleton] at hf
    rcases hf with rfl | rfl | rfl
    · exact hq
    · exact hi
    · exact absurd rfl hfe
  have hmem : connectionCol A x y z (.inl (![a, b, c] k)) ∈
      span F (connectionCol A x y z '' (Sum.inl '' (Sum.inl ⁻¹' S))) :=
    subset_span ⟨_, ⟨_, hkS, rfl⟩, rfl⟩
  rcases hk with hk | hk
  · rwa [hk]
  · rw [← neg_sub, hk]
    exact neg_mem hmem

end IsTriangleForm

example : connectionCol (iK4 : Matrix (Fin 4) (Fin 6) ℚ) 0 1 2 (.inl 0) none = 0 :=
  none_eq_zero_of_mem_span (T := Set.univ) (subset_span ⟨.inl 0, ⟨0, trivial, rfl⟩, rfl⟩)

example := mem_of_eq_sub (s := (⊤ : Submodule ℚ ℚ)) (show (1 : ℚ) = 3 - 2 by norm_num)

/-- The triangle `{a, b, c}` of `M(K₄)`, inside the matroid of the parallel connection. -/
example := (iK4_isTriangleForm (F := ℚ)).mem_span_of_mem_k4Triangles
  (C := {.inl 0, .inl 1, .inl 2}) (by decide) (e := .inl 0) (by decide) (S := Set.univ)
  fun _ _ _ => trivial

example := (iK4_isTriangleForm (F := ℚ)).shiftNone_mem_span (S := Set.univ) (i := 0) trivial
  (fun _ _ _ _ _ => trivial) (subset_span ⟨.inr 0, trivial, rfl⟩)

end Transformer.GainMatroid
