module

public import Transformer.GainMatroid.Section2_ConnectionFlats
public import Transformer.GainMatroid.Section2_MatrixMatroids

/-!
# The matroid of the parallel connection across a triangle

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, citing Brylawski. Let `A` represent `M`, with
the columns of `X = {a, b, c}` the last three columns of `I(K₄)` on the rows `x, y, z`
(`IsTriangleForm`). Then `a, b, c` are distinct (`IsTriangleForm.a_ne_b`, `IsTriangleForm.a_ne_c`,
`IsTriangleForm.b_ne_c`) and `X` is a triangle of `M` (`IsTriangleForm.isCircuit`). The flats of
the matroid of `connectionMatrix A x y z` are the sets whose elements of `M` form a flat of `M` and
whose elements of `K₄` form a flat of `M(K₄)` (`IsTriangleForm.isFlat_connection_iff`): it is the
generalized parallel connection of `M` and `M(K₄)` across `X` that `IsDeltaMatroid` asks for.
-/

@[expose] public section

namespace Transformer.GainMatroid

open Submodule

variable {E R F : Type*} [DecidableEq R] [Field F] {A : Matrix R E F} {a b c : E} {x y z : R}

namespace IsTriangleForm

theorem a_ne_b (hform : IsTriangleForm A a b c x y z) : a ≠ b := fun h => by
  have hx := hform.col_a x
  rw [h, hform.col_b x] at hx
  simp [hform.ne_xy, hform.ne_xz] at hx

theorem a_ne_c (hform : IsTriangleForm A a b c x y z) : a ≠ c := fun h => by
  have hx := hform.col_a x
  rw [h, hform.col_c x] at hx
  simp [hform.ne_xy, hform.ne_xz] at hx

theorem b_ne_c (hform : IsTriangleForm A a b c x y z) : b ≠ c := fun h => by
  have hy := hform.col_b y
  rw [h, hform.col_c y] at hy
  simp [hform.ne_xy.symm, hform.ne_yz] at hy

/-- `X` is a triangle of a matroid that `A` represents. -/
theorem isCircuit {M : Matroid E} (hform : IsTriangleForm A a b c x y z)
    (hA : IsMatrixRepresentation M A) : M.IsCircuit {a, b, c} := by
  have hab := hform.a_ne_b
  have hac := hform.a_ne_c
  have hbc := hform.b_ne_c
  have hsub (X : Set E) : X ⊆ M.E := hA.1 ▸ Set.subset_univ X
  have hval (r : R) (e : E) (s t : F) (f : E)
      (h : s • (fun r => A r e) + t • (fun r => A r f) = 0) : s * A r e + t * A r f = 0 := by
    simpa using congrFun h r
  have hy := hform.ne_xy.symm
  have hzy := hform.ne_yz.symm
  rw [Matroid.isCircuit_iff_dep_forall_sdiff_singleton_indep]
  refine ⟨(Matroid.not_indep_iff (hsub _)).1 fun hI => ?_, fun e he => ?_⟩
  · rw [hA.2, linearIndepOn_insert (by simp [hab, hac])] at hI
    refine hI.2 ?_
    have hrel : (fun r => A r a) = (fun r => A r b) - fun r => A r c := by
      funext r
      simp only [Pi.sub_apply, hform.col_a, hform.col_b, hform.col_c]
      ring
    rw [hrel]
    exact sub_mem (subset_span ⟨b, by simp, rfl⟩) (subset_span ⟨c, by simp, rfl⟩)
  · rw [hA.2]
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at he
    rcases he with rfl | rfl | rfl
    · rw [show ({e, b, c} : Set E) \ {e} = {b, c} by ext; simp; aesop]
      refine (LinearIndepOn.pair_iff _ hbc).2 fun s t h => ?_
      have h₁ := hval z b s t c h
      have h₂ := hval y b s t c h
      simp [hform.col_b, hform.col_c, hform.ne_xz.symm, hzy, hy, hform.ne_yz] at h₁ h₂
      exact ⟨h₁, by simpa [h₁] using h₂⟩
    · rw [show ({a, e, c} : Set E) \ {e} = {a, c} by ext; simp; aesop]
      refine (LinearIndepOn.pair_iff _ hac).2 fun s t h => ?_
      have h₁ := hval x a s t c h
      have h₂ := hval y a s t c h
      simp [hform.col_a, hform.col_c, hform.ne_xy, hform.ne_xz, hy, hform.ne_yz] at h₁ h₂
      exact ⟨by simpa [h₁] using h₂, h₁⟩
    · rw [show ({a, b, e} : Set E) \ {e} = {a, b} by ext; simp; aesop]
      refine (LinearIndepOn.pair_iff _ hab).2 fun s t h => ?_
      have h₁ := hval x a s t b h
      have h₂ := hval y a s t b h
      simp [hform.col_a, hform.col_b, hform.ne_xy, hform.ne_xz, hy, hform.ne_yz] at h₁ h₂
      exact ⟨h₂, h₁⟩

/-- **The flats of the parallel connection** (§2.4, Brylawski): a set is a flat of the matroid of
`connectionMatrix A x y z` exactly when its elements of `M` form a flat of `M` and its elements of
`K₄` a flat of `M(K₄)`. -/
theorem isFlat_connection_iff [Finite E] [DecidableEq E] {M : Matroid E}
    (hform : IsTriangleForm A a b c x y z) (hA : IsMatrixRepresentation M A)
    (S : Set (E ⊕ Fin 3)) :
    (matrixMatroid (connectionMatrix A x y z)).IsFlat S ↔
      M.IsFlat (Sum.inl ⁻¹' S) ∧ k4Matroid.IsFlat (deltaEmbed a b c ⁻¹' S) := by
  have hP : (matrixMatroid (connectionMatrix A x y z)).IsFlat S ↔
      ∀ q, connectionCol A x y z q ∈ span F (connectionCol A x y z '' S) → q ∈ S :=
    (matrixMatroid_isMatrixRepresentation _).isFlat_iff S
  rw [hP, hA.isFlat_iff, k4Matroid_isFlat_iff]
  constructor
  · intro h
    refine ⟨fun e he => h (.inl e) ?_, fun C hC e he hCe =>
      h (deltaEmbed a b c e) (hform.mem_span_of_mem_k4Triangles hC he hCe)⟩
    have he' : connectionCol A x y z (.inl e) ∈
        span F (connectionCol A x y z '' (Sum.inl '' (Sum.inl ⁻¹' S))) := by
      rw [connectionCol_inl, mem_span_inl_iff]
      exact he
    exact span_mono (Set.image_mono (Set.image_preimage_subset _ _)) he'
  rintro ⟨hM, hK⟩ q hq
  by_cases hstar : ∃ i, Sum.inr i ∈ S
  · obtain ⟨i, hi⟩ := hstar
    have hsh := hform.shiftNone_mem_span hi hK hq
    rcases q with e | j
    · rw [shiftNone_inl, connectionCol_inl, mem_span_inl_iff] at hsh
      exact hM e hsh
    rw [shiftNone_inr] at hsh
    obtain rfl | hji := eq_or_ne j i
    · exact hi
    obtain ⟨k, hC, hk⟩ := hform.exists_triangle j i hji
    have hkV : connectionCol A x y z (.inl (![a, b, c] k)) ∈
        span F (connectionCol A x y z '' (Sum.inl '' (Sum.inl ⁻¹' S))) := by
      rcases hk with hk | hk
      · rwa [← hk]
      · rw [← hk, ← neg_sub]
        exact neg_mem hsh
    rw [connectionCol_inl, mem_span_inl_iff] at hkV
    exact hK _ hC (.inr j) (by simp) fun f hf hfe => by
      simp only [Finset.mem_insert, Finset.mem_singleton] at hf
      rcases hf with rfl | rfl | rfl
      · exact absurd rfl hfe
      · exact hi
      · exact hM _ hkV
  push Not at hstar
  have hsub : S ⊆ Sum.inl '' (Sum.inl ⁻¹' S) := by
    rintro (e | j) hq
    · exact ⟨e, hq, rfl⟩
    · exact absurd hq (hstar j)
  have hq' := span_mono (Set.image_mono hsub) hq
  rcases q with e | j
  · rw [connectionCol_inl, mem_span_inl_iff] at hq'
    exact hM e hq'
  · have h0 := none_eq_zero_of_mem_span hq'
    rw [connectionCol_inr_none] at h0
    exact absurd h0 (neg_ne_zero.2 one_ne_zero)

end IsTriangleForm

example : (5 : Fin 6) ≠ 4 := (iK4_isTriangleForm (F := ℚ)).a_ne_b

example : (5 : Fin 6) ≠ 3 := (iK4_isTriangleForm (F := ℚ)).a_ne_c

example : (4 : Fin 6) ≠ 3 := (iK4_isTriangleForm (F := ℚ)).b_ne_c

/-- The triangle `{a, b, c}` of `M(K₄)`. -/
example : (matrixMatroid (iK4 : Matrix (Fin 4) (Fin 6) ℚ)).IsCircuit {5, 4, 3} :=
  iK4_isTriangleForm.isCircuit (matrixMatroid_isMatrixRepresentation _)


/-- The flats of the parallel connection of `M(K₄)` with itself across a triangle. -/
example := (iK4_isTriangleForm (F := ℚ)).isFlat_connection_iff
  (matrixMatroid_isMatrixRepresentation _)

end Transformer.GainMatroid
