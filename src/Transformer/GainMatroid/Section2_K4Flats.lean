module

public import Transformer.GainMatroid.Section2_MatroidExchange

/-!
# The flats of `M(K₄)`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. With the labels of §2.4, `M(K₄)`
(`k4Matroid`) has the triangle `{a, b, c} = inl 0, 1, 2` and the triangles `{a, b′, c′}`,
`{a′, b, c′}` and `{a′, b′, c}` through `a′, b′, c′ = inr 0, 1, 2` (`k4Triangles`). A set of edges
is a flat of `M(K₄)` exactly when it holds the third edge of every triangle with two edges in it
(`k4Matroid_isFlat_iff`).
-/

@[expose] public section

namespace Transformer.GainMatroid

/-- The four triangles of `M(K₄)` with the labels of §2.4. -/
def k4Triangles : Finset (Finset (Fin 3 ⊕ Fin 3)) :=
  {{.inl 0, .inl 1, .inl 2}, {.inl 0, .inr 1, .inr 2}, {.inr 0, .inl 1, .inr 2},
    {.inr 0, .inr 1, .inl 2}}

set_option maxRecDepth 100000 in
private theorem k4_flatTest_iff : ∀ T : Finset (Fin 3 ⊕ Fin 3), FlatTest k4Indep T ↔
    ∀ C ∈ k4Triangles, ∀ e ∈ C, (∀ f ∈ C, f ≠ e → f ∈ T) → e ∈ T := by
  decide +kernel

/-- The flats of `M(K₄)` are the sets of edges that hold the third edge of every triangle with two
edges in them. -/
theorem k4Matroid_isFlat_iff (T : Set (Fin 3 ⊕ Fin 3)) :
    k4Matroid.IsFlat T ↔ ∀ C ∈ k4Triangles, ∀ e ∈ C, (∀ f ∈ C, f ≠ e → f ∈ T) → e ∈ T := by
  obtain ⟨T, rfl⟩ := T.toFinite.exists_finset_coe
  simp only [Finset.mem_coe]
  exact (matroidOfFinset_isFlat_iff T).trans (k4_flatTest_iff T)

end Transformer.GainMatroid
