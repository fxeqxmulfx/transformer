module

public import Transformer.GainMatroid.Section2_DeltaDual

/-!
# The triangle of `Δ_X M` is an independent set and, if coindependent in `M`, a triad

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. The matroid `Δ_X M` is unique
(`IsDeltaMatroid.eq`), since its parallel connection is unique by its flats
(`IsDeltaMatroid.eq_comap`). Its rank function `deltaRk` gives the rank of `X = {a, b, c}` in
`Δ_X M` as the rank `2` of the triangle plus one, so `X` is independent in `Δ_X M`
(`IsDeltaMatroid.indep_triangle`).

When `X` is coindependent in `M`, it is a triad of `Δ_X M` (`IsDeltaMatroid.dual_isCircuit`). It is
dependent in the dual, since `Δ_X M` has rank `r(M) + 1` and the complement of `X` has rank
`r_M(E \ X) ≤ r(M)`. For `s ∈ X`, `X \ {s} = {u, w}` is coindependent in `Δ_X M` exactly when
`s` added to the exchange of `{u, w}`, that is `X`, is coindependent in `M`
(`IsDeltaMatroid.dual_indep_insert_swap_iff`). Then `∇_X Δ_X M = M`
(`IsDeltaMatroid.eq_of_isNablaMatroid`), by `IsDeltaMatroid.dual`.
-/

@[expose] public section

namespace Transformer.GainMatroid

open Set

variable {E : Type*} [DecidableEq E] [Finite E] {M N N' : Matroid E} {a b c : E}

/-- **`Δ_X M` is unique.** -/
theorem IsDeltaMatroid.eq (h : IsDeltaMatroid N a b c M) (h' : IsDeltaMatroid N a b c N') :
    M = N' :=
  h.eq_comap.trans h'.eq_comap.symm

/-- The triangle `X` is independent in `Δ_X M`. -/
theorem IsDeltaMatroid.indep_triangle (h : IsDeltaMatroid N a b c M) : M.Indep {a, b, c} := by
  rw [indep_iff_natRk, h.natRk_eq h.triangle.subset_ground,
    deltaRk_of_subset subset_rfl ⟨a, mem_insert a _⟩]
  exact natRk_add_one_of_isCircuit h.triangle

omit [Finite E] in
/-- `{u, w}`, exchanged and with `s` added, is `{u, w, s}`. -/
theorem insert_swap_pair (s u w : E) :
    insert s (Equiv.swap u w '' {u, w}) = ({u, w, s} : Set E) := by
  rw [image_pair, Equiv.swap_apply_left, Equiv.swap_apply_right]
  ext e
  simp only [mem_insert_iff, mem_singleton_iff]
  tauto

/-- **A coindependent triangle of `M` is a triad of `Δ_X M`.** -/
theorem IsDeltaMatroid.dual_isCircuit (h : IsDeltaMatroid N a b c M)
    (hco : N✶.Indep {a, b, c}) : M✶.IsCircuit {a, b, c} := by
  have hE := h.ground_eq
  have hXE : ({a, b, c} : Set E) ⊆ N.E := h.triangle.subset_ground
  have hab := h.ne_ab
  have hac := h.ne_ac
  have hbc := h.ne_bc
  rw [Matroid.isCircuit_iff_dep_forall_sdiff_singleton_indep]
  refine ⟨⟨fun hi => ?_, by rw [Matroid.dual_ground, hE]; exact hXE⟩, fun t ht => ?_⟩
  · rw [dual_indep_iff_natRk (hE ▸ hXE), hE, h.natRk_eq sdiff_subset, h.natRk_eq subset_rfl,
      deltaRk_of_disjoint h.ne_ab disjoint_sdiff_left,
      deltaRk_of_subset hXE ⟨a, mem_insert a _⟩] at hi
    have := natRk_mono N (sdiff_subset : N.E \ {a, b, c} ⊆ N.E)
    omega
  obtain ⟨u, w, hX, hut, hwt, huw, hsub⟩ : ∃ u w : E, ({u, w, t} : Set E) = {a, b, c} ∧
      u ≠ t ∧ w ≠ t ∧ u ≠ w ∧ ({a, b, c} : Set E) \ {t} = {u, w} := by
    rcases ht with rfl | rfl | rfl
    · refine ⟨b, c, triple_bca_eq, hab.symm, hac.symm, hbc, ?_⟩
      ext e
      simp only [mem_sdiff, mem_insert_iff, mem_singleton_iff]
      grind
    · refine ⟨a, c, triple_acb_eq, hab, hbc.symm, hac, ?_⟩
      ext e
      simp only [mem_sdiff, mem_insert_iff, mem_singleton_iff]
      grind
    · refine ⟨a, b, rfl, hac, hbc, hab, ?_⟩
      ext e
      simp only [mem_sdiff, mem_insert_iff, mem_singleton_iff]
      grind
  rw [hsub, ← h.dual_indep_insert_swap_iff hX hut hwt huw (by simp [hut.symm, hwt.symm]),
    insert_swap_pair, hX]
  exact hco

/-- **`∇_X Δ_X M = M`** when `X` is a triad of `Δ_X M`. -/
theorem IsDeltaMatroid.eq_of_isNablaMatroid (h : IsDeltaMatroid N a b c M)
    (ht : M✶.IsCircuit {a, b, c}) (h' : IsNablaMatroid M a b c N') : N' = N := by
  obtain ⟨Q, hQ, rfl⟩ := h'
  rw [← (h.dual ht).eq hQ, Matroid.dual_dual]

example := insert_swap_pair (0 : Fin 3) 1 2

end Transformer.GainMatroid
