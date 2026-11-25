module

public import Transformer.GainMatroid.Section2_DeltaIndep

/-!
# `∇_X` undoes `Δ_X`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4: `∇_Y M = (Δ_Y M*)*` for a triad `Y` of `M`.
The generalized parallel connection is determined by its flats (`eq_of_isFlat_iff`), so `Δ_X N`
is the relabelled deletion of `deltaConnection` (`IsDeltaMatroid.eq_comap`), of rank `deltaRk`
(`IsDeltaMatroid.natRk_eq`).

A set `J` is coindependent when deleting it keeps the rank (`dual_indep_iff_natRk`). If `M = Δ_X N`,
the complement of a set `I` of `M` through `s ∈ X` is `s` added to the exchange of the complement of
`s` added to the exchange of `I` (`sdiff_eq_insert_swap`), so the rank formula of `Δ_X` describes
the coindependent sets of `M` by those of `N`. So if `X` is a triad of `M`, then `Δ_X M* = N*`
(`IsDeltaMatroid.dual`), that is, `N = ∇_X M` (`IsDeltaMatroid.isNablaMatroid`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open Set

/-- Two matroids with the same elements and the same flats are equal. -/
theorem eq_of_isFlat_iff {α : Type*} {P Q : Matroid α} (hE : P.E = Q.E)
    (h : ∀ S, P.IsFlat S ↔ Q.IsFlat S) : P = Q :=
  Matroid.ext_closure fun X => by simp only [Matroid.closure_def, hE, h]

variable {E : Type*} {M N : Matroid E} {a b c : E}

/-- An exchange of two elements of `S` keeps membership in `S`. -/
theorem swap_mem_iff [DecidableEq E] {S : Set E} (ha : a ∈ S) (hb : b ∈ S) {e : E} :
    Equiv.swap a b e ∈ S ↔ e ∈ S := by
  rw [Equiv.swap_apply_def]
  split_ifs with h₁ h₂
  · exact ⟨fun _ => h₁ ▸ ha, fun _ => hb⟩
  · exact ⟨fun _ => h₂ ▸ hb, fun _ => ha⟩
  · exact Iff.rfl

/-- The complement of a set `I` off `c` is `c` added to the exchange of the complement of `c` added
to the exchange of `I`. -/
theorem sdiff_eq_insert_swap [DecidableEq E] {S I : Set E} (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S)
    (hac : a ≠ c) (hbc : b ≠ c) (hcI : c ∉ I) :
    S \ I = insert c (Equiv.swap a b '' (S \ insert c (Equiv.swap a b '' I))) := by
  have hswap : ∀ e, Equiv.swap a b e = c ↔ e = c := fun e => by
    rw [← Equiv.swap_apply_of_ne_of_ne hac.symm hbc.symm, (Equiv.swap a b).injective.eq_iff]
    rw [Equiv.swap_apply_of_ne_of_ne hac.symm hbc.symm]
  ext e
  simp only [mem_sdiff, mem_insert_iff, mem_image_equiv, Equiv.symm_swap, Equiv.swap_apply_self,
    swap_mem_iff ha hb, hswap]
  by_cases hec : e = c
  · subst hec
    exact ⟨fun _ => Or.inl rfl, fun _ => ⟨hc, hcI⟩⟩
  · tauto


/-- A set holding `X` has rank `deltaRk` one more than its rank. -/
theorem deltaRk_of_subset {X Y : Set E} (hXY : X ⊆ Y) (hX : X.Nonempty) :
    deltaRk N X Y = natRk N Y + 1 := by
  have hspan : starSpan X Y = X := Set.ext fun t =>
    ⟨fun ht => ht.1, fun ht => ⟨ht, sdiff_subset.trans hXY⟩⟩
  rw [deltaRk, hspan, sdiff_union_of_subset hXY,
    ite_eq_left (hX.mono (subset_inter hXY subset_rfl))]

variable [Finite E]

/-- The rank of the dual: `r*(X) + r(E) = r(E \ X) + |X ∩ E|`. -/
theorem natRk_dual_add (M : Matroid E) (X : Set E) :
    natRk M✶ X + natRk M M.E = natRk M (M.E \ X) + (X ∩ M.E).ncard := by
  have := M.eRk_dual_add_eRank' X
  rw [Matroid.eRank_def, ← natRk_cast, ← natRk_cast, ← natRk_cast,
    ← (toFinite (X ∩ M.E)).cast_ncard_eq] at this
  exact_mod_cast this

/-- A set is coindependent when deleting it keeps the rank. -/
theorem dual_indep_iff_natRk {J : Set E} (hJ : J ⊆ M.E) :
    M✶.Indep J ↔ natRk M (M.E \ J) = natRk M M.E := by
  have := natRk_dual_add M J
  rw [inter_eq_left.2 hJ] at this
  rw [indep_iff_natRk]
  omega

variable [DecidableEq E]

/-- **The parallel connection is unique.** `Δ_X N` is the relabelled deletion of
`deltaConnection`. -/
theorem IsDeltaMatroid.eq_comap (h : IsDeltaMatroid N a b c M) :
    M = (deltaConnection N a b c h.triangle).comap (deltaRelabel a b c) := by
  obtain ⟨P, hPE, hP, hM⟩ := h.exists_connection
  refine hM.trans (congrArg (fun Q : Matroid (E ⊕ Fin 3) => Q.comap (deltaRelabel a b c)) ?_)
  refine eq_of_isFlat_iff (by rw [hPE, deltaConnection_E, connGround]) fun S => ?_
  rw [hP, isFlat_deltaConnection_iff h.triangle h.ne_ab h.ne_ac h.ne_bc]

/-- **The rank of `Δ_X N`** is `deltaRk`. -/
theorem IsDeltaMatroid.natRk_eq (h : IsDeltaMatroid N a b c M) {I : Set E} (hI : I ⊆ N.E) :
    natRk M I = deltaRk N {a, b, c} I := by
  rw [h.eq_comap, natRk_comap, natRk_deltaConnection h.triangle (by
    rw [image_subset_iff, deltaRelabel_preimage_connGround h.triangle]; exact hI),
    connRk_image_deltaRelabel h.ne_ab h.ne_ac h.ne_bc]

/-- Coindependence across `Δ_X` off `X`. -/
theorem IsDeltaMatroid.dual_indep_iff_of_disjoint (h : IsDeltaMatroid N a b c M) {I : Set E}
    (hI : Disjoint I {a, b, c}) : N✶.Indep I ↔ M✶.Indep I := by
  have hE := h.ground_eq
  by_cases hIE : I ⊆ N.E
  · have hIM : I ⊆ M.E := by
      rw [hE]
      exact hIE
    have hX : ({a, b, c} : Set E) ⊆ N.E \ I :=
      subset_sdiff.2 ⟨h.triangle.subset_ground, hI.symm⟩
    have hne : ({a, b, c} : Set E).Nonempty := ⟨a, Or.inl rfl⟩
    rw [dual_indep_iff_natRk hIE, dual_indep_iff_natRk hIM, hE, h.natRk_eq sdiff_subset,
      h.natRk_eq subset_rfl, deltaRk_of_subset hX hne,
      deltaRk_of_subset h.triangle.subset_ground hne]
    omega
  · exact iff_of_false (fun h' => hIE h'.subset_ground) fun h' => hIE (hE ▸ h'.subset_ground)

/-- Coindependence across `Δ_X` through an element `s` of `X`: `s` added to the exchange of `I` is
coindependent in `N` exactly when `I` is coindependent in `Δ_X N`. -/
theorem IsDeltaMatroid.dual_indep_insert_swap_iff (h : IsDeltaMatroid N a b c M) {s u w : E}
    (hX : ({u, w, s} : Set E) = {a, b, c}) (hus : u ≠ s) (hws : w ≠ s) (huw : u ≠ w) {I : Set E}
    (hsI : s ∉ I) : N✶.Indep (insert s (Equiv.swap u w '' I)) ↔ M✶.Indep I := by
  have hE := h.ground_eq
  have hXE : ({u, w, s} : Set E) ⊆ N.E := hX ▸ h.triangle.subset_ground
  have hu : u ∈ N.E := hXE (Or.inl rfl)
  have hw : w ∈ N.E := hXE (Or.inr (Or.inl rfl))
  have hs : s ∈ N.E := hXE (Or.inr (Or.inr rfl))
  by_cases hIE : I ⊆ N.E
  · have hIM : I ⊆ M.E := by
      rw [hE]
      exact hIE
    have hJE : insert s (Equiv.swap u w '' I) ⊆ N.E := by
      refine insert_subset hs ?_
      rintro _ ⟨e, he, rfl⟩
      exact (swap_mem_iff hu hw).2 (hIE he)
    have hT : N.IsCircuit {u, w, s} := hX ▸ h.triangle
    have hsK : s ∉ N.E \ insert s (Equiv.swap u w '' I) := fun h' => h'.2 (mem_insert s _)
    have hrk := deltaRk_insert_swap hT huw hus hws hsK
    rw [hX] at hrk
    rw [dual_indep_iff_natRk hJE, dual_indep_iff_natRk hIM, hE, h.natRk_eq sdiff_subset,
      h.natRk_eq subset_rfl, sdiff_eq_insert_swap hu hw hs hus hws hsI, hrk,
      deltaRk_of_subset h.triangle.subset_ground ⟨a, Or.inl rfl⟩]
    omega
  · refine iff_of_false (fun h' => hIE fun e he => ?_) fun h' => hIE (hE ▸ h'.subset_ground)
    exact (swap_mem_iff hu hw).1 (h'.subset_ground (mem_insert_of_mem s (mem_image_of_mem _ he)))

/-- **`Δ_X` of the dual.** If `M = Δ_X N` and `X` is a triad of `M`, then `Δ_X M* = N*`. -/
theorem IsDeltaMatroid.dual (h : IsDeltaMatroid N a b c M) (htriad : M✶.IsCircuit {a, b, c}) :
    IsDeltaMatroid M✶ a b c N✶ :=
  isDeltaMatroid_of_indep_iff htriad h.ne_ab h.ne_ac h.ne_bc
    (by rw [Matroid.dual_ground, Matroid.dual_ground, h.ground_eq])
    (fun _ hI => h.dual_indep_iff_of_disjoint hI)
    (fun _ hI => h.dual_indep_insert_swap_iff rfl h.ne_ac h.ne_bc h.ne_ab hI)
    (fun _ hI => h.dual_indep_insert_swap_iff triple_acb_eq h.ne_ab h.ne_bc.symm h.ne_ac hI)
    (fun _ hI => h.dual_indep_insert_swap_iff triple_bca_eq h.ne_ab.symm h.ne_ac.symm h.ne_bc hI)

/-- **`∇_X` undoes `Δ_X`.** If `M = Δ_X N` and `X` is a triad of `M`, then `N = ∇_X M`. -/
theorem IsDeltaMatroid.isNablaMatroid (h : IsDeltaMatroid N a b c M)
    (htriad : M✶.IsCircuit {a, b, c}) : IsNablaMatroid M a b c N :=
  ⟨N✶, h.dual htriad, (Matroid.dual_dual N).symm⟩

end Transformer.GainMatroid
