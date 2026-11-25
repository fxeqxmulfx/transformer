module

public import Transformer.GainMatroid.Section2_ConnectionRank

/-!
# The generalized parallel connection with `M(K₄)`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4: "take the generalized parallel connection of
`M` and `M(K₄)` across `X`", citing Oxley, §11.4. For a triangle `X = {a, b, c}` of a finite matroid
`M`, `deltaConnection` is the matroid on the elements of `M` and three new elements
`a′, b′, c′ = inr 0, 1, 2` whose independent sets are the sets with rank `connRk` equal to their
size. As `connRk` is at most the size, monotone and submodular (`Section2_ConnectionRank`), these
sets satisfy the axioms of independence, and `connRk` is the rank function of the matroid
(`natRk_deltaConnection`).

Its restriction to the elements of `M` is `M` (`deltaConnection_comap_inl`), and its restriction to
`X` and the new elements is `M(K₄)` with the labels of §2.4 (`deltaConnection_comap_deltaEmbed`):
there the rank of a set is that of its triangle part with the elements its new part spans, at most
two, plus one if it has a new element, which matches `k4Indep`.
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {E : Type*} {M : Matroid E} {a b c : E}

/-- The old elements of a set of old elements are the set, and it has no new elements. -/
theorem connRk_image_inl (J : Set E) : connRk M a b c (Sum.inl '' J) = natRk M J := by
  have hK : newIdx (Sum.inl '' J : Set (E ⊕ Fin 3)) = ∅ := by
    ext i
    simp [mem_newIdx]
  rw [connRk, hK, Set.preimage_image_eq _ Sum.inl_injective]
  have : spanIdx ∅ = ∅ := by decide
  simp [this, nonemptyInd]

variable [Finite E]

/-- The elements of `M` and the three new elements. -/
def connGround (M : Matroid E) : Set (E ⊕ Fin 3) := Sum.inl '' M.E ∪ Set.range Sum.inr

theorem connRk_eq_of_forall_insert (hT : M.IsCircuit {a, b, c}) {Y D : Set (E ⊕ Fin 3)}
    (h : ∀ e ∈ D, connRk M a b c (insert e Y) = connRk M a b c Y) :
    connRk M a b c (Y ∪ D) = connRk M a b c Y :=
  eq_of_forall_insert_eq (fun _ _ => connRk_mono) (connRk_submod hT) D.toFinite h

theorem connRk_insert_le (hT : M.IsCircuit {a, b, c}) (e : E ⊕ Fin 3) (Y : Set (E ⊕ Fin 3)) :
    connRk M a b c (insert e Y) ≤ connRk M a b c Y + 1 := by
  have h1 := connRk_submod hT Y {e}
  have h2 := connRk_le_ncard hT {e}
  rw [Set.union_singleton, Set.ncard_singleton] at *
  omega

/-- A set of rank equal to its size, in a monotone submodular rank, keeps this for its subsets. -/
theorem connRk_eq_ncard_of_subset (hT : M.IsCircuit {a, b, c}) {I J : Set (E ⊕ Fin 3)}
    (hJ : connRk M a b c J = J.ncard) (hIJ : I ⊆ J) : connRk M a b c I = I.ncard := by
  have h1 := connRk_submod hT I (J \ I)
  have h2 := connRk_le_ncard hT (J \ I)
  have h3 := connRk_le_ncard hT I
  have h4 := Set.ncard_sdiff_add_ncard_of_subset hIJ
  rw [Set.inter_sdiff_self, Set.union_sdiff_cancel hIJ] at h1
  omega

/-- If no element of `J` can be added to `I` without the rank falling short of the size, then `J`
is no larger than `I`. -/
theorem ncard_le_of_forall_insert (hT : M.IsCircuit {a, b, c}) {I J : Set (E ⊕ Fin 3)}
    (hI : connRk M a b c I = I.ncard) (hJ : connRk M a b c J = J.ncard)
    (h : ∀ e ∈ J, e ∉ I → connRk M a b c (insert e I) ≠ (insert e I).ncard) :
    J.ncard ≤ I.ncard := by
  have hins : ∀ e ∈ J, connRk M a b c (insert e I) = connRk M a b c I := by
    intro e he
    by_cases heI : e ∈ I
    · rw [Set.insert_eq_of_mem heI]
    · have h1 := h e he heI
      have h2 := connRk_insert_le hT e I
      have h3 := connRk_mono (M := M) (a := a) (b := b) (c := c) (Set.subset_insert e I)
      rw [Set.ncard_insert_of_notMem heI] at h1
      omega
  have := connRk_eq_of_forall_insert hT hins
  have := connRk_mono (M := M) (a := a) (b := b) (c := c) (Set.subset_union_right (s := I) (t := J))
  omega

/-- **The generalized parallel connection** of `M` and `M(K₄)` across the triangle `{a, b, c}`
(§2.4, after Oxley §11.4): the sets of elements of `M` and new elements whose rank `connRk` is their
size. -/
noncomputable def deltaConnection (M : Matroid E) (a b c : E) (hT : M.IsCircuit {a, b, c}) :
    Matroid (E ⊕ Fin 3) :=
  (IndepMatroid.ofFinite (E := connGround M) (Set.toFinite _)
    (fun Z => Z ⊆ connGround M ∧ connRk M a b c Z = Z.ncard)
    ⟨Set.empty_subset _, by have := connRk_le_ncard hT ∅; rw [Set.ncard_empty] at *; omega⟩
    (fun _ _ hJ hIJ => ⟨hIJ.trans hJ.1, connRk_eq_ncard_of_subset hT hJ.2 hIJ⟩)
    (fun I J hI hJ hlt => by
      by_contra hcon
      exact absurd hlt (not_lt.2 (ncard_le_of_forall_insert hT hI.2 hJ.2 fun e he heI hr =>
        hcon ⟨e, he, heI, Set.insert_subset (hJ.1 he) hI.1, hr⟩)))
    (fun _ hI => hI.1)).matroid

theorem deltaConnection_E (hT : M.IsCircuit {a, b, c}) :
    (deltaConnection M a b c hT).E = connGround M := rfl

theorem deltaConnection_indep_iff (hT : M.IsCircuit {a, b, c}) {Z : Set (E ⊕ Fin 3)} :
    (deltaConnection M a b c hT).Indep Z ↔ Z ⊆ connGround M ∧ connRk M a b c Z = Z.ncard :=
  Iff.rfl

/-- `connRk` is the rank function of the generalized parallel connection. -/
theorem natRk_deltaConnection (hT : M.IsCircuit {a, b, c}) {Z : Set (E ⊕ Fin 3)}
    (hZ : Z ⊆ connGround M) : natRk (deltaConnection M a b c hT) Z = connRk M a b c Z := by
  obtain ⟨I, hI⟩ := (deltaConnection M a b c hT).exists_isBasis Z hZ
  have hIr := ((deltaConnection_indep_iff hT).1 hI.indep).2
  rw [natRk_eq_ncard_of_isBasis hI, ← hIr, ← Set.union_eq_self_of_subset_left hI.subset]
  refine (connRk_eq_of_forall_insert hT fun e he => ?_).symm
  by_cases heI : e ∈ I
  · rw [Set.insert_eq_of_mem heI]
  · have hdep := hI.insert_dep ⟨he, heI⟩
    have h1 : connRk M a b c (insert e I) ≠ (insert e I).ncard := fun h =>
      hdep.not_indep ⟨Set.insert_subset (hZ he) (hI.subset.trans hZ), h⟩
    have h2 := connRk_insert_le hT e I
    have h3 := connRk_mono (M := M) (a := a) (b := b) (c := c) (Set.subset_insert e I)
    rw [Set.ncard_insert_of_notMem heI] at h1
    omega

/-- **The generalized parallel connection restricts to `M`** on the elements of `M`. -/
theorem deltaConnection_comap_inl (hT : M.IsCircuit {a, b, c}) :
    (deltaConnection M a b c hT).comap Sum.inl = M := by
  refine Matroid.ext_indep ?_ fun I hI => ?_
  · ext e
    simp [deltaConnection_E, connGround]
  · rw [Matroid.comap_ground_eq, deltaConnection_E] at hI
    have hIE : I ⊆ M.E := fun e he => by simpa [connGround] using hI he
    rw [Matroid.comap_indep_iff, deltaConnection_indep_iff, connRk_image_inl,
      Set.ncard_image_of_injective _ Sum.inl_injective, indep_iff_natRk]
    exact ⟨fun h => h.1.2, fun h => ⟨⟨Set.image_mono hIE |>.trans Set.subset_union_left, h⟩,
      Sum.inl_injective.injOn⟩⟩

end Transformer.GainMatroid
