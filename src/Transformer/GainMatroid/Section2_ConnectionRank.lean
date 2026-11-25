module

public import Transformer.GainMatroid.Section2_ConnectionIndices
public import Transformer.GainMatroid.Section2_NatRank

/-!
# The rank function of the parallel connection with `M(K₄)`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, take the generalized parallel connection of a
matroid `M` and `M(K₄)` across a triangle `X = {a, b, c}` of `M` (Oxley, §11.4). On the elements of
`M` and three new elements `a′, b′, c′ = inr 0, 1, 2` its rank (`connRk`) is the rank in `M` of the
old elements together with the elements of `X` that the new ones span in `M(K₄)`, plus one if there
is a new element. The new elements with indices in `K` span the element `k` of `X` exactly when both
other indices lie in `K` (`spanIdx`).

With `X` a circuit of `M` this function is at most the size (`connRk_le_ncard`), monotone
(`connRk_mono`) and submodular (`connRk_submod`). For submodularity, the elements of `X` spanned by
the union of two sets of new elements lie in the closure of those spanned by each, together with one
more element of `X` when the two sets are disjoint and nonempty (`natRk_union_spanIdx_union`, from
facts about `Fin 3` decided over all pairs).
-/

@[expose] public section

namespace Transformer.GainMatroid

/-- The indices `k` whose two other indices lie in `K`: in `M(K₄)` the new elements with indices in
`K` span the element `k` of the triangle exactly for these (§2.4). -/
def spanIdx (K : Finset (Fin 3)) : Finset (Fin 3) :=
  Finset.univ.filter fun k => ∀ i, i ≠ k → i ∈ K

/-- The closure of a set of indices of the triangle: two of its elements span the third. -/
def triClosure (A : Finset (Fin 3)) : Finset (Fin 3) :=
  if 2 ≤ A.card then Finset.univ else A

/-- `1` for a nonempty set of indices, `0` for the empty one. -/
def nonemptyInd (K : Finset (Fin 3)) : ℕ := if K.Nonempty then 1 else 0

set_option maxRecDepth 10000 in
private theorem spanIdx_facts :
    (∀ K L : Finset (Fin 3), K ⊆ L → spanIdx K ⊆ spanIdx L ∧ nonemptyInd K ≤ nonemptyInd L) ∧
    (∀ K : Finset (Fin 3), (spanIdx K).card + nonemptyInd K ≤
      K.card + if spanIdx K = Finset.univ then 1 else 0) ∧
    (∀ K L : Finset (Fin 3), (spanIdx (K ∪ L) ⊆ triClosure (spanIdx K ∪ spanIdx L) ∧
        nonemptyInd (K ∪ L) + nonemptyInd (K ∩ L) ≤ nonemptyInd K + nonemptyInd L) ∨
      ∃ t, spanIdx (K ∪ L) ⊆ triClosure (insert t (spanIdx K ∪ spanIdx L)) ∧
        nonemptyInd (K ∪ L) + nonemptyInd (K ∩ L) + 1 ≤ nonemptyInd K + nonemptyInd L) ∧
    (∀ (A : Finset (Fin 3)) (k : Fin 3), 2 ≤ A.card → k ∉ A → ∀ j, j ≠ k → j ∈ A) := by
  decide +kernel

theorem spanIdx_mono {K L : Finset (Fin 3)} (h : K ⊆ L) : spanIdx K ⊆ spanIdx L :=
  (spanIdx_facts.1 K L h).1

theorem nonemptyInd_mono {K L : Finset (Fin 3)} (h : K ⊆ L) : nonemptyInd K ≤ nonemptyInd L :=
  (spanIdx_facts.1 K L h).2

variable {E : Type*} {M : Matroid E} {a b c : E}

/-- In a triangle, the elements with indices in the closure of `A` lie in the closure of those with
indices in `A`. -/
theorem image_triClosure_subset (hT : M.IsCircuit {a, b, c}) (A : Finset (Fin 3)) :
    ![a, b, c] '' ↑(triClosure A) ⊆ M.closure (![a, b, c] '' ↑A) := by
  have hE : ![a, b, c] '' ↑A ⊆ M.E := by
    rintro _ ⟨k, -, rfl⟩
    exact hT.subset_ground (triple_mem a b c k)
  unfold triClosure
  split_ifs with h
  · rintro _ ⟨k, -, rfl⟩
    by_cases hk : k ∈ A
    · exact M.subset_closure _ hE ⟨k, hk, rfl⟩
    · have hsub : ({a, b, c} : Set E) \ {![a, b, c] k} ⊆ ![a, b, c] '' ↑A := by
        rintro e ⟨he, hek⟩
        obtain ⟨j, rfl⟩ := exists_triple_eq he
        exact ⟨j, spanIdx_facts.2.2.2 A k h hk j fun hjk => hek (by rw [hjk]; rfl), rfl⟩
      exact M.closure_subset_closure hsub
        (hT.subset_closure_sdiff_singleton (![a, b, c] k) (triple_mem a b c k))
  · exact M.subset_closure _ hE

/-- **The rank function of the generalized parallel connection** of `M` and `M(K₄)` across the
triangle `{a, b, c}` (§2.4): the rank in `M` of the old elements together with the elements of the
triangle that the new elements span, plus one if there is a new element. -/
noncomputable def connRk (M : Matroid E) (a b c : E) (Z : Set (E ⊕ Fin 3)) : ℕ :=
  natRk M (Sum.inl ⁻¹' Z ∪ ![a, b, c] '' ↑(spanIdx (newIdx Z))) + nonemptyInd (newIdx Z)

variable [Finite E]

theorem connRk_mono {Z W : Set (E ⊕ Fin 3)} (h : Z ⊆ W) : connRk M a b c Z ≤ connRk M a b c W :=
  Nat.add_le_add (natRk_mono M (Set.union_subset_union (Set.preimage_mono h)
    (Set.image_mono (Finset.coe_subset.2 (spanIdx_mono (newIdx_mono h))))))
    (nonemptyInd_mono (newIdx_mono h))

theorem connRk_le_ncard (hT : M.IsCircuit {a, b, c}) (Z : Set (E ⊕ Fin 3)) :
    connRk M a b c Z ≤ Z.ncard := by
  have h1 := natRk_union_le M (Sum.inl ⁻¹' Z) (![a, b, c] '' ↑(spanIdx (newIdx Z)))
  have h2 := natRk_le_ncard M (Sum.inl ⁻¹' Z)
  have h3 : natRk M (![a, b, c] '' ↑(spanIdx (newIdx Z))) +
      (if spanIdx (newIdx Z) = Finset.univ then 1 else 0) ≤ (spanIdx (newIdx Z)).card := by
    split_ifs with hK
    · have himage : ![a, b, c] '' ↑(spanIdx (newIdx Z)) = ({a, b, c} : Set E) := by
        rw [hK, Finset.coe_univ, Set.image_univ]
        ext e
        exact ⟨fun ⟨k, hk⟩ => hk ▸ triple_mem a b c k, exists_triple_eq⟩
      have h4 := natRk_add_one_of_isCircuit hT
      have h5 : ({a, b, c} : Set E).ncard ≤ 3 :=
        (Set.ncard_insert_le _ _).trans (Nat.succ_le_succ ((Set.ncard_insert_le _ _).trans
          (by rw [Set.ncard_singleton])))
      rw [himage, hK, Finset.card_univ, Fintype.card_fin]
      omega
    · exact (natRk_le_ncard _ _).trans ((Set.ncard_image_le (Finset.finite_toSet _)).trans
        (Set.ncard_coe_finset _).le)
  have h4 := spanIdx_facts.2.1 (newIdx Z)
  rw [connRk, ncard_eq_inl_add_newIdx Z]
  split_ifs at h3 h4 <;> omega

/-- The elements of the triangle spanned by the union of two sets of new elements add to the rank
of the old elements and of those spanned by each set at most what the indicator of their union
and intersection takes away. -/
theorem natRk_union_spanIdx_union (hT : M.IsCircuit {a, b, c}) (X : Set E)
    (K L : Finset (Fin 3)) :
    natRk M (X ∪ ![a, b, c] '' ↑(spanIdx (K ∪ L))) + nonemptyInd (K ∪ L) + nonemptyInd (K ∩ L) ≤
      natRk M (X ∪ ![a, b, c] '' ↑(spanIdx K ∪ spanIdx L)) + nonemptyInd K + nonemptyInd L := by
  set A := spanIdx K ∪ spanIdx L
  have hcl (B : Finset (Fin 3)) (hB : spanIdx (K ∪ L) ⊆ triClosure B) :
      natRk M (X ∪ ![a, b, c] '' ↑(spanIdx (K ∪ L))) ≤ natRk M (X ∪ ![a, b, c] '' ↑B) :=
    calc natRk M (X ∪ ![a, b, c] '' ↑(spanIdx (K ∪ L)))
        ≤ natRk M ((X ∪ ![a, b, c] '' ↑B) ∪ ![a, b, c] '' ↑(triClosure B)) :=
          natRk_mono M (Set.union_subset_union Set.subset_union_left
            (Set.image_mono (Finset.coe_subset.2 hB)))
      _ = natRk M (X ∪ ![a, b, c] '' ↑B) := natRk_union_of_subset_closure M
          ((image_triClosure_subset hT B).trans (M.closure_subset_closure Set.subset_union_right))
  rcases spanIdx_facts.2.2.1 K L with ⟨hsub, hind⟩ | ⟨t, hsub, hind⟩
  · have := hcl A hsub
    omega
  · have h1 := hcl (insert t A) hsub
    have h2 : natRk M (X ∪ ![a, b, c] '' ↑(insert t A)) ≤
        natRk M (X ∪ ![a, b, c] '' ↑A) + 1 := by
      rw [Finset.coe_insert, Set.image_insert_eq, Set.union_insert]
      exact natRk_insert_le M _ _
    omega

theorem connRk_submod (hT : M.IsCircuit {a, b, c}) (Z W : Set (E ⊕ Fin 3)) :
    connRk M a b c (Z ∩ W) + connRk M a b c (Z ∪ W) ≤ connRk M a b c Z + connRk M a b c W := by
  unfold connRk
  rw [newIdx_inter, newIdx_union, Set.preimage_inter, Set.preimage_union]
  generalize newIdx Z = K
  generalize newIdx W = L
  have key := natRk_union_spanIdx_union hT (Sum.inl ⁻¹' Z ∪ Sum.inl ⁻¹' W) K L
  have hsub := natRk_submod M (Sum.inl ⁻¹' Z ∪ ![a, b, c] '' ↑(spanIdx K))
    (Sum.inl ⁻¹' W ∪ ![a, b, c] '' ↑(spanIdx L))
  have hmono : natRk M (Sum.inl ⁻¹' Z ∩ Sum.inl ⁻¹' W ∪ ![a, b, c] '' ↑(spanIdx (K ∩ L))) ≤
      natRk M ((Sum.inl ⁻¹' Z ∪ ![a, b, c] '' ↑(spanIdx K)) ∩
        (Sum.inl ⁻¹' W ∪ ![a, b, c] '' ↑(spanIdx L))) := by
    refine natRk_mono M (Set.union_subset (Set.inter_subset_inter Set.subset_union_left
      Set.subset_union_left) (Set.subset_inter ?_ ?_))
    · exact (Set.image_mono (Finset.coe_subset.2 (spanIdx_mono Finset.inter_subset_left))).trans
        Set.subset_union_right
    · exact (Set.image_mono (Finset.coe_subset.2 (spanIdx_mono Finset.inter_subset_right))).trans
        Set.subset_union_right
  have hunion : (Sum.inl ⁻¹' Z ∪ ![a, b, c] '' ↑(spanIdx K)) ∪
      (Sum.inl ⁻¹' W ∪ ![a, b, c] '' ↑(spanIdx L)) =
      (Sum.inl ⁻¹' Z ∪ Sum.inl ⁻¹' W) ∪ ![a, b, c] '' ↑(spanIdx K ∪ spanIdx L) := by
    rw [Finset.coe_union, Set.image_union]
    ext e
    simp only [Set.mem_union]
    tauto
  rw [hunion] at hsub
  omega

end Transformer.GainMatroid
