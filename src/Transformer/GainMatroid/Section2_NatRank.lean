module

public import Mathlib.Combinatorics.Matroid.Map
public import Mathlib.Combinatorics.Matroid.Rank.ENat

/-!
# Ranks of finite matroids as natural numbers

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2, take matroids as in Oxley's text, which are
finite. Mathlib's rank `Matroid.eRk` takes values in `ℕ∞`. On a finite type it is finite, and
`natRk` reads it as a natural number, so that rank computations become arithmetic in `ℕ`:
independence (`indep_iff_natRk`), monotonicity (`natRk_mono`), submodularity (`natRk_submod`),
closure (`natRk_union_of_subset_closure`, `natRk_insert_of_notMem_closure`) and flats
(`isFlat_iff_natRk`) read through it.

A monotone submodular function that does not grow by adding any single element of a finite set does
not grow by adding the whole set (`eq_of_forall_insert_eq`).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {α β : Type*} {M : Matroid α} {X Y I F C : Set α} {e : α}

/-- The rank of a set in a matroid as a natural number: the value of `Matroid.eRk`, finite on a
finite type. -/
noncomputable def natRk (M : Matroid α) (X : Set α) : ℕ := (M.eRk X).toNat

/-- The rank of a set in a comap is the rank of its image. -/
theorem natRk_comap (M : Matroid β) (f : α → β) (X : Set α) :
    natRk (M.comap f) X = natRk M (f '' X) := by
  rw [natRk, natRk, Matroid.eRk_comap]

/-- A monotone submodular function that is unchanged by adding each element of a finite set `D` is
unchanged by adding all of `D`. -/
theorem eq_of_forall_insert_eq {r : Set β → ℕ} (hmono : ∀ X Y, X ⊆ Y → r X ≤ r Y)
    (hsub : ∀ X Y, r (X ∩ Y) + r (X ∪ Y) ≤ r X + r Y) {Y D : Set β} (hD : D.Finite)
    (h : ∀ e ∈ D, r (insert e Y) = r Y) : r (Y ∪ D) = r Y := by
  induction D, hD using Set.Finite.induction_on with
  | empty => rw [Set.union_empty]
  | @insert e D heD _ ih =>
    have hY := h e (Set.mem_insert e D)
    have ih := ih fun f hf => h f (Set.mem_insert_of_mem e hf)
    have hinter : (Y ∪ D) ∩ insert e Y = Y := by
      ext f
      simp only [Set.mem_inter_iff, Set.mem_union, Set.mem_insert_iff]
      constructor
      · rintro ⟨hf | hf, rfl | hf'⟩
        · exact hf
        · exact hf
        · exact absurd hf heD
        · exact hf'
      · intro hf
        exact ⟨Or.inl hf, Or.inr hf⟩
    have hunion : (Y ∪ D) ∪ insert e Y = Y ∪ insert e D := by
      ext f
      simp only [Set.mem_union, Set.mem_insert_iff]
      tauto
    have := hsub (Y ∪ D) (insert e Y)
    rw [hinter, hunion, ih, hY] at this
    exact le_antisymm (by omega) (hmono _ _ Set.subset_union_left)

variable [Finite α]

theorem natRk_cast (M : Matroid α) (X : Set α) : (natRk M X : ℕ∞) = M.eRk X :=
  ENat.natCast_toNat ((M.eRk_le_encard X).trans_lt X.toFinite.encard_lt_top).ne

theorem natRk_le_ncard (M : Matroid α) (X : Set α) : natRk M X ≤ X.ncard := by
  have h := M.eRk_le_encard X
  rw [← natRk_cast, ← X.toFinite.cast_ncard_eq] at h
  exact_mod_cast h

theorem natRk_mono (M : Matroid α) (h : X ⊆ Y) : natRk M X ≤ natRk M Y := by
  have := M.eRk_mono h
  rw [← natRk_cast, ← natRk_cast] at this
  exact_mod_cast this

theorem natRk_submod (M : Matroid α) (X Y : Set α) :
    natRk M (X ∩ Y) + natRk M (X ∪ Y) ≤ natRk M X + natRk M Y := by
  have := M.eRk_inter_add_eRk_union_le X Y
  rw [← natRk_cast, ← natRk_cast, ← natRk_cast, ← natRk_cast] at this
  exact_mod_cast this

theorem natRk_union_le (M : Matroid α) (X Y : Set α) :
    natRk M (X ∪ Y) ≤ natRk M X + natRk M Y :=
  le_trans (Nat.le_add_left _ _) (natRk_submod M X Y)

theorem natRk_insert_le (M : Matroid α) (e : α) (X : Set α) :
    natRk M (insert e X) ≤ natRk M X + 1 := by
  have := M.eRk_insert_le_add_one e X
  rw [← natRk_cast, ← natRk_cast] at this
  exact_mod_cast this

/-- Adding a subset of the closure keeps the rank. -/
theorem natRk_union_of_subset_closure (M : Matroid α) (h : Y ⊆ M.closure X) :
    natRk M (X ∪ Y) = natRk M X := by
  refine le_antisymm ?_ (natRk_mono M Set.subset_union_left)
  calc natRk M (X ∪ Y) ≤ natRk M (X ∪ M.closure X) :=
        natRk_mono M (Set.union_subset_union_right X h)
    _ = natRk M X := by rw [natRk, natRk, M.eRk_union_closure_right_eq, Set.union_self]

theorem natRk_insert_of_notMem_closure (he : e ∈ M.E \ M.closure X) :
    natRk M (insert e X) = natRk M X + 1 := by
  have := Matroid.eRk_insert_eq_add_one he
  rw [← natRk_cast, ← natRk_cast] at this
  exact_mod_cast this

theorem indep_iff_natRk : M.Indep I ↔ natRk M I = I.ncard := by
  rw [Matroid.indep_iff_eRk_eq_encard_of_finite I.toFinite, ← natRk_cast,
    ← I.toFinite.cast_ncard_eq]
  exact Nat.cast_inj

theorem mem_closure_iff_natRk (he : e ∈ M.E) :
    e ∈ M.closure X ↔ natRk M (insert e X) = natRk M X := by
  refine ⟨fun h => ?_, fun h => by_contra fun hcl => ?_⟩
  · rw [Set.insert_eq, Set.union_comm]
    exact natRk_union_of_subset_closure M (Set.singleton_subset_iff.2 h)
  · rw [natRk_insert_of_notMem_closure ⟨he, hcl⟩] at h
    omega

/-- A set is a flat exactly when adding any element of the ground set outside it raises its
rank. -/
theorem isFlat_iff_natRk :
    M.IsFlat F ↔ F ⊆ M.E ∧ ∀ e ∈ M.E, e ∉ F → natRk M F < natRk M (insert e F) := by
  rw [Matroid.isFlat_iff_closure_eq]
  constructor
  · intro h
    refine ⟨h ▸ M.closure_subset_ground F, fun e he heF => ?_⟩
    rw [natRk_insert_of_notMem_closure ⟨he, by rw [h]; exact heF⟩]
    omega
  · rintro ⟨hF, h⟩
    refine Set.Subset.antisymm (fun e he => by_contra fun heF => ?_) (M.subset_closure F hF)
    have := h e (M.closure_subset_ground F he) heF
    rw [(mem_closure_iff_natRk (M.closure_subset_ground F he)).1 he] at this
    exact lt_irrefl _ this

/-- The rank of a set is the size of any of its bases. -/
theorem natRk_eq_ncard_of_isBasis (h : M.IsBasis I X) : natRk M X = I.ncard := by
  have := h.eRk_eq_encard
  rw [← natRk_cast, ← I.toFinite.cast_ncard_eq] at this
  exact_mod_cast this

/-- A circuit has rank one less than its size. -/
theorem natRk_add_one_of_isCircuit (h : M.IsCircuit C) : natRk M C + 1 = C.ncard := by
  have := h.eRk_add_one_eq
  rw [← natRk_cast, ← C.toFinite.cast_ncard_eq] at this
  exact_mod_cast this

end Transformer.GainMatroid
