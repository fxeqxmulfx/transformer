module

public import Transformer.GainMatroid.Section2_DeltaRank

/-!
# A Δ-Y exchange through its independent sets

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. Let `X = {a, b, c}` be a triangle of a finite
matroid `M`. A matroid `M'` on the same elements is `Δ_X M` (`IsDeltaMatroid`) when its independent
sets off `X` are those of `M`, and for each element `s` of `X`, its independent sets through `s` are
those of `M \ s` with the other two elements of `X` exchanged and `s` added
(`isDeltaMatroid_of_indep_iff`): contracting `s` in `Δ_X M` gives `M \ s` with these two exchanged.

The witness is `deltaConnection`, whose flats are those that `IsDeltaMatroid` asks for
(`isFlat_deltaConnection_iff`). Its relabelled deletion of `X` has as independent sets those of rank
`deltaRk` equal to their size (`connRk_image_deltaRelabel`), and so has `M'`: every set through
an element `s` of `X` is `s` added to the exchange of a set avoiding `s`
(`exists_eq_insert_swap`), whose rank `deltaRk_insert_swap` gives, and every other set misses `X`
(`deltaRk_of_disjoint`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open Set

theorem triple_acb_eq {E : Type*} {a b c : E} : ({a, c, b} : Set E) = {a, b, c} := by
  rw [pair_comm c b]

theorem triple_bca_eq {E : Type*} {a b c : E} : ({b, c, a} : Set E) = {a, b, c} := by
  rw [pair_comm c a, insert_comm b a]

variable {E : Type*} [DecidableEq E] {M M' : Matroid E} {a b c : E}

/-- A set through `s` is `s` added to the image, under the exchange of `u` and `w`, of a set
avoiding `s`. -/
theorem exists_eq_insert_swap {I : Set E} {s u w : E} (hs : s ∈ I) (hsu : s ≠ u) (hsw : s ≠ w) :
    ∃ J, s ∉ J ∧ I = insert s (Equiv.swap u w '' J) := by
  refine ⟨Equiv.swap u w '' (I \ {s}), ?_, ?_⟩
  · rw [mem_image_equiv, Equiv.symm_swap, Equiv.swap_apply_of_ne_of_ne hsu hsw]
    exact fun h => h.2 rfl
  · rw [image_image]
    simp only [Equiv.swap_apply_self, image_id']
    rw [insert_sdiff_singleton, insert_eq_of_mem hs]

/-- Adding `s` to the exchange of a set avoiding `s` adds one element. -/
theorem ncard_insert_swap [Finite E] {J : Set E} {s u w : E} (hsJ : s ∉ J) (hsu : s ≠ u)
    (hsw : s ≠ w) : (insert s (Equiv.swap u w '' J)).ncard = J.ncard + 1 := by
  rw [ncard_insert_of_notMem, ncard_image_of_injective _ (Equiv.injective _)]
  rw [mem_image_equiv, Equiv.symm_swap, Equiv.swap_apply_of_ne_of_ne hsu hsw]
  exact hsJ

/-- The relabelled deletion of `X` has the elements of `M`. -/
theorem deltaRelabel_preimage_connGround (hT : M.IsCircuit {a, b, c}) :
    deltaRelabel a b c ⁻¹' connGround M = M.E := by
  have hX : ({a, b, c} : Set E) ⊆ M.E := hT.subset_ground
  ext e
  rw [mem_preimage, connGround]
  unfold deltaRelabel
  split_ifs with h1 h2 h3
  · simpa [h1] using hX (Or.inl rfl)
  · simpa [h2] using hX (Or.inr (Or.inl rfl))
  · simpa [h3] using hX (Or.inr (Or.inr rfl))
  · simp

variable [Finite E]

/-- **A Δ-Y exchange from its independent sets** (§2.4). If the independent sets of `M'` off the
triangle `X = {a, b, c}` are those of `M`, and through each element `s` of `X` they are the sets of
`M \ s` with the other two elements of `X` exchanged and `s` added, then `M' = Δ_X M`. -/
theorem isDeltaMatroid_of_indep_iff (hT : M.IsCircuit {a, b, c}) (hab : a ≠ b) (hac : a ≠ c)
    (hbc : b ≠ c) (hE : M'.E = M.E)
    (h₀ : ∀ I, Disjoint I {a, b, c} → (M'.Indep I ↔ M.Indep I))
    (hc : ∀ I, c ∉ I → (M'.Indep (insert c (Equiv.swap a b '' I)) ↔ M.Indep I))
    (hb : ∀ I, b ∉ I → (M'.Indep (insert b (Equiv.swap a c '' I)) ↔ M.Indep I))
    (ha : ∀ I, a ∉ I → (M'.Indep (insert a (Equiv.swap b c '' I)) ↔ M.Indep I)) :
    IsDeltaMatroid M a b c M' := by
  refine ⟨hab, hac, hbc, hT, deltaConnection M a b c hT, rfl,
    isFlat_deltaConnection_iff hT hab hac hbc, ?_⟩
  have hρ := deltaRelabel_injective a b c
  refine Matroid.ext_indep ?_ fun I hI => ?_
  · rw [hE, Matroid.comap_ground_eq, deltaConnection_E, deltaRelabel_preimage_connGround hT]
  rw [hE] at hI
  have hsub : deltaRelabel a b c '' I ⊆ connGround M := by
    rw [image_subset_iff, deltaRelabel_preimage_connGround hT]
    exact hI
  rw [Matroid.comap_indep_iff, deltaConnection_indep_iff, connRk_image_deltaRelabel hab hac hbc,
    ncard_image_of_injective _ hρ, and_iff_left hρ.injOn, and_iff_right hsub]
  have hcase (s u w : E) (J : Set E) (hsJ : s ∉ J) (hsu : s ≠ u) (hsw : s ≠ w)
      (hind : M'.Indep (insert s (Equiv.swap u w '' J)) ↔ M.Indep J)
      (hrk : deltaRk M {a, b, c} (insert s (Equiv.swap u w '' J)) = natRk M J + 1) :
      M'.Indep (insert s (Equiv.swap u w '' J)) ↔
        deltaRk M {a, b, c} (insert s (Equiv.swap u w '' J)) =
          (insert s (Equiv.swap u w '' J)).ncard := by
    rw [hind, hrk, ncard_insert_swap hsJ hsu hsw, indep_iff_natRk]
    omega
  by_cases hcI : c ∈ I
  · obtain ⟨J, hcJ, rfl⟩ := exists_eq_insert_swap hcI hac.symm hbc.symm
    exact hcase c a b J hcJ hac.symm hbc.symm (hc J hcJ) (deltaRk_insert_swap hT hab hac hbc hcJ)
  by_cases hbI : b ∈ I
  · obtain ⟨J, hbJ, rfl⟩ := exists_eq_insert_swap hbI hab.symm hbc
    have hT' : M.IsCircuit {a, c, b} := by rwa [triple_acb_eq]
    have hrk := deltaRk_insert_swap hT' hac hab hbc.symm hbJ
    rw [triple_acb_eq] at hrk
    exact hcase b a c J hbJ hab.symm hbc (hb J hbJ) hrk
  by_cases haI : a ∈ I
  · obtain ⟨J, haJ, rfl⟩ := exists_eq_insert_swap haI hab hac
    have hT' : M.IsCircuit {b, c, a} := by rwa [triple_bca_eq]
    have hrk := deltaRk_insert_swap hT' hbc hab.symm hac.symm haJ
    rw [triple_bca_eq] at hrk
    exact hcase a b c J haJ hab hac (ha J haJ) hrk
  have hdisj : Disjoint I {a, b, c} := by
    rw [disjoint_left]
    rintro e he (rfl | rfl | rfl)
    exacts [haI he, hbI he, hcI he]
  rw [h₀ I hdisj, deltaRk_of_disjoint hab hdisj, indep_iff_natRk]

end Transformer.GainMatroid
