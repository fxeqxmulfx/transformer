module

public import Mathlib.Algebra.Field.ZMod
public import Transformer.GainMatroid.Section2_MatroidExchange

/-!
# Matrix representations of a matroid, and exchanges of uniform matroids

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. Proposition 2.9 compares the matrices that
represent a matroid `M` with those that represent an exchange of `M`, up to the projective
equivalence of §1.1. This file defines these matrices and their classes.

`isDeltaMatroid_of_finset` decides a Δ-Y exchange of finite matroids from their independent sets.
It checks the exchanges of `Section2_RepresentationExchange`.
-/

@[expose] public section

namespace Transformer.GainMatroid

section Representation

variable {E R F : Type*} [Field F]

/-- The matrix `A` represents `M`: a set of elements is independent in `M` exactly when its columns
are linearly independent. -/
def IsMatrixRepresentation (M : Matroid E) (A : Matrix R E F) : Prop :=
  M.E = Set.univ ∧ ∀ I : Set E, M.Indep I ↔ LinearIndepOn F (fun e r => A r e) I

variable (F) in
/-- `M` is `F`-representable. -/
def IsRepresentable (M : Matroid E) : Prop :=
  ∃ (m : ℕ) (A : Matrix (Fin m) E F), IsMatrixRepresentation M A

variable (F) in
/-- The `F`-matrices representing `M`, of any number of rows. -/
def RepMatrix (M : Matroid E) : Type _ :=
  Σ m : ℕ, {A : Matrix (Fin m) E F // IsMatrixRepresentation M A}

variable (F) in
/-- Projective equivalence on the `F`-matrices representing `M`. -/
def repSetoid (M : Matroid E) : Setoid (RepMatrix F M) where
  r A B := ProjEquiv A.2.1 B.2.1
  iseqv := ⟨fun _ => ProjEquiv.refl _, fun h => h.symm, fun h h' => h.trans h'⟩

variable (F) in
/-- The projective equivalence classes of `F`-representations of `M`. -/
def ProjClass (M : Matroid E) : Type _ := Quotient (repSetoid F M)

/-- A matroid has a class of `F`-representations exactly when it is `F`-representable. -/
theorem nonempty_projClass_iff (M : Matroid E) :
    Nonempty (ProjClass F M) ↔ IsRepresentable F M := by
  constructor
  · rintro ⟨c⟩
    obtain ⟨⟨m, A, hA⟩, -⟩ := Quotient.exists_rep c
    exact ⟨m, A, hA⟩
  · rintro ⟨m, A, hA⟩
    exact ⟨Quotient.mk _ ⟨m, A, hA⟩⟩

end Representation

section Uniform

variable {α : Type*} [DecidableEq α]

variable (α) in
/-- The uniform matroid of rank `r`, from its finite independent sets. -/
noncomputable def finsetUniform (r : ℕ) : Matroid α :=
  matroidOfFinset (fun I => I.card ≤ r) (by simp)
    (fun _ _ hJ hIJ => (Finset.card_le_card hIJ).trans hJ) fun I J _ hJ hlt => by
      obtain ⟨e, heJ, heI⟩ := Finset.exists_mem_notMem_of_card_lt_card hlt
      exact ⟨e, heJ, heI, by rw [Finset.card_insert_of_notMem heI]; omega⟩

theorem finsetUniform_indep (r : ℕ) (I : Finset α) :
    (finsetUniform α r).Indep (I : Set α) ↔ I.card ≤ r :=
  matroidOfFinset_indep I

omit [DecidableEq α] in
/-- A preimage of a flat is a flat of the comap. -/
theorem isFlat_comap_preimage {β : Type*} {P : Matroid β} {S : Set β} (hS : P.IsFlat S)
    (f : α → β) (hE : P.E = Set.univ) : (P.comap f).IsFlat (f ⁻¹' S) := by
  rw [Matroid.isFlat_iff_closure_eq, Matroid.comap_closure_eq]
  refine Set.Subset.antisymm (Set.preimage_mono ?_) fun x hx => ?_
  · exact (P.closure_subset_closure (Set.image_preimage_subset f S)).trans hS.closure.subset
  · exact P.subset_closure (f '' (f ⁻¹' S)) (hE ▸ Set.subset_univ _) ⟨x, hx, rfl⟩

/-- A matroid built from independent sets is a comap when they are read through an injection. -/
theorem matroidOfFinset_eq_comap {β : Type*} [Finite α] [DecidableEq β] {i : Finset α → Prop}
    {j : Finset β → Prop} {h₁ : i ∅} {s₁ : ∀ I J : Finset α, i J → I ⊆ J → i I}
    {g₁ : ∀ I J : Finset α, i I → i J → I.card < J.card → ∃ e ∈ J, e ∉ I ∧ i (insert e I)}
    {h₂ : j ∅} {s₂ : ∀ I J : Finset β, j J → I ⊆ J → j I}
    {g₂ : ∀ I J : Finset β, j I → j J → I.card < J.card → ∃ e ∈ J, e ∉ I ∧ j (insert e I)}
    {f : α → β} (hf : Function.Injective f) (h : ∀ I : Finset α, i I ↔ j (I.image f)) :
    matroidOfFinset i h₁ s₁ g₁ = (matroidOfFinset j h₂ s₂ g₂).comap f := by
  refine Matroid.ext_indep rfl fun I _ => ?_
  obtain ⟨I, rfl⟩ := (Set.toFinite I).exists_finset_coe
  rw [Matroid.comap_indep_iff]
  have himg : f '' (I : Set α) = ((I.image f : Finset _) : Set _) := by simp
  rw [himg]
  constructor
  · intro hI
    exact ⟨(matroidOfFinset_indep _).2 ((h I).1 ((matroidOfFinset_indep I).1 hI)), hf.injOn⟩
  · rintro ⟨hI, -⟩
    exact (matroidOfFinset_indep I).2 ((h I).2 ((matroidOfFinset_indep _).1 hI))

/-- A Δ-Y exchange of finite matroids, from tests on their independent sets: `{a, b, c}` is a
circuit of the first matroid, `iP` restricts to both parts, every set restricting to flats of both
is a flat of `iP`, and `i₂` is `iP` with `a`, `b`, `c` replaced by the new elements. -/
theorem isDeltaMatroid_of_finset [Fintype α] {i₁ i₂ : Finset α → Prop}
    {iP : Finset (α ⊕ Fin 3) → Prop} {h₁ : i₁ ∅} {s₁ : ∀ I J : Finset α, i₁ J → I ⊆ J → i₁ I}
    {g₁ : ∀ I J : Finset α, i₁ I → i₁ J → I.card < J.card → ∃ e ∈ J, e ∉ I ∧ i₁ (insert e I)}
    {h₂ : i₂ ∅} {s₂ : ∀ I J : Finset α, i₂ J → I ⊆ J → i₂ I}
    {g₂ : ∀ I J : Finset α, i₂ I → i₂ J → I.card < J.card → ∃ e ∈ J, e ∉ I ∧ i₂ (insert e I)}
    (hP : iP ∅) (sP : ∀ I J : Finset (α ⊕ Fin 3), iP J → I ⊆ J → iP I)
    (gP : ∀ I J : Finset (α ⊕ Fin 3), iP I → iP J → I.card < J.card →
      ∃ e ∈ J, e ∉ I ∧ iP (insert e I)) {a b c : α} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c)
    (hcirc : ¬i₁ {a, b, c} ∧ ∀ e ∈ ({a, b, c} : Finset α), i₁ (({a, b, c} : Finset α).erase e))
    (hres₁ : ∀ I : Finset α, i₁ I ↔ iP (I.image Sum.inl))
    (hres₂ : ∀ I : Finset (Fin 3 ⊕ Fin 3), k4Indep I ↔ iP (I.image (deltaEmbed a b c)))
    (hflat : ∀ S : Finset (α ⊕ Fin 3), FlatTest i₁ (Finset.univ.filter fun e => Sum.inl e ∈ S) →
      FlatTest k4Indep (Finset.univ.filter fun t => deltaEmbed a b c t ∈ S) → FlatTest iP S)
    (hcomap : ∀ I : Finset α, i₂ I ↔ iP (I.image (deltaRelabel a b c))) :
    IsDeltaMatroid (matroidOfFinset i₁ h₁ s₁ g₁) a b c (matroidOfFinset i₂ h₂ s₂ g₂) := by
  have hM : matroidOfFinset i₁ h₁ s₁ g₁ = (matroidOfFinset iP hP sP gP).comap Sum.inl :=
    matroidOfFinset_eq_comap Sum.inl_injective hres₁
  have hK : k4Matroid = (matroidOfFinset iP hP sP gP).comap (deltaEmbed a b c) :=
    matroidOfFinset_eq_comap (deltaEmbed_injective hab hac hbc) hres₂
  refine ⟨hab, hac, hbc, ?_, matroidOfFinset iP hP sP gP, ?_, fun S => ⟨fun hS => ⟨?_, ?_⟩, ?_⟩,
    matroidOfFinset_eq_comap (deltaRelabel_injective a b c) hcomap⟩
  · have h : ({a, b, c} : Set α) = (({a, b, c} : Finset α) : Set α) := by simp
    rw [h]
    exact (matroidOfFinset_isCircuit_iff _).2 hcirc
  · change Set.univ = Sum.inl '' Set.univ ∪ Set.range Sum.inr
    rw [Set.image_univ, Set.range_inl_union_range_inr]
  · rw [hM]
    exact isFlat_comap_preimage hS _ rfl
  · rw [hK]
    exact isFlat_comap_preimage hS _ rfl
  · rintro ⟨hS₁, hS₂⟩
    obtain ⟨S, rfl⟩ := (Set.toFinite S).exists_finset_coe
    have h1 : Sum.inl ⁻¹' (S : Set (α ⊕ Fin 3)) =
        ((Finset.univ.filter fun e => Sum.inl e ∈ S : Finset α) : Set α) := by
      ext
      simp
    have h2 : deltaEmbed a b c ⁻¹' (S : Set (α ⊕ Fin 3)) =
        ((Finset.univ.filter fun t => deltaEmbed a b c t ∈ S :
          Finset (Fin 3 ⊕ Fin 3)) : Set (Fin 3 ⊕ Fin 3)) := by
      ext
      simp
    rw [h1] at hS₁
    rw [h2] at hS₂
    exact (matroidOfFinset_isFlat_iff S).2 (hflat S ((matroidOfFinset_isFlat_iff _).1 hS₁)
      ((matroidOfFinset_isFlat_iff _).1 hS₂))

end Uniform

end Transformer.GainMatroid
