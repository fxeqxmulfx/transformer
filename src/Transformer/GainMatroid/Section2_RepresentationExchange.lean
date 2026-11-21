module

public import Transformer.GainMatroid.Section2_Representable

/-!
# Proposition 2.9: representations under a Δ-Y exchange

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, Proposition 2.9, credited to Whittle: "Let
`M'` be a matroid obtained from the matroid `M` by a single Δ-Y exchange, and let `𝔽` be a field.
(i) `M` is `𝔽`-representable if and only if `M'` is `𝔽`-representable. (ii) The projective
equivalence classes of `𝔽`-representations of `M` are in one-to-one correspondence with the
projective equivalence classes of `𝔽`-representations of `M'`."

Both parts fail as printed (`proposition2_9_fails`). In `U_{2,4}` take the triangle `X = {0, 1, 2}`.
Its exchange is `U_{3,4}`, which is binary, although `U_{2,4}` is not. Here `X` is not
coindependent: the element `3` alone does not span the line. Whittle's paper is not among the
sources of this repository, so only the printed statement is refuted here.

The corrected statements add that `X` is coindependent, and for a Y-Δ exchange that `Y` is
independent. The source's use in the proof of Lemma 4.6, on a star of `K₄`, satisfies this.
-/

@[expose] public section

namespace Transformer.GainMatroid

/-- The independent sets of the generalized parallel connection of `U_{2,n}` and `M(K₄)` across
`{0, 1, 2}`: at most three elements, at most two on the line of `U_{2,n}`, and none of the three
lines through two of the new elements. -/
def lineConnectionIndep (n : ℕ) [NeZero n] (I : Finset (Fin n ⊕ Fin 3)) : Prop :=
  I.card ≤ 3 ∧ (I.filter fun t => t.isLeft = true).card ≤ 2 ∧
    ¬(.inl 0 ∈ I ∧ .inr 1 ∈ I ∧ .inr 2 ∈ I) ∧ ¬(.inr 0 ∈ I ∧ .inl 1 ∈ I ∧ .inr 2 ∈ I) ∧
    ¬(.inr 0 ∈ I ∧ .inr 1 ∈ I ∧ .inl 2 ∈ I)

instance (n : ℕ) [NeZero n] : DecidablePred (lineConnectionIndep n) := fun I => by
  unfold lineConnectionIndep
  infer_instance

set_option maxRecDepth 100000 in
set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
/-- **`U_{3,4}` is the Δ-Y exchange of `U_{2,4}` on the triangle `{0, 1, 2}`.** -/
theorem u24_isDeltaMatroid :
    IsDeltaMatroid (finsetUniform (Fin 4) 2) 0 1 2 (finsetUniform (Fin 4) 3) :=
  isDeltaMatroid_of_finset (i₁ := fun I => I.card ≤ 2) (i₂ := fun I => I.card ≤ 3)
    (iP := lineConnectionIndep 4) (by decide) (by decide +kernel) (by decide +kernel) (by decide)
    (by decide) (by decide) (by decide +kernel) (by decide +kernel) (by decide +kernel)
    (by decide +kernel) (by decide +kernel)

set_option maxRecDepth 100000 in
set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
private theorem line5_subset :
    ∀ J, lineConnectionIndep 5 J → ∀ I, I ⊆ J → lineConnectionIndep 5 I := by
  decide +kernel

set_option maxRecDepth 100000 in
set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
private theorem line5_aug : ∀ I, lineConnectionIndep 5 I → ∀ J, lineConnectionIndep 5 J →
    I.card < J.card → ∃ e ∈ J, e ∉ I ∧ lineConnectionIndep 5 (insert e I) := by
  decide +kernel

set_option maxRecDepth 100000 in
set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
private theorem line5_flat : ∀ S : Finset (Fin 5 ⊕ Fin 3),
    FlatTest (fun I : Finset (Fin 5) => I.card ≤ 2) (Finset.univ.filter fun e => Sum.inl e ∈ S) →
      FlatTest k4Indep (Finset.univ.filter fun t => deltaEmbed (0 : Fin 5) 1 2 t ∈ S) →
        FlatTest (lineConnectionIndep 5) S := by
  decide +kernel

set_option maxRecDepth 100000 in
set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
/-- **`U_{3,5}` is the Δ-Y exchange of `U_{2,5}` on the triangle `{0, 1, 2}`.** -/
theorem u25_isDeltaMatroid :
    IsDeltaMatroid (finsetUniform (Fin 5) 2) 0 1 2 (finsetUniform (Fin 5) 3) :=
  isDeltaMatroid_of_finset (i₁ := fun I => I.card ≤ 2) (i₂ := fun I => I.card ≤ 3)
    (iP := lineConnectionIndep 5) (by decide) (fun I J hJ h => line5_subset J hJ I h)
    (fun I J hI hJ => line5_aug I hI J hJ) (by decide) (by decide) (by decide) (by decide +kernel)
    (by decide +kernel) (by decide +kernel) line5_flat (by decide +kernel)

/-- `U_{2,4}` is not binary: a line over `𝔽₂` has only three nonzero vectors. -/
theorem uniformTwoFour_not_binary : ¬IsRepresentable (ZMod 2) (finsetUniform (Fin 4) 2) := by
  rintro ⟨m, A, -, hA⟩
  set v : Fin 4 → Fin m → ZMod 2 := fun e r => A r e
  have hind (I : Finset (Fin 4)) (hI : I.card ≤ 2) : LinearIndepOn (ZMod 2) v (I : Set (Fin 4)) :=
    (hA _).1 ((finsetUniform_indep 2 I).2 hI)
  have hdep (I : Finset (Fin 4)) (hI : ¬I.card ≤ 2) : ¬LinearIndepOn (ZMod 2) v (I : Set (Fin 4)) :=
    fun h => hI ((finsetUniform_indep 2 I).1 ((hA _).2 h))
  have hpair (i j : Fin 4) (hij : i ≠ j) : v i ≠ 0 ∧ v i ≠ v j := by
    have h := hind {i, j} Finset.card_le_two
    exact ⟨h.ne_zero (by simp), fun heq => hij (h.injOn (by simp) (by simp) heq)⟩
  have hbit : ∀ x : ZMod 2, x = 0 ∨ x = 1 := by decide
  have key (k : Fin 4) (hk0 : k ≠ 0) (hk1 : k ≠ 1) : v k = v 0 + v 1 := by
    have hcard : ¬({k, 0, 1} : Finset (Fin 4)).card ≤ 2 := by
      revert k
      decide
    have hset : (({k, 0, 1} : Finset (Fin 4)) : Set (Fin 4)) = insert k {0, 1} := by simp
    have hmem : k ∉ ({0, 1} : Set (Fin 4)) := by simp [hk0, hk1]
    have h := hdep _ hcard
    rw [hset, linearIndepOn_insert hmem] at h
    have h01 : LinearIndepOn (ZMod 2) v ({0, 1} : Set (Fin 4)) := by
      simpa using hind {0, 1} (by decide)
    have hspan : v k ∈ Submodule.span (ZMod 2) (v '' {0, 1}) := by
      by_contra hn
      exact h ⟨h01, hn⟩
    rw [Set.image_pair, Submodule.mem_span_pair] at hspan
    obtain ⟨a, b, hab⟩ := hspan
    rcases hbit a with rfl | rfl <;> rcases hbit b with rfl | rfl
    · exact absurd (by simpa using hab.symm) (hpair k 0 hk0).1
    · exact absurd (by simpa using hab.symm) (hpair k 1 hk1).2
    · exact absurd (by simpa using hab.symm) (hpair k 0 hk0).2
    · simpa using hab.symm
  exact (hpair 2 3 (by decide)).2
    ((key 2 (by decide) (by decide)).trans (key 3 (by decide) (by decide)).symm)

/-- The three unit vectors of `𝔽₂³` and their sum. -/
def uniformThreeFourMatrix : Matrix (Fin 3) (Fin 4) (ZMod 2) :=
  !![1, 0, 0, 1; 0, 1, 0, 1; 0, 0, 1, 1]

set_option maxRecDepth 100000 in
set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
private theorem uniformThreeFourMatrix_test : ∀ I : Finset (Fin 4),
    I.card ≤ 3 ↔ ∀ f : Fin 4 → ZMod 2,
      ∑ i ∈ I, f i • (fun r => uniformThreeFourMatrix r i) = 0 → ∀ i ∈ I, f i = 0 := by
  decide +kernel

/-- `U_{3,4}` is binary. -/
theorem uniformThreeFour_binary : IsRepresentable (ZMod 2) (finsetUniform (Fin 4) 3) := by
  refine ⟨3, uniformThreeFourMatrix, rfl, fun I => ?_⟩
  obtain ⟨I, rfl⟩ := (Set.toFinite I).exists_finset_coe
  rw [finsetUniform_indep, linearIndepOn_finset_iff]
  exact uniformThreeFourMatrix_test I

/-- **Proposition 2.9 fails as printed.** The Δ-Y exchange of `U_{2,4}` on a triangle is `U_{3,4}`.
Over `𝔽₂` the second has a representation and the first has none, so neither "(i) `M` is
`𝔽`-representable if and only if `M'` is" nor the bijection of (ii) holds. -/
theorem proposition2_9_fails :
    ∃ M M' : Matroid (Fin 4), IsDeltaMatroid M 0 1 2 M' ∧
      ¬(IsRepresentable (ZMod 2) M ↔ IsRepresentable (ZMod 2) M') ∧
        IsEmpty (ProjClass (ZMod 2) M ≃ ProjClass (ZMod 2) M') := by
  refine ⟨finsetUniform (Fin 4) 2, finsetUniform (Fin 4) 3, u24_isDeltaMatroid,
    fun h => uniformTwoFour_not_binary (h.2 uniformThreeFour_binary), ⟨fun e => ?_⟩⟩
  obtain ⟨c⟩ := (nonempty_projClass_iff _).2 uniformThreeFour_binary
  exact uniformTwoFour_not_binary ((nonempty_projClass_iff _).1 ⟨e.symm c⟩)

/-- The triangle of `proposition2_9_fails` is not coindependent, which is the hypothesis that the
corrected statement adds. -/
theorem u24_triangle_not_coindependent : ¬(finsetUniform (Fin 4) 2)✶.Indep {0, 1, 2} := by
  have h : ({0, 1, 2} : Set (Fin 4)) = (({0, 1, 2} : Finset (Fin 4)) : Set (Fin 4)) := by simp
  rw [h]
  intro hc
  obtain ⟨B, hB, hdis⟩ := (matroidOfFinset_dual_indep_iff _).1 hc
  revert B
  decide +kernel

/-- The triangle `{0, 1, 2}` of `U_{2,5}` is coindependent: `{3, 4}` is a base. -/
theorem u25_triangle_coindependent : (finsetUniform (Fin 5) 2)✶.Indep {0, 1, 2} := by
  have h : ({0, 1, 2} : Set (Fin 5)) = (({0, 1, 2} : Finset (Fin 5)) : Set (Fin 5)) := by simp
  rw [h]
  exact (matroidOfFinset_dual_indep_iff _).2 ⟨{3, 4}, by decide +kernel, by decide⟩

/-- The dual of `u25_isDeltaMatroid` is a Y-Δ exchange on an independent set. -/
theorem u25_dual_isNablaMatroid :
    IsNablaMatroid (finsetUniform (Fin 5) 2)✶ 0 1 2 (finsetUniform (Fin 5) 3)✶ :=
  ⟨finsetUniform (Fin 5) 3, by rw [Matroid.dual_dual]; exact u25_isDeltaMatroid, rfl⟩

section Corrected

variable {E : Type*} [DecidableEq E] [Finite E] {M M' : Matroid E} {a b c : E} (F : Type*) [Field F]

/-- **Proposition 2.9**, Δ-Y exchange, with the added hypothesis that the triangle `{a, b, c}` is
coindependent, without which it fails (`proposition2_9_fails`). Part (i) is the first conjunct,
part (ii) the second. -/
theorem IsDeltaMatroid.representable_and_projClass (h : IsDeltaMatroid M a b c M')
    (hco : M✶.Indep {a, b, c}) :
    (IsRepresentable F M ↔ IsRepresentable F M') ∧ Nonempty (ProjClass F M ≃ ProjClass F M') := by
  sorry

/-- **Proposition 2.9**, Y-Δ exchange, with the added hypothesis that the triad `{a, b, c}` is
independent, dual to the hypothesis of `IsDeltaMatroid.representable_and_projClass`. -/
theorem IsNablaMatroid.representable_and_projClass (h : IsNablaMatroid M a b c M')
    (hi : M.Indep {a, b, c}) :
    (IsRepresentable F M ↔ IsRepresentable F M') ∧ Nonempty (ProjClass F M ≃ ProjClass F M') := by
  sorry

end Corrected

example := u25_isDeltaMatroid.representable_and_projClass ℚ u25_triangle_coindependent

example := u25_dual_isNablaMatroid.representable_and_projClass ℚ u25_triangle_coindependent

end Transformer.GainMatroid
