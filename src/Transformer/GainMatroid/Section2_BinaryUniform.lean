module

public import Transformer.GainMatroid.Section2_Representable

/-!
# `U_{2,4}` is not binary, and `U_{3,4}` is

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. Proposition 2.9 as printed fails on the Δ-Y
exchange of `U_{2,4}` on a triangle, which is `U_{3,4}` (`proposition2_9_fails`, in
`Section2_RepresentationExchange`). Over `𝔽₂` the first has no representation, since a line over
`𝔽₂` has only three nonzero vectors (`uniformTwoFour_not_binary`), and the second has one: the
three unit vectors of `𝔽₂³` and their sum (`uniformThreeFour_binary`).
-/

@[expose] public section

namespace Transformer.GainMatroid

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

end Transformer.GainMatroid
