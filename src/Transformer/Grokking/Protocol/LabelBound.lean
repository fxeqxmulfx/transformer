module

public import Transformer.Grokking.Protocol.LabelWitness
public import Transformer.Grokking.Protocol.SplitCount

/-!
# A split labels every residue with a probability of at least `1 - 97 / 2 ^ 96`

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*: half
of the equations, chosen at random, are the training set;
`experiments/grok_division/PROTOCOL.md`, *The split*; `review/grok_division_convergence.md`,
section 2.

`LabelCoverage` shows that a true form of the target admits labelled splits only, and that to
be labelled is not sure. Here is how unsure. The split trains on half of the equations, so
`SplitCount` applies. A residue `r` is the answer of 96 equations (`withAnswer`,
`le_card_withAnswer`), and a split leaves `r` unlabelled when it keeps them all out of
training (`unlabelled_eq`): that has a probability of at most `2⁻⁹⁶`
(`uniformPerm_unlabelled_le_inv`). There are 97 residues, so a split is not labelled with a
probability of at most `97 / 2 ^ 96` (`uniformPerm_compl_labelled_le`), which is about
`1.2 · 10⁻²⁷`; and it is labelled, hence anchored, with a probability of at least
`1 - 97 / 2 ^ 96` (`le_uniformPerm_labelled`, `le_uniformPerm_anchored`).

These are bounds. The probability that a split is not labelled is about `7.5 · 10⁻²⁸` by a
count that is not formalized here.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open MeasureTheory

/-- The equations with the answer `r`, as places in the list of equations. -/
def withAnswer (r : Fin 97) : Finset (Fin (97 * 96)) :=
  Finset.univ.filter fun j => (equations j).1 = r

/-- A residue is the answer of at least 96 equations: one for each denominator. -/
lemma le_card_withAnswer (r : Fin 97) : 96 ≤ (withAnswer r).card := by
  have h := Finset.card_le_card_of_injOn (s := (Finset.univ : Finset (Fin 96)))
    (t := withAnswer r) (fun d => finProdFinEquiv (r, d))
    (fun d _ => Finset.mem_coe.2 (Finset.mem_filter.2 ⟨Finset.mem_univ _, by simp [equations]⟩))
    (fun a _ b _ hab => (Prod.mk.inj (finProdFinEquiv.injective hab)).2)
  simpa using h

/-- A split leaves `r` unlabelled when it keeps every equation with the answer `r` out of the
4656 training slots. -/
lemma unlabelled_eq (r : Fin 97) :
    unlabelled r = {π | ∀ x ∈ withAnswer r, 4656 ≤ (π.symm x : ℕ)} := by
  ext π
  simp only [unlabelled, Set.mem_ofPred_eq]
  refine ⟨fun hπ x hx => ?_, fun h i hi => ?_⟩
  · by_contra hlt
    rw [not_le] at hlt
    refine hπ ⟨(π.symm x : ℕ), hlt⟩ ?_
    have hcast : Fin.castLE (by norm_num : 4656 ≤ 97 * 96) ⟨(π.symm x : ℕ), hlt⟩ = π.symm x :=
      Fin.ext rfl
    simp only [trainEq, trainAt, hcast, Equiv.apply_symm_apply]
    exact (Finset.mem_filter.1 hx).2
  · have hx : π (Fin.castLE (by norm_num : 4656 ≤ 97 * 96) i) ∈ withAnswer r :=
      Finset.mem_filter.2 ⟨Finset.mem_univ _, hi⟩
    have hk := h _ hx
    rw [Equiv.symm_apply_apply] at hk
    simp only [Fin.val_castLE] at hk
    omega

/-- **A split leaves a given residue unlabelled with a probability of at most `2⁻⁹⁶`.** -/
theorem uniformPerm_unlabelled_le_inv (r : Fin 97) :
    uniformPerm (97 * 96) (unlabelled r) ≤ (2 ^ 96)⁻¹ := by
  rw [unlabelled_eq]
  exact (uniformPerm_keptOut_le 4656 (by norm_num) (withAnswer r)).trans
    (ENNReal.inv_le_inv.2 (pow_le_pow_right₀ one_le_two (le_card_withAnswer r)))

/-- **A split is not labelled with a probability of at most `97 / 2 ^ 96`.** -/
theorem uniformPerm_compl_labelled_le : uniformPerm (97 * 96) labelledᶜ ≤ 97 / 2 ^ 96 := by
  have hcover : labelledᶜ ⊆ ⋃ r : Fin 97, unlabelled r := fun π hπ =>
    Set.mem_iUnion.2 (not_mem_labelled_iff.1 hπ)
  calc uniformPerm (97 * 96) labelledᶜ
      ≤ uniformPerm (97 * 96) (⋃ r : Fin 97, unlabelled r) := measure_mono hcover
    _ ≤ ∑ r : Fin 97, uniformPerm (97 * 96) (unlabelled r) := measure_iUnion_fintype_le _ _
    _ ≤ ∑ _r : Fin 97, (2 ^ 96 : ENNReal)⁻¹ :=
        Finset.sum_le_sum fun r _ => uniformPerm_unlabelled_le_inv r
    _ = 97 / 2 ^ 96 := by simp [div_eq_mul_inv]

/-- **A split is labelled with a probability of at least `1 - 97 / 2 ^ 96`.** -/
theorem le_uniformPerm_labelled : 1 - 97 / 2 ^ 96 ≤ uniformPerm (97 * 96) labelled := by
  rw [tsub_le_iff_right]
  calc (1 : ENNReal) = uniformPerm (97 * 96) labelled + uniformPerm (97 * 96) labelledᶜ :=
        (uniformPerm_add_compl labelled).symm
    _ ≤ uniformPerm (97 * 96) labelled + 97 / 2 ^ 96 :=
        add_le_add le_rfl uniformPerm_compl_labelled_le

/-- **A split is anchored with a probability of at least `1 - 97 / 2 ^ 96`.** -/
theorem le_uniformPerm_anchored : 1 - 97 / 2 ^ 96 ≤ uniformPerm (97 * 96) anchored :=
  le_uniformPerm_labelled.trans (measure_mono labelled_subset_anchored)

end Transformer.Grokking.Protocol
