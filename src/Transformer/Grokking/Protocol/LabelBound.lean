module

public import Transformer.Grokking.Protocol.LabelWitness
public import Transformer.Grokking.Protocol.SplitProbability

/-!
# How probable it is that a split labels every residue

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*: half
of the equations, chosen at random, are the training set;
`experiments/grok_division/PROTOCOL.md`, *The split*; `review/grok_division_convergence.md`,
section 2.

`LabelCoverage` shows that a true form of the target admits labelled splits only, and that to
be labelled is not sure. Here is how unsure. A residue `r` is the answer of 96 equations
(`withAnswer`, `card_withAnswer`), and a split leaves `r` unlabelled when it keeps them all
out of its 4656 training slots (`unlabelled_eq`). By the count of `SplitProbability` that has
the probability `4656 · 4655 ⋯ 4561 / (9312 · 9311 ⋯ 9217)` (`uniformPerm_unlabelled_eq`),
which is between `7 · 10⁻³⁰` and `8 · 10⁻³⁰` (`uniformPerm_unlabelled_mem_Icc`) and at most
`2⁻⁹⁶` (`uniformPerm_unlabelled_le_inv`). There are 97 residues, so a split is not labelled
with a probability of at least `7 · 10⁻³⁰` and at most `97 / 2 ^ 96`
(`le_uniformPerm_compl_labelled`, `uniformPerm_compl_labelled_le`), which is about
`1.2 · 10⁻²⁷`; and it is labelled with a probability of at least `1 - 97 / 2 ^ 96`
(`le_uniformPerm_labelled`).

The probability that a split is not labelled is about `7.5 · 10⁻²⁸` by a count that is not
formalized here: the bounds above enclose it and do not give it.

For the anchored splits of `OrbitAnchors` the probability is exact. Only 96 equations have the
quotient zero, so a split always trains on an equation with a nonzero quotient, and it is not
anchored exactly when it leaves the residue `0` unlabelled (`compl_anchored_eq`). So a split
is anchored with the probability `1 - 4656 ⋯ 4561 / (9312 ⋯ 9217)`
(`uniformPerm_anchored_eq`), between `1 - 8 · 10⁻³⁰` and `1 - 7 · 10⁻³⁰`
(`uniformPerm_anchored_mem_Icc`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open MeasureTheory

/-- The equations with the answer `r`, as places in the list of equations. -/
def withAnswer (r : Fin 97) : Finset (Fin (97 * 96)) :=
  Finset.univ.filter fun j => (equations j).1 = r

/-- A residue is the answer of exactly 96 equations: one for each denominator. -/
lemma card_withAnswer (r : Fin 97) : (withAnswer r).card = 96 := by
  refine le_antisymm (card_quotient_eq_le r) ?_
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

/-- **The probability that a split leaves a given residue unlabelled**:
`4656 · 4655 ⋯ 4561 / (9312 · 9311 ⋯ 9217)`. -/
theorem uniformPerm_unlabelled_eq (r : Fin 97) :
    uniformPerm (97 * 96) (unlabelled r) =
      (Nat.descFactorial 4656 96 : ENNReal) / (Nat.descFactorial 9312 96 : ENNReal) := by
  rw [unlabelled_eq, uniformPerm_keptOut_eq, card_withAnswer]

/-- That ratio is between `7 · 10⁻³⁰` and `8 · 10⁻³⁰`. -/
lemma descFactorial_ratio_bounds :
    7 * Nat.descFactorial 9312 96 ≤ Nat.descFactorial 4656 96 * 10 ^ 30 ∧
      Nat.descFactorial 4656 96 * 10 ^ 30 ≤ 8 * Nat.descFactorial 9312 96 := by
  decide

/-- **A split leaves a given residue unlabelled with a probability between `7 · 10⁻³⁰` and
`8 · 10⁻³⁰`.** -/
theorem uniformPerm_unlabelled_mem_Icc (r : Fin 97) :
    uniformPerm (97 * 96) (unlabelled r) ∈ Set.Icc (7 / 10 ^ 30 : ENNReal) (8 / 10 ^ 30) := by
  have hd : Nat.descFactorial 9312 96 ≠ 0 := (Nat.descFactorial_pos.2 (by norm_num)).ne'
  rw [uniformPerm_unlabelled_eq]
  constructor
  · exact_mod_cast natCast_div_le_div (a := 7) (b := 10 ^ 30) (by positivity) hd
      descFactorial_ratio_bounds.1
  · exact_mod_cast natCast_div_le_div (c := 8) (d := 10 ^ 30) hd (by positivity)
      descFactorial_ratio_bounds.2

/-- **A split leaves a given residue unlabelled with a probability of at most `2⁻⁹⁶`.** -/
theorem uniformPerm_unlabelled_le_inv (r : Fin 97) :
    uniformPerm (97 * 96) (unlabelled r) ≤ (2 ^ 96)⁻¹ := by
  rw [unlabelled_eq]
  exact (uniformPerm_keptOut_le 4656 (by norm_num) (withAnswer r)).trans
    (ENNReal.inv_le_inv.2 (pow_le_pow_right₀ one_le_two (card_withAnswer r).ge))

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

/-- **A split is not labelled with a probability of at least `7 · 10⁻³⁰`.** -/
theorem le_uniformPerm_compl_labelled : 7 / 10 ^ 30 ≤ uniformPerm (97 * 96) labelledᶜ :=
  (uniformPerm_unlabelled_mem_Icc 0).1.trans
    (measure_mono fun _ hπ => not_mem_labelled_iff.2 ⟨0, hπ⟩)

/-- **A split is labelled with a probability of at least `1 - 97 / 2 ^ 96`.** -/
theorem le_uniformPerm_labelled : 1 - 97 / 2 ^ 96 ≤ uniformPerm (97 * 96) labelled := by
  rw [tsub_le_iff_right]
  calc (1 : ENNReal) = uniformPerm (97 * 96) labelled + uniformPerm (97 * 96) labelledᶜ :=
        (uniformPerm_add_compl labelled).symm
    _ ≤ uniformPerm (97 * 96) labelled + 97 / 2 ^ 96 :=
        add_le_add le_rfl uniformPerm_compl_labelled_le

/-- **A split is not anchored exactly when it leaves the residue `0` unlabelled**: only 96
equations have the quotient zero, so the 4656 training equations are not all among them, and
what a split that is not anchored lacks is a training equation with the quotient zero. -/
theorem compl_anchored_eq : anchoredᶜ = unlabelled 0 := by
  ext π
  have hne : ∃ i : Fin 4656, quotient (trainEq π i) ≠ 0 := by
    by_contra hall
    have hzero : ∀ i : Fin 4656, quotient (trainEq π i) = 0 := fun i =>
      not_not.1 (not_exists.1 hall i)
    have h := Finset.card_le_card_of_injOn (s := (Finset.univ : Finset (Fin 4656)))
      (t := Finset.univ.filter fun j : Fin (97 * 96) => quotient (equations j) = 0)
      (fun i => π (Fin.castLE (by norm_num : 4656 ≤ 97 * 96) i))
      (fun i _ => Finset.mem_coe.2 (Finset.mem_filter.2 ⟨Finset.mem_univ _, hzero i⟩))
      (fun a _ b _ hab => Fin.castLE_injective _ (π.injective hab))
    have h96 := card_quotient_zero_le
    rw [Finset.card_univ, Fintype.card_fin] at h
    omega
  simp only [Set.mem_compl_iff, anchored, unlabelled, Set.mem_ofPred_eq, not_and, not_exists]
  exact ⟨fun h i => h hne i, fun h _ i => h i⟩

/-- **The probability that a split is not anchored**:
`4656 · 4655 ⋯ 4561 / (9312 · 9311 ⋯ 9217)`. -/
theorem uniformPerm_compl_anchored_eq :
    uniformPerm (97 * 96) anchoredᶜ =
      (Nat.descFactorial 4656 96 : ENNReal) / (Nat.descFactorial 9312 96 : ENNReal) := by
  rw [compl_anchored_eq, uniformPerm_unlabelled_eq]

/-- **The probability that a split is anchored.** -/
theorem uniformPerm_anchored_eq :
    uniformPerm (97 * 96) anchored =
      1 - (Nat.descFactorial 4656 96 : ENNReal) / (Nat.descFactorial 9312 96 : ENNReal) := by
  rw [← uniformPerm_compl_anchored_eq]
  exact ENNReal.eq_sub_of_add_eq (measure_ne_top _ _) (uniformPerm_add_compl anchored)

/-- **A split is anchored with a probability between `1 - 8 · 10⁻³⁰` and `1 - 7 · 10⁻³⁰`.** -/
theorem uniformPerm_anchored_mem_Icc :
    uniformPerm (97 * 96) anchored ∈ Set.Icc (1 - 8 / 10 ^ 30 : ENNReal) (1 - 7 / 10 ^ 30) := by
  have h := uniformPerm_unlabelled_mem_Icc 0
  rw [uniformPerm_unlabelled_eq] at h
  rw [uniformPerm_anchored_eq]
  exact ⟨tsub_le_tsub_left h.2 1, tsub_le_tsub_left h.1 1⟩

end Transformer.Grokking.Protocol
