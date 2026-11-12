module

public import Transformer.Grokking.Protocol.LabelBound
public import Transformer.Grokking.Protocol.Bonferroni

/-!
# The probability that a split labels every residue, to three digits

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*: half
of the equations, chosen at random, are the training set;
`experiments/grok_division/PROTOCOL.md`, *The split*; `review/grok_division_convergence.md`,
section 2. The paper states no probability.

A split is not labelled when it leaves one of the 97 residues unlabelled
(`biUnion_unlabelled`), and `LabelBound` gives the probability of that for one residue, between
`7.695 · 10⁻³⁰` and `7.696 · 10⁻³⁰`. The 97 events overlap, but hardly: two given residues are
both unlabelled when 192 equations are kept out of training, which has a probability of at
most `2⁻¹⁹²` (`uniformPerm_unlabelled_inter_le`). The union bound from above and Bonferroni's
inequality from below (`Bonferroni`) enclose the probability that a split is not labelled
between `7.46 · 10⁻²⁸` and `7.47 · 10⁻²⁸` (`uniformPerm_compl_labelled_mem_Icc`); so a split is
labelled with a probability between `1 - 7.47 · 10⁻²⁸` and `1 - 7.46 · 10⁻²⁸`
(`uniformPerm_labelled_mem_Icc`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open MeasureTheory

/-- **A split leaves two given residues unlabelled with a probability of at most `2⁻¹⁹²`**:
it keeps the 192 equations with these answers out of training. -/
theorem uniformPerm_unlabelled_inter_le {r s : Fin 97} (h : r ≠ s) :
    uniformPerm (97 * 96) (unlabelled r ∩ unlabelled s) ≤ (2 ^ 192)⁻¹ := by
  have hset : unlabelled r ∩ unlabelled s =
      {π | ∀ x ∈ withAnswer r ∪ withAnswer s, 4656 ≤ (π.symm x : ℕ)} := by
    rw [unlabelled_eq, unlabelled_eq]
    ext π
    simp only [Set.mem_inter_iff, Set.mem_ofPred_eq, Finset.mem_union]
    exact ⟨fun hπ x hx => hx.elim (hπ.1 x) (hπ.2 x),
      fun hπ => ⟨fun x hx => hπ x (Or.inl hx), fun x hx => hπ x (Or.inr hx)⟩⟩
  have hcard : (withAnswer r ∪ withAnswer s).card = 192 := by
    rw [Finset.card_union_of_disjoint, card_withAnswer, card_withAnswer]
    exact Finset.disjoint_filter.2 fun j _ hr hs => h (hr.symm.trans hs)
  rw [hset]
  exact (uniformPerm_keptOut_le 4656 (by norm_num) _).trans (by rw [hcard])

/-- The arithmetic of the lower bound: 97 events of probability at least `7.695 · 10⁻³⁰`
each, of which two occur together with a probability of at most `2⁻¹⁹²`. -/
lemma le_of_two_mul_le_add {P : ENNReal}
    (h : (2 : ENNReal) * (97 * (7695 / 10 ^ 33)) ≤ 2 * P + 97 * (96 * (2 ^ 192)⁻¹)) :
    746 / 10 ^ 30 ≤ P := by
  rcases eq_or_ne P ⊤ with rfl | hP
  · exact le_top
  have h3 := ENNReal.toReal_mono (by finiteness) h
  rw [ENNReal.toReal_add (by finiteness) (by finiteness)] at h3
  simp only [ENNReal.toReal_mul, ENNReal.toReal_ofNat, ENNReal.toReal_div, ENNReal.toReal_pow,
    ENNReal.toReal_inv] at h3
  rw [← ENNReal.ofReal_toReal hP,
    ENNReal.le_ofReal_iff_toReal_le (by finiteness) ENNReal.toReal_nonneg]
  simp only [ENNReal.toReal_div, ENNReal.toReal_ofNat, ENNReal.toReal_pow]
  norm_num at h3 ⊢
  linarith

/-- The splits that are not labelled are those that leave some residue unlabelled. -/
lemma biUnion_unlabelled :
    (⋃ r ∈ (Finset.univ : Finset (Fin 97)), unlabelled r) = labelledᶜ := by
  ext π
  simp only [Set.mem_iUnion, Finset.mem_univ, exists_const, Set.mem_compl_iff,
    not_mem_labelled_iff]

/-- **A split is not labelled with a probability between `7.46 · 10⁻²⁸` and
`7.47 · 10⁻²⁸`.** -/
theorem uniformPerm_compl_labelled_mem_Icc :
    uniformPerm (97 * 96) labelledᶜ ∈ Set.Icc (746 / 10 ^ 30 : ENNReal) (747 / 10 ^ 30) := by
  constructor
  · have h := two_mul_sum_uniformPerm_le (N := 97 * 96) unlabelled Finset.univ
    rw [biUnion_unlabelled] at h
    have hl : (2 : ENNReal) * (97 * (7695 / 10 ^ 33)) ≤
        2 * ∑ r : Fin 97, uniformPerm (97 * 96) (unlabelled r) := by
      gcongr
      calc (97 : ENNReal) * (7695 / 10 ^ 33) = ∑ _r : Fin 97, (7695 / 10 ^ 33 : ENNReal) := by
            simp
        _ ≤ _ := Finset.sum_le_sum fun r _ => (uniformPerm_unlabelled_mem_Icc r).1
    have hr : ∑ r : Fin 97, ∑ s ∈ Finset.univ.erase r,
        uniformPerm (97 * 96) (unlabelled r ∩ unlabelled s) ≤ 97 * (96 * (2 ^ 192)⁻¹) := by
      calc _ ≤ ∑ r : Fin 97, ∑ _s ∈ Finset.univ.erase r, ((2 : ENNReal) ^ 192)⁻¹ :=
            Finset.sum_le_sum fun r _ => Finset.sum_le_sum fun s hs =>
              uniformPerm_unlabelled_inter_le (Finset.ne_of_mem_erase hs).symm
        _ = 97 * (96 * (2 ^ 192)⁻¹) := by simp
    exact le_of_two_mul_le_add (hl.trans (h.trans (add_le_add le_rfl hr)))
  · calc uniformPerm (97 * 96) labelledᶜ
        ≤ ∑ r : Fin 97, uniformPerm (97 * 96) (unlabelled r) := by
          rw [← biUnion_unlabelled]
          exact measure_biUnion_finset_le _ _
      _ ≤ ∑ _r : Fin 97, (7696 / 10 ^ 33 : ENNReal) :=
          Finset.sum_le_sum fun r _ => (uniformPerm_unlabelled_mem_Icc r).2
      _ ≤ 747 / 10 ^ 30 := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, ← mul_div_assoc]
          exact_mod_cast natCast_div_le_div (a := 97 * 7696) (b := 10 ^ 33) (c := 747)
            (d := 10 ^ 30) (by positivity) (by positivity) (by norm_num)

/-- **A split is labelled with a probability between `1 - 7.47 · 10⁻²⁸` and
`1 - 7.46 · 10⁻²⁸`.** -/
theorem uniformPerm_labelled_mem_Icc :
    uniformPerm (97 * 96) labelled ∈
      Set.Icc (1 - 747 / 10 ^ 30 : ENNReal) (1 - 746 / 10 ^ 30) := by
  have h := uniformPerm_compl_labelled_mem_Icc
  have heq : uniformPerm (97 * 96) labelled = 1 - uniformPerm (97 * 96) labelledᶜ :=
    ENNReal.eq_sub_of_add_eq (measure_ne_top _ _) (uniformPerm_add_compl labelled)
  rw [heq]
  exact ⟨tsub_le_tsub_left h.2 1, tsub_le_tsub_left h.1 1⟩

/-! The witnesses: the residues `0` and `1` are two; and the arithmetic at the probability
`1`, where its hypothesis holds because `97 · 7.695 · 10⁻³⁰ ≤ 1`. -/

example : uniformPerm (97 * 96) (unlabelled 0 ∩ unlabelled 1) ≤ (2 ^ 192)⁻¹ :=
  uniformPerm_unlabelled_inter_le (by decide)

example : (746 / 10 ^ 30 : ENNReal) ≤ 1 := by
  refine le_of_two_mul_le_add (le_add_right ?_)
  gcongr
  rw [← mul_div_assoc]
  exact ENNReal.div_le_of_le_mul (by norm_num)

end Transformer.Grokking.Protocol
