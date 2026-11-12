module

public import Transformer.Grokking.Protocol.LabelCoverage

/-!
# Splits that are anchored and show every residue, and still leave an answer unlabelled

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*;
openai/grok at 3d64b1d8, `grok/data.py`, where the list of the equations has the quotient in
the outer loop; `experiments/grok_division/PROTOCOL.md`, *The split*;
`review/grok_division_convergence.md`, sections 2 and 4.

`LabelCoverage` shows that a condition on the split under which the target holds admits
labelled splits only. This file compares `labelled` with the conditions that the development
already has, and exhibits the splits.

* A labelled split is anchored (`labelled_subset_anchored`), and an uncovered split leaves
  the residue `1` unlabelled (`uncovered_subset_unlabelled`), so it is not labelled.
* The split `1`, which trains on the first 4656 equations of the source's list, trains on
  quotients from `0` to `48` only (`val_fst_trainEq_one`), so it leaves the residue `96`
  unlabelled (`one_mem_unlabelled`). It is anchored (`one_mem_anchored`), and every residue is
  shown by one of its training equations, as a denominator or as the quotient `0`
  (`occurring`, `one_mem_occurring`). Hence the target is false under the condition that the
  split is anchored and shows every residue (`not_ae_eventually_anchored_occurring`): that
  every residue occurs in training does not suffice.
* The split that lists the equations with the denominator in the outer loop is labelled
  (`denominatorFirst_mem_labelled`), so labelled splits have positive probability
  (`uniformPerm_labelled_pos`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter MeasureTheory

/-- **A split that labels every residue is anchored.** -/
theorem labelled_subset_anchored : labelled ⊆ anchored := fun π hπ => by
  obtain ⟨i₁, h₁⟩ := hπ 1
  obtain ⟨i₀, h₀⟩ := hπ 0
  refine ⟨⟨i₁, ?_⟩, ⟨i₀, ?_⟩⟩
  · rw [quotient, h₁]
    exact one_ne_zero
  · rw [quotient, h₀]
    rfl

/-- **An uncovered split leaves the residue `1` unlabelled.** -/
theorem uncovered_subset_unlabelled : uncovered ⊆ unlabelled 1 := fun π hπ i h =>
  (hπ i).2.2 (by rw [h]; decide)

/-- The quotient of the equation in the slot `i` of the split that trains on the first 4656
equations of the list. -/
lemma val_fst_trainEq_one (i : Fin 4656) : ((trainEq 1 i).1 : ℕ) = i / 96 := rfl

/-- The denominator, less one, of the equation in the slot `i` of that split. -/
lemma val_snd_trainEq_one (i : Fin 4656) : ((trainEq 1 i).2 : ℕ) = i % 96 := rfl

/-- **The split that trains on the first 4656 equations leaves the residue `96` unlabelled**:
its training equations have the quotients `0` to `48`. -/
theorem one_mem_unlabelled : (1 : Equiv.Perm (Fin (97 * 96))) ∈ unlabelled 96 := fun i h => by
  have hv : (i : ℕ) / 96 = 96 := congrArg Fin.val h
  have := i.isLt
  omega

/-- That split is not labelled. -/
theorem one_notMem_labelled : (1 : Equiv.Perm (Fin (97 * 96))) ∉ labelled :=
  not_mem_labelled_iff.2 ⟨96, one_mem_unlabelled⟩

/-- **The splits whose training equations show every residue**: each residue is the
numerator, the denominator or the quotient of a training equation. -/
def occurring : Set (Equiv.Perm (Fin (97 * 96))) :=
  {π | ∀ r : Fin 97, ∃ i, numerator (trainEq π i) = r ∨ denominator (trainEq π i) = r ∨
    (trainEq π i).1 = r}

/-- **The split that trains on the first 4656 equations shows every residue**: `0` is the
quotient of `0 / 1 = 0`, and `r ≠ 0` is the denominator of `0 / r = 0`. -/
theorem one_mem_occurring : (1 : Equiv.Perm (Fin (97 * 96))) ∈ occurring := fun r => by
  have hr := r.isLt
  rcases Nat.eq_zero_or_pos (r : ℕ) with h0 | hpos
  · refine ⟨⟨0, by norm_num⟩, Or.inr (Or.inr (Fin.ext ?_))⟩
    show (0 : ℕ) / 96 = r
    omega
  · refine ⟨⟨r - 1, by omega⟩, Or.inr (Or.inl (Fin.ext ?_))⟩
    show ((r : ℕ) - 1) % 96 + 1 = r
    omega

/-- **Anchors and every residue shown in training do not make the target true**: it is false
that for almost every sample whose split is anchored and shows every residue every held-out
equation is eventually answered, for Adam and for AdamW at every weight decay. -/
theorem not_ae_eventually_anchored_occurring (decay : ℝ) :
    ¬ ∀ᵐ ω ∂law (97 * 96) 4656 (initMeasure paperSizes), ω.1 ∈ anchored ∩ occurring →
      ∀ᶠ t in atTop, ω ∉ wrongAt decay t :=
  not_ae_eventually_of_not_mem_labelled decay ⟨one_mem_anchored, one_mem_occurring⟩
    one_notMem_labelled

/-- The split that lists the equations with the denominator in the outer loop: its slots `0`
to `96` hold the equations `r / 1 = r`. -/
def denominatorFirst : Equiv.Perm (Fin (97 * 96)) :=
  (finCongr (by norm_num : 97 * 96 = 96 * 97)).trans
    ((finProdFinEquiv (m := 96) (n := 97)).symm.trans
      ((Equiv.prodComm (Fin 96) (Fin 97)).trans (finProdFinEquiv (m := 97) (n := 96))))

/-- **Some split labels every residue.** -/
theorem denominatorFirst_mem_labelled : denominatorFirst ∈ labelled := fun r => by
  have hr := r.isLt
  refine ⟨⟨r, by omega⟩, ?_⟩
  simp only [trainEq, trainAt, equations, denominatorFirst, Equiv.trans_apply,
    Equiv.symm_apply_apply, Equiv.prodComm_apply, Prod.fst_swap]
  exact Fin.ext (Nat.mod_eq_of_lt hr)

/-- **A uniformly random split labels every residue with positive probability.** -/
theorem uniformPerm_labelled_pos : 0 < uniformPerm (97 * 96) labelled :=
  uniformPerm_pos_of_mem denominatorFirst_mem_labelled

/-! The witnesses: a labelled split is anchored; an uncovered split exists and leaves `1`
unlabelled; the equations in two slots of the split `1`; and the target is false under the
condition `anchored` alone. -/

example : denominatorFirst ∈ anchored := labelled_subset_anchored denominatorFirst_mem_labelled

example : ∃ π, π ∈ unlabelled 1 := exists_uncovered.imp fun _ h => uncovered_subset_unlabelled h

example : ((trainEq 1 ⟨4655, by norm_num⟩).1 : ℕ) = 48 := val_fst_trainEq_one _

example : ((trainEq 1 ⟨4655, by norm_num⟩).2 : ℕ) = 47 := val_snd_trainEq_one _

example (decay : ℝ) : ¬ ∀ᵐ ω ∂law (97 * 96) 4656 (initMeasure paperSizes), ω.1 ∈ anchored →
    ∀ᶠ t in atTop, ω ∉ wrongAt decay t :=
  not_ae_eventually_of_not_mem_labelled decay one_mem_anchored one_notMem_labelled

end Transformer.Grokking.Protocol
