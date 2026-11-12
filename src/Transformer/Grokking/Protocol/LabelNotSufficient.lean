module

public import Transformer.Grokking.Protocol.MeanAccuracy
public import Transformer.Grokking.Protocol.TwinWitness

/-!
# The label condition does not suffice

Source: arXiv:2201.02177v1 (Power et al.), Figure 1 and section 1, "distinct abstract
symbols"; `experiments/grok_division/PROTOCOL.md`, *The split*;
`review/grok_division_convergence.md`, section 2. The general argument is that of
`Transformer.Grokking.Protocol.TwistedSplits`.

`LabelCoverage` shows that a true form of the target admits labelled splits only. The converse
fails. A twin split (`TwinSplit`) holds out `2 / 3 = 33` and `95 / 3 = 64`
(`mem_heldout_of_mem_twin`); the exchange of the tokens of `2` and `95` turns the first into
the second up to the position of `=` and does not move their answers. So the labelled twin
splits are a set with a symmetry that moves a held-out answer (`twisted_twin`), and they have
positive probability (`uniformPerm_twin_labelled_pos`). Hence, under Adam and under AdamW at
every weight decay:

* given that the split is a labelled twin split, a held-out answer is wrong with a conditional
  probability of at least one half at every update (`uniformPerm_twin_labelled_le`);
* under every condition on the split that the labelled twin splits meet the target is false
  almost surely, in probability and in mean (`not_ae_eventually_of_twin_subset`,
  `not_tendsto_law_of_twin_subset`, `not_tendsto_setIntegral_of_twin_subset`);
* in particular the target is false on the labelled splits (`not_ae_eventually_labelled`).

To be labelled is necessary and it is not sufficient. The twin splits are one family of
labelled splits with such a symmetry; no description of the splits without one is given here.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology MeasureTheory

variable {π : Equiv.Perm (Fin (97 * 96))}

/-- **A twin split holds out every other equation that shows a twin residue.** -/
lemma mem_heldout_of_mem_twin (hπ : π ∈ twin) {x : Equation} (h1 : ¬ AvoidsTwins x)
    (h2 : x ≠ twinTwo) (h3 : x ≠ twinMinus) : x ∈ heldout equations 4656 π := by
  rcases mem_heldout_or_exists_trainEq π x with h | ⟨i, hi⟩
  · exact h
  · rcases hπ i with h | h | h
    · exact absurd (hi ▸ h) h1
    · exact absurd (hi.symm.trans h) h2
    · exact absurd (hi.symm.trans h) h3

/-- **The labelled twin splits are a set with a symmetry that moves a held-out answer**: the
exchange turns `2 / 3 = 33` into `95 / 3 = 64` up to the position of `=`. -/
theorem twisted_twin : Twisted twinSwap twinOrder (twin ∩ labelled) (33, 2) (64, 2) where
  closed _ hπ := ⟨twinOrder_mul_mem_twin hπ.1, twinOrder_mul_mem_labelled hπ.1 hπ.2⟩
  carries _ hπ := carries_twin hπ.1
  heldout_left _ hπ := mem_heldout_of_mem_twin hπ.1 (by decide) (by decide) (by decide)
  heldout_right _ hπ := mem_heldout_of_mem_twin hπ.1 (by decide) (by decide) (by decide)
  read_eq := by decide
  fixed_left := by decide
  fixed_right := by decide
  ne := by decide

/-- **Given that the split is a labelled twin split, a held-out answer is wrong with a
conditional probability of at least one half, at every update.** -/
theorem uniformPerm_twin_labelled_le (decay : ℝ) (t : ℕ) :
    uniformPerm (97 * 96) (twin ∩ labelled) ≤ 2 * law (97 * 96) 4656 (initMeasure paperSizes)
      (splitIn (twin ∩ labelled) ∩ wrongAt decay t) :=
  uniformPerm_le_of_twisted decay t twisted_twin

variable {S : Set (Equiv.Perm (Fin (97 * 96)))}

/-- **Under a condition on the split that the labelled twin splits meet the target is
false.** -/
theorem not_ae_eventually_of_twin_subset (decay : ℝ) (hS : twin ∩ labelled ⊆ S) :
    ¬ ∀ᵐ ω ∂law (97 * 96) 4656 (initMeasure paperSizes), ω.1 ∈ S →
      ∀ᶠ t in atTop, ω ∉ wrongAt decay t := by
  obtain ⟨π, hπ⟩ := exists_twin_labelled
  exact not_ae_eventually_of_twisted decay twisted_twin hS hπ

/-- **Under such a condition the target fails in probability.** -/
theorem not_tendsto_law_of_twin_subset (decay : ℝ) (hS : twin ∩ labelled ⊆ S) :
    ¬ Tendsto (fun t => law (97 * 96) 4656 (initMeasure paperSizes)
      (splitIn S ∩ wrongAt decay t)) atTop (𝓝 0) := by
  obtain ⟨π, hπ⟩ := exists_twin_labelled
  exact not_tendsto_law_of_twisted decay twisted_twin hS hπ

/-- **Under such a condition the target fails in mean**: the integral of the held-out accuracy
over the runs that meet the condition does not tend to the probability of the condition. -/
theorem not_tendsto_setIntegral_of_twin_subset (decay : ℝ) (hS : twin ∩ labelled ⊆ S) :
    ¬ Tendsto (fun t => ∫ ω in splitIn S, divisionAccuracy decay t ω
      ∂law (97 * 96) 4656 (initMeasure paperSizes)) atTop
        (𝓝 ((uniformPerm (97 * 96)).real S)) := by
  obtain ⟨π, hπ⟩ := exists_twin_labelled
  exact not_tendsto_setIntegral_of_half decay hS hπ fun t => uniformPerm_twin_labelled_le decay t

/-- **The target is false on the labelled splits**: it is not so that for almost every run
with a labelled split every held-out equation is eventually answered, under Adam or under
AdamW at any weight decay. -/
theorem not_ae_eventually_labelled (decay : ℝ) :
    ¬ ∀ᵐ ω ∂law (97 * 96) 4656 (initMeasure paperSizes), ω.1 ∈ labelled →
      ∀ᶠ t in atTop, ω ∉ wrongAt decay t :=
  not_ae_eventually_of_twin_subset decay Set.inter_subset_right

/-! The witnesses: `2 / 3 = 33` is held out by some labelled twin split; and the condition
`labelled`, and the condition `anchored`, which every labelled split meets. -/

example : ∃ π, ((33 : Fin 97), (2 : Fin 96)) ∈ heldout equations 4656 π := by
  obtain ⟨π, hπ⟩ := exists_twin_labelled
  exact ⟨π, mem_heldout_of_mem_twin hπ.1 (by decide) (by decide) (by decide)⟩

example (decay : ℝ) : ¬ Tendsto (fun t => law (97 * 96) 4656 (initMeasure paperSizes)
    (splitIn labelled ∩ wrongAt decay t)) atTop (𝓝 0) :=
  not_tendsto_law_of_twin_subset decay Set.inter_subset_right

example (decay : ℝ) : ¬ Tendsto (fun t => ∫ ω in splitIn labelled, divisionAccuracy decay t ω
    ∂law (97 * 96) 4656 (initMeasure paperSizes)) atTop
      (𝓝 ((uniformPerm (97 * 96)).real labelled)) :=
  not_tendsto_setIntegral_of_twin_subset decay Set.inter_subset_right

example (decay : ℝ) : ¬ ∀ᵐ ω ∂law (97 * 96) 4656 (initMeasure paperSizes), ω.1 ∈ anchored →
    ∀ᶠ t in atTop, ω ∉ wrongAt decay t :=
  not_ae_eventually_of_twin_subset decay fun _ hπ => labelled_subset_anchored hπ.2

end Transformer.Grokking.Protocol
