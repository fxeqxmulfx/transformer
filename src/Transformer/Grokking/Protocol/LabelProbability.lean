module

public import Transformer.Grokking.Protocol.LabelWitness

/-!
# On the splits that are not labelled a held-out answer is wrong half the time, at every update

Source: arXiv:2201.02177v1 (Power et al.), Figure 1 and the appendix *Additional experimental
details*; openai/grok at 3d64b1d8; `experiments/grok_division/PROTOCOL.md`, *The split*;
`review/grok_division_convergence.md`, section 2.

`LabelCoverage` refutes the almost sure target under a condition on the split that admits a
split that is not labelled. The relabelling does not move the split, so its argument can be
made on the samples of one split: on a split that is not labelled, at every update, the
samples with a wrong held-out answer have at least half the probability of the split
(`uniformPerm_singleton_le`). The splits are finitely many, so the same holds for every set
of splits that are not labelled (`uniformPerm_le_of_subset_compl_labelled`, by the sum over
the splits of `SymmetricLaw`): given that the split is not labelled, a held-out answer is
wrong with a conditional probability of at least one half at every update, for Adam and for
AdamW at every weight decay.

So under a condition on the split that admits a split that is not labelled, the probability
that the condition holds and a held-out answer is wrong does not tend to zero
(`not_tendsto_law_inter_wrongAt`): the target fails in probability too, not only almost
surely.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology MeasureTheory

/-- The samples whose split is in `S`. -/
abbrev splitIn (S : Set (Equiv.Perm (Fin (97 * 96)))) :
    Set (Sample (97 * 96) 4656 (Param paperSizes)) :=
  S ×ˢ (Set.univ : Set ((Param paperSizes → ℝ) × (ℕ → Equiv.Perm (Fin 4656))))

/-- **On a split that is not labelled, at every update the samples with a wrong held-out
answer have at least half the probability of the split.** -/
theorem uniformPerm_singleton_le (decay : ℝ) (t : ℕ) {π : Equiv.Perm (Fin (97 * 96))}
    (hπ : π ∉ labelled) :
    uniformPerm (97 * 96) {π} ≤
      2 * law (97 * 96) 4656 (initMeasure paperSizes) (splitIn {π} ∩ wrongAt decay t) := by
  obtain ⟨r, hr⟩ := not_mem_labelled_iff.1 hπ
  exact uniformPerm_le_two_mul_of_or (measurePreserving_relabelSample r) (fun _ hω => hω)
    fun ω hω => mem_wrongAt_or_relabel decay t (by rwa [Set.mem_singleton_iff.1 hω])

/-- **Given that the split is in a set of splits that are not labelled, a held-out answer is
wrong with a conditional probability of at least one half, at every update.** -/
theorem uniformPerm_le_of_subset_compl_labelled (decay : ℝ) (t : ℕ)
    {S : Set (Equiv.Perm (Fin (97 * 96)))} (hS : S ⊆ labelledᶜ) :
    uniformPerm (97 * 96) S ≤
      2 * law (97 * 96) 4656 (initMeasure paperSizes) (splitIn S ∩ wrongAt decay t) :=
  uniformPerm_le_two_mul_of_singleton fun _ hπ => uniformPerm_singleton_le decay t (hS hπ)

/-- **Under a condition on the split that admits a split that is not labelled the target
fails in probability**: the probability that the condition holds and a held-out answer is
wrong does not tend to zero. -/
theorem not_tendsto_law_inter_wrongAt (decay : ℝ) {S : Set (Equiv.Perm (Fin (97 * 96)))}
    {π : Equiv.Perm (Fin (97 * 96))} (hπS : π ∈ S) (hπ : π ∉ labelled) :
    ¬ Tendsto (fun t => law (97 * 96) 4656 (initMeasure paperSizes)
      (splitIn S ∩ wrongAt decay t)) atTop (𝓝 0) := fun hlim => by
  have hbound : ∀ t : ℕ, uniformPerm (97 * 96) {π} ≤
      2 * law (97 * 96) 4656 (initMeasure paperSizes) (splitIn S ∩ wrongAt decay t) := fun t =>
    (uniformPerm_singleton_le decay t hπ).trans (mul_le_mul_right (measure_mono
      (Set.inter_subset_inter_left _ (Set.prod_mono (Set.singleton_subset_iff.2 hπS) le_rfl))) 2)
  have hzero := ge_of_tendsto' (ENNReal.Tendsto.const_mul hlim (Or.inr ENNReal.ofNat_ne_top))
    hbound
  rw [mul_zero] at hzero
  exact (uniformPerm_pos_of_mem (Set.mem_singleton π)).not_ge hzero

/-! The witnesses: the split `1`, which is anchored and not labelled; the set of all the
splits that are not labelled; and the condition `anchored`, under which the target fails in
probability. -/

example (decay : ℝ) (t : ℕ) : uniformPerm (97 * 96) {1} ≤
    2 * law (97 * 96) 4656 (initMeasure paperSizes) (splitIn {1} ∩ wrongAt decay t) :=
  uniformPerm_singleton_le decay t one_notMem_labelled

example (decay : ℝ) (t : ℕ) : uniformPerm (97 * 96) labelledᶜ ≤
    2 * law (97 * 96) 4656 (initMeasure paperSizes) (splitIn labelledᶜ ∩ wrongAt decay t) :=
  uniformPerm_le_of_subset_compl_labelled decay t subset_rfl

example (decay : ℝ) : ¬ Tendsto (fun t => law (97 * 96) 4656 (initMeasure paperSizes)
    (splitIn anchored ∩ wrongAt decay t)) atTop (𝓝 0) :=
  not_tendsto_law_inter_wrongAt decay one_mem_anchored one_notMem_labelled

end Transformer.Grokking.Protocol
