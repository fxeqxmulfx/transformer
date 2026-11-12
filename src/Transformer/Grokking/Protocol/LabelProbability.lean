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
of splits that are not labelled (`uniformPerm_le_of_subset_compl_labelled`): given that the
split is not labelled, a held-out answer is wrong with a conditional probability of at least
one half at every update, for Adam and for AdamW at every weight decay.

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
  set μ := law (97 * 96) 4656 (initMeasure paperSizes) with hμ
  set E := splitIn {π} ∩ wrongAt decay t with hE
  have hF : μ (splitIn {π}) = uniformPerm (97 * 96) {π} := by
    rw [hμ, law, Measure.prod_prod, measure_univ, mul_one]
  have hpre : μ (relabelSample r ⁻¹' E) ≤ μ E :=
    calc μ (relabelSample r ⁻¹' E)
        ≤ μ (relabelSample r ⁻¹' toMeasurable μ E) :=
          measure_mono (Set.preimage_mono (subset_toMeasurable _ _))
      _ = μ (toMeasurable μ E) := (measurePreserving_relabelSample r).measure_preimage
            (measurableSet_toMeasurable _ _).nullMeasurableSet
      _ = μ E := measure_toMeasurable _
  calc uniformPerm (97 * 96) {π} = μ (splitIn {π}) := hF.symm
    _ ≤ μ (E ∪ relabelSample r ⁻¹' E) := measure_mono fun ω hω => by
        have hω1 : ω.1 = π := hω.1
        rcases mem_wrongAt_or_relabel decay t (ω := ω) (hω1 ▸ hr) with h | h
        · exact Or.inl ⟨hω, h⟩
        · exact Or.inr ⟨⟨hω.1, Set.mem_univ _⟩, h⟩
    _ ≤ μ E + μ (relabelSample r ⁻¹' E) := measure_union_le _ _
    _ ≤ μ E + μ E := add_le_add le_rfl hpre
    _ = 2 * μ E := (two_mul _).symm

/-- **Given that the split is in a set of splits that are not labelled, a held-out answer is
wrong with a conditional probability of at least one half, at every update.** -/
theorem uniformPerm_le_of_subset_compl_labelled (decay : ℝ) (t : ℕ)
    {S : Set (Equiv.Perm (Fin (97 * 96)))} (hS : S ⊆ labelledᶜ) :
    uniformPerm (97 * 96) S ≤
      2 * law (97 * 96) 4656 (initMeasure paperSizes) (splitIn S ∩ wrongAt decay t) := by
  classical
  set ν := (law (97 * 96) 4656 (initMeasure paperSizes)).restrict (wrongAt decay t) with hν
  have hmeas : ∀ A : Set (Equiv.Perm (Fin (97 * 96))), MeasurableSet (splitIn A) := fun A =>
    (MeasurableSpace.measurableSet_top).prod MeasurableSet.univ
  have hνA : ∀ A : Set (Equiv.Perm (Fin (97 * 96))), ν (splitIn A) =
      law (97 * 96) 4656 (initMeasure paperSizes) (splitIn A ∩ wrongAt decay t) := fun A => by
    rw [hν, Measure.restrict_apply (hmeas A)]
  set F := S.toFinite.toFinset with hFdef
  have hmem : ∀ π, π ∈ F ↔ π ∈ S := fun π => Set.Finite.mem_toFinset _
  have hS1 : S = ⋃ π ∈ F, ({π} : Set (Equiv.Perm (Fin (97 * 96)))) := by
    ext π
    simp [hmem]
  have hS2 : splitIn S = ⋃ π ∈ F, splitIn {π} := by
    ext ω
    simp [splitIn, hmem]
  have hsum1 : uniformPerm (97 * 96) S = ∑ π ∈ F, uniformPerm (97 * 96) {π} := by
    conv_lhs => rw [hS1]
    exact measure_biUnion_finset (fun a _ b _ hab => Set.disjoint_singleton.2 hab)
      fun _ _ => MeasurableSpace.measurableSet_top
  have hsum2 : ν (splitIn S) = ∑ π ∈ F, ν (splitIn {π}) := by
    conv_lhs => rw [hS2]
    refine measure_biUnion_finset (fun a _ b _ hab => ?_) fun π _ => hmeas {π}
    exact Set.disjoint_left.2 fun ω ha hb => hab (ha.1.symm.trans hb.1)
  rw [← hνA, hsum1, hsum2, Finset.mul_sum]
  refine Finset.sum_le_sum fun π hπ => ?_
  rw [hνA]
  exact uniformPerm_singleton_le decay t (hS ((hmem π).1 hπ))

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
