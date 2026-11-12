module

public import Transformer.Grokking.Protocol.Target

/-!
# A symmetry of the law of a run halves a set of splits

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*;
`experiments/grok_division/PROTOCOL.md`, *The split* and *Randomness*. The law of a run is
`law` of `Transformer.Grokking.Protocol.Target`.

The refutations of the target share one step. A map of samples preserves the law of a run and
keeps the split in a set `A` of splits; and of every sample with a split in `A`, the sample or
its image is in a set `W` of samples. Then the samples of `W` with a split in `A` have at
least half the probability of `A` (`uniformPerm_le_two_mul_of_or`): they and their preimage,
which has no more probability, cover the samples with a split in `A`. And a set of splits on
each of which that holds has the same bound (`uniformPerm_le_two_mul_of_singleton`): the splits
are finitely many, and the samples with a given split are an event. `W` need not be
measurable.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open MeasureTheory

variable {ι : Type*} {N k : ℕ}

/-- **A symmetry of the law that keeps a set of splits halves it**: if a map of samples
preserves the law of a run and keeps the split in `A`, and of every sample with a split in `A`
the sample or its image is in `W`, then the samples of `W` with a split in `A` have at least
half the probability of `A`. -/
theorem uniformPerm_le_two_mul_of_or {ν : Measure (ι → ℝ)} [IsProbabilityMeasure ν]
    {R : Sample N k ι → Sample N k ι} (hR : MeasurePreserving R (law N k ν) (law N k ν))
    {A : Set (Equiv.Perm (Fin N))} {W : Set (Sample N k ι)}
    (hRA : ∀ ω, ω.1 ∈ A → (R ω).1 ∈ A) (hW : ∀ ω, ω.1 ∈ A → ω ∈ W ∨ R ω ∈ W) :
    uniformPerm N A ≤ 2 * law N k ν ((A ×ˢ Set.univ) ∩ W) := by
  set μ := law N k ν with hμ
  set E := (A ×ˢ (Set.univ : Set ((ι → ℝ) × (ℕ → Equiv.Perm (Fin k))))) ∩ W with hE
  have hF : μ (A ×ˢ Set.univ) = uniformPerm N A := by
    rw [hμ, law, Measure.prod_prod, measure_univ, mul_one]
  have hpre : μ (R ⁻¹' E) ≤ μ E :=
    calc μ (R ⁻¹' E) ≤ μ (R ⁻¹' toMeasurable μ E) :=
          measure_mono (Set.preimage_mono (subset_toMeasurable _ _))
      _ = μ (toMeasurable μ E) :=
          hR.measure_preimage (measurableSet_toMeasurable _ _).nullMeasurableSet
      _ = μ E := measure_toMeasurable _
  calc uniformPerm N A = μ (A ×ˢ Set.univ) := hF.symm
    _ ≤ μ (E ∪ R ⁻¹' E) := measure_mono fun ω hω => by
        rcases hW ω hω.1 with h | h
        · exact Or.inl ⟨hω, h⟩
        · exact Or.inr ⟨⟨hRA ω hω.1, Set.mem_univ _⟩, h⟩
    _ ≤ μ E + μ (R ⁻¹' E) := measure_union_le _ _
    _ ≤ μ E + μ E := add_le_add le_rfl hpre
    _ = 2 * μ E := (two_mul _).symm

/-- The same with the whole of `W`. -/
theorem uniformPerm_le_two_mul_law_of_or {ν : Measure (ι → ℝ)} [IsProbabilityMeasure ν]
    {R : Sample N k ι → Sample N k ι} (hR : MeasurePreserving R (law N k ν) (law N k ν))
    {A : Set (Equiv.Perm (Fin N))} {W : Set (Sample N k ι)}
    (hRA : ∀ ω, ω.1 ∈ A → (R ω).1 ∈ A) (hW : ∀ ω, ω.1 ∈ A → ω ∈ W ∨ R ω ∈ W) :
    uniformPerm N A ≤ 2 * law N k ν W :=
  (uniformPerm_le_two_mul_of_or hR hRA hW).trans
    (mul_le_mul_right (measure_mono Set.inter_subset_right) 2)

/-- **A halving that holds on each split of a set holds on the set**: the splits are finitely
many, and the samples with a given split are an event. -/
theorem uniformPerm_le_two_mul_of_singleton {ν : Measure (ι → ℝ)} [IsProbabilityMeasure ν]
    {S : Set (Equiv.Perm (Fin N))} {W : Set (Sample N k ι)}
    (h : ∀ π ∈ S, uniformPerm N {π} ≤ 2 * law N k ν (({π} ×ˢ Set.univ) ∩ W)) :
    uniformPerm N S ≤ 2 * law N k ν ((S ×ˢ Set.univ) ∩ W) := by
  classical
  set μ := (law N k ν).restrict W with hμ
  have hmeas : ∀ A : Set (Equiv.Perm (Fin N)), MeasurableSet
      (A ×ˢ (Set.univ : Set ((ι → ℝ) × (ℕ → Equiv.Perm (Fin k))))) := fun A =>
    (MeasurableSpace.measurableSet_top).prod MeasurableSet.univ
  have hμA : ∀ A : Set (Equiv.Perm (Fin N)), μ (A ×ˢ Set.univ) =
      law N k ν ((A ×ˢ Set.univ) ∩ W) := fun A => by
    rw [hμ, Measure.restrict_apply (hmeas A)]
  set F := S.toFinite.toFinset with hFdef
  have hmem : ∀ π, π ∈ F ↔ π ∈ S := fun π => Set.Finite.mem_toFinset _
  have hS1 : S = ⋃ π ∈ F, ({π} : Set (Equiv.Perm (Fin N))) := by
    ext π
    simp [hmem]
  have hS2 : (S ×ˢ (Set.univ : Set ((ι → ℝ) × (ℕ → Equiv.Perm (Fin k))))) =
      ⋃ π ∈ F, ({π} : Set (Equiv.Perm (Fin N))) ×ˢ Set.univ := by
    ext ω
    simp [hmem]
  have hsum1 : uniformPerm N S = ∑ π ∈ F, uniformPerm N {π} := by
    conv_lhs => rw [hS1]
    exact measure_biUnion_finset (fun a _ b _ hab => Set.disjoint_singleton.2 hab)
      fun _ _ => MeasurableSpace.measurableSet_top
  have hsum2 : μ (S ×ˢ Set.univ) =
      ∑ π ∈ F, μ (({π} : Set (Equiv.Perm (Fin N))) ×ˢ Set.univ) := by
    conv_lhs => rw [hS2]
    refine measure_biUnion_finset (fun a _ b _ hab => ?_) fun π _ => hmeas {π}
    exact Set.disjoint_left.2 fun ω ha hb => hab (ha.1.symm.trans hb.1)
  rw [← hμA, hsum1, hsum2, Finset.mul_sum]
  refine Finset.sum_le_sum fun π hπ => ?_
  rw [hμA]
  exact h π ((hmem π).1 hπ)

/-! The witnesses: two shuffles of two equations, the identity as the symmetry and every sample
in `W`; the split `1`, and every split by the sum over the splits. -/

example : uniformPerm 2 {1} ≤
    2 * law 2 1 (Measure.dirac (0 : Unit → ℝ)) (({1} ×ˢ Set.univ) ∩ Set.univ) :=
  uniformPerm_le_two_mul_of_or (MeasurePreserving.id _) (fun _ h => h) fun _ _ => Or.inl trivial

example : uniformPerm 2 {1} ≤ 2 * law 2 1 (Measure.dirac (0 : Unit → ℝ)) Set.univ :=
  uniformPerm_le_two_mul_law_of_or (MeasurePreserving.id _) (fun _ h => h)
    fun _ _ => Or.inl trivial

example : uniformPerm 2 Set.univ ≤
    2 * law 2 1 (Measure.dirac (0 : Unit → ℝ)) ((Set.univ ×ˢ Set.univ) ∩ Set.univ) :=
  uniformPerm_le_two_mul_of_singleton fun π _ => uniformPerm_le_two_mul_of_or
    (A := {π}) (MeasurePreserving.id _) (fun _ h => h) fun _ _ => Or.inl trivial

end Transformer.Grokking.Protocol
