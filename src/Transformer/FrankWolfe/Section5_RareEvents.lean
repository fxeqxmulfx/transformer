/-
# Attention's forward pass and Frank-Wolfe — counting rare dependent steps

The binomial estimate in the proof of `lem: metastab.1` needs only an upper
bound on the probability of a bad step conditional on any event of the past.
The steps need not be independent. Intersecting at prescribed times gives
`p^m`; a union bound over the `m`-element subsets gives `choose(N,m) p^m`.

The events below are data on the actual path space of `IsSAProcess`. The
conditional bound is an explicit input, to be supplied by its transition
identity and the score estimate `cl: group.cluster.bound`.
Source: arXiv:2508.09628v1, §5, proof of `lem: metastab.1`,
`eq: union.bound`, `eq: binomial.2`.
-/

import Transformer.FrankWolfe.Section5_Transition
import Mathlib.Data.Finset.Powerset

open scoped ENNReal BigOperators
open MeasureTheory

namespace Transformer.FrankWolfe

variable {d n : ℕ}

/-- A finite intersection of events determined by the same past is still
determined by that past. Source: arXiv:2508.09628v1, §5, auxiliary to
`eq: union.bound`. -/
theorem pastDet_finset_biInter (t : ℕ) (B : Finset ℕ)
    (E : ℕ → Set (ℕ → Idx n → EucSpace d)) (hE : ∀ k ∈ B, PastDet t (E k)) :
    PastDet t (⋂ k ∈ B, E k) := by
  refine ⟨MeasurableSet.iInter fun k => MeasurableSet.iInter fun hk => (hE k hk).1, ?_⟩
  intro x y hxy
  simp only [Set.mem_iInter]
  exact forall_congr' fun k => forall_congr' fun hk => (hE k hk).2 x y hxy

example (t : ℕ) (B : Finset ℕ) :
    ∀ k ∈ B, PastDet t (Set.univ : Set (ℕ → Idx 1 → EucSpace 1)) :=
  fun _ _ => pastDet_univ t

/-- At prescribed times, a conditional rare-step bound multiplies, without
any independence hypothesis. Source: arXiv:2508.09628v1, §5, auxiliary to
the binomial estimate in the proof of `lem: metastab.1`. -/
theorem measure_biInter_rare_steps_le
    (P : Measure (ℕ → Idx n → EucSpace d)) [IsProbabilityMeasure P]
    (E : ℕ → Set (ℕ → Idx n → EucSpace d)) (p : ℝ≥0∞)
    (hE : ∀ t, PastDet (t + 1) (E t))
    (hstep : ∀ t S, PastDet t S → P (S ∩ E t) ≤ p * P S)
    (N : ℕ) (B : Finset ℕ) (hB : B ⊆ Finset.range N) :
    P (⋂ k ∈ B, E k) ≤ p ^ B.card := by
  classical
  induction N generalizing B with
  | zero =>
    have he : B = ∅ := Finset.subset_empty.mp (by simpa using hB)
    subst B
    simp
  | succ N ih =>
    by_cases hN : N ∈ B
    · have hC : B.erase N ⊆ Finset.range N := by
        intro k hk
        have hkb := Finset.mem_of_mem_erase hk
        have hlt := Finset.mem_range.mp (hB hkb)
        have hne := (Finset.mem_erase.mp hk).1
        exact Finset.mem_range.mpr (by omega)
      have hpast : PastDet N (⋂ k ∈ B.erase N, E k) :=
        pastDet_finset_biInter N (B.erase N) E fun k hk =>
          (hE k).mono (by have := Finset.mem_range.mp (hC hk); omega)
      have he : (⋂ k ∈ B, E k) = (⋂ k ∈ B.erase N, E k) ∩ E N := by
        ext x
        simp only [Set.mem_iInter, Set.mem_inter_iff]
        constructor
        · intro h
          exact ⟨fun k hk => h k (Finset.mem_of_mem_erase hk), h N hN⟩
        · rintro ⟨h, hlast⟩ k hk
          by_cases hkN : k = N
          · simpa only [hkN] using hlast
          · exact h k (Finset.mem_erase.mpr ⟨hkN, hk⟩)
      rw [he]
      calc P ((⋂ k ∈ B.erase N, E k) ∩ E N)
          ≤ p * P (⋂ k ∈ B.erase N, E k) := hstep N _ hpast
        _ ≤ p * p ^ (B.erase N).card :=
          mul_le_mul_of_nonneg_left (ih _ hC) zero_le
        _ = p ^ B.card := by
          rw [← Finset.card_erase_add_one hN, pow_succ, mul_comm]
    · apply ih
      intro k hk
      have hlt := Finset.mem_range.mp (hB hk)
      have hne : k ≠ N := fun h => hN (h ▸ hk)
      exact Finset.mem_range.mpr (by omega)

/-- Empty rare-step events realize both inputs, on a genuine probability
space and with conditional bound zero. -/
example (P : Measure (ℕ → Idx 1 → EucSpace 1)) :
    (∀ t, PastDet (t + 1) (∅ : Set (ℕ → Idx 1 → EucSpace 1))) ∧
      (∀ t S, PastDet t S → P (S ∩ ∅) ≤ (0 : ℝ≥0∞) * P S) := by
  refine ⟨fun _ => ⟨MeasurableSet.empty, fun _ _ _ => Iff.rfl⟩, ?_⟩
  intro t S _
  simp

/-- The observed times of the event family before the finite horizon `N`.
This is the actual count used in the binomial union estimate. -/
noncomputable def rareStepTimes (E : ℕ → Set (ℕ → Idx n → EucSpace d)) (N : ℕ)
    (x : ℕ → Idx n → EucSpace d) : Finset ℕ := by
  classical
  exact (Finset.range N).filter (fun k => x ∈ E k)

/-- The count of rare steps is bounded by the binomial union estimate even
when the steps are dependent. Source: arXiv:2508.09628v1, §5, the binomial
estimate in the proof of `lem: metastab.1`. -/
theorem measure_count_rare_steps_le
    (P : Measure (ℕ → Idx n → EucSpace d)) [IsProbabilityMeasure P]
    (E : ℕ → Set (ℕ → Idx n → EucSpace d)) (p : ℝ≥0∞)
    (hE : ∀ t, PastDet (t + 1) (E t))
    (hstep : ∀ t S, PastDet t S → P (S ∩ E t) ≤ p * P S) (N M : ℕ) :
    P {x | M ≤ (rareStepTimes E N x).card} ≤
      (N.choose M : ℝ≥0∞) * p ^ M := by
  classical
  have hsub : {x | M ≤ (rareStepTimes E N x).card} ⊆
      ⋃ B ∈ (Finset.range N).powersetCard M, ⋂ k ∈ B, E k := by
    intro x hx
    obtain ⟨B, hB, hcard⟩ := Finset.exists_subset_card_eq hx
    have hrange : B ⊆ Finset.range N := hB.trans (Finset.filter_subset _ _)
    refine Set.mem_iUnion.mpr ⟨B, Set.mem_iUnion.mpr ⟨?_, ?_⟩⟩
    · exact Finset.mem_powersetCard.mpr ⟨hrange, hcard⟩
    · simp only [Set.mem_iInter]
      exact fun k hk => (Finset.mem_filter.mp (hB hk)).2
  calc P {x | M ≤ (rareStepTimes E N x).card}
      ≤ P (⋃ B ∈ (Finset.range N).powersetCard M, ⋂ k ∈ B, E k) := measure_mono hsub
    _ ≤ ∑ B ∈ (Finset.range N).powersetCard M, P (⋂ k ∈ B, E k) :=
      measure_biUnion_finset_le _ _
    _ ≤ ∑ B ∈ (Finset.range N).powersetCard M, p ^ M := by
      apply Finset.sum_le_sum
      intro B hB
      obtain ⟨hrange, hcard⟩ := Finset.mem_powersetCard.mp hB
      simpa only [hcard] using measure_biInter_rare_steps_le P E p hE hstep N B hrange
    _ = (N.choose M : ℝ≥0∞) * p ^ M := by simp

/-- The tail estimate's inputs hold at the Dirac law of the stationary origin
path, with no rare events. The counted family need not be empty. -/
example :
    IsProbabilityMeasure (Measure.dirac (fun _ : ℕ => fun _ : Idx 1 => (0 : EucSpace 1))) ∧
      (∀ t, PastDet (t + 1) (∅ : Set (ℕ → Idx 1 → EucSpace 1))) ∧
      (∀ t S, PastDet t S →
        (Measure.dirac (fun _ : ℕ => fun _ : Idx 1 => (0 : EucSpace 1))) (S ∩ ∅) ≤
          (0 : ℝ≥0∞) * (Measure.dirac (fun _ : ℕ => fun _ : Idx 1 => (0 : EucSpace 1))) S) := by
  refine ⟨inferInstance, fun _ => ⟨MeasurableSet.empty, fun _ _ _ => Iff.rfl⟩, ?_⟩
  intro t S _
  simp

end Transformer.FrankWolfe
