/-
# Accumulating transported mass along a ball chain

ArXiv:2411.04551v3, Appendix, the final induction in the proof
of `lem: tubular.mass.movement`.

Nonadjacent regions are disjoint. The portion of the next region outside
the current region is therefore disjoint from the entire earlier prefix.
A prefix particle map which fixes that complement preserves the portion's
original mass exactly.

At the next transport step, the previous retained mass contributes one
more factor, while fresh mass contributes its full original value. The
retained fraction is at most one, so the latter also dominates the new
power times the fresh mass. The disjoint union decomposition then gives
the same lower bound for the enlarged chain's initial mass.

All measures are actual pushforwards. The preceding mass bound is an
explicit induction input and the next bound is derived from it.
-/

import Transformer.Interpolation.Section3_FixedRegionMass

open Real MeasureTheory Set
open scoped ENNReal

namespace Transformer.Interpolation

variable {X : Type*}

/-- Adding the next region extends exactly the finite prefix union.
Auxiliary for arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`. -/
theorem chain_union_succ (B : ℕ → Set X) (K : ℕ) :
    (⋃ k ∈ Iic (K + 1), B k) = (⋃ k ∈ Iic K, B k) ∪ B (K + 1) := by
  ext x
  simp only [mem_iUnion, mem_Iic, mem_union]
  constructor
  · rintro ⟨k, hk, hx⟩
    by_cases h : k ≤ K
    · exact Or.inl ⟨k, h, hx⟩
    · have he : k = K + 1 := by omega
      exact Or.inr (he ▸ hx)
  · rintro (⟨k, hk, hx⟩ | hx)
    · exact ⟨k, by omega, hx⟩
    · exact ⟨K + 1, le_rfl, hx⟩

/-- Nonadjacent disjointness keeps the new difference outside every
preceding region. Source: arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`, the backward mass calculation. -/
theorem chain_fresh_disjoint (B : ℕ → Set X) (K : ℕ)
    (hd : ∀ j : ℕ, j + 2 ≤ K + 1 → Disjoint (B j) (B (K + 1))) :
    Disjoint (⋃ k ∈ Iic K, B k) (B (K + 1) \ B K) := by
  apply disjoint_left.mpr
  intro x hx hy
  obtain ⟨k, hx⟩ := mem_iUnion.mp hx
  obtain ⟨hk, hx⟩ := mem_iUnion.mp hx
  by_cases he : k = K
  · exact hy.2 (he ▸ hx)
  · exact disjoint_left.mp (hd k (by change k ≤ K at hk; omega)) hx hy.1

variable [MeasurableSpace X]

/-- The actual prefix pushforward leaves the next fresh mass unchanged.
Source: arXiv:2411.04551v3, Appendix, the difference-set equality
in the induction for `lem: tubular.mass.movement`. -/
theorem map_chain_fresh_mass (μ : Measure X) (f : X → X) (hf : Function.Injective f)
    (hm : Measurable f) (B : ℕ → Set X) (K : ℕ)
    (hB : ∀ k, MeasurableSet (B k))
    (hd : ∀ j : ℕ, j + 2 ≤ K + 1 → Disjoint (B j) (B (K + 1)))
    (hfix : ∀ x ∉ ⋃ k ∈ Iic K, B k, f x = x) :
    Measure.map f μ (B (K + 1) \ B K) = μ (B (K + 1) \ B K) := by
  apply map_apply_eq_of_injective_fix_compl μ f hf hm (⋃ k ∈ Iic K, B k) hfix
    (B (K + 1) \ B K) ((hB _).diff (hB _))
  exact fun _ hx hy => disjoint_left.mp (chain_fresh_disjoint B K hd) hy hx

/-- One more actual transport multiplies the chain's retained fraction
by one more factor. Source: arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`, the induction step. -/
theorem chain_mass_step (μ₀ μ : Measure X) (f : X → X) (hf : Function.Injective f)
    (hm : Measurable f) (B : ℕ → Set X) (K : ℕ)
    (hB : ∀ k, MeasurableSet (B k))
    (hd : ∀ j : ℕ, j + 2 ≤ K + 1 → Disjoint (B j) (B (K + 1)))
    (hfix : ∀ x ∉ B K, f x = x) (a : ℝ≥0∞) (ha : a ≤ 1)
    (hprev : a ^ K * μ₀ (⋃ k ∈ Iic K, B k) ≤ μ (B K))
    (hfresh : μ (B (K + 1) \ B K) = μ₀ (B (K + 1) \ B K))
    (hmove : a * μ (B K) ≤ Measure.map f μ (B K ∩ B (K + 1))) :
    a ^ (K + 1) * μ₀ (⋃ k ∈ Iic (K + 1), B k) ≤ Measure.map f μ (B (K + 1)) := by
  have hU : (⋃ k ∈ Iic (K + 1), B k) =
      (⋃ k ∈ Iic K, B k) ∪ (B (K + 1) \ B K) := by
    rw [chain_union_succ]
    ext x
    simp only [mem_union, Set.mem_sdiff]
    have hsub : x ∈ B K → x ∈ ⋃ k ∈ Iic K, B k :=
      fun hx => mem_iUnion.mpr ⟨K, mem_iUnion.mpr ⟨by change K ≤ K; exact le_rfl, hx⟩⟩
    tauto
  rw [hU, measure_union (chain_fresh_disjoint B K hd) ((hB _).diff (hB _)), mul_add]
  have hp : a ^ (K + 1) ≤ 1 := pow_le_one₀ (by positivity) ha
  calc
    _ ≤ a * μ (B K) + μ₀ (B (K + 1) \ B K) := add_le_add
      (by simpa only [pow_succ', mul_assoc] using (mul_le_mul_of_nonneg_left hprev (by positivity : 0 ≤ a)))
      (by simpa only [one_mul] using
        (mul_le_mul_of_nonneg_right hp (by positivity : 0 ≤ μ₀ (B (K + 1) \ B K))))
    _ = a * μ (B K) + μ (B (K + 1) \ B K) := by rw [hfresh]
    _ ≤ _ := map_successor_mass_add μ f hf hm (B K) (B (K + 1)) (hB _) (hB _) hfix a hmove

/-- Three explicit finite regions have adjacent overlaps and disjoint
end regions. Moving a Dirac mass from zero to one realizes the previous
bound, and the next identity step realizes every induction premise. -/
example : ∃ (B : ℕ → Set (Fin 3)) (μ₀ μ : Measure (Fin 3)) (f : Fin 3 → Fin 3),
    (∀ k, MeasurableSet (B k)) ∧
    (∀ j : ℕ, j + 2 ≤ 2 → Disjoint (B j) (B 2)) ∧
    Function.Injective f ∧ Measurable f ∧
    (∀ x ∉ B 1, f x = x) ∧ (1 : ℝ≥0∞) ≤ 1 ∧
    1 ^ 1 * μ₀ (⋃ k ∈ Iic 1, B k) ≤ μ (B 1) ∧
    μ (B 2 \ B 1) = μ₀ (B 2 \ B 1) ∧
    1 * μ (B 1) ≤ Measure.map f μ (B 1 ∩ B 2) := by
  let B : ℕ → Set (Fin 3) := fun k => if k = 0 then {0} else if k = 1 then {0, 1} else {1, 2}
  have hmeas (k : ℕ) : MeasurableSet (B k) := (B k).toFinite.measurableSet
  have hdisj (j : ℕ) (hj : j + 2 ≤ 2) : Disjoint (B j) (B 2) := by
    have he : j = 0 := by omega
    subst j
    simp [B]
  refine ⟨B, Measure.dirac 0, Measure.dirac 1, id, hmeas, hdisj,
    Function.injective_id, measurable_id, fun _ _ => rfl, le_rfl, ?_, ?_, ?_⟩
  · simp only [one_pow, one_mul]
    calc
      _ ≤ (Measure.dirac (0 : Fin 3)) univ := measure_mono (subset_univ _)
      _ = (Measure.dirac (1 : Fin 3)) (B 1) := by simp [B]
  · simp [B]
  · simp [B]

/-- A nontrivial prefix swap realizes the fixed-prefix and measurable
pushforward premises for preservation of fresh mass. -/
example : ∃ (B : ℕ → Set (Fin 3)) (f : Fin 3 → Fin 3),
    (∀ k, MeasurableSet (B k)) ∧ Function.Injective f ∧ Measurable f ∧
    (∀ j : ℕ, j + 2 ≤ 2 → Disjoint (B j) (B 2)) ∧
    (∀ x ∉ ⋃ k ∈ Iic 1, B k, f x = x) := by
  let B : ℕ → Set (Fin 3) := fun k => if k = 0 then {0} else if k = 1 then {0, 1} else {1, 2}
  refine ⟨B, Equiv.swap 0 1, fun k => (B k).toFinite.measurableSet,
    (Equiv.swap 0 1).injective, measurable_of_countable _, ?_, ?_⟩
  · intro j hj
    have he : j = 0 := by omega
    subst j
    simp [B]
  · intro x hx
    have hnot : x ∉ B 1 := fun h => hx (mem_iUnion.mpr ⟨1, mem_iUnion.mpr ⟨by simp, h⟩⟩)
    fin_cases x <;> simp_all [B, Equiv.swap_apply_def]

end Transformer.Interpolation
