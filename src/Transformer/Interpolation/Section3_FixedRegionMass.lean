/-
# Untouched mass and the one-step ball-chain estimate

ArXiv:2411.04551v3, Appendix, the proof of
`lem: tubular.mass.movement` after `lem: two.balls`.

An injective particle map which fixes the complement of an active region
cannot move mass across that region's boundary. Every measurable subset
of the complement has exactly its original pushforward mass.

The next region splits into its overlap with the active region and the
untouched difference. The overlap's transported mass and that untouched
mass add without subtraction of extended nonnegative quantities. If the
retained fraction is at most one, this also bounds the retained fraction
of the union's original mass.

These statements concern the actual pushforward of a measure by the
particle map; neither an invariant law nor a mass estimate is assumed
beyond the stated overlap bound supplied by the preceding ball control.
-/

import Transformer.Interpolation.BallTransport

open Real MeasureTheory Set Filter
open scoped ENNReal

namespace Transformer.Interpolation

variable {X : Type*}

/-- Injectivity and a fixed complement make the active region invariant.
Auxiliary for arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`, the one-step mass calculation. -/
theorem mem_iff_of_injective_fix_compl (f : X → X) (hf : Function.Injective f)
    (S : Set X) (hfix : ∀ x ∉ S, f x = x) (x : X) : f x ∈ S ↔ x ∈ S := by
  constructor
  · intro hx
    by_contra hs
    rw [hfix x hs] at hx
    exact hs hx
  · intro hx
    by_contra hs
    have he : f x = x := hf (hfix (f x) hs)
    rw [he] at hs
    exact hs hx

/-- Every subset of the untouched complement has its exact original
preimage. Source: arXiv:2411.04551v3, Appendix, the fixed-region clause
used in the proof of `lem: tubular.mass.movement`. -/
theorem preimage_eq_of_injective_fix_compl (f : X → X) (hf : Function.Injective f)
    (S : Set X) (hfix : ∀ x ∉ S, f x = x) (A : Set X) (hAS : A ⊆ Sᶜ) : f ⁻¹' A = A := by
  ext x
  constructor
  · intro hx
    have he : f x = x := hf (hfix (f x) (hAS hx))
    change f x ∈ A at hx
    rwa [he] at hx
  · intro hx
    change f x ∈ A
    rwa [hfix x (hAS hx)]

/-- The active region has precisely itself as preimage.
Auxiliary for arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`. -/
theorem preimage_self_of_injective_fix_compl (f : X → X) (hf : Function.Injective f)
    (S : Set X) (hfix : ∀ x ∉ S, f x = x) : f ⁻¹' S = S := by
  ext x
  exact mem_iff_of_injective_fix_compl f hf S hfix x

variable [MeasurableSpace X]

/-- The actual pushforward preserves every measurable untouched subset.
Source: arXiv:2411.04551v3, Appendix, the difference-set equality
in the proof of `lem: tubular.mass.movement`. -/
theorem map_apply_eq_of_injective_fix_compl (μ : Measure X) (f : X → X) (hf : Function.Injective f)
    (hfm : Measurable f) (S : Set X) (hfix : ∀ x ∉ S, f x = x)
    (A : Set X) (hA : MeasurableSet A) (hAS : A ⊆ Sᶜ) : Measure.map f μ A = μ A := by
  rw [Measure.map_apply hfm hA, preimage_eq_of_injective_fix_compl f hf S hfix A hAS]

/-- The total mass of the active region is conserved.
Auxiliary for arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`. -/
theorem map_self_of_injective_fix_compl (μ : Measure X) (f : X → X) (hf : Function.Injective f)
    (hfm : Measurable f) (S : Set X) (hS : MeasurableSet S) (hfix : ∀ x ∉ S, f x = x) :
    Measure.map f μ S = μ S := by
  rw [Measure.map_apply hfm hS, preimage_self_of_injective_fix_compl f hf S hfix]

/-- Transported overlap mass and untouched new mass both contribute to
the next region. Source: arXiv:2411.04551v3, Appendix, the first inequality
in the proof of `lem: tubular.mass.movement`. -/
theorem map_successor_mass_add (μ : Measure X) (f : X → X) (hf : Function.Injective f)
    (hfm : Measurable f) (S B : Set X) (hS : MeasurableSet S) (hB : MeasurableSet B)
    (hfix : ∀ x ∉ S, f x = x) (a : ℝ≥0∞)
    (hmove : a * μ S ≤ Measure.map f μ (S ∩ B)) :
    a * μ S + μ (B \ S) ≤ Measure.map f μ B := by
  have hdiff : Measure.map f μ (B \ S) = μ (B \ S) :=
    map_apply_eq_of_injective_fix_compl μ f hf hfm S hfix (B \ S) (hB.diff hS) (fun _ hx => hx.2)
  have hBunion : B = (S ∩ B) ∪ (B \ S) := by
    ext x
    simp only [mem_union, Set.mem_sdiff, mem_inter_iff]
    tauto
  have hdisj : Disjoint (S ∩ B) (B \ S) :=
    disjoint_left.mpr fun _ hx hx' => hx'.2 hx.1
  calc
    _ ≤ Measure.map f μ (S ∩ B) + Measure.map f μ (B \ S) := add_le_add hmove hdiff.ge
    _ = Measure.map f μ B := by
      conv_rhs => rw [hBunion, measure_union hdisj (hB.diff hS)]

/-- A retained fraction in `[0,1]` of the union's original mass reaches
the next region. Source: arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`, the one-step estimate. -/
theorem map_successor_mass_bound (μ : Measure X) (f : X → X) (hf : Function.Injective f)
    (hfm : Measurable f) (S B : Set X) (hS : MeasurableSet S) (hB : MeasurableSet B)
    (hfix : ∀ x ∉ S, f x = x) (a : ℝ≥0∞) (ha : a ≤ 1)
    (hmove : a * μ S ≤ Measure.map f μ (S ∩ B)) :
    a * μ (S ∪ B) ≤ Measure.map f μ B := by
  have hU : S ∪ B = S ∪ (B \ S) := by
    ext x
    simp only [mem_union, Set.mem_sdiff]
    tauto
  have hdisj : Disjoint S (B \ S) := disjoint_left.mpr fun _ hx hx' => hx'.2 hx
  rw [hU, measure_union hdisj (hB.diff hS), mul_add]
  calc
    _ ≤ a * μ S + μ (B \ S) := add_le_add le_rfl
      (by simpa only [one_mul] using mul_le_mul_of_nonneg_right ha (by positivity : 0 ≤ μ (B \ S)))
    _ ≤ Measure.map f μ B := map_successor_mass_add μ f hf hfm S B hS hB hfix a hmove

/-- A nontrivial swap inside two points fixes the third point, which lies
in the next region's untouched difference. -/
example : ∃ (f : Fin 3 → Fin 3) (S B : Set (Fin 3)),
    Function.Injective f ∧ (∀ x ∉ S, f x = x) ∧ B \ S ⊆ Sᶜ ∧
    (0 : Fin 3) ∈ S ∧ (2 : Fin 3) ∈ B \ S := by
  refine ⟨Equiv.swap 0 1, {0, 1}, {1, 2}, (Equiv.swap 0 1).injective, ?_,
    fun _ hx => hx.2, by simp, by norm_num⟩
  intro x hx
  fin_cases x <;> simp_all [Equiv.swap_apply_def]

/-- Actual Dirac mass moves from the first point to the overlap, with
measurable maps and regions and retention one. All mass-estimate premises
are jointly realized, including a nonempty untouched difference. -/
example : ∃ (μ : Measure (Fin 3)) (f : Fin 3 → Fin 3) (S B : Set (Fin 3)),
    Function.Injective f ∧ Measurable f ∧ MeasurableSet S ∧ MeasurableSet B ∧
    (∀ x ∉ S, f x = x) ∧ B \ S ⊆ Sᶜ ∧ (1 : ℝ≥0∞) ≤ 1 ∧
    1 * μ S ≤ Measure.map f μ (S ∩ B) := by
  let f : Fin 3 → Fin 3 := Equiv.swap 0 1
  let S : Set (Fin 3) := {0, 1}
  let B : Set (Fin 3) := {1, 2}
  have hfix : ∀ x ∉ S, f x = x := by
    intro x hx
    fin_cases x <;> simp_all [f, S, Equiv.swap_apply_def]
  refine ⟨Measure.dirac 0, f, S, B, (Equiv.swap 0 1).injective,
    measurable_of_countable f, S.toFinite.measurableSet, B.toFinite.measurableSet,
    hfix, fun _ hx => hx.2, le_rfl, ?_⟩
  simp [f, S, B]

end Transformer.Interpolation
