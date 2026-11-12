module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery.Basic

/-!
# The codings of the training rectangles depend only on the set of observations

Source: `review/flat_spectrum.md`, CM17, *The linear system*: "The solution space depends only
on the set of observed (cell, label) pairs: not on the order of the samples, their repetitions
or their grouping into batches." Native reference:
`experiments/convex_stress_relabel/src/coding.rs` at 414431c7, whose elimination reads the
observations in the order given.

More observations allow fewer codings (`rectangleCodes_anti`), so two training sets with the
same observed cells and labels allow the same ones (`rectangleCodes_congr`,
`pinnedCodes_congr`). In particular reading the samples through a surjection, which covers a
reordering, a repetition and a stream of batches that together show every sample, changes
nothing (`rectangleCodes_comp_surjective`).

This is the invariance of the linear system. It is not a claim that the coding can be found
from a stream without keeping the observations.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery

variable {p : ℕ} {C n S S' : Type*} {cell : S → n × n} {labels : S → C} {cell' : S' → n × n}
  {labels' : S' → C}

/-- More observations allow fewer codings. -/
lemma rectangleCodes_anti
    (h : Set.range (fun t => (cell' t, labels' t)) ⊆ Set.range fun t => (cell t, labels t)) :
    rectangleCodes p cell labels ≤ rectangleCodes p cell' labels' := by
  intro z hz t₁ t₂ t₃ t₄ hr
  obtain ⟨s₁, h₁⟩ := h ⟨t₁, rfl⟩
  obtain ⟨s₂, h₂⟩ := h ⟨t₂, rfl⟩
  obtain ⟨s₃, h₃⟩ := h ⟨t₃, rfl⟩
  obtain ⟨s₄, h₄⟩ := h ⟨t₄, rfl⟩
  simp only [Prod.mk.injEq] at h₁ h₂ h₃ h₄
  have key := hz s₁ s₂ s₃ s₄ (by simpa only [Rectangle, h₁.1, h₂.1, h₃.1, h₄.1] using hr)
  rwa [h₁.2, h₂.2, h₃.2, h₄.2] at key

/-- **The codings depend only on the set of observed cells with their labels** (CM17): the
order of the samples, their repetitions and their grouping into batches do not matter. -/
theorem rectangleCodes_congr
    (h : Set.range (fun t => (cell t, labels t)) = Set.range fun t => (cell' t, labels' t)) :
    rectangleCodes p cell labels = rectangleCodes p cell' labels' :=
  le_antisymm (rectangleCodes_anti h.ge) (rectangleCodes_anti h.le)

/-- The pinned codings depend only on the set of observed cells with their labels. -/
theorem pinnedCodes_congr (c₀ : C)
    (h : Set.range (fun t => (cell t, labels t)) = Set.range fun t => (cell' t, labels' t)) :
    pinnedCodes p cell labels c₀ = pinnedCodes p cell' labels' c₀ := by
  rw [pinnedCodes, pinnedCodes, rectangleCodes_congr h]

/-- **Reading the samples through a surjection keeps the codings**: a reordering, a repetition
of samples, a stream of batches that together show every sample. -/
theorem rectangleCodes_comp_surjective (cell : S → n × n) (labels : S → C) {f : S' → S}
    (hf : Function.Surjective f) :
    rectangleCodes p (cell ∘ f) (labels ∘ f) = rectangleCodes p cell labels :=
  rectangleCodes_congr (hf.range_comp fun t => (cell t, labels t))

/-! The witness of `Basic`: a part of its samples, their reverse order, and two passes. -/

/-- The first eight samples of the witness allow at least the codings of all twelve. -/
example : rectangleCodes 5 fiveCells fiveLabels ≤
    rectangleCodes 5 (fiveCells ∘ Fin.castLE (by norm_num : 8 ≤ 12))
      (fiveLabels ∘ Fin.castLE (by norm_num : 8 ≤ 12)) :=
  rectangleCodes_anti (Set.range_comp_subset_range _ fun t => (fiveCells t, fiveLabels t))

/-- The samples in the reverse order allow the same codings. -/
example : rectangleCodes 5 (fiveCells ∘ Fin.rev) (fiveLabels ∘ Fin.rev) =
      rectangleCodes 5 fiveCells fiveLabels ∧
    pinnedCodes 5 (fiveCells ∘ Fin.rev) (fiveLabels ∘ Fin.rev) 0 =
      pinnedCodes 5 fiveCells fiveLabels 0 :=
  have h := Fin.rev_surjective.range_comp fun t => (fiveCells t, fiveLabels t)
  ⟨rectangleCodes_congr h, pinnedCodes_congr 0 h⟩

/-- Two passes over the samples, the second in the reverse order, allow the same codings. -/
example : rectangleCodes 5
    (fiveCells ∘ fun t : Bool × Fin 12 => if t.1 then t.2 else Fin.rev t.2)
    (fiveLabels ∘ fun t : Bool × Fin 12 => if t.1 then t.2 else Fin.rev t.2) =
      rectangleCodes 5 fiveCells fiveLabels :=
  rectangleCodes_comp_surjective fiveCells fiveLabels fun t => ⟨(true, t), rfl⟩

end Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery
