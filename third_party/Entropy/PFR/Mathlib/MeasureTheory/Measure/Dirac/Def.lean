module

public import Mathlib.MeasureTheory.Measure.Dirac.Def

/-!
# Dirac probabilities

Real-valued evaluations of point-mass measures.
-/

public section

namespace MeasureTheory.Measure
variable {α : Type*} [MeasurableSpace α] {s : Set α} {a : α}

@[simp]
lemma dirac_real_apply' (a : α) (hs : MeasurableSet s) : (dirac a).real s = s.indicator 1 a := by
  by_cases ha : a ∈ s <;> simp [Measure.real, *]

end MeasureTheory.Measure
