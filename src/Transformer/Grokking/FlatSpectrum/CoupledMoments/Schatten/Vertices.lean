module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Basic
public import Mathlib.Analysis.Convex.Extreme

/-!
# Convexification cannot create a missing deterministic table

Source: `review/flat_spectrum.md`, CM9. A one-hot table is an extreme point of the
product of probability simplexes. If every member of a family is probability-valued,
its convex hull contains that table exactly when the original family does. This
separates adding genuinely new spectral budgets from only mixing old candidates.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

open Finset

variable {m n C : Type*} [Fintype C] [DecidableEq C]

/-- A deterministic table is an extreme probability-valued prediction (CM9). -/
theorem pointMass_extreme (labels : m → n → C) :
    pointMass labels ∈ Set.extremePoints ℝ {P : Table m n C | Rows P} := by
  refine mem_extremePoints_iff_left.mpr ⟨rows_pointMass labels, ?_⟩
  intro P hP Q hQ hsegment
  obtain ⟨a, b, ha, hb, _, heq⟩ := hsegment
  have hz (i : m) (j : n) (c : C) (hc : c ≠ labels i j) : P i j c = 0 := by
    have he := congrFun (congrFun (congrFun heq i) j) c
    change a * P i j c + b * Q i j c = pointMass labels i j c at he
    simp only [pointMass, ite_eq_right hc] at he
    nlinarith [hP.1 i j c, mul_nonneg hb.le (hQ.1 i j c)]
  ext i j c
  by_cases hc : c = labels i j
  · subst c
    have hs := hP.2 i j
    rw [sum_eq_single (labels i j) (fun d _ hd => hz i j d hd) (by simp)] at hs
    simpa [pointMass] using hs
  · simp [pointMass, hc, hz i j c hc]

/-- A family of probability predictions cannot acquire a previously excluded one-hot
table merely by taking its convex hull (CM9). -/
theorem pointMass_mem_convexHull_iff (labels : m → n → C) (A : Set (Table m n C))
    (hA : ∀ P ∈ A, Rows P) : pointMass labels ∈ convexHull ℝ A ↔ pointMass labels ∈ A := by
  constructor
  · intro h
    have hsub : convexHull ℝ A ⊆ {P | Rows P} :=
      convexHull_min hA rows_convex
    have he := inter_extremePoints_subset_extremePoints_of_subset hsub
      ⟨h, pointMass_extreme labels⟩
    exact extremePoints_convexHull_subset he
  · intro h
    exact subset_convexHull ℝ A h

example := pointMass_mem_convexHull_iff (fun i j : Bool => i && j)
  {P | Rows P} (fun _ h => h)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
