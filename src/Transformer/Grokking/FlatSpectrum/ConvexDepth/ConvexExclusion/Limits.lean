module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ConvexExclusion.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.RawRegions.Head

/-!
# Exterior identification and the limits of candidate closure

Source: `review/flat_spectrum.md`, CM49. Observable conflicts identify two
queries outside every observed hull under a closed candidate pool. A new
convex class breaks identification without closure; an interior gap remains
ambiguous even with closure. These are target-class hypotheses, not spectral
consequences or claims that arbitrary observed affine spans extrapolate.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ConvexExclusion

open RawRegions

/-- Both exterior queries are outside every observed raw-class hull, CM49.
Thus the exclusion examples strictly extend CM48's direct hull coverage. -/
theorem exterior_uncovered (c : Fin 4) :
    (-4 : ℝ) ∉ convexHull ℝ (observedClass regionPoints regionLabels c) ∧
      (4 : ℝ) ∉ convexHull ℝ (observedClass regionPoints regionLabels c) := by
  have hs : observedClass regionPoints regionLabels c ⊆ Set.Icc (-3 : ℝ) 3 := by
    rintro z ⟨i, _, rfl⟩
    fin_cases i <;> norm_num [regionPoints]
  have hh := convexHull_min hs (convex_Icc (-3 : ℝ) 3 : Convex ℝ (Set.Icc (-3 : ℝ) 3))
  constructor
  · intro hx
    have hb := (hh hx).1
    norm_num at hb
  · intro hx
    have hb := (hh hx).2
    norm_num at hb

/-- The same observable certificates identify both exterior answers for
any compatible target whose two query labels belong to the declared pool. -/
theorem exterior_identification (target : ℝ → Fin 4)
    (hconv : ConvexClasses target) (hobs : ∀ i, target (regionPoints i) = regionLabels i)
    (hl : target (-4) ∈ rawPool) (hr : target 4 ∈ rawPool) :
    target (-4) = 0 ∧ target 4 = 2 := by
  constructor
  · exact target_eq_of_exclusions regionPoints regionLabels target hconv hobs rawPool
      hl left_exclusions
  · exact target_eq_of_exclusions regionPoints regionLabels target hconv hobs rawPool
      hr right_exclusions

/-- A target with an additional unobserved raw class, CM49. Its three nonempty
regions are intervals; observations cannot exclude its new exterior class. -/
noncomputable def freshTarget (x : ℝ) : Fin 4 :=
  if x < -(7 / 2 : ℝ) then 1 else cutTarget 0 x

/-- The counterexample meets exactly the same convex-region assumption. -/
theorem freshTarget_convex : ConvexClasses freshTarget := by
  intro c
  fin_cases c
  · convert (convex_Icc (-(7 / 2 : ℝ)) 0 : Convex ℝ (Set.Icc (-(7 / 2 : ℝ)) 0)) using 1
    ext x
    by_cases hx : x < -(7 / 2 : ℝ) <;> by_cases hz : x ≤ 0 <;>
      simp [freshTarget, cutTarget, hx, hz]
    linarith
  · convert (convex_Iio (-(7 / 2 : ℝ)) : Convex ℝ (Set.Iio (-(7 / 2 : ℝ)))) using 1
    ext x
    by_cases hx : x < -(7 / 2 : ℝ) <;> by_cases hz : x ≤ 0 <;>
      simp [freshTarget, cutTarget, hx, hz]
  · convert (convex_Ioi (0 : ℝ) : Convex ℝ (Set.Ioi (0 : ℝ))) using 1
    ext x
    by_cases hx : x < -(7 / 2 : ℝ) <;> by_cases hz : x ≤ 0 <;>
      simp [freshTarget, cutTarget, hx, hz] <;> linarith
  · convert (convex_empty : Convex ℝ (∅ : Set ℝ)) using 1
    ext x
    by_cases hx : x < -(7 / 2 : ℝ) <;> by_cases hz : x ≤ 0 <;>
      simp [freshTarget, cutTarget, hx, hz]

/-- Every original observation has the same raw answer under the new target. -/
theorem freshTarget_observed (i : Fin 4) : freshTarget (regionPoints i) = regionLabels i := by
  fin_cases i <;> norm_num [freshTarget, cutTarget, regionPoints, regionLabels]

/-- The alternative exterior answer was never an observed raw label,
CM49. It belongs to the original four-class vocabulary; no new output
coordinate or vocabulary extension is involved. Closure to observed labels
would exclude a compatible target. -/
theorem fresh_label_unobserved (i : Fin 4) : regionLabels i ≠ freshTarget (-4) := by
  fin_cases i <;> norm_num [regionLabels, freshTarget]

/-- Without candidate closure the same observations and convexity allow a
different answer at the otherwise identified exterior query, CM49. -/
theorem missing_pool_counterexample :
    ConvexClasses freshTarget ∧
      (∀ i, freshTarget (regionPoints i) = regionLabels i) ∧
      freshTarget (-4) ≠ cutTarget 0 (-4) ∧ freshTarget (-4) ∉ rawPool := by
  refine ⟨freshTarget_convex, freshTarget_observed, ?_, ?_⟩ <;>
    norm_num [freshTarget, cutTarget, rawPool]

/-- No observable exclusion certificate can rule out the additional class:
it is realized by a compatible convex target. -/
theorem fresh_label_not_excluded : ¬ Excluded regionPoints regionLabels (-4) 1 := by
  intro hex
  have hne := target_ne_of_excluded regionPoints regionLabels freshTarget
    freshTarget_convex freshTarget_observed hex
  norm_num [freshTarget] at hne

/-- The fresh-class and two-class targets give identical ordinary observed
CE for every source and complete head tensor, CM49. The missing answer is
therefore not recoverable from that training objective alone. -/
theorem fresh_observed_ce_indistinguishable {K : Type*} [Fintype K]
    (source : K → Fin 4 → ℝ) (W : K → Fin 4 → ℝ) :
    headLoss source id (fun i => freshTarget (regionPoints i)) W =
      headLoss source id (fun i => cutTarget 0 (regionPoints i)) W := by
  have heq : (fun i => freshTarget (regionPoints i)) =
      (fun i => cutTarget 0 (regionPoints i)) := by
    funext i
    exact (freshTarget_observed i).trans (cutTargets_observed i).1.symm
  rw [heq]

/-- Both possible labels at the interior gap belong to the closed pool,
yet compatible convex targets disagree there, CM49. -/
theorem closed_pool_ambiguity :
    ConvexClasses (cutTarget 0) ∧ ConvexClasses (cutTarget (1 / 2)) ∧
      (∀ i, cutTarget 0 (regionPoints i) = regionLabels i ∧
        cutTarget (1 / 2) (regionPoints i) = regionLabels i) ∧
      cutTarget 0 (1 / 4) ∈ rawPool ∧ cutTarget (1 / 2) (1 / 4) ∈ rawPool ∧
      cutTarget 0 (1 / 4) ≠ cutTarget (1 / 2) (1 / 4) := by
  exact ⟨cutTarget_convex 0, cutTarget_convex (1 / 2), cutTargets_observed,
    cut_in_pool 0 (1 / 4), cut_in_pool (1 / 2) (1 / 4), uncovered_ambiguity.2.2.2.2⟩

/-- Neither possible label has an exclusion certificate at the ambiguous gap.
The rule consequently refuses to identify this query rather than guessing. -/
theorem gap_labels_not_excluded :
    ¬ Excluded regionPoints regionLabels (1 / 4) 0 ∧
      ¬ Excluded regionPoints regionLabels (1 / 4) 2 := by
  constructor
  · intro hex
    have hne := target_ne_of_excluded regionPoints regionLabels (cutTarget (1 / 2))
      (cutTarget_convex (1 / 2)) (fun i => (cutTargets_observed i).2) hex
    norm_num [cutTarget] at hne
  · intro hex
    have hne := target_ne_of_excluded regionPoints regionLabels (cutTarget 0)
      (cutTarget_convex 0) (fun i => (cutTargets_observed i).1) hex
    norm_num [cutTarget] at hne

example := exterior_identification (cutTarget 0) (cutTarget_convex 0)
  (fun i => (cutTargets_observed i).1) (cut_in_pool 0 (-4)) (cut_in_pool 0 4)
example := exterior_identification (cutTarget (1 / 2)) (cutTarget_convex (1 / 2))
  (fun i => (cutTargets_observed i).2) (cut_in_pool (1 / 2) (-4)) (cut_in_pool (1 / 2) 4)

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ConvexExclusion
