module

public import Transformer.Grokking.Convex.ConvexDepth.ContextInterpolation.Loss
public import Mathlib.Analysis.Convex.Combination

/-!
# Raw labels identified by observed convex hulls

Source: `review/convex/moments.md`, CM48. Convexity of each unknown raw-class
region is an additional structural hypothesis, independent of flat spectrum.
The hull certificate uses only observed coordinates and their raw labels.
The statements can be applied separately inside each fixed latent block.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.RawRegions

open scoped BigOperators

variable {E I C : Type*} [AddCommGroup E] [Module ℝ E]

/-- The observed input points bearing one raw label, CM48. -/
def observedClass (point : I → E) (labels : I → C) (c : C) : Set E :=
  point '' {i | labels i = c}

/-- Convex raw-class regions are a hypothesis about the unknown target,
CM48, not a condition inferred from a flat code or observed loss. -/
def ConvexClasses (target : E → C) : Prop :=
  ∀ c, Convex ℝ {x | target x = c}

/-- Every compatible target assigns the observed label throughout its
observed class hull, CM48. The query label is a conclusion. -/
theorem label_of_hull (point : I → E) (labels : I → C) (target : E → C)
    (hconv : ConvexClasses target) (hobs : ∀ i, target (point i) = labels i)
    {x : E} {c : C} (hx : x ∈ convexHull ℝ (observedClass point labels c)) :
    target x = c := by
  have hsubset : observedClass point labels c ⊆ {z | target z = c} := by
    rintro z ⟨i, hi, rfl⟩
    exact (hobs i).trans hi
  exact convexHull_min hsubset (hconv c) hx

/-- All convex-region targets fitting the same observations agree on a
covered query, CM48. No selected learner or unknown answer is a hypothesis. -/
theorem compatible_targets_agree (point : I → E) (labels : I → C)
    (f g : E → C) (hf : ConvexClasses f) (hg : ConvexClasses g)
    (hfobs : ∀ i, f (point i) = labels i) (hgobs : ∀ i, g (point i) = labels i)
    {x : E} {c : C} (hx : x ∈ convexHull ℝ (observedClass point labels c)) :
    f x = g x := by
  have hfx := label_of_hull point labels f hf hfobs hx
  have hgx := label_of_hull point labels g hg hgobs hx
  exact hfx.trans hgx.symm

/-- Overlapping observed hulls cannot carry distinct raw labels under
the structural hypothesis, CM48. Such an overlap refutes realizability. -/
theorem labels_eq_of_hull_overlap (point : I → E) (labels : I → C)
    (target : E → C) (hconv : ConvexClasses target)
    (hobs : ∀ i, target (point i) = labels i) {x : E} {c d : C}
    (hc : x ∈ convexHull ℝ (observedClass point labels c))
    (hd : x ∈ convexHull ℝ (observedClass point labels d)) : c = d := by
  have hcx := label_of_hull point labels target hconv hobs hc
  have hdx := label_of_hull point labels target hconv hobs hd
  exact hcx.symm.trans hdx

/-- A finite barycentric certificate uses only source coordinates, observed
labels and nonnegative weights, CM48. Zero-weight sources may have any label. -/
theorem barycenter_mem_hull [Fintype I] (point : I → E) (labels : I → C)
    (w : I → ℝ) {c : C} (hw : ∀ i, 0 ≤ w i) (hs : ∑ i, w i = 1)
    (hlabels : ∀ i, w i ≠ 0 → labels i = c) :
    (∑ i, w i • point i) ∈ convexHull ℝ (observedClass point labels c) := by
  have hsum : (∑ᶠ i, w i) = 1 := by
    simpa only [finsum_eq_sum_of_fintype] using hs
  have hmem : ∀ i, w i ≠ 0 → point i ∈ convexHull ℝ (observedClass point labels c) := by
    intro i hi
    apply subset_convexHull
    exact ⟨i, hlabels i hi, rfl⟩
  simpa only [finsum_eq_sum_of_fintype] using
    (convex_convexHull ℝ (observedClass point labels c)).finsum_mem hw hsum hmem

/-- The same finite weights can certify the missing target and construct
its context feature row, CM48. Convex regions remain an explicit premise. -/
theorem label_of_barycenter [Fintype I] (point : I → E) (labels : I → C)
    (target : E → C) (hconv : ConvexClasses target)
    (hobs : ∀ i, target (point i) = labels i) (w : I → ℝ) {x : E} {c : C}
    (hw : ∀ i, 0 ≤ w i) (hs : ∑ i, w i = 1)
    (hlabels : ∀ i, w i ≠ 0 → labels i = c)
    (hx : ∑ i, w i • point i = x) : target x = c := by
  have hmem := barycenter_mem_hull point labels w hw hs hlabels
  rw [hx] at hmem
  exact label_of_hull point labels target hconv hobs hmem

/-- A nonconstant family of compatible raw targets for CM48's witnesses. -/
noncomputable def cutTarget (t x : ℝ) : Fin 4 := if x ≤ t then 0 else 2

/-- Every member of the threshold family has convex raw-class regions. -/
theorem cutTarget_convex (t : ℝ) : ConvexClasses (cutTarget t) := by
  intro c
  fin_cases c
  · convert (convex_Iic t : Convex ℝ (Set.Iic t)) using 1
    ext x
    simp [cutTarget]
  · convert (convex_empty : Convex ℝ (∅ : Set ℝ)) using 1
    ext x
    by_cases hx : x ≤ t <;> simp [cutTarget, hx]
  · convert (convex_Ioi t : Convex ℝ (Set.Ioi t)) using 1
    ext x
    simp [cutTarget]
  · convert (convex_empty : Convex ℝ (∅ : Set ℝ)) using 1
    ext x
    by_cases hx : x ≤ t <;> simp [cutTarget, hx]

/-- Four observed inputs and their two raw aliases, CM48. -/
def regionPoints : Fin 4 → ℝ := ![-3, -1, 1, 3]
def regionLabels : Fin 4 → Fin 4 := ![0, 0, 2, 2]

/-- Both targets fit all observations, although their unseen labels can differ. -/
theorem cutTargets_observed (i : Fin 4) :
    cutTarget 0 (regionPoints i) = regionLabels i ∧
      cutTarget (1 / 2) (regionPoints i) = regionLabels i := by
  fin_cases i <;> norm_num [cutTarget, regionPoints, regionLabels]

/-- The unseen point -2 has a nontrivial same-label barycentric certificate. -/
noncomputable def leftWeights : Fin 4 → ℝ := ![1 / 2, 1 / 2, 0, 0]

theorem left_certificate :
    (∀ i, 0 ≤ leftWeights i) ∧ (∑ i, leftWeights i = 1) ∧
      (∀ i, leftWeights i ≠ 0 → regionLabels i = 0) ∧
      (∑ i, leftWeights i • regionPoints i = -2) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro i
    fin_cases i <;> norm_num [leftWeights]
  · norm_num [leftWeights, Fin.sum_univ_four]
  · intro i hi
    fin_cases i <;> norm_num [leftWeights, regionLabels] at *
  · norm_num [leftWeights, regionPoints, Fin.sum_univ_four]

/-- The certificate covers a genuinely unobserved input, CM48. -/
theorem left_covered_unseen :
    (-2 : ℝ) ∈ convexHull ℝ (observedClass regionPoints regionLabels 0) ∧
      (-2 : ℝ) ∉ Set.range regionPoints := by
  constructor
  · have h := barycenter_mem_hull regionPoints regionLabels leftWeights
      left_certificate.1 left_certificate.2.1 left_certificate.2.2.1
    rwa [left_certificate.2.2.2] at h
  · rintro ⟨i, hi⟩
    fin_cases i <;> norm_num [regionPoints] at hi

-- Every conditional geometric theorem has a nonconstant, held-out instance.
example := label_of_hull regionPoints regionLabels (cutTarget 0)
  (cutTarget_convex 0) (fun i => (cutTargets_observed i).1) left_covered_unseen.1
example := compatible_targets_agree regionPoints regionLabels (cutTarget 0) (cutTarget (1 / 2))
  (cutTarget_convex 0) (cutTarget_convex (1 / 2))
  (fun i => (cutTargets_observed i).1) (fun i => (cutTargets_observed i).2) left_covered_unseen.1
example := labels_eq_of_hull_overlap regionPoints regionLabels (cutTarget 0)
  (cutTarget_convex 0) (fun i => (cutTargets_observed i).1)
  left_covered_unseen.1 left_covered_unseen.1
example := barycenter_mem_hull regionPoints regionLabels leftWeights
  left_certificate.1 left_certificate.2.1 left_certificate.2.2.1
example := label_of_barycenter regionPoints regionLabels (cutTarget 0)
  (cutTarget_convex 0) (fun i => (cutTargets_observed i).1) leftWeights
  left_certificate.1 left_certificate.2.1 left_certificate.2.2.1 left_certificate.2.2.2

end Transformer.Grokking.Convex.ConvexDepth.RawRegions
