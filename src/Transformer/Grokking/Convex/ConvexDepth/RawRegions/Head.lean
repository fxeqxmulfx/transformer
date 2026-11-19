module

public import Transformer.Grokking.Convex.ConvexDepth.RawRegions.Basic

/-!
# Geometry certificates in the actual complex head

Source: `review/convex/moments.md`, CM48. Observed raw labels and an input
barycentric certificate replace CM42's assumption about an unknown query
label. Convex target regions remain a separate structural hypothesis. The
interpolation weights stay fixed during ordinary complete-head CE training.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.RawRegions

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {E K I X C : Type*} [AddCommGroup E] [Module ℝ E]
  [Fintype K] [Fintype I]

/-- A same-label input-hull certificate transfers observed source margins
to the unknown query target, CM48, using actual finite head products. -/
theorem certified_margin (point : I → E) (labels : I → C) (target : E → C)
    (hconv : ConvexClasses target) (hobs : ∀ i, target (point i) = labels i)
    (query : X → E) (w : X → I → ℝ) (source : K → I → ℝ)
    (W : K → C → ℝ) (x : X) (c d : C) (m : ℝ)
    (hw : ∀ i, 0 ≤ w x i) (hs : ∑ i, w x i = 1)
    (hlabels : ∀ i, w x i ≠ 0 → labels i = c)
    (hx : ∑ i, w x i • point i = query x)
    (hmargin : ∀ i, w x i ≠ 0 →
      m ≤ headLogits source W i (labels i) - headLogits source W i d) :
    m ≤ headLogits (ContextInterpolation.features w source) W x (target (query x)) -
      headLogits (ContextInterpolation.features w source) W x d := by
  have hy := label_of_barycenter point labels target hconv hobs (w x) hw hs hlabels hx
  rw [hy]
  apply ContextInterpolation.margin_le w source W x c d hw hs
  intro i hi
  have h := hmargin i hi
  rwa [hlabels i hi] at h

/-- Low ordinary observed raw CE transfers to a hull-certified query,
CM48. All classes remain in the softmax; no missing label is supplied. -/
theorem certified_ce [Fintype C]
    (point : I → E) (labels : I → C) (target : E → C)
    (hconv : ConvexClasses target) (hobs : ∀ i, target (point i) = labels i)
    (query : X → E) (w : X → I → ℝ) (source : K → I → ℝ)
    (W : K → C → ℝ) (x : X) (c : C) (ε : ℝ)
    (hw : ∀ i, 0 ≤ w x i) (hs : ∑ i, w x i = 1)
    (hlabels : ∀ i, w x i ≠ 0 → labels i = c)
    (hx : ∑ i, w x i • point i = query x)
    (hloss : ∀ i, w x i ≠ 0 → crossEntropy (headLogits source W i) (labels i) ≤ ε) :
    crossEntropy (headLogits (ContextInterpolation.features w source) W x)
      (target (query x)) ≤ ε := by
  have hy := label_of_barycenter point labels target hconv hobs (w x) hw hs hlabels hx
  rw [hy]
  apply ContextInterpolation.ce_le_of_local w source W x c hw hs
  intro i hi
  have h := hloss i hi
  rwa [hlabels i hi] at h

omit [AddCommGroup E] [Module ℝ E] [Fintype K] in
/-- Same-label source weights yield exact raw margin logits in the complete
unitary head, CM48. This algebraic identity does not require nonnegative weights;
the geometric certificate and feature bound do require them. -/
theorem same_label_logits [DecidableEq I] [DecidableEq C]
    (U : Matrix I I ℂ) (hU : U ∈ Matrix.unitaryGroup I ℂ)
    (labels : I → C) (w : X → I → ℝ) (x : X) (c : C) (a : ℝ)
    (hs : ∑ i, w x i = 1) (hlabels : ∀ i, w x i ≠ 0 → labels i = c) :
    headLogits (ContextInterpolation.features w (complexFeatures U))
      (complexCoefficients (complexInitializer U labels a)) x =
      fun d => if d = c then a else 0 := by
  funext d
  rw [ContextInterpolation.initializer_logits U hU]
  unfold ContextInterpolation.labelMass
  by_cases hd : d = c
  · subst d
    have hsum : (∑ i, if labels i = c then w x i else 0) = 1 := by
      rw [← hs]
      apply Finset.sum_congr rfl
      intro i _
      by_cases hi : w x i = 0
      · simp [hi]
      · simp [hlabels i hi]
    rw [hsum]
    simp
  · have hsum : (∑ i, if labels i = d then w x i else 0) = 0 := by
      apply Finset.sum_eq_zero
      intro i _
      by_cases hi : w x i = 0
      · simp [hi]
      · simp [hlabels i hi, Ne.symm hd]
    rw [hsum]
    simp [hd]

/-- The same observations admit two convex-region targets with different
raw labels at one quarter, CM48. That query belongs to no observed class hull. -/
theorem uncovered_ambiguity :
    ConvexClasses (cutTarget 0) ∧ ConvexClasses (cutTarget (1 / 2)) ∧
      (∀ i, cutTarget 0 (regionPoints i) = regionLabels i ∧
        cutTarget (1 / 2) (regionPoints i) = regionLabels i) ∧
      (∀ c, (1 / 4 : ℝ) ∉ convexHull ℝ (observedClass regionPoints regionLabels c)) ∧
      cutTarget 0 (1 / 4) ≠ cutTarget (1 / 2) (1 / 4) := by
  have hneq : cutTarget 0 (1 / 4) ≠ cutTarget (1 / 2) (1 / 4) := by
    norm_num [cutTarget]
  refine ⟨cutTarget_convex 0, cutTarget_convex (1 / 2), cutTargets_observed, ?_, hneq⟩
  intro c hc
  have heq := compatible_targets_agree regionPoints regionLabels
    (cutTarget 0) (cutTarget (1 / 2)) (cutTarget_convex 0) (cutTarget_convex (1 / 2))
    (fun i => (cutTargets_observed i).1) (fun i => (cutTargets_observed i).2) hc
  exact hneq heq

/-- Any fixed prediction at the uncovered query is wrong for at least one
compatible target, CM48. No deterministic training-only choice resolves both. -/
theorem uncovered_prediction_wrong (predict : ℝ → Fin 4) :
    predict (1 / 4) ≠ cutTarget 0 (1 / 4) ∨
      predict (1 / 4) ≠ cutTarget (1 / 2) (1 / 4) := by
  by_cases h : predict (1 / 4) = cutTarget 0 (1 / 4)
  · right
    rw [h]
    exact uncovered_ambiguity.2.2.2.2
  · exact Or.inl h

/-- All original raw CE objectives coincide for the two possible targets,
for every complete coefficient tensor, CM48. Observed fit cannot choose one. -/
theorem observed_ce_indistinguishable {J : Type*} [Fintype J]
    (source : J → Fin 4 → ℝ) (W : J → Fin 4 → ℝ) :
    headLoss source id (fun i => cutTarget 0 (regionPoints i)) W =
      headLoss source id (fun i => cutTarget (1 / 2) (regionPoints i)) W := by
  have hlabels : (fun i => cutTarget 0 (regionPoints i)) =
      (fun i => cutTarget (1 / 2) (regionPoints i)) := by
    funext i
    exact (cutTargets_observed i).1.trans (cutTargets_observed i).2.symm
  rw [hlabels]

end Transformer.Grokking.Convex.ConvexDepth.RawRegions
