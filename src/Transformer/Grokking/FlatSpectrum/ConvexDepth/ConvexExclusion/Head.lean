module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ConvexExclusion.Limits
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ComplexHeadBudget

/-!
# Actual complex-head guarantees after candidate exclusion

Source: `review/convex/moments.md`, CM49. Geometry first identifies a query's
raw label. A fixed positive mixture of observed sources bearing that label
then transfers actual margins and ordinary full-vocabulary CE. Its input
barycenter need not be the query: the identifying and prediction certificates
have different jobs. Only the complete output head is trained; the pool,
geometry, source labels and feature mixtures are fixed.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ConvexExclusion

open RawRegions
open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {E K I X C : Type*} [AddCommGroup E] [Module ℝ E]
  [Fintype K] [Fintype I]

/-- Excluding alternative admissible labels replaces the unknown-target
premise of CM42's margin transfer, CM49. No query-coordinate interpolation
hypothesis is needed for the positive prediction weights. -/
theorem certified_margin (point : I → E) (labels : I → C) (target : E → C)
    (hconv : ConvexClasses target) (hobs : ∀ i, target (point i) = labels i)
    (allowed : Set C) (query : X → E) (w : X → I → ℝ) (source : K → I → ℝ)
    (W : K → C → ℝ) (x : X) (c d : C) (m : ℝ)
    (hpool : target (query x) ∈ allowed)
    (hex : ∀ b ∈ allowed, b ≠ c → Excluded point labels (query x) b)
    (hw : ∀ i, 0 ≤ w x i) (hs : ∑ i, w x i = 1)
    (hlabels : ∀ i, w x i ≠ 0 → labels i = c)
    (hmargin : ∀ i, w x i ≠ 0 →
      m ≤ headLogits source W i (labels i) - headLogits source W i d) :
    m ≤ headLogits (ContextInterpolation.features w source) W x (target (query x)) -
      headLogits (ContextInterpolation.features w source) W x d := by
  have hy := target_eq_of_exclusions point labels target hconv hobs allowed hpool hex
  rw [hy]
  apply ContextInterpolation.margin_le w source W x c d hw hs
  intro i hi
  have h := hmargin i hi
  rwa [hlabels i hi] at h

/-- Ordinary observed raw CE bounds the identified query's ordinary CE,
CM49. Every class stays in the softmax and the missing label is a conclusion. -/
theorem certified_ce [Fintype C]
    (point : I → E) (labels : I → C) (target : E → C)
    (hconv : ConvexClasses target) (hobs : ∀ i, target (point i) = labels i)
    (allowed : Set C) (query : X → E) (w : X → I → ℝ) (source : K → I → ℝ)
    (W : K → C → ℝ) (x : X) (c : C) (ε : ℝ)
    (hpool : target (query x) ∈ allowed)
    (hex : ∀ b ∈ allowed, b ≠ c → Excluded point labels (query x) b)
    (hw : ∀ i, 0 ≤ w x i) (hs : ∑ i, w x i = 1)
    (hlabels : ∀ i, w x i ≠ 0 → labels i = c)
    (hloss : ∀ i, w x i ≠ 0 → crossEntropy (headLogits source W i) (labels i) ≤ ε) :
    crossEntropy (headLogits (ContextInterpolation.features w source) W x)
      (target (query x)) ≤ ε := by
  have hy := target_eq_of_exclusions point labels target hconv hobs allowed hpool hex
  rw [hy]
  apply ContextInterpolation.ce_le_of_local w source W x c hw hs
  intro i hi
  have h := hloss i hi
  rwa [hlabels i hi] at h

omit [Fintype K] in
/-- The actual complete unitary initializer gives exact logits for the
geometrically identified answer, CM49. This algebraic identity only needs
unit mass; nonnegativity is separately required for the feature bound. -/
theorem certified_initializer_logits [DecidableEq I] [DecidableEq C]
    (point : I → E) (labels : I → C) (target : E → C)
    (hconv : ConvexClasses target) (hobs : ∀ i, target (point i) = labels i)
    (allowed : Set C) (query : X → E) (w : X → I → ℝ)
    (U : Matrix I I ℂ) (hU : U ∈ Matrix.unitaryGroup I ℂ) (x : X) (c : C) (a : ℝ)
    (hpool : target (query x) ∈ allowed)
    (hex : ∀ b ∈ allowed, b ≠ c → Excluded point labels (query x) b)
    (hs : ∑ i, w x i = 1) (hlabels : ∀ i, w x i ≠ 0 → labels i = c) :
    headLogits (ContextInterpolation.features w (complexFeatures U))
      (complexCoefficients (complexInitializer U labels a)) x =
      fun d => if d = target (query x) then a else 0 := by
  have hy := target_eq_of_exclusions point labels target hconv hobs allowed hpool hex
  rw [hy]
  exact same_label_logits U hU labels w x c a hs hlabels

omit [Fintype K] in
/-- The identified answer's exact ordinary CE uses the full raw vocabulary,
CM49. A many-to-one decoder does not pool probabilities in this loss. -/
theorem certified_initializer_ce [Fintype C] [DecidableEq I] [DecidableEq C]
    (point : I → E) (labels : I → C) (target : E → C)
    (hconv : ConvexClasses target) (hobs : ∀ i, target (point i) = labels i)
    (allowed : Set C) (query : X → E) (w : X → I → ℝ)
    (U : Matrix I I ℂ) (hU : U ∈ Matrix.unitaryGroup I ℂ) (x : X) (c : C) (a : ℝ)
    (hpool : target (query x) ∈ allowed)
    (hex : ∀ b ∈ allowed, b ≠ c → Excluded point labels (query x) b)
    (hs : ∑ i, w x i = 1) (hlabels : ∀ i, w x i ≠ 0 → labels i = c) :
    crossEntropy (headLogits (ContextInterpolation.features w (complexFeatures U))
      (complexCoefficients (complexInitializer U labels a)) x) (target (query x)) =
      Real.log (1 + ((Fintype.card C : ℝ) - 1) * Real.exp (-a)) := by
  rw [certified_initializer_logits point labels target hconv hobs allowed query w U hU
    x c a hpool hex hs hlabels]
  exact fixed_margin_ce (target (query x)) a

omit [Fintype K] in
/-- A single complete-head realization satisfies identification, bounded
source features and the full coefficient budget together, CM49. Geometry
does not supply an extra trainable decoder or uncharged output coefficients. -/
theorem certified_initializer [Fintype C] [DecidableEq I] [DecidableEq C]
    (point : I → E) (labels : I → C) (target : E → C)
    (hconv : ConvexClasses target) (hobs : ∀ i, target (point i) = labels i)
    (allowed : Set C) (query : X → E) (w : X → I → ℝ)
    (U : Matrix I I ℂ) (hU : U ∈ Matrix.unitaryGroup I ℂ) (x : X) (c : C) (a : ℝ)
    (hpool : target (query x) ∈ allowed)
    (hex : ∀ b ∈ allowed, b ≠ c → Excluded point labels (query x) b)
    (hw : ∀ i, 0 ≤ w x i) (hs : ∑ i, w x i = 1)
    (hlabels : ∀ i, w x i ≠ 0 → labels i = c) :
    headLogits (ContextInterpolation.features w (complexFeatures U))
        (complexCoefficients (complexInitializer U labels a)) x =
        (fun d => if d = target (query x) then a else 0) ∧
      (∀ k, |ContextInterpolation.features w (complexFeatures U) k x| ≤ 1) ∧
      complexCoefficients (complexInitializer U labels a) ∈
        headBudget (2 * |a| * (Fintype.card I : ℝ)) := by
  refine ⟨?_, ?_, ?_⟩
  · exact certified_initializer_logits point labels target hconv hobs allowed query w U hU
      x c a hpool hex hs hlabels
  · exact ContextInterpolation.complex_features_bound U hU w x hw hs
  · exact complexInitializer_budget U hU labels a

omit [AddCommGroup E] [Module ℝ E] [Fintype K] in
/-- Positive feature mixtures preserve both the source-coordinate bound
and the actual complete-head logit bound, CM49. Input extrapolation does
not require extrapolating features with signed or excessive weights. -/
theorem positive_complex_bounds [DecidableEq I]
    (U : Matrix I I ℂ) (hU : U ∈ Matrix.unitaryGroup I ℂ)
    (w : X → I → ℝ) (x : X) (hw : ∀ i, 0 ≤ w x i) (hs : ∑ i, w x i = 1)
    (W : I × Bool → C → ℝ) (R : ℝ) (hW : W ∈ headBudget R) :
    (∀ k, |ContextInterpolation.features w (complexFeatures U) k x| ≤ 1) ∧
      (∀ c, |headLogits (ContextInterpolation.features w (complexFeatures U)) W x c| ≤ R) := by
  constructor
  · exact ContextInterpolation.complex_features_bound U hU w x hw hs
  · intro c
    simpa only [mul_one] using ContextInterpolation.logits_bound w (complexFeatures U)
      x hw hs (by norm_num : (0 : ℝ) ≤ 1)
      (fun k i => complexFeatures_bound U hU k i) hW c

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ConvexExclusion
