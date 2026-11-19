module

public import Transformer.Grokking.Convex.ConvexDepth.ClassGeometry.Head

/-!
# A nonconstant unseen-class witness and the role of its prior

Source: `review/convex/moments.md`, CM56. Training observes raw labels 1 and 2;
the heldout input requires label 3. Two distinct feasible margins share that
prediction through the actual complete complex head. Removing the affine
decoder prior admits a target agreeing at both anchors but requiring label 4.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.ClassGeometry

open scoped BigOperators

-- Nonconstant examples use labels 1/2 as anchors and leave raw label 3 unseen.
noncomputable def exampleCoordinates (c : Fin 5) : ℝ := (c.val : ℝ) / 4
noncomputable def exampleParameters : Fin 3 → ℝ := ![2, 1, 4]

example := fitted_anchors exampleCoordinates 0 (1 / 2) 1 2 4 (by norm_num)

example : exampleParameters 0 = exampleParameters 2 * (1 / 2) ∧
    exampleParameters 1 = exampleParameters 2 * (1 / 4) := by
  apply anchors_identify exampleCoordinates exampleParameters 0 (1 / 2) (1 / 2) (1 / 4) 1 2
  all_goals norm_num [exampleCoordinates, exampleParameters]

example : score exampleCoordinates exampleParameters 1 3 -
    score exampleCoordinates exampleParameters 1 2 =
      exampleParameters 2 * (exampleCoordinates 3 - exampleCoordinates 2) ^ 2 := by
  apply score_difference
  norm_num [exampleParameters, exampleCoordinates]

noncomputable def exampleFeature : Fin 3 → ℝ := ![0, 1 / 2, 1]
def exampleTarget : Fin 3 → Fin 5 := ![1, 2, 3]
def alternativeTarget : Fin 3 → Fin 5 := ![1, 2, 4]
noncomputable def halfParameters : Fin 3 → ℝ := ![1, 1 / 2, 2]
private noncomputable def trainingInputs : Fin 3 → ℝ := ![0, 0, 1 / 2]
private def trainingLabels : Fin 3 → Fin 5 := ![1, 1, 2]

/-- The five supplied class coordinates are distinct, CM56. -/
theorem exampleCoordinates_injective : Function.Injective exampleCoordinates := by
  intro a b h
  apply Fin.ext
  have hv : (a.val : ℝ) = b.val := by
    dsimp [exampleCoordinates] at h
    linarith
  exact_mod_cast hv

/-- All five output classes meet the complete expansion's bound, CM56. -/
theorem exampleCoordinates_bound (c : Fin 5) : |exampleCoordinates c| ≤ 1 := by
  fin_cases c <;> norm_num [exampleCoordinates]

/-- A nonconstant target satisfies the identifying prior, CM56. -/
theorem example_realizable (x : Fin 3) :
    exampleCoordinates (exampleTarget x) = (1 / 2) * exampleFeature x + 1 / 4 := by
  fin_cases x <;> norm_num [exampleCoordinates, exampleTarget, exampleFeature]

/-- A nonzero feasible decoder has complete weighted cost 20, CM56. -/
theorem example_feasible : exampleParameters ∈ domain exampleCoordinates 0 (1 / 2) 1 2 32 1 := by
  norm_num [domain, ParameterSharing.weightedDomain, costs, exampleParameters,
    exampleCoordinates, Fin.sum_univ_succ]

/-- A different margin remains feasible, so the domain is not a singleton,
CM56. Its complete weighted cost is 10. -/
theorem half_feasible : halfParameters ∈ domain exampleCoordinates 0 (1 / 2) 1 2 32 1 := by
  norm_num [domain, ParameterSharing.weightedDomain, costs, halfParameters,
    exampleCoordinates, Fin.sum_univ_succ]

/-- This feasible domain permits genuine margin variation, CM56. -/
theorem distinct_feasible :
    exampleParameters ≠ halfParameters ∧
    exampleParameters ∈ domain exampleCoordinates 0 (1 / 2) 1 2 32 1 ∧
    halfParameters ∈ domain exampleCoordinates 0 (1 / 2) 1 2 32 1 := by
  refine ⟨?_, example_feasible, half_feasible⟩
  intro h
  have h0 := congrFun h 0
  norm_num [exampleParameters, halfParameters] at h0

/-- A different raw answer remains observationally possible when the affine
coordinate prior is removed, despite using the same fixed flat feature code,
CM56. The countertarget fails that prior; it is not a second realizable decoder. -/
theorem missing_prior_ambiguity :
    alternativeTarget 0 = exampleTarget 0 ∧ alternativeTarget 1 = exampleTarget 1 ∧
    alternativeTarget 2 ≠ exampleTarget 2 ∧
    ¬ ∃ a b : ℝ, ∀ x, exampleCoordinates (alternativeTarget x) = a * exampleFeature x + b := by
  refine ⟨by decide, by decide, by decide, ?_⟩
  rintro ⟨a, b, h⟩
  have h0 := h 0
  have h1 := h 1
  have h2 := h 2
  norm_num [alternativeTarget, exampleCoordinates, exampleFeature] at h0 h1 h2
  linarith

example : exampleTarget 2 ≠ trainingLabels 0 ∧ exampleTarget 2 ≠ trainingLabels 1 ∧
    exampleTarget 2 ≠ trainingLabels 2 := by decide

example : score exampleCoordinates exampleParameters 1 2 <
    score exampleCoordinates exampleParameters 1 3 := by
  apply strict_score exampleCoordinates exampleCoordinates_injective
  · norm_num [exampleParameters]
  · decide
  · norm_num [exampleParameters, exampleCoordinates]

example : score exampleCoordinates (fitted exampleCoordinates 0 (1 / 2) 1 2 4) 1 4 <
    score exampleCoordinates (fitted exampleCoordinates 0 (1 / 2) 1 2 4) 1 3 := by
  exact fitted_prediction exampleCoordinates exampleCoordinates_injective
    exampleFeature exampleTarget (1 / 2) (1 / 4) example_realizable 0 1
    (by norm_num [exampleFeature]) 4 (by norm_num) 2 4 (by decide)

/-- The actual complex head prefers the unseen label to every other raw class,
CM56. Its decoder was constrained only at the two observed feature anchors. -/
theorem example_unseen_prediction (c : Fin 5) (hc : c ≠ 3) :
    headLogits features (head exampleCoordinates exampleParameters) 1 c <
    headLogits features (head exampleCoordinates exampleParameters) 1 3 := by
  exact identified_prediction exampleCoordinates exampleCoordinates_injective
    exampleFeature exampleTarget (1 / 2) (1 / 4) example_realizable 0 1
    (by norm_num [exampleFeature]) (by norm_num : (0 : ℝ) < 1) example_feasible 2 c hc

example := example_unseen_prediction 4 (by decide)

example : head exampleCoordinates exampleParameters ∈ headBudget 32 :=
  head_budget exampleCoordinates exampleCoordinates_bound example_feasible.1

example : head exampleCoordinates exampleParameters (1, true) 4 = 1 := by
  norm_num [head, ParameterSharing.parameterMap, ParameterSharing.coefficientMap,
    expansion, complexFeatures, fourier4, exampleParameters, exampleCoordinates,
    Fin.sum_univ_succ]

example (p : Fin 3) (c : Fin 5) : ∑ k, |expansion exampleCoordinates p k c| ≤ costs p :=
  expansion_bound exampleCoordinates exampleCoordinates_bound p c

example := loss_convex exampleCoordinates exampleCoordinates_bound trainingInputs trainingLabels 32
example := batchLoss_convex exampleCoordinates exampleCoordinates_bound trainingInputs trainingLabels
  (Finset.univ : Finset (Fin 3)) 32
example := batchLoss_convex exampleCoordinates exampleCoordinates_bound trainingInputs trainingLabels
  ({2} : Finset (Fin 3)) 32
example := domain_loss_convex exampleCoordinates exampleCoordinates_bound trainingInputs trainingLabels
  0 (1 / 2) 1 2 32 1
example := domain_batchLoss_convex exampleCoordinates exampleCoordinates_bound trainingInputs trainingLabels
  ({2} : Finset (Fin 3)) 0 (1 / 2) 1 2 32 1
example := relabeling (Equiv.swap (1 : Fin 5) 4) exampleCoordinates 0 (1 / 2) 1 2 4 1 3

end Transformer.Grokking.Convex.ConvexDepth.ClassGeometry
