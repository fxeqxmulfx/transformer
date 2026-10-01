/-
Copyright (c) 2025 Rémy Degenne. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Rémy Degenne
-/
module

public import KolmogorovExtension4.KolmogorovExtension
public import Mathlib.Probability.BrownianMotion.GaussianProjectiveFamily
public import Mathlib.Probability.HasLaw

@[expose] public section

open MeasureTheory
open scoped NNReal

namespace ProbabilityTheory

noncomputable abbrev gaussianProjectiveFamily := BrownianReal.projectiveFamily

noncomputable def gaussianLimit : Measure (ℝ≥0 → ℝ) :=
  projectiveLimit gaussianProjectiveFamily BrownianReal.isProjectiveMeasureFamily_projectiveFamily

instance IsProbabilityMeasure_gaussianLimit : IsProbabilityMeasure gaussianLimit :=
  isProbabilityMeasure_projectiveLimit BrownianReal.isProjectiveMeasureFamily_projectiveFamily

lemma isProjectiveLimit_gaussianLimit :
    IsProjectiveLimit gaussianLimit gaussianProjectiveFamily :=
  isProjectiveLimit_projectiveLimit BrownianReal.isProjectiveMeasureFamily_projectiveFamily

lemma _root_.MeasureTheory.IsProjectiveLimit.hasLaw_restrict {ι : Type*} {X : ι → Type*}
    {mX : ∀ i, MeasurableSpace (X i)} {μ : Measure (Π i, X i)}
    {P : (I : Finset ι) → Measure (Π i : I, X i)} (h : IsProjectiveLimit μ P) {I : Finset ι} :
    HasLaw I.restrict (P I) μ where
  map_eq := h I

lemma hasLaw_restrict_gaussianLimit {I : Finset ℝ≥0} :
    HasLaw I.restrict (gaussianProjectiveFamily I) gaussianLimit :=
  isProjectiveLimit_gaussianLimit.hasLaw_restrict

lemma hasLaw_eval_gaussianLimit {t : ℝ≥0} :
    HasLaw (fun x ↦ x t) (gaussianReal 0 t) gaussianLimit :=
  (BrownianReal.measurePreserving_eval_projectiveFamily
    (⟨t, by simp⟩ : ({t} : Finset ℝ≥0))).hasLaw.comp hasLaw_restrict_gaussianLimit

lemma covariance_eval_gaussianLimit {s t : ℝ≥0} :
    cov[fun x ↦ x s, fun x ↦ x t; gaussianLimit] = min s t := by
  convert (hasLaw_restrict_gaussianLimit (I := {s, t})).covariance_fun_comp
    (f := Function.eval ⟨s, by simp⟩) (g := Function.eval ⟨t, by simp⟩) ?_ ?_
  · rfl
  · rfl
  · rw [BrownianReal.covariance_eval_projectiveFamily]
  all_goals exact Measurable.aemeasurable (by fun_prop)

end ProbabilityTheory
