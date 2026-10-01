/-
# Real analytic germs: Coordinates

Adapted from Bochao Kong's LocalComplexGeometry, revision
029697242294aea7989bf5d391199696a43490df, Germs/Coordinates.lean.
The coefficient field and analytic germs are real. Only the required
dependency closure is retained. Apache-2.0 license:
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.RealAnalyticGerms.Pullback

open Filter
open scoped BigOperators Topology

noncomputable section
namespace Transformer.RealAnalyticGerms

open Transformer.AnalyticPreparation

/-- Projection from `ℝⁿ⁺¹` to its first `n` coordinates.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def baseProjectionCLM (n : ℕ) :
    Base (n + 1) →L[ℝ] Base n :=
  (ContinuousLinearMap.fst ℝ (Base n) ℝ).comp
    (wptAmbientEquiv n : Base (n + 1) →L[ℝ]
      Transformer.AnalyticPreparation.Ambient n)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem baseProjectionCLM_apply (n : ℕ) (x : Base (n + 1)) :
    baseProjectionCLM n x = fun i ↦ x i.castSucc :=
  rfl

/-- The zero-last-coordinate section `z ↦ (z, 0)` in the standard model.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def baseSectionCLM (n : ℕ) :
    Base n →L[ℝ] Base (n + 1) :=
  (wptAmbientEquiv n).symm.toContinuousLinearMap.comp
    (ContinuousLinearMap.inl ℝ (Base n) ℝ)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem baseSectionCLM_castSucc (n : ℕ) (z : Base n) (i : Fin n) :
    baseSectionCLM n z i.castSucc = z i := by
  simp [baseSectionCLM]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem baseSectionCLM_last (n : ℕ) (z : Base n) :
    baseSectionCLM n z (Fin.last n) = 0 := by
  simp [baseSectionCLM]

/-- Extraction of the last coordinate of `ℝⁿ⁺¹`.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def lastCoordinateCLM (n : ℕ) : Base (n + 1) →L[ℝ] ℝ :=
  (ContinuousLinearMap.snd ℝ (Base n) ℝ).comp
    (wptAmbientEquiv n : Base (n + 1) →L[ℝ]
      Transformer.AnalyticPreparation.Ambient n)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem lastCoordinateCLM_apply (n : ℕ) (x : Base (n + 1)) :
    lastCoordinateCLM n x = x (Fin.last n) :=
  rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem baseProjectionCLM_comp_baseSectionCLM (n : ℕ) :
    (baseProjectionCLM n).comp (baseSectionCLM n) =
      ContinuousLinearMap.id ℝ (Base n) := by
  apply ContinuousLinearMap.ext
  intro z
  funext i
  simp

/-- Include a base germ as a germ independent of the last coordinate.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def lowerDimensionalInclusion (n : ℕ) :
    AnalyticGerm n →+* AnalyticGerm (n + 1) :=
  analyticGermPullbackHom (baseProjectionCLM n)

/-- Restrict an ambient germ to the zero-last-coordinate base section.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def lowerDimensionalRestriction (n : ℕ) :
    AnalyticGerm (n + 1) →+* AnalyticGerm n :=
  analyticGermPullbackHom (baseSectionCLM n)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem lowerDimensionalInclusion_ofFunction {n : ℕ}
    (f : Base n → ℝ) (hf : AnalyticAt ℝ f 0) :
    lowerDimensionalInclusion n (AnalyticGerm.ofFunction f hf) =
      AnalyticGerm.ofFunction (f ∘ baseProjectionCLM n)
        (by
          have hf' : AnalyticAt ℝ f (baseProjectionCLM n 0) := by
            rw [map_zero]
            exact hf
          simpa using hf'.compContinuousLinearMap
            (u := baseProjectionCLM n) (x := 0)) :=
  analyticGermPullbackHom_ofFunction (baseProjectionCLM n) f hf

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem lowerDimensionalRestriction_ofFunction {n : ℕ}
    (f : Base (n + 1) → ℝ) (hf : AnalyticAt ℝ f 0) :
    lowerDimensionalRestriction n (AnalyticGerm.ofFunction f hf) =
      AnalyticGerm.ofFunction (f ∘ baseSectionCLM n)
        (by
          have hf' : AnalyticAt ℝ f (baseSectionCLM n 0) := by
            rw [map_zero]
            exact hf
          simpa using hf'.compContinuousLinearMap
            (u := baseSectionCLM n) (x := 0)) :=
  analyticGermPullbackHom_ofFunction (baseSectionCLM n) f hf

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem evalAtOrigin_lowerDimensionalInclusion {n : ℕ}
    (φ : AnalyticGerm n) :
    evalAtOrigin (lowerDimensionalInclusion n φ) = evalAtOrigin φ :=
  evalAtOrigin_analyticGermPullbackHom (baseProjectionCLM n) φ

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem evalAtOrigin_lowerDimensionalRestriction {n : ℕ}
    (φ : AnalyticGerm (n + 1)) :
    evalAtOrigin (lowerDimensionalRestriction n φ) = evalAtOrigin φ :=
  evalAtOrigin_analyticGermPullbackHom (baseSectionCLM n) φ

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem lowerDimensionalRestriction_comp_inclusion (n : ℕ) :
    (lowerDimensionalRestriction n).comp (lowerDimensionalInclusion n) =
      RingHom.id (AnalyticGerm n) := by
  unfold lowerDimensionalRestriction lowerDimensionalInclusion
  rw [← analyticGermPullbackHom_comp,
    baseProjectionCLM_comp_baseSectionCLM, analyticGermPullbackHom_id]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem lowerDimensionalInclusion_injective (n : ℕ) :
    Function.Injective (lowerDimensionalInclusion n) := by
  apply Function.LeftInverse.injective (g := lowerDimensionalRestriction n)
  intro φ
  change ((lowerDimensionalRestriction n).comp
    (lowerDimensionalInclusion n)) φ = φ
  rw [lowerDimensionalRestriction_comp_inclusion]
  rfl

/-- The germ of the last coordinate `w` on `ℝⁿ⁺¹`.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def lastCoordinateGerm (n : ℕ) : AnalyticGerm (n + 1) :=
  AnalyticGerm.ofFunction (lastCoordinateCLM n)
    ((lastCoordinateCLM n).analyticAt 0)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem evalAtOrigin_lastCoordinateGerm (n : ℕ) :
    evalAtOrigin (lastCoordinateGerm n) = 0 := by
  simp [lastCoordinateGerm]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem lowerDimensionalRestriction_lastCoordinateGerm (n : ℕ) :
    lowerDimensionalRestriction n (lastCoordinateGerm n) = 0 := by
  apply Subtype.ext
  apply Filter.Germ.coe_eq.mpr
  exact Filter.Eventually.of_forall fun z ↦ by
    simp

/-! ## The ambient germ ring as an algebra over the base germ ring -/

/-- The natural algebra structure induced by germs independent of the last coordinate.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
instance analyticGermSuccAlgebra (n : ℕ) :
    Algebra (AnalyticGerm n) (AnalyticGerm (n + 1)) :=
  (lowerDimensionalInclusion n).toAlgebra

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem algebraMap_analyticGermSucc (n : ℕ) :
    algebraMap (AnalyticGerm n) (AnalyticGerm (n + 1)) =
      lowerDimensionalInclusion n :=
  rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem algebraMap_analyticGermSucc_apply (n : ℕ) (φ : AnalyticGerm n) :
    algebraMap (AnalyticGerm n) (AnalyticGerm (n + 1)) φ =
      lowerDimensionalInclusion n φ :=
  rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticGermSucc_smul_eq_mul (n : ℕ)
    (a : AnalyticGerm n) (f : AnalyticGerm (n + 1)) :
    a • f = lowerDimensionalInclusion n a * f :=
  rfl

end Transformer.RealAnalyticGerms
