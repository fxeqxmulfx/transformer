/-
# Real analytic germs: FunctionPullback

Adapted from Bochao Kong's LocalComplexGeometry, revision
029697242294aea7989bf5d391199696a43490df, Germs/Coordinates.lean.
The coefficient field and analytic germs are real. Only the required
dependency closure is retained. Apache-2.0 license:
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.RealAnalyticGerms.StandardCoordinates

open Filter
open scoped BigOperators Topology

noncomputable section
namespace Transformer.RealAnalyticGerms

open Transformer.AnalyticPreparation

/-- Precomposition of function germs by a continuous linear map fixing the origin.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def functionGermPullbackHom {n m : ℕ}
    (L : Base n →L[ℝ] Base m) :
    FunctionGerm m →+* FunctionGerm n where
  toFun φ := φ.compTendsto L (by
    have hL : Tendsto L (𝓝 (0 : Base n)) (𝓝 (L 0)) :=
      L.continuous.continuousAt
    rw [L.map_zero] at hL
    exact hL)
  map_zero' := rfl
  map_one' := rfl
  map_add' φ ψ := by
    refine Filter.Germ.inductionOn φ ?_
    intro f
    refine Filter.Germ.inductionOn ψ ?_
    intro g
    rfl
  map_mul' φ ψ := by
    refine Filter.Germ.inductionOn φ ?_
    intro f
    refine Filter.Germ.inductionOn ψ ?_
    intro g
    rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem functionGermPullbackHom_coe {n m : ℕ}
    (L : Base n →L[ℝ] Base m)
    (f : Base m → ℝ) :
    functionGermPullbackHom L (f : FunctionGerm m) =
      ((f ∘ L : Base n → ℝ) : FunctionGerm n) :=
  rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem functionGermPullbackHom_value {n m : ℕ}
    (L : Base n →L[ℝ] Base m)
    (φ : FunctionGerm m) :
    Filter.Germ.value (functionGermPullbackHom L φ) = Filter.Germ.value φ := by
  refine Filter.Germ.inductionOn φ ?_
  intro f
  rw [functionGermPullbackHom_coe]
  simp only [Filter.Germ.value_ofFun, Function.comp_apply, map_zero]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem functionGermPullbackHom_id (n : ℕ) :
    functionGermPullbackHom (ContinuousLinearMap.id ℝ (Base n)) =
      RingHom.id (FunctionGerm n) := by
  ext φ
  refine Filter.Germ.inductionOn φ ?_
  intro f
  rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem functionGermPullbackHom_comp {n m k : ℕ}
    (L : Base n →L[ℝ] Base m)
    (M : Base m →L[ℝ] Base k) :
    functionGermPullbackHom (M.comp L) =
      (functionGermPullbackHom L).comp (functionGermPullbackHom M) := by
  ext φ
  refine Filter.Germ.inductionOn φ ?_
  intro f
  rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem functionGermPullbackHom_leftInverse {n m : ℕ}
    (L : Base n ≃L[ℝ] Base m) :
    Function.LeftInverse
      (functionGermPullbackHom
        (L.symm : Base m →L[ℝ] Base n))
      (functionGermPullbackHom
        (L : Base n →L[ℝ] Base m)) := by
  intro φ
  change ((functionGermPullbackHom
      (L.symm : Base m →L[ℝ] Base n)).comp
    (functionGermPullbackHom
      (L : Base n →L[ℝ] Base m))) φ = φ
  rw [← DFunLike.congr_fun (functionGermPullbackHom_comp
    (L.symm : Base m →L[ℝ] Base n)
    (L : Base n →L[ℝ] Base m)) φ]
  have hcomp :
      (L : Base n →L[ℝ] Base m).comp
          (L.symm : Base m →L[ℝ] Base n) =
        ContinuousLinearMap.id ℝ (Base m) := by
    apply ContinuousLinearMap.ext
    intro x
    exact L.apply_symm_apply x
  rw [hcomp, DFunLike.congr_fun (functionGermPullbackHom_id m) φ]
  rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem functionGermPullbackHom_rightInverse {n m : ℕ}
    (L : Base n ≃L[ℝ] Base m) :
    Function.RightInverse
      (functionGermPullbackHom
        (L.symm : Base m →L[ℝ] Base n))
      (functionGermPullbackHom
        (L : Base n →L[ℝ] Base m)) := by
  intro φ
  change ((functionGermPullbackHom
      (L : Base n →L[ℝ] Base m)).comp
    (functionGermPullbackHom
      (L.symm : Base m →L[ℝ] Base n))) φ = φ
  rw [← DFunLike.congr_fun (functionGermPullbackHom_comp
    (L : Base n →L[ℝ] Base m)
    (L.symm : Base m →L[ℝ] Base n)) φ]
  have hcomp :
      (L.symm : Base m →L[ℝ] Base n).comp
          (L : Base n →L[ℝ] Base m) =
        ContinuousLinearMap.id ℝ (Base n) := by
    apply ContinuousLinearMap.ext
    intro x
    exact L.symm_apply_apply x
  rw [hcomp, DFunLike.congr_fun (functionGermPullbackHom_id n) φ]
  rfl


end Transformer.RealAnalyticGerms
