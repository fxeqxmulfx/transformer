/-
# Real analytic germs: Pullback

Adapted from Bochao Kong's LocalComplexGeometry, revision
029697242294aea7989bf5d391199696a43490df, Germs/Coordinates.lean.
The coefficient field and analytic germs are real. Only the required
dependency closure is retained. Apache-2.0 license:
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.RealAnalyticGerms.FunctionPullback

open Filter
open scoped BigOperators Topology

noncomputable section
namespace Transformer.RealAnalyticGerms

open Transformer.AnalyticPreparation

/-- Pullback of analytic germs by a continuous real-linear map.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def analyticGermPullbackHom {n m : ℕ}
    (L : Base n →L[ℝ] Base m) :
    AnalyticGerm m →+* AnalyticGerm n where
  toFun φ :=
    ⟨functionGermPullbackHom L φ.1, by
      obtain ⟨f, hf, hφ⟩ := φ.property
      refine ⟨f ∘ L, ?_, ?_⟩
      · have hf' : AnalyticAt ℝ f (L 0) := by simpa using hf
        simpa using hf'.compContinuousLinearMap (u := L) (x := 0)
      · rw [← hφ]
        rfl⟩
  map_zero' := by
    apply Subtype.ext
    exact map_zero (functionGermPullbackHom L)
  map_one' := by
    apply Subtype.ext
    exact map_one (functionGermPullbackHom L)
  map_add' φ ψ := by
    apply Subtype.ext
    exact map_add (functionGermPullbackHom L) φ.1 ψ.1
  map_mul' φ ψ := by
    apply Subtype.ext
    exact map_mul (functionGermPullbackHom L) φ.1 ψ.1

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem analyticGermPullbackHom_coe {n m : ℕ}
    (L : Base n →L[ℝ] Base m)
    (φ : AnalyticGerm m) :
    ((analyticGermPullbackHom L φ : AnalyticGerm n) : FunctionGerm n) =
      functionGermPullbackHom L (φ : FunctionGerm m) :=
  rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem analyticGermPullbackHom_ofFunction {n m : ℕ}
    (L : Base n →L[ℝ] Base m)
    (f : Base m → ℝ) (hf : AnalyticAt ℝ f 0) :
    analyticGermPullbackHom L (AnalyticGerm.ofFunction f hf) =
      AnalyticGerm.ofFunction (f ∘ L)
        (by
          have hf' : AnalyticAt ℝ f (L 0) := by simpa using hf
          simpa using hf'.compContinuousLinearMap (u := L) (x := 0)) := by
  apply Subtype.ext
  rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem evalAtOrigin_analyticGermPullbackHom {n m : ℕ}
    (L : Base n →L[ℝ] Base m)
    (φ : AnalyticGerm m) :
    evalAtOrigin (analyticGermPullbackHom L φ) = evalAtOrigin φ := by
  exact functionGermPullbackHom_value L (φ : FunctionGerm m)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem analyticGermPullbackHom_id (n : ℕ) :
    analyticGermPullbackHom (ContinuousLinearMap.id ℝ (Base n)) =
      RingHom.id (AnalyticGerm n) := by
  ext φ
  exact DFunLike.congr_fun (functionGermPullbackHom_id n) φ.1

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticGermPullbackHom_comp {n m k : ℕ}
    (L : Base n →L[ℝ] Base m)
    (M : Base m →L[ℝ] Base k) :
    analyticGermPullbackHom (M.comp L) =
      (analyticGermPullbackHom L).comp (analyticGermPullbackHom M) := by
  ext φ
  exact DFunLike.congr_fun (functionGermPullbackHom_comp L M) φ.1

/-- Pullback by a continuous real-linear equivalence, as a ring equivalence.

The direction is contravariant: an equivalence `L : ℝⁿ ≃L[ℝ] ℝᵐ` induces an
equivalence from germs on `ℝᵐ` to germs on `ℝⁿ`.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def coordinatePullback {n m : ℕ}
    (L : Base n ≃L[ℝ] Base m) :
    AnalyticGerm m ≃+* AnalyticGerm n where
  toFun := analyticGermPullbackHom
    (L : Base n →L[ℝ] Base m)
  invFun := analyticGermPullbackHom
    (L.symm : Base m →L[ℝ] Base n)
  left_inv φ := by
    apply Subtype.ext
    exact functionGermPullbackHom_leftInverse L (φ : FunctionGerm m)
  right_inv φ := by
    apply Subtype.ext
    exact functionGermPullbackHom_rightInverse L (φ : FunctionGerm n)
  map_add' φ ψ := (analyticGermPullbackHom
    (L : Base n →L[ℝ] Base m)).map_add φ ψ
  map_mul' φ ψ := (analyticGermPullbackHom
    (L : Base n →L[ℝ] Base m)).map_mul φ ψ

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem coordinatePullback_apply {n m : ℕ}
    (L : Base n ≃L[ℝ] Base m)
    (φ : AnalyticGerm m) :
    coordinatePullback L φ = analyticGermPullbackHom
      (L : Base n →L[ℝ] Base m) φ :=
  rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem coordinatePullback_symm_apply {n m : ℕ}
    (L : Base n ≃L[ℝ] Base m)
    (φ : AnalyticGerm n) :
    (coordinatePullback L).symm φ = analyticGermPullbackHom
      (L.symm : Base m →L[ℝ] Base n) φ :=
  rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem evalAtOrigin_coordinatePullback {n m : ℕ}
    (L : Base n ≃L[ℝ] Base m)
    (φ : AnalyticGerm m) :
    evalAtOrigin (coordinatePullback L φ) = evalAtOrigin φ :=
  evalAtOrigin_analyticGermPullbackHom
    (L : Base n →L[ℝ] Base m) φ

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem coordinatePullback_refl (n : ℕ) :
    coordinatePullback (ContinuousLinearEquiv.refl ℝ (Base n)) =
      RingEquiv.refl (AnalyticGerm n) := by
  ext φ
  exact DFunLike.congr_fun (functionGermPullbackHom_id n) φ.1

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem coordinatePullback_trans {n m k : ℕ}
    (L : Base n ≃L[ℝ] Base m)
    (M : Base m ≃L[ℝ] Base k) :
    coordinatePullback (L.trans M) =
      (coordinatePullback M).trans (coordinatePullback L) := by
  ext φ
  exact DFunLike.congr_fun
    (functionGermPullbackHom_comp (L : Base n →L[ℝ] Base m)
      (M : Base m →L[ℝ] Base k)) φ.1

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem coordinatePullback_symm {n m : ℕ}
    (L : Base n ≃L[ℝ] Base m) :
    coordinatePullback L.symm = (coordinatePullback L).symm := by
  ext φ
  rfl


end Transformer.RealAnalyticGerms
