/-
# Real analytic germs: StandardCoordinates

Adapted from Bochao Kong's LocalComplexGeometry, revision
029697242294aea7989bf5d391199696a43490df, Germs/Coordinates.lean.
The coefficient field and analytic germs are real. Only the required
dependency closure is retained. Apache-2.0 license:
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.RealAnalyticGerms.Basic
import Mathlib.Topology.Algebra.Module.FiniteDimension
import Mathlib.Data.Fin.Tuple.Basic

open Filter
open scoped BigOperators Topology

noncomputable section
namespace Transformer.RealAnalyticGerms

open Transformer.AnalyticPreparation

/-- Split the last coordinate of `ℝⁿ⁺¹`, as a real-linear equivalence.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def wptAmbientLinearEquiv (n : ℕ) :
    Base (n + 1) ≃ₗ[ℝ] Transformer.AnalyticPreparation.Ambient n where
  toFun x := (fun i ↦ x i.castSucc, x (Fin.last n))
  invFun x := Fin.lastCases x.2 x.1
  left_inv x := by
    funext i
    cases i using Fin.lastCases <;> simp
  right_inv x := by
    ext i <;> simp
  map_add' x y := by
    ext i <;> simp
  map_smul' c x := by
    ext i <;> simp

/-- The continuous real-linear splitting of the last coordinate of `ℝⁿ⁺¹`.

The codomain is definitionally WPT's `(Fin n → ℝ) × ℝ` ambient space.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def wptAmbientEquiv (n : ℕ) :
    Base (n + 1) ≃L[ℝ] Transformer.AnalyticPreparation.Ambient n :=
  (wptAmbientLinearEquiv n).toContinuousLinearEquiv

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem wptAmbientEquiv_apply (n : ℕ) (x : Base (n + 1)) :
    wptAmbientEquiv n x = (fun i ↦ x i.castSucc, x (Fin.last n)) :=
  rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem wptAmbientEquiv_symm_apply (n : ℕ)
    (x : Transformer.AnalyticPreparation.Ambient n) :
    (wptAmbientEquiv n).symm x = Fin.lastCases x.2 x.1 :=
  rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem wptAmbientEquiv_zero (n : ℕ) :
    wptAmbientEquiv n (0 : Base (n + 1)) = 0 :=
  map_zero (wptAmbientEquiv n)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem wptAmbientEquiv_symm_zero (n : ℕ) :
    (wptAmbientEquiv n).symm (0 : Transformer.AnalyticPreparation.Ambient n) = 0 :=
  map_zero (wptAmbientEquiv n).symm

/-- Analyticity at the origin is preserved and reflected by the standard/WPT
ambient-coordinate equivalence.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticAt_comp_wptAmbientEquiv_iff (n : ℕ)
    (f : Transformer.AnalyticPreparation.Ambient n → ℝ) :
    AnalyticAt ℝ (f ∘ wptAmbientEquiv n) 0 ↔ AnalyticAt ℝ f 0 := by
  constructor
  · intro hf
    have hf' : AnalyticAt ℝ (f ∘ wptAmbientEquiv n)
        ((wptAmbientEquiv n).symm 0) := by
      rw [map_zero]
      exact hf
    have hcomp := hf'.compContinuousLinearMap
      (u := (wptAmbientEquiv n).symm.toContinuousLinearMap) (x := 0)
    simpa [Function.comp_def] using hcomp
  · intro hf
    have hf' : AnalyticAt ℝ f (wptAmbientEquiv n 0) := by
      rw [map_zero]
      exact hf
    simpa using hf'.compContinuousLinearMap
      (u := (wptAmbientEquiv n).toContinuousLinearMap) (x := 0)

/-- Neighborhood equality at the origin is preserved and reflected by the
standard/WPT ambient-coordinate equivalence.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem eventuallyEq_comp_wptAmbientEquiv_iff (n : ℕ)
    (f g : Transformer.AnalyticPreparation.Ambient n → ℝ) :
    (f ∘ wptAmbientEquiv n) =ᶠ[𝓝 0] (g ∘ wptAmbientEquiv n) ↔
      f =ᶠ[𝓝 0] g := by
  constructor
  · intro h
    have ht : Tendsto (wptAmbientEquiv n).symm (𝓝 0) (𝓝 0) := by
      have ht' : Tendsto (wptAmbientEquiv n).symm (𝓝 0)
          (𝓝 ((wptAmbientEquiv n).symm 0)) :=
        (wptAmbientEquiv n).symm.continuousAt
      rw [map_zero] at ht'
      exact ht'
    simpa [Function.comp_def] using h.comp_tendsto ht
  · intro h
    have ht : Tendsto (wptAmbientEquiv n) (𝓝 0) (𝓝 0) := by
      have ht' : Tendsto (wptAmbientEquiv n) (𝓝 0)
          (𝓝 (wptAmbientEquiv n 0)) :=
        (wptAmbientEquiv n).continuousAt
      rw [map_zero] at ht'
      exact ht'
    exact h.comp_tendsto ht


end Transformer.RealAnalyticGerms
