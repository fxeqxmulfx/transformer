/-
# Real analytic germs: ZeroDimension

Adapted from Bochao Kong's LocalComplexGeometry, revision
029697242294aea7989bf5d391199696a43490df, Germs/Ring.lean.
The coefficient field and analytic germs are real. Only the required
dependency closure is retained. Apache-2.0 license:
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.RealAnalyticGerms.Basic
import Mathlib.RingTheory.Noetherian.Basic
import Mathlib.RingTheory.PrincipalIdealDomain

open Filter
open scoped BigOperators Topology

noncomputable section
namespace Transformer.RealAnalyticGerms

open Transformer.AnalyticPreparation

/-- Embed a real number as a constant analytic germ.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def constantGermHom (n : ℕ) : ℝ →+* AnalyticGerm n where
  toFun c := AnalyticGerm.ofFunction (fun _ ↦ c) analyticAt_const
  map_zero' := by
    apply Subtype.ext
    rfl
  map_one' := by
    apply Subtype.ext
    rfl
  map_add' _ _ := by
    apply Subtype.ext
    rfl
  map_mul' _ _ := by
    apply Subtype.ext
    rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem evalAtOrigin_constantGerm (n : ℕ) (c : ℝ) :
    evalAtOriginHom n (constantGermHom n c) = c :=
  rfl

/-- Evaluation at the origin is onto, with the constant-germ map as a section.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem evalAtOrigin_surjective (n : ℕ) :
    Function.Surjective (evalAtOriginHom n) := by
  intro c
  exact ⟨constantGermHom n c, by simp⟩

/-- In real dimension zero, evaluation is injective.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem evalAtOrigin_injective_zero :
    Function.Injective (evalAtOriginHom 0) := by
  intro φ ψ h
  obtain ⟨f, hf, hφ⟩ := φ.property
  obtain ⟨g, hg, hψ⟩ := ψ.property
  have hfg0 : f 0 = g 0 := by
    change Filter.Germ.value (φ : FunctionGerm 0) =
      Filter.Germ.value (ψ : FunctionGerm 0) at h
    rw [← hφ, ← hψ] at h
    simpa using h
  apply Subtype.ext
  rw [← hφ, ← hψ]
  apply Filter.Germ.coe_eq.mpr
  apply Filter.Eventually.of_forall
  intro x
  have hx : x = (0 : Base 0) := Subsingleton.elim _ _
  subst x
  exact hfg0

/-- The zero-dimensional analytic germ ring is canonically `ℝ`.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def analyticGermZeroEquiv : AnalyticGerm 0 ≃+* ℝ :=
  RingEquiv.ofBijective (evalAtOriginHom 0)
    ⟨evalAtOrigin_injective_zero, evalAtOrigin_surjective 0⟩

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem analyticGermZeroEquiv_apply (f : AnalyticGerm 0) :
    analyticGermZeroEquiv f = evalAtOrigin f :=
  rfl

/-- The zero-dimensional case of Rückert's basis theorem.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticGerm_isNoetherian_zero :
    IsNoetherianRing (AnalyticGerm 0) :=
  isNoetherianRing_of_ringEquiv ℝ analyticGermZeroEquiv.symm

end Transformer.RealAnalyticGerms
