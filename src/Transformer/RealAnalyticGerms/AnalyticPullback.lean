/-
# Pullback by arbitrary real analytic maps fixing the origin

Composition with an analytic parameterization induces an intrinsic ring
homomorphism on analytic germs. Finite equation bases consequently control
vanishing along nonlinear analytic curves as well as linear coordinate maps.
-/

import Transformer.RealAnalyticGerms.FiniteEquations

open Filter
open scoped Topology

noncomputable section
namespace Transformer.RealAnalyticGerms

open AnalyticPreparation

/-- A real analytic map fixing the origin pulls function germs back by
actual composition. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
def analyticFunctionPullbackHom {n m : ℕ}
    (T : Base n → Base m) (hT : AnalyticAt ℝ T 0) (hT0 : T 0 = 0) :
    FunctionGerm m →+* FunctionGerm n where
  toFun phi := phi.compTendsto T (by simpa only [hT0] using hT.continuousAt.tendsto)
  map_zero' := rfl
  map_one' := rfl
  map_add' phi psi := by
    refine Filter.Germ.inductionOn phi ?_
    intro f
    refine Filter.Germ.inductionOn psi ?_
    intro g
    rfl
  map_mul' phi psi := by
    refine Filter.Germ.inductionOn phi ?_
    intro f
    refine Filter.Germ.inductionOn psi ?_
    intro g
    rfl

/-- Analytic composition descends to a ring homomorphism on real analytic
germs. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def analyticPullbackHom {n m : ℕ}
    (T : Base n → Base m) (hT : AnalyticAt ℝ T 0) (hT0 : T 0 = 0) :
    AnalyticGerm m →+* AnalyticGerm n where
  toFun phi := ⟨analyticFunctionPullbackHom T hT hT0 phi.1, by
    obtain ⟨f, hf, hrep⟩ := phi.property
    refine ⟨f ∘ T, ?_, ?_⟩
    · have hf' : AnalyticAt ℝ f (T 0) := by simpa only [hT0] using hf
      exact hf'.comp hT
    · rw [← hrep]
      rfl⟩
  map_zero' := Subtype.ext (map_zero (analyticFunctionPullbackHom T hT hT0))
  map_one' := Subtype.ext (map_one (analyticFunctionPullbackHom T hT hT0))
  map_add' phi psi := Subtype.ext (map_add (analyticFunctionPullbackHom T hT hT0) phi.1 psi.1)
  map_mul' phi psi := Subtype.ext (map_mul (analyticFunctionPullbackHom T hT hT0) phi.1 psi.1)

/-- The pullback agrees with pointwise composition of every analytic
representative; there is no choice of quotient representative. Auxiliary
for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticPullbackHom_ofFunction {n m : ℕ}
    (T : Base n → Base m) (hT : AnalyticAt ℝ T 0) (hT0 : T 0 = 0)
    (f : Base m → ℝ) (hf : AnalyticAt ℝ f 0) :
    (analyticPullbackHom T hT hT0 (AnalyticGerm.ofFunction f hf) : FunctionGerm n) =
      ((f ∘ T : Base n → ℝ) : FunctionGerm n) := rfl

/-- A finite subfamily of any real analytic germ equations determines
their simultaneous vanishing along every analytic parameterization fixing
the origin. Equalities are equalities of germs, with their neighborhoods
allowed to depend on each original equation. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticGerm_finite_parameterized_equations {n : ℕ}
    (S : Set (AnalyticGerm n)) :
    ∃ F : Finset (AnalyticGerm n), (F : Set (AnalyticGerm n)) ⊆ S ∧
      ∀ (m : ℕ) (T : Base m → Base n) (hT : AnalyticAt ℝ T 0) (hT0 : T 0 = 0),
        (∀ f ∈ S, analyticPullbackHom T hT hT0 f = 0) ↔
          ∀ f ∈ F, analyticPullbackHom T hT hT0 f = 0 := by
  obtain ⟨F, hFS, hfinite⟩ := analyticGerm_finite_pullback_equations S
  exact ⟨F, hFS, fun m T hT hT0 => hfinite (analyticPullbackHom T hT hT0)⟩

end Transformer.RealAnalyticGerms
