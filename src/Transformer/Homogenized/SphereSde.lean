/-
# Homogenized Transformers — well-posedness of the SDE on the sphere product

Formalization of `prop:existence_SDE_sphere` of arXiv:2604.01978v1,
*Homogenized Transformers*, Appendix: the Itô SDE

  `dX = B(X) dt + ∫_Θ G(X,θ) W(dθ,dt)`

on `(𝕊^{d-1})^n` has, for every starting configuration, a unique global
solution.  The source proves it by citing `[gess2024rsgd, Proposition 4.2]`
and noting that `(𝕊^{d-1})^n` is compact, so the explosion time is infinite.

**What the source says and what is carried here.**

* The solution concept is `def: ito.formula`, which `Generator.lean` renders in
  the only form Mathlib can carry — Mathlib has no stochastic integral and no
  cylindrical Wiener process — namely the one-dimensional marginal flow
  `d/dt E φ(X(t)) = E (𝖫 φ)(X(t))` of `IsItoSolution`.  Existence is therefore
  existence of a process with that marginal flow.

* "Unique" in the source is pathwise uniqueness of a strong solution, which
  cannot be written at all without the stochastic integral.  What is written
  here is its shadow at the level `IsItoSolution` lives at: any two processes
  with the marginal flow and the same starting configuration have the same
  one-dimensional marginals.  That is a genuine consequence of the source's
  claim — pathwise uniqueness gives uniqueness in law, hence well-posedness of
  the martingale problem, hence uniqueness of the marginals — and it is
  strictly weaker; the difference is what the missing stochastic integral costs.

* `B ∈ C¹((𝕊^{d-1})^n; T(𝕊^{d-1})^n)` and `G(·,θ) ∈ C²` are written as
  `Kernel.lean` writes them: ambient `ContDiff` for the smoothness, plus the
  tangency `⟨x_i, B(x)_i⟩ = 0` at unit tokens for the bundle.

* "with the integrability needed to make the stochastic integral well-defined"
  is made explicit as `G(x,·) ∈ L²(ρ*)` at every configuration and token, which
  is the condition under which `∫_Θ G W(dθ,dt)` is defined at all.

Source: arXiv:2604.01978v1, `prop:existence_SDE_sphere`, `def: ito.formula`.
-/

import Transformer.Homogenized.Generator

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-- **Proposition (prop:existence_SDE_sphere).**  Let `B ∈ C¹` and, for
`ρ*`-a.e. `θ`, `G(·,θ) ∈ C²` be tangent fields on `(𝕊^{d-1})^n` with
`G(x,·) ∈ L²(ρ*)`.  Then for every `x ∈ (𝕊^{d-1})^n` the Itô SDE
`eq: ito.sde.final` has a global solution started at `x`, and its
one-dimensional marginals are unique.

See the module docstring for the four readings this statement makes of the
source, of which the one that matters is that "unique strong solution" is
carried as uniqueness of the marginals, `IsItoSolution` being all that can be
written without a stochastic integral.

Not proved here.

Source: arXiv:2604.01978v1, `prop:existence_SDE_sphere`. -/
theorem existence_SDE_sphere {d n : ℕ} (ρ : Measure (HeadParam d)) [IsProbabilityMeasure ρ]
    (B : (Idx n → EucSpace d) → Idx n → EucSpace d)
    (G : HeadParam d → (Idx n → EucSpace d) → Idx n → EucSpace d)
    (hB : ContDiff ℝ 1 B)
    (hBtan : ∀ y : Idx n → EucSpace d, (∀ i, ‖y i‖ = 1) → ∀ i : Idx n,
      inner (𝕜 := ℝ) (y i) (B y i) = 0)
    (hG : ∀ᵐ θ ∂ρ, ContDiff ℝ 2 (G θ))
    (hGtan : ∀ (θ : HeadParam d) (y : Idx n → EucSpace d), (∀ i, ‖y i‖ = 1) → ∀ i : Idx n,
      inner (𝕜 := ℝ) (y i) (G θ y i) = 0)
    (hGsq : ∀ (y : Idx n → EucSpace d) (i : Idx n), Integrable (fun θ => ‖G θ y i‖ ^ 2) ρ)
    (x : Idx n → EucSpace d) (hx : ∀ i : Idx n, ‖x i‖ = 1) :
    ∃ (Ω : Type) (_ : MeasurableSpace Ω) (P : Measure Ω)
      (X : ℝ → Ω → (Idx n → EucSpace d)),
      IsProbabilityMeasure P ∧ (∀ ω, X 0 ω = x) ∧ IsItoSolution ρ B G P X ∧
      ∀ (Ω' : Type) (_ : MeasurableSpace Ω') (P' : Measure Ω')
        (Y : ℝ → Ω' → (Idx n → EucSpace d)),
        IsProbabilityMeasure P' → (∀ ω, Y 0 ω = x) → IsItoSolution ρ B G P' Y →
        ∀ φ : (Idx n → EucSpace d) → ℝ, ContDiff ℝ (⊤ : ℕ∞) φ →
        ∀ t ∈ Set.Ici (0 : ℝ), ∫ ω, φ (Y t ω) ∂P' = ∫ ω, φ (X t ω) ∂P := by
  sorry

/-- The hypotheses of `existence_SDE_sphere` are satisfiable: the degenerate
head law `ρ* = δ_0`, the zero drift and the zero noise kernel — which are
tangent, `C^∞` and square integrable — and the constant configuration at a unit
vector. -/
example (d n : ℕ) (e : EucSpace (d + 1)) (he : ‖e‖ = 1) :
    IsProbabilityMeasure (Measure.dirac (0 : HeadParam (d + 1))) ∧
      ContDiff ℝ 1 (0 : (Idx n → EucSpace (d + 1)) → Idx n → EucSpace (d + 1)) ∧
      (∀ y : Idx n → EucSpace (d + 1), (∀ i, ‖y i‖ = 1) → ∀ i : Idx n,
        inner (𝕜 := ℝ) (y i) ((0 : (Idx n → EucSpace (d + 1)) → Idx n → EucSpace (d + 1)) y i)
          = 0) ∧
      (∀ᵐ θ ∂(Measure.dirac (0 : HeadParam (d + 1))),
        ContDiff ℝ 2 ((0 : HeadParam (d + 1) → (Idx n → EucSpace (d + 1)) →
          Idx n → EucSpace (d + 1)) θ)) ∧
      (∀ (y : Idx n → EucSpace (d + 1)) (i : Idx n),
        Integrable (fun θ : HeadParam (d + 1) =>
          ‖(0 : HeadParam (d + 1) → (Idx n → EucSpace (d + 1)) → Idx n → EucSpace (d + 1))
            θ y i‖ ^ 2) (Measure.dirac 0)) ∧
      ∀ _i : Idx n, ‖e‖ = 1 :=
  ⟨inferInstance, contDiff_const, fun _ _ _ => by simp, .of_forall fun _ => contDiff_const,
    fun _ _ => by simp, fun _ => he⟩

end Homogenized
end Transformer
