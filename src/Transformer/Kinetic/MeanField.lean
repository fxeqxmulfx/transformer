/-
# Kinetic theory for Transformers — the mean-field limit

Formalization of `th:mf-gpt`(i) of arXiv:2605.09213v1,
*Kinetic theory for Transformers and the lost-in-the-middle phenomenon*, §1.3.

The empirical measure of the ALiBi dynamics converges to the unique weak
solution of a layered McKean-Vlasov equation on `(0,1] × 𝕋`, the layering being
the causal structure: a token at cursor `σ` sees only `σ' < σ`, weighted by the
graphon `k_λ`.
-/

import Transformer.Kinetic.Defs
import Transformer.Kinetic.Riemann
import Mathlib.MeasureTheory.Measure.ProbabilityMeasure
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Periodic
import Mathlib.Analysis.Calculus.Deriv.Shift

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Kinetic

instance instFactTwoPiPos : Fact (0 < 2 * π) := ⟨by positivity⟩

/-- The torus `𝕋 = ℝ/2πℤ` on which the minimal decoder's tokens live.

Source: arXiv:2605.09213v1, §1. -/
noncomputable abbrev Torus := AddCircle (2 * π)

/-- The periodic derivative `w_β'` as a function on `𝕋`. -/
noncomputable def wDerivT (β : ℝ) : Torus → ℝ := (wBetaDeriv_periodic β).lift

/-- The derivative of a periodic function is periodic. -/
theorem periodic_deriv {f : ℝ → ℝ} {c : ℝ} (h : Function.Periodic f c) :
    Function.Periodic (deriv f) c := by
  intro x
  have h2 : (fun y : ℝ => f (y + c)) = f := funext fun y => h y
  have h1 : deriv (fun y : ℝ => f (y + c)) x = deriv f (x + c) := deriv_comp_add_const f c x
  rw [h2] at h1
  exact h1.symm

/-- The convolution `w_β' ∗_θ μ` on `𝕋`. -/
noncomputable def wConv (β : ℝ) (μ : ProbabilityMeasure Torus) (x : Torus) : ℝ :=
  ∫ y, wDerivT β (x - y) ∂(μ : Measure Torus)

/-- The velocity field of `eq:mfl-lambda`,

  `V[f](t,σ,θ) = ∫_0^σ k_λ(σ,σ') (w_β' ∗_θ f)(t,σ',θ) dσ'`.

Source: arXiv:2605.09213v1, `eq:mfl-lambda`. -/
noncomputable def mfVelocity (lam β : ℝ) (f : ℝ → ℝ → ProbabilityMeasure Torus)
    (t σ : ℝ) (x : Torus) : ℝ :=
  ∫ σ' in (0 : ℝ)..σ, graphon lam σ σ' * wConv β (f t σ') x

/-- **A cursor kernel.**  A family `σ ↦ g(σ) ∈ 𝒫(𝕋)` — the disintegration over the cursor of a
measure on `(0,1] × 𝕋` — is *measurable in `σ`*: `σ ↦ ∫ ψ dg(σ)` is almost everywhere
measurable on `(0,1]` for every continuous `ψ` on `𝕋`.

This is what gives sense to `∫_0^σ k_λ(σ,σ') (w_β' ∗_θ g(σ')) dσ'` and to the integral over
the cursor in `EmpiricalTendsto`: an integral of a non-measurable function is `0` in Lean, and
a family of measures on `𝕋` with no measurability in `σ` is not a measure on `(0,1] × 𝕋`. -/
def IsCursorKernel (g : ℝ → ProbabilityMeasure Torus) : Prop :=
  ∀ ψ : Torus → ℝ, Continuous ψ →
    AEMeasurable (fun σ => ∫ x, ψ x ∂(g σ : Measure Torus))
      (volume.restrict (Set.Ioc (0 : ℝ) 1))

/-- **Equation (eq:mfl-lambda).**  `f` is a weak solution of the layered
McKean-Vlasov equation

  `∂_t f(t,σ,θ) = -∂_θ(f(t,σ,θ) V[f](t,σ,θ))`,  `f|_{t=0} = f_∘`.

The equation carries no derivative in `σ` — the cursor is a label, not a
direction of transport — so the weak formulation is taken layer by layer: for
almost every `σ ∈ (0,1]` and each smooth `2π`-periodic test function `ψ` on `𝕋`,

  `d/dt ∫_𝕋 ψ df(t,σ) = ∫_𝕋 ψ' V[f](t,σ,·) df(t,σ)`.

The `σ`-marginal of the limit is Lebesgue on `(0,1]`, as it is for the
empirical measures, so `f` is written through its disintegration
`σ ↦ f(t,σ,·) ∈ 𝒫(𝕋)`, which is what the source's `f(t,σ,θ)` denotes and which a measure on
`(0,1] × 𝕋` determines only for almost every `σ`: the initial condition and the equation are
asked for almost every `σ`, the velocity `V[f](t,σ,·)` depends on `f(t,σ',·)` for `σ' < σ`
through an integral and so not on the choice of the kernel.  The disintegration is measurable in
`σ` (`IsCursorKernel`), which is part of being a solution: without it the velocity is not
defined.  As the source's solutions are weakly continuous in `t`, the equation is asked as a
time derivative at every `t ≥ 0`.

Source: arXiv:2605.09213v1, `eq:mfl-lambda`. -/
def IsMeanFieldSolution (lam β : ℝ) (f₀ : ℝ → ProbabilityMeasure Torus)
    (f : ℝ → ℝ → ProbabilityMeasure Torus) : Prop :=
  (∀ᵐ σ ∂(volume.restrict (Set.Ioc (0 : ℝ) 1)), f 0 σ = f₀ σ) ∧
  (∀ t ∈ Set.Ici (0 : ℝ), IsCursorKernel (f t)) ∧
  ∀ᵐ σ ∂(volume.restrict (Set.Ioc (0 : ℝ) 1)),
    ∀ ψ : ℝ → ℝ, ContDiff ℝ (⊤ : ℕ∞) ψ →
      ∀ hψ : Function.Periodic ψ (2 * π), ∀ t ∈ Set.Ici (0 : ℝ),
        HasDerivWithinAt (fun s => ∫ x, hψ.lift x ∂(f s σ : Measure Torus))
          (∫ x, (periodic_deriv hψ).lift x * mfVelocity lam β f t σ x
            ∂(f t σ : Measure Torus))
          (Set.Ici 0) t

/-- Weak-* convergence of the empirical measures `μ_N = N⁻¹ Σ_j δ_{(j/N, θ_j)}`
to the profile `g : (0,1] → 𝒫(𝕋)`, tested against `φ ∈ C((0,1] × 𝕋)`.

Source: arXiv:2605.09213v1, `eq:emp-meas`. -/
def EmpiricalTendsto (ϑ : (N : ℕ) → Idx N → ℝ) (g : ℝ → ProbabilityMeasure Torus) : Prop :=
  ∀ φ : ℝ × Torus → ℝ, Continuous φ →
    Filter.Tendsto
      (fun N : ℕ => (N : ℝ)⁻¹ * ∑ j : Idx N, φ (((j : ℝ) + 1) / N, ((ϑ N j : ℝ) : Torus)))
      Filter.atTop
      (nhds (∫ σ in (0 : ℝ)..1, ∫ x, φ (σ, x) ∂(g σ : Measure Torus)))

/-- **Theorem (th:mf-gpt)(i), qualitative mean-field limit.**

If `μ_N|_{t=0} ⇀ f_∘` in `𝒫((0,1] × 𝕋)`, then `μ_N(t) ⇀ f(t)` for every
`t ≥ 0`, where `f` is the unique weak solution of `eq:mfl-lambda`.

Stated as existence of the limit together with its uniqueness as a weak
solution, so that the limit is produced rather than assumed.

Not proved here.

**What the source says and what is changed here.**  The source's limit `f_∘` and
solution `f` are measures on `(0,1] × 𝕋` — elements of `L^∞(ℝ_{≥0}; 𝒫((0,1] × 𝕋))` —
and `f(t,σ,θ)` denotes their disintegration over the cursor `σ`, which is defined only for
almost every `σ`.

* The earlier statement took `f_∘` and `f` to be arbitrary maps `σ ↦ 𝒫(𝕋)`, with no
  measurability in `σ`.  For such maps the velocity `V[f]` and the limit integral of
  `EmpiricalTendsto` are integrals of non-measurable functions, which vanish in Lean, so
  the statement was about junk.  The initial profile is now a cursor kernel (`hf₀`) and so
  is each `f t` (a clause of `IsMeanFieldSolution`).  The `L^∞` bound in `t` is automatic for
  probability measures.
* The earlier statement asked the initial condition and the equation at *every* `σ`, and
  `g t σ = f t σ` for every `σ`.  A disintegration is determined only almost everywhere, and the
  source identifies solutions as measures (elements of `L^∞`), so the initial condition, the
  equation and the uniqueness are now almost everywhere in `σ ∈ (0,1]`.
* The source fixes `β > 0` (`eq:wbeta`); `hβ` is added (the earlier statement had every real
  `β`).  `λ ∈ ℝ` is as in the source.

Source: arXiv:2605.09213v1, `th:mf-gpt`(i). -/
theorem mean_field_limit (lam β : ℝ) (hβ : 0 < β) (f₀ : ℝ → ProbabilityMeasure Torus)
    (hf₀ : IsCursorKernel f₀) (θ : (N : ℕ) → ℝ → Idx N → ℝ)
    (hflow : ∀ N : ℕ, IsGPTFlow lam β N (θ N))
    (hinit : EmpiricalTendsto (fun N => θ N 0) f₀) :
    ∃ f : ℝ → ℝ → ProbabilityMeasure Torus,
      IsMeanFieldSolution lam β f₀ f ∧
      (∀ t ∈ Set.Ici (0 : ℝ), EmpiricalTendsto (fun N => θ N t) (f t)) ∧
      ∀ g : ℝ → ℝ → ProbabilityMeasure Torus, IsMeanFieldSolution lam β f₀ g →
        ∀ t ∈ Set.Ici (0 : ℝ),
          ∀ᵐ σ ∂(volume.restrict (Set.Ioc (0 : ℝ) 1)), g t σ = f t σ := by
  sorry

/-- The hypotheses of `mean_field_limit` are satisfiable: tokens that never move, all at the
origin of `𝕋`, whose empirical measures converge to `δ_0` at every cursor (Riemann sums,
`tendsto_rightRiemann`).  The flow is witnessed for every `N`, the profile `σ ↦ δ_0` is
constant and so a cursor kernel. -/
example : ∃ lam β : ℝ, 0 < β ∧ ∃ f₀ : ℝ → ProbabilityMeasure Torus, IsCursorKernel f₀ ∧
    (∀ N : ℕ, IsGPTFlow lam β N (fun _ _ => 0)) ∧
    EmpiricalTendsto (fun N => (fun _ _ => 0 : ℝ → Idx N → ℝ) 0) f₀ := by
  refine ⟨0, 1, one_pos, fun _ => ⟨Measure.dirac 0, inferInstance⟩, fun ψ _ => aemeasurable_const,
    ?_, ?_⟩
  · intro N t j
    have : gptField 0 1 N (fun _ => (0 : ℝ)) j = 0 := by
      simp [gptField, wBetaDeriv]
    rw [this]
    exact hasDerivAt_const t 0
  · intro φ hφ
    have h := tendsto_rightRiemann (g := fun σ => φ (σ, 0))
      (hφ.comp (continuous_id.prodMk continuous_const))
    simpa [integral_dirac] using h

/-- `IsMeanFieldSolution` is inhabited: the profile `δ_0` at every cursor and every time solves
the equation with initial datum `δ_0`, because `w_β'(0) = 0` makes the velocity vanish on the
support of the measure. -/
example (lam β : ℝ) :
    IsMeanFieldSolution lam β (fun _ => ⟨Measure.dirac 0, inferInstance⟩)
      (fun _ _ => ⟨Measure.dirac 0, inferInstance⟩) := by
  refine ⟨Filter.Eventually.of_forall fun _ => rfl, fun t _ ψ _ => aemeasurable_const,
    Filter.Eventually.of_forall fun σ ψ _ hψ t _ => ?_⟩
  have hw : wDerivT β 0 = 0 := by
    have h := (wBetaDeriv_periodic β).lift_coe 0
    rw [AddCircle.coe_zero] at h
    simp [wDerivT, h, wBetaDeriv]
  have hv : mfVelocity lam β (fun _ _ => (⟨Measure.dirac 0, inferInstance⟩ :
      ProbabilityMeasure Torus)) t σ 0 = 0 := by
    simp [mfVelocity, wConv, hw]
  show HasDerivWithinAt (fun _ : ℝ => ∫ x, hψ.lift x ∂(Measure.dirac (0 : Torus)))
    (∫ x, (periodic_deriv hψ).lift x * mfVelocity lam β _ t σ x ∂(Measure.dirac (0 : Torus)))
    (Set.Ici 0) t
  rw [integral_dirac, integral_dirac, hv, mul_zero]
  exact hasDerivWithinAt_const t (Set.Ici (0 : ℝ)) _

end Kinetic
end Transformer
