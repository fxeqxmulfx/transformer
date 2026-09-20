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

/-- **Equation (eq:mfl-lambda).**  `f` is a weak solution of the layered
McKean-Vlasov equation

  `∂_t f(t,σ,θ) = -∂_θ(f(t,σ,θ) V[f](t,σ,θ))`,  `f|_{t=0} = f_∘`.

The equation carries no derivative in `σ` — the cursor is a label, not a
direction of transport — so the weak formulation is taken layer by layer: for
each `σ ∈ (0,1]` and each smooth `2π`-periodic test function `ψ` on `𝕋`,

  `d/dt ∫_𝕋 ψ df(t,σ) = ∫_𝕋 ψ' V[f](t,σ,·) df(t,σ)`.

The `σ`-marginal of the limit is Lebesgue on `(0,1]`, as it is for the
empirical measures, so `f` is written through its disintegration
`σ ↦ f(t,σ,·) ∈ 𝒫(𝕋)`, which is what the source's `f(t,σ,θ)` denotes.

Source: arXiv:2605.09213v1, `eq:mfl-lambda`. -/
def IsMeanFieldSolution (lam β : ℝ) (f₀ : ℝ → ProbabilityMeasure Torus)
    (f : ℝ → ℝ → ProbabilityMeasure Torus) : Prop :=
  (∀ σ ∈ Set.Ioc (0 : ℝ) 1, f 0 σ = f₀ σ) ∧
  ∀ σ ∈ Set.Ioc (0 : ℝ) 1, ∀ ψ : ℝ → ℝ, ContDiff ℝ (⊤ : ℕ∞) ψ →
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

Source: arXiv:2605.09213v1, `th:mf-gpt`(i). -/
theorem mean_field_limit (lam β : ℝ) (f₀ : ℝ → ProbabilityMeasure Torus)
    (θ : (N : ℕ) → ℝ → Idx N → ℝ)
    (hflow : ∀ N : ℕ, IsGPTFlow lam β N (θ N))
    (hinit : EmpiricalTendsto (fun N => θ N 0) f₀) :
    ∃ f : ℝ → ℝ → ProbabilityMeasure Torus,
      IsMeanFieldSolution lam β f₀ f ∧
      (∀ t ∈ Set.Ici (0 : ℝ), EmpiricalTendsto (fun N => θ N t) (f t)) ∧
      ∀ g : ℝ → ℝ → ProbabilityMeasure Torus, IsMeanFieldSolution lam β f₀ g →
        ∀ t ∈ Set.Ici (0 : ℝ), ∀ σ ∈ Set.Ioc (0 : ℝ) 1, g t σ = f t σ := by
  sorry

/-- The hypotheses of `mean_field_limit` are satisfiable: tokens that never
move, all at the origin of `𝕋`, whose empirical measures are `δ_0` at every
cursor.

The flow hypothesis is witnessed for every `N`; the initial convergence is the
source's assumption on the prompt generator, and witnessing it means computing
the weak limit of the constant configurations, which is the content of the
statement rather than of its binders. -/
example (lam β : ℝ) : ∀ N : ℕ, IsGPTFlow lam β N (fun _ _ => 0) := by
  intro N t j
  have : gptField lam β N (fun _ => (0 : ℝ)) j = 0 := by
    simp [gptField, wBetaDeriv]
  rw [this]
  exact hasDerivAt_const t 0

end Kinetic
end Transformer
