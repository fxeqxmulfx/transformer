/-
# Kinetic theory for Transformers — the limiting correlations

Formalization of `th:lost` of arXiv:2605.09213v1,
*Kinetic theory for Transformers and the lost-in-the-middle phenomenon*, §1.3.

The autocorrelation `Â^N_φ` of a token with its own initial value converges, at rate `N^{-ζ}`, to
the solution `A_φ` of the transported equation `eq:Aphi-lambda`; the cross-correlation
`N Ĉ^N_φ` of a token with an earlier one converges to the solution `C_φ` of the linearization
`eq:Cphi-lambda`, forced at the source position through the graphon.  The equations and the
observables are in `Transformer.Kinetic.CorrelationEquations`.
-/

import Transformer.Kinetic.CorrelationEquations

open scoped BigOperators
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Kinetic

variable {Ω : Type*} [MeasurableSpace Ω]

/-- **`f_∘ ∈ C⁰([0,1]; 𝒫(𝕋))`, for a profile carrying a density.**  For every `σ ∈ [0,1]` the
function `f_∘(σ,·)` is the density, for the normalized integral on `𝕋`, of a probability
measure: integrable on a period, nonnegative almost everywhere there, of normalized mass `1`;
and `σ ↦ f_∘(σ,·)` is continuous into `𝒫(𝕋)` for its topology of weak convergence — that is,
`σ ↦ ∫_𝕋 ψ f_∘(σ,·)` is continuous on `[0,1]` for every continuous `2π`-periodic `ψ`.

Source: arXiv:2605.09213v1, `th:lost`: "some limit profile `f_∘ ∈ C⁰([0,1]; 𝒫(𝕋))`". -/
def IsProbabilityProfile (f₀ : ℝ → ℝ → ℝ) : Prop :=
  (∀ σ ∈ Set.Icc (0 : ℝ) 1, IntervalIntegrable (f₀ σ) volume 0 (2 * π) ∧
      (∀ᵐ θ ∂(volume.restrict (Set.Ioc (0 : ℝ) (2 * π))), 0 ≤ f₀ σ θ) ∧
      (2 * π)⁻¹ * ∫ θ in (0 : ℝ)..(2 * π), f₀ σ θ = 1) ∧
  ∀ ψ : ℝ → ℝ, Continuous ψ → Function.Periodic ψ (2 * π) →
    ContinuousOn (fun σ => (2 * π)⁻¹ * ∫ θ in (0 : ℝ)..(2 * π), ψ θ * f₀ σ θ) (Set.Icc 0 1)

/-- **Theorem (th:lost).**  For independent initial data satisfying
`eq:init-conv` with `δ > 0`, and for `ζ ≤ δ` with `ζ < 1`,

  `|(Â^N_φ - Â_φ)(t,σ,n)| ≲_{ζ,φ} N^{-ζ} σ^{-2} ⟨n⟩^C e^{Ct}`,
  `|(N Ĉ^N_φ - Ĉ_φ)(t,σ,n;σ₀)| ≲_{ζ,φ} N^{-ζ} σ₀^{-2} ⟨n⟩^C e^{Ct}`,

where `A_φ` and `C_φ` are the unique solutions of `eq:Aphi-lambda` and
`eq:Cphi-lambda`.

Uniqueness is stated at the level of the Fourier coefficients the estimates
use, since a weak solution is determined only up to a null set in `θ`.

**How the two constants are quantified.**  The source's `≲_{ζ,φ}` names its
implicit prefactor's dependence: on `ζ`, on the test function `φ`, and — as
data fixed by the standing assumption — on `δ`, `γ` and the constant of
`eq:init-conv`.  The exponent `C` of `⟨n⟩^C e^{Ct}` is written explicitly and
is a constant of the model alone.  So `C` is quantified outside `φ` and outside
that data, and the prefactor `K` after them.  A single constant uniform in `ζ`
and `φ` would be a strengthening the source does not claim, and is the shape
this statement had before the audit.

**What the source says and what is changed here.**

* *The initial profile.*  The source takes `f_∘ ∈ C⁰([0,1]; 𝒫(𝕋))`.  The statement had an
  arbitrary function `f₀ : ℝ → ℝ → ℝ` — with no positivity, no unit mass and no continuity — and
  now assumes `IsProbabilityProfile f₀`.
* *The test function.*  The source takes `φ ∈ C^∞(𝕋)`; the statement had every `φ : ℝ → ℝ`, for
  which the covariances are integrals of non-measurable or non-integrable functions and the
  initial datum of `eq:Aphi-lambda` is junk.  It is now smooth and `2π`-periodic.
* *The cross-correlation is estimated for distinct tokens.*  The source states `eq:estim-CNC` for
  all `σ₀ < σ`.  When `⌈Nσ₀⌉ = ⌈Nσ⌉` — both in one cell of the grid — `Ĉ^N_φ = 0` by the
  indicator of `eq:def-Covn-2`, whereas `Ĉ_φ(t,σ,n;σ₀)` does not tend to `0` as `σ ↓ σ₀`: the
  coefficient `k_λ(σ,σ₀)` of the forcing in `eq:Cphi-lambda` tends to `λ (1 - e^{-λσ₀})⁻¹ > 0`
  (`σ₀⁻¹` at `λ = 0`).  For `f_∘ ≡ 1` one has `V[f] = 0`, `A_φ = φ - ∫_𝕋 φ`, and in Fourier
  `∂_t Ĉ_φ = a_n(∫_{σ₀}^σ k_λ(σ,σ') Ĉ_φ(t,σ') dσ' + k_λ(σ,σ₀) φ̂(n))`, so
  `Ĉ_φ(t,σ,n;σ₀) → t a_n λ (1 - e^{-λσ₀})⁻¹ φ̂(n)` as `σ ↓ σ₀`.  Along `N → ∞` with `σ` and `σ₀`
  in one cell the bound `N^{-ζ} → 0` cannot hold.  The source's proof (Step 1) works with the
  token indices `ℓ ≤ j`, and the case `ℓ = j` carries no cross-correlation; the estimate is
  therefore asked for `j₀ < j` (the autocorrelation estimate is unchanged).  This derivation is
  from the equations, not checked in Lean.
* *Densities, not measures (not repaired).*  The source's `f_∘`, `f`, `A_φ`, `C_φ` are measures or
  distributions on `𝕋` — its own example `f_∘ = Σ_m p_m(σ) δ_{ϑ_m}` is not a density, and for such
  data `C_φ` carries derivatives of the atoms, which is why its estimates are on Fourier
  coefficients.  Here they are densities, the form in which `eq:Aphi-lambda` and `eq:Cphi-lambda`
  are written in `CorrelationEquations`.  The theorem is thus for profiles whose correlation
  solutions are functions; the existence of those solutions is not asserted, only their uniqueness
  (at the Fourier level) and the estimates for solutions that exist.

Not proved here.

Source: arXiv:2605.09213v1, `th:lost`, `eq:estim-ANA`, `eq:estim-CNC`. -/
theorem lost_correlations (lam β : ℝ) (f₀ : ℝ → ℝ → ℝ) (hf₀ : IsProbabilityProfile f₀)
    (f : ℝ → ℝ → ℝ → ℝ) (hf : IsDensitySolution lam β f₀ f) :
    ∃ C : ℝ, 0 < C ∧
      ∀ φ : ℝ → ℝ, ContDiff ℝ (⊤ : ℕ∞) φ → Function.Periodic φ (2 * π) →
      ∀ A : ℝ → ℝ → ℝ → ℝ, IsAphiSolution lam β f₀ f φ A →
      ∀ Cφ : ℝ → ℝ → ℝ → ℝ → ℝ, IsCphiSolution lam β f A Cφ →
      (∀ A' : ℝ → ℝ → ℝ → ℝ, IsAphiSolution lam β f₀ f φ A' →
          ∀ t ∈ Set.Ici (0 : ℝ), ∀ σ ∈ Set.Ioc (0 : ℝ) 1, ∀ n : ℤ,
            fourierDensity (A' t σ) n = fourierDensity (A t σ) n) ∧
        (∀ C' : ℝ → ℝ → ℝ → ℝ → ℝ, IsCphiSolution lam β f A C' →
          ∀ t ∈ Set.Ici (0 : ℝ), ∀ σ ∈ Set.Ioc (0 : ℝ) 1, ∀ σ₀ ∈ Set.Ioc (0 : ℝ) 1, ∀ n : ℤ,
            fourierDensity (C' t σ σ₀) n = fourierDensity (Cφ t σ σ₀) n) ∧
        ∀ δ ζ Cγ γ : ℝ, 0 < δ → ζ ≤ δ → ζ < 1 →
        ∃ K : ℝ, 0 < K ∧
          ∀ (P : Measure Ω), IsProbabilityMeasure P →
          ∀ (N : ℕ) (ϑ : Ω → ℝ → Idx N → ℝ),
            (∀ ω, IsGPTFlow lam β N (ϑ ω)) →
            iIndepFun (fun (j : Idx N) (ω : Ω) => ϑ ω 0 j) P →
            InitConvD P N (fun ω => ϑ ω 0) f₀ δ γ Cγ →
          ∀ t ∈ Set.Ici (0 : ℝ), ∀ σ ∈ Set.Ioc (0 : ℝ) 1, ∀ σ₀ ∈ Set.Ioc (0 : ℝ) 1, σ₀ < σ →
          ∀ n : ℤ, ∀ j j₀ : Idx N,
            (j : ℕ) + 1 = ⌈(N : ℝ) * σ⌉₊ → (j₀ : ℕ) + 1 = ⌈(N : ℝ) * σ₀⌉₊ →
            ‖AhatN P N ϑ φ t j n - fourierDensity (A t σ) n‖
                ≤ K * (N : ℝ) ^ (-ζ) * σ⁻¹ ^ 2 * bracket n ^ C * Real.exp (C * t) ∧
              (j₀ < j →
                ‖(N : ℂ) * ChatN P N ϑ φ t j j₀ n - fourierDensity (Cφ t σ σ₀) n‖
                  ≤ K * (N : ℝ) ^ (-ζ) * σ₀⁻¹ ^ 2 * bracket n ^ C * Real.exp (C * t)) := by
  sorry

/-- The hypotheses of `lost_correlations` are satisfiable: the uniform prompt
`f_∘ ≡ 1` of the source's most homogeneous baseline, which is a probability profile and is
stationary because `w_β' ∗ 1 = 0`, together with the smooth periodic `φ ≡ 0`, for which
`A ≡ 0` and `C ≡ 0`.

The velocity `V[f]` need not vanish for the correlations: `A` and `C` are
identically zero, so both transported quantities have zero flux, and the
initial condition of `eq:Aphi-lambda` is `f_∘(σ,θ)(0 - 0) = 0`. -/
example (lam β : ℝ) :
    IsProbabilityProfile (fun _ _ => 1) ∧
      ContDiff ℝ (⊤ : ℕ∞) (fun _ : ℝ => (0 : ℝ)) ∧
      Function.Periodic (fun _ : ℝ => (0 : ℝ)) (2 * π) ∧
      IsDensitySolution lam β (fun _ _ => 1) (fun _ _ _ => 1) ∧
      IsAphiSolution lam β (fun _ _ => 1) (fun _ _ _ => 1) (fun _ => 0) (fun _ _ _ => 0) ∧
      IsCphiSolution lam β (fun _ _ _ => 1) (fun _ _ _ => 0) (fun _ _ _ _ => 0) := by
  refine ⟨⟨fun σ _ => ⟨intervalIntegrable_const, Filter.Eventually.of_forall fun _ => zero_le_one,
      ?_⟩, fun ψ _ _ => continuousOn_const⟩, contDiff_const, fun _ => rfl,
    ⟨by intro σ θ; simp, ?_⟩, ⟨by intro σ θ; simp, ?_⟩, ?_⟩
  · have hπ : (π : ℝ) ≠ 0 := Real.pi_pos.ne'
    simp [intervalIntegral.integral_const]
    field_simp
  · intro σ _ ψ _ _ t _
    simpa [velD] using
      hasDerivWithinAt_const t (Set.Ici (0 : ℝ)) (∫ θ in (0 : ℝ)..(2 * π), ψ θ)
  · intro σ _ ψ _ _ t _
    simpa using (hasDerivWithinAt_const t (Set.Ici (0 : ℝ)) (0 : ℝ))
  · refine ⟨by intro σ σ₀ θ; simp, ?_⟩
    intro σ _ σ₀ _ ψ _ _ t _
    simpa using (hasDerivWithinAt_const t (Set.Ici (0 : ℝ)) (0 : ℝ))

end Kinetic
end Transformer
