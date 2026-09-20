/-
# Kinetic theory for Transformers — the limiting correlations

Formalization of `th:lost` of arXiv:2605.09213v1,
*Kinetic theory for Transformers and the lost-in-the-middle phenomenon*, §1.3.

The mean-field limit says nothing about retrieval: the signal sits in the time
correlations, which vanish at that order.  The autocorrelation `A_φ` is `O(1)`
and is transported by the mean-field flow; the cross-correlation `C_φ` is
`O(N⁻¹)` and solves the linearization, forced at the source position `σ₀`
through the graphon `k_λ(σ,σ₀)`.  That forcing is what makes the retrieval
profile depend on where the answer was placed.

**The normalization of `∫_𝕋`.**  The source writes `f̂(n) = ∫_𝕋 e^{-inθ} f(θ) dθ`
and `(w_β' ∗_θ f)`, with `∫_𝕋` the *normalized* integral `(2π)⁻¹ ∫_0^{2π}`.
Three of its own statements force that reading: the iid uniform prompt has
`f_∘ ≡ 1` in `eq:init-conv`, where the left-hand side `E[e^{inθ⁰}]` is `≤ 1`;
`ŵ_β(n) = I_n(β)` is the normalized coefficient, which is what `a_n` is
calibrated against; and `A_φ ≡ φ - ∫_𝕋 φ` of §`sec:litm` is centered only for
the normalized mean.  So `torusConv` and `fourierDensity` carry the factor
`(2π)⁻¹`, densities here are densities for the normalized measure — the uniform
profile is `1`, not `(2π)⁻¹` — and `wConv` of `Transformer.Kinetic.MeanField`,
an integral against a probability measure, is the same convolution.
-/

import Transformer.Kinetic.Fluctuations

open scoped BigOperators
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Kinetic

variable {Ω : Type*} [MeasurableSpace Ω]

/-- The convolution `(w_β' ∗_θ g)(θ) = ∫_𝕋 w_β'(θ - θ') g(θ') dθ'` of a
`2π`-periodic density on `𝕋`, for the normalized integral — the same
convolution as `wConv`, written for a density instead of a measure.

Source: arXiv:2605.09213v1, `eq:mfl-lambda`. -/
noncomputable def torusConv (β : ℝ) (g : ℝ → ℝ) (θ : ℝ) : ℝ :=
  (2 * π)⁻¹ * ∫ θ' in (0 : ℝ)..(2 * π), wBetaDeriv β (θ - θ') * g θ'

/-- `w_β'` integrates to zero over a period, so a constant density is
stationary: `w_β' ∗_θ c = 0`.  This is what makes the uniform profile a
solution of `eq:mfl-lambda`. -/
@[simp]
theorem torusConv_const (β c θ : ℝ) : torusConv β (fun _ => c) θ = 0 := by
  have hcont : Continuous (wBetaDeriv β) := by
    unfold wBetaDeriv
    fun_prop
  have hsub : (∫ θ' in (0 : ℝ)..(2 * π), wBetaDeriv β (θ - θ')) = 0 := by
    rw [intervalIntegral.integral_comp_sub_left (wBetaDeriv β) θ, sub_zero,
      intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => hasDerivAt_wBeta β x)
        (hcont.intervalIntegrable _ _), (wBeta_periodic β).sub_eq θ, sub_self]
  simp [torusConv, intervalIntegral.integral_mul_const, hsub]

/-- The Fourier coefficient `ĝ(n) = ∫_𝕋 e^{-inθ} g(θ) dθ` of a density, for the
normalized integral on `𝕋`; see the header on that normalization.

Source: arXiv:2605.09213v1, §1.3. -/
noncomputable def fourierDensity (g : ℝ → ℝ) (n : ℤ) : ℂ :=
  (2 * (π : ℂ))⁻¹ *
    ∫ θ in (0 : ℝ)..(2 * π), Complex.exp (-(n : ℂ) * (θ : ℂ) * Complex.I) * (g θ : ℂ)

/-- The mean-field velocity `∫_0^σ k_λ(σ,σ')(w_β' ∗_θ f)(t,σ',θ) dσ'`, written
for a limit profile carrying a density — which is the form `eq:Aphi-lambda` and
`eq:Cphi-lambda` use.

Source: arXiv:2605.09213v1, `eq:mfl-lambda`. -/
noncomputable def velD (lam β : ℝ) (f : ℝ → ℝ → ℝ → ℝ) (t σ θ : ℝ) : ℝ :=
  ∫ σ' in (0 : ℝ)..σ, graphon lam σ σ' * torusConv β (f t σ') θ

/-- The weak formulation of `∂_t u = -∂_θ(u · V)` on the layer `σ`, with
velocity field `V`: for every smooth `2π`-periodic `ψ`,

  `d/dt ∫_𝕋 ψ u(t,·) = ∫_𝕋 ψ' u(t,θ) V(t,θ) dθ`. -/
def IsTransportedBy (V : ℝ → ℝ → ℝ) (u : ℝ → ℝ → ℝ) : Prop :=
  ∀ ψ : ℝ → ℝ, ContDiff ℝ (⊤ : ℕ∞) ψ → Function.Periodic ψ (2 * π) →
    ∀ t ∈ Set.Ici (0 : ℝ),
      HasDerivWithinAt (fun s => ∫ θ in (0 : ℝ)..(2 * π), ψ θ * u s θ)
        (∫ θ in (0 : ℝ)..(2 * π), deriv ψ θ * (u t θ * V t θ)) (Set.Ici 0) t

/-- **Equation (eq:mfl-lambda), density form.**  `f` solves the layered
McKean-Vlasov equation with initial profile `f_∘`, as a density in `θ`.

This is `IsMeanFieldSolution` written for a solution that has a density, which
is the form the correlation equations of §1.3 use — they multiply `f` by other
densities.

Source: arXiv:2605.09213v1, `eq:mfl-lambda`. -/
def IsDensitySolution (lam β : ℝ) (f₀ : ℝ → ℝ → ℝ) (f : ℝ → ℝ → ℝ → ℝ) : Prop :=
  (∀ σ θ, f 0 σ θ = f₀ σ θ) ∧
  ∀ σ ∈ Set.Ioc (0 : ℝ) 1, IsTransportedBy (fun t => velD lam β f t σ) (fun t => f t σ)

/-- **Equation (eq:Aphi-lambda).**  The limiting autocorrelation is transported
by the mean-field flow, starting from the centred observable:

  `∂_t A_φ = -∂_θ(A_φ V[f])`,  `A_φ(0,σ,θ) = f_∘(σ,θ)(φ(θ) - ∫_𝕋 φ f_∘(σ,·))`.

Source: arXiv:2605.09213v1, `eq:Aphi-lambda`. -/
def IsAphiSolution (lam β : ℝ) (f₀ : ℝ → ℝ → ℝ) (f : ℝ → ℝ → ℝ → ℝ) (φ : ℝ → ℝ)
    (A : ℝ → ℝ → ℝ → ℝ) : Prop :=
  (∀ σ θ, A 0 σ θ =
      f₀ σ θ * (φ θ - (2 * π)⁻¹ * ∫ θ' in (0 : ℝ)..(2 * π), φ θ' * f₀ σ θ')) ∧
  ∀ σ ∈ Set.Ioc (0 : ℝ) 1, IsTransportedBy (fun t => velD lam β f t σ) (fun t => A t σ)

/-- **Equation (eq:Cphi-lambda).**  The limiting cross-correlation solves the
linearization of `eq:mfl-lambda`, forced at the source position `σ₀`:

  `∂_t C_φ = -∂_θ(f ∫_{σ₀}^σ k_λ(σ,σ')(w_β' ∗_θ C_φ)(t,σ',·;σ₀) dσ')`
            `- ∂_θ(C_φ V[f]) - ∂_θ(f k_λ(σ,σ₀)(w_β' ∗_θ A_φ)(t,σ₀,·))`,
  `C_φ(0,σ,θ;σ₀) = 0`.

The three transport terms are combined into the single flux
`f·(linear term) + C_φ·V[f] + f·k_λ(σ,σ₀)(w_β' ∗ A_φ)(t,σ₀,·)` whose
`θ`-derivative the equation subtracts, so the weak form is one integration by
parts against `ψ'`.

Source: arXiv:2605.09213v1, `eq:Cphi-lambda`. -/
def IsCphiSolution (lam β : ℝ) (f : ℝ → ℝ → ℝ → ℝ) (A : ℝ → ℝ → ℝ → ℝ)
    (C : ℝ → ℝ → ℝ → ℝ → ℝ) : Prop :=
  (∀ σ σ₀ θ, C 0 σ σ₀ θ = 0) ∧
  ∀ σ ∈ Set.Ioc (0 : ℝ) 1, ∀ σ₀ ∈ Set.Ioc (0 : ℝ) 1,
    ∀ ψ : ℝ → ℝ, ContDiff ℝ (⊤ : ℕ∞) ψ → Function.Periodic ψ (2 * π) →
      ∀ t ∈ Set.Ici (0 : ℝ),
        HasDerivWithinAt (fun s => ∫ θ in (0 : ℝ)..(2 * π), ψ θ * C s σ σ₀ θ)
          (∫ θ in (0 : ℝ)..(2 * π), deriv ψ θ *
            (f t σ θ * ∫ σ' in σ₀..σ, graphon lam σ σ' * torusConv β (C t σ' σ₀) θ
              + C t σ σ₀ θ * velD lam β f t σ θ
              + f t σ θ * graphon lam σ σ₀ * torusConv β (A t σ₀) θ))
          (Set.Ici 0) t

/-- The covariance of a complex observable with a real one. -/
noncomputable def cplxCov (P : Measure Ω) (X : Ω → ℂ) (Y : Ω → ℝ) : ℂ :=
  (∫ ω, X ω * (Y ω : ℂ) ∂P) - (∫ ω, X ω ∂P) * ((∫ ω, Y ω ∂P : ℝ) : ℂ)

/-- `eq:def-Covn` in Fourier: `Â^N_φ(t,σ,n) = Cov(e^{-inθ_{⌈Nσ⌉}(t)}, φ(θ⁰_{⌈Nσ⌉}))`.

Source: arXiv:2605.09213v1, `eq:def-Covn`. -/
noncomputable def AhatN (P : Measure Ω) (N : ℕ) (ϑ : Ω → ℝ → Idx N → ℝ) (φ : ℝ → ℝ)
    (t : ℝ) (j : Idx N) (n : ℤ) : ℂ :=
  cplxCov P (fun ω => Complex.exp (-(n : ℂ) * (ϑ ω t j : ℂ) * Complex.I))
    (fun ω => φ (ϑ ω 0 j))

/-- `eq:def-Covn-2` in Fourier:
`Ĉ^N_φ(t,σ,n;σ₀) = Cov(e^{-inθ_{⌈Nσ⌉}(t)}, φ(θ⁰_{⌈Nσ₀⌉})) 1_{⌈Nσ₀⌉<⌈Nσ⌉}`.

Source: arXiv:2605.09213v1, `eq:def-Covn-2`. -/
noncomputable def ChatN (P : Measure Ω) (N : ℕ) (ϑ : Ω → ℝ → Idx N → ℝ) (φ : ℝ → ℝ)
    (t : ℝ) (j j₀ : Idx N) (n : ℤ) : ℂ :=
  if j₀ < j then
    cplxCov P (fun ω => Complex.exp (-(n : ℂ) * (ϑ ω t j : ℂ) * Complex.I))
      (fun ω => φ (ϑ ω 0 j₀))
  else 0

/-- `eq:init-conv` for a limit profile carrying a density, as `th:lost` uses it.

The sign remark of `InitConv` applies verbatim: the limit of `E[e^{inθ⁰}]` is
`f̂_∘(σ,-n)`, and since the condition is imposed for every `n ∈ ℤ` with
`⟨n⟩ = ⟨-n⟩`, that is the same assumption as the source's.

Source: arXiv:2605.09213v1, `eq:init-conv`. -/
def InitConvD (P : Measure Ω) (N : ℕ) (ϑ : Ω → Idx N → ℝ) (f₀ : ℝ → ℝ → ℝ)
    (δ γ Cst : ℝ) : Prop :=
  ∀ σ ∈ Set.Ioc (0 : ℝ) 1, ∀ n : ℤ, ∀ j : Idx N, (j : ℕ) + 1 = ⌈(N : ℝ) * σ⌉₊ →
    ‖(∫ ω, Complex.exp ((n : ℂ) * (ϑ ω j : ℂ) * Complex.I) ∂P) -
        fourierDensity (f₀ σ) (-n)‖
      ≤ Cst * (N : ℝ) ^ (-δ) * bracket n ^ γ

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

Not proved here.

Source: arXiv:2605.09213v1, `th:lost`, `eq:estim-ANA`, `eq:estim-CNC`. -/
theorem lost_correlations (lam β : ℝ) (f₀ : ℝ → ℝ → ℝ) (f : ℝ → ℝ → ℝ → ℝ)
    (hf : IsDensitySolution lam β f₀ f) :
    ∃ C : ℝ, 0 < C ∧
      ∀ (φ : ℝ → ℝ) (A : ℝ → ℝ → ℝ → ℝ), IsAphiSolution lam β f₀ f φ A →
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
              ‖(N : ℂ) * ChatN P N ϑ φ t j j₀ n - fourierDensity (Cφ t σ σ₀) n‖
                ≤ K * (N : ℝ) ^ (-ζ) * σ₀⁻¹ ^ 2 * bracket n ^ C * Real.exp (C * t) := by
  sorry

/-- The hypotheses of `lost_correlations` are satisfiable: the uniform prompt
`f_∘ ≡ 1` of the source's most homogeneous baseline, which is stationary
because `w_β' ∗ 1 = 0`, together with `φ ≡ 0`, for which `A ≡ 0` and `C ≡ 0`.

The velocity `V[f]` need not vanish for the correlations: `A` and `C` are
identically zero, so both transported quantities have zero flux, and the
initial condition of `eq:Aphi-lambda` is `f_∘(σ,θ)(0 - 0) = 0`. -/
example (lam β : ℝ) :
    IsDensitySolution lam β (fun _ _ => 1) (fun _ _ _ => 1) ∧
      IsAphiSolution lam β (fun _ _ => 1) (fun _ _ _ => 1) (fun _ => 0) (fun _ _ _ => 0) ∧
      IsCphiSolution lam β (fun _ _ _ => 1) (fun _ _ _ => 0) (fun _ _ _ _ => 0) := by
  refine ⟨⟨by intro σ θ; simp, ?_⟩, ⟨by intro σ θ; simp, ?_⟩, ?_⟩
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
