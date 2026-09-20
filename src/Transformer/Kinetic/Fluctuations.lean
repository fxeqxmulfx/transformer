/-
# Kinetic theory for Transformers — fluctuations

Formalization of `th:mf-gpt`(ii) and `lem:correl` of arXiv:2605.09213v1,
*Kinetic theory for Transformers and the lost-in-the-middle phenomenon*.

The quantitative half of the mean-field limit, and the Glauber-calculus
correlation estimates that drive it: a two-point covariance of size `i⁻¹` and a
third cumulant of size `(ij)⁻¹`, both with the nonstandard factor
`e^{√(CT log(i/j))}` that the singular cumulative influence of small indices
produces.
-/

import Transformer.Kinetic.MeanField
import Mathlib.Probability.Independence.Basic
import Mathlib.Probability.Moments.Covariance
import Mathlib.Analysis.Fourier.AddCircle

open scoped BigOperators
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Kinetic

variable {Ω : Type*} [MeasurableSpace Ω]

/-- The Japanese bracket `⟨n⟩ = √(1 + n²)`.

Source: arXiv:2605.09213v1, §1.3. -/
noncomputable def bracket (n : ℤ) : ℝ := Real.sqrt (1 + (n : ℝ) ^ 2)

/-- **Equation (eq:init-conv).**  The initial token laws converge to the
profile `f_∘` at rate `N^{-δ}` with polynomial loss in the frequency:

  `|E[e^{inθ⁰_{⌈Nσ⌉}}] - f̂_∘(σ,n)| ≤ C N^{-δ} ⟨n⟩^γ`  for all `σ ∈ (0,1]`, `n ∈ ℤ`.

**What the source says and what is changed here.**  The source pairs
`E[e^{+inθ⁰}]` with `f̂_∘(σ,n) = ∫_𝕋 e^{-inθ} f_∘(σ,dθ)`, whose limit is in fact
`f̂_∘(σ,-n)`.  Since the condition is imposed for every `n ∈ ℤ` and
`⟨n⟩ = ⟨-n⟩`, the two readings define the same assumption; the one written here
is `∫_𝕋 e^{inθ} f_∘(σ,dθ)`, which is `fourier n`.

The position `⌈Nσ⌉` of the source is `1`-based, so it is the token of index
`j : Idx N` with `j + 1 = ⌈Nσ⌉`.

Source: arXiv:2605.09213v1, `eq:init-conv`. -/
def InitConv (P : Measure Ω) (N : ℕ) (ϑ : Ω → Idx N → ℝ)
    (f₀ : ℝ → ProbabilityMeasure Torus) (δ γ Cst : ℝ) : Prop :=
  ∀ σ ∈ Set.Ioc (0 : ℝ) 1, ∀ n : ℤ, ∀ j : Idx N, (j : ℕ) + 1 = ⌈(N : ℝ) * σ⌉₊ →
    ‖(∫ ω, fourier (T := 2 * π) n ((ϑ ω j : ℝ) : Torus) ∂P) -
        ∫ x, fourier (T := 2 * π) n x ∂(f₀ σ : Measure Torus)‖
      ≤ Cst * (N : ℝ) ^ (-δ) * bracket n ^ γ

/-- The third joint cumulant `κ_{1,1,1}(Y,Y',Y'') = E[(Y-EY)(Y'-EY')(Y''-EY'')]`.

Source: arXiv:2605.09213v1, before `lem:correl`. -/
noncomputable def thirdCumulant (P : Measure Ω) (Y Y' Y'' : Ω → ℝ) : ℝ :=
  ∫ ω, (Y ω - ∫ ω', Y ω' ∂P) * (Y' ω - ∫ ω', Y' ω' ∂P) * (Y'' ω - ∫ ω', Y'' ω' ∂P) ∂P

/-- **Lemma (lem:correl).**  For independent initial data, `φ ∈ C²(𝕋)`,
`t,t',t'' ∈ [0,T]` and `i > j > k`,

  `|Cov(φ(θ_i(t)), φ(θ_j(t')))| ≲ i⁻¹ e^{√(CT log(i/j))} e^{CT} ‖φ'‖_∞²`,
  `|κ_{1,1,1}(φ(θ_i(t)), φ(θ_j(t')), φ(θ_k(t'')))|
      ≲ (ij)⁻¹ e^{√(CT log(i/k))} e^{CT} ‖φ'‖_∞² ‖φ'‖_{W^{1,∞}}`.

The source's `i, j, k` are `1`-based positions, so they are `i+1, j+1, k+1`
here.  The two norms of `φ'` are carried as bounds `K ≥ ‖φ'‖_∞` and
`K' ≥ ‖φ'‖_{W^{1,∞}} = max{‖φ'‖_∞, ‖φ''‖_∞}`, which is what `≲ ‖·‖` means and
avoids the junk value an unbounded supremum would take.

Not proved here.

Source: arXiv:2605.09213v1, `lem:correl`, `eq:est-cov-th`, `eq:est-kap3-th`. -/
theorem correlation_bounds (lam β : ℝ) :
    ∃ C : ℝ, 0 < C ∧ ∀ (N : ℕ) (P : Measure Ω), IsProbabilityMeasure P →
      ∀ ϑ : Ω → ℝ → Idx N → ℝ,
        (∀ ω, IsGPTFlow lam β N (ϑ ω)) →
        iIndepFun (fun (j : Idx N) (ω : Ω) => ϑ ω 0 j) P →
      ∀ (φ dφ ddφ : ℝ → ℝ), (∀ x, HasDerivAt φ (dφ x) x) → (∀ x, HasDerivAt dφ (ddφ x) x) →
      ∀ K K' : ℝ, (∀ x, |dφ x| ≤ K) → (∀ x, |dφ x| ≤ K') → (∀ x, |ddφ x| ≤ K') →
      ∀ T : ℝ, 0 ≤ T → ∀ t ∈ Set.Icc (0 : ℝ) T, ∀ t' ∈ Set.Icc (0 : ℝ) T,
        ∀ t'' ∈ Set.Icc (0 : ℝ) T, ∀ i j k : Idx N, k < j → j < i →
          |covariance (fun ω => φ (ϑ ω t i)) (fun ω => φ (ϑ ω t' j)) P|
              ≤ C * ((i : ℝ) + 1)⁻¹ *
                Real.exp (Real.sqrt (C * T * Real.log (((i : ℝ) + 1) / ((j : ℝ) + 1)))) *
                Real.exp (C * T) * K ^ 2 ∧
            |thirdCumulant P (fun ω => φ (ϑ ω t i)) (fun ω => φ (ϑ ω t' j))
                (fun ω => φ (ϑ ω t'' k))|
              ≤ C * (((i : ℝ) + 1) * ((j : ℝ) + 1))⁻¹ *
                Real.exp (Real.sqrt (C * T * Real.log (((i : ℝ) + 1) / ((k : ℝ) + 1)))) *
                Real.exp (C * T) * K ^ 2 * K' := by
  sorry

/-- The hypotheses of `correlation_bounds` are satisfiable: the system of a
single token, which never moves, on the one-point probability space.  The index
hypotheses `k < j < i` are then unsatisfiable in `Idx 1`, which is why they are
quantified inside the statement, where the source puts them, rather than in the
binders. -/
example : IsProbabilityMeasure (Measure.dirac () : Measure Unit) ∧
    IsGPTFlow (1 : ℝ) 1 1 (fun _ _ => 0) ∧
    iIndepFun (fun (_ : Idx 1) (_ : Unit) => (0 : ℝ)) (Measure.dirac ()) := by
  refine ⟨inferInstance, ?_, iIndepFun.of_subsingleton⟩
  intro t j
  have : gptField (1 : ℝ) 1 1 (fun _ => (0 : ℝ)) j = 0 := by
    simp [gptField, wBetaDeriv]
  rw [this]
  exact hasDerivAt_const t 0

/-- **Theorem (th:mf-gpt)(ii), error estimates.**  Under `eq:init-conv` with
`δ > 0`, for every `t ≥ 0`, `φ ∈ C^∞([0,1] × 𝕋)` and `m > γ ∨ 2 + 1/2`,

  `E[|∫ φ (μ_N(t) - f(t))|²]^{1/2} ≲ N^{-δ∧1/2} e^{Ct} ‖φ‖_{L^∞([0,1];W^{m,∞}(𝕋))}`.

The Sobolev norm on the right is carried as a bound `K` on the `θ`-derivatives
of `φ` up to order `m`, uniformly in `σ ∈ [0,1]`: that is what
`‖φ‖_{L^∞([0,1];W^{m,∞}(𝕋))} ≤ K` says, and it avoids the junk value an
unbounded supremum would take.

Not proved here.

Source: arXiv:2605.09213v1, `th:mf-gpt`(ii), `eq:conv-rate-muNf`. -/
theorem mean_field_rate (lam β : ℝ) (f₀ : ℝ → ProbabilityMeasure Torus)
    (f : ℝ → ℝ → ProbabilityMeasure Torus) (hf : IsMeanFieldSolution lam β f₀ f) :
    ∃ C : ℝ, 0 < C ∧ ∀ (δ γ Cst : ℝ), 0 < δ →
      ∀ (P : Measure Ω), IsProbabilityMeasure P →
      ∀ (N : ℕ) (ϑ : Ω → ℝ → Idx N → ℝ),
        (∀ ω, IsGPTFlow lam β N (ϑ ω)) →
        iIndepFun (fun (j : Idx N) (ω : Ω) => ϑ ω 0 j) P →
        InitConv P N (fun ω => ϑ ω 0) f₀ δ γ Cst →
      ∀ (m : ℕ), (m : ℝ) > max γ 2 + 1 / 2 →
      ∀ (φ : ℝ × ℝ → ℝ) (hper : ∀ σ, Function.Periodic (fun u => φ (σ, u)) (2 * π)),
        ∀ K : ℝ, (∀ σ ∈ Set.Icc (0 : ℝ) 1, ∀ r ≤ m, ∀ u : ℝ,
            |iteratedDeriv r (fun u => φ (σ, u)) u| ≤ K) →
      ∀ t ∈ Set.Ici (0 : ℝ),
        Real.sqrt (∫ ω, ((N : ℝ)⁻¹ * ∑ j : Idx N, φ (((j : ℝ) + 1) / N, ϑ ω t j)
            - ∫ σ in (0 : ℝ)..1, ∫ x, (hper σ).lift x ∂(f t σ : Measure Torus)) ^ 2 ∂P)
          ≤ C * (N : ℝ) ^ (-(min δ (1 / 2))) * Real.exp (C * t) * K := by
  sorry

/-- The hypotheses of `mean_field_rate` are satisfiable.  Its one binder
hypothesis is that `f` solves `eq:mfl-lambda`, which is exactly what
`mean_field_limit` produces from an initial profile.  The conditions quantified
inside the statement are witnessed here in the regime they constrain: `δ > 0`,
an admissible Sobolev order `m = 3` for `γ = 0`, and a test function periodic
in `θ`. -/
example : (0 : ℝ) < 1 ∧ ((3 : ℕ) : ℝ) > max 0 2 + 1 / 2 ∧
    ∀ σ : ℝ, Function.Periodic (fun u => ((fun p : ℝ × ℝ => Real.cos p.2) (σ, u))) (2 * π) := by
  refine ⟨by norm_num, by norm_num, fun σ u => ?_⟩
  simp [Real.cos_add_two_pi]

end Kinetic
end Transformer
