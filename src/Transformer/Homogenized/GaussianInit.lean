/-
# Homogenized Transformers — Gaussian initialization

Formalization of `eq: tformers.at.initialization` (assumption (G)), its two
consequences, and `cor:weak_error_centered` of arXiv:2604.01978v1,
*Homogenized Transformers*.

At Gaussian initialization the value matrix is centered, so the drift
`b_{ρ*}[μ]` of `eq: first.sde` vanishes identically and the limiting dynamics
is driven entirely by the fluctuation term; and the variance proxy is
computable, `ς² = σ_V²(d-1)`, so `α = (η/H) σ_V²(d-1)`.  The error bound then
holds up to a macroscopic time of order `1/α ≫ 1`, with `e^{C t_L}` improved
to `e^{C t_L α}` and `max(1,α)` to `max(η,α)`.

The limiting object of `cor:weak_error_centered` is `eq:Diffusive_gaussian_case`,
which is `eq: first.sde` read under `b_{ρ*} ≡ 0`; it is therefore carried here
as `IsFirstSde`, and `bField_gaussian` is what identifies the two.  Both
consequences are proved downstream, where the facts about the head law they
rest on live: (i), `ς² = σ_V²(d-1)`, in `Homogenized.GaussianVariance`
(`varianceProxy_gaussian`, `alphaOf_gaussian`), and (ii), `b_{ρ*}[μ] ≡ 0`, in
`Homogenized.GaussianDrift` (`meanFieldOf_gaussian`, `bField_gaussian`).

Source: arXiv:2604.01978v1, §2.3.3, `eq: tformers.at.initialization`,
`cor:weak_error_centered`.
-/

import Transformer.Homogenized.WeakError
import Mathlib.Probability.Distributions.Gaussian.Real

open scoped BigOperators NNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

/-- **Assumption (G), `eq: tformers.at.initialization`**: a Transformer at
Gaussian initialization.  `V_{ij} ∼ 𝒩(0, σ_V²)` i.i.d., `A = W W'ᵀ` with
`W_{ij}, W'_{ij} ∼ 𝒩(0, σ_A²)` i.i.d., all entries independent.

Source: arXiv:2604.01978v1, `eq: tformers.at.initialization`. -/
def IsGaussianHeadLaw (d : ℕ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)) : Prop :=
  ∃ (Ω : Type) (_ : MeasurableSpace Ω) (P : Measure Ω)
    (Vr Wr Wr' : Ω → Matrix (Fin d) (Fin d) ℝ),
    IsProbabilityMeasure P ∧ Measurable Vr ∧ Measurable Wr ∧ Measurable Wr' ∧
      iIndepFun
        (fun p : Fin 3 × Fin d × Fin d => fun ω =>
          if p.1 = 0 then Vr ω p.2.1 p.2.2
          else if p.1 = 1 then Wr ω p.2.1 p.2.2 else Wr' ω p.2.1 p.2.2) P ∧
      (∀ i j, Measure.map (fun ω => Vr ω i j) P = gaussianReal 0 (σV ^ 2)) ∧
      (∀ i j, Measure.map (fun ω => Wr ω i j) P = gaussianReal 0 (σA ^ 2)) ∧
      (∀ i j, Measure.map (fun ω => Wr' ω i j) P = gaussianReal 0 (σA ^ 2)) ∧
      Measure.map (fun ω => ((Vr ω, Wr ω * (Wr' ω).transpose) : HeadParam d)) P = ρ

/-- **Equation (eq:Diffusive_gaussian_case).**  The homogenized model at
Gaussian initialization,

  `dx_i = √α 𝐏_{x_i} ∫ B_θ[μ](x_i) W(dθ,dt) - (α/2) x_i ∫ ‖𝐏_{x_i} B_θ[μ](x_i)‖² ρ*(dθ)`,

which is `eq: first.sde` read under `b_{ρ*} ≡ 0` — the centered fluctuation
`ξ_θ` is then `B_θ` itself, by `bField_gaussian`.  The source rescales time to
set `α = 1` and writes the noise without the `1/ς` normalization, so this is
`IsFirstSde` at `α = ς = 1`; the Itô correction is the one the intrinsic
formulation of `Transformer.Homogenized.Generator` carries.

Source: arXiv:2604.01978v1, `eq:Diffusive_gaussian_case`. -/
def IsDiffusiveSde {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    (X : ℝ → Ω → (Idx n → EucSpace d)) : Prop :=
  IsFirstSde β 1 1 ρ P X

/-- `IsGaussianHeadLaw` is satisfiable: at `σ_V = σ_A = 0` every Gaussian is a
Dirac mass at `0`, and the law of the head is `δ_0`. -/
theorem isGaussianHeadLaw_dirac_zero (d : ℕ) :
    IsGaussianHeadLaw d 0 0 (Measure.dirac (0 : HeadParam d)) := by
  refine ⟨Unit, inferInstance, Measure.dirac (), 0, 0, 0, inferInstance,
    measurable_const, measurable_const, measurable_const, iIndepFun_of_unit _,
    ?_, ?_, ?_, ?_⟩
  · intro i j; simp [gaussianReal_zero_var]
  · intro i j; simp [gaussianReal_zero_var]
  · intro i j; simp [gaussianReal_zero_var]
  · simp
    rfl

/-- **Corollary (cor:weak_error_centered).**  Under
`eq: tformers.at.initialization`, for any `φ ∈ C⁴((𝕊^{d-1})^n)`,

  `sup_{t ∈ [0,t_L]} |𝔼φ(X(t)) - 𝔼φ(X^η(t))| ≤ C e^{C t_L α} η (t_L+1) max(η,α)`,

with `C ≥ 1` depending on `‖φ‖_{C⁴}` but not on `η, α, L`.  The exponential now
carries `t_L α`, so the approximation survives to a macroscopic time of order
`1/α`.

The same reading of the initial condition, and the same positivity of the
number of heads, as in `weak_error_clean` are written into the statement.

Not proved here.

Source: arXiv:2604.01978v1, `cor:weak_error_centered`. -/
theorem weak_error_centered {d n H : ℕ} (hH : 0 < H) (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ)
    (φ : (Idx n → EucSpace d) → ℝ) (hφ : ContDiff ℝ 4 φ) :
    ∃ C : ℝ, 1 ≤ C ∧
      ∀ (η s : ℝ), 0 < η → IsVarianceProxy d n β ρ s →
      ∀ (L : ℕ) (x₀ : Idx n → EucSpace d), (∀ i, ‖x₀ i‖ = 1) →
      ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)
        (Θ : ℕ → Idx H → Ω → HeadParam d) (Xd : Ω → ℕ → Idx n → EucSpace d),
        IsRandomChain η β ρ P Θ Xd x₀ →
      ∀ (Ω' : Type) [MeasurableSpace Ω'] (P' : Measure Ω')
        (X : ℝ → Ω' → (Idx n → EucSpace d)),
        IsFirstSde β (alphaOf η s H) s ρ P' X → (∀ ω', X 0 ω' = x₀) →
      ∀ t ∈ Set.Icc (0 : ℝ) (η * L),
        |(∫ ω', φ (X t ω') ∂P') - ∫ ω, φ (interpChain η (Xd ω) t) ∂P| ≤
          C * Real.exp (C * (η * L) * alphaOf η s H) * η * (η * L + 1) *
            max η (alphaOf η s H) := by
  sorry

/-- The hypotheses of `weak_error_centered` are satisfiable. -/
example (d n : ℕ) :
    0 < 1 ∧ IsGaussianHeadLaw d 0 0 (Measure.dirac (0 : HeadParam d)) ∧
      ContDiff ℝ 4 (fun _ : Idx n → EucSpace d => (0 : ℝ)) :=
  ⟨Nat.one_pos, isGaussianHeadLaw_dirac_zero d, contDiff_const⟩

/-- The degenerate solution witnesses `IsDiffusiveSde`. -/
theorem isDiffusiveSde_dirac_zero {d n : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (β : ℝ) (x : Idx n → EucSpace d) (hx : ∀ i, ‖x i‖ = 1) :
    IsDiffusiveSde β (Measure.dirac (0 : HeadParam d)) P (fun _ _ => x) :=
  isItoSolution_dirac_zero P β 1 1 x hx

/-- The hypothesis of `isDiffusiveSde_dirac_zero` is satisfiable. -/
example (d : ℕ) : ∀ _i : Idx 1, ‖((basePoint d : SSphere (d + 1)) : EucSpace (d + 1))‖ = 1 :=
  fun _ => by simp [basePoint, PiLp.norm_single]

end Homogenized
end Transformer
