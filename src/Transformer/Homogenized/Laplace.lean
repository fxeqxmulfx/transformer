/-
# Homogenized Transformers — the barycenter at low temperature

Formalization of `lemma:Laplace_method` and `lem:delta_method` of
arXiv:2604.01978v1, *Homogenized Transformers*: the two estimates that make the
drift of the overlap collapse to the logistic one, and so carry the proof of
`thm:large_beta_meta`.

The first says that at large `β` the softmax barycenter `m_{β,A}[μ](x)` of
`Barycenter.lean` collapses onto the unit vector `A x / ‖A x‖` — the Laplace
method on the sphere — and the second expands the resulting
`E⟨Ax/‖Ax‖, Ay/‖Ay‖⟩` in `1/d` by the delta method.  Together they are
`eq:kernel_expansion`, which is a line of the proof of `thm:large_beta_meta`
and is not restated here.
-/

import Transformer.Homogenized.Barycenter
import Transformer.Homogenized.Metastability

open scoped BigOperators ENNReal NNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### The two estimates -/

/-- **Lemma (lemma:Laplace_method).**  For `μ` as in `ass:low-temperature`,

  `E⟨m_{β,A}[μ](x), m_{β,A}[μ](y)⟩ = E⟨Ax/‖Ax‖, Ay/‖Ay‖⟩ + O(d^{-3/2} + β^{-1/2})`

for all `x, y ∈ 𝕊^{d-1}`.

**What the source says and what is written here.**

* The `β` of the error is again the temperature of the normalized query-key
  matrix, as in `large_beta_metastability`: the proof's Laplace expansion is in
  `λ_x = β‖A x‖`, and at `A = W W'ᵀ` of `eq: tformers.at.initialization` the
  norm `‖A x‖` is of order `d σ_A²`, so `λ_x ≍ β d σ_A² = effBeta`.  Writing
  `β^{-1/2}` here would be the statement at `‖A x‖ ≍ 1`, which is not the
  ensemble the lemma is applied to.
* The lemma is stated at a Gaussian head law and not at an arbitrary one: the
  proof discards the event `{‖A x‖ < ε}` by a Gaussian concentration
  inequality, and that is where `e^{-c(ε)d} = O(d^{-3/2})` comes from.  At an
  arbitrary law the claim is false — `ρ* = δ_0` makes the right-hand side a
  junk value while the left-hand side is `⟨m_{β,0}[μ](x), m_{β,0}[μ](y)⟩`.
* `ass:low-temperature` is an assumption on a *path* `t ↦ μ(t)`; the lemma uses
  it at one time, so it is instantiated at the constant path.
* `ρ_min`, `ρ_max`, `L` come before `C`, which is how the source's "all
  constants here depend only on bounds on `ρ` and `∇ρ`" is said.

Not proved here.

Source: arXiv:2604.01978v1, `lemma:Laplace_method`. -/
theorem laplace_method (ρmin ρmax L : ℝ) :
    ∃ C : ℝ, 0 < C ∧
      ∀ (d : ℕ), 0 < d → ∀ β : ℝ, 0 < β → ∀ σV σA : ℝ≥0, 0 < σA →
      ∀ ρ : Measure (HeadParam d), IsGaussianHeadLaw d σV σA ρ →
      ∀ (σ μ : Measure (EucSpace d)) (dens : EucSpace d → ℝ),
        IsLowTemperature d σ (fun _ => μ) (fun _ => dens) ρmin ρmax L →
      ∀ x y : EucSpace d, ‖x‖ = 1 → ‖y‖ = 1 →
        |baryCorr β ρ μ x y
            - ∫ θ, inner (𝕜 := ℝ)
                (normalizeLayer (qkMap θ x)) (normalizeLayer (qkMap θ y)) ∂ρ|
          ≤ C * ((d : ℝ) ^ (-(3 : ℝ) / 2) + effBeta d β σA ^ (-(1 : ℝ) / 2)) := by
  sorry

/-- **Lemma (lem:delta_method).**  For `A = (dσ_A²)⁻¹ Wᵀ W'`,

  `E⟨Ax/‖Ax‖, Ay/‖Ay‖⟩ = ⟨x,y⟩ + (⟨x,y⟩³ - ⟨x,y⟩)/(2d) + O(d^{-3/2})`.

**What the source says and what is written here.**  The statement is written
at the head law's own `A = W W'ᵀ` of `eq: tformers.at.initialization` rather
than at the source's `Â`.  The two differ by the positive constant `(dσ_A²)⁻¹`,
which `normalizeLayer_smul` shows the left-hand side does not see, and by
transposing both factors, which is a relabeling of independent identically
distributed entries and so does not change the law of `A`.  Stating it at `Â`
would therefore be the same lemma written in a matrix the rest of the
development never forms.

Not proved here.

Source: arXiv:2604.01978v1, `lem:delta_method`. -/
theorem delta_method :
    ∃ C : ℝ, 0 < C ∧
      ∀ (d : ℕ), 0 < d → ∀ σV σA : ℝ≥0, 0 < σA →
      ∀ ρ : Measure (HeadParam d), IsGaussianHeadLaw d σV σA ρ →
      ∀ x y : EucSpace d, ‖x‖ = 1 → ‖y‖ = 1 →
        |(∫ θ, inner (𝕜 := ℝ)
              (normalizeLayer (qkMap θ x)) (normalizeLayer (qkMap θ y)) ∂ρ)
            - (inner (𝕜 := ℝ) x y
                + ((inner (𝕜 := ℝ) x y : ℝ) ^ 3 - inner (𝕜 := ℝ) x y) / (2 * (d : ℝ)))|
          ≤ C * (d : ℝ) ^ (-(3 : ℝ) / 2) := by
  sorry

/-- The hypotheses of `laplace_method` are satisfiable, at the constants
`ρ_min = ρ_max = 1`, `L = 0` its statement is quantified after: dimension `1`,
temperature `1`, the Gaussian head law at `σ_V² = 1/d`, and the uniform measure
with density `1`, tested at the unit vector against itself. -/
example :
    (0 : ℕ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ≥0) < 1 ∧
      IsGaussianHeadLaw 1 (stdSigmaV 1) 1 (gaussHeadLaw 1 (stdSigmaV 1) 1) ∧
      IsLowTemperature 1 (uniformAmbient 1) (fun _ => uniformAmbient 1)
        (fun _ => (fun _ => 1)) 1 1 0 ∧
      ‖EuclideanSpace.single (0 : Fin 1) (1 : ℝ)‖ = 1 :=
  ⟨one_pos, one_pos, one_pos, isGaussianHeadLaw_gaussHeadLaw 1 _ _,
    isLowTemperature_uniformAmbient one_pos, by simp [PiLp.norm_single]⟩

/-- The hypotheses of `delta_method` are satisfiable: the same Gaussian head
law, with no assumption on the token measure. -/
example :
    (0 : ℕ) < 1 ∧ (0 : ℝ≥0) < 1 ∧
      IsGaussianHeadLaw 1 (stdSigmaV 1) 1 (gaussHeadLaw 1 (stdSigmaV 1) 1) ∧
      ‖EuclideanSpace.single (0 : Fin 1) (1 : ℝ)‖ = 1 :=
  ⟨one_pos, one_pos, isGaussianHeadLaw_gaussHeadLaw 1 _ _, by simp [PiLp.norm_single]⟩

end Homogenized
end Transformer
