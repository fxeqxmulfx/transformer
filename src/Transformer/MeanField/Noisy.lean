/-
# Mean-Field Dynamics — Noisy Transformers (§7 of 2512.01868v4)

* `eq: noisy.SDE`         — noisy Transformer SDE on the sphere, stated
                            pathwise for a fixed realization of the noise,
* `eq: Fokker`            — Fokker–Planck equation, in weak form.

`eq: McKV`, the McKean–Vlasov limit, is not stated: it is an SDE whose drift
depends on the law of its own solution, and laws of processes are not
modelled here.  `eq: Fokker` is the equation that law satisfies.

Two objects the statements need are carried as parameters rather than built:
the realization `W` of the driving Brownian motion, and the Laplace–Beltrami
operator `Δ` of the sphere.  Neither the Itô/Stratonovich correction that keeps
the solution on the sphere nor the construction of `Δ` is formalized here.

The pitchfork bifurcation at `κ = 2` of the homogeneous (`β = 0`) Kuramoto
limit is also not formalized: the stationary densities are `∝ e^{κ r cos θ}`
and the threshold comes from the self-consistency `r = I₁(κ r) / I₀(κ r)`,
which needs modified Bessel functions.
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS
import Transformer.Perspective.Section2_GradientFlow
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace MeanField

variable (d n : ℕ)

/-- The drift of `eq: noisy.SDE`:

  `b_i(X) = Proj_{X_i}( (1/n) Σ_j e^{β ⟨X_i, X_j⟩} X_j )`.

Source: arXiv:2512.01868v4, §7. -/
noncomputable def noisyDrift
    (β : ℝ) (X : SphereTuple d n) (i : Idx n) : EucSpace d :=
  proj d ((X i : EucSpace d))
    (((n : ℝ)⁻¹) • ∑ j : Idx n,
      Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))
        • (X j : EucSpace d))

/-- The drift is tangent to the sphere at `X_i`: `⟨X_i, b_i(X)⟩ = 0`.  This is
what makes `eq: noisy.SDE` an equation on `(𝕊^{d-1})^n` rather than on
`(ℝ^d)^n`. -/
theorem inner_noisyDrift_eq_zero (β : ℝ) (X : SphereTuple d n) (i : Idx n) :
    inner (𝕜 := ℝ) ((X i : EucSpace d)) (noisyDrift d n β X i) = 0 := by
  have hx : ‖(X i : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp (X i).2
  exact inner_proj_eq_zero hx _

/-- **Equation (eq: noisy.SDE).**  Noisy Transformer SDE on the sphere:

  `dX_i(t) = Proj_{X_i}( (1/n) Σ_j e^{β ⟨X_i, X_j⟩} X_j ) dt + √(2/κ) dW_i(t)`.

Stated pathwise, as the integral equation the display abbreviates, for a fixed
realization `W` of the driving noise: the stochastic integral of the constant
diffusion coefficient `√(2/κ)` is `√(2/κ) W_i(t)`.  The correction term that
distinguishes the Itô and Stratonovich readings on the sphere — the one that
keeps `X_i(t)` on `𝕊^{d-1}` — is not formalized.
Source: arXiv:2512.01868v4, §7. -/
def noisyTransformerSDE
    (β κ : ℝ) (X : ℝ → SphereTuple d n) (W : ℝ → Idx n → EucSpace d) : Prop :=
  ∀ (t : ℝ) (i : Idx n),
    (X t i : EucSpace d)
      = (X 0 i : EucSpace d)
        + (∫ s in (0 : ℝ)..t, noisyDrift d n β (X s) i)
        + Real.sqrt (2 / κ) • (W t i - W 0 i)

/-- **Equation (eq: Fokker).**  Fokker–Planck equation of `eq: McKV`, in weak
form: for every `C²` test function `φ`,

  `d/dt ∫ φ dμ_t = ∫ ( κ⁻¹ Δφ + ⟨∇φ, v_t⟩ ) dμ_t`,
  `v_t(x) = Proj_x ∫ e^{β⟨x,y⟩} y dμ_t(y)`,

which is `∂_t μ_t = κ⁻¹ Δ μ_t - ∇ · (μ_t v_t)`.  The drift `v_t` is the
`USA` field of `eq:continuity` (`Perspective.usaVectorField`), unnormalized,
as in `eq: McKV`.

**What the source says and what is changed here.**  The display prints

  `∂_t μ_t + κ⁻¹ Δ μ_t = ∇ · ( μ_t ∫ e^{β⟨·,y⟩} y μ_t(dy) )`,

with both signs reversed.  The sentence after it says the equation "reduces
to `eq:continuity` as `κ → ∞`", and `eq:continuity` is
`∂_t μ_t + ∇ · (μ_t v_t) = 0`, the opposite sign of the transport term; and
`+κ⁻¹Δμ_t` on the left is the backward heat equation, not the law of the
forward diffusion `√(2κ⁻¹) dW`.  The equation stated is the one the text
describes: the law of `eq: McKV`, reducing to `eq:continuity`.  The printed
drift also omits `Proj_x`; in weak form the drift is paired with the
gradient of a test function along the sphere, which is tangent, so only the
tangential part of the drift enters and the two agree.

`Δ` is the Laplace–Beltrami operator of the sphere, carried as a parameter:
this development does not construct it.
Source: arXiv:2512.01868v4, §7, `eq:Fokker`. -/
def fokkerPlanck
    (β κ : ℝ) (Δ : (EucSpace d → ℝ) → EucSpace d → ℝ)
    (μ : ℝ → Perspective.ProbSphere d) : Prop :=
  ∀ φ : EucSpace d → ℝ, ContDiff ℝ 2 φ → ∀ t : ℝ,
    HasDerivAt (fun s => ∫ x, φ (x : EucSpace d) ∂(μ s : Measure (SSphere d)))
      (∫ x, (κ⁻¹ * Δ φ (x : EucSpace d)
          + inner (𝕜 := ℝ) (gradient φ (x : EucSpace d))
              (Perspective.usaVectorField d β (μ t) (x : EucSpace d)))
        ∂(μ t : Measure (SSphere d))) t

end MeanField
end Transformer
