/-
# Measure-to-measure interpolation — Basic definitions

Formalization of arXiv:2411.04551v3 — "Measure-to-measure interpolation
using Transformers" (Geshkovski, Karagodin, Polyanskiy, Rigollet).

This file collects:

* `eq: ips`                 — interacting-particle system on `𝕊^{d-1}`,
* `eq: vf`                  — full vector field with the attention `𝒜_𝐁`
                              and perceptron components,
* `eq: cauchy.pb`           — Cauchy problem for the continuity equation,
* `eq: average.vf`          — simplified vector field used in
                              `prop: separation`,
* parameter type `Θ = (𝐕, 𝐁, 𝐖, 𝐔, b)`,
* the flow map `Φ^t_θ : 𝒫(𝕊^{d-1}) → 𝒫(𝕊^{d-1})`,
* the hyperplane / spherical-cap shorthands `H_ε^γ` and `𝒮_q(ε)`.
-/

import Transformer.Basic
import Transformer.Perspective.Section2_FlowMap
import Mathlib.MeasureTheory.Measure.MeasureSpaceDef
import Mathlib.Analysis.Calculus.Gradient.Basic

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Interpolation

variable (d : ℕ)

/-- The parameter space `Θ = M_{d×d}(ℝ)^4 × ℝ^d` of all admissible parameter
tuples `(𝐕, 𝐁, 𝐖, 𝐔, b)`. -/
structure Params (d : ℕ) where
  V : ParamMatrix d
  B : ParamMatrix d
  W : ParamMatrix d
  U : ParamMatrix d
  b : EucSpace d

/-- A time-dependent parameter is a curve in `Params d`. -/
abbrev TimeParams (d : ℕ) : Type := ℝ → Params d

/-- **Equation (eq: vf).** The full vector field driving the Transformer
particle system on `𝕊^{d-1}`:

  `𝐯[μ](t, x) = Proj_x (𝐕(t) · 𝒜_𝐁[μ](t, x) + 𝐖(t) · (𝐔(t) x + b(t))_+)`. -/
noncomputable def fullVF
    (θ : TimeParams d) (μ : Perspective.ProbSphere d)
    (t : ℝ) (x : EucSpace d) : EucSpace d :=
  let Bt := (θ t).B
  let Vt := (θ t).V
  let Wt := (θ t).W
  let Ut := (θ t).U
  let bt := (θ t).b
  let num : EucSpace d :=
    ∫ x', Real.exp (inner (𝕜 := ℝ) (Bt x) ((x' : EucSpace d)))
            • (x' : EucSpace d) ∂(μ : Measure (SSphere d))
  let den : ℝ :=
    ∫ ζ, Real.exp (inner (𝕜 := ℝ) (Bt x) ((ζ : EucSpace d)))
      ∂(μ : Measure (SSphere d))
  let attn : EucSpace d := den⁻¹ • num
  proj d x (Vt attn +
    Wt (EuclideanSpace.equiv _ ℝ |>.symm
        (fun i =>
          max ((EuclideanSpace.equiv _ ℝ (Ut x + bt)) i) 0)))

/-- **Equation (eq: average.vf).**  Simplified vector field with
`𝐁 ≡ 0` (so attention is just the barycenter `𝔼_μ[z]`):

  `𝐯[μ](t, x) = Proj_x (𝐕(t) · 𝔼_μ[z] + 𝐖(t) · (𝐔(t) x + b(t))_+)`. -/
noncomputable def averageVF
    (θ : TimeParams d) (μ : Perspective.ProbSphere d)
    (t : ℝ) (x : EucSpace d) : EucSpace d :=
  let Vt := (θ t).V
  let Wt := (θ t).W
  let Ut := (θ t).U
  let bt := (θ t).b
  let barycenter : EucSpace d :=
    ∫ x', (x' : EucSpace d) ∂(μ : Measure (SSphere d))
  proj d x (Vt barycenter +
    Wt (EuclideanSpace.equiv _ ℝ |>.symm
        (fun i =>
          max ((EuclideanSpace.equiv _ ℝ (Ut x + bt)) i) 0)))

/-- **Equation (eq: cauchy.pb).** Continuity equation on the sphere,

  `∂_t μ(t) + div(μ(t) · 𝐯[μ(t)]) = 0`,

in distributional form: for every `C¹` test function `φ` on the ambient space,

  `d/dt ∫ φ dμ(t) = ∫ ⟨∇φ(x), 𝐯[μ(t)](t,x)⟩ dμ(t)(x)`.

The initial condition `μ(0) = μ_0` is imposed separately by the statements that
use this, since it is what varies between them.  As in
`Perspective.continuityEquation`, the ambient gradient is the right pairing
because `fullVF` ends in `proj`. -/
def cauchyPB
    (θ : TimeParams d) (μ : ℝ → Perspective.ProbSphere d) : Prop :=
  ∀ φ : EucSpace d → ℝ, ContDiff ℝ 1 φ → ∀ t : ℝ,
    HasDerivAt (fun s => ∫ x, φ (x : EucSpace d) ∂(μ s : Measure (SSphere d)))
      (∫ x, inner (𝕜 := ℝ) (gradient φ (x : EucSpace d))
          (fullVF d θ (μ t) t (x : EucSpace d))
        ∂(μ t : Measure (SSphere d))) t

/-- **Flow map associated with parameters `θ`** — the solution operator of
`eq: cauchy.pb`.

It is a property of a candidate `Φ`, not a construction: `Φ^t_θ μ₀` is the
solution of the Cauchy problem started at `μ₀`, and producing one is exactly
the well-posedness the paper assumes.  Stating it this way keeps that
assumption visible wherever a flow map is used. -/
def IsFlowMap
    (θ : TimeParams d) (Φ : ℝ → Perspective.ProbSphere d → Perspective.ProbSphere d) :
    Prop :=
  ∀ μ₀ : Perspective.ProbSphere d, Φ 0 μ₀ = μ₀ ∧ cauchyPB d θ (fun t => Φ t μ₀)

/-- The hyperplane `H_ε^γ = { x ∈ 𝕊^{d-1} : |⟨x, γ⟩| ≤ ε }`. -/
def Hε (γ : SSphere d) (ε : ℝ) : Set (SSphere d) :=
  { x | |inner (𝕜 := ℝ) ((x : EucSpace d)) ((γ : EucSpace d))| ≤ ε }

/-- The positive quadrant of `𝕊^{d-1}`:

  `ℚ_1^{d-1} = 𝕊^{d-1} ∩ (ℝ_{>0})^d`. -/
def positiveQuadrant : Set (SSphere d) :=
  { x | ∀ i : Fin d, 0 < (EuclideanSpace.equiv _ ℝ ((x : EucSpace d))) i }

end Interpolation
end Transformer
