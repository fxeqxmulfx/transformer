/-
# §8 — BBGKY hierarchy

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes §8 of the survey, which considers a hierarchy of
correlation functions for the dynamics on the circle.  Main object:

* `e:transport` — closed-up-to-`g(t,x)` transport equation for the
  two-particle correlation `ψ(t, ·) : 𝕋 → ℝ_{≥0}`.
-/

import Transformer.Basic
import Transformer.Perspective.Section6_Circle
import Mathlib.MeasureTheory.Measure.MeasureSpaceDef
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

open Perspective

variable (n : ℕ)

/-- The kernel `h_β'(θ) = -β sin(θ) e^{β cos θ}`. -/
noncomputable def h_β' (β θ : ℝ) : ℝ :=
  -β * Real.sin θ * Real.exp (β * Real.cos θ)

/-- **The conditional expectation `g(t, x)` of the hierarchy.**

  `g(t, x) = 𝔼[ -h_β'(θ_3(t)) | θ_1(t) = 0, θ_2(t) = x ]`.

Written against the densities it is a conditional expectation of: with `ψ₂` the
two-particle and `ψ₃` the three-particle correlation, the conditioning is the
identity

  `g(t,x) ψ₂(t,x) = ∫_𝕋 -h_β'(y) ψ₃(t,x,y) dy`.

This is the place where the hierarchy fails to close: `ψ₂` is determined by
`ψ₃`, and the paper leaves the closure ansatz open. -/
def gBBGKY (β : ℝ) (ψ₂ : ℝ → ℝ → ℝ) (ψ₃ : ℝ → ℝ → ℝ → ℝ) (g : ℝ → ℝ → ℝ) : Prop :=
  ∀ t x : ℝ, g t x * ψ₂ t x = ∫ y in (0 : ℝ)..(2 * Real.pi), -h_β' β y * ψ₃ t x y

/-- The transport velocity of `e:transport`:

  `v(t,x) = (2 / (β n)) h_β'(x) - (2 (n-2) / (β n)) g(t,x)`.

The first term is the pair's own interaction, the second is everything the
other `n - 2` particles do, which is where `g` enters. -/
noncomputable def bbgkyVelocity (β : ℝ) (g : ℝ → ℝ → ℝ) (t x : ℝ) : ℝ :=
  (2 / (β * (n : ℝ))) * h_β' β x - (2 * ((n : ℝ) - 2) / (β * (n : ℝ))) * g t x

/-- **Equation (e:transport).** *Two-particle transport equation.*

  `∂_t ψ(t,x) + ∂_x (v(t,x) ψ(t,x)) = 0`,   `(t,x) ∈ ℝ_{≥0} × 𝕋`,
  `ψ(0,x) = 1/(2π)`,

with `v = bbgkyVelocity`.  The `ψ` of the hierarchy is an honest function, so
the equation is stated strongly: the time derivative of `ψ(·,x)` is minus the
space derivative of the flux `v ψ` at `x`. -/
def bbgkyTransport
    (β : ℝ) (ψ : ℝ → ℝ → ℝ) (g : ℝ → ℝ → ℝ) : Prop :=
  -- Initial condition: ψ(0, x) = 1/(2π) for all x ∈ 𝕋.
  (∀ x : ℝ, ψ 0 x = (2 * Real.pi)⁻¹) ∧
  -- Transport equation: `∂_t ψ = -∂_x (v ψ)`.
  ∀ t x : ℝ,
    HasDerivAt (fun s => ψ s x)
      (-deriv (fun y => bbgkyVelocity n β g t y * ψ t y) x) t

end Perspective
end Transformer
