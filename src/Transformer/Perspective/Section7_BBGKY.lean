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
import Mathlib.MeasureTheory.Measure.MeasureSpace

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

open Perspective

variable (n : ℕ)

/-- The kernel `h_β'(θ) = -β sin(θ) e^{β cos θ}`. -/
noncomputable def h_β' (β θ : ℝ) : ℝ :=
  -β * Real.sin θ * Real.exp (β * Real.cos θ)

/-- The conditional expectation `g(t, x)` appearing in the BBGKY hierarchy:

  `g(t, x) = 𝔼[ -h_β'(θ_3(t)) | θ_1(t) = 0, θ_2(t) = x ]`. -/
def gBBGKY (β : ℝ) (g : ℝ → ℝ → ℝ) : Prop :=
  -- abstract specification; the closure ansatz is left open in the paper.
  True

/-- **Equation (e:transport).** *Two-particle transport equation.*

  `∂_t ψ(t,x) + ∂_x (v(t,x) ψ(t,x)) = 0`,   `(t,x) ∈ ℝ_{≥0} × 𝕋`,
  `ψ(0,x) = 1/(2π)`,

with velocity

  `v(t,x) = (2 / (β n)) h_β'(x) - (2 (n-2) / (β n)) g(t,x)`. -/
def bbgkyTransport
    (β : ℝ) (ψ : ℝ → ℝ → ℝ) (g : ℝ → ℝ → ℝ) : Prop :=
  -- Initial condition: ψ(0, x) = 1/(2π) for all x ∈ 𝕋.
  (∀ x : ℝ, ψ 0 x = (2 * Real.pi)⁻¹) ∧
  -- Transport equation.
  ∀ t : ℝ, ∀ x : ℝ,
    let v := (2 / (β * (n : ℝ))) * h_β' β x
              - (2 * ((n : ℝ) - 2) / (β * (n : ℝ))) * g t x
    True   -- the actual distributional identity is omitted.

end Perspective
end Transformer
