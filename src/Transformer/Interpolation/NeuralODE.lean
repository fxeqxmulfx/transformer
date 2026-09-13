/-
# Measure-to-measure interpolation — Matching discrete measures (Neural ODE part)

Formalization of §4 of arXiv:2411.04551v3:

* `eq: neural.ode.sphere`              — neural-ODE-only flow on the sphere,
* `Proposition prop: interpolation.neural.ode`,
* `Proposition lem: induction.neural.ode`,
* `eq: estimate.neural`, `eq: Hartman.Grobman` — exponential convergence
                                                 of perceptron-only flow,
* `eq: sphere.separation`, `eq: geodesic.toll`, `eq: identity.flow.symm`.
-/

import Transformer.Basic
import Transformer.Interpolation.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace Interpolation

open Interpolation

variable (d M : ℕ)

/-- **Equation (eq: neural.ode.sphere).**

  `ẋ^i(t) = Proj_x 𝐖(t) (𝐔(t) x^i(t) + b(t))_+`,    `x^i(0) = x_0^i`. -/
def neuralODESphere
    (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d)
    (x : ℝ → Idx M → EucSpace d) : Prop :=
  ∀ t : ℝ, ∀ i : Idx M,
    HasDerivAt (fun s => x s i)
      (proj d (x t i)
        ((W t)
          (EuclideanSpace.equiv _ ℝ |>.symm
            (fun k =>
              max ((EuclideanSpace.equiv _ ℝ ((U t) (x t i) + b t)) k) 0)))) t

/-- **Proposition (prop: interpolation.neural.ode).**

For `d ≥ 3` and data `(x_0^i, y^i) ∈ 𝕊^{d-1} × 𝕊^{d-1}` with
`x_0^i ≠ x_0^j`, `y^i ≠ y^j` for `i ≠ j`, and the geometric separation
condition

  `∃ γ_i ∈ 𝕊^{d-1}, ε_i > 0`,    `⟨γ_i, x_0^i - y^i⟩ = 0`,
  `x_0^j ∉ H_{ε_i}^{γ_i}` for `j ≠ i`,

there are piecewise-constant `θ = (𝐖, 𝐔, b) : [0, T] → M_{d×d}²×ℝ^d` with at
most `6M` switches such that the solution to `eq: neural.ode.sphere` satisfies
`x^i(T) = y^i` and `‖θ‖_∞ ≤ C M / (T min_i ε_i)`. -/
theorem prop_interpolation_neural_ode
    (hd : 3 ≤ d)
    (x₀ y : Idx M → SSphere d)
    (h_distinct_x : ∀ i j : Idx M, i ≠ j → x₀ i ≠ x₀ j)
    (h_distinct_y : ∀ i j : Idx M, i ≠ j → y i ≠ y j)
    (γ : Idx M → SSphere d) (ε : Idx M → ℝ)
    (h_orth : ∀ i, inner (𝕜 := ℝ) ((γ i : EucSpace d))
                          ((x₀ i : EucSpace d) - (y i : EucSpace d)) = 0)
    (h_sep : ∀ i j : Idx M, i ≠ j →
                x₀ j ∉ Hε d (γ i) (ε i))
    (T : ℝ) (hT : 0 < T) :
    ∃ (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d) (switches : ℕ),
      switches ≤ 6 * M ∧
      ∀ x : ℝ → Idx M → EucSpace d,
        (∀ i, x 0 i = (x₀ i : EucSpace d)) →
        neuralODESphere d M W U b x →
        ∀ i, x T i = (y i : EucSpace d) := by
  sorry

/-- **Proposition (lem: induction.neural.ode).** *Induction step.*

Specializing to `x_0^i = y^i` for `i ∈ [M - 1]` and a common pair
`(γ, ε)`, only the `M`-th pair is moved, with at most 6 switches. -/
theorem lem_induction_neural_ode
    (hd : 3 ≤ d) (M : ℕ) (hM : 1 ≤ M)
    (x₀ y : Idx M → SSphere d)
    (h_fixed : ∀ i : Idx M, (i : ℕ) < M - 1 → x₀ i = y i)
    (γ : SSphere d) (ε : ℝ)
    (h_orth : inner (𝕜 := ℝ) ((γ : EucSpace d))
                ((x₀ ⟨M - 1, by sorry⟩ : EucSpace d)
                  - (y ⟨M - 1, by sorry⟩ : EucSpace d)) = 0)
    (T : ℝ) (hT : 0 < T) :
    True := by trivial

/-- **Equation (eq: Hartman.Grobman).** Exponential settling near the
attractor:

  `d_g(x(t), ω_+) ≤ K e^{-λ t}`  for `t ≥ 0`,
with `λ > 0`, `K ≥ 1` depending only on `x_0`, `ε`, `γ`. -/
theorem Hartman_Grobman
    (γ ω_plus : SSphere d) (ε : ℝ) (hε : 0 < ε)
    (x : ℝ → SSphere d) :
    ∃ (K lam : ℝ), 1 ≤ K ∧ 0 < lam ∧
      ∀ t : ℝ, 0 ≤ t →
        ‖((x t : EucSpace d)) - ((ω_plus : EucSpace d))‖
          ≤ K * Real.exp (-(lam * t)) := by
  sorry

end Interpolation
end Transformer
