/-
# §6.2-§6.3 — Quantitative convergence and the phase-transition curve

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

Continues `Perspective.Section5_HighD`:

* `Theorem thm: orthogonal`        — orthogonal-initial dynamics,
* `eq: ybeta`, `eq: ybetaUSA`      — the scalar ODE for the angle,
* `Theorem thm: phase.transition.curve` — `d ≫ n` quantitative result,
* `eq: upto-t`,
* `eq: gamma.infty`                — phase-transition curve.
-/

import Transformer.Perspective.Section5_HighD
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

open Perspective

variable (d n : ℕ)

/-! ### §6.2 — More precise quantitative convergence -/

/-- The scalar ODE driving the angle between pairwise orthogonal particles
under `SA`:

  `γ̇_β(t) = 2 e^{β γ_β(t)} (1 - γ_β(t)) ((n-1) γ_β(t) + 1)
             / (e^β + (n-1) e^{β γ_β(t)})`,
  `γ_β(0) = 0`.

This is **Equation (eq: ybeta).** -/
def ybetaODE_SA (n : ℕ) (β : ℝ) (γ : ℝ → ℝ) : Prop :=
  γ 0 = 0 ∧
  ∀ t : ℝ, HasDerivAt γ
    (2 * Real.exp (β * γ t) * (1 - γ t) * ((n - 1 : ℝ) * γ t + 1)
      / (Real.exp β + (n - 1 : ℝ) * Real.exp (β * γ t))) t

/-- The scalar ODE for `USA` (eq: ybetaUSA):

  `γ̇_β(t) = (2/n) e^{β γ_β(t)} (1 - γ_β(t)) ((n-1) γ_β(t) + 1)`. -/
def ybetaODE_USA (n : ℕ) (β : ℝ) (γ : ℝ → ℝ) : Prop :=
  γ 0 = 0 ∧
  ∀ t : ℝ, HasDerivAt γ
    ((2 / (n : ℝ)) * Real.exp (β * γ t) * (1 - γ t) * ((n - 1 : ℝ) * γ t + 1)) t

/-- **A solution of `eq: ybeta`.**  At `n = 1` and `β = 0` the equation is
`γ̇ = 2(1 - γ)`, `γ(0) = 0`, whose solution is `γ(t) = 1 - e^{-2t}`: the angle
closes at an exponential rate.

It is the one solution of `eq: ybeta` available in closed form, and it is what
witnesses that the hypothesis `ybetaODE_SA` of the estimates below is
satisfiable. -/
theorem ybetaODE_SA_one_zero :
    ybetaODE_SA 1 0 (fun t => 1 - Real.exp (-2 * t)) := by
  refine ⟨by simp, fun t => ?_⟩
  have hlin : HasDerivAt (fun s : ℝ => -2 * s) (-2 : ℝ) t := by
    simpa using HasDerivAt.const_mul (-2 : ℝ) (hasDerivAt_id t)
  have h : HasDerivAt (fun s : ℝ => 1 - Real.exp (-2 * s))
      (-(Real.exp (-2 * t) * -2)) t := hlin.exp.const_sub 1
  refine h.congr_deriv ?_
  norm_num
  ring

/-- **The derivative of `tanh`,** `tanh' = 1 - tanh²`.

Not in Mathlib, and what makes `tanh` the solution of both scalar ODEs at
`n = 2`, `β = 0`: there the right-hand side of `eq: ybeta` and of
`eq: ybetaUSA` both collapse to `(1 - γ)(1 + γ)`. -/
theorem hasDerivAt_tanh (t : ℝ) : HasDerivAt Real.tanh (1 - Real.tanh t ^ 2) t := by
  have hcosh : Real.cosh t ≠ 0 := ne_of_gt (Real.cosh_pos t)
  have hfun : Real.tanh = fun s : ℝ => Real.sinh s / Real.cosh s :=
    funext Real.tanh_eq_sinh_div_cosh
  have htanh : HasDerivAt Real.tanh
      ((Real.cosh t * Real.cosh t - Real.sinh t * Real.sinh t)
        / Real.cosh t ^ 2) t := by
    rw [hfun]
    exact (Real.hasDerivAt_sinh t).div (Real.hasDerivAt_cosh t) hcosh
  refine htanh.congr_deriv ?_
  rw [Real.tanh_eq_sinh_div_cosh]
  field_simp

/-- **A second solution of `eq: ybeta`.**  At `n = 2` and `β = 0` the equation
reads `γ̇ = (1 - γ)(1 + γ) = 1 - γ²`, `γ(0) = 0`, whose solution is `tanh`.

Unlike `ybetaODE_SA_one_zero` this one has two particles, so the pair estimates
of `Perspective.AppendixD_Assembly`, which say nothing when `i ≠ j` is
unsatisfiable, are witnessed non-vacuously by it. -/
theorem ybetaODE_SA_two_zero : ybetaODE_SA 2 0 Real.tanh := by
  refine ⟨Real.tanh_zero, fun t => ?_⟩
  refine (hasDerivAt_tanh t).congr_deriv ?_
  norm_num
  ring

/-- **A solution of `eq: ybetaUSA`.**  At `n = 2` and `β = 0` the `USA`
equation is `γ̇ = (2/2)(1 - γ)(1 + γ)`, the same `1 - γ²` as `eq: ybeta` at
those parameters, so `tanh` solves it too.

This is what witnesses that the hypothesis `ybetaODE_USA` of
`Perspective.usa_analogue` is satisfiable at the `n ≥ 2` the estimate needs. -/
theorem ybetaODE_USA_two_zero : ybetaODE_USA 2 0 Real.tanh := by
  refine ⟨Real.tanh_zero, fun t => ?_⟩
  refine (hasDerivAt_tanh t).congr_deriv ?_
  norm_num
  ring

/-- **Theorem (thm: orthogonal).** *Orthogonal initial sequence.*

Let `β ≥ 0`, `d, n ≥ 2`.  If `(x_i(0))_{i ∈ [n]}` are pairwise orthogonal on
`𝕊^{d-1}`, then the angle `θ(t) := ∠(x_i(t), x_j(t))` is the same for all
distinct `i, j`, and `γ_β(t) := cos θ(t)` satisfies `eq: ybeta` (for `SA`) or
`eq: ybetaUSA` (for `USA`). -/
theorem orthogonal_initial
    (β : ℝ) (hβ : 0 ≤ β) (hd : 2 ≤ d) (hn : 2 ≤ n)
    (X₀ : SphereTuple d n)
    (h_ortho : ∀ i j : Idx n, i ≠ j →
                inner (𝕜 := ℝ) ((X₀ i : EucSpace d)) ((X₀ j : EucSpace d)) = 0) :
    ∃ γ : ℝ → ℝ, ybetaODE_SA n β γ ∧
      ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → Perspective.SA d n β X →
        ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n, i ≠ j →
          inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d)) = γ t := by
  sorry

/-- **Theorem (thm: phase.transition.curve), eq: upto-t.**

For each `n ≥ 2` and `β ≥ 0`, there exists `d⋆(n, β) ≥ n` such that for all
`d ≥ d⋆(n, β)` and an i.i.d. uniform initial sequence `(x_i(0))_{i ∈ [n]}`,
the solution to the `SA` Cauchy problem satisfies, with probability at least
`1 - 2 n² d^{-1/64}`,

  `|⟨x_i(t), x_j(t)⟩ - γ_β(t)| ≤ min{ 2 c(β)^{n t} √(log d / d), C e^{-λ t} }`

for all `i ≠ j` and `t ≥ 0`, where `c(β) = e^{10 max(1,β)}` and `γ_β` is the
unique solution to `eq: ybeta`.

Appendix D assembles this from two halves, each covering one branch of the
minimum past the threshold `eq: d.large`: `e:ineqfirstpart` — the
`√(log d / d)` branch, from `Perspective.almost_orthogonal` — and
`e:ineqsecondpart` — the exponentially decaying branch, from
`Perspective.ineq_second_part`, which is proved from the two Grönwall estimates
it deduces it from, carried as hypotheses.  The estimates themselves are not
proved, nor is the first half, nor the assembly.

Not proved here.

Source: arXiv:2312.10794v5, §6.2, `thm: phase.transition.curve`, and
Appendix D for the assembly. -/
theorem phase_transition_curve
    (β : ℝ) (hβ : 0 ≤ β) (hn : 2 ≤ n) :
    ∃ d_star : ℕ, n ≤ d_star ∧ ∀ d : ℕ, d_star ≤ d →
      ∃ (C lam : ℝ), 0 < C ∧ 0 < lam ∧
        -- with probability at least `1 - 2 n² d^{-1/64}` (under uniform init),
        ∀ (X₀ : SphereTuple d n),
          ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → Perspective.SA d n β X →
            ∀ γ : ℝ → ℝ, ybetaODE_SA n β γ →
              ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n, i ≠ j →
                |inner (𝕜 := ℝ)
                    ((X t i : EucSpace d)) ((X t j : EucSpace d)) - γ t|
                  ≤ min
                      (2 * (Real.exp (10 * max 1 β))^(n * t) *
                          Real.sqrt (Real.log d / d))
                      (C * Real.exp (-(lam * t))) := by
  sorry

/-! ### §6.3 — Phase transition curve -/

/-- The phase-transition curve `Γ_{d,δ}` in `(t, β)`-space: the frontier of
the region of times and temperatures at which the `SA` dynamics started from a
pairwise-orthogonal sequence of `n` points of `𝕊^{d-1}` has brought every
pairwise inner product to `1 - δ` or above.

`ΓInf` below is its `d → ∞` limit, where the pairwise inner product is the
solution `γ_β` of `eq: ybeta` and the condition becomes `γ_β(t) = 1 - δ`. -/
def Γ (d n : ℕ) (δ : ℝ) : Set (ℝ × ℝ) :=
  frontier { p : ℝ × ℝ | 0 ≤ p.1 ∧ 0 ≤ p.2 ∧
    ∀ X : ℝ → SphereTuple d n, Perspective.SA d n p.2 X →
      (∀ i j : Idx n, i ≠ j →
        inner (𝕜 := ℝ) ((X 0 i : EucSpace d)) ((X 0 j : EucSpace d)) = 0) →
      ∀ i j : Idx n,
        1 - δ ≤ inner (𝕜 := ℝ) ((X p.1 i : EucSpace d)) ((X p.1 j : EucSpace d)) }

/-- **Equation (eq: gamma.infty).** Limiting phase-transition curve:

  `Γ_{∞, δ} = {(t, β) ≥ 0 : γ_β(t) = 1 - δ}`. -/
def ΓInf (n : ℕ) (δ : ℝ) : Set (ℝ × ℝ) :=
  { p | ∃ γ : ℝ → ℝ, ybetaODE_SA n p.2 γ ∧ γ p.1 = 1 - δ }

end Perspective
end Transformer
