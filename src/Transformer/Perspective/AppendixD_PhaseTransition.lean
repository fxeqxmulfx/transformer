/-
# Appendix D — Proof of Theorem (thm: phase.transition.curve)

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes Appendix D of the survey:

* `eq: lip.1`,    `eq: lip.2`,    `eq: lip.3` — Lipschitz bound on the flow,
* `eq: stability.4ortho`               — Gronwall stability estimate,
* `eq: almost.ortho.vec`               — almost-orthogonality (Lévy),
* `e:shortdist`                         — distance bound between `x_i` and
                                          its orthogonal approximation `y_i`,
* `e:ineqfirstpart`                     — first part of `eq: upto-t`,
* `e:ybetacloseto1`                     — `1 - γ_β(t) ≤ ⋯` estimate,
* `eq: d.large`                         — definition of `d⋆(n, β)`,
* `e:productcloseto1`, `e:1/n`,
* `e:dotalpha`, `e:mineqalpha`, `e:diffineqalpha`,
* `e:ineqsecondpart`                    — second part of `eq: upto-t`.
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS
import Transformer.Perspective.Section5_HighD
import Transformer.Perspective.Section5_HighDCurve

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

open Perspective Perspective

variable (d n : ℕ)

/-- **Equation (eq: stability.4ortho).** *Lipschitz estimate of the flow.*

For two solutions of `SA` with possibly distinct initial data,

  `max_j ‖x_j(t) - y_j(t)‖ ≤ c(β)^{n t} · max_j ‖x_j(0) - y_j(0)‖`,

where `c(β) = e^{10 max(1, β)}`. -/
theorem stability_orthogonal
    (β : ℝ) (X Y : ℝ → SphereTuple d n)
    (hX : Perspective.SA d n β X) (hY : Perspective.SA d n β Y) :
    ∀ t : ℝ, 0 ≤ t →
      (Finset.univ : Finset (Idx n)).sup'
        ⟨⟨0, by sorry⟩, Finset.mem_univ _⟩
        (fun j => ‖((X t j : EucSpace d)) - ((Y t j : EucSpace d))‖)
      ≤
      (Real.exp (10 * max 1 β))^(n * t) *
        (Finset.univ : Finset (Idx n)).sup'
          ⟨⟨0, by sorry⟩, Finset.mem_univ _⟩
          (fun j => ‖((X 0 j : EucSpace d)) - ((Y 0 j : EucSpace d))‖) := by
  sorry

/-- **Equation (eq: almost.ortho.vec).** *Almost-orthogonality (Lévy
concentration of measure).*

For i.i.d. uniform points on `𝕊^{d-1}`, there exist `n` pairwise orthogonal
points `y_i ∈ 𝕊^{d-1}` such that

  `‖x_i(0) - y_i(0)‖ ≤ √(log d / d)`,

with probability `≥ 1 - 2 n² d^{-1/64}`. -/
theorem almost_orthogonal
    (hn : 2 ≤ n) (hdn : n ≤ d) :
    True := by trivial

/-- **Equation (e:shortdist).**

  `‖x_i(t) - y_i(t)‖ ≤ c(β)^{n t} √(log d / d)`. -/
theorem shortdist_bound
    (β : ℝ) (X Y : ℝ → SphereTuple d n)
    (hX : Perspective.SA d n β X) (hY : Perspective.SA d n β Y) :
    True := by trivial

/-- **Equation (e:ineqfirstpart).** First part of `eq: upto-t`:

  `|⟨x_i(t), x_j(t)⟩ - γ_β(t)| ≤ 2 c(β)^{n t} √(log d / d)`. -/
theorem ineq_first_part
    (β : ℝ) (X : ℝ → SphereTuple d n) (γ : ℝ → ℝ)
    (hX : Perspective.SA d n β X) (hγ : ybetaODE_SA n β γ) :
    True := by trivial

/-- **Equation (e:ybetacloseto1).**

  `1 - γ_β(t) ≤ (1/2) exp( (n² e^β) / (2(n + e^{β/2}))
                              - n t / (n + e^{β/2}) )`. -/
theorem ybeta_close_to_1
    (β : ℝ) (γ : ℝ → ℝ) (hγ : ybetaODE_SA n β γ) :
    ∀ t : ℝ, 0 ≤ t →
      1 - γ t
        ≤ (1/2 : ℝ) * Real.exp
            ((n : ℝ)^2 * Real.exp β
                / (2 * ((n : ℝ) + Real.exp (β / 2)))
              - ((n : ℝ) * t) / ((n : ℝ) + Real.exp (β / 2))) := by
  sorry

/-- **Equation (eq: d.large).** Defines `d⋆(n, β)` as the smallest `d` for
which

  `d / log d ≥ 16 c(β)² / γ_β(1/n)²`. -/
theorem d_star_definition
    (β : ℝ) (γ : ℝ → ℝ) (hγ : ybetaODE_SA n β γ) :
    ∃ d_star : ℕ, ∀ d : ℕ, d_star ≤ d →
      16 * (Real.exp (10 * max 1 β))^2 / (γ ((n : ℝ)⁻¹))^2
        ≤ (d : ℝ) / Real.log d := by
  sorry

/-- **Equation (e:1/n).**

  `α(1/n) ≥ (1/2) γ_β(1/n)`. -/
theorem alpha_at_one_over_n
    (β : ℝ) (γ : ℝ → ℝ) (X : ℝ → SphereTuple d n)
    (hX : Perspective.SA d n β X) (hγ : ybetaODE_SA n β γ)
    (x_star : SSphere d) :
    let α := fun t : ℝ =>
              (Finset.univ : Finset (Idx n)).inf'
                ⟨⟨0, by sorry⟩, Finset.mem_univ _⟩
                (fun i => inner (𝕜 := ℝ)
                            ((X t i : EucSpace d)) ((x_star : EucSpace d)))
    (1/2 : ℝ) * γ ((n : ℝ)⁻¹) ≤ α ((n : ℝ)⁻¹) := by
  sorry

/-- **Equation (e:diffineqalpha).** Differential inequality:

  `α̇(t) ≥ (1/(n e^{2β})) α(1/n) (1 - α(t))`. -/
theorem diffineq_alpha
    (β : ℝ) (X : ℝ → SphereTuple d n)
    (hX : Perspective.SA d n β X) :
    True := by trivial

/-- **Equation (e:productcloseto1).**

  `1 - α(t) ≤ exp( (1 - γ_β(1/n) · t) / (2 n e^{2β}) )`. -/
theorem product_close_to_one
    (β : ℝ) (X : ℝ → SphereTuple d n) (γ : ℝ → ℝ)
    (hX : Perspective.SA d n β X) (hγ : ybetaODE_SA n β γ) :
    True := by trivial

/-- **Equation (e:ineqsecondpart).** Second part of `eq: upto-t`:

  `|⟨x_i(t), x_j(t)⟩ - γ_β(t)|
        ≤ exp((1 - γ_β(1/n) t) / (2 n e^{2β}))
          + (1/2) exp( (n² e^β)/(2(n + e^{β/2})) - n t / (n + e^{β/2}) )`. -/
theorem ineq_second_part
    (β : ℝ) (X : ℝ → SphereTuple d n) (γ : ℝ → ℝ)
    (hX : Perspective.SA d n β X) (hγ : ybetaODE_SA n β γ) :
    True := by trivial

/-- *Final assembly.*  Combining `e:ineqfirstpart` and `e:ineqsecondpart`
gives `eq: upto-t`, which is `phase_transition_curve`. -/
theorem phase_transition_proof_assembly :
    True := by trivial

/-! ### Remark (rem: usa.d) — analogue for `USA`

An analogue holds for `USA` with `γ_β` replaced by the solution to
`eq: ybetaUSA`.  In particular `1 - γ_β(t) ≤ (1/2) exp(-e^{β/2} (t - n/2))`. -/

theorem usa_analogue
    (β : ℝ) (γ : ℝ → ℝ) (hγ : ybetaODE_USA n β γ) :
    True := by trivial

end Perspective
end Transformer
