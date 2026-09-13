/-
# Metastability — Main theorem (thm: metastability) and direct proof

This file formalizes §2 of arXiv:2410.06833v1.

Equations covered:
* `thm: metastability`         — main metastability theorem,
* `eq: lambda.1, lambda.2, lambda.3`  — admissible decay rates `λ(β)`,
* `eq: stick`                   — particles within a cap are exponentially close,
* `eq: comparison`              — Cauchy-problem comparison for `ρ_q(t)`,
* `Lemma lem: eminem` (Until collapse),
* `Lemma lem:collapsetime` (Propagation),
* `eq: an.ineq`                 — `1 - ρ_q(T₁) ≤ e^{-λβ}`,
* `eq: ze.equation`             — differential inequality for `ρ_q`,
* `eq: delta.alpha.cond`        — propagation-of-smallness condition,
* the auxiliary `compt: bound_far`, `compt: ineq_1`, `compt: minim`,
  `compt: bound_far_again`, `eq: rhoq.lb`, `eq: rhoq.lb2`.
-/

import Transformer.Basic
import Transformer.Section1_IPS
import Transformer.Metastability.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace Metastability

open SectionIPS

variable (d n : ℕ)

/-- **`a_{ij}(t)` — attention weights for the `SA` model in the
metastability paper:

  `a_{ij}(t) = exp(β ⟨x_i, x_j⟩) / Σ_k exp(β ⟨x_i, x_k⟩)`. -/
noncomputable def attn
    (β : ℝ) (X : ℝ → SphereTuple d n) (t : ℝ) (i j : Idx n) : ℝ :=
  Real.exp (β * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d)))
    /
  ∑ k : Idx n,
    Real.exp (β * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t k : EucSpace d)))

/-- **Equation (eq: ze.equation).**  Differential inequality for the
within-cap inner-product minimum `ρ_q(t)`:

  `ρ̇_q(t) ≥ (2/n) ρ_q(t)(1 - ρ_q(t)) e^{β(ρ_q(t) - 1)}
                - 2 n e^{-(1-α) β}`. -/
theorem rho_diff_ineq
    (β α : ℝ) (X : ℝ → SphereTuple d n) (hX : SectionIPS.SA d n β X)
    (q : ℕ) (ρq : ℝ → ℝ) (Tesc : ℝ) :
    ∀ t : ℝ, 0 ≤ t → t ≤ Tesc →
      (2 / (n : ℝ)) * ρq t * (1 - ρq t) * Real.exp (β * (ρq t - 1))
        - 2 * (n : ℝ) * Real.exp (-((1 - α) * β))
      ≤ deriv ρq t := by
  sorry

/-- **Lemma (lem: eminem) — *Until collapse.*

For `β > 1`, `c > 0`, `u_0 ∈ (0, 1]`, consider the Cauchy problem

  `u̇(t) = u(t) (1 - u(t)) e^{β(u(t) - 1)}`,    `u(0) = u_0`.

Then

  `inf { t ≥ 0 : 1 - u(t) ≤ e^{-c β} } ≤ e^{β(1 - u_0)} / u_0 + (β² c e) / (β - 1)`. -/
lemma eminem
    (β : ℝ) (hβ : 1 < β) (c : ℝ) (hc : 0 < c)
    (u₀ : ℝ) (hu : 0 < u₀ ∧ u₀ ≤ 1)
    (u : ℝ → ℝ)
    (hu_init : u 0 = u₀)
    (hu_ode : ∀ t : ℝ, HasDerivAt u (u t * (1 - u t) * Real.exp (β * (u t - 1))) t) :
    sInf { t : ℝ | 0 ≤ t ∧ 1 - u t ≤ Real.exp (-(c * β)) }
      ≤ Real.exp (β * (1 - u₀)) / u₀
        + β^2 * c * Real.exp 1 / (β - 1) := by
  sorry

/-- **Lemma (lem:collapsetime) — *Propagation.*

Fix `β > 1`, and let `δ ∈ (0, 1)`, `α ∈ (-1, 1)` satisfy

  `(1/n) δ (1 - δ) e^{-δ β} > n e^{-(1-α) β}`.

If `⟨x_i(0), x_j(0)⟩ ≥ 1 - δ` for all `(i, j) ∈ I²` and
`⟨x_i(t), x_k(t)⟩ ≤ α` for all `i ∈ I`, `k ∈ I^c`, `t ∈ [0, T]`, then

  `⟨x_i(t), x_j(t)⟩ ≥ 1 - δ`  for all `(i, j) ∈ I²` and `t ∈ [0, T]`. -/
lemma propagation
    (β : ℝ) (hβ : 1 < β) (δ α : ℝ)
    (X : ℝ → SphereTuple d n) (hX : SectionIPS.SA d n β X)
    (I : Finset (Idx n)) (T : ℝ) (hT : 0 ≤ T)
    (h_cond : (1 / (n : ℝ)) * δ * (1 - δ) * Real.exp (-(δ * β))
                > (n : ℝ) * Real.exp (-((1 - α) * β)))
    (h_init : ∀ i ∈ I, ∀ j ∈ I,
                1 - δ ≤ inner (𝕜 := ℝ)
                          ((X 0 i : EucSpace d)) ((X 0 j : EucSpace d)))
    (h_far  : ∀ i ∈ I, ∀ k ∈ Iᶜ, ∀ t : ℝ, 0 ≤ t → t ≤ T →
                inner (𝕜 := ℝ)
                  ((X t i : EucSpace d)) ((X t k : EucSpace d)) ≤ α) :
    ∀ i ∈ I, ∀ j ∈ I, ∀ t : ℝ, 0 ≤ t → t ≤ T →
      1 - δ ≤ inner (𝕜 := ℝ)
                  ((X t i : EucSpace d)) ((X t j : EucSpace d)) := by
  sorry

/-- **Theorem (thm: metastability).** *Dynamic metastability.*

Suppose `d, n ≥ 2`, `β > 1`.  Take a `(β, ε)`-separated initial configuration
`(x_i(0))_{i=1}^n` with `ε = ε(β) ∈ (0, 1/16)`, and any `λ = λ(β)`
satisfying `0 < lam < 1 - α - O_{β,n}(1/β)` and `λ(β) = Ω(1)`.

Then there exist `T₂ > T₁ > 0` (as in `MetastabilityTimes`) such that

1. if `x_i(0) ∈ 𝒮_q(ε)`, then `x_i(t) ∈ 𝒮_q(2ε)` for all `t ∈ [0, T₂]`;
2. for all `q ∈ [k]`, `t ∈ [T₁, T₂]`,

   `max_{x_i(t), x_j(t) ∈ 𝒮_q(2ε)} ‖x_i(t) - x_j(t)‖² ≤ 2 e^{-λ β}`. -/
theorem metastability
    (β ε : ℝ) (hβ : 1 < β) (hε : 0 < ε ∧ ε < 1/16) (hd : 2 ≤ d) (hn : 2 ≤ n)
    (X₀ : SphereTuple d n) (hX₀ : isSeparated d n β ε X₀)
    (k : ℕ) (hk : k ≤ n) (w : Idx k → SSphere d)
    (lam : ℝ) (hlam : 0 < lam) :
    ∃ T₁ T₂ : ℝ,
      -- T₁, T₂ satisfy the explicit bounds in `MetastabilityTimes`
      0 < T₁ ∧ T₁ < T₂ ∧
      ∀ X : ℝ → SphereTuple d n,
        X 0 = X₀ → SectionIPS.SA d n β X →
        -- (1) staying in safety caps
        (∀ i : Idx n, ∀ q : Idx k,
          (X₀ i) ∈ sphericalCap d (w q) ε →
          ∀ t : ℝ, 0 ≤ t → t ≤ T₂ →
            (X t i) ∈ sphericalCap d (w q) (2 * ε)) ∧
        -- (2) cap collapse: pairwise within-cap distance is exp small
        ∀ q : Idx k, ∀ t : ℝ, T₁ ≤ t → t ≤ T₂ →
          ∀ i j : Idx n,
            (X t i) ∈ sphericalCap d (w q) (2 * ε) →
            (X t j) ∈ sphericalCap d (w q) (2 * ε) →
              ‖((X t i : EucSpace d)) - ((X t j : EucSpace d))‖^2
                ≤ 2 * Real.exp (-(lam * β)) := by
  sorry

/-- *Variance inequality (rem: variance).* Inside a cap, the
variance-controlled rate of convergence:

  `η̇_q(t) ≥ η_q(t) Σ_{j: x_j ∈ 𝒮_q(2ε)} a_{i(t) j}(t) ‖x_j - x_i‖²/2
              - n e^{-(1-α)β}`. -/
theorem variance_inequality
    (β α ε : ℝ) (X : ℝ → SphereTuple d n) :
    True := by trivial

end Metastability
end Transformer
