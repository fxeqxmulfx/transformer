/-
# Metastability — the PL inequality for `𝖤_β` on `𝕋^n` (§3.2 of 2410.06833v1)

The slow manifold `𝒩_β`, the kernel `g`, and `lem: PL.borjan`;
`eq: otto.attention` is in `Transformer.Metastability.OttoCorollary`.  The framework they apply is in
`Transformer.Metastability.OttoReznikoff`; the hypotheses of `PL_borjan` are
witnessed by `Metastability.circle_witness`.
-/

import Transformer.Metastability.OttoReznikoff
import Transformer.Metastability.CircleWitness

open scoped BigOperators
open Real

namespace Transformer
namespace Metastability

variable (n : ℕ)

open Perspective Metastability

/-- The slow manifold `𝒩_β` of `Lemma lem: PL.borjan`:

  `𝒩_β = { (θ_1,…,θ_n) : max_q max_{θ_i, θ_j ∈ 𝒮_q(2τ)} |θ_i - θ_j|
              ≤ e^{-λ β / 2} }`. -/
def slowManifold
    (β τ lam : ℝ) (k : ℕ) (ω : Idx k → ℝ) : Set (Idx n → ℝ) :=
  { θ | ∀ q : Idx k, ∀ i j : Idx n,
        (1 - 2*τ ≤ Real.cos (θ i - ω q)) →
        (1 - 2*τ ≤ Real.cos (θ j - ω q)) →
        |θ i - θ j| ≤ Real.exp (-(lam * β / 2)) }

/-- The kernel
`g(x) = (cos x - β sin² x) e^{β (cos x - 1)}` used in §3.2. -/
noncomputable def g_OR (β x : ℝ) : ℝ :=
  (Real.cos x - β * Real.sin x ^ 2) * Real.exp (β * (Real.cos x - 1))

/-- **Lemma (lem: PL.borjan) — PL inequality.**

Let `β > 1`, `n ≥ 2`, and let `Θ` be `(β, τ)`-separated with caps around
`ω_1, …, ω_k`, where (eq: tau.small)

  `|u - v| ≤ (1/8) √((1 - δ) / (β + 1/2))`  for all `(u, v) ∈ 𝒮_q(2τ)²`,

for some `8(1 + β) e^{-(1-α)β} e^{-1/2} < δ < 1`, and let `λ > 0` be as in
`eq: lambda.1`.  Then there exist `U ∈ 𝒩_β` and `κ(β, n) > 0` with

  `𝖤_β(U) - 𝖤_β(Θ) ≤ (1/(2 κ(β, n))) ‖∇𝖤_β(Θ)‖²`.

**What the source says and how it is read here.**

*`κ(β, n)` does not depend on `Θ`.*  It is chosen before the configuration
(after `β`, `n` and the constants `δ`, `α` its formula in the proof uses).
Chosen after `Θ` the statement would be nearly empty: `𝖤_β` is bounded, so a
small enough `κ` absorbs any gap once `∇𝖤_β(Θ) ≠ 0`.

*The separation is spelled out,* with the caps `ω` its witnesses rather than
free: every `θ_i` in some `𝒮_q(τ)`, `α` a bound on the cosine across distinct
caps `𝒮_q(2τ)`, `𝒮_p(2τ)` (`eq: alpha.dist.2` — the maximum itself is one,
and every condition only weakens as `α` falls), and `γ > 0`.

*`λ` is bounded by `eq: lambda.3`,* the precise form of `eq: lambda.1` given
in `rem: lambda.gamma` (with `τ` for `ε`, as in §3).  The asymptotic
requirements `γ(β) = Ω(1)` and `λ(β) = Ω(1)` concern a family in `β` and are
dropped.

*The torus is carried by real representatives,* and the distances of
`eq: tau.small` are those of the chart centred at `ω_q`: it is asserted for
`u, v` within `π` of `ω_q`.  Asserted for every representative it would fail
at `u = ω_q`, `v = ω_q + 2π`, and the lemma would hold vacuously.

*The norm is Euclidean,* written out as `∑ᵢ (∂_{θ_i} 𝖤_β)²`.

Not proved here.

Source: arXiv:2410.06833v1, §3.2, `lem: PL.borjan`. -/
theorem PL_borjan (n : ℕ) (β δ α : ℝ) (hβ : 1 < β) (hn : 2 ≤ n) :
    ∃ κ : ℝ, 0 < κ ∧ ∀ (τ lam : ℝ) (k : ℕ) (ω : Idx k → ℝ) (Θ : Idx n → ℝ),
      0 < τ → τ < 1 / 16 → k ≤ n →
      (∀ i : Idx n, ∃ q : Idx k, 1 - τ ≤ Real.cos (Θ i - ω q)) →
      (∀ p p' : Idx k, p ≠ p' → ∀ u v : ℝ,
        1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p') →
          Real.cos (u - v) ≤ α) →
      0 < 1 - α - 8 * τ - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / τ) →
      (∀ q : Idx k, ∀ u v : ℝ, |u - ω q| ≤ π → |v - ω q| ≤ π →
        1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q) →
          |u - v| ≤ (1 / 8) * Real.sqrt ((1 - δ) / (β + 1 / 2))) →
      8 * (1 + β) * Real.exp (-((1 - α) * β)) * Real.exp (-(1 / 2 : ℝ)) < δ → δ < 1 →
      0 < lam →
      lam < min
        (Real.exp ((1 - α - β⁻¹ * Real.log ((β - 1) * τ / (β ^ 2 * (n : ℝ) ^ 2 * Real.exp 1)))
            * β)
          * (1 - Real.exp (-((1 - α - 8 * τ - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / τ)) * β))))
        (1 - α - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2
            / (1 - Real.exp (-(β⁻¹ * Real.log (1 / (8 * τ)) * β))))
          - Real.exp (-(β⁻¹ * Real.log (1 / (8 * τ)) * β))) →
      ∃ U : Idx n → ℝ, U ∈ slowManifold n β τ lam k ω ∧
        angularEβ n β U - angularEβ n β Θ
          ≤ (1 / (2 * κ)) * ∑ i : Idx n, (angularGrad n β Θ i) ^ 2 := by
  sorry

/-- The hypotheses of `PL_borjan` are satisfiable, at `n = 3`, `k = 2`,
`β = 1000`, `τ = 10⁻⁸`, `δ = 1/2`, `λ = 1`: the configuration of
`circle_witness`. -/
example : let τ : ℝ := 1 / 10 ^ 8
    let α : ℝ := 1 - 2 * (1 - 2 * τ) ^ 2
    let ω : Idx 2 → ℝ := ![0, π]
    let Θ : Idx 3 → ℝ := ![0, 1 / 10 ^ 4, π]
    (1 : ℝ) < 1000 ∧ 2 ≤ 3 ∧ 0 < τ ∧ τ < 1 / 16 ∧ 2 ≤ 3 ∧
    (∀ i : Idx 3, ∃ q : Idx 2, 1 - τ ≤ Real.cos (Θ i - ω q)) ∧
    (∀ p p' : Idx 2, p ≠ p' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p') →
        Real.cos (u - v) ≤ α) ∧
    0 < 1 - α - 8 * τ - (1000 : ℝ)⁻¹ * Real.log (2 * ((3 : ℕ) : ℝ) ^ 2 / τ) ∧
    (∀ q : Idx 2, ∀ u v : ℝ, |u - ω q| ≤ π → |v - ω q| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - 1 / 2) / (1000 + 1 / 2))) ∧
    8 * (1 + 1000) * Real.exp (-((1 - α) * 1000)) * Real.exp (-(1 / 2 : ℝ)) < 1 / 2 ∧
    (1 / 2 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (1 : ℝ) < min
      (Real.exp ((1 - α - (1000 : ℝ)⁻¹ * Real.log ((1000 - 1) * τ
          / (1000 ^ 2 * ((3 : ℕ) : ℝ) ^ 2 * Real.exp 1))) * 1000)
        * (1 - Real.exp (-((1 - α - 8 * τ
            - (1000 : ℝ)⁻¹ * Real.log (2 * ((3 : ℕ) : ℝ) ^ 2 / τ)) * 1000))))
      (1 - α - (1000 : ℝ)⁻¹ * Real.log (2 * ((3 : ℕ) : ℝ) ^ 2
          / (1 - Real.exp (-((1000 : ℝ)⁻¹ * Real.log (1 / (8 * τ)) * 1000))))
        - Real.exp (-((1000 : ℝ)⁻¹ * Real.log (1 / (8 * τ)) * 1000))) := by
  intro τ α ω Θ
  obtain ⟨-, h2, h3, h4, h5, h6, -, -, -, -, h11⟩ := circle_witness
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, h11, h2, h3, h4,
    h5, by norm_num, by norm_num, h6⟩

end Metastability
end Transformer
