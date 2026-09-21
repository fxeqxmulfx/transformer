/-
# Metastability — `claim: 1` of the PL inequality (§3.2 of 2410.06833v1)

The comparison of the partial derivatives of `𝖤_β` inside one cluster that
the proof of `lem: PL.borjan` rests on (`eq: Ht.third.lb`), stated at the
configuration where the source uses it, with its hypotheses witnessed by `Metastability.circle_witness`.
-/

import Transformer.Metastability.OttoAttention

open Real

namespace Transformer
namespace Metastability

/-- **Claim (claim: 1), equation (eq: Ht.third.lb).**

Let `θ_{i_1} < ⋯ < θ_{i_r}` be the particles of one cap `𝒮_q(2τ)`, `1 ≤ r < n`,
at a configuration off the slow manifold through that cap (`eq: cond.sine`).
Then

  `max_ℓ |∂_{θ_{i_ℓ}} 𝖤_β(Θ)| ≤ (e/2) max{ |∂_{θ_{i_1}} 𝖤_β(Θ)|, |∂_{θ_{i_r}} 𝖤_β(Θ)| }`.

**The setting, as the source uses it.**  The claim sits inside the proof of
`lem: PL.borjan` and is about `Θ(t)`, `t ∈ [0, T)`, on the flow started at a
`(β, τ)`-separated configuration.  Its proof uses of `Θ(t)` exactly the
following, which are the hypotheses here, at one configuration `Θ`:

* every particle lies in some cap `𝒮_p(2τ)` (`hcaps`, from `eq: stick`), and
  `α` bounds the cosine between points of distinct caps (`hα`, `eq:
  alpha.dist.2` — the maximum itself satisfies this, and every condition below
  only weakens as `α` falls);
* `γ > 0` (`d: condition_ineq_2`), `eq: tau.small` with its `δ`, and `λ` below
  the bound `eq: lambda.3` of `rem: lambda.gamma` (with `τ` for `ε`, as in §3);
* `q` is a cap satisfying `eq: cond.sine`, which is what `Θ(t) ∉ 𝒩_β` provides;
* the particles of `𝒮_q(2τ)` are relabelled `idx 0, …, idx (r-1)` with
  increasing angles.

The torus is carried by real representatives.  The relabelling
`θ_1 < ⋯ < θ_r` and the distances of `eq: tau.small` are those of the chart
centred at `ω_q`, so the members of a cap are taken within `π` of its centre
(`hchart`, and the same restriction inside `hsmall`); without it a particle and
its translate by `2π` would both sit in the cap, and `eq: tau.small` could not
hold.  The asymptotic requirements `γ(β) = Ω(1)` and `λ(β) = Ω(1)` concern a
family in `β`, not one configuration, and are dropped.

Not proved here.  The source's own last step — "using the fact that
`Θ(t) ∉ 𝒩_β`, `max_ℓ |∂_{θ_ℓ} 𝖤_β| ≥ 2e(1+β)e^{-(1-α)β}`" — is asserted
without derivation.

Source: arXiv:2410.06833v1, §3.2, proof of `lem: PL.borjan`, `claim: 1`,
`eq: Ht.third.lb`. -/
theorem claim_one (n : ℕ) (β τ δ α lam : ℝ) (k : ℕ) (ω : Idx k → ℝ) (Θ : Idx n → ℝ)
    (hβ : 1 < β) (hτ : 0 < τ) (hτ16 : τ < 1 / 16) (hk : k ≤ n)
    (hcaps : ∀ i : Idx n, ∃ p : Idx k, 1 - 2 * τ ≤ Real.cos (Θ i - ω p))
    (hα : ∀ p p' : Idx k, p ≠ p' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p') →
        Real.cos (u - v) ≤ α)
    (hγ : 0 < 1 - α - 8 * τ - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / τ))
    (hsmall : ∀ p : Idx k, ∀ u v : ℝ, |u - ω p| ≤ π → |v - ω p| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - δ) / (β + 1 / 2)))
    (hδ : 8 * (1 + β) * Real.exp (-((1 - α) * β)) * Real.exp (-(1 / 2 : ℝ)) < δ)
    (hδ1 : δ < 1) (hlam : 0 < lam)
    (hlam3 : lam < min
      (Real.exp ((1 - α - β⁻¹ * Real.log ((β - 1) * τ / (β ^ 2 * (n : ℝ) ^ 2 * Real.exp 1)))
          * β)
        * (1 - Real.exp (-((1 - α - 8 * τ - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / τ)) * β))))
      (1 - α - β⁻¹ * Real.log (2 * (n : ℝ) ^ 2
          / (1 - Real.exp (-(β⁻¹ * Real.log (1 / (8 * τ)) * β))))
        - Real.exp (-(β⁻¹ * Real.log (1 / (8 * τ)) * β))))
    (q : Idx k) (r : ℕ) (idx : Fin r → Idx n) (hr : 0 < r) (hrn : r < n)
    (hmono : StrictMono (Θ ∘ idx))
    (hchart : ∀ j : Fin r, |Θ (idx j) - ω q| ≤ π)
    (hmem : ∀ i : Idx n, 1 - 2 * τ ≤ Real.cos (Θ i - ω q) ↔ i ∈ Set.range idx)
    (hsine : ∃ a b : Fin r, Real.exp (-(lam * β / 2)) ≤ |Θ (idx a) - Θ (idx b)|) :
    ∀ l : Fin r, |angularGrad n β Θ (idx l)|
      ≤ (Real.exp 1 / 2) * max |angularGrad n β Θ (idx ⟨0, hr⟩)|
          |angularGrad n β Θ (idx ⟨r - 1, by omega⟩)| := by
  sorry

/-- The hypotheses of `claim_one` are satisfiable, at `n = 3`, `k = 2`,
`q = 0`, `r = 2`, `β = 1000`, `τ = 10⁻⁸`, `δ = 1/2`, `λ = 1`: the
configuration of `circle_witness`. -/
example : let τ : ℝ := 1 / 10 ^ 8
    let α : ℝ := 1 - 2 * (1 - 2 * τ) ^ 2
    let ω : Idx 2 → ℝ := ![0, π]
    let Θ : Idx 3 → ℝ := ![0, 1 / 10 ^ 4, π]
    let idx : Fin 2 → Idx 3 := ![0, 1]
    (1 : ℝ) < 1000 ∧ 0 < τ ∧ τ < 1 / 16 ∧ 2 ≤ 3 ∧ (1 / 2 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    0 < 2 ∧ 2 < 3 ∧
    (∀ i : Idx 3, ∃ p : Idx 2, 1 - 2 * τ ≤ Real.cos (Θ i - ω p)) ∧
    (∀ p p' : Idx 2, p ≠ p' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p') →
        Real.cos (u - v) ≤ α) ∧
    0 < 1 - α - 8 * τ - (1000 : ℝ)⁻¹ * Real.log (2 * ((3 : ℕ) : ℝ) ^ 2 / τ) ∧
    (∀ p : Idx 2, ∀ u v : ℝ, |u - ω p| ≤ π → |v - ω p| ≤ π →
      1 - 2 * τ ≤ Real.cos (u - ω p) → 1 - 2 * τ ≤ Real.cos (v - ω p) →
        |u - v| ≤ (1 / 8) * Real.sqrt ((1 - 1 / 2) / (1000 + 1 / 2))) ∧
    8 * (1 + 1000) * Real.exp (-((1 - α) * 1000)) * Real.exp (-(1 / 2 : ℝ)) < 1 / 2 ∧
    (1 : ℝ) < min
      (Real.exp ((1 - α - (1000 : ℝ)⁻¹ * Real.log ((1000 - 1) * τ
          / (1000 ^ 2 * ((3 : ℕ) : ℝ) ^ 2 * Real.exp 1))) * 1000)
        * (1 - Real.exp (-((1 - α - 8 * τ
            - (1000 : ℝ)⁻¹ * Real.log (2 * ((3 : ℕ) : ℝ) ^ 2 / τ)) * 1000))))
      (1 - α - (1000 : ℝ)⁻¹ * Real.log (2 * ((3 : ℕ) : ℝ) ^ 2
          / (1 - Real.exp (-((1000 : ℝ)⁻¹ * Real.log (1 / (8 * τ)) * 1000))))
        - Real.exp (-((1000 : ℝ)⁻¹ * Real.log (1 / (8 * τ)) * 1000))) ∧
    StrictMono (Θ ∘ idx) ∧
    (∀ j : Fin 2, |Θ (idx j) - ω 0| ≤ π) ∧
    (∀ i : Idx 3, 1 - 2 * τ ≤ Real.cos (Θ i - ω 0) ↔ i ∈ Set.range idx) ∧
    (∃ a b : Fin 2, Real.exp (-(1 * 1000 / 2)) ≤ |Θ (idx a) - Θ (idx b)|) := by
  intro τ α ω Θ idx
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, -⟩ := circle_witness
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10⟩

end Metastability
end Transformer
