/-
# Metastability — `eq: otto.attention` (§3.2 of 2410.06833v1)

The corollary of the Otto–Reznikoff framework for `𝖤_β` on `𝕋^n`, in the
setting of `Metastability.PL_borjan`, its configuration witnessed by
`Metastability.circle_witness`.
-/

import Transformer.Metastability.OttoAttention

open scoped BigOperators
open Real

namespace Transformer
namespace Metastability

/-- **Corollary (eq: otto.attention).**

For `β > 1` and a `(β, τ)`-separated configuration meeting (eq: tau.small),
the conclusion of `thm: Otto result` holds with `δ = e^{-λ β / 2}`: the
angular energy along the flow approaches, at the exponential rate of
`thm: Otto result`, that of a point of the slow manifold, up to
`C_ε e^{-λ β / 2}`.

**The setting** is that of `PL_borjan`, spelled out the same way: the caps
`ω` are the witnesses of the separation of `Θ`, `α` bounds the cosine across
distinct caps, `γ > 0`, `eq: tau.small` holds in the chart at each `ω_q` for
some `δ` in the range `lem: PL.borjan` requires, and `λ` is below
`eq: lambda.3`.  An earlier version left `k`, `ω` and `λ` free and dropped
`eq: tau.small`; with `k = 0` the slow manifold is everything.

**What the source says and what is changed here.**  Four things, all of them
forced by what the corollary is a corollary *of*.

*`C_ε` depends on `ε` alone.*  It is the constant of `thm: Otto result`, and
with `β` quantified before it the statement would say nothing: `𝖤_β` is
bounded — `0 < 𝖤_β ≤ 1/(2β)` — so a `C_ε` allowed to depend on `β` could be
chosen to swallow the whole left-hand side, and the inequality would hold
with no dynamics at all.  `ε` is therefore the outermost binder and everything
else, `n` included, comes after `C_ε`, exactly as `C_ε` is written.

*The sign.*  `V(t)` lies on the slow manifold and the flow moves towards it,
so it is `𝖤_β(V) - 𝖤_β(U)` that decays, not the difference the other way
round: the energy of `eq: otto.gf` — the one that *falls* along its flow, as
`PL_borjan` and `quantitative_inequality` both read it — is `-𝖤_β`.  Written
with the difference reversed the square roots are identically `0` below the
slow manifold and the statement is empty.

*`U` is the flow and `V` is its (H1)-projection.*  Both were free in an
earlier version, which made the statement false for trivial reasons.  `U` solves
`U̇ = -∇(-𝖤_β)(U) = ∇𝖤_β(U)`, whose components are `angularGrad`, from `Θ`;
`V(t)` is a point of the slow manifold satisfying (H1) against `U(t)`.  The
angular `USA` dynamics of the paper is this same trajectory traversed at the
constant speed `n e^β` — `angularUSA`'s velocity is `n e^β` times
`angularGrad` — so it is the same curve up to a time change, and the rate
`e^{-(1-ε)t}` is stated in the time of `thm: Otto result`, the result this
one is a corollary of.

*The norm in (H1) is Euclidean.*  `Idx n → ℝ` carries the sup norm in
Mathlib, and the `‖u - v‖²` of (H1) is the Euclidean one, so it is written
out as `∑ᵢ (U t i - V t i)²`.

Not proved here.  The paper derives it from `thm: Otto result` applied to
`-𝖤_β` on `𝕋^n`, but that theorem is false as printed (`not_otto_reznikoff`,
it fails at `t = 0`), so the derivation does not stand; a proof has to go
through a corrected form of it.  The verification of (H1), (H2) for `𝖤_β` —
`PL_borjan` — is `sorry` as well.

Source: arXiv:2410.06833v1, §3.2, `eq: otto.attention`. -/
theorem otto_attention (ε : ℝ) (hε : 0 < ε) (hε1 : ε < 1) :
    ∃ Cε : ℝ, 0 < Cε ∧
      ∀ (n : ℕ) (α β τ δ lam : ℝ) (k : ℕ) (ω : Idx k → ℝ) (Θ : Idx n → ℝ),
        1 < β → 2 ≤ n → 0 < τ → τ < 1 / 16 → k ≤ n →
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
        ∀ U V : ℝ → Idx n → ℝ,
          U 0 = Θ →
          (∀ t : ℝ, ∀ i : Idx n,
            HasDerivAt (fun s => U s i) (angularGrad n β (U t) i) t) →
          (∀ t : ℝ, V t ∈ slowManifold n β τ lam k ω ∧
            (1/2 : ℝ) * ∑ i : Idx n, (U t i - V t i) ^ 2
              ≤ angularEβ n β (V t) - angularEβ n β (U t) ∧
            angularEβ n β (V t) - angularEβ n β (U t)
              ≤ (1/2 : ℝ) * ∑ i : Idx n, (angularGrad n β (U t) i) ^ 2) →
          ∀ t : ℝ, 0 ≤ t →
            Real.sqrt (angularEβ n β (V t) - angularEβ n β (U t))
              ≤ Real.exp (-(1 - ε) * t)
                  * Real.sqrt (angularEβ n β (V 0) - angularEβ n β (U 0))
                + Cε * Real.exp (-(lam * β / 2)) := by
  sorry

/-- The hypotheses of `otto_attention` on the constant and the configuration
are satisfiable: `ε = 1/2`, and the configuration of `circle_witness` at
`n = 3`, `k = 2`, `β = 1000`, `τ = 10⁻⁸`, `δ = 1/2`, `λ = 1`.  The flow `U`
out of it and a projection `V` satisfying (H1) along it are not constructed. -/
example : let τ : ℝ := 1 / 10 ^ 8
    let α : ℝ := 1 - 2 * (1 - 2 * τ) ^ 2
    let ω : Idx 2 → ℝ := ![0, π]
    let Θ : Idx 3 → ℝ := ![0, 1 / 10 ^ 4, π]
    (0 : ℝ) < 1 / 2 ∧ (1 : ℝ) / 2 < 1 ∧
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
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, h11, h2, h3, h4, h5, by norm_num, by norm_num, h6⟩

end Metastability
end Transformer
