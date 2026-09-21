/-
# Metastability — the PL inequality for `𝖤_β` on `𝕋^n` (§3.2 of 2410.06833v1)

The angular form of a separated configuration, the slow manifold `𝒩_β`,
`lem: PL.borjan` and `eq: otto.attention`.  The framework they apply is in
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

/-- **Definition (hyp: init.theta).** Angular form of `(β, τ)`-separated
configurations:

`(θ_1,…,θ_n) ∈ 𝕋^n` is `(β, τ)`-separated if there are
`ω_1,…,ω_k ∈ 𝕋` such that each `θ_i ∈ ⋃_q 𝒮_q(τ)`, where

  `𝒮_q(τ) = { θ ∈ 𝕋 : cos(θ - ω_q) ≥ 1 - τ }`,

and `γ(β) := 1 - α - 8τ - β⁻¹ log(2 n²/τ) > 0`, which is the last conjunct.
The asymptotic side of the paper's condition, `γ(β) = Ω(1)`, is not part of
the definition: it is a statement about a family of configurations, not about
one.  Source: arXiv:2410.06833v1, §3.2, `hyp: init.theta`. -/
def isSeparatedAngles
    (α β τ : ℝ) (θ : Idx n → ℝ) : Prop :=
  ∃ (k : ℕ), k ≤ n ∧ ∃ ω : Idx k → ℝ,
    (∀ i : Idx n, ∃ q : Idx k, 1 - τ ≤ Real.cos (θ i - ω q)) ∧
    0 < 1 - α - 8 * τ - β⁻¹ * Real.log (2 * (n : ℝ)^2 / τ)

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

/-- **Corollary (eq: otto.attention).**

For `β > 1` and a `(β, τ)`-separated configuration meeting (eq: tau.small),
the conclusion of `thm: Otto result` holds with `δ = e^{-λ β / 2}`: the
angular energy along the flow approaches, at the exponential rate of
`thm: Otto result`, that of a point of the slow manifold, up to
`C_ε e^{-λ β / 2}`.

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
      ∀ (n : ℕ) (α β τ lam : ℝ) (Θ : Idx n → ℝ) (k : ℕ) (ω : Idx k → ℝ),
        1 < β → isSeparatedAngles n α β τ Θ →
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

/-- The hypotheses of `otto_attention` are satisfiable: `ε = 1/2`.  Everything
else stays inside the statement — `isSeparatedAngles` carries `γ(β) > 0`,
which ties `α`, `τ` and `β` together, and no configuration meeting it, and no
flow out of one, is built in this file. -/
example : (0 : ℝ) < 1 / 2 ∧ (1 : ℝ) / 2 < 1 := by norm_num

end Metastability
end Transformer
