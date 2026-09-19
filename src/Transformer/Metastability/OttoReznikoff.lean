/-
# Metastability — Otto–Reznikoff framework and PL inequality (§3 of 2410.06833v1)

Equations and statements covered:

* `eq: otto.gf`             — abstract gradient flow,
* `(H1), (H2)`              — the Otto–Reznikoff hypotheses,
* `eq: first.inequality`    — Polyak–Łojasiewicz-like bound,
* `Theorem thm: Otto result`,
* `eq: otto.1, otto.2`      — the consequence of the Otto–Reznikoff theorem,
* `Lemma lem: bakry-emery`,
* `ineq: Almost Hessian`,
* `Lemma lem: PL.borjan`    — PL inequality for `𝖤_β` on `𝕋^n`,
* `eq: tau.small`, `eq: cond.sine`, `eq: Ht.first.lb`, `eq: Ht.second.lb`,
  `eq: Ht.third.lb`, `Claim claim: 1`,
* `Lemma lem: quantitative inequality` — in
  `Transformer.Metastability.QuantitativeInequality`, which proves it with the
  sign and the constant its argument supports,
* `Corollary eq: otto.attention`,
* `Remark rem: sa.extension` — the extension to `SA`,
* `eq: hessian.lb.reverse.pl`  — acceleration / reverse PL inequality.
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS
import Transformer.Perspective.Section6_Circle
import Transformer.Metastability.Basic
import Transformer.Metastability.MainTheorem
import Transformer.Metastability.AngularEnergy

open scoped BigOperators
open Real

namespace Transformer
namespace Metastability

variable (d n : ℕ)

/-! ### §3.1 — Abstract framework -/

/-- **Equation (eq: otto.gf).** Abstract gradient flow on a manifold:

  `u̇(t) = -∇𝖤(u(t))`,  `u(0) = u_0`.

The manifold is flattened to a normed space and the gradient is carried as an
abstract field `gradE`: the Riemannian structure of §3.1 is not formalized.
Source: arXiv:2410.06833v1, §3.1, `eq: otto.gf`. -/
def abstractGF
    {M : Type*} [NormedAddCommGroup M] [NormedSpace ℝ M]
    (gradE : M → M)
    (u₀ : M) (u : ℝ → M) : Prop :=
  u 0 = u₀ ∧ ∀ t : ℝ, HasDerivAt u (-(gradE (u t))) t

/-- **Hypothesis (H1).** For every `u ∈ ℳ` there is `v ∈ 𝒩` with

  `(1/2) ‖u - v‖² ≤ 𝖤(u) - 𝖤(v) ≤ (1/2) ‖∇𝖤(u)‖²`. -/
def H1
    {M : Type*} [NormedAddCommGroup M] (E : M → ℝ) (gradNorm : M → ℝ)
    (𝒩 : Set M) : Prop :=
  ∀ u : M, ∃ v ∈ 𝒩,
    (1/2 : ℝ) * ‖u - v‖^2 ≤ E u - E v ∧
    E u - E v ≤ (1/2 : ℝ) * (gradNorm u)^2

/-- **Hypothesis (H2).** There is `δ > 0` such that for all `v₁, v₂ ∈ 𝒩`,

  `|𝖤(v₁) - 𝖤(v₂)| ≤ δ ‖v₁ - v₂‖`. -/
def H2
    {M : Type*} [NormedAddCommGroup M] (E : M → ℝ) (𝒩 : Set M) (δ : ℝ) : Prop :=
  ∀ v₁ ∈ 𝒩, ∀ v₂ ∈ 𝒩, |E v₁ - E v₂| ≤ δ * ‖v₁ - v₂‖

/-- **Theorem (thm: Otto result), eq: otto.1, eq: otto.2.**

Under hypotheses (H1) and (H2) the gradient flow is drawn into a
`δ`-neighborhood of the slow manifold `𝒩` exponentially:

  `‖u(t) - v(t)‖ + √(𝖤(u(t)) - 𝖤(v(t)))
        ≤ e^{-(1-ε) t} √(𝖤(u(0)) - 𝖤(v(0))) + C_ε δ`. -/
theorem otto_reznikoff
    {M : Type*} [NormedAddCommGroup M] (E : M → ℝ) (gradNorm : M → ℝ)
    (𝒩 : Set M) (δ : ℝ) (hδ : 0 < δ)
    (h1 : H1 E gradNorm 𝒩) (h2 : H2 E 𝒩 δ) :
    ∀ (ε : ℝ), 0 < ε → ε < 1 →
      ∃ Cε : ℝ, 0 < Cε ∧
        ∀ u : ℝ → M, ∀ v : ℝ → M, ∀ t : ℝ, 0 ≤ t →
          ‖u t - v t‖ + Real.sqrt (E (u t) - E (v t))
            ≤ Real.exp (-(1 - ε) * t)
                * Real.sqrt (E (u 0) - E (v 0))
              + Cε * δ := by
  sorry

/-- **Lemma (lem: bakry-emery), ineq: Almost Hessian.**

If `⟨∇𝖤(X(t)), Hess 𝖤(X(t)) ∇𝖤(X(t))⟩ ≤ -c ‖∇𝖤(X(t))‖²` for all
`t ∈ [0, T]`, with `X(T) = v` and `X(0) = u`, then

  `𝖤(v) - 𝖤(u) ≤ (1/(2c)) ‖∇𝖤(u)‖²`.

The Hessian quadratic form is carried as an abstract `gradHess`, the way
`reversePL` below carries it: the Riemannian Hessian of `𝖤_β` on `𝕋^n` is not
formalized.  Source: arXiv:2410.06833v1, §3.1, `lem: bakry-emery`. -/
lemma bakry_emery
    {M : Type*} [NormedAddCommGroup M]
    (E : M → ℝ) (gradNorm gradHess : M → ℝ)
    (u v : M) (c T : ℝ) (hc : 0 < c) (hT : 0 < T)
    (X : ℝ → M) (hX0 : X 0 = u) (hXT : X T = v)
    (hHess : ∀ t : ℝ, 0 ≤ t → t ≤ T →
      gradHess (X t) ≤ -(c * (gradNorm (X t))^2)) :
    E v - E u ≤ (1 / (2 * c)) * (gradNorm u)^2 := by
  sorry

/-- The hypotheses of `bakry_emery` are satisfiable: a constant flow with a
vanishing gradient meets the Hessian bound at every `c > 0`. -/
example (E : ℝ → ℝ) (u : ℝ) (c : ℝ) (hc : 0 < c) :
    E u - E u ≤ (1 / (2 * c)) * (0 : ℝ)^2 :=
  bakry_emery E (fun _ => 0) (fun _ => 0) u u c 1 hc one_pos (fun _ => u) rfl rfl
    (fun _ _ _ => by simp)

/-! ### §3.2 — Application to `𝖤_β` on `𝕋^n` -/

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

Under the smallness condition (eq: tau.small)

  `|u - v| ≤ (1/8) √((1 - δ) / (β + 1/2))`  for all `(u, v) ∈ 𝒮_q(2τ)²`,

and the lower-bound condition `8(1 + β) e^{-(1-α)β} e^{-1/2} < δ < 1`, the
energy satisfies

  `𝖤_β(U) - 𝖤_β(Θ) ≤ (1/(2 κ(β, n))) ‖∇𝖤_β(Θ)‖²`

for some `U ∈ 𝒩_β` and `κ(β, n) > 0`. -/
lemma PL_borjan
    (β τ δ α lam : ℝ) (hβ : 1 < β) (hn : 2 ≤ n)
    (Θ : Idx n → ℝ) (hsep : isSeparatedAngles n α β τ Θ)
    (k : ℕ) (hk : k ≤ n) (ω : Idx k → ℝ)
    (h_tau_small : ∀ q : Idx k, ∀ u v : ℝ,
                    (1 - 2*τ ≤ Real.cos (u - ω q)) →
                    (1 - 2*τ ≤ Real.cos (v - ω q)) →
                    |u - v| ≤ (1/8 : ℝ) * Real.sqrt ((1 - δ) / (β + 1/2)))
    (h_delta : 8 * (1 + β) * Real.exp (-((1 - α) * β)) * Real.exp (-(1/2 : ℝ)) < δ
                ∧ δ < 1) :
    ∃ (U : Idx n → ℝ) (κ : ℝ),
      U ∈ slowManifold n β τ lam k ω ∧
      0 < κ ∧
      angularEβ n β U - angularEβ n β Θ
      ≤ (1 / (2 * κ)) * ∑ i : Idx n, (angularGrad n β Θ i)^2 := by
  sorry

/-- **Claim (claim: 1).**

  `max_ℓ |∂_{θ_ℓ} 𝖤_β(Θ)|
     ≤ (e/2) max{ |∂_{θ_1} 𝖤_β(Θ)|, |∂_{θ_r} 𝖤_β(Θ)| }`:

inside a cluster the largest partial derivative of the angular energy is
controlled by the two at the ends of the cluster, `θ_1` and `θ_r`.

A `Prop`-valued definition and not a lemma: the claim is proved in the paper
by a monotonicity argument along the cluster that is not formalized here.
The partial derivative is `angularGrad`, which `hasDerivAt_angularEβ` proves
to be one.  Source: arXiv:2410.06833v1, §3.2, `claim: 1`. -/
def Claim1
    (n : ℕ) (β : ℝ) (Θ : Idx n → ℝ) (r : ℕ)
    (h0 : 0 < n) (hr1 : 1 ≤ r) (hrn : r ≤ n) : Prop :=
  1 < β →
    ∀ l : Idx n, |angularGrad n β Θ l|
      ≤ (Real.exp 1 / 2)
        * max |angularGrad n β Θ ⟨0, h0⟩| |angularGrad n β Θ ⟨r - 1, by omega⟩|

/-- **Corollary (eq: otto.attention).**

For `β > 1` and a `(β, τ)`-separated configuration meeting (eq: tau.small),
the conclusion of `thm: Otto result` holds for the angular `USA` dynamics
with `δ = e^{-λ β / 2}`: the angular energy along the flow approaches, at the
exponential rate of `otto_reznikoff`, that of a point of the slow manifold,
up to `C_ε e^{-λ β / 2}`.

A `Prop`-valued definition and not a theorem: it is `otto_reznikoff` applied
to `𝖤_β` on `𝕋^n`, and both that theorem and the verification of (H1), (H2)
for `𝖤_β` — which is `PL_borjan` — are `sorry` here.  Source:
arXiv:2410.06833v1, §3.2, `eq: otto.attention`. -/
def OttoAttention
    (n : ℕ) (α β τ lam : ℝ) (Θ : Idx n → ℝ) (k : ℕ) (ω : Idx k → ℝ) : Prop :=
  1 < β → isSeparatedAngles n α β τ Θ →
  ∀ ε : ℝ, 0 < ε → ε < 1 →
    ∃ Cε : ℝ, 0 < Cε ∧
      ∀ U : ℝ → Idx n → ℝ, ∀ V : ℝ → Idx n → ℝ,
        (∀ t : ℝ, V t ∈ slowManifold n β τ lam k ω) →
        ∀ t : ℝ, 0 ≤ t →
          Real.sqrt (angularEβ n β (U t) - angularEβ n β (V t))
            ≤ Real.exp (-(1 - ε) * t)
                * Real.sqrt (angularEβ n β (U 0) - angularEβ n β (V 0))
              + Cε * Real.exp (-(lam * β / 2))

/-! ### §3.3 — Acceleration of the gradient between metastable states -/

/-- **Equation (eq: hessian.lb.reverse.pl).**

  `⟨Hess 𝖤(X(t)) ∇𝖤(X(t)), ∇𝖤(X(t))⟩ ≥ c ‖∇𝖤(X(t))‖²`. -/
def reversePL
    {M : Type*} [NormedAddCommGroup M]
    (gradHess : M → ℝ) (gradNorm : M → ℝ)
    (c T : ℝ) (X : ℝ → M) : Prop :=
  ∀ t : ℝ, 0 ≤ t → t ≤ T →
    c * (gradNorm (X t))^2 ≤ gradHess (X t)

/-- *Reverse PL inequality:* if (eq: hessian.lb.reverse.pl) holds, then

  `𝖤(v) - 𝖤(u) ≥ c ‖∇𝖤(v)‖²`,

and one obtains the exponential acceleration

  `‖∇𝖤(X(t))‖² ≥ e^{2 c t} ‖∇𝖤(X(0))‖²`. -/
theorem reverse_PL_acceleration
    {M : Type*} [NormedAddCommGroup M]
    (E : M → ℝ) (gradHess gradNorm : M → ℝ)
    (X : ℝ → M) (c T : ℝ) (hc : 0 < c) (hT : 0 < T)
    (hrev : reversePL gradHess gradNorm c T X) :
    ∀ t : ℝ, 0 ≤ t → t ≤ T →
      (gradNorm (X 0))^2 * Real.exp (2 * c * t)
        ≤ (gradNorm (X t))^2 := by
  sorry

end Metastability
end Transformer
