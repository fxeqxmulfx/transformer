/-
# Metastability — Otto–Reznikoff framework and PL inequality (§3 of 2410.06833v1)

Equations and statements covered:

* `eq: otto.gf`             — abstract gradient flow,
* `(H1), (H2)`              — the Otto–Reznikoff hypotheses,
* `eq: first.inequality`    — Polyak–Łojasiewicz-like bound,
* `Theorem thm: Otto result` — false as printed, even for the true gradient
  flow and its (H1)-projection: refuted by `not_otto_reznikoff`,
* `eq: otto.1, otto.2`      — the consequence of the Otto–Reznikoff theorem,
* `Lemma lem: bakry-emery` with `ineq: Almost Hessian` — in
  `Transformer.Metastability.BakryEmery`, which proves it with the flow and
  the gradient the paper assumes, and refutes the form that leaves them free,
* `Lemma lem: PL.borjan`    — PL inequality for `𝖤_β` on `𝕋^n`,
* `eq: tau.small`, `eq: cond.sine`, `eq: Ht.first.lb`, `eq: Ht.second.lb`,
* `eq: Ht.third.lb`, `Claim claim: 1` — in
  `Transformer.Metastability.OttoClaimOne`, stated for one ordered cluster off
  the slow manifold, as the source uses it,
* `Lemma lem: quantitative inequality` — in
  `Transformer.Metastability.QuantitativeInequality`, which proves it with the
  sign and the constant its argument supports,
* `Corollary eq: otto.attention`,
* `Remark rem: sa.extension` — the extension to `SA`.

`eq: hessian.lb.reverse.pl`, the acceleration of §3.3, is in
`Transformer.Metastability.ReversePL`, which proves it with the chain rule its
derivation uses and refutes the form that leaves the two scalar fields free.
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS
import Transformer.Perspective.Section6_Circle
import Transformer.Metastability.Basic
import Transformer.Metastability.MainTheorem
import Transformer.Metastability.AngularEnergy
import Mathlib.Analysis.Calculus.Gradient.Basic

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

/-- **`thm: Otto result` is false as printed, with the true gradient.**

The paper states, "verbatim" from Otto–Reznikoff: under (H1) and (H2), for
the gradient flow `u` of `eq: otto.gf` and any `v` with `v(t) ∈ 𝒩` satisfying
`eq: first.inequality` against `u(t)`, and every `ε ∈ (0,1)`, some `C_ε > 0`
gives, for all `t ≥ 0`,

  `‖u(t) - v(t)‖ + √(𝖤(u(t)) - 𝖤(v(t)))
        ≤ e^{-(1-ε) t} √(𝖤(u(0)) - 𝖤(v(0))) + C_ε δ`.   (`eq: otto.1`)

At `t = 0` the two square roots cancel and what is left is
`‖u(0) - v(0)‖ ≤ C_ε δ` for every initial datum, which no constant can give.
The counterexample is as honest as the setting allows: `M = ℝ`,
`𝖤(x) = x²/2 ≥ 0` smooth, `∇𝖤(x) = x` its actual gradient, `𝒩 = {0}`,
`δ = 1`.  (H1) holds with `v = 0` and equality on both sides, (H2) is
trivial on a point, the flow is `u(t) = u₀ e^{-t}`, and `v ≡ 0` satisfies
`eq: first.inequality` at every time; `u₀ = C_ε + 1` then breaks `eq: otto.1`.

By (H1) `‖u - v‖ ≤ √(2(𝖤(u) - 𝖤(v)))`, so a version with a constant factor in
front of the first term would survive this example; which factor the original
has is not recoverable from this source, and the statement is therefore
refuted rather than repaired.  `eq: otto.2` falls with it on the same flow
(`s` small, `t - s ≈ 2`, `u₀` large) and is not stated separately.

Source: arXiv:2410.06833v1, §3.1, `thm: Otto result`, `eq: otto.1`, with
(H1), (H2) and `eq: otto.gf` of the same section. -/
theorem not_otto_reznikoff :
    ¬ ∀ (E : ℝ → ℝ) (gradE : ℝ → ℝ) (𝒩 : Set ℝ) (δ : ℝ), 0 < δ →
        (∀ u, 0 ≤ E u) → (∀ u, HasGradientAt E (gradE u) u) →
        H1 E (fun u => ‖gradE u‖) 𝒩 → H2 E 𝒩 δ →
        ∀ ε : ℝ, 0 < ε → ε < 1 →
          ∃ Cε : ℝ, 0 < Cε ∧
            ∀ (u₀ : ℝ) (u v : ℝ → ℝ),
              abstractGF gradE u₀ u →
              (∀ t : ℝ, v t ∈ 𝒩 ∧
                (1/2 : ℝ) * ‖u t - v t‖ ^ 2 ≤ E (u t) - E (v t) ∧
                E (u t) - E (v t) ≤ (1/2 : ℝ) * ‖gradE (u t)‖ ^ 2) →
              ∀ t : ℝ, 0 ≤ t →
                ‖u t - v t‖ + Real.sqrt (E (u t) - E (v t))
                  ≤ Real.exp (-(1 - ε) * t) * Real.sqrt (E (u 0) - E (v 0))
                    + Cε * δ := by
  intro h
  have hgrad : ∀ u : ℝ, HasGradientAt (fun x : ℝ => x ^ 2 / 2) u u := fun u => by
    apply HasDerivAt.hasGradientAt'
    simpa using (hasDerivAt_pow 2 u).div_const 2
  have hH1 : ∀ u : ℝ, (1/2 : ℝ) * ‖u - 0‖ ^ 2 ≤ u ^ 2 / 2 - 0 ^ 2 / 2 ∧
      u ^ 2 / 2 - 0 ^ 2 / 2 ≤ (1/2 : ℝ) * ‖u‖ ^ 2 := fun u => by
    simp only [sub_zero, Real.norm_eq_abs, sq_abs]; constructor <;> linarith
  obtain ⟨C, hC, hkey⟩ :=
    h (fun x => x ^ 2 / 2) id {0} 1 one_pos (fun u => by positivity) hgrad
      (fun u => ⟨0, rfl, hH1 u⟩)
      (fun v₁ h₁ v₂ h₂ => by simp_all)
      (1/2) (by norm_num) (by norm_num)
  have hflow : abstractGF id (C + 1) (fun t => (C + 1) * Real.exp (-t)) := by
    refine ⟨by simp, fun t => ?_⟩
    have := ((Real.hasDerivAt_exp (-t)).comp t (hasDerivAt_neg t)).const_mul (C + 1)
    convert this using 1
    all_goals first | rfl | (show -((C + 1) * Real.exp (-t)) = _; ring)
  have hbad := hkey (C + 1) _ (fun _ => 0) hflow (fun t => ⟨rfl, hH1 _⟩) 0 le_rfl
  have hnorm : ‖(C + 1) * Real.exp (-0) - 0‖ = C + 1 := by
    rw [neg_zero, Real.exp_zero, mul_one, sub_zero, Real.norm_eq_abs,
      abs_of_pos (by linarith)]
  rw [hnorm] at hbad
  simp only [mul_zero, Real.exp_zero] at hbad
  linarith

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
