/-
# Metastability — Otto–Reznikoff framework (§3.1 of 2410.06833v1)

Equations and statements covered, here and in the modules named:

* `eq: otto.gf`             — abstract gradient flow,
* `(H1), (H2)`              — the Otto–Reznikoff hypotheses,
* `eq: first.inequality`    — Polyak–Łojasiewicz-like bound,
* `Theorem thm: Otto result` — false as printed, even for the true gradient
  flow and its (H1)-projection: refuted by `not_otto_reznikoff`,
* `eq: otto.1, otto.2`      — the consequence of the Otto–Reznikoff theorem,
* `Lemma lem: bakry-emery` with `ineq: Almost Hessian` — in
  `Transformer.Metastability.BakryEmery`, which proves it with the flow and
  the gradient the paper assumes, and refutes the form that leaves them free,
* `Lemma lem: PL.borjan`    — PL inequality for `𝖤_β` on `𝕋^n`, in
  `Transformer.Metastability.OttoAttention`,
* `eq: tau.small`, `eq: cond.sine`, `eq: Ht.first.lb`, `eq: Ht.second.lb`,
* `eq: Ht.third.lb`, `Claim claim: 1` — in
  `Transformer.Metastability.OttoClaimOne`, stated for one ordered cluster off
  the slow manifold, as the source uses it,
* `Lemma lem: quantitative inequality` — in
  `Transformer.Metastability.QuantitativeInequality`, which proves it with the
  sign and the constant its argument supports,
* `Corollary eq: otto.attention` — in `Transformer.Metastability.OttoCorollary`,
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

end Metastability
end Transformer
