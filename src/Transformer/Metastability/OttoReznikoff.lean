/-
# Metastability — Otto–Reznikoff framework and PL inequality (§3 of 2410.06833v1)

Equations and statements covered:

* `eq: otto.gf`             — abstract gradient flow,
* `(H1), (H2)`              — the Otto–Reznikoff hypotheses,
* `eq: first.inequality`    — Polyak–Łojasiewicz-like bound,
* `Theorem thm: Otto result` — restated over the gradient flow and its
  (H1)-projection, the form that leaves both curves free being refuted by
  `not_forall_otto_reznikoff`,
* `eq: otto.1, otto.2`      — the consequence of the Otto–Reznikoff theorem,
* `Lemma lem: bakry-emery` with `ineq: Almost Hessian` — in
  `Transformer.Metastability.BakryEmery`, which proves it with the flow and
  the gradient the paper assumes, and refutes the form that leaves them free,
* `Lemma lem: PL.borjan`    — PL inequality for `𝖤_β` on `𝕋^n`,
* `eq: tau.small`, `eq: cond.sine`, `eq: Ht.first.lb`, `eq: Ht.second.lb`,
* `eq: Ht.third.lb`, `Claim claim: 1` — refuted in the form that leaves the
  cluster free, by `not_forall_claim_one`,
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
        ≤ e^{-(1-ε) t} √(𝖤(u(0)) - 𝖤(v(0))) + C_ε δ`.

**What the source says and what is changed here.**  `u` and `v` are not two
arbitrary curves.  `u` is *the* gradient flow `eq: otto.gf` of `𝖤`, and `v(t)`
is *the* point of `𝒩` that (H1) attaches to `u(t)` — the paper writes
`v(t) ∈ 𝒩` for the projection whose existence (H1) asserts, and its proof uses
both halves of (H1) at that point.  Carried with `u` and `v` free, as they
were, the statement is false: `not_forall_otto_reznikoff` refutes it at
`M = ℝ` with `𝖤 ≡ 0`, where (H1) and (H2) hold, by letting `u` sit at
`C_ε + 1` and `v` at `0`.

`gradNorm` is therefore no longer a free scalar field either: it is the norm
of the field `gradE` that drives the flow, which is what `‖∇𝖤(u)‖` means.

Not proved here.

Source: arXiv:2410.06833v1, §3.1, `thm: Otto result`, `eq: otto.1`,
`eq: otto.2`. -/
theorem otto_reznikoff
    {M : Type*} [NormedAddCommGroup M] [NormedSpace ℝ M]
    (E : M → ℝ) (gradE : M → M)
    (𝒩 : Set M) (δ : ℝ) (hδ : 0 < δ)
    (h1 : H1 E (fun u => ‖gradE u‖) 𝒩) (h2 : H2 E 𝒩 δ) :
    ∀ (ε : ℝ), 0 < ε → ε < 1 →
      ∃ Cε : ℝ, 0 < Cε ∧
        ∀ (u₀ : M) (u v : ℝ → M),
          abstractGF gradE u₀ u →
          (∀ t : ℝ, v t ∈ 𝒩 ∧
            (1/2 : ℝ) * ‖u t - v t‖ ^ 2 ≤ E (u t) - E (v t) ∧
            E (u t) - E (v t) ≤ (1/2 : ℝ) * ‖gradE (u t)‖ ^ 2) →
          ∀ t : ℝ, 0 ≤ t →
            ‖u t - v t‖ + Real.sqrt (E (u t) - E (v t))
              ≤ Real.exp (-(1 - ε) * t)
                  * Real.sqrt (E (u 0) - E (v 0))
                + Cε * δ := by
  sorry

/-- The hypotheses of `otto_reznikoff` are satisfiable: `M = ℝ`, `𝖤 ≡ 0`,
`∇𝖤 ≡ 0`, `𝒩 = ℝ` and `δ = 1`.  (H1) holds with `v = u`, and (H2) reads
`0 ≤ ‖v₁ - v₂‖`. -/
example :
    H1 (fun _ : ℝ => (0 : ℝ)) (fun u : ℝ => ‖(fun _ : ℝ => (0 : ℝ)) u‖) Set.univ ∧
      H2 (fun _ : ℝ => (0 : ℝ)) Set.univ 1 :=
  ⟨fun u => ⟨u, Set.mem_univ u, by simp, by simp⟩,
    fun _ _ _ _ => by simp⟩

/-- **`thm: Otto result` is false over two free curves.**

The theorem is about the gradient flow `u` of `eq: otto.gf` and the
`𝒩`-projection `v` that (H1) attaches to it.  With `u` and `v` free it claims
that *any* two curves whatever stay within `C_ε δ` of each other, and that is
false as soon as one of them is far away: at `M = ℝ` with `𝖤 ≡ 0`,
`∇𝖤 ≡ 0`, `𝒩 = ℝ` and `δ = 1` both hypotheses hold — (H1) with `v = u`, (H2)
because `0 ≤ ‖v₁ - v₂‖` — and, reading the conclusion at `t = 0` with
`u ≡ C_ε + 1` and `v ≡ 0`, both square roots vanish and it says
`C_ε + 1 ≤ C_ε`.

Source: arXiv:2410.06833v1, §3.1, `thm: Otto result`. -/
theorem not_forall_otto_reznikoff :
    ¬ ∀ (E : ℝ → ℝ) (gradNorm : ℝ → ℝ) (𝒩 : Set ℝ) (δ : ℝ), 0 < δ →
        H1 E gradNorm 𝒩 → H2 E 𝒩 δ →
        ∀ ε : ℝ, 0 < ε → ε < 1 →
          ∃ Cε : ℝ, 0 < Cε ∧
            ∀ u v : ℝ → ℝ, ∀ t : ℝ, 0 ≤ t →
              ‖u t - v t‖ + Real.sqrt (E (u t) - E (v t))
                ≤ Real.exp (-(1 - ε) * t) * Real.sqrt (E (u 0) - E (v 0))
                  + Cε * δ := by
  intro h
  obtain ⟨C, hC, hkey⟩ :=
    h (fun _ => 0) (fun _ => 0) Set.univ 1 one_pos
      (fun u => ⟨u, Set.mem_univ u, by simp, by simp⟩)
      (fun _ _ _ _ => by simp)
      (1/2) (by norm_num) (by norm_num)
  have hbad := hkey (fun _ => C + 1) (fun _ => 0) 0 le_rfl
  have hnorm : ‖(C + 1 : ℝ) - 0‖ = C + 1 := by
    rw [sub_zero, Real.norm_eq_abs, abs_of_pos (by linarith)]
  simp only [sub_self, Real.sqrt_zero, mul_zero, add_zero, mul_one, hnorm] at hbad
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

/-- **Claim (claim: 1) is false in the form it was carried here.**

The paper's claim is

  `max_ℓ |∂_{θ_ℓ} 𝖤_β(Θ)|
     ≤ (e/2) max{ |∂_{θ_1} 𝖤_β(Θ)|, |∂_{θ_r} 𝖤_β(Θ)| }`,

and it is stated about *one cluster*: `θ_1, …, θ_r` are the particles of a
single spherical cap `𝒮_q(2τ)`, **relabelled so that `θ_1 < ⋯ < θ_r`**, at a
configuration `Θ ∉ 𝒩_β` off the slow manifold.  Its proof uses all of that —
the ordering, to compare `sin(θ_k - θ_j)` with `sin(θ_k - θ_1)`; the cap, for
`|sin(θ_j - θ_i)| ≤ β^{-1/2}`; and `Θ ∉ 𝒩_β`, for the final absorption of
`(1 + β) e^{-(1-α)β}`.

Carried into Lean with `Θ` and `r` free, the claim asserts that for *every*
configuration of angles the `ℓ`-th partial derivative is controlled by those
at the indices `0` and `r - 1`, and that is false.  Take `n = 3`, `β = 2`,
`r = 1` and `Θ = (0, π/2, -π/2)`: the two neighbours of `θ_0` are symmetric
about it, so `∂_{θ_0} 𝖤_β(Θ) = 0` and the right-hand side vanishes, while

  `∂_{θ_1} 𝖤_β(Θ) = -(1/9) e^{-2} ≠ 0`.

None of the cluster structure is present in this file — `θ_1 < ⋯ < θ_r` in
particular is a relabelling of the particles of `𝒮_q(2τ)`, which is not
constructed here — so the claim is recorded as refuted in the form it was
stated, and not restated.

Source: arXiv:2410.06833v1, §3.2, `claim: 1`, `eq: Ht.third.lb`. -/
theorem not_forall_claim_one :
    ¬ ∀ (n : ℕ) (β : ℝ) (Θ : Idx n → ℝ) (r : ℕ)
        (h0 : 0 < n) (hr1 : 1 ≤ r) (hrn : r ≤ n), 1 < β →
        ∀ l : Idx n, |angularGrad n β Θ l|
          ≤ (Real.exp 1 / 2)
            * max |angularGrad n β Θ ⟨0, h0⟩|
                  |angularGrad n β Θ ⟨r - 1, by omega⟩| := by
  intro h
  have hbad := h 3 2 ![0, Real.pi/2, -(Real.pi/2)] 1 (by norm_num) le_rfl
    (by norm_num) (by norm_num) 1
  rw [show (⟨0, by norm_num⟩ : Idx 3) = 1 - 1 from rfl] at hbad
  simp only [angularGrad, Fin.sum_univ_three, Matrix.cons_val_zero,
    Matrix.cons_val_one, Matrix.head_cons, Matrix.cons_val_two, Matrix.tail_cons,
    sub_zero, zero_sub, sub_self, sub_neg_eq_add] at hbad
  norm_num [Real.sin_pi_div_two, Real.cos_pi_div_two, Real.sin_neg,
    Real.cos_neg, show Real.pi/2 + Real.pi/2 = Real.pi by ring,
    Real.sin_pi, Real.cos_pi] at hbad
  have hpos : (0 : ℝ) < Real.exp (-2) := Real.exp_pos _
  linarith

/-- **Corollary (eq: otto.attention).**

For `β > 1` and a `(β, τ)`-separated configuration meeting (eq: tau.small),
the conclusion of `thm: Otto result` holds with `δ = e^{-λ β / 2}`: the
angular energy along the flow approaches, at the exponential rate of
`otto_reznikoff`, that of a point of the slow manifold, up to
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

*`U` is the flow and `V` is its (H1)-projection.*  Both were free, which is
what `not_forall_otto_reznikoff` refutes one import away.  `U` solves
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

Not proved here: it is `otto_reznikoff` applied to `-𝖤_β` on `𝕋^n`, and both
that theorem and the verification of (H1), (H2) for `𝖤_β` — which is
`PL_borjan` — are `sorry` here.

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
