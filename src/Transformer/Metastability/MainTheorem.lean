/-
# Metastability — Main theorem (thm: metastability) and direct proof

This file formalizes §2 of arXiv:2410.06833v1.

Equations covered:
* `thm: metastability`         — main metastability theorem,
* `eq: lambda.1, lambda.2, lambda.3`  — admissible decay rates `λ(β)`,
* `eq: stick`                   — particles within a cap are exponentially close,
* `eq: comparison`              — Cauchy-problem comparison for `ρ_q(t)`,
* `Lemma lem: eminem` (Until collapse) — proved, from the scalar ODE
  analysis of `Metastability.CollapseODE`,
* `Lemma lem:collapsetime` (Propagation),
* `eq: an.ineq`                 — `1 - ρ_q(T₁) ≤ e^{-λβ}`,
* `eq: ze.equation`             — differential inequality for `ρ_q`,
* `eq: delta.alpha.cond`        — propagation-of-smallness condition,
* the auxiliary `compt: bound_far`, `compt: ineq_1`, `compt: minim`,
  `compt: bound_far_again`, `eq: rhoq.lb`, `eq: rhoq.lb2`.

`rem: variance`, which closes §2, is in `Metastability.CapVariance`.
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS
import Transformer.Metastability.Basic
import Transformer.Metastability.CollapseODE
import Transformer.Metastability.PairVelocitySum
import Mathlib.Analysis.InnerProductSpace.Calculus

open scoped BigOperators
open Real

namespace Transformer
namespace Metastability

open Perspective
open scoped Classical

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
                - 2 n e^{-(1-α) β}`.

`ρ_q` is not an arbitrary function: `hρq_le` and `hρq_att` say that `ρ_q(t)`
is the minimum of `⟨x_i(t), x_j(t)⟩` over the cap `I = 𝒮_q`, and `h_far` is
the `α`-separation of the cap from its complement on `[0, Tesc]` that bounds
the leakage term.  Without them the statement is false —
`not_rho_diff_ineq_of_free` refutes it.  The minimum of finitely many smooth
functions has corners, so its differentiability on `[0, Tesc]` is a hypothesis
as well; the paper reads the inequality in the Dini sense instead.

The proof is the envelope argument.  Fix `t` and a pair `(i, j) ∈ I²`
attaining the minimum; since `I` does not move, `s ↦ ρ_q(s) - ⟨x_i(s), x_j(s)⟩`
has a *global* maximum at `t`, so `ρ̇_q(t) = ⟨v_i, x_j⟩ + ⟨x_i, v_j⟩`, and the
two terms are estimated by `Metastability.inner_proj_softmax_pair_sum`.

**What the source says and what is changed here.**  Three things.

*`β ≥ 0` is a hypothesis.*  §2 runs at `β > 1` throughout, and the leakage
estimate `a_{ik} ≤ e^{βα}/e^{β}` and the partition bound `Σ_l e^{β⟨x_i,x_l⟩}
≤ n e^{β}` both reverse without it; stated for an arbitrary `β` it has to be
assumed.

*`ρ_q ≥ 0` is a hypothesis.*  A token `x_k` inside the cap contributes
`a_{ik}(⟨x_k, x_j⟩ - ⟨x_i, x_k⟩ ρ_q)` to `⟨v_i, x_j⟩`, and this is nonnegative
because `⟨x_k, x_j⟩ ≥ ρ_q` and `⟨x_i, x_k⟩ ρ_q ≤ ρ_q` — the second step needs
`ρ_q ≥ 0`.  In §2 `ρ_q ≥ 1 - δ` with `δ < 1`, so the hypothesis holds where
the inequality is used; stated for an arbitrary minimum it has to be assumed.

*The constant is the paper's `2 n`.*  Each of the two velocity terms leaks at
most `n e^{-(1-α)β}` — at most `n` tokens outside the cap, each with weight
at most `e^{-(1-α)β}` against a bracket `⟨x_k, x_j - ρ_q x_i⟩ ≥ -√(1 - ρ_q²)
≥ -1` — and there are two of them, `⟨v_i, x_j⟩` and `⟨x_i, v_j⟩`.

Source: arXiv:2410.06833v1, §2, `eq: ze.equation`. -/
theorem rho_diff_ineq
    (β α : ℝ) (hβ : 0 ≤ β)
    (X : ℝ → SphereTuple d n) (hX : Perspective.SA d n β X)
    (I : Finset (Idx n)) (ρq : ℝ → ℝ) (Tesc : ℝ)
    (hρq_le : ∀ t : ℝ, ∀ i ∈ I, ∀ j ∈ I,
        ρq t ≤ inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d)))
    (hρq_att : ∀ t : ℝ, ∃ i ∈ I, ∃ j ∈ I,
        ρq t = inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d)))
    (hρq_diff : ∀ t : ℝ, 0 ≤ t → t ≤ Tesc → DifferentiableAt ℝ ρq t)
    (hρq_nonneg : ∀ t : ℝ, 0 ≤ t → t ≤ Tesc → 0 ≤ ρq t)
    (h_far : ∀ i ∈ I, ∀ k ∈ Iᶜ, ∀ t : ℝ, 0 ≤ t → t ≤ Tesc →
        inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t k : EucSpace d)) ≤ α) :
    ∀ t : ℝ, 0 ≤ t → t ≤ Tesc →
      (2 / (n : ℝ)) * ρq t * (1 - ρq t) * Real.exp (β * (ρq t - 1))
        - 2 * (n : ℝ) * Real.exp (-((1 - α) * β))
      ≤ deriv ρq t := by
  intro t ht0 htT
  obtain ⟨i, hi, j, hj, hρ⟩ := hρq_att t
  have hnorm : ∀ k : Idx n, ‖((X t k : EucSpace d))‖ = 1 :=
    fun k => mem_sphere_zero_iff_norm.mp (X t k).2
  -- both tokens move with the `SA` velocity of the frozen configuration
  have hvi : HasDerivAt (fun s => ((X s i : EucSpace d)))
      (softmaxVel d n β (fun k => ((X t k : EucSpace d))) i) t := hX t i
  have hvj : HasDerivAt (fun s => ((X s j : EucSpace d)))
      (softmaxVel d n β (fun k => ((X t k : EucSpace d))) j) t := hX t j
  have hF : HasDerivAt
      (fun s => inner (𝕜 := ℝ) ((X s i : EucSpace d)) ((X s j : EucSpace d)))
      (inner (𝕜 := ℝ) ((X t i : EucSpace d))
            (softmaxVel d n β (fun k => ((X t k : EucSpace d))) j)
        + inner (𝕜 := ℝ) (softmaxVel d n β (fun k => ((X t k : EucSpace d))) i)
            ((X t j : EucSpace d))) t :=
    hvi.inner ℝ hvj
  -- the envelope argument: `I` does not move, so the touching is a global max
  have hmax : IsLocalMax
      (fun s => ρq s
        - inner (𝕜 := ℝ) ((X s i : EucSpace d)) ((X s j : EucSpace d))) t := by
    refine Filter.Eventually.of_forall fun s => ?_
    have h1 := hρq_le s i hi j hj
    simp only
    linarith
  have hderiv : deriv ρq t
      = inner (𝕜 := ℝ) ((X t i : EucSpace d))
            (softmaxVel d n β (fun k => ((X t k : EucSpace d))) j)
        + inner (𝕜 := ℝ) (softmaxVel d n β (fun k => ((X t k : EucSpace d))) i)
            ((X t j : EucSpace d)) := by
    have hsub := (hρq_diff t ht0 htT).hasDerivAt.sub hF
    have h0 : deriv ρq t
        - (inner (𝕜 := ℝ) ((X t i : EucSpace d))
              (softmaxVel d n β (fun k => ((X t k : EucSpace d))) j)
          + inner (𝕜 := ℝ) (softmaxVel d n β (fun k => ((X t k : EucSpace d))) i)
              ((X t j : EucSpace d))) = 0 := by
      rw [← hsub.deriv]
      exact hmax.deriv_eq_zero
    linarith
  rw [hderiv]
  linarith [inner_proj_softmax_pair_sum d n β α (ρq t) hβ
    (fun k => ((X t k : EucSpace d))) I i j hi hj hnorm (hρq_nonneg t ht0 htT) hρ
    (fun k hk l hl => hρq_le t k hk l hl)
    (fun k hk l hl => h_far k hk l (Finset.mem_compl.mpr hl) t ht0 htT)]

/-- **The inequality is about the within-cap minimum, not about an arbitrary
curve.**

Dropping `hρq_le` and `hρq_att` from `rho_diff_ineq` makes it false: at
`d = n = 1` the constant configuration solves `eq: SA`, and the constant
`ρ_q ≡ 1/2` — which has nothing to do with it — has `ρ̇_q ≡ 0` while at
`β = 2`, `α = -1` the right-hand side is `e^{-1}/2 - 2 e^{-4} > 0`, because
`e³ > 8`. -/
theorem not_rho_diff_ineq_of_free :
    ¬ ∀ (β α : ℝ) (X : ℝ → SphereTuple 1 1), Perspective.SA 1 1 β X →
        ∀ (ρq : ℝ → ℝ) (Tesc : ℝ), ∀ t : ℝ, 0 ≤ t → t ≤ Tesc →
          (2 / ((1 : ℕ) : ℝ)) * ρq t * (1 - ρq t) * Real.exp (β * (ρq t - 1))
            - 2 * ((1 : ℕ) : ℝ) * Real.exp (-((1 - α) * β))
          ≤ deriv ρq t := by
  intro h
  have hSA : Perspective.SA 1 1 2 (fun _ _ => basePoint 0) :=
    Perspective.SA_const_consensus 1 1 one_pos 2 (basePoint 0)
  have hineq := h 2 (-1) (fun _ _ => basePoint 0) hSA (fun _ => 1 / 2) 1 0 le_rfl zero_le_one
  rw [deriv_const] at hineq
  have he : (2 : ℝ) < Real.exp 1 := by
    have h1 := Real.add_one_lt_exp (x := (1 : ℝ)) (by norm_num)
    linarith
  have h8 : (8 : ℝ) < Real.exp 3 := by
    have h3 : Real.exp 3 = Real.exp 1 * Real.exp 1 * Real.exp 1 := by
      rw [← Real.exp_add, ← Real.exp_add]; norm_num
    rw [h3]; nlinarith [Real.exp_pos (1 : ℝ)]
  have hsplit : Real.exp (-1 : ℝ) = Real.exp 3 * Real.exp (-4 : ℝ) := by
    rw [← Real.exp_add]; norm_num
  have hpos : (0 : ℝ) < Real.exp (-4 : ℝ) := Real.exp_pos _
  norm_num at hineq
  rw [hsplit] at hineq
  nlinarith [mul_pos (sub_pos.mpr h8) hpos]

/-- The hypotheses of `rho_diff_ineq` are satisfiable: at `d = n = 1` the
constant configuration solves `eq: SA`, its cap `I = {0}` has empty
complement, and the within-cap minimum is the constant `1`. -/
example (α : ℝ) :
    ∀ t : ℝ, 0 ≤ t → t ≤ 1 →
      (2 / ((1 : ℕ) : ℝ)) * (fun _ : ℝ => (1 : ℝ)) t * (1 - (fun _ : ℝ => (1 : ℝ)) t)
          * Real.exp (2 * ((fun _ : ℝ => (1 : ℝ)) t - 1))
        - 2 * ((1 : ℕ) : ℝ) * Real.exp (-((1 - α) * 2))
      ≤ deriv (fun _ : ℝ => (1 : ℝ)) t := by
  have hinner : inner (𝕜 := ℝ) ((basePoint 0 : EucSpace 1)) ((basePoint 0 : EucSpace 1))
      = (1 : ℝ) := by
    have hx : ‖(basePoint 0 : EucSpace 1)‖ = 1 :=
      mem_sphere_zero_iff_norm.mp (basePoint 0).2
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  exact rho_diff_ineq 1 1 2 α (by norm_num) (fun _ _ => basePoint 0)
    (Perspective.SA_const_consensus 1 1 one_pos 2 (basePoint 0))
    Finset.univ (fun _ => 1) 1
    (fun _ _ _ _ _ => by rw [hinner])
    (fun _ => ⟨0, Finset.mem_univ 0, 0, Finset.mem_univ 0, by rw [hinner]⟩)
    (fun _ _ _ => differentiableAt_const 1)
    (fun _ _ _ => zero_le_one)
    (fun _ _ k hk _ _ _ => absurd (Finset.mem_univ k) (Finset.mem_compl.mp hk))

/-- **Lemma (lem: eminem) — *Until collapse.*

For `β > 1`, `c > 0`, `u_0 ∈ (0, 1]`, consider the Cauchy problem

  `u̇(t) = u(t) (1 - u(t)) e^{β(u(t) - 1)}`,    `u(0) = u_0`.

Then

  `inf { t ≥ 0 : 1 - u(t) ≤ e^{-c β} } ≤ e^{β(1 - u_0)} / u_0 + (β² c e) / (β - 1)`.

The analysis is `Metastability.exists_collapse_time`, which exhibits a time
`t ≤ e^{β(1-u_0)}/u_0 + β² c e/(β-1)` at which `1 - u` has already fallen to
`e^{-cβ}`; the infimum above is then at most that `t`.

Source: arXiv:2410.06833v1, §2, `lem: eminem`. -/
lemma eminem
    (β : ℝ) (hβ : 1 < β) (c : ℝ) (hc : 0 < c)
    (u₀ : ℝ) (hu : 0 < u₀ ∧ u₀ ≤ 1)
    (u : ℝ → ℝ)
    (hu_init : u 0 = u₀)
    (hu_ode : ∀ t : ℝ, HasDerivAt u (u t * (1 - u t) * Real.exp (β * (u t - 1))) t) :
    sInf { t : ℝ | 0 ≤ t ∧ 1 - u t ≤ Real.exp (-(c * β)) }
      ≤ Real.exp (β * (1 - u₀)) / u₀
        + β^2 * c * Real.exp 1 / (β - 1) := by
  obtain ⟨t, ht0, htB, htu⟩ :=
    exists_collapse_time β hβ c hc u (by rw [hu_init]; exact hu.1) hu_ode
  rw [hu_init] at htB
  exact le_trans (csInf_le ⟨0, fun s hs => hs.1⟩ ⟨ht0, htu⟩) htB

/-- The hypotheses of `eminem` are satisfiable: `β = 2`, `c = 1`, and the
constant solution `u ≡ 1` of the Cauchy problem at `u₀ = 1`. -/
example : (1 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (0 < (1 : ℝ) ∧ (1 : ℝ) ≤ 1) ∧
    (fun _ : ℝ => (1 : ℝ)) 0 = 1 ∧
    ∀ t : ℝ, HasDerivAt (fun _ : ℝ => (1 : ℝ))
      ((fun _ : ℝ => (1 : ℝ)) t * (1 - (fun _ : ℝ => (1 : ℝ)) t) *
        Real.exp (2 * ((fun _ : ℝ => (1 : ℝ)) t - 1))) t := by
  refine ⟨by norm_num, by norm_num, ⟨by norm_num, le_rfl⟩, rfl, fun t => ?_⟩
  simpa using hasDerivAt_const t (1 : ℝ)

/-- **Lemma (lem:collapsetime) — *Propagation.*

Fix `β > 1`, and let `δ ∈ (0, 1)`, `α ∈ (-1, 1)` satisfy

  `(1/n) δ (1 - δ) e^{-δ β} > n e^{-(1-α) β}`.

If `⟨x_i(0), x_j(0)⟩ ≥ 1 - δ` for all `(i, j) ∈ I²` and
`⟨x_i(t), x_k(t)⟩ ≤ α` for all `i ∈ I`, `k ∈ I^c`, `t ∈ [0, T]`, then

  `⟨x_i(t), x_j(t)⟩ ≥ 1 - δ`  for all `(i, j) ∈ I²` and `t ∈ [0, T]`. -/
lemma propagation
    (β : ℝ) (hβ : 1 < β) (δ α : ℝ)
    (X : ℝ → SphereTuple d n) (hX : Perspective.SA d n β X)
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
        X 0 = X₀ → Perspective.SA d n β X →
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

end Metastability
end Transformer
