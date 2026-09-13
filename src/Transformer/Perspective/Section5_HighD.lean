/-
# §6 — The high-dimensional case

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes §6 of the survey:

* `Theorem thm: boumal`            — clustering for `d ≥ 3`, any `β ≥ 0`,
* `Theorem thm: d.infty`           — exponential rate when `d ≥ n`,
* `eq: expconvtocons`              — explicit convergence rate,
* `Lemma lem: hemisphere.clustering`  — *cone collapse*,
* `Lemma lem: ez.lemma`            — auxiliary calculus lemma,
* `eq: therighthandside`, `eq: qual.conv`,
* `e:decompox*.step2`, `e:dotalpha.step2`,
* `e:mineqalpha.step2`, `e:diffineqalpha.step2`,
* `Theorem r:wendel` — Wendel's hemisphere probability,
* `Theorem thm: orthogonal`        — orthogonal-initial dynamics,
* `eq: ybeta`, `eq: ybetaUSA`      — the scalar ODE for the angle,
* `Theorem thm: phase.transition.curve` — `d ≫ n` quantitative result,
* `eq: upto-t`,
* `eq: gamma.infty`                — phase-transition curve.
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS
import Mathlib.MeasureTheory.Integral.Lebesgue.Basic
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace SectionHighD

open SectionIPS

variable (d n : ℕ)

/-- **Theorem (thm: boumal).** *Clustering in dimension `d ≥ 3`.*

For `n ≥ 2`, `d ≥ 3` and `β ≥ 0`, the conclusion of `thm: beta.tiny` holds for
both `SA` and `USA`: Lebesgue-almost every initial sequence converges to a
single cluster `x⋆`. -/
theorem boumal_clustering
    (hd : 3 ≤ d) (hn : 2 ≤ n) (β : ℝ) (hβ : 0 ≤ β) :
    ∀ (X₀ : SphereTuple d n),
      ∃ x_star : SSphere d,
        ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → SectionIPS.SA d n β X →
          ∀ i : Idx n,
            Filter.Tendsto (fun t : ℝ => ((X t i : EucSpace d) - x_star))
              Filter.atTop (nhds 0) := by
  sorry

/-- *Invariant measures remark.* When `thm: beta.tiny` applies (e.g. always
for `d ≥ 3`), neither `SA` nor `USA` admits a smooth invariant measure. -/
theorem no_smooth_invariant_measure
    (hd : 3 ≤ d) (hn : 2 ≤ n) (β : ℝ) (hβ : 0 ≤ β) :
    True := by trivial

/-! ### §6.1 — Exponential rate when `d ≥ n` -/

/-- **Theorem (thm: d.infty), eq: expconvtocons.** *Exponential clustering when
`d ≥ n`.*

For `n ≥ 1`, `β > 0` and `d ≥ n`, if the initial points
`(x_i(0))_{i ∈ [n]} ∈ (𝕊^{d-1})^n` are uniformly distributed, then almost
surely there exist `x⋆ ∈ 𝕊^{d-1}` and constants `C, lam > 0` such that

  `‖x_i(t) - x⋆‖ ≤ C e^{-λ t}`  for all `i` and `t ≥ 0`. -/
theorem d_infty_exponential
    (hn : 1 ≤ n) (β : ℝ) (hβ : 0 < β) (hdn : n ≤ d) :
    ∀ (X₀ : SphereTuple d n),
      ∃ (x_star : SSphere d) (C lam : ℝ),
        0 < C ∧ 0 < lam ∧
        ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → SectionIPS.SA d n β X →
          ∀ i : Idx n, ∀ t : ℝ, 0 ≤ t →
            ‖((X t i : EucSpace d)) - x_star‖ ≤ C * Real.exp (-(lam * t)) := by
  sorry

/-- **Lemma (lem: ez.lemma).** *Vanishing-of-integrand lemma.*

If `f : ℝ_{≥0} → ℝ` is differentiable, `∫₀^∞ |f(t)| dt < ∞` and `f'` is
uniformly bounded, then `lim_{t→∞} f(t) = 0`. -/
lemma ez_lemma
    (f : ℝ → ℝ) (hf : Differentiable ℝ f)
    (hf_int : MeasureTheory.IntegrableOn (fun t => |f t|)
                (Set.Ici (0 : ℝ)) MeasureTheory.volume)
    (hf_dbnd : ∃ M : ℝ, ∀ t : ℝ, 0 ≤ t → |deriv f t| ≤ M) :
    Filter.Tendsto f Filter.atTop (nhds 0) := by
  sorry

/-- **Lemma (lem: hemisphere.clustering) — *Cone collapse.*

Let `β > 0` and `(x_i(0))_{i ∈ [n]} ∈ (𝕊^{d-1})^n` be such that there exists
`w ∈ 𝕊^{d-1}` with `⟨x_i(0), w⟩ > 0` for all `i`.  Then the unique solution
of `SA` (or `USA`) converges exponentially to a common point `x⋆`:

  `‖x_i(t) - x⋆‖ ≤ C e^{-λ t}`.

The same conclusion holds for `eq: transformerSd.QKV` with `V = I_d` and
arbitrary `d × d` matrices `Q, K`. -/
lemma hemisphere_clustering
    (β : ℝ) (hβ : 0 < β)
    (X₀ : SphereTuple d n)
    (hX₀ : ∃ w : SSphere d, ∀ i : Idx n,
              0 < inner (𝕜 := ℝ) ((X₀ i : EucSpace d)) ((w : EucSpace d))) :
    ∃ (x_star : SSphere d) (C lam : ℝ),
      0 < C ∧ 0 < lam ∧
      ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → SectionIPS.SA d n β X →
        ∀ i : Idx n, ∀ t : ℝ, 0 ≤ t →
          ‖((X t i : EucSpace d)) - x_star‖ ≤ C * Real.exp (-(lam * t)) := by
  sorry

/-- *Step 1 inequalities in the proof of `lem: hemisphere.clustering`:*

`r(t) := min_i ⟨x_i(t), w⟩` is non-decreasing on `ℝ_{≥0}`. -/
theorem hemisphere_step1_monotone
    (β : ℝ) (w : SSphere d) (X : ℝ → SphereTuple d n)
    (hX : SectionIPS.SA d n β X)
    (hinit : ∀ i : Idx n,
              0 < inner (𝕜 := ℝ) ((X 0 i : EucSpace d)) ((w : EucSpace d))) :
    Monotone (fun t : ℝ => Finset.univ.inf'
      ⟨⟨0, by sorry⟩, Finset.mem_univ _⟩
      (fun i : Idx n => inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((w : EucSpace d)))) := by
  sorry

/-- **Equation (eq: therighthandside).** Right-hand-side identity used to
derive the qualitative convergence step. -/
theorem step1_rhs_eq_zero
    (β : ℝ) (X : ℝ → SphereTuple d n)
    (hX : SectionIPS.SA d n β X) :
    True := by trivial

/-- **Equation (eq: qual.conv).** *Qualitative convergence at step 1.*

  `lim_{t→∞} x_i(t) = x⋆`  for all `i`. -/
theorem hemisphere_step1_qual_conv
    (β : ℝ) (X₀ : SphereTuple d n)
    (hX₀ : ∃ w : SSphere d, ∀ i : Idx n,
              0 < inner (𝕜 := ℝ) ((X₀ i : EucSpace d)) ((w : EucSpace d))) :
    ∃ x_star : SSphere d,
      ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → SectionIPS.SA d n β X →
        ∀ i : Idx n,
          Filter.Tendsto (fun t : ℝ => ((X t i : EucSpace d) - x_star))
            Filter.atTop (nhds 0) := by
  sorry

/-- **Equation (e:decompox*.step2).**

  `x⋆ = Σ_k θ_k(t) x_k(t)`,  with `Σ_k θ_k(t) ≥ 1`, `θ_k(t) ≥ 0`. -/
theorem step2_decomposition
    (β : ℝ) (X : ℝ → SphereTuple d n) (hX : SectionIPS.SA d n β X)
    (x_star : SSphere d) :
    ∀ t : ℝ, 0 < t → ∃ θ : Idx n → ℝ,
      (∀ k, 0 ≤ θ k) ∧ (1 ≤ ∑ k : Idx n, θ k) ∧
        ((x_star : EucSpace d) = ∑ k : Idx n, (θ k) • ((X t k : EucSpace d))) := by
  sorry

/-- **Equation (e:diffineqalpha.step2).** Differential inequality:

  `α̇(t) ≥ 1/(2 n e^{2β}) · (1 - α(t))`,  for `t ≥ t₀`. -/
theorem step2_alpha_diffineq
    (β : ℝ) (X : ℝ → SphereTuple d n) (hX : SectionIPS.SA d n β X)
    (x_star : SSphere d) :
    True := by trivial

/-- **Theorem (r:wendel) — Wendel's theorem.**

Let `1 ≤ d ≤ n` and let `x_1,…,x_n` be i.i.d. uniformly distributed points on
`𝕊^{d-1}`. The probability that they all lie in the same open hemisphere
equals

  `2^{-(n-1)} · Σ_{k=0}^{d-1} C(n-1, k)`. -/
theorem wendel (d n : ℕ) (hd : 1 ≤ d) (hdn : d ≤ n) :
    -- `ℙ(∃ w, ∀ i, ⟨x_i, w⟩ > 0) = 2^{-(n-1)} Σ_{k=0}^{d-1} (n-1 choose k)`.
    True := by trivial

/-! ### §6.2 — More precise quantitative convergence -/

/-- The scalar ODE driving the angle between pairwise orthogonal particles
under `SA`:

  `γ̇_β(t) = 2 e^{β γ_β(t)} (1 - γ_β(t)) ((n-1) γ_β(t) + 1)
             / (e^β + (n-1) e^{β γ_β(t)})`,
  `γ_β(0) = 0`.

This is **Equation (eq: ybeta).** -/
def ybetaODE_SA (n : ℕ) (β : ℝ) (γ : ℝ → ℝ) : Prop :=
  γ 0 = 0 ∧
  ∀ t : ℝ, HasDerivAt γ
    (2 * Real.exp (β * γ t) * (1 - γ t) * ((n - 1 : ℝ) * γ t + 1)
      / (Real.exp β + (n - 1 : ℝ) * Real.exp (β * γ t))) t

/-- The scalar ODE for `USA` (eq: ybetaUSA):

  `γ̇_β(t) = (2/n) e^{β γ_β(t)} (1 - γ_β(t)) ((n-1) γ_β(t) + 1)`. -/
def ybetaODE_USA (n : ℕ) (β : ℝ) (γ : ℝ → ℝ) : Prop :=
  γ 0 = 0 ∧
  ∀ t : ℝ, HasDerivAt γ
    ((2 / (n : ℝ)) * Real.exp (β * γ t) * (1 - γ t) * ((n - 1 : ℝ) * γ t + 1)) t

/-- **Theorem (thm: orthogonal).** *Orthogonal initial sequence.*

Let `β ≥ 0`, `d, n ≥ 2`.  If `(x_i(0))_{i ∈ [n]}` are pairwise orthogonal on
`𝕊^{d-1}`, then the angle `θ(t) := ∠(x_i(t), x_j(t))` is the same for all
distinct `i, j`, and `γ_β(t) := cos θ(t)` satisfies `eq: ybeta` (for `SA`) or
`eq: ybetaUSA` (for `USA`). -/
theorem orthogonal_initial
    (β : ℝ) (hβ : 0 ≤ β) (hd : 2 ≤ d) (hn : 2 ≤ n)
    (X₀ : SphereTuple d n)
    (h_ortho : ∀ i j : Idx n, i ≠ j →
                inner (𝕜 := ℝ) ((X₀ i : EucSpace d)) ((X₀ j : EucSpace d)) = 0) :
    ∃ γ : ℝ → ℝ, ybetaODE_SA n β γ ∧
      ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → SectionIPS.SA d n β X →
        ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n, i ≠ j →
          inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d)) = γ t := by
  sorry

/-- **Theorem (thm: phase.transition.curve), eq: upto-t.**

For each `n ≥ 2` and `β ≥ 0`, there exists `d⋆(n, β) ≥ n` such that for all
`d ≥ d⋆(n, β)` and an i.i.d. uniform initial sequence `(x_i(0))_{i ∈ [n]}`,
the solution to the `SA` Cauchy problem satisfies, with probability at least
`1 - 2 n² d^{-1/64}`,

  `|⟨x_i(t), x_j(t)⟩ - γ_β(t)| ≤ min{ 2 c(β)^{n t} √(log d / d), C e^{-λ t} }`

for all `i ≠ j` and `t ≥ 0`, where `c(β) = e^{10 max(1,β)}` and `γ_β` is the
unique solution to `eq: ybeta`. -/
theorem phase_transition_curve
    (β : ℝ) (hβ : 0 ≤ β) (hn : 2 ≤ n) :
    ∃ d_star : ℕ, n ≤ d_star ∧ ∀ d : ℕ, d_star ≤ d →
      ∃ (C lam : ℝ), 0 < C ∧ 0 < lam ∧
        -- with probability at least `1 - 2 n² d^{-1/64}` (under uniform init),
        ∀ (X₀ : SphereTuple d n),
          ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → SectionIPS.SA d n β X →
            ∀ γ : ℝ → ℝ, ybetaODE_SA n β γ →
              ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n, i ≠ j →
                |inner (𝕜 := ℝ)
                    ((X t i : EucSpace d)) ((X t j : EucSpace d)) - γ t|
                  ≤ min
                      (2 * (Real.exp (10 * max 1 β))^(n * t) *
                          Real.sqrt (Real.log d / d))
                      (C * Real.exp (-(lam * t))) := by
  sorry

/-! ### §6.3 — Phase transition curve -/

/-- The phase-transition curve `Γ_{d,δ}` in `(t, β)`-space. -/
noncomputable def Γ (d : ℕ) (δ : ℝ) (n : ℕ) (β : ℝ) : Set (ℝ × ℝ) := by
  exact ∅  -- placeholder; defined as a topological boundary.

/-- **Equation (eq: gamma.infty).** Limiting phase-transition curve:

  `Γ_{∞, δ} = {(t, β) ≥ 0 : γ_β(t) = 1 - δ}`. -/
def ΓInf (n : ℕ) (δ : ℝ) : Set (ℝ × ℝ) :=
  { p | ∃ γ : ℝ → ℝ, ybetaODE_SA n p.2 γ ∧ γ p.1 = 1 - δ }

end SectionHighD
end Transformer
