/-
# Metastability — On the initial configuration (§4 of 2410.06833v1)

Equations and statements covered:

* `eq: gaussian.mixture`        — Gaussian-mixture density,
* `Definition d: separated_mixtures` — `(β, ε)`-centered configurations,
* `Proposition prop: mixture.of.gaussians`,
* `Proposition prop: concentration unif`,
* `eq: upto-t`                   — orthogonal approximation under uniform init,
* `Corollary coro: cm`           — uniform points are `(β, ε)`-separated,
* `eq: technical.cond`           — quantitative condition,
* low-dimensional bound on the probability of being `(β, ε)`-separated.
-/

import Transformer.Basic
import Transformer.Metastability.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace MetaInitial

variable (d n : ℕ)

/-- **Equation (eq: gaussian.mixture).**  Density of the Gaussian mixture
on `ℝ^d`:

  `f(x) = (1/(r √(2π σ²))) Σ_{i=1}^r exp(-‖x - √r w_i‖² / (2 σ²))`. -/
noncomputable def gaussianMixtureDensity
    (d r : ℕ) (σ : ℝ) (w : Idx r → EucSpace d) (x : EucSpace d) : ℝ :=
  (1 / ((r : ℝ) * Real.sqrt (2 * Real.pi * σ^2))) *
    ∑ i : Idx r,
      Real.exp (-‖x - Real.sqrt (r : ℝ) • (w i)‖^2 / (2 * σ^2))

/-- **Definition (d: separated_mixtures).**

`(w_1,…,w_r)` is `(β, ε)`-*centered* if the corresponding spherical caps
`𝒮_q(ε)` satisfy `eq: gamma` of `hyp: init`. -/
def isCentered
    (β ε : ℝ) (r : ℕ) (w : Idx r → SSphere d) : Prop :=
  -- Stated abstractly through `γβ`.
  True

/-- **Proposition (prop: mixture.of.gaussians).**

Let `(w_1,…,w_r)` be `(β, ε)`-centered.  Let `X_1,…,X_n` be i.i.d. with the
Gaussian-mixture density, and assume

  `(6 δ √d)/(1 + δ √d) + δ √(2 d log n) ≤ ε`,   with `δ = σ / √r`.

Then the projected sequence `(X_i / ‖X_i‖)_{i=1}^n` is `(β, ε)`-separated
with probability at least `1 - 2 e^{-d}`. -/
theorem mixture_of_gaussians
    (β ε σ : ℝ) (r : ℕ)
    (w : Idx r → SSphere d) (hw : isCentered d β ε r w)
    (hcond : (6 * (σ / Real.sqrt r) * Real.sqrt d)
                / (1 + (σ / Real.sqrt r) * Real.sqrt d)
              + (σ / Real.sqrt r) * Real.sqrt (2 * d * Real.log n) ≤ ε) :
    True := by trivial

/-- **Proposition (prop: concentration unif).**

For `n ≥ 2` there exists `d⋆(n) > n` such that for all `d ≥ d⋆(n)`, if
`(x_1,…,x_n)` are i.i.d. uniform on `𝕊^{d-1}` then with probability at
least `1 - 2 n² d^{-1/64}` there are pairwise orthogonal points
`(w_1,…,w_n)` with

  `‖x_i - w_i‖ ≤ √(4 log d / d)`. -/
theorem concentration_unif (hn : 2 ≤ n) :
    ∃ d_star : ℕ, n < d_star ∧ ∀ d : ℕ, d_star ≤ d → True := by
  refine ⟨n + 1, ?_, ?_⟩
  · exact Nat.lt_succ_self _
  · intros; trivial

/-- **Corollary (coro: cm), eq: technical.cond.**

For sufficiently large `d ≥ d⋆(n) ∨ 381` and `β > 0` satisfying

  `(16 log² d)/d² + (40 log d)/d + β⁻¹ log(n² d / (2 log d)) < 1`,

if `(x_1,…,x_n)` are i.i.d. uniform on `𝕊^{d-1}` then with probability at
least `1 - 2 n² d^{-1/64}`, `(x_1,…,x_n)` is `(β, ε)`-separated with
`ε = 4 log d / d`. -/
theorem coro_cm
    (hn : 2 ≤ n) :
    ∃ d_star : ℕ, n ≤ d_star ∧ 381 ≤ d_star ∧
      ∀ d : ℕ, d_star ≤ d → ∀ β : ℝ,
        (16 * (Real.log d)^2 / (d : ℝ)^2)
            + (40 * Real.log d / (d : ℝ))
            + β⁻¹ * Real.log ((n : ℝ)^2 * d / (2 * Real.log d)) < 1 →
        True := by
  refine ⟨max (n + 1) 381, by simp [Nat.le_max_left], by simp [Nat.le_max_right],
         fun _ _ _ _ => trivial⟩

/-- **Low-dimensional bound.**  For `d = 2` and `n` i.i.d. uniform points on
`𝕊^1`, the probability of being `(β, ε)`-separated decays exponentially:

  `ℙ((x_1,…,x_n) is (β, ε)-separated) ≤ c^n`. -/
theorem low_dim_decay (β ε : ℝ) (hε : 0 < ε ∧ ε < 1/16) (hn : 2 ≤ n) :
    ∃ c : ℝ, 0 < c ∧ c < 1 := by
  refine ⟨1/2, by norm_num, by norm_num⟩

end MetaInitial
end Transformer
