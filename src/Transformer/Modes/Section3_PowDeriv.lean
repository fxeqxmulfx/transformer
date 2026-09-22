import Mathlib.Analysis.Calculus.IteratedDeriv.Lemmas
import Mathlib.Analysis.Calculus.Deriv.Pow

/-
# The number of modes of a Gaussian KDE — third derivatives along a chain

The moment generating function of a sum of `n` i.i.d. summands is the `n`-th
power of the moment generating function of one of them, and the third moment of
the sum is read off the third derivative of that power at `0`.  Two pieces of
one-variable calculus do it, both stated for a *chain* `F, F₁, F₂, F₃` of
derivative functions rather than for a function known to be three times
differentiable, because differentiating under the integral sign supplies
exactly such a chain and nothing more:

* `iteratedDeriv_three_of_chain` — `F''' = F₃` along a chain;
* `iteratedDeriv_three_pow` — `(Gⁿ)''' = n G₃` at a point where `G = 1` and
  `G' = 0`, the normalization a moment generating function has at the origin
  when the law is centred.  Every term of the third derivative of a power
  carries a factor `G'` except the last one.

Source: arXiv:2412.09080v3, §3.1, the display after `eq:psi`.
-/

open Filter
open scoped Topology

namespace Transformer
namespace Modes

variable {F F₁ F₂ F₃ G G₁ G₂ G₃ : ℝ → ℝ} {x : ℝ}

/-- **The third derivative along a chain of derivative functions.**  If `F₁` is
a derivative of `F` near `x`, `F₂` one of `F₁` near `x`, and `F₃ x` one of `F₂`
at `x`, then `iteratedDeriv 3 F x = F₃ x`.

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
theorem iteratedDeriv_three_of_chain (h0 : ∀ᶠ t in 𝓝 x, HasDerivAt F (F₁ t) t)
    (h1 : ∀ᶠ t in 𝓝 x, HasDerivAt F₁ (F₂ t) t) (h2 : HasDerivAt F₂ (F₃ x) x) :
    iteratedDeriv 3 F x = F₃ x := by
  have e0 : deriv F =ᶠ[𝓝 x] F₁ := h0.mono fun _ ht => ht.deriv
  have e1 : deriv F₁ =ᶠ[𝓝 x] F₂ := h1.mono fun _ ht => ht.deriv
  rw [iteratedDeriv_succ', e0.iteratedDeriv_eq 2, iteratedDeriv_succ', e1.iteratedDeriv_eq 1,
    iteratedDeriv_one, h2.deriv]

/-- The hypotheses of `iteratedDeriv_three_of_chain` are satisfiable: the
constant function `0`. -/
example : (∀ᶠ t in 𝓝 (0 : ℝ), HasDerivAt (fun _ : ℝ => (0 : ℝ)) ((fun _ : ℝ => (0 : ℝ)) t) t) ∧
    (∀ᶠ t in 𝓝 (0 : ℝ), HasDerivAt (fun _ : ℝ => (0 : ℝ)) ((fun _ : ℝ => (0 : ℝ)) t) t) ∧
    HasDerivAt (fun _ : ℝ => (0 : ℝ)) ((fun _ : ℝ => (0 : ℝ)) 0) 0 :=
  ⟨Eventually.of_forall fun _ => hasDerivAt_const _ _,
    Eventually.of_forall fun _ => hasDerivAt_const _ _, hasDerivAt_const _ _⟩

/-- **The third derivative of an `n`-th power at a normalized point.**  Along a
chain `G, G₁, G₂, G₃` of derivative functions with `G x = 1` and `G₁ x = 0`,
`iteratedDeriv 3 (Gⁿ) x = n · G₃ x`: the three other terms of the third
derivative of a power each carry a factor `G₁ x`.

This is the additivity of the third cumulant over independent summands, in the
only form used here.

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
theorem iteratedDeriv_three_pow (n : ℕ) (h0 : ∀ᶠ t in 𝓝 x, HasDerivAt G (G₁ t) t)
    (h1 : ∀ᶠ t in 𝓝 x, HasDerivAt G₁ (G₂ t) t) (h2 : HasDerivAt G₂ (G₃ x) x)
    (hG : G x = 1) (hG₁ : G₁ x = 0) :
    iteratedDeriv 3 (fun t => G t ^ n) x = (n : ℝ) * G₃ x := by
  have h0x : HasDerivAt G (G₁ x) x := h0.self_of_nhds
  have h1x : HasDerivAt G₁ (G₂ x) x := h1.self_of_nhds
  refine iteratedDeriv_three_of_chain (F₁ := fun t => (n : ℝ) * G t ^ (n - 1) * G₁ t)
    (F₂ := fun t => (n : ℝ) * (((n - 1 : ℕ) : ℝ) * G t ^ (n - 1 - 1) * G₁ t) * G₁ t
      + (n : ℝ) * G t ^ (n - 1) * G₂ t)
    (F₃ := fun _ => (n : ℝ) * G₃ x) ?_ ?_ ?_
  · filter_upwards [h0] with t ht using ht.fun_pow n
  · filter_upwards [h0, h1] with t ht ht1
    exact (((ht.fun_pow (n - 1)).const_mul (n : ℝ)).fun_mul ht1).congr_deriv (by ring)
  · have hA : HasDerivAt (fun t => (n : ℝ) * (((n - 1 : ℕ) : ℝ) * G t ^ (n - 1 - 1) * G₁ t) * G₁ t)
        0 x := by
      refine (((((h0x.fun_pow (n - 1 - 1)).const_mul (((n - 1 : ℕ) : ℝ))).fun_mul
        h1x).const_mul (n : ℝ)).fun_mul h1x).congr_deriv ?_
      rw [hG₁]; ring
    have hB : HasDerivAt (fun t => (n : ℝ) * G t ^ (n - 1) * G₂ t) ((n : ℝ) * G₃ x) x := by
      refine (((h0x.fun_pow (n - 1)).const_mul (n : ℝ)).fun_mul h2).congr_deriv ?_
      rw [hG, hG₁, one_pow]; ring
    simpa using hA.fun_add hB

/-- The hypotheses of `iteratedDeriv_three_pow` are satisfiable: the constant
function `1`, whose chain of derivatives is `0`. -/
example : (∀ᶠ t in 𝓝 (0 : ℝ), HasDerivAt (fun _ : ℝ => (1 : ℝ)) ((fun _ : ℝ => (0 : ℝ)) t) t) ∧
    (∀ᶠ t in 𝓝 (0 : ℝ), HasDerivAt (fun _ : ℝ => (0 : ℝ)) ((fun _ : ℝ => (0 : ℝ)) t) t) ∧
    HasDerivAt (fun _ : ℝ => (0 : ℝ)) ((fun _ : ℝ => (0 : ℝ)) 0) 0 ∧
    (fun _ : ℝ => (1 : ℝ)) 0 = 1 ∧ (fun _ : ℝ => (0 : ℝ)) 0 = 0 :=
  ⟨Eventually.of_forall fun _ => hasDerivAt_const _ _,
    Eventually.of_forall fun _ => hasDerivAt_const _ _, hasDerivAt_const _ _, rfl, rfl⟩

end Modes
end Transformer
