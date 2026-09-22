import Mathlib.Analysis.Calculus.IteratedDeriv.Lemmas
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

/-
# The number of modes of a Gaussian KDE — the derivatives of a log-MGF

The cumulants of arXiv:2412.09080v3, §3.1, are mixed derivatives at `0` of
`log 𝔼 e^{⟨u, Y⟩}`.  Differentiating a logarithm produces a quotient and then
a quotient of quotients, and at the origin of a *standardized* law almost all
of it collapses: the moment generating function is `1` there and its first
derivatives vanish.  This module is the one-variable calculus that records the
collapse, stated for a chain of derivative functions rather than for the
moments themselves:

* `iteratedDeriv_two_log` — the general second derivative of `log a`;
* `iteratedDeriv_two_div` — `(b/c)'' = b''` when `c = 1`, `c' = 0`, `b = 0`;
* `iteratedDeriv_three_log` — hence `(log a)''' = a'''` when `a = 1`, `a' = 0`;
* `deriv_mul_sub_sq_div_sq` — `((PQ - R²)/Q²)' = P'` when `Q = 1`, `Q' = 0`,
  `R = 0`, which is the outer derivative of the mixed cumulant `κ^{(1,2)}`.

The hypotheses are `HasDerivAt` on a neighbourhood, never differentiability of
the derivative function, because that is exactly what differentiating under
the integral sign supplies.

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`.
-/

open Real Filter
open scoped Topology

namespace Transformer
namespace Modes

variable {a a₁ a₂ a₃ b b₁ b₂ c c₁ c₂ : ℝ → ℝ} {x : ℝ}

/-- **The second derivative of a logarithm**, `(log a)'' = (a''a - a'a')/a²`.

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
theorem iteratedDeriv_two_log (ha : ∀ᶠ t in 𝓝 x, HasDerivAt a (a₁ t) t)
    (ha₁ : HasDerivAt a₁ (a₂ x) x) (hpos : ∀ᶠ t in 𝓝 x, 0 < a t) :
    iteratedDeriv 2 (fun t => Real.log (a t)) x
      = (a₂ x * a x - a₁ x * a₁ x) / (a x * a x) := by
  have key : deriv (fun t => Real.log (a t)) =ᶠ[𝓝 x] fun t => a₁ t / a t := by
    filter_upwards [ha, hpos] with t hta htp
    exact (hta.log htp.ne').deriv
  have hax : HasDerivAt a (a₁ x) x := ha.self_of_nhds
  have hne : a x ≠ 0 := hpos.self_of_nhds.ne'
  rw [iteratedDeriv_succ, iteratedDeriv_one, key.deriv_eq, (ha₁.fun_div hax hne).deriv, pow_two]

/-- The hypotheses of `iteratedDeriv_two_log` are satisfiable: the constant
function `1`. -/
example : (∀ᶠ t in 𝓝 (0 : ℝ), HasDerivAt (fun _ : ℝ => (1 : ℝ)) ((fun _ : ℝ => (0 : ℝ)) t) t) ∧
    HasDerivAt (fun _ : ℝ => (0 : ℝ)) ((fun _ : ℝ => (0 : ℝ)) 0) 0 ∧
    (∀ᶠ t in 𝓝 (0 : ℝ), 0 < (fun _ : ℝ => (1 : ℝ)) t) :=
  ⟨Eventually.of_forall fun _ => hasDerivAt_const _ _, hasDerivAt_const _ _,
    Eventually.of_forall fun _ => one_pos⟩

/-- **The second derivative of a quotient at a normalized point.**  Where the
denominator is `1` with vanishing derivative and the numerator vanishes,
`(b/c)'' = b''`.

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
theorem iteratedDeriv_two_div (hb : ∀ᶠ t in 𝓝 x, HasDerivAt b (b₁ t) t)
    (hb₁ : HasDerivAt b₁ (b₂ x) x) (hc : ∀ᶠ t in 𝓝 x, HasDerivAt c (c₁ t) t)
    (hc₁ : HasDerivAt c₁ (c₂ x) x) (hc0 : ∀ᶠ t in 𝓝 x, c t ≠ 0) (hcx : c x = 1)
    (hc'x : c₁ x = 0) (hbx : b x = 0) :
    iteratedDeriv 2 (fun t => b t / c t) x = b₂ x := by
  have key : deriv (fun t => b t / c t) =ᶠ[𝓝 x]
      fun t => (b₁ t * c t - b t * c₁ t) / (c t * c t) := by
    filter_upwards [hb, hc, hc0] with t htb htc htc0
    rw [(htb.fun_div htc htc0).deriv, pow_two]
  have hbx' : HasDerivAt b (b₁ x) x := hb.self_of_nhds
  have hcx' : HasDerivAt c (c₁ x) x := hc.self_of_nhds
  have hne : c x * c x ≠ 0 := by rw [hcx]; norm_num
  rw [iteratedDeriv_succ, iteratedDeriv_one, key.deriv_eq,
    (((hb₁.fun_mul hcx').fun_sub (hbx'.fun_mul hc₁)).fun_div (hcx'.fun_mul hcx') hne).deriv,
    hcx, hc'x, hbx]
  ring

/-- The hypotheses of `iteratedDeriv_two_div` are satisfiable: `b = 0`,
`c = 1`. -/
example : (∀ᶠ t in 𝓝 (0 : ℝ), HasDerivAt (fun _ : ℝ => (0 : ℝ)) ((fun _ : ℝ => (0 : ℝ)) t) t) ∧
    HasDerivAt (fun _ : ℝ => (0 : ℝ)) ((fun _ : ℝ => (0 : ℝ)) 0) 0 ∧
    (∀ᶠ t in 𝓝 (0 : ℝ), (fun _ : ℝ => (1 : ℝ)) t ≠ 0) ∧
    (fun _ : ℝ => (1 : ℝ)) 0 = 1 ∧ (fun _ : ℝ => (0 : ℝ)) 0 = 0 :=
  ⟨Eventually.of_forall fun _ => hasDerivAt_const _ _, hasDerivAt_const _ _,
    Eventually.of_forall fun _ => one_ne_zero, rfl, rfl⟩

/-- **The third derivative of a logarithm at a normalized point.**  Along a
chain `a, a₁, a₂, a₃` of derivatives, with `a` positive, `a x = 1` and
`a₁ x = 0`, the third derivative of `log a` is `a₃ x`.

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
theorem iteratedDeriv_three_log (ha : ∀ᶠ t in 𝓝 x, HasDerivAt a (a₁ t) t)
    (ha₁ : ∀ᶠ t in 𝓝 x, HasDerivAt a₁ (a₂ t) t) (ha₂ : HasDerivAt a₂ (a₃ x) x)
    (hpos : ∀ᶠ t in 𝓝 x, 0 < a t) (h0 : a x = 1) (h1 : a₁ x = 0) :
    iteratedDeriv 3 (fun t => Real.log (a t)) x = a₃ x := by
  have key : deriv (fun t => Real.log (a t)) =ᶠ[𝓝 x] fun t => a₁ t / a t := by
    filter_upwards [ha, hpos] with t hta htp
    exact (hta.log htp.ne').deriv
  rw [iteratedDeriv_succ', key.iteratedDeriv_eq 2]
  exact iteratedDeriv_two_div ha₁ ha₂ ha ha₁.self_of_nhds (hpos.mono fun t ht => ht.ne') h0 h1 h1

/-- The hypotheses of `iteratedDeriv_three_log` are satisfiable: the constant
function `1`. -/
example : (∀ᶠ t in 𝓝 (0 : ℝ), HasDerivAt (fun _ : ℝ => (1 : ℝ)) ((fun _ : ℝ => (0 : ℝ)) t) t) ∧
    (∀ᶠ t in 𝓝 (0 : ℝ), HasDerivAt (fun _ : ℝ => (0 : ℝ)) ((fun _ : ℝ => (0 : ℝ)) t) t) ∧
    HasDerivAt (fun _ : ℝ => (0 : ℝ)) ((fun _ : ℝ => (0 : ℝ)) 0) 0 ∧
    (∀ᶠ t in 𝓝 (0 : ℝ), 0 < (fun _ : ℝ => (1 : ℝ)) t) ∧
    (fun _ : ℝ => (1 : ℝ)) 0 = 1 ∧ (fun _ : ℝ => (0 : ℝ)) 0 = 0 :=
  ⟨Eventually.of_forall fun _ => hasDerivAt_const _ _,
    Eventually.of_forall fun _ => hasDerivAt_const _ _, hasDerivAt_const _ _,
    Eventually.of_forall fun _ => one_pos, rfl, rfl⟩

/-- **The derivative at a normalized point of the second-derivative-of-log
expression**, `((PQ - R²)/Q²)' = P'` when `Q = 1`, `Q' = 0` and `R = 0`.

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
theorem deriv_mul_sub_sq_div_sq {P P₁ Q Q₁ R R₁ : ℝ → ℝ} (hP : HasDerivAt P (P₁ x) x)
    (hQ : HasDerivAt Q (Q₁ x) x) (hR : HasDerivAt R (R₁ x) x) (hQx : Q x = 1) (hQ'x : Q₁ x = 0)
    (hRx : R x = 0) :
    deriv (fun t => (P t * Q t - R t * R t) / (Q t * Q t)) x = P₁ x := by
  have hne : Q x * Q x ≠ 0 := by rw [hQx]; norm_num
  rw [(((hP.fun_mul hQ).fun_sub (hR.fun_mul hR)).fun_div (hQ.fun_mul hQ) hne).deriv, hQx, hQ'x,
    hRx]
  ring

/-- The hypotheses of `deriv_mul_sub_sq_div_sq` are satisfiable: `P = R = 0`,
`Q = 1`. -/
example : HasDerivAt (fun _ : ℝ => (0 : ℝ)) ((fun _ : ℝ => (0 : ℝ)) 0) 0 ∧
    HasDerivAt (fun _ : ℝ => (1 : ℝ)) ((fun _ : ℝ => (0 : ℝ)) 0) 0 ∧
    (fun _ : ℝ => (1 : ℝ)) 0 = 1 ∧ (fun _ : ℝ => (0 : ℝ)) 0 = 0 :=
  ⟨hasDerivAt_const _ _, hasDerivAt_const _ _, rfl, rfl⟩

end Modes
end Transformer
