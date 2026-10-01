/-
# Real analytic preparation: Basic

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d:
https://github.com/BochaoKong/classical-complex-wpt
Only the existence proof and its coefficient infrastructure are copied.
The scalar field is real; no complex root or complex differentiation result
is used. Apache-2.0 license: third_party/classical-complex-wpt/LICENSE.
-/

import Mathlib.Analysis.Analytic.Constructions
import Mathlib.Analysis.Calculus.IteratedDeriv.Defs
import Mathlib.Analysis.Analytic.Order

open Filter Finset
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
set_option maxHeartbeats 5000000

namespace Transformer.AnalyticPreparation

variable {A I J : Type*} [AddCommMonoid A] [Finset.HasAntidiagonal A]

/-- The real parameter space `ℝⁿ`. -/
abbrev Base (n : ℕ) := Fin n → ℝ

/-- The parameter space together with the distinguished real variable. -/
abbrev Ambient (n : ℕ) := Base n × ℝ

/-- The additive zero in the ambient product is the pair of coordinate zeros.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem ambient_zero_eq (n : ℕ) : (0 : Ambient n) = ((0 : Base n), 0) := by
  ext <;> simp

/-- Restriction of a function to the distinguished-variable axis. -/
def lastSlice {n : ℕ} (f : Ambient n → ℝ) : ℝ → ℝ :=
  fun w ↦ f (0, w)

/--
The distinguished-variable slice has a zero of exact order `d` at the origin:
all derivatives of order below `d` vanish and the derivative of order `d` does
not vanish.
-/
def ExactOrderInLastVariable {n : ℕ} (f : Ambient n → ℝ) (d : ℕ) : Prop :=
  (∀ k < d, iteratedDeriv k (lastSlice f) 0 = 0) ∧
    iteratedDeriv d (lastSlice f) 0 ≠ 0

/-- The monic degree-`d` polynomial in the distinguished variable. -/
def preparedPolynomial {n : ℕ} (d : ℕ) (a : Fin d → Base n → ℝ)
    (x : Ambient n) : ℝ :=
  x.2 ^ d + ∑ i : Fin d, a i x.1 * x.2 ^ (i : ℕ)

/--
`f` is locally the product of a nonvanishing analytic unit and a monic
distinguished-variable polynomial whose lower coefficients are analytic and
vanish at the base origin.
-/
def IsWeierstrassPreparation {n : ℕ} (f : Ambient n → ℝ) (d : ℕ)
    (a : Fin d → Base n → ℝ) (u : Ambient n → ℝ) : Prop :=
  (∀ i, AnalyticAt ℝ (a i) 0) ∧
    (∀ i, a i 0 = 0) ∧
    AnalyticAt ℝ u 0 ∧
    u 0 ≠ 0 ∧
    f =ᶠ[𝓝 0] fun x ↦ u x * preparedPolynomial d a x

end Transformer.AnalyticPreparation
