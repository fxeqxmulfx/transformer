/-
# Measure-to-measure interpolation — The flow is the identity off the support

Formalization of `eq: identity.flow` of §3 of arXiv:2411.04551v3.

The paper's statement is `Φ^T(x) = x` for every
`x ∈ 𝕊^{d-1} \ (conv_g supp μ_0 ∪ conv_g supp ν_0)`.  What makes it true in
the construction of §3 is not the geometry of the geodesic hulls — which this
development does not define — but the fact that the parameters are chosen so
that the vector field `eq: vf` vanishes identically off those hulls: `𝐕` is
switched off, and the affine form `𝐔 x + b` is nonpositive in every
coordinate, so the ReLU kills the perceptron term as well.

That implication is what is proved here, in two steps: the field vanishes
under those two conditions, and a characteristic along which the field
vanishes is constant.  The geometric input — that the two conditions hold
exactly off the hulls — is the part that is not formalized.
-/

import Transformer.Basic
import Transformer.Interpolation.Basic
import Mathlib.Analysis.Calculus.MeanValue

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Interpolation

open Interpolation Perspective

variable (d : ℕ)

/-- The vector field `eq: vf` vanishes at `x` as soon as `𝐕(t) = 0` and every
coordinate of `𝐔(t) x + b(t)` is `≤ 0`: the attention term is multiplied by
`𝐕`, and the ReLU turns the perceptron term into `𝐖(t) · 0`.

Source: arXiv:2411.04551v3, §3 (the hypotheses of `eq: identity.flow`). -/
theorem fullVF_eq_zero
    (θ : TimeParams d) (μ : ProbSphere d) (t : ℝ) (x : EucSpace d)
    (hV : (θ t).V = 0)
    (hb : ∀ i : Fin d, (EuclideanSpace.equiv (Fin d) ℝ ((θ t).U x + (θ t).b)) i ≤ 0) :
    fullVF d θ μ t x = 0 := by
  have hrelu :
      (fun i : Fin d =>
        max ((EuclideanSpace.equiv (Fin d) ℝ ((θ t).U x + (θ t).b)) i) 0) = 0 := by
    funext i
    simpa using max_eq_right (hb i)
  have hz : (EuclideanSpace.equiv (Fin d) ℝ).symm
      (fun i : Fin d =>
        max ((EuclideanSpace.equiv (Fin d) ℝ ((θ t).U x + (θ t).b)) i) 0) = 0 := by
    rw [hrelu, map_zero]
  simp only [fullVF, hV, zero_apply, zero_add, hz, map_zero]
  simp [proj]

/-- **Equation (eq: identity.flow).**  A characteristic of `eq: cauchy.pb`
that stays in the region where `𝐕(t) = 0` and `(𝐔(t) x + b(t))_+ = 0` does not
move at all: `Φ^T(x) = x`.

Source: arXiv:2411.04551v3, §3. -/
theorem identity_flow
    (θ : TimeParams d) (μ : ℝ → ProbSphere d) (x : ℝ → EucSpace d) (T : ℝ)
    (hx : ∀ t : ℝ, HasDerivAt x (fullVF d θ (μ t) t (x t)) t)
    (hV : ∀ t : ℝ, (θ t).V = 0)
    (hb : ∀ t : ℝ, ∀ i : Fin d,
      (EuclideanSpace.equiv (Fin d) ℝ ((θ t).U (x t) + (θ t).b)) i ≤ 0) :
    x T = x 0 := by
  have hzero : ∀ t : ℝ, HasDerivAt x 0 t := fun t =>
    (hx t).congr_deriv (fullVF_eq_zero d θ (μ t) t (x t) (hV t) (hb t))
  exact is_const_of_deriv_eq_zero (fun t => (hzero t).differentiableAt)
    (fun t => (hzero t).deriv) T 0

/-- The hypotheses are satisfiable: with all parameters zero the field is zero
everywhere, and a constant curve is a characteristic of it. -/
example (y : SSphere d) (x₀ : EucSpace d) :
    ∃ (θ : TimeParams d) (μ : ℝ → ProbSphere d),
      (∀ t : ℝ, HasDerivAt (fun _ : ℝ => x₀) (fullVF d θ (μ t) t x₀) t) ∧
        (∀ t : ℝ, (θ t).V = 0) ∧
        (∀ t : ℝ, ∀ i : Fin d,
          (EuclideanSpace.equiv (Fin d) ℝ ((θ t).U x₀ + (θ t).b)) i ≤ 0) := by
  refine ⟨fun _ => { V := 0, B := 0, W := 0, U := 0, b := 0 },
    fun _ => ⟨Measure.dirac y, inferInstance⟩, fun t => ?_, fun _ => rfl, fun _ i => by simp⟩
  exact (hasDerivAt_const t x₀).congr_deriv
    (fullVF_eq_zero d _ _ t x₀ rfl (fun i => by simp)).symm

end Interpolation
end Transformer
