/-
# Measure-to-measure interpolation — Mass concentration estimates

Formalization of the two technical claims of arXiv:2411.04551v3 that turn a
Wasserstein bound into a statement about mass:

* `Lemma lem: mass.concentration.Q1` — a measure that puts `1 - η` of its mass
  in the positive quadrant is `O(η)`-close to a measure carried by it,
* `Claim cl: W.to.ball`             — a measure `W₂`-close to a Dirac mass puts
  almost all of its mass in a small ball.

Mathlib has no Wasserstein distance, so `W₂` is a parameter, as in
`Interpolation/Main.lean` and `Interpolation/Clustering.lean`; here it is a
family indexed by the dimension, so that the survey's *universal* constant `C`
can be quantified before the dimension and not after it.
-/

import Transformer.Basic
import Transformer.Interpolation.Basic

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Interpolation

/-- A Wasserstein distance on each sphere at once: the parameter the two
statements of this file share. -/
abbrev W2Family : Type :=
  ∀ d : ℕ, Measure (SSphere d) → Measure (SSphere d) → ℝ

/-- **Lemma (lem: mass.concentration.Q1).**

Let `ν_0^i ∈ 𝒫(𝕊^{d-1})` and `η > 0`, and let `φ_η : 𝕊^{d-1} → 𝕊^{d-1}` be
Lipschitz and invertible with

  `(φ_η)_# ν_0^i (ℚ_1^{d-1}) = 1 - η`   for every `i`.

Then for every `i` there is a `μ_0^i ∈ 𝒫(ℚ_1^{d-1})` with

  `W_2((φ_η)_# ν_0^i, μ_0^i) ≤ C η`,

`C > 0` universal.  "`μ_0^i ∈ 𝒫(ℚ_1^{d-1})`" is read as a probability measure
on the sphere giving no mass to the complement of the quadrant, since
`ℚ_1^{d-1}` is not carried as a type here.

Not proved here.

Source: arXiv:2411.04551v3, Appendix "On condition (eq: assumption.hole)",
`lem: mass.concentration.Q1`. -/
theorem mass_concentration_Q1 (W₂ : W2Family) :
    ∃ C : ℝ, 0 < C ∧
      ∀ (d N : ℕ) (ν : Idx N → Perspective.ProbSphere d) (η : ℝ), 0 < η →
        ∀ (φ : SSphere d → SSphere d) (L : NNReal), LipschitzWith L φ →
          Function.Bijective φ →
          (∀ i : Idx N,
            Measure.map φ (ν i : Measure (SSphere d)) (positiveQuadrant d)
              = ENNReal.ofReal (1 - η)) →
          ∀ i : Idx N, ∃ μ : Perspective.ProbSphere d,
            (μ : Measure (SSphere d)) (positiveQuadrant d)ᶜ = 0 ∧
            W₂ d (Measure.map φ (ν i : Measure (SSphere d)))
                (μ : Measure (SSphere d)) ≤ C * η := by
  sorry

/-- **Claim (cl: W.to.ball).**

If `μ ∈ 𝒫(𝕊^{d-1})` and `x_0 ∈ 𝕊^{d-1}` satisfy `W_2(μ, δ_{x_0}) ≤ η_2`, then

  `1 - μ(B(x_0, η_3)) ≤ C η_2 / η_3`   for every `η_3 > 0`,

with `C > 0` universal.  The paper's proof goes through Kantorovich–Rubinstein
duality for `W_1`; none of it is formalized.

Source: arXiv:2411.04551v3, Appendix, `cl: W.to.ball`. -/
theorem W_to_ball (W₂ : W2Family) :
    ∃ C : ℝ, 0 < C ∧
      ∀ (d : ℕ) (μ : Perspective.ProbSphere d) (x₀ : SSphere d) (η₂ : ℝ),
        W₂ d (μ : Measure (SSphere d)) (Measure.dirac x₀) ≤ η₂ →
        ∀ η₃ : ℝ, 0 < η₃ →
          1 - ((μ : Measure (SSphere d)) (Metric.ball x₀ η₃)).toReal
            ≤ C * η₂ / η₃ := by
  sorry

end Interpolation
end Transformer
