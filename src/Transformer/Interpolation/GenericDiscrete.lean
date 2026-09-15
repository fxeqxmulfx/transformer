/-
# Measure-to-measure interpolation — Generic discrete inputs

Formalization of `Proposition prop:generic-discrete` of arXiv:2411.04551v3:
for `d ≥ 3` and uniformly sampled discrete inputs, the pure-attention flow
`𝐁 ≡ β I_d`, `𝐕 ≡ I_d`, `𝐖 ≡ 0` sends different inputs to different cluster
points almost surely — the `O(1)`-switch regime the survey contrasts with the
`O(d · N)` bound of `prop: compression`.
-/

import Transformer.Basic
import Transformer.Interpolation.Basic
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Measure.ProbabilityMeasure

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Interpolation

/-- The parameters of `prop:generic-discrete`: `𝐕 ≡ I_d`, `𝐁 ≡ β I_d`,
`𝐖 ≡ 0`, so the vector field is pure self-attention. -/
noncomputable def genericParams (d : ℕ) (β : ℝ) : TimeParams d :=
  fun _ =>
    { V := ContinuousLinearMap.id ℝ (EucSpace d)
      B := β • ContinuousLinearMap.id ℝ (EucSpace d)
      W := 0
      U := 0
      b := 0 }

/-- **Proposition (prop:generic-discrete).**

Let `d ≥ 3`, let `ref` be the uniform measure on `(𝕊^{d-1})^n` — identified with
the empirical measures `𝒮_n` — and sample `μ_0^1, …, μ_0^N` i.i.d. from `π`.
For `eq: cauchy.pb` with `𝐁 ≡ β I_d`, `𝐕 ≡ I_d`, `𝐖 ≡ 0`, let `x_*^i` be the
almost surely existing cluster point of `μ^i(t)`.  Then

  `ℙ[x_*^i ≠ x_*^j] = 1`   for `i ≠ j`.

The cluster point is carried by a function `f` on configurations, whose
defining property — almost sure weak convergence of the flow to `δ_{f(X)}` —
is a hypothesis, since it is itself quoted from the survey of
`Transformer.Perspective` rather than proved.  `ref`, the reference measure, is a parameter: its
rotation invariance, which the survey's proof uses to rule out atoms, is not
recorded, and neither is the proof.

A `Prop`-valued definition and not a theorem: nothing of it is proved here.

Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
def GenericDiscreteDistinct
    (d n N : ℕ) (β : ℝ) (ref : Measure (SphereTuple d n)) : Prop :=
  3 ≤ d → 0 ≤ β →
  ∀ f : SphereTuple d n → SSphere d,
    (∀ᵐ X : SphereTuple d n ∂ref,
      ∀ μ : ℝ → Perspective.ProbSphere d,
        (μ 0 : Measure (SSphere d)) = Perspective.empiricalMeasure d n X →
        cauchyPB d (genericParams d β) μ →
        Filter.Tendsto μ Filter.atTop
          (nhds (⟨Measure.dirac (f X), inferInstance⟩ :
            Perspective.ProbSphere d))) →
    ∀ i j : Idx N, i ≠ j →
      (Measure.pi fun _ : Idx N => ref)
          { X : Idx N → SphereTuple d n | f (X i) ≠ f (X j) } = 1

end Interpolation
end Transformer
