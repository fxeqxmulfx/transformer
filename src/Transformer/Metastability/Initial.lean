/-
# Metastability — On the initial configuration, Gaussian mixtures
  (§4 of 2410.06833v1)

* `eq: gaussian.mixture`             — Gaussian-mixture density,
* `Definition d: separated_mixtures` — `(β, ε)`-centered configurations,
* `Proposition prop: mixture.of.gaussians`.

The uniform-initialization half of §4 (`prop: concentration unif`,
`coro: cm`, the low-dimensional bound) is in
`Transformer.Metastability.InitialUniform`.

`prop: mixture.of.gaussians` bounds a probability, so it is stated against the
law `mixtureLaw` of the sample — a product of measures with the density
`eq: gaussian.mixture` — as a `Prop`-valued definition: nothing here proves
that this law exists as a probability measure (the density is not shown to
integrate to one), which is the hypothesis the paper starts from.
-/

import Transformer.Basic
import Transformer.Metastability.Basic
import Transformer.Perspective.Section2_FlowMap
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.Analysis.Normed.Lp.MeasurableSpace
import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Metastability

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

`(w_1,…,w_r)` is `(β, ε)`-*centered* (for a sample of size `n`) if there are
at most `n` centres and the caps `𝒮_q(ε)` around them satisfy `eq: gamma` of
`hyp: init`:

  `γ(β) = 1 - α(ε) - 8 ε - β⁻¹ log(2 n² / ε) > 0`.

This is `isSeparated` with the membership clause dropped: it constrains the
centres alone, not the sample.  Source: arXiv:2410.06833v1, §4. -/
def isCentered
    (β ε : ℝ) (r : ℕ) (w : Idx r → SSphere d) : Prop :=
  r ≤ n ∧ 0 < γβ n β (αDist d r w ε) ε

/-- The law of `n` i.i.d. draws from the Gaussian mixture
`eq: gaussian.mixture` — the `n`-fold product of the measure with that
density.  It is a probability measure exactly when the density integrates to
one, which is not proved here. -/
noncomputable def mixtureLaw
    (r : ℕ) (σ : ℝ) (w : Idx r → EucSpace d) : Measure (Idx n → EucSpace d) :=
  Measure.pi fun _ : Idx n =>
    volume.withDensity fun x => ENNReal.ofReal (gaussianMixtureDensity d r σ w x)

/-- The event that the radial projections `X_i / ‖X_i‖` of a sample form a
`(β, ε)`-separated configuration on the sphere.  A sample with some `X_i = 0`
has no projection and is outside the event, since no point of the sphere
equals `‖0‖⁻¹ • 0 = 0`. -/
def projectedSeparated (β ε : ℝ) : Set (Idx n → EucSpace d) :=
  { X | ∃ Y : SphereTuple d n,
      (∀ i : Idx n, (Y i : EucSpace d) = ‖X i‖⁻¹ • X i) ∧ isSeparated d n β ε Y }

/-- **Proposition (prop: mixture.of.gaussians).**

Let `(w_1,…,w_r)` be `(β, ε)`-centered, let `X_1,…,X_n` be i.i.d. with the
Gaussian-mixture density, and assume, with `δ = σ / √r`,

  `(6 δ √d)/(1 + δ √d) + δ √(2 d log n) ≤ ε`.

Then `(X_i / ‖X_i‖)_{i=1}^n` is `(β, ε)`-separated with probability at least
`1 - 2 e^{-d}`.

Source: arXiv:2410.06833v1, §4. -/
def MixtureSeparation
    (β ε σ : ℝ) (r : ℕ) (w : Idx r → SSphere d) : Prop :=
  isCentered d n β ε r w →
  (6 * (σ / Real.sqrt r) * Real.sqrt d)
        / (1 + (σ / Real.sqrt r) * Real.sqrt d)
      + (σ / Real.sqrt r) * Real.sqrt (2 * d * Real.log n) ≤ ε →
    1 - 2 * Real.exp (-(d : ℝ))
      ≤ (mixtureLaw d n r σ fun q => ((w q : EucSpace d))).real
          (projectedSeparated d n β ε)

end Metastability
end Transformer
