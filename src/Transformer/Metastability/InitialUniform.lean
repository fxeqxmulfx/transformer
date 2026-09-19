/-
# Metastability — On the initial configuration, uniform initialization
  (§4 of 2410.06833v1)

* `Proposition prop: concentration unif` — `eq: upto-t`, the orthogonal
                                           approximation of a uniform sample,
* `Corollary coro: cm`, `eq: technical.cond` — uniform points are
                                           `(β, ε)`-separated,
* the low-dimensional bound on the probability of being `(β, ε)`-separated.

All three bound the probability of an event under the *uniform* measure on
`𝕊^{d-1}`, which this development does not construct.  It is pinned down
instead of built: `IsUniformFamily` asks for a probability measure on each
sphere invariant under every linear isometry of `ℝ^d`, and such a measure is
unique, so quantifying over all families satisfying it is not a strengthening.
This is `Perspective.UniformTuple`'s device, one sphere at a time.

None of the three is proved.
-/

import Transformer.Basic
import Transformer.Metastability.Basic
import Transformer.Perspective.Section2_FlowMap
import Mathlib.MeasureTheory.Constructions.Pi

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Metastability

/-- The uniform law on `𝕊^{d-1}`: a probability measure invariant under every
linear isometry of the ambient `ℝ^d`.

Such a measure is unique — a rotation-invariant Borel probability measure on
`𝕊^{d-1}` *is* `σ_d` — so the statements below, read against every measure
satisfying this, say exactly what the survey says about the uniform one.  The
invariance is stated through `Perspective.sphereMap`, which keeps the Haar
machinery out. -/
def IsUniformOn (d : ℕ) (ν : Measure (SSphere d)) : Prop :=
  IsProbabilityMeasure ν ∧
    ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d,
      ν.map (Perspective.sphereMap d U) = ν

/-- The same, dimension by dimension: a uniform law on every sphere at once. -/
def IsUniformFamily (σ : ∀ d : ℕ, Measure (SSphere d)) : Prop :=
  ∀ d : ℕ, IsUniformOn d (σ d)

/-- The law of `n` i.i.d. draws from a measure `ν` on `𝕊^{d-1}`. -/
noncomputable def iidSphere (d n : ℕ) (ν : Measure (SSphere d)) :
    Measure (Idx n → SSphere d) :=
  Measure.pi fun _ : Idx n => ν

/-- **Equation (eq: upto-t).**  The event that the sample is uniformly close
to a pairwise-orthogonal configuration:

  `∃ (w_1,…,w_n)` pairwise orthogonal with `‖x_i - w_i‖ ≤ √(4 log d / d)`. -/
def nearOrthogonal (d n : ℕ) : Set (Idx n → SSphere d) :=
  { X | ∃ w : Idx n → SSphere d,
      (∀ i j : Idx n, i ≠ j →
        inner (𝕜 := ℝ) ((w i : EucSpace d)) ((w j : EucSpace d)) = 0) ∧
      ∀ i : Idx n, ‖(X i : EucSpace d) - (w i : EucSpace d)‖
        ≤ Real.sqrt (4 * Real.log d / (d : ℝ)) }

/-- **Proposition (prop: concentration unif).**

For `n ≥ 2` there is `d⋆(n) > n` such that for every `d ≥ d⋆(n)`, an i.i.d.
uniform sample on `𝕊^{d-1}` lies in `nearOrthogonal` with probability at
least `1 - 2 n² d^{-1/64}`.

Not proved here.

Source: arXiv:2410.06833v1, §4. -/
theorem concentration_unif (n : ℕ) (hn : 2 ≤ n) :
    ∀ σ : ∀ d : ℕ, Measure (SSphere d), IsUniformFamily σ →
      ∃ d_star : ℕ, n < d_star ∧ ∀ d : ℕ, d_star ≤ d →
        1 - 2 * (n : ℝ)^2 * (d : ℝ) ^ (-(1 : ℝ) / 64)
          ≤ (iidSphere d n (σ d)).real (nearOrthogonal d n) := by
  sorry

/-- The hypothesis of `concentration_unif` is satisfiable: `n = 2`. -/
example : 2 ≤ 2 := le_rfl

/-- **Corollary (coro: cm) with equation (eq: technical.cond).**

For `d ≥ max(d⋆(n), 381)` and `β > 0` satisfying

  `(16 log² d)/d² + (40 log d)/d + β⁻¹ log(n² d / (2 log d)) < 1`,

an i.i.d. uniform sample on `𝕊^{d-1}` is `(β, ε)`-separated with
`ε = 4 log d / d` with probability at least `1 - 2 n² d^{-1/64}`.

Not proved here; it rests on `concentration_unif`, which is not proved
either.

Source: arXiv:2410.06833v1, §4. -/
theorem uniform_separated (n : ℕ) (hn : 2 ≤ n) :
    ∀ σ : ∀ d : ℕ, Measure (SSphere d), IsUniformFamily σ →
      ∃ d_star : ℕ, n ≤ d_star ∧ 381 ≤ d_star ∧
        ∀ d : ℕ, d_star ≤ d → ∀ β : ℝ, 0 < β →
          (16 * (Real.log d)^2 / (d : ℝ)^2)
              + (40 * Real.log d / (d : ℝ))
              + β⁻¹ * Real.log ((n : ℝ)^2 * d / (2 * Real.log d)) < 1 →
            1 - 2 * (n : ℝ)^2 * (d : ℝ) ^ (-(1 : ℝ) / 64)
              ≤ (iidSphere d n (σ d)).real
                  { X | isSeparated d n β (4 * Real.log d / (d : ℝ)) X } := by
  sorry

/-- The hypothesis of `uniform_separated` is satisfiable: `n = 2`. -/
example : 2 ≤ 2 := le_rfl

/-- **Low-dimensional bound.**

For `d = 2`, `n ≥ 2` and `0 < ε < 1/16`, the probability that an i.i.d.
uniform sample on `𝕊^1` is `(β, ε)`-separated decays exponentially in `n`:
there is `c ∈ (0, 1)`, depending on `β` and `ε` alone, with

  `ℙ((x_1,…,x_n) is (β, ε)-separated) ≤ c^n`.

Not proved here.

Source: arXiv:2410.06833v1, §4. -/
theorem low_dim_decay (β ε : ℝ) (hε : 0 < ε) (hε16 : ε < 1 / 16) :
    ∀ σ : ∀ d : ℕ, Measure (SSphere d), IsUniformFamily σ →
      ∃ c : ℝ, 0 < c ∧ c < 1 ∧
        ∀ n : ℕ, 2 ≤ n →
          (iidSphere 2 n (σ 2)).real { X | isSeparated 2 n β ε X } ≤ c ^ n := by
  sorry

/-- The hypotheses of `low_dim_decay` are satisfiable: `ε = 1/32`. -/
example : (0 : ℝ) < 1 / 32 ∧ (1 : ℝ) / 32 < 1 / 16 := by norm_num

end Metastability
end Transformer
