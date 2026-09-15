/-
# Normalization — Initial and terminal velocities (§4.2–4.3 of 2510.22026v2)

* `Theorem thm: initial-velocity` — at a uniform initialization the attention
  vector is small, `‖A_j(0)‖ ≤ C(√(log n / n) + log n / d)` for every token at
  once with probability `1 - n^{-c}`;
* `Theorem thm: preln-slow (ii)` — from a local-cone initialization the
  intra-cluster variance decays at the per-scheme rate
  `d/dt Var(t) = -Θ(Var(t) / scale(t))`.

Part (i) of `thm: preln-slow`, the radial growth `r_k(t) ≥ (1 - δ) t`, is
proved in `Normalization.Velocities`; the bound `‖A_j‖ ≤ 1` that holds at every
configuration is there too.

Both statements here are `Prop`-valued definitions.  The first is almost-sure
with respect to the uniform measure on `(𝕊^{d-1})^{⊗ n}`, which this
development does not construct, and carries it as a parameter.  The second is
an asymptotic two-sided bound: the `Θ(·)` is spelled out as a pair of
constants `0 < c ≤ C` sandwiching the derivative, which is what `Θ` means, and
neither is proved here.
-/

import Transformer.Basic
import Transformer.Normalization.Basic
import Transformer.Normalization.Velocities
import Mathlib.MeasureTheory.Constructions.BorelSpace.Basic
import Mathlib.Analysis.Normed.Lp.MeasurableSpace

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Normalization

variable (d n : ℕ)

/-- **Theorem (thm: initial-velocity).** *The attention vector is small at a
uniform initialization.*

For `β = 1` and `max{‖Q^⊤ K‖_op, ‖V‖_op} ≤ 1`, with `θ_j(0)` drawn i.i.d.
uniform on `𝕊^{d-1}` and `e^{√d} ≥ n log n ≥ d`, there are absolute constants
`c, C > 0` such that with probability `1 - n^{-c}`, simultaneously for all
`j ∈ [n]`,

  `‖A_j(0)‖ ≤ C (√(log n / n) + log n / d)`.

`‖Q^⊤ K‖_op ≤ 1` is stated on the bilinear form it is the norm of,
`|⟨Q x, K y⟩| ≤ ‖x‖ ‖y‖`, so that no adjoint has to be formed.  `σ` is the
reference measure on `(𝕊^{d-1})^{⊗ n}`, carried as a parameter: the uniform
measure is not constructed here, and the independence of the coordinates is
part of what it would have to supply.

Not proved here.

Source: arXiv:2510.22026v2, §4.2, `thm: initial-velocity`. -/
def InitialVelocitySmall (σ : Measure (SphereTuple d n)) : Prop :=
  ∃ c C : ℝ, 0 < c ∧ 0 < C ∧
    ∀ Q K V : ParamMatrix d,
      (∀ x y : EucSpace d, |inner (𝕜 := ℝ) (Q x) (K y)| ≤ ‖x‖ * ‖y‖) →
      ‖V‖ ≤ 1 →
      (n : ℝ) * Real.log n ≤ Real.exp (Real.sqrt d) →
      (d : ℝ) ≤ (n : ℝ) * Real.log n →
      1 - ENNReal.ofReal ((n : ℝ) ^ (-c))
        ≤ σ { Θ : SphereTuple d n | ∀ j : Idx n,
            ‖attentionVec d n 1 Q K V (tupleCoe Θ) j‖
              ≤ C * (Real.sqrt (Real.log n / n) + Real.log n / d) }

/-- The time scale by which `thm: preln-slow (ii)` divides the intra-cluster
variance, one row per scheme: `1` for Post-LN, `t` for Pre-LN, Mix-LN and
Peri-LN, the step size `α_t` for nGPT, and `√t` for CoD.

Source: arXiv:2510.22026v2, §4.3, the display of `thm: preln-slow (ii)`. -/
noncomputable def varScale (α : ℝ → ℝ) (scheme : Scheme) (t : ℝ) : ℝ :=
  match scheme with
  | .post => 1
  | .pre  => t
  | .mix  => t
  | .peri => t
  | .nGPT => α t
  | .CoD  => Real.sqrt t

/-- **Theorem (thm: preln-slow), (ii).** *Speed of cluster collapse.*

For `V = I_d` and arbitrary `Q, K` with `‖Q^⊤ K‖_op ≤ 1`, started in a local
cone `⟨θ_j(0), θ_k(0)⟩ ≥ 1 - δ` with `δ < 1 / (100 n² β²)`, the intra-cluster
variance obeys

  `d/dt Var(t) = -Θ(Var(t) / scale(t))`,

with `scale` the per-scheme time scale of `varScale`: Post-LN clusters
exponentially, CoD at `√t`, and Pre-LN, Mix-LN, Peri-LN polynomially, while
nGPT controls the rate through `α_t`.

`Θ(·)` is written out as a pair of absolute constants `0 < c ≤ C` that
sandwich the derivative from both sides for all positive times; this is the
whole content of the statement, the paper giving no explicit constants.
Part (i) is `radialDerivative_pre_ge_of_localCone`.

Not proved here.

Source: arXiv:2510.22026v2, §4.3, `thm: preln-slow` (ii). -/
def ClusteringRate (β δ : ℝ) (α : ℝ → ℝ) (τ : ℝ) (scheme : Scheme) : Prop :=
  δ < 1 / (100 * (n : ℝ) ^ 2 * β ^ 2) →
  ∀ Q K : ℝ → ParamMatrix d,
    (∀ t : ℝ, ∀ x y : EucSpace d,
      |inner (𝕜 := ℝ) (Q t x) (K t y)| ≤ ‖x‖ * ‖y‖) →
    ∃ c C : ℝ, 0 < c ∧ c ≤ C ∧
      ∀ (θ : ℝ → Idx n → EucSpace d) (r : ℝ → Idx n → ℝ),
        (∀ j k : Idx n, 1 - δ ≤ inner (𝕜 := ℝ) (θ 0 j) (θ 0 k)) →
        SchemeDynamics d n β Q K (idParams d) α τ scheme θ r →
          ∀ t : ℝ, 0 < t → ∃ v : ℝ,
            HasDerivAt (intraClusterVar d n θ) v t ∧
            -(C * intraClusterVar d n θ t / varScale α scheme t) ≤ v ∧
            v ≤ -(c * intraClusterVar d n θ t / varScale α scheme t)

end Normalization
end Transformer
