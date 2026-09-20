/-
# The emergence of clusters in self-attention dynamics — well-posedness of the
  ODEs

§6 of arXiv:2305.05465v6: `p:wellposedparticles` and
`p:wellposednessrescaledparticles`, global existence and uniqueness for
`eq:trans_dyn` and for its rescaled form `e:Rres`.

**What the source says and what is carried here.**

* Both propositions assert a *unique Lipschitz continuous* `t ↦ x(t)` defined
  on all of `ℝ`.  "Lipschitz continuous" cannot be read globally on `ℝ`: for
  `n = 1` the only attention weight is `1`, the dynamics is the linear
  `ẋ = Vx`, and its solution `x(t) = e^{tV}x⁰` grows exponentially, so it is
  not Lipschitz on `ℝ` for any `V` with a positive eigenvalue.  It is read
  here as Lipschitz on every compact interval, which is what the proof of
  `c:wellposedtransformers` delivers and what the uniqueness argument uses.
  This is the only deviation from the source.

* Uniqueness is asserted among curves with that regularity, so it is carried
  inside the `∃!`.

* `p:wellposednessrescaledparticles` is deduced in the source from
  `p:wellposedparticles` through the change of variables `x_i = e^{tV}z_i`.
  It is stated here independently rather than derived, because a derivation
  would rest on `transformerDynamics_iff_rescaled`, which is itself unproved.

Source: arXiv:2305.05465v6, `p:wellposedparticles`,
`p:wellposednessrescaledparticles`.
-/

import Transformer.Clusters.Section3_Rescaled
import Mathlib.Topology.MetricSpace.Lipschitz

open scoped BigOperators NNReal
open Real

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-- **Lipschitz on every compact interval**, the regularity
`p:wellposedparticles` asks of its solution. -/
def IsLocLipschitzCurve (X : ℝ → Idx n → EucSpace d) : Prop :=
  ∀ (i : Idx n) (a b : ℝ), ∃ L : ℝ≥0, LipschitzOnWith L (fun t => X t i) (Set.Icc a b)

/-- A constant curve is Lipschitz on every interval, with constant `0`. -/
theorem isLocLipschitzCurve_const (X : Idx n → EucSpace d) :
    IsLocLipschitzCurve (fun _ => X) :=
  fun i _ _ => ⟨0, LipschitzOnWith.of_dist_le_mul fun _ _ _ _ => by simp⟩

/-- **Proposition (p:wellposedparticles).**  For any initial sequence of `n`
tokens there is exactly one curve, Lipschitz on compact intervals, solving
`eq:trans_dyn` from it, and it is defined for all time.

Not proved here.

Source: arXiv:2305.05465v6, `p:wellposedparticles`. -/
theorem wellposed_particles (Q K V : ParamMatrix d) (X₀ : Idx n → EucSpace d) :
    ∃! X : ℝ → Idx n → EucSpace d,
      TransformerDynamics Q K V X ∧ X 0 = X₀ ∧ IsLocLipschitzCurve X := by
  sorry

/-- **Proposition (p:wellposednessrescaledparticles).**  The same for the
rescaled dynamics `e:Rres`.

Not proved here.

Source: arXiv:2305.05465v6, `p:wellposednessrescaledparticles`. -/
theorem wellposed_rescaled_particles (Q K V : ParamMatrix d) (Z₀ : Idx n → EucSpace d) :
    ∃! Z : ℝ → Idx n → EucSpace d,
      RescaledDynamics Q K V Z ∧ Z 0 = Z₀ ∧ IsLocLipschitzCurve Z := by
  sorry

/-- The stationary curve of `transformerDynamics_const` has all three
properties at `V = 0`, so the conjunction inside the two `∃!` above is not
empty of witnesses. -/
example (Q K : ParamMatrix d) (X₀ : Idx n → EucSpace d) :
    TransformerDynamics Q K 0 (fun _ => X₀) ∧ (fun _ => X₀) 0 = X₀ ∧
      IsLocLipschitzCurve (fun _ : ℝ => X₀) :=
  ⟨transformerDynamics_const Q K X₀, rfl, isLocLipschitzCurve_const X₀⟩

/-- The same for the rescaled dynamics. -/
example (Q K : ParamMatrix d) (Z₀ : Idx n → EucSpace d) :
    RescaledDynamics Q K 0 (fun _ => Z₀) ∧ (fun _ => Z₀) 0 = Z₀ ∧
      IsLocLipschitzCurve (fun _ : ℝ => Z₀) :=
  ⟨rescaledDynamics_const Q K Z₀, rfl, isLocLipschitzCurve_const Z₀⟩

end Clusters
end Transformer
