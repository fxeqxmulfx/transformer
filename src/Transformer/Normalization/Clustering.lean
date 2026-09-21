/-
# Normalization — Asymptotic clustering (§3 of 2510.22026v2)

* `Theorem thm: convergence` — for `Q = K = V = I_d` the normalized attention
  dynamics clusters almost surely: unconditionally for Post-LN, nGPT and CoD
  from a uniform initialization, and up to a stalling radial velocity for
  Pre-LN, Mix-LN and Peri-LN from a Gaussian one;
* its corollary — for Pre-LN and Peri-LN with `n ≤ e^β` the second alternative
  is excluded and the synchronization is unconditional.

Neither is proved here.  The uniform law is `Perspective.UniformTuple`, which
pins it down by rotation invariance, and the Gaussian one is Mathlib's
`stdGaussian`, taken `n` times independently.

*Deviation from the source.*  The source states both for every `d`; each is
stated here for `d ≥ 2`.  On `𝕊⁰ ⊆ ℝ¹` the tangent space is `0`, so every
configuration is a rest point, and two tokens of opposite signs — an event of
positive probability under both laws — never meet: `ClusteringLine` refutes all
three statements at `d = 1`, `n = 2`.

The gradient-flow identification the proof starts from is in
`Normalization.Convergence`, the velocity lower bound behind the corollary in
`Normalization.Radial`, and the convergence machinery of the proof in
`Normalization.Lojasiewicz`.
-/

import Transformer.Basic
import Transformer.Normalization.Basic
import Transformer.Normalization.Radial
import Transformer.Perspective.Section3_SmallBeta
import Mathlib.Probability.Distributions.Gaussian.Multivariate
import Mathlib.Order.LiminfLimsup
import Mathlib.MeasureTheory.Constructions.BorelSpace.Basic
import Mathlib.Analysis.Normed.Lp.MeasurableSpace

open scoped BigOperators
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Normalization

variable (d n : ℕ)

/-- **The tokens synchronize to one cluster:** every `θ_j(t)` converges, as
`t → ∞`, to one and the same point.

Source: arXiv:2510.22026v2, §3, the event of `thm: convergence`. -/
def Synchronizes (θ : ℝ → Idx n → EucSpace d) : Prop :=
  ∃ c : EucSpace d,
    ∀ j : Idx n, Filter.Tendsto (fun t => θ t j) Filter.atTop (nhds c)

/-- **The radial velocity stalls:** `min_{j ∈ [n]} liminf_{t → ∞} ṙ_j(t) = 0`.

Written as `IsLeast` over the values of the `n` lower limits rather than as a
`Finset.inf'`, so that the nonemptiness of the index set does not have to be
carried inside the statement.

Source: arXiv:2510.22026v2, §3, the second event of `thm: convergence`. -/
def RadialStalls
    (β : ℝ) (τ : ℝ) (scheme : Scheme) (θ : ℝ → Idx n → EucSpace d) : Prop :=
  IsLeast
    { c : ℝ | ∃ j : Idx n, c =
        Filter.liminf
          (fun t => radialDerivative d n β (idParams d) (idParams d) (idParams d)
            θ τ scheme t j) Filter.atTop }
    0

/-- **Theorem (thm: convergence), first half.** *Post-LN, nGPT and CoD cluster.*

For `d ≥ 2`, `Q = K = V = I_d` and a uniformly sampled
`Θ(0) ∈ (𝕊^{d-1})^{⊗ n}`, the Post-LN, nGPT and CoD dynamics synchronize to one
cluster with probability `1`.

The uniform law is read as every `σ` with `Perspective.UniformTuple d n σ`,
which holds for exactly one `σ`.  The source has no `d ≥ 2`, and without it the
statement is false: `not_clusters_from_uniform_one`.  The paper also
records that the same conclusion holds for constant `Q, K` and
`V = Q^⊤ K = K^⊤ Q`; only the identity case is stated here.

Not proved here.

Source: arXiv:2510.22026v2, §3, `thm: convergence`. -/
theorem clusters_from_uniform (hd : 2 ≤ d) (β : ℝ) (α : ℝ → ℝ) (τ : ℝ) (scheme : Scheme)
    (hscheme : scheme = Scheme.post ∨ scheme = Scheme.nGPT ∨ scheme = Scheme.CoD) :
    ∀ σ : Measure (SphereTuple d n), Perspective.UniformTuple d n σ →
    ∀ᵐ Θ₀ ∂σ, ∀ (θ : ℝ → Idx n → EucSpace d) (r : ℝ → Idx n → ℝ),
      θ 0 = tupleCoe Θ₀ →
        SchemeDynamics d n β (idParams d) (idParams d) (idParams d) α τ scheme θ r →
          Synchronizes d n θ := by
  sorry

/-- The hypotheses of `clusters_from_uniform` are satisfiable: `d = 2`, Post-LN. -/
example : 2 ≤ 2 ∧ (Scheme.post = Scheme.post ∨ Scheme.post = Scheme.nGPT ∨
    Scheme.post = Scheme.CoD) :=
  ⟨le_rfl, Or.inl rfl⟩

/-- **Theorem (thm: convergence), second half.** *Pre-LN, Mix-LN and Peri-LN.*

For `d ≥ 2`, `Q = K = V = I_d` and a standard Gaussian sample
`X(0) = r(0) · Θ(0) ∈ (ℝ^d)^{⊗ n}`, the
Pre-LN, Mix-LN and Peri-LN dynamics almost surely either synchronize to one
cluster or have a stalling radial velocity,

  `P[{tokens synchronize} ∪ {min_j liminf_{t → ∞} ṙ_j(t) = 0}] = 1`.

The initialization is read off the Gaussian sample: `θ_j(0) = x_j / ‖x_j‖` and
`r_j(0) = ‖x_j‖`, so the hypothesis `x_j ≠ 0` — which the Gaussian satisfies
almost surely — is carried explicitly.  The source has no `d ≥ 2`, and without
it the statement is false: `not_clusters_or_stalls_from_gaussian_one`.

Not proved here.

Source: arXiv:2510.22026v2, §3, `thm: convergence`. -/
theorem clusters_or_stalls_from_gaussian (hd : 2 ≤ d) (β : ℝ) (α : ℝ → ℝ) (τ : ℝ)
    (scheme : Scheme)
    (hscheme : scheme = Scheme.pre ∨ scheme = Scheme.mix ∨ scheme = Scheme.peri) :
    ∀ᵐ X₀ ∂(Measure.pi fun _ : Idx n => stdGaussian (EucSpace d)), (∀ j : Idx n, X₀ j ≠ 0) →
      ∀ (θ : ℝ → Idx n → EucSpace d) (r : ℝ → Idx n → ℝ),
        (∀ j : Idx n, θ 0 j = ‖X₀ j‖⁻¹ • X₀ j) → (∀ j : Idx n, r 0 j = ‖X₀ j‖) →
          SchemeDynamics d n β (idParams d) (idParams d) (idParams d) α τ scheme θ r →
            Synchronizes d n θ ∨ RadialStalls d n β τ scheme θ := by
  sorry

/-- The hypotheses of `clusters_or_stalls_from_gaussian` are satisfiable:
`d = 2`, Pre-LN. -/
example : 2 ≤ 2 ∧ (Scheme.pre = Scheme.pre ∨ Scheme.pre = Scheme.mix ∨
    Scheme.pre = Scheme.peri) :=
  ⟨le_rfl, Or.inl rfl⟩

/-- **Corollary.** *Unconditional synchronization for Pre-LN and Peri-LN.*

For Pre-LN and Peri-LN at `n ≤ e^β` the radial velocity is bounded below by
`1 / (n e^β) > 0` — `radialDerivative_pre_lower_bound` — so the second event of
`thm: convergence` is impossible and the first one has probability `1`.

Not proved here: what is available is the velocity bound, not the implication
from it to clustering, which is the content of `thm: convergence` itself.
Stated for `d ≥ 2`, as `thm: convergence` is; without it the corollary is
false: `not_unconditional_synchronization_one`.

Source: arXiv:2510.22026v2, §3, the corollary to `thm: convergence`. -/
theorem unconditional_synchronization (hd : 2 ≤ d) (β : ℝ) (α : ℝ → ℝ) (τ : ℝ)
    (scheme : Scheme) (hβ : 0 < β) (hn : (n : ℝ) ≤ Real.exp β)
    (hscheme : scheme = Scheme.pre ∨ scheme = Scheme.peri) :
    ∀ᵐ X₀ ∂(Measure.pi fun _ : Idx n => stdGaussian (EucSpace d)), (∀ j : Idx n, X₀ j ≠ 0) →
      ∀ (θ : ℝ → Idx n → EucSpace d) (r : ℝ → Idx n → ℝ),
        (∀ j : Idx n, θ 0 j = ‖X₀ j‖⁻¹ • X₀ j) → (∀ j : Idx n, r 0 j = ‖X₀ j‖) →
          SchemeDynamics d n β (idParams d) (idParams d) (idParams d) α τ scheme θ r →
            Synchronizes d n θ := by
  sorry

/-- The hypotheses of `unconditional_synchronization` are satisfiable: `d = 2`,
one token, `β = 1`, Pre-LN, since `1 ≤ e`. -/
example : 2 ≤ 2 ∧ (0 : ℝ) < 1 ∧ ((1 : ℕ) : ℝ) ≤ Real.exp 1 ∧
    (Scheme.pre = Scheme.pre ∨ Scheme.pre = Scheme.peri) :=
  ⟨le_rfl, one_pos, by simp, Or.inl rfl⟩

end Normalization
end Transformer
