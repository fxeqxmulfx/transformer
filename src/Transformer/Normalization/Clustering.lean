/-
# Normalization — Asymptotic clustering (§3 of 2510.22026v2)

* `Theorem thm: convergence` — for `Q = K = V = I_d` the normalized attention
  dynamics clusters almost surely: unconditionally for Post-LN, nGPT and CoD
  from a uniform initialization, and up to a stalling radial velocity for
  Pre-LN, Mix-LN and Peri-LN from a Gaussian one;
* its corollary — for Pre-LN and Peri-LN with `n ≤ e^β` the second alternative
  is excluded and the synchronization is unconditional.

The two clustering alternatives remain open. The corollary is proved from
the second as an explicit input. The uniform law is `Perspective.UniformTuple`,
and the Gaussian law is Mathlib's `stdGaussian`, taken independently `n` times.

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
import Transformer.Normalization.Section3_RadialBounds
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

/-- **Conditional corollary.** Pre-LN and Peri-LN at `n ≤ e^β` synchronize
assuming the open Gaussian clustering alternative, stated in full in
`hcluster`. No proof depends on the still-open theorem. Sphere invariance
follows from the actual equations; the radial velocity lies in
`[(n e^β)⁻¹, 1]`, so each lower limit is positive and stalling is impossible.
The hypothesis `d ≥ 2` is retained; the source's `d = 1` case is refuted by
`not_unconditional_synchronization_one`.
Source: arXiv:2510.22026v2, §3, corollary to `thm:convergence`. -/
theorem unconditional_synchronization (hd : 2 ≤ d) (β : ℝ) (α : ℝ → ℝ) (τ : ℝ)
    (scheme : Scheme) (hβ : 0 < β) (hn : (n : ℝ) ≤ Real.exp β)
    (hscheme : scheme = Scheme.pre ∨ scheme = Scheme.peri)
    (hcluster : 2 ≤ d →
      (scheme = Scheme.pre ∨ scheme = Scheme.mix ∨ scheme = Scheme.peri) →
      ∀ᵐ X₀ ∂(Measure.pi fun _ : Idx n => stdGaussian (EucSpace d)),
        (∀ j : Idx n, X₀ j ≠ 0) →
        ∀ (θ : ℝ → Idx n → EucSpace d) (r : ℝ → Idx n → ℝ),
          (∀ j, θ 0 j = ‖X₀ j‖⁻¹ • X₀ j) → (∀ j, r 0 j = ‖X₀ j‖) →
          SchemeDynamics d n β (idParams d) (idParams d) (idParams d) α τ scheme θ r →
            Synchronizes d n θ ∨ RadialStalls d n β τ scheme θ) :
    ∀ᵐ X₀ ∂(Measure.pi fun _ : Idx n => stdGaussian (EucSpace d)), (∀ j : Idx n, X₀ j ≠ 0) →
      ∀ (θ : ℝ → Idx n → EucSpace d) (r : ℝ → Idx n → ℝ),
        (∀ j : Idx n, θ 0 j = ‖X₀ j‖⁻¹ • X₀ j) → (∀ j : Idx n, r 0 j = ‖X₀ j‖) →
          SchemeDynamics d n β (idParams d) (idParams d) (idParams d) α τ scheme θ r →
            Synchronizes d n θ := by
  have hs : scheme = Scheme.pre ∨ scheme = Scheme.mix ∨ scheme = Scheme.peri :=
    hscheme.elim Or.inl (fun h => Or.inr (Or.inr h))
  filter_upwards [hcluster hd hs] with X₀ hX
  intro hX₀ θ r hθ₀ hr₀ hD
  rcases hX hX₀ θ r hθ₀ hr₀ hD with hsync | hstall
  · exact hsync
  · obtain ⟨j, hj⟩ := hstall.1
    have h0 : ∀ j, ‖θ 0 j‖ = 1 := fun i => by
      rw [hθ₀ i]; exact norm_smul_inv_norm (hX₀ i)
    have hp := hD.radial_liminf_pos hβ hn hscheme h0 j
    rw [← hj] at hp
    exact (lt_irrefl 0 hp).elim

/-- Every conditional-corollary hypothesis is realizable: `d = 2`, one token,
`β = 1`, Pre-LN. The clustering input itself is proved, using the actual
singleton dynamics rather than an open theorem. -/
example : 2 ≤ 2 ∧ (0 : ℝ) < 1 ∧ ((1 : ℕ) : ℝ) ≤ Real.exp 1 ∧
    (Scheme.pre = .pre ∨ Scheme.pre = .peri) ∧
    (2 ≤ 2 → (Scheme.pre = .pre ∨ Scheme.pre = .mix ∨ Scheme.pre = .peri) →
      ∀ᵐ X₀ ∂(Measure.pi fun _ : Idx 1 => stdGaussian (EucSpace 2)),
        (∀ j : Idx 1, X₀ j ≠ 0) →
        ∀ (θ : ℝ → Idx 1 → EucSpace 2) (r : ℝ → Idx 1 → ℝ),
          (∀ j, θ 0 j = ‖X₀ j‖⁻¹ • X₀ j) → (∀ j, r 0 j = ‖X₀ j‖) →
          SchemeDynamics 2 1 1 (idParams 2) (idParams 2) (idParams 2) 0 0 .pre θ r →
            Synchronizes 2 1 θ ∨ RadialStalls 2 1 1 0 .pre θ) := by
  refine ⟨le_rfl, one_pos, by simp, Or.inl rfl, ?_⟩
  intro _ _
  refine Filter.Eventually.of_forall ?_
  intro X₀ hX₀ θ r hθ₀ _ hD
  have h0 : ∀ j, ‖θ 0 j‖ = 1 := fun i => by
    rw [hθ₀ i]; exact norm_smul_inv_norm (hX₀ i)
  refine Or.inl ⟨θ 0 0, fun j => ?_⟩
  apply Filter.Tendsto.congr' (f₁ := fun _ : ℝ => θ 0 0) ?_ tendsto_const_nhds
  refine (Filter.eventually_ge_atTop (0 : ℝ)).mono fun t ht => ?_
  have hc := hD.direction_constant_singleton (Or.inl rfl) h0 t ht j
  simpa only [Fin.fin_one_eq_zero j] using hc.symm

end Normalization
end Transformer
