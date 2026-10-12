/-
# Perceptrons and attention's mean-field landscape — the anti-concentration bound

Formalization of `thm: bound` and `cor: bound` of arXiv:2601.21366v2, §3.2:
the source's local concavity estimate for `θ ↦ e^{β cos θ}` bounds the mass
of a cluster in a SOPD critical point — a cluster of angular width
`1/(2√β)` carries at most `0.5742 + O(e^{-β})` of the mass at large `β`.
The source additionally excludes a single such cluster for every `β > 0`
when the weights are small. That universal claim is false:
`Section3_ClusterCounterexample` proves a counterexample and its negation.

**What the source says and what is carried here.**

* "for `β` sufficiently large … `≤ 0.5742 + O(e^{-β})`" is one statement about
  the fixed perceptron `ϑ = (ω_j, a_j)_j` of the theorem: constants `C, β₀`
  produced after `ϑ` but *before* `β`, the activation and the configuration,
  with the bound `0.5742 + C e^{-β}` for every `β ≥ β₀`.  `C` may depend on
  `ϑ`, and in the source's proof it does: the bound derived there is
  `(2e^{β-3/2} + C_ϑ)/(2e^{β-3/2} + (3/8)e^{β-1/8})` with
  `C_ϑ = 2 Σ_j |ω_j| ‖a_j‖²`, whose `O(e^{-β})` part is `C_ϑ e^{-β}` up to a
  numerical factor.  A `C` uniform in `ϑ` would be a stronger claim than the
  source makes or proves.  Read the other way round — `C` chosen after the
  configuration — the claim would say nothing, since `C` could absorb any
  cluster mass.

* `max_{i,j ∈ S} min_{k ∈ ℤ} |θ_i - θ_j + 2πk| ≤ 1/(2√β)` is carried as: for
  every `i, j ∈ S` there is a `k` with `|θ_i - θ_j + 2πk| ≤ 1/(2√β)`.  The
  minimum over `k` is attained and the maximum over a finite set is a bound on
  every entry, so this is the same condition.

* The arcs `I_j` of `cor: bound` are carried by their endpoints, with
  `L` a bound on their lengths, and `supp μ ⊂ ⋃_j I_j` as "every angle lies in
  some `I_j`" — the support of `Σ m_i δ_{θ_i}` with `m_i > 0` is exactly the
  set of its atoms.

* `σ` globally Lipschitz with constant `1` is `LipschitzWith 1 σ`.

* The source assumes a SOPD critical point. Here the first mass conclusion
  is strengthened to every stationary critical point: the interaction-force
  increment between two distinct cluster endpoints is bounded using the
  exact cluster scale, and stationarity balances it with the globally
  Lipschitz potential force. This also avoids requiring a second derivative
  of the Lipschitz activation at an atomic support point.

* `rem: thm.bound.multid` states no theorem: it says that the restriction to
  `d = 2` is an artefact of the proof — the angular parametrization of `𝕊¹` and
  the scalar concavity estimate `lem: concavity` — and that `d ≥ 3` would need
  a Hessian estimate on small geodesic caps.  It names no such estimate and
  claims no bound in `d ≥ 3`, so there is nothing to put on the books; the
  dimension hypothesis of both theorems above is the record of it.

The conditional atom-count corollary is proved in `Section3_AtomCount`.
Its cluster-mass estimate is an explicit input, and every input is witnessed
there by the actual antipodal SOPD measure.

The mass-bound inputs have a complete joint witness at the actual threshold
`β=4`: the quadratic symmetric pair at angles `1/8,2π-1/8`, with equal
positive masses and one weight `-exp(4 cos(1/4))`. The actual measure is
stationary and the circular cluster width is exactly `1/4`. The primitive,
Lipschitz activation, zero activation value, distinct atoms, and full
pairwise circular-distance condition are all included in the witness.

Source: arXiv:2601.21366v2, `thm: bound`, `cor: bound`.
-/

import Transformer.Perceptron.Atoms
import Transformer.Perceptron.Geodesic
import Transformer.Perceptron.Section3_ClusterCounterexample
import Transformer.Perceptron.Section5_StationaryCluster
import Mathlib.Analysis.Real.Pi.Bounds

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Perceptron

/-- **Theorem (thm: bound).**  Let `d = 2`, `β > 0` and `σ` globally Lipschitz
with constant `1` and `σ(0) = 0`. Let `μ = Σ_i m_i δ_{θ_i}` be a stationary
Wasserstein critical point as in `eq: atomic.thm.bound`, and let
`S ⊆ ⟦1,N⟧` with `|S| ≥ 2` satisfy the cluster condition

  `max_{i,j ∈ S} min_{k ∈ ℤ} |θ_i - θ_j + 2πk| ≤ 1/(2√β)`.

Then, for `β` large enough, `Σ_{i ∈ S} m_i ≤ 0.5742 + O(e^{-β})`, the
constant of `O(e^{-β})` depending on the weights `ϑ` — see the module
docstring.

The source assumes SOPD; the proof here needs only stationarity, which
strengthens this first conclusion. The constants can be chosen as
`β₀=4` and `C=2 Σ_j |ω_j| ‖a_j‖²+1`, before the temperature,
activation and configuration, with the exact printed cluster scale.

Source: arXiv:2601.21366v2, §3.2, `thm: bound`, and §5.4, `eq: conclusion`. -/
theorem bound_cluster_mass :
    ∀ (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2),
    ∃ C β₀ : ℝ, 0 < C ∧ 0 < β₀ ∧
      ∀ β : ℝ, β₀ ≤ β → ∀ φ σ : ℝ → ℝ, (∀ s : ℝ, HasDerivAt φ (2 * σ s) s) →
        LipschitzWith 1 σ → σ 0 = 0 →
      ∀ (N : ℕ) (m θ : Idx N → ℝ) (μ : Perspective.ProbSphere 2),
        IsAtomicOnCircle N m θ μ → IsStationary β σ ω a μ →
      ∀ S : Finset (Idx N), 2 ≤ S.card →
        (∀ i ∈ S, ∀ j ∈ S, ∃ k : ℤ,
          |θ i - θ j + 2 * π * (k : ℝ)| ≤ 1 / (2 * Real.sqrt β)) →
        ∑ i ∈ S, m i ≤ 0.5742 + C * Real.exp (-β) := by
  intro ω a
  refine ⟨2 * ∑ j : Idx 2, |ω j| * ‖a j‖ ^ 2 + 1, 4, by positivity, by norm_num, ?_⟩
  intro β hβ φ σ hφ hσ hzero N m θ μ hatom hμ S hcard hpair
  exact cluster_mass_bound_of_stationary hatom β hβ φ σ hφ hσ hzero ω a hμ S hcard hpair

/-- Every mass-bound input is jointly realized at `β=4` by the actual
quadratic two-atom stationary measure, including the complete cluster
condition across angle zero and the actual positive-mass presentation.
Source: arXiv:2601.21366v2, §3.2, `thm: bound`, and §5.4, `eq:dist.cond`. -/
example : (4 : ℝ) ≤ 4 ∧
    (∀ r : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * id r) r) ∧
    LipschitzWith 1 (id : ℝ → ℝ) ∧ id (0 : ℝ) = 0 ∧
    IsAtomicOnCircle 2 (fun _ => 1 / 2) quadraticClusterAngles quadraticClusterProb ∧
    IsStationary 4 id quadraticClusterWeights quadraticClusterNeuron quadraticClusterProb ∧
    2 ≤ (Finset.univ : Finset (Idx 2)).card ∧
    (∀ i ∈ (Finset.univ : Finset (Idx 2)), ∀ j ∈ Finset.univ, ∃ k : ℤ,
      |quadraticClusterAngles i - quadraticClusterAngles j + 2 * π * (k : ℝ)| ≤
        1 / (2 * Real.sqrt 4)) := quadraticCluster_stationary_witness

/-- The covering data in **Corollary (cor: bound)** are intervals of length
at most `L`. The conditional counting result is `bound_atom_count` in
`Section3_AtomCount`, with the mass estimate as an explicit input.
If the support of `μ` is covered by `M` arcs
of length at most `L < 2π`, the number `N_ε` of atoms of mass at least `ε`
satisfies, for `β` large enough,

  `N_ε ≤ (M/ε) (1 + 2L√β) (0.5742 + O(e^{-β}))`,

the constant of `O(e^{-β})` depending on the weights `ϑ`, as in
`bound_cluster_mass`.

Source: arXiv:2601.21366v2, §3.2, `cor: bound`. The covering data are
satisfiable: one arc of
length `1` containing three angles `0, δ, 2δ` at `δ=1/(2√β)`,
and any positive `ε`. -/
example (β : ℝ) (hβ : 1 ≤ β) :
    (1 : ℕ) ≤ 1 ∧ (1 : ℝ) < 2 * π ∧
      (∀ j : Idx 1, (fun _ : Idx 1 => (1 : ℝ)) j - (fun _ : Idx 1 => (0 : ℝ)) j ≤ 1) ∧
      (∀ i : Idx 3, ∃ j : Idx 1, ((i : ℕ) : ℝ) * (1 / (2 * Real.sqrt β)) ∈
        Set.Icc ((fun _ : Idx 1 => (0 : ℝ)) j) ((fun _ : Idx 1 => (1 : ℝ)) j)) ∧
      (0 : ℝ) < 1 := by
  have hs1 : (1 : ℝ) ≤ Real.sqrt β := by simpa using Real.sqrt_le_sqrt hβ
  have hs : (0 : ℝ) < Real.sqrt β := lt_of_lt_of_le one_pos hs1
  have hδ : 0 < 1 / (2 * Real.sqrt β) := by positivity
  have hδhalf : 1 / (2 * Real.sqrt β) ≤ 1 / 2 :=
    one_div_le_one_div_of_le (by norm_num) (by linarith)
  refine ⟨le_rfl, by linarith [Real.pi_gt_three], fun _ => by norm_num, fun i => ⟨0, ?_, ?_⟩,
    one_pos⟩
  · positivity
  · have hi2 : ((i : ℕ) : ℝ) ≤ 2 := by exact_mod_cast i.is_le
    nlinarith [Nat.cast_nonneg (α := ℝ) (i : ℕ)]

end Perceptron
end Transformer
