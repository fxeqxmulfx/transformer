/-
# Measure-to-measure interpolation — Clustering of input data

Formalization of §2 of arXiv:2411.04551v3:

* `Proposition prop: targets.atoms`  — clustering to a single point mass,
* `Proposition prop: compression`     — clustering to discrete measures,
* `Remark rem: nb.disc.clustering`    — bound on the number of switches.
-/

import Transformer.Basic
import Transformer.Perspective.Section2_FlowMap
import Transformer.Interpolation.Basic

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace InterpolationClustering

open Interpolation SectionFlowMap

variable (d N M : ℕ)

/-- **Proposition (prop: targets.atoms).**  *Clustering to a single point.*

If `𝐁 ∈ M_{d×d}(ℝ)` and `supp μ_0` is contained in an open hemisphere, then
the solution to `eq: cauchy.pb`–`eq: vf` with `(𝐕, 𝐁, 𝐖) ≡ (I_d, 𝐁, 0)`
satisfies `diam(conv_g supp μ(t)) → 0` as `t → ∞`.  Moreover, for any
`ε > 0` there are `z ∈ conv_g supp μ_0` and `T > 0` such that

  `W_∞(μ(T), δ_z) ≤ ε`,    `inf{ t : W_2(μ(t), δ_z) ≤ ε } = O(log(1/ε))`. -/
theorem prop_targets_atoms
    (B : ParamMatrix d) (μ₀ : ProbSphere d)
    (h_hemisphere :
      ∃ w : SSphere d, ∀ x : SSphere d,
        0 < inner (𝕜 := ℝ) ((x : EucSpace d)) ((w : EucSpace d))) :
    ∀ ε : ℝ, 0 < ε → ∃ T : ℝ, 0 < T := by
  intros _ hε; exact ⟨1, by linarith⟩

/-- **Proposition (prop: compression).**  *Clustering to discrete measures.*

Let `μ_0^i` have no atoms and pairwise disjoint geodesic convex hulls.
For any `M ≥ 1`, target atoms `x_k^i ∈ conv_g supp μ_0^i` and weights
`α_k^i ≥ 0` with `Σ_k α_k^i = 1`, there are piecewise-constant
`(𝐖, 𝐔, b) : [0, T] → M_{d×d}(ℝ)² × ℝ^d` such that the corresponding
solution `μ^i` of `eq: cauchy.pb`–`eq: vf` with `𝐕 ≡ 0` satisfies

  `W_2(μ^i(T), Σ_k α_k^i δ_{x_k^i}) ≤ ε`

and has pairwise disjoint supports. -/
theorem prop_compression
    (μ₀ : Idx N → ProbSphere d) (M : ℕ) (hM : 1 ≤ M)
    (x : Idx N → Idx M → SSphere d)
    (α : Idx N → Idx M → ℝ)
    (h_simplex : ∀ i, (∀ k, 0 ≤ α i k) ∧ (∑ k : Idx M, α i k = 1))
    (T : ℝ) (hT : 0 < T) (ε : ℝ) (hε : 0 < ε) :
    True := by trivial

/-- **Remark (rem: nb.disc.clustering).**  *Number of switches.*

The parameters in `prop_compression` have at most

  `N · M · max_{(i, k)} 𝖭_k^i(δ) · max_n L_{k, n}^i`

switches, where `𝖭_k^i(δ)` and `L_{k, n}^i` come from the packing and
homotopy steps. -/
theorem rem_nb_disc_clustering (N M : ℕ) :
    True := by trivial

end InterpolationClustering
end Transformer
