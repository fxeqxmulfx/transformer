/-
# Mean-Field Dynamics — the pairing phase (§5 of 2512.01868v4)

* `eq: init_clust`      — a discrete multi-cluster initial datum
                          `μ_0 = Σ_j α_j δ_{x_j(0)}`, and `eq: SA` read there
                          (`clusterSA`);
* the pairing dynamics of `Theorem thm: agazzi_merge` (Bruno, Pasqualotto &
  Agazzi), in which the closest pair attends to each other and every other
  cluster stands still (`hardmaxPair`).

The initial weights `α` enter through the dynamics: `clusterSA` is `eq: SA` read
at the measure `Σ_j α_j δ_{x_j(t)}`, which is the mean-field field
`Perspective.vectorField` evaluated there, written out as a finite sum so that
no probability-measure bundling is needed.

Self-attention is kept in `clusterSA`, as `eq: SA` has it; the `argmax` limit
of the survey's footnote forbids it, and `hardmaxPair` — where each of the two
merging clusters follows the *other* one — is that limit.

`thm: agazzi_merge`, the convergence of the first to the second as `β → ∞`, is
false as the survey prints it: `MeanField.not_agazzi_merge`.  What
`hardmaxPair` itself does — the clusters outside the pair standing still, and
the pair approaching each other without ever arriving — is proved in
`MeanField.PairMerge`.
-/

import Transformer.Basic
import Transformer.MeanField.Basic
import Transformer.Perspective.Section2_FlowMap

open scoped BigOperators
open Real

namespace Transformer
namespace MeanField

variable (d : ℕ)

/-- **Equation (eq: SA) at a discrete multi-cluster configuration
(eq: init_clust).**

For `μ(t) = Σ_j α_j δ_{x_j(t)}` the mean-field field reads

  `ẋ_i = Proj_{x_i} ( (Σ_k α_k e^{β⟨x_i,x_k⟩})⁻¹ Σ_j α_j e^{β⟨x_i,x_j⟩} x_j )`.

Source: arXiv:2512.01868v4, §5, `eq: init_clust` with `eq: SA`. -/
def clusterSA (K : ℕ) (α : Idx K → ℝ) (β : ℝ) (X : ℝ → SphereTuple d K) : Prop :=
  ∀ t : ℝ, ∀ i : Idx K,
    HasDerivAt (fun s => (X s i : EucSpace d))
      (proj d ((X t i : EucSpace d))
        ((∑ k : Idx K,
            α k * Real.exp (β * inner (𝕜 := ℝ)
              ((X t i : EucSpace d)) ((X t k : EucSpace d))))⁻¹ •
          ∑ j : Idx K,
            (α j * Real.exp (β * inner (𝕜 := ℝ)
              ((X t i : EucSpace d)) ((X t j : EucSpace d))))
              • ((X t j : EucSpace d)))) t

/-- **The limiting pairing dynamics.**

  `ẏ_ī = Proj_{y_ī}(y_j̄)`,  `ẏ_j̄ = Proj_{y_j̄}(y_ī)`,  `ẏ_k = 0` otherwise.

This is the `argmax` rule of the survey's footnote once `(ī, j̄)` is the
unique closest pair: each of the two attends to the other, and every other
cluster attends to one of them at an exponentially slower rate, which the
rescaling of time sends to zero.

Source: arXiv:2512.01868v4, §5, `thm: agazzi_merge`. -/
def hardmaxPair (K : ℕ) (ibar jbar : Idx K) (Y : ℝ → SphereTuple d K) : Prop :=
  ∀ s : ℝ, ∀ k : Idx K,
    HasDerivAt (fun r => (Y r k : EucSpace d))
      (if k = ibar then proj d ((Y s ibar : EucSpace d)) ((Y s jbar : EucSpace d))
       else if k = jbar then proj d ((Y s jbar : EucSpace d)) ((Y s ibar : EucSpace d))
       else 0) s

end MeanField
end Transformer
