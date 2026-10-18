/-
# Almost every start clusters, for `d ≥ 3` and `β > 0`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

For `β ≠ 0` every solution of `SA` or `USA` converges to a critical point of `𝖤_β`
(`exists_tendsto_isCriticalEBeta_of_SA`, `exists_tendsto_isCriticalEBeta_of_USA`: the
Łojasiewicz theorem), and for `d ≥ 3` and `β > 0` almost every start has a solution that
converges to no critical point with two distinct tokens (`ae_SA_not_tendsto_nonTrivial`,
`ae_USA_not_tendsto_nonTrivial`: these are strict saddles, which the center-stable manifold
theorem avoids).  So for almost every start the limit is a consensus, and the solution is
unique (`SA_unique`, `USA_unique`): the start lies in `𝒮_β` and in its `USA` counterpart
(`ae_mem_clusteringSet_of_pos`).  The step from limits to clusters holds for any
dynamics with unique solutions (`exists_cluster_of_not_tendsto_nonTrivial`).

Source: arXiv:2312.10794v5, §6, `thm: boumal`, and the outline of the gradient-flow
proofs at the end of §5; arXiv:2405.18273, `thm:spheres`, for the landscape.
-/

import Transformer.Perspective.SphereFlowAvoidance
import Transformer.Perspective.Section2_SphereFlow

open Filter MeasureTheory
open scoped Topology

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **No unsynchronized limit means one cluster.**  For dynamics `D` with unique
solutions that converge to critical points of `𝖤_β`: if no solution from `X₀` converges
to a critical point with two distinct tokens, then every solution from `X₀` converges to
one point `x⋆` of the sphere. -/
theorem exists_cluster_of_not_tendsto_nonTrivial (hn : 0 < n) {β : ℝ}
    {D : (ℝ → SphereTuple d n) → Prop} (hD : ∀ X Y, D X → D Y → X 0 = Y 0 → X = Y)
    (hlim : ∀ X, D X → ∃ Xstar : SphereTuple d n, IsCriticalEBeta d n β Xstar ∧
      ∀ i, Tendsto (fun t => (X t i : EucSpace d)) atTop (𝓝 (Xstar i : EucSpace d)))
    {X₀ : SphereTuple d n}
    (hX₀ : ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → D X →
      ∀ Xstar : SphereTuple d n, IsCriticalEBeta d n β Xstar → NonTrivialTuple d n Xstar →
        ¬∀ i, Tendsto (fun t => (X t i : EucSpace d)) atTop (𝓝 (Xstar i : EucSpace d))) :
    ∃ x_star : SSphere d, ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → D X →
      ∀ i, Tendsto (fun t => (X t i : EucSpace d) - x_star) atTop (𝓝 0) := by
  by_cases hex : ∃ X, X 0 = X₀ ∧ D X
  · obtain ⟨X, hX0, hX⟩ := hex
    obtain ⟨Xstar, hcrit, hconv⟩ := hlim X hX
    have htriv : ∀ i j, (Xstar i : EucSpace d) = Xstar j := by
      by_contra h
      push Not at h
      exact hX₀ X hX0 hX Xstar hcrit h hconv
    refine ⟨Xstar ⟨0, hn⟩, fun Y hY0 hY i => ?_⟩
    rw [← hD X Y hX hY (hX0.trans hY0.symm), htriv ⟨0, hn⟩ i]
    exact tendsto_sub_nhds_zero_iff.2 (hconv i)
  · push Not at hex
    exact ⟨X₀ ⟨0, hn⟩, fun X hX0 hX => absurd hX (hex X hX0)⟩

/-- The hypotheses of `exists_cluster_of_not_tendsto_nonTrivial` hold for two tokens of
`𝕊²` under dynamics with no solution. -/
example := exists_cluster_of_not_tendsto_nonTrivial (d := 3) (n := 2) (β := 1) two_pos
  (D := fun _ => False) (fun _ _ h => h.elim) (fun _ h => h.elim)
  (X₀ := fun _ => basePoint 2) fun _ _ h => h.elim

/-- **Almost every start clusters, for `d ≥ 3` and `β > 0`:** under the uniform law,
almost every initial sequence lies in `𝒮_β` and in its `USA` counterpart. -/
theorem ae_mem_clusteringSet_of_pos (hd : 3 ≤ d) (hn : 0 < n) {β : ℝ} (hβ : 0 < β)
    (P : Measure (SphereTuple d n)) (hP : UniformTuple d n P) :
    ∀ᵐ X₀ ∂P, X₀ ∈ clusteringSet d n β ∧ X₀ ∈ clusteringSetUSA d n β := by
  have hd1 : 1 ≤ d := by omega
  refine (ae_iff_ae_volume hd1 hP).2 ?_
  filter_upwards [ae_SA_not_tendsto_nonTrivial hd1 hd hβ,
    ae_USA_not_tendsto_nonTrivial hd1 hd hβ] with x hSA hUSA
  exact ⟨exists_cluster_of_not_tendsto_nonTrivial hn (fun X Y hX hY h0 => SA_unique β X Y hX hY h0)
      (fun _ hX => exists_tendsto_isCriticalEBeta_of_SA hd1 hβ.ne' hX) hSA,
    exists_cluster_of_not_tendsto_nonTrivial hn (fun _ _ hX hY h0 => USA_unique β hX hY h0)
      (fun _ hX => exists_tendsto_isCriticalEBeta_of_USA hd1 hβ.ne' hX) hUSA⟩

/-- The hypotheses of `ae_mem_clusteringSet_of_pos` hold for two tokens of `𝕊²`, `β = 1`
and the product of shell laws. -/
example := ae_mem_clusteringSet_of_pos (d := 3) (n := 2) le_rfl two_pos one_pos _
  (uniformTuple_shellLaw (d := 3) (by norm_num))

end Perspective
end Transformer
