/-
# Every leading token converges to a candidate of the limiting polytope

ArXiv:2305.05465v6, §10, the adaptation of Steps 3--4 from §8.1 in
the proof of `t:multiplicity`. The actual projected hulls are nested.
Their compact intersection is a finite polytope, and every projected
token cluster point lies in that intersection.

The vanishing leading gain supplies its supporting inequality in the
attention form `A π_F`. The proved candidate-finiteness argument then
applies to this genuine leading form. Compact confinement and continuity
force each projected token to converge to one candidate. No convergence
input or commutation of the attention root with a projection is assumed.
-/

import Transformer.Clusters.Section10_ProjectedGain
import Transformer.Clusters.Section8_ClusterLimits

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d n : ℕ}

/-- Every actual projected-token cluster point is in each projected
token hull. Source: arXiv:2305.05465v6, §10, the nested leading hulls
and the limiting polytope in the proof of `t:multiplicity`. -/
theorem RescaledDynamics.projected_cluster_mem_tokenHull (Q K V proj : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z) (i : Idx n)
    (p : EucSpace d) (hp : MapClusterPt p atTop (fun t => proj (Z t i))) (t : ℝ) :
    p ∈ tokenHull (fun j => proj (Z t j)) := by
  apply (isCompact_tokenHull (fun j => proj (Z t j))).isClosed.mem_of_mapClusterPt hp
  filter_upwards [eventually_ge_atTop t] with s hs
  exact hZ.tokenHull_projected_antitone Q K V proj F G lam hV hproj Z hs (mem_tokenHull _ i)

/-- All splitting, projection, ODE, and cluster-point inputs coexist
at a stationary token. Source: arXiv:2305.05465v6, §3, `e:Rres`, and §10. -/
example (z : EucSpace 1) : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1 (fun _ (_ : Idx 2) => z) ∧
    MapClusterPt z atTop (fun _ : ℝ => (1 : ParamMatrix 1) z) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, rescaledDynamics_one_const 1 1 z,
    tendsto_const_nhds.mapClusterPt⟩

/-- At every projected cluster point the leading attention displacement
is nonpositive toward every point of the future-hull intersection.
Source: arXiv:2305.05465v6, §10, the corrected candidate condition in
Steps 2'--4. A cluster sequence transfers the actual vanishing-gain bound. -/
theorem RescaledDynamics.projected_inner_gap_nonpos_at_cluster
    (Q K V proj A : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (hA : IsAttentionRoot Q K A) (Z : ℝ → Idx n → EucSpace d)
    (hZ : RescaledDynamics Q K V Z) (i : Idx n) (p : EucSpace d)
    (hp : MapClusterPt p atTop (fun t => proj (Z t i))) (y : EucSpace d)
    (hy : ∀ t : ℝ, 0 ≤ t → y ∈ tokenHull (fun j => proj (Z t j))) :
    inner (𝕜 := ℝ) (A p) (A (y - p)) ≤ 0 := by
  obtain ⟨τ, hlim, htime⟩ := hp.exists_seq_tendsto
  change Tendsto (fun k => proj (Z (τ k) i)) atTop (nhds p) at hlim
  have hc : Continuous (fun x : EucSpace d => inner (𝕜 := ℝ) (A x) (A (y - x))) := by
    fun_prop
  apply le_of_tendsto_of_tendsto (hc.tendsto p |>.comp hlim)
    ((hZ.tendsto_projected_attentionGain_zero Q K V proj A F G lam hV hproj hA Z i).comp htime)
  filter_upwards [htime.eventually (eventually_ge_atTop (0 : ℝ))] with k hk
  exact inner_gap_le_attentionGain_of_mem_hull A (fun j => proj (Z (τ k) j)) i y (hy (τ k) hk)

/-- The root, actual dynamics, cluster point, and full future-hull
membership hypothesis hold simultaneously for stationary tokens.
Source: arXiv:2305.05465v6, §3, `e:Rres`, and §10, Steps 2'--4. -/
example (z : EucSpace 1) : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧ IsAttentionRoot (1 : ParamMatrix 1) 1 1 ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1 (fun _ (_ : Idx 2) => z) ∧
    MapClusterPt z atTop (fun _ : ℝ => (1 : ParamMatrix 1) z) ∧
    ∀ t : ℝ, 0 ≤ t → z ∈ tokenHull (fun _ : Idx 2 => (1 : ParamMatrix 1) z) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, isAttentionRoot_id 1,
    rescaledDynamics_one_const 1 1 z, tendsto_const_nhds.mapClusterPt,
    fun _ _ => mem_tokenHull _ 0⟩

/-- Every actual projected cluster point satisfies the exact
vertex-based candidate equality for the limiting hull.
Source: arXiv:2305.05465v6, §10, the candidate set used in Steps 2'--4,
with the corrected leading form `A π_F`. -/
theorem RescaledDynamics.projected_cluster_mem_limitCandidates_future
    (Q K V proj A : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (hA : IsAttentionRoot Q K A) (Z : ℝ → Idx n → EucSpace d)
    (hZ : RescaledDynamics Q K V Z) (i : Idx n) (p : EucSpace d)
    (hp : MapClusterPt p atTop (fun t => proj (Z t i))) :
    p ∈ limitCandidates A {x | ∀ t : ℝ, 0 ≤ t → x ∈ tokenHull (fun j => proj (Z t j))} := by
  have hpHull : ∀ t : ℝ, 0 ≤ t → p ∈ tokenHull (fun j => proj (Z t j)) :=
    fun t _ => hZ.projected_cluster_mem_tokenHull Q K V proj F G lam hV hproj Z i p hp t
  refine ⟨hpHull, IsAttentionMax.of_inner_le A _
    (isCompact_future_tokenHull (fun t j => proj (Z t j))) p hpHull ?_⟩
  intro y hy
  have hgap := hZ.projected_inner_gap_nonpos_at_cluster Q K V proj A F G lam hV hproj hA Z i p hp y hy
  rw [map_sub, inner_sub_right, real_inner_self_eq_norm_sq] at hgap
  linarith

/-- All candidate-identification inputs are realized by actual nonzero
stationary rescaled tokens and their cluster point.
Source: arXiv:2305.05465v6, §3, `e:Rres`, and §10. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧ IsAttentionRoot (1 : ParamMatrix 1) 1 1 ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) ∧
    MapClusterPt (basePoint 0 : EucSpace 1) atTop
      (fun _ : ℝ => (1 : ParamMatrix 1) (basePoint 0 : EucSpace 1)) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, isAttentionRoot_id 1,
    rescaledDynamics_one_const 1 1 _, tendsto_const_nhds.mapClusterPt⟩

/-- **Each actual leading projected token converges to one candidate.**
Source: arXiv:2305.05465v6, §10, the conclusion of Steps 3--4 in
`t:multiplicity`. Injectivity of the true attention root makes the
candidate set finite; the actual dynamics supplies every cluster input. -/
theorem RescaledDynamics.exists_projected_limit_in_candidates
    (Q K V proj A : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (hA : IsAttentionRoot Q K A) (Z : ℝ → Idx n → EucSpace d)
    (hZ : RescaledDynamics Q K V Z) (i : Idx n) :
    ∃ p ∈ limitCandidates A {x | ∀ t : ℝ, 0 ≤ t → x ∈ tokenHull (fun j => proj (Z t j))},
      Tendsto (fun t => proj (Z t i)) atTop (nhds p) := by
  have hn : 0 < n := Fin.pos_iff_nonempty.mpr ⟨i⟩
  obtain ⟨S, _, _, hHull, _⟩ :=
    exists_polytope_limit_of_tokenHull_antitone (fun t j => proj (Z t j)) hn
      (fun _ _ _ hst => hZ.tokenHull_projected_antitone Q K V proj F G lam hV hproj Z hst)
  have hfinite : (limitCandidates A
      {x | ∀ t : ℝ, 0 ≤ t → x ∈ tokenHull (fun j => proj (Z t j))}).Finite := by
    rw [← hHull]
    exact finite_limitCandidates_convexHull A hA.injective S
  apply exists_limit_of_finite_cluster_set (fun t => proj (Z t i))
    (proj.continuous.comp (hZ.continuous_token Q K V Z i))
    (tokenHull (fun j => proj (Z 0 j))) _ (isCompact_tokenHull _) hfinite
  · filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
    exact hZ.mem_projected_tokenHull_zero Q K V proj F G lam hV hproj Z i ht
  · exact fun p _ hp =>
      hZ.projected_cluster_mem_limitCandidates_future Q K V proj A F G lam hV hproj hA Z i p hp

/-- The attention root, splitting, projection, and actual dynamics all
hold for a nonzero stationary example.
Source: arXiv:2305.05465v6, §3, `e:Rres`, and §10, `t:multiplicity`. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧ IsAttentionRoot (1 : ParamMatrix 1) 1 1 ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, isAttentionRoot_id 1,
    rescaledDynamics_one_const 1 1 _⟩

end Transformer.Clusters
