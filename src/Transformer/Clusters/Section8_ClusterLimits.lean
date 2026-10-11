/-
# Every rescaled token converges to a candidate

ArXiv:2305.05465v6, §8.1, `t:Idcase11` and the candidate condition `e:Ax15`.
Any cluster point belongs to every future hull by nesting and closedness.
The vanishing maximal attention gain bounds its supporting functional on
their intersection. An exposed face supplies a vertex attaining this
bound, so the cluster point is a candidate in the exact vertex-based
definition used in the paper.

The intersection is a finite polytope. If `A` is injective, its candidate
set is finite. Compact confinement and continuity then force each token
to converge to one candidate. No token-convergence input, quantitative
candidate-distance estimate, or unproved paper claim is assumed.
-/

import Transformer.Clusters.Section8_GainLimit
import Transformer.Clusters.Section8_HullLimit
import Transformer.Clusters.Section8_CandidateFinite
import Mathlib.Analysis.Convex.Exposed
import Mathlib.Topology.Bases

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d n : ℕ}

/-- A supporting inequality on a compact set is attained at an extreme
point of its exposed face. Source: arXiv:2305.05465v6, §8.1, the maximum
over vertices in `e:Ax15`. This verifies attainment as well as the bound. -/
theorem IsAttentionMax.of_inner_le (A : ParamMatrix d) (P : Set (EucSpace d))
    (hP : IsCompact P) (x : EucSpace d) (hx : x ∈ P)
    (hle : ∀ y ∈ P, inner (𝕜 := ℝ) (A x) (A y) ≤ ‖A x‖ ^ 2) :
    IsAttentionMax A P x := by
  let f : EucSpace d →L[ℝ] ℝ := (innerSL ℝ (A x)).comp A
  let C : Set (EucSpace d) := {y ∈ P | ∀ z ∈ P, f z ≤ f y}
  have hface : IsExposed ℝ P C := fun _ => ⟨f, rfl⟩
  have hxC : x ∈ C := by
    refine ⟨hx, ?_⟩
    intro y hy
    simpa only [f, ContinuousLinearMap.comp_apply, innerSL_apply_apply,
      real_inner_self_eq_norm_sq] using hle y hy
  obtain ⟨v, hv⟩ := (hface.isCompact hP).extremePoints_nonempty ⟨x, hxC⟩
  have hvC : v ∈ C := extremePoints_subset hv
  have hvalue : f v = ‖A x‖ ^ 2 := by
    apply le_antisymm (hle v hvC.1)
    simpa only [f, ContinuousLinearMap.comp_apply, innerSL_apply_apply,
      real_inner_self_eq_norm_sq] using hvC.2 x hx
  refine ⟨⟨v, hface.isExtreme.extremePoints_subset_extremePoints hv, hvalue⟩, ?_⟩
  rintro _ ⟨y, hy, rfl⟩
  exact hle y (extremePoints_subset hy)

/-- Compactness, membership, and the full supporting inequality all hold
at a singleton vertex. Source: arXiv:2305.05465v6, §8.1, `e:Ax15`. -/
example (A : ParamMatrix d) (x : EucSpace d) :
    IsCompact ({x} : Set (EucSpace d)) ∧ x ∈ ({x} : Set (EucSpace d)) ∧
      ∀ y ∈ ({x} : Set (EucSpace d)), inner (𝕜 := ℝ) (A x) (A y) ≤ ‖A x‖ ^ 2 := by
  refine ⟨isCompact_singleton, Set.mem_singleton _, ?_⟩
  rintro y rfl
  exact (real_inner_self_eq_norm_sq _).le

/-- Any cluster point of a token is in every token hull, by nesting.
Source: arXiv:2305.05465v6, §8.1, `p:noninc` and the limiting hull in
`t:Idcase11`. The formal dynamics give nesting at all real times. -/
theorem IdRescaledDynamics.cluster_mem_tokenHull (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) (i : Idx n)
    (p : EucSpace d) (hp : MapClusterPt p atTop (fun t => Z t i)) (t : ℝ) :
    p ∈ tokenHull (Z t) := by
  apply (isCompact_tokenHull (Z t)).isClosed.mem_of_mapClusterPt hp
  filter_upwards [eventually_ge_atTop t] with s hs
  exact tokenHull_antitone A Z hZ hs (mem_tokenHull (Z s) i)

/-- An actual stationary solution realizes both the ODE and cluster-point
hypotheses, at a nonzero unit token. Source: arXiv:2305.05465v6, §8.1. -/
example (z : EucSpace 1) : IdRescaledDynamics (1 : ParamMatrix 1)
    (fun _ (_ : Idx 2) => z) ∧ MapClusterPt z atTop (fun _ : ℝ => z) :=
  ⟨idRescaledDynamics_const 1 z, tendsto_const_nhds.mapClusterPt⟩

/-- The supporting displacement at a cluster point is nonpositive for
every point of the future-hull intersection. Source: arXiv:2305.05465v6,
§8.1, `e:Ax15` and Step 2 of `t:Idcase11`. A cluster sequence transfers
the actual hull bound and the vanishing gain to the limit. -/
theorem IdRescaledDynamics.inner_gap_nonpos_at_cluster (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) (i : Idx n)
    (p : EucSpace d) (hp : MapClusterPt p atTop (fun t => Z t i)) (y : EucSpace d)
    (hy : ∀ t : ℝ, 0 ≤ t → y ∈ tokenHull (Z t)) :
    inner (𝕜 := ℝ) (A p) (A (y - p)) ≤ 0 := by
  obtain ⟨τ, hlim, htime⟩ := hp.exists_seq_tendsto
  change Tendsto (fun k => Z (τ k) i) atTop (nhds p) at hlim
  have hc : Continuous (fun x : EucSpace d => inner (𝕜 := ℝ) (A x) (A (y - x))) := by
    fun_prop
  apply le_of_tendsto_of_tendsto (hc.tendsto p |>.comp hlim)
    ((hZ.tendsto_attentionGain_zero A Z i).comp htime)
  filter_upwards [htime.eventually (eventually_ge_atTop (0 : ℝ))] with k hk
  exact inner_gap_le_attentionGain_of_mem_hull A (Z (τ k)) i y (hy (τ k) hk)

/-- All ODE, cluster-point and future-hull inputs hold for coincident
stationary tokens. Source: arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
example (z : EucSpace 1) : IdRescaledDynamics (1 : ParamMatrix 1)
    (fun _ (_ : Idx 2) => z) ∧ MapClusterPt z atTop (fun _ : ℝ => z) ∧
    ∀ t : ℝ, 0 ≤ t → z ∈ tokenHull (fun _ : Idx 2 => z) :=
  ⟨idRescaledDynamics_const 1 z, tendsto_const_nhds.mapClusterPt,
    fun _ _ => mem_tokenHull _ 0⟩

/-- Every actual cluster point is in the exact candidate set of the
limiting hull. Source: arXiv:2305.05465v6, §8.1, `t:Idcase11` and `e:Ax15`.
Positive definiteness is not needed until candidate finiteness is invoked. -/
theorem IdRescaledDynamics.cluster_mem_limitCandidates_future (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) (i : Idx n)
    (p : EucSpace d) (hp : MapClusterPt p atTop (fun t => Z t i)) :
    p ∈ limitCandidates A {x | ∀ t : ℝ, 0 ≤ t → x ∈ tokenHull (Z t)} := by
  have hpHull : ∀ t : ℝ, 0 ≤ t → p ∈ tokenHull (Z t) :=
    fun t _ => hZ.cluster_mem_tokenHull A Z i p hp t
  refine ⟨hpHull, IsAttentionMax.of_inner_le A _ (isCompact_future_tokenHull Z) p hpHull ?_⟩
  intro y hy
  have hgap := hZ.inner_gap_nonpos_at_cluster A Z i p hp y hy
  rw [map_sub, inner_sub_right, real_inner_self_eq_norm_sq] at hgap
  linarith

/-- The candidate-identification inputs are realized by genuine
stationary unit tokens. Source: arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
example : IdRescaledDynamics (1 : ParamMatrix 1)
    (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) ∧
    MapClusterPt (basePoint 0 : EucSpace 1) atTop (fun _ : ℝ => (basePoint 0 : EucSpace 1)) :=
  ⟨idRescaledDynamics_const 1 _, tendsto_const_nhds.mapClusterPt⟩

/-- **Each token converges to a candidate.** Source: arXiv:2305.05465v6,
§8.1, `t:Idcase11`. The exact limiting hull is its future intersection;
injectivity of `A` makes the candidate set finite. The token index supplies
`n > 0`, so the limiting polytope is nonempty. -/
theorem IdRescaledDynamics.exists_limit_in_candidates (A : ParamMatrix d)
    (hA : Function.Injective A) (Z : ℝ → Idx n → EucSpace d)
    (hZ : IdRescaledDynamics A Z) (i : Idx n) :
    ∃ p ∈ limitCandidates A {x | ∀ t : ℝ, 0 ≤ t → x ∈ tokenHull (Z t)},
      Tendsto (fun t => Z t i) atTop (nhds p) := by
  have hn : 0 < n := Fin.pos_iff_nonempty.mpr ⟨i⟩
  obtain ⟨S, -, -, hHull, -⟩ := hZ.exists_tokenHull_polytope_limit A Z hn
  have hfinite : (limitCandidates A {x | ∀ t : ℝ, 0 ≤ t → x ∈ tokenHull (Z t)}).Finite := by
    rw [← hHull]
    exact finite_limitCandidates_convexHull A hA S
  apply exists_limit_of_finite_cluster_set (fun t => Z t i) (hZ.continuous_token A Z i)
    (tokenHull (Z 0)) _ (isCompact_tokenHull (Z 0)) hfinite
  · filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
    exact mem_tokenHull_zero A Z hZ i ht
  · exact fun p _ hp => hZ.cluster_mem_limitCandidates_future A Z i p hp

/-- Injectivity and the complete dynamical input hold simultaneously for
the identity operator and stationary unit tokens; arXiv:2305.05465v6, §8.1. -/
example : Function.Injective (ContinuousLinearMap.id ℝ (EucSpace 1)) ∧
    IdRescaledDynamics (1 : ParamMatrix 1)
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨Function.injective_id, idRescaledDynamics_const 1 _⟩

end Transformer.Clusters
