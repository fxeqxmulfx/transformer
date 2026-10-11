/-
# Two limit principles for the identity-value proof

ArXiv:2305.05465v6, §8.1, Steps 2--4 of the proof of `t:Idcase11`.
The first principle rules out persistent growth away from the candidates:
a nonnegative uniformly continuous function dominated by the derivative
of a convergent energy tends to zero. The proof uses a fixed short
interval around any hypothetical late positive value.

The second principle rules out travel between finitely many candidates.
A continuous path confined eventually to a compact set, with all its
cluster points in a finite set, converges to one of those points. A small
sphere around one cluster point contains no candidate, so compactness
makes the path eventually avoid the sphere. Connectedness of a future
time interval then keeps the path on the sphere's inner side.

Both principles are proved independently of a particular ODE. Their
inputs are verified for the actual dynamics in subsequent modules.
-/

import Transformer.Basic
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Topology.MetricSpace.Pseudo.Defs
import Mathlib.Topology.Compactness.Compact

open Real Filter Topology

namespace Transformer.Clusters

/-- Persistent positive attention gain would force a fixed energy
increase, contradicting energy convergence. Source: arXiv:2305.05465v6,
§8.1, Step 2 of `t:Idcase11`; the abstract form uses uniform continuity
instead of the paper's quantitative candidate-distance estimate. -/
theorem tendsto_zero_of_deriv_domination (F F' g : ℝ → ℝ) (L : ℝ)
    (hF : ∀ t : ℝ, HasDerivAt F (F' t) t)
    (hdom : ∀ t : ℝ, 0 ≤ t → g t ≤ F' t)
    (hnonneg : ∀ t : ℝ, 0 ≤ t → 0 ≤ g t)
    (huniform : UniformContinuousOn g (Set.Ici 0))
    (hlim : Tendsto F atTop (nhds L)) : Tendsto g atTop (nhds 0) := by
  have hFc : Continuous F := continuous_iff_continuousAt.mpr fun t => (hF t).continuousAt
  rw [tendsto_order]
  refine ⟨fun a ha => ?_, fun ε hε => ?_⟩
  · filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
    exact ha.trans_le (hnonneg t ht)
  · by_contra hne
    have hfreq : ∃ᶠ t in atTop, ε ≤ g t := by
      simpa only [not_eventually, not_lt] using hne
    obtain ⟨δ, hδ, hU⟩ := Metric.uniformContinuousOn_iff.mp huniform (ε / 2) (half_pos hε)
    let τ := δ / 2
    have hτ : 0 < τ := half_pos hδ
    have hτδ : τ < δ := half_lt_self hδ
    have hdiff : Tendsto (fun t => F (t + τ) - F t) atTop (nhds 0) := by
      simpa only [Function.comp_apply, id_eq, sub_self] using
        (hlim.comp (tendsto_atTop_add_const_right _ τ tendsto_id)).sub hlim
    obtain ⟨t, ⟨ht, ht0⟩, hsmall⟩ :=
      ((hfreq.and_eventually (eventually_ge_atTop (0 : ℝ))).and_eventually
        (hdiff.eventually_lt_const (mul_pos (half_pos hε) hτ))).exists
    have hlow (s : ℝ) (hs : s ∈ Set.Icc t (t + τ)) : ε / 2 ≤ g s := by
      have hd : dist t s < δ := by
        rw [Real.dist_eq, abs_of_nonpos (sub_nonpos.mpr hs.1)]
        linarith [hs.2]
      have hnear := hU t ht0 s (ht0.trans hs.1) hd
      rw [Real.dist_eq] at hnear
      linarith [(abs_lt.mp hnear).2]
    have htu : t ≤ t + τ := by linarith
    have hgrowth := (convex_Icc t (t + τ)).mul_sub_le_image_sub_of_le_deriv
      hFc.continuousOn (fun s _ => (hF s).differentiableAt.differentiableWithinAt)
      (fun s hs => by
        rw [(hF s).deriv]
        have hs' := interior_subset hs
        exact (hlow s hs').trans (hdom s (ht0.trans hs'.1)))
      t ⟨le_rfl, htu⟩ (t + τ) ⟨htu, le_rfl⟩ htu
    simp only [add_sub_cancel_left] at hgrowth
    linarith

/-- All inputs are realized by a constant energy with zero gain; no
derivative or convergence input is left unproved. Source: arXiv:2305.05465v6,
§8.1, the stationary case of the energy argument. -/
example : (∀ t : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) 0 t) ∧
    (∀ t : ℝ, 0 ≤ t → (0 : ℝ) ≤ 0) ∧ (∀ t : ℝ, 0 ≤ t → (0 : ℝ) ≤ 0) ∧
    UniformContinuousOn (fun _ : ℝ => (0 : ℝ)) (Set.Ici 0) ∧
    Tendsto (fun _ : ℝ => (0 : ℝ)) atTop (nhds 0) :=
  ⟨fun t => hasDerivAt_const t 0, fun _ _ => le_rfl, fun _ _ => le_rfl,
    uniformContinuous_const.uniformContinuousOn, tendsto_const_nhds⟩

/-- A finite candidate set admits a ball around any point containing no
other candidate. Source: arXiv:2305.05465v6, §8.1, Step 1 of `t:Idcase11`,
the separation used to define `S_δ` and its connected components. -/
theorem exists_isolating_ball_of_finite {E : Type*} [MetricSpace E]
    (C : Set E) (hC : C.Finite) (p : E) :
    ∃ δ : ℝ, 0 < δ ∧ Metric.ball p δ ∩ C ⊆ {p} := by
  have hnot : p ∈ (C \ {p})ᶜ := fun h => h.2 (Set.mem_singleton p)
  obtain ⟨δ, hδ, hball⟩ := Metric.mem_nhds_iff.mp
    (hC.sdiff.isClosed.isOpen_compl.mem_nhds hnot)
  refine ⟨δ, hδ, ?_⟩
  intro q hq
  by_contra hne
  exact hball hq.1 ⟨hq.2, hne⟩

/-- The finiteness hypothesis holds for a nonempty singleton;
arXiv:2305.05465v6, §8.1, the stationary candidate set. -/
example (p : EucSpace 1) : ({p} : Set (EucSpace 1)).Finite := Set.finite_singleton p

/-- A continuous, eventually compact path whose cluster points lie in a
finite set converges to one member of that set.
Source: arXiv:2305.05465v6, §8.1, Steps 3--4 of `t:Idcase11`.
The inputs express compact confinement and an independently proved
description of every cluster point, rather than assuming path convergence. -/
theorem exists_limit_of_finite_cluster_set {E : Type*} [MetricSpace E]
    (X : ℝ → E) (hX : Continuous X) (K C : Set E) (hK : IsCompact K) (hC : C.Finite)
    (hmem : ∀ᶠ t in atTop, X t ∈ K)
    (hcluster : ∀ p ∈ K, MapClusterPt p atTop X → p ∈ C) :
    ∃ p ∈ C, Tendsto X atTop (nhds p) := by
  obtain ⟨p, hpK, hp⟩ := hK.exists_mapClusterPt_of_frequently hmem.frequently
  refine ⟨p, hcluster p hpK hp, Metric.tendsto_nhds.mpr ?_⟩
  intro ε hε
  obtain ⟨δ, hδ, hball⟩ := exists_isolating_ball_of_finite C hC p
  let r := min ε δ / 2
  have hmin : 0 < min ε δ := lt_min hε hδ
  have hr : 0 < r := half_pos hmin
  have hrε : r < ε := (half_lt_self hmin).trans_le (min_le_left _ _)
  have hrδ : r < δ := (half_lt_self hmin).trans_le (min_le_right _ _)
  have havoid : ∀ᶠ t in atTop, dist (X t) p ≠ r := by
    by_contra hne
    have hfreq : ∃ᶠ t in atTop, dist (X t) p = r := by
      simpa only [not_eventually, not_not] using hne
    have hlevel : IsCompact (K ∩ {q | dist q p = r}) :=
      hK.inter_right (isClosed_eq (continuous_id.dist continuous_const) continuous_const)
    obtain ⟨q, ⟨hqK, hqr⟩, hq⟩ := hlevel.exists_mapClusterPt_of_frequently
      ((hfreq.and_eventually hmem).mono fun _ ht => ⟨ht.2, ht.1⟩)
    change dist q p = r at hqr
    have hqp : q ≠ p := by
      intro heq
      rw [heq, dist_self] at hqr
      linarith
    have hqball : q ∈ Metric.ball p δ := by change dist q p < δ; rw [hqr]; exact hrδ
    exact hqp (Set.mem_singleton_iff.mp (hball ⟨hqball, hcluster q hqK hq⟩))
  obtain ⟨T, hT⟩ := eventually_atTop.mp havoid
  obtain ⟨u, hu, huT⟩ :=
    ((hp.frequently (Metric.ball_mem_nhds p hr)).and_eventually (eventually_ge_atTop T)).exists
  filter_upwards [eventually_ge_atTop T] with t ht
  have hinside : dist (X t) p < r := isPreconnected_Ici.gt_of_ne
    (hX.dist continuous_const).continuousOn hT ⟨u, huT, hu⟩ ht
  exact hinside.trans hrε

/-- Every compactness, continuity, finiteness, confinement and cluster
input is satisfied by a constant path in its singleton.
Source: arXiv:2305.05465v6, §8.1, stationary-token limit. -/
example (p : EucSpace 1) : Continuous (fun _ : ℝ => p) ∧
    IsCompact ({p} : Set (EucSpace 1)) ∧ ({p} : Set (EucSpace 1)).Finite ∧
    (∀ᶠ t : ℝ in atTop, (fun _ : ℝ => p) t ∈ ({p} : Set (EucSpace 1))) ∧
    (∀ q ∈ ({p} : Set (EucSpace 1)), MapClusterPt q atTop (fun _ : ℝ => p) →
      q ∈ ({p} : Set (EucSpace 1))) :=
  ⟨continuous_const, isCompact_singleton, Set.finite_singleton _,
    Eventually.of_forall (fun _ => Set.mem_singleton _), fun _ hq _ => hq⟩

end Transformer.Clusters
