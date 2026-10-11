/-
# The shrinking token hull converges to a polytope

The opening step in the proof of `t:Idcase11`, arXiv:2305.05465v6, §8.1.
The paper infers convergence to a polytope from `p:noninc`. Here compactness
of the initial hull gives a convergent subsequence of the entire token tuple.
Its limiting tuple generates the intersection of all future hulls, with at
most `n` generators. Nesting then gives Hausdorff convergence at all times.

This step does not require convergence of the individual token paths,
positive definiteness of the attention operator, or a candidate-limit set.
For `n > 0` the limiting generating set is explicitly nonempty, and its hull
is identified as a set. Thus the real-valued Hausdorff distance cannot hide
an empty proposed limit. Every input is witnessed by an actual stationary
solution of `e:dynIdzi`.
-/

import Transformer.Clusters.Section8_HullContinuity
import Mathlib.Topology.Sequences

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d n : ℕ}

/-- The intersection of all future hulls is closed for any token path;
arXiv:2305.05465v6, §8.1, the limiting hull in `t:Idcase11`.
Each finite hull is compact, hence closed. -/
theorem isClosed_future_tokenHull (Z : ℝ → Idx n → EucSpace d) :
    IsClosed {x | ∀ t : ℝ, 0 ≤ t → x ∈ tokenHull (Z t)} := by
  have heq : {x | ∀ t : ℝ, 0 ≤ t → x ∈ tokenHull (Z t)} =
      ⋂ (t : ℝ) (_ : 0 ≤ t), tokenHull (Z t) := by ext x; simp
  rw [heq]
  exact isClosed_iInter fun t => isClosed_iInter fun _ => (isCompact_tokenHull (Z t)).isClosed

/-- The future-hull intersection is compact, being closed inside the initial
hull. Source: arXiv:2305.05465v6, §8.1, `p:noninc` and `t:Idcase11`.
This compactness alone does not assert that the intersection is a polytope. -/
theorem isCompact_future_tokenHull (Z : ℝ → Idx n → EucSpace d) :
    IsCompact {x | ∀ t : ℝ, 0 ≤ t → x ∈ tokenHull (Z t)} :=
  (isCompact_tokenHull (Z 0)).of_isClosed_subset (isClosed_future_tokenHull Z)
    (fun _ hx => hx 0 le_rfl)

/-- A nested finite token hull converges to a nonempty polytope with at most
`n` generators, exactly the intersection of its future hulls.
Source: arXiv:2305.05465v6, §8.1, proof of `t:Idcase11`, first paragraph.
Only the nesting of `p:noninc` is needed; no convergence of tokens is assumed. -/
theorem exists_polytope_limit_of_tokenHull_antitone
    (Z : ℝ → Idx n → EucSpace d) (hn : 0 < n)
    (hmono : ∀ s t : ℝ, 0 ≤ s → s ≤ t → tokenHull (Z t) ⊆ tokenHull (Z s)) :
    ∃ S : Finset (EucSpace d), S.Nonempty ∧ S.card ≤ n ∧
      convexHull ℝ (S : Set (EucSpace d)) =
        {x | ∀ t : ℝ, 0 ≤ t → x ∈ tokenHull (Z t)} ∧
      Tendsto (fun t => Metric.hausdorffDist (tokenHull (Z t))
        (convexHull ℝ (S : Set (EucSpace d)))) atTop (nhds 0) := by
  classical
  have hcompact : IsCompact {X : Idx n → EucSpace d | ∀ i, X i ∈ tokenHull (Z 0)} :=
    isCompact_pi_infinite fun _ => isCompact_tokenHull (Z 0)
  obtain ⟨Y, -, φ, hφ, hlim⟩ := hcompact.tendsto_subseq
    (x := fun k : ℕ => Z (k : ℝ)) (fun k i =>
      hmono 0 k le_rfl (Nat.cast_nonneg k) (mem_tokenHull _ i))
  change Tendsto (fun k => Z (φ k : ℝ)) atTop (nhds Y) at hlim
  have htime : Tendsto (fun k => (φ k : ℝ)) atTop atTop :=
    tendsto_natCast_atTop_atTop.comp hφ.tendsto_atTop
  have hsub : ∀ t : ℝ, 0 ≤ t → tokenHull Y ⊆ tokenHull (Z t) := by
    intro t ht
    apply tokenHull_subset_of_tendsto (fun k => Z (φ k : ℝ)) Y (tokenHull (Z t))
      (isCompact_tokenHull _).isClosed (convex_convexHull ℝ _) hlim
    filter_upwards [htime.eventually (eventually_ge_atTop t)] with k hk
    intro i
    exact hmono t (φ k) ht hk (mem_tokenHull _ i)
  have hnorm : Tendsto (fun k => ‖Z (φ k : ℝ) - Y‖) atTop (nhds 0) := by
    simpa only [sub_self, norm_zero] using
      (hlim.sub (tendsto_const_nhds (x := Y))).norm
  have hinter : tokenHull Y = {x | ∀ t : ℝ, 0 ≤ t → x ∈ tokenHull (Z t)} := by
    ext x
    constructor
    · intro hx t ht
      exact hsub t ht hx
    · intro hx
      have hnear (k : ℕ) : ∃ y ∈ tokenHull Y, ‖x - y‖ ≤ ‖Z (φ k : ℝ) - Y‖ :=
        exists_tokenHull_norm_sub_le _ _ _
          (fun i => norm_le_pi_norm (Z (φ k : ℝ) - Y) i)
          (hx (φ k) (Nat.cast_nonneg _))
      choose y hy hxy using hnear
      have hy_lim : Tendsto y atTop (nhds x) := by
        apply Metric.tendsto_nhds.mpr
        intro ε hε
        filter_upwards [hnorm.eventually_lt_const hε] with k hk
        rw [dist_eq_norm, norm_sub_rev]
        exact (hxy k).trans_lt hk
      exact (isCompact_tokenHull Y).isClosed.mem_of_tendsto hy_lim
        (Eventually.of_forall hy)
  have hHaus : Tendsto (fun t => Metric.hausdorffDist (tokenHull (Z t)) (tokenHull Y))
      atTop (nhds 0) := by
    apply Metric.tendsto_nhds.mpr
    intro ε hε
    obtain ⟨k, hk⟩ := (hnorm.eventually_lt_const (half_pos hε)).exists
    filter_upwards [eventually_ge_atTop (φ k : ℝ)] with t ht
    have ht0 : 0 ≤ t := (Nat.cast_nonneg (φ k)).trans ht
    have hbound := hausdorffDist_le_of_between_tokenHulls (Z (φ k : ℝ)) Y
      (tokenHull (Z t)) (ε / 2) (half_pos hε).le (hsub t ht0)
      (hmono (φ k) t (Nat.cast_nonneg _) ht)
      (fun i => (norm_le_pi_norm (Z (φ k : ℝ) - Y) i).trans hk.le)
    rw [dist_zero_right, Real.norm_of_nonneg Metric.hausdorffDist_nonneg]
    exact hbound.trans_lt (half_lt_self hε)
  let S := Finset.univ.image Y
  have hS : S.Nonempty :=
    ⟨Y ⟨0, hn⟩, Finset.mem_image_of_mem Y (Finset.mem_univ _)⟩
  have hcard : S.card ≤ n := by
    simpa only [Finset.card_univ, Fintype.card_fin] using
      (Finset.card_image_le (s := Finset.univ) (f := Y))
  have hset : (S : Set (EucSpace d)) = Set.range Y := by
    simp only [S, Finset.coe_image, Finset.coe_univ, Set.image_univ]
  have hHull : convexHull ℝ (S : Set (EucSpace d)) = tokenHull Y := by rw [hset]; rfl
  exact ⟨S, hS, hcard, hHull.trans hinter, by simpa only [hHull] using hHaus⟩

/-- Both geometric hypotheses hold for an actual stationary solution with
two tokens. Source: arXiv:2305.05465v6, §8.1, `e:dynIdzi` and `p:noninc`. -/
example (z : EucSpace 1) : (0 : ℕ) < 2 ∧
    IdRescaledDynamics (1 : ParamMatrix 1) (fun _ (_ : Idx 2) => z) ∧
    ∀ s t : ℝ, 0 ≤ s → s ≤ t →
      tokenHull ((fun _ (_ : Idx 2) => z) t) ⊆
        tokenHull ((fun _ (_ : Idx 2) => z) s) :=
  ⟨by norm_num, idRescaledDynamics_const 1 z, fun _ _ _ _ => subset_rfl⟩

/-- **The hull-convergence step of `t:Idcase11`.** For the actual
identity-value dynamics, the limiting polytope is the intersection of all
future token hulls and has at most `n` generating points.
Source: arXiv:2305.05465v6, §8.1, proof of `t:Idcase11`, first paragraph.
The source's positive-definiteness condition is unnecessary for this step. -/
theorem IdRescaledDynamics.exists_tokenHull_polytope_limit
    (A : ParamMatrix d) (Z : ℝ → Idx n → EucSpace d)
    (hZ : IdRescaledDynamics A Z) (hn : 0 < n) :
    ∃ S : Finset (EucSpace d), S.Nonempty ∧ S.card ≤ n ∧
      convexHull ℝ (S : Set (EucSpace d)) =
        {x | ∀ t : ℝ, 0 ≤ t → x ∈ tokenHull (Z t)} ∧
      Tendsto (fun t => Metric.hausdorffDist (tokenHull (Z t))
        (convexHull ℝ (S : Set (EucSpace d)))) atTop (nhds 0) :=
  exists_polytope_limit_of_tokenHull_antitone Z hn
    (fun _ _ _ hst => tokenHull_antitone A Z hZ hst)

/-- All dynamical hypotheses are realized by two stationary unit tokens;
arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
example : IdRescaledDynamics (1 : ParamMatrix 1)
    (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) ∧ (0 : ℕ) < 2 :=
  ⟨idRescaledDynamics_const 1 _, by norm_num⟩

end Transformer.Clusters
