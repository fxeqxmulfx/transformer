/-
# Continuity of the finite token hull

The geometric input in arXiv:2305.05465v6, §8.1, the opening paragraph of
the proof of `t:Idcase11`. Moving each of finitely many tokens by at most
`ε` moves its entire convex hull by at most `ε` in Hausdorff distance.
Convex combinations retain the same coefficients in the comparison hull.

These estimates prepare the compactness proof that a nested family of
token hulls has a polytope as its limit. They also retain the genuine
set inclusions needed to identify that limit; the distance alone cannot
distinguish an empty set under the real-valued Hausdorff convention.
-/

import Transformer.Clusters.Section8_Polytope

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d n : ℕ}

/-- A point in the first hull has a nearby point in the second hull when
each corresponding token is nearby. Source: arXiv:2305.05465v6, §8.1,
proof of `t:Idcase11`, compactness of the shrinking hull. -/
theorem exists_tokenHull_norm_sub_le (X Y : Idx n → EucSpace d) (ε : ℝ)
    (hXY : ∀ i, ‖X i - Y i‖ ≤ ε) {x : EucSpace d} (hx : x ∈ tokenHull X) :
    ∃ y ∈ tokenHull Y, ‖x - y‖ ≤ ε := by
  let T : Set (EucSpace d) := {x | ∃ y ∈ tokenHull Y, ‖x - y‖ ≤ ε}
  have hconv : Convex ℝ T := by
    rintro x ⟨u, hu, hxu⟩ y ⟨v, hv, hyv⟩ a b ha hb hab
    refine ⟨a • u + b • v, (convex_convexHull ℝ _ hu hv ha hb hab), ?_⟩
    calc ‖(a • x + b • y) - (a • u + b • v)‖
        = ‖a • (x - u) + b • (y - v)‖ := by congr 1; simp only [smul_sub]; abel
      _ ≤ ‖a • (x - u)‖ + ‖b • (y - v)‖ := norm_add_le _ _
      _ = a * ‖x - u‖ + b * ‖y - v‖ := by
        rw [norm_smul, norm_smul, Real.norm_of_nonneg ha, Real.norm_of_nonneg hb]
      _ ≤ a * ε + b * ε := add_le_add
        (mul_le_mul_of_nonneg_left hxu ha) (mul_le_mul_of_nonneg_left hyv hb)
      _ = ε := by rw [← add_mul, hab, one_mul]
  exact convexHull_min (fun z hz => by
    obtain ⟨i, rfl⟩ := hz
    exact ⟨Y i, mem_tokenHull Y i, hXY i⟩) hconv hx

/-- Every proximity hypothesis holds for two coincident stationary tokens;
arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
example (z : EucSpace 1) :
    (∀ i : Idx 2, ‖(fun _ : Idx 2 => z) i - (fun _ : Idx 2 => z) i‖ ≤ 0) ∧
    z ∈ tokenHull (fun _ : Idx 2 => z) :=
  ⟨fun _ => by simp, mem_tokenHull _ 0⟩

/-- The finite hull inherits a uniform token-displacement bound;
arXiv:2305.05465v6, §8.1, proof of `t:Idcase11`. -/
theorem hausdorffDist_tokenHull_le (X Y : Idx n → EucSpace d) (ε : ℝ)
    (hε : 0 ≤ ε) (hXY : ∀ i, ‖X i - Y i‖ ≤ ε) :
    Metric.hausdorffDist (tokenHull X) (tokenHull Y) ≤ ε := by
  apply Metric.hausdorffDist_le_of_mem_dist hε
  · intro x hx
    obtain ⟨y, hy, hxy⟩ := exists_tokenHull_norm_sub_le X Y ε hXY hx
    exact ⟨y, hy, by simpa only [dist_eq_norm] using hxy⟩
  · intro y hy
    have hYX : ∀ i, ‖Y i - X i‖ ≤ ε := fun i => by rw [norm_sub_rev]; exact hXY i
    obtain ⟨x, hx, hyx⟩ := exists_tokenHull_norm_sub_le Y X ε hYX hy
    exact ⟨x, hx, by simpa only [dist_eq_norm] using hyx⟩

/-- Both bound hypotheses hold at zero displacement;
arXiv:2305.05465v6, §8.1, proof of `t:Idcase11`. -/
example (z : EucSpace 1) : (0 : ℝ) ≤ 0 ∧
    ∀ i : Idx 2, ‖(fun _ : Idx 2 => z) i - (fun _ : Idx 2 => z) i‖ ≤ 0 :=
  ⟨le_rfl, fun _ => by simp⟩

/-- The Hausdorff displacement is at most the norm of the tuple displacement;
arXiv:2305.05465v6, §8.1, compactness step of `t:Idcase11`.
The tuple has the usual finite-product supremum norm. -/
theorem hausdorffDist_tokenHull_le_norm (X Y : Idx n → EucSpace d) :
    Metric.hausdorffDist (tokenHull X) (tokenHull Y) ≤ ‖X - Y‖ :=
  hausdorffDist_tokenHull_le X Y _ (norm_nonneg _) (fun i => norm_le_pi_norm (X - Y) i)

/-- A common token bound also bounds every convex combination;
arXiv:2305.05465v6, §8.1, `c:theziunifbdd`. -/
theorem tokenHull_subset_closedBall (X : Idx n → EucSpace d) (R : ℝ)
    (hR : ∀ i, ‖X i‖ ≤ R) : tokenHull X ⊆ Metric.closedBall 0 R := by
  apply convexHull_min ?_ (convex_closedBall _ _)
  rintro _ ⟨i, rfl⟩
  exact mem_closedBall_zero_iff.mpr (hR i)

/-- The common bound holds for the stationary zero configuration;
arXiv:2305.05465v6, §8.1, `c:theziunifbdd`. -/
example : ∀ i : Idx 2, ‖(fun _ : Idx 2 => (0 : EucSpace 1)) i‖ ≤ 1 := by simp

/-- Convergence of a finite token tuple implies convergence of its hull;
arXiv:2305.05465v6, §8.1, proof of `t:Idcase11`. -/
theorem tendsto_hausdorffDist_tokenHull {ι : Type*} {l : Filter ι}
    (X : ι → Idx n → EucSpace d) (Y : Idx n → EucSpace d)
    (hX : Tendsto X l (nhds Y)) :
    Tendsto (fun k => Metric.hausdorffDist (tokenHull (X k)) (tokenHull Y)) l (nhds 0) := by
  apply squeeze_zero' (Eventually.of_forall fun _ => Metric.hausdorffDist_nonneg)
    (Eventually.of_forall fun k => hausdorffDist_tokenHull_le_norm (X k) Y)
  simpa only [sub_self, norm_zero] using
    (hX.sub (tendsto_const_nhds (x := Y))).norm

/-- The convergence hypothesis holds for the actual stationary configuration;
arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
example (z : EucSpace 1) :
    Tendsto (fun _ : ℕ => fun _ : Idx 2 => z) atTop (nhds (fun _ : Idx 2 => z)) :=
  tendsto_const_nhds

/-- A tuple limit remains in a closed convex set that eventually contains
every token; arXiv:2305.05465v6, §8.1, proof of `t:Idcase11`.
Nontriviality of the filter is needed to pass membership to the limit. -/
theorem tokenHull_subset_of_tendsto {ι : Type*} {l : Filter ι} [l.NeBot]
    (X : ι → Idx n → EucSpace d) (Y : Idx n → EucSpace d)
    (K : Set (EucSpace d)) (hclosed : IsClosed K) (hconv : Convex ℝ K)
    (hX : Tendsto X l (nhds Y)) (hmem : ∀ᶠ k in l, ∀ i, X k i ∈ K) :
    tokenHull Y ⊆ K := by
  apply convexHull_min ?_ hconv
  rintro _ ⟨i, rfl⟩
  apply hclosed.mem_of_tendsto (((continuous_apply i).tendsto Y).comp hX)
  exact hmem.mono fun k hk => hk i

/-- All closed-set, convexity, convergence and membership hypotheses are
realized by the stationary tuple and its singleton hull;
arXiv:2305.05465v6, §8.1, proof of `t:Idcase11`. -/
example (z : EucSpace 1) : IsClosed ({z} : Set (EucSpace 1)) ∧
    Convex ℝ ({z} : Set (EucSpace 1)) ∧
    Tendsto (fun _ : ℕ => fun _ : Idx 2 => z) atTop (nhds (fun _ : Idx 2 => z)) ∧
    ∀ᶠ k : ℕ in atTop, ∀ i : Idx 2, (fun _ : ℕ => fun _ : Idx 2 => z) k i ∈
      ({z} : Set (EucSpace 1)) :=
  ⟨isClosed_singleton, convex_singleton _, tendsto_const_nhds,
    Eventually.of_forall fun _ _ => Set.mem_singleton z⟩

/-- A set between the comparison hull and the original hull is also close
to the comparison hull. Source: arXiv:2305.05465v6, §8.1,
the nested-hull compactness step of `t:Idcase11`. -/
theorem hausdorffDist_le_of_between_tokenHulls (X Y : Idx n → EucSpace d)
    (K : Set (EucSpace d)) (ε : ℝ) (hε : 0 ≤ ε)
    (hKY : tokenHull Y ⊆ K) (hKX : K ⊆ tokenHull X)
    (hXY : ∀ i, ‖X i - Y i‖ ≤ ε) :
    Metric.hausdorffDist K (tokenHull Y) ≤ ε := by
  apply Metric.hausdorffDist_le_of_mem_dist hε
  · intro x hx
    obtain ⟨y, hy, hxy⟩ := exists_tokenHull_norm_sub_le X Y ε hXY (hKX hx)
    exact ⟨y, hy, by simpa only [dist_eq_norm] using hxy⟩
  · intro y hy
    exact ⟨y, hKY hy, by simpa using hε⟩

/-- Every intermediate-set bound input holds for the actual stationary hull;
arXiv:2305.05465v6, §8.1, proof of `t:Idcase11`. -/
example (z : EucSpace 1) : (0 : ℝ) ≤ 0 ∧
    tokenHull (fun _ : Idx 2 => z) ⊆ tokenHull (fun _ : Idx 2 => z) ∧
    tokenHull (fun _ : Idx 2 => z) ⊆ tokenHull (fun _ : Idx 2 => z) ∧
    ∀ i : Idx 2, ‖(fun _ : Idx 2 => z) i - (fun _ : Idx 2 => z) i‖ ≤ 0 :=
  ⟨le_rfl, subset_rfl, subset_rfl, fun _ => by simp⟩

end Transformer.Clusters
