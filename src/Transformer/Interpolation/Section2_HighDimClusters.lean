module

public import Transformer.Interpolation.Section2_RoutedCollapses
public import Transformer.Interpolation.Section2_SmallBalls
public import Transformer.Interpolation.Section2_BallGrouping
public import Transformer.Interpolation.Section2_ClusterCoupling
public import Transformer.Interpolation.AtomlessSphere

/-!
# Clustering one law in dimension at least three

ArXiv:2411.04551v3, §2.2, proof of `prop: compression`, Steps 1–3, for
one law: its mass in an open region closed under geodesic hulls is
gathered near prescribed points of the region with prescribed weights by
phases local in the region. On a sphere of dimension at least two
(`d ≥ 3`) finitely many small disjoint balls carry almost all the mass
and are grouped into the weights; the balls of each group are collapsed
and routed near its point. The masses near the points then bound the
distance to the atomic target.
-/

@[expose] public section

open scoped ENNReal NNReal
open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- **Clustering one law, `d ≥ 3`.** In dimension `d ≥ 3`, a law without
atoms carried by an open region closed under geodesic hulls, points of
the region, and weights summing to one: for every `ε > 0`, phases local
in the region bring the law within `ε` of the atomic law of the points
with these weights in `W_2`. Source: arXiv:2411.04551v3, §2.2, proof of
`prop: compression`, Steps 1–3, for one law, with the region in place of
the geodesic convex hull of its support. -/
theorem exists_cluster_phases_of_three_le (hd : 3 ≤ d) (ν : Perspective.ProbSphere d)
    (hatom : ∀ y : SSphere d, (ν : Measure (SSphere d)) {y} = 0) (U : Set (SSphere d))
    (hU : IsOpen U) (hconv : ∀ A ⊆ U, convG d A ⊆ U) (hνU : (ν : Measure (SSphere d)) U = 1)
    {M : ℕ} (x : Fin M → SSphere d) (hx : ∀ k, x k ∈ U) (α : Fin M → ℝ≥0)
    (hα : ∑ k, α k = 1) (ε : ℝ) (hε : 0 < ε) :
    ∃ L : List (AutonomousPhase d), AllLocalIn L U ∧
      W2 d (phaseSequenceLaw L ν : Measure (SSphere d)) (∑ k, α k • Measure.dirac (x k)) ≤
        ε := by
  classical
  obtain ⟨M, rfl⟩ : ∃ M', M = M' + 1 := Nat.exists_eq_succ_of_ne_zero fun h => by
    subst h
    simp at hα
  set ε₁ := min ε 1
  have hε₁ : 0 < ε₁ := lt_min hε one_pos
  have hε₁1 : ε₁ ≤ 1 := min_le_right _ _
  set e := ε₁ ^ 2 / (16 * (M + 1)) with he_def
  have he : 0 < e := by positivity
  have he1 : e ≤ 1 := div_le_one_of_le₀ (by nlinarith [(M.cast_nonneg : (0 : ℝ) ≤ M)])
    (by positivity)
  obtain ⟨P, z, ρ, hρ, hBU, hdisj, hsmall, hcover⟩ := exists_small_disjoint_balls
    (ν : Measure (SSphere d)) hatom U hU 2 two_pos (ENNReal.ofReal (e / 2))
    (ENNReal.ofReal_pos.mpr (half_pos he))
  have hsum : ∑ k, (α k : ℝ≥0∞) ≤
      ∑ p ∈ Finset.univ, (ν : Measure (SSphere d)) (Metric.ball (z p) (ρ p)) +
        ENNReal.ofReal (e / 2) := by
    rw [← ENNReal.ofNNReal_finsetSum, hα, ENNReal.coe_one, ← hνU]
    exact hcover
  obtain ⟨g, hg⟩ := exists_grouping M Finset.univ
    (fun p => (ν : Measure (SSphere d)) (Metric.ball (z p) (ρ p))) (ENNReal.ofReal (e / 2))
    (fun p _ => hsmall p) α hsum
  obtain ⟨L, c, r, hL, hr, hball, hdisjc, hmass⟩ := exists_routed_collapses hd ν U hU hconv z ρ
    hρ hBU hdisj (fun p => x (g p)) (fun p => hx (g p)) (ε₁ / 2) (half_pos hε₁) (e / 2)
    (half_pos he) (by linarith)
  refine ⟨L, hL, ?_⟩
  set ν' := phaseSequenceLaw L ν
  let E : Fin (M + 1) → Set (SSphere d) := fun k =>
    ⋃ p ∈ Finset.univ.filter fun p => g p = k, Metric.ball (c p) r
  have hdisjE : Pairwise (Function.onFun Disjoint E) := fun k k' hkk' => by
    refine disjoint_iUnion₂_left.mpr fun p hp => disjoint_iUnion₂_right.mpr fun q hq =>
      hdisjc fun hpq => hkk' ?_
    rw [← (Finset.mem_filter.mp hp).2, ← (Finset.mem_filter.mp hq).2, hpq]
  have hEs : ∀ k, ∀ y ∈ E k, dist y (x k) ≤ ε₁ / 2 := fun k y hy => by
    simp only [E, mem_iUnion, Finset.mem_filter, Finset.mem_univ, true_and, exists_prop] at hy
    obtain ⟨p, rfl, hy⟩ := hy
    exact (Metric.mem_ball.mp (hball p hy)).le
  have hmassE : ∀ k, (α k : ℝ≥0∞) ≤ (ν' : Measure (SSphere d)) (E k) + ENNReal.ofReal e :=
      fun k => by
    set F := Finset.univ.filter fun p => g p = k
    have hS : ∑ p ∈ F, (ν : Measure (SSphere d)) (Metric.ball (z p) (ρ p)) ≤ 1 := by
      rw [← measure_biUnion_finset (fun p _ q _ hpq => hdisj hpq)
        fun p _ => Metric.isOpen_ball.measurableSet]
      exact prob_le_one
    have hE : (ν' : Measure (SSphere d)) (E k) =
        ∑ p ∈ F, (ν' : Measure (SSphere d)) (Metric.ball (c p) r) :=
      measure_biUnion_finset (fun p _ q _ hpq => hdisjc hpq)
        fun p _ => Metric.isOpen_ball.measurableSet
    have h1 : ENNReal.ofReal (1 - e / 2) + ENNReal.ofReal (e / 2) = 1 := by
      rw [← ENNReal.ofReal_add (by linarith) (by linarith), sub_add_cancel, ENNReal.ofReal_one]
    calc (α k : ℝ≥0∞)
        ≤ ∑ p ∈ F, (ν : Measure (SSphere d)) (Metric.ball (z p) (ρ p)) +
            ENNReal.ofReal (e / 2) := hg k
      _ = ENNReal.ofReal (1 - e / 2) *
            ∑ p ∈ F, (ν : Measure (SSphere d)) (Metric.ball (z p) (ρ p)) +
          ENNReal.ofReal (e / 2) *
            ∑ p ∈ F, (ν : Measure (SSphere d)) (Metric.ball (z p) (ρ p)) +
          ENNReal.ofReal (e / 2) := by rw [← add_mul, h1, one_mul]
      _ ≤ (ν' : Measure (SSphere d)) (E k) + ENNReal.ofReal (e / 2) * 1 +
          ENNReal.ofReal (e / 2) := by
          refine add_le_add (add_le_add ?_ (mul_le_mul_right hS _)) le_rfl
          rw [hE, Finset.mul_sum]
          exact Finset.sum_le_sum fun p _ => hmass p
      _ = (ν' : Measure (SSphere d)) (E k) + ENNReal.ofReal e := by
          rw [mul_one, add_assoc, ← ENNReal.ofReal_add (by linarith) (by linarith), add_halves]
  refine (W2_le_of_cluster_masses (ν' : Measure (SSphere d)) x α hα E
    (fun k => (isOpen_biUnion fun p _ => Metric.isOpen_ball).measurableSet) hdisjE (ε₁ / 2) e
    he.le hEs hmassE).trans ?_
  have h4 : 4 * ((M + 1 : ℕ) : ℝ) * e = ε₁ ^ 2 / 4 := by
    rw [he_def]
    push_cast
    field_simp
    ring
  rw [Real.sqrt_le_iff, h4]
  refine ⟨hε.le, ?_⟩
  have hε₁ε : ε₁ ≤ ε := min_le_left _ _
  nlinarith

/-- The hypotheses of `exists_cluster_phases_of_three_le` are satisfiable:
the uniform law on the sphere of `ℝ³`, the whole sphere, and one point
of weight one. -/
example : ∃ ν : Perspective.ProbSphere 3,
    (∀ y : SSphere 3, (ν : Measure (SSphere 3)) {y} = 0) ∧
    (ν : Measure (SSphere 3)) univ = 1 ∧ IsOpen (univ : Set (SSphere 3)) ∧
    (∀ A ⊆ (univ : Set (SSphere 3)), convG 3 A ⊆ univ) ∧ ∑ _k : Fin 1, (1 : ℝ≥0) = 1 := by
  obtain ⟨ν, hν, -⟩ := exists_diffuse_probSphere 3 (by norm_num) isClosed_univ (by
    have := MeanField.isProbabilityMeasure_uniformLaw 3 (by norm_num)
    rw [measure_univ]
    exact one_pos)
  exact ⟨ν, hν, measure_univ, isOpen_univ, fun _ _ => subset_univ _, by simp⟩

end Transformer.Interpolation
