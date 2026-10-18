/-
# Saddle avoidance — orbits that stay near a point, and sets met once by lines

Two facts the local argument combines.

* `stayNear g z r` is closed when `g` is continuous, lies in the ball, and
  contains the tail of every orbit that eventually stays in the ball
  (`iterate_mem_stayNear`).
* A bounded measurable set that meets every line of a fixed direction `e` at
  most once is null (`measure_eq_zero_of_line_meets_once`): its translates by
  `t • e`, `t = 1/(n+1)`, are pairwise disjoint, have its measure, and lie in
  one bounded set.  This is the step that replaces the dimension count of the
  local center-stable manifold in the proof of `thm:main` of arXiv:1602.04915.
-/

import Transformer.Landscape.SaddleAvoidance.Defs
import Mathlib.MeasureTheory.Measure.Lebesgue.EqHaar

open Set Metric MeasureTheory

namespace Transformer.Landscape.SaddleAvoidance

section Orbit

variable {X : Type*} [PseudoMetricSpace X]

theorem stayNear_subset (g : X → X) (z : X) (r : ℝ) : stayNear g z r ⊆ closedBall z r :=
  fun _ hx => hx 0

theorem isClosed_stayNear {g : X → X} (hg : Continuous g) (z : X) (r : ℝ) :
    IsClosed (stayNear g z r) := by
  have : stayNear g z r = ⋂ k : ℕ, g^[k] ⁻¹' closedBall z r := by
    ext x
    simp [stayNear]
  rw [this]
  exact isClosed_iInter fun k => isClosed_closedBall.preimage (hg.iterate k)

/-- An orbit that is in the closed ball from step `m` on enters `stayNear` at
step `m`. -/
theorem iterate_mem_stayNear {g : X → X} {z x : X} {r : ℝ} {m : ℕ}
    (h : ∀ k, m ≤ k → g^[k] x ∈ closedBall z r) : g^[m] x ∈ stayNear g z r := by
  intro k
  rw [← Function.iterate_add_apply]
  exact h (k + m) (Nat.le_add_left m k)

/-- An orbit converging to a point of an open set `U` is in `U` from some step
on, so its tail lies in `stayNear g z r` for any closed ball containing `U`. -/
theorem exists_iterate_mem_stayNear {g : X → X} {x w z : X} {r : ℝ}
    (hx : Filter.Tendsto (fun k => g^[k] x) Filter.atTop (nhds w))
    (hw : w ∈ ball z r) : ∃ m : ℕ, g^[m] x ∈ stayNear g z r := by
  obtain ⟨m, hm⟩ := Filter.eventually_atTop.1 (hx.eventually (isOpen_ball.mem_nhds hw))
  exact ⟨m, iterate_mem_stayNear fun k hk => ball_subset_closedBall (hm k hk)⟩

/-- The hypotheses of the three lemmas above hold for the halving map of the
line: it is continuous, and the orbit `2⁻ᵏ` of `1` stays in `B̄(0, 1)` from step
`0` on and converges to `0 ∈ B(0, 1)`. -/
example : Continuous (fun x : ℝ => (1 / 2 : ℝ) • x) ∧
    (∀ k, 0 ≤ k → (fun x : ℝ => (1 / 2 : ℝ) • x)^[k] 1 ∈ closedBall (0 : ℝ) 1) ∧
    Filter.Tendsto (fun k => (fun x : ℝ => (1 / 2 : ℝ) • x)^[k] 1) Filter.atTop (nhds 0) ∧
    (0 : ℝ) ∈ ball (0 : ℝ) 1 := by
  simp only [smul_iterate]
  simp only [smul_eq_mul, mul_one]
  refine ⟨continuous_const.mul continuous_id, fun k _ => ?_,
    tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num), mem_ball_self one_pos⟩
  rw [mem_closedBall, Real.dist_eq, sub_zero, abs_of_nonneg (by positivity)]
  exact pow_le_one₀ (by norm_num) (by norm_num)

end Orbit

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]
  [MeasurableSpace E] [BorelSpace E]

/-- **A bounded measurable set met at most once by each line of direction `e`
is null.**  The translates `W + tₙ e`, `tₙ = 1/(n+1)`, are pairwise disjoint
(two of them meeting would put two points of one line in `W`), each has the
measure of `W`, and all lie in the ball of radius `R + ‖e‖` around `0` when
`W` lies in the ball of radius `R`; infinitely many disjoint sets of equal
measure in a set of finite measure are null. -/
theorem measure_eq_zero_of_line_meets_once (μ : Measure E) [μ.IsAddHaarMeasure] {W : Set E}
    (hW : MeasurableSet W) (hb : Bornology.IsBounded W) {e : E}
    (he : ∀ x ∈ W, ∀ t : ℝ, x + t • e ∈ W → t = 0) : μ W = 0 := by
  by_contra h0
  obtain ⟨R, hR⟩ := hb.subset_closedBall 0
  set t : ℕ → ℝ := fun n => 1 / ((n : ℝ) + 1) with ht
  have ht0 : ∀ n, 0 ≤ t n := fun n => by positivity
  have ht1 : ∀ n, t n ≤ 1 := fun n => by
    rw [ht, div_le_one (by positivity)]
    linarith [(Nat.cast_nonneg n : (0 : ℝ) ≤ n)]
  have htinj : Function.Injective t := by
    intro n m hnm
    simp only [ht, one_div, inv_inj, add_left_inj, Nat.cast_inj] at hnm
    exact hnm
  set s : ℕ → Set E := fun n => (fun y => y + (-t n) • e) ⁻¹' W with hs
  have hs_meas : ∀ n, MeasurableSet (s n) := fun n => hW.preimage (measurable_add_const _)
  have hs_eq : ∀ n, μ (s n) = μ W := fun n => measure_preimage_add_right μ _ W
  have hdisj : Pairwise (Function.onFun Disjoint s) := by
    intro n m hnm
    rw [Function.onFun, Set.disjoint_left]
    intro y hyn hym
    simp only [hs, mem_preimage] at hyn hym
    have hx : y + (-t n) • e + (t n - t m) • e = y + (-t m) • e := by
      rw [neg_smul, neg_smul, sub_smul]
      abel
    have h := he _ hyn (t n - t m) (by rw [hx]; exact hym)
    exact hnm (htinj (sub_eq_zero.1 h))
  have hsub : (⋃ n, s n) ⊆ closedBall 0 (R + ‖e‖) := by
    intro y hy
    obtain ⟨n, hyn⟩ := mem_iUnion.1 hy
    have hyR := mem_closedBall_zero_iff.1 (hR hyn)
    rw [mem_closedBall_zero_iff]
    have hte : ‖t n • e‖ ≤ ‖e‖ := by
      rw [norm_smul, Real.norm_of_nonneg (ht0 n)]
      exact mul_le_of_le_one_left (norm_nonneg e) (ht1 n)
    calc ‖y‖ = ‖(y + (-t n) • e) + t n • e‖ := by rw [neg_smul]; abel_nf
      _ ≤ ‖y + (-t n) • e‖ + ‖t n • e‖ := norm_add_le _ _
      _ ≤ R + ‖e‖ := add_le_add hyR hte
  have htop : ∑' n, μ (s n) = ⊤ := by
    simp_rw [hs_eq]
    exact ENNReal.tsum_const_eq_top_of_ne_zero h0
  rw [← measure_iUnion hdisj hs_meas] at htop
  have := (measure_mono hsub).trans_lt (measure_closedBall_lt_top (μ := μ))
  rw [htop] at this
  exact lt_irrefl _ this

/-- The hypotheses of `measure_eq_zero_of_line_meets_once` hold for a point of
the line and the direction `1`. -/
example : MeasurableSet ({0} : Set ℝ) ∧ Bornology.IsBounded ({0} : Set ℝ) ∧
    ∀ x ∈ ({0} : Set ℝ), ∀ t : ℝ, x + t • (1 : ℝ) ∈ ({0} : Set ℝ) → t = 0 := by
  refine ⟨measurableSet_singleton 0, Bornology.isBounded_singleton, fun x hx t ht => ?_⟩
  simp_all

end Transformer.Landscape.SaddleAvoidance
