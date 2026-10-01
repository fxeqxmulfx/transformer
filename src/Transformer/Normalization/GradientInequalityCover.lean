/-
# Local gradient inequalities and the fixed-energy open cover

Appendix D.1 of arXiv:2510.22026v2: regular points require only continuity;
points away from the limiting energy are excluded from the late trajectory.
-/

import Transformer.Normalization.ModulatedCritical
import Mathlib.Analysis.SpecialFunctions.Pow.Continuity

namespace Transformer.Normalization

/-- At a noncritical point, continuity alone yields the local gradient
inequality of Appendix D.1, `lem: loj`, in arXiv:2510.22026v2 with
exponent `1/2`. This isolates the difficult analytic input at critical points. -/
theorem local_gradient_inequality_of_noncritical {N : ℕ} (E : EucSpace N → ℝ)
    (z : EucSpace N) (hE : ContinuousAt E z)
    (hgrad : ContinuousAt (gradient E) z) (hregular : gradient E z ≠ 0) :
    ∃ alpha k : ℝ, ∃ V : Set (EucSpace N),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
        ∀ y ∈ V, |E y - E z| ^ alpha ≤ k * ‖gradient E y‖ := by
  have hg : 0 < ‖gradient E z‖ := norm_pos_iff.mpr hregular
  obtain ⟨r, hr, hballE⟩ := Metric.continuousAt_iff.1 hE 1 zero_lt_one
  obtain ⟨s, hs, hballG⟩ := Metric.continuousAt_iff.1 hgrad.norm
    (‖gradient E z‖ / 2) (half_pos hg)
  refine ⟨1 / 2, 2 / ‖gradient E z‖, Metric.ball z (min r s), by norm_num,
    by norm_num, div_pos (by norm_num) hg, Metric.isOpen_ball,
    Metric.mem_ball_self (lt_min hr hs), ?_⟩
  intro y hy
  have he := hballE (hy.trans_le (min_le_left r s))
  have hn := hballG (hy.trans_le (min_le_right r s))
  rw [Real.dist_eq] at he hn
  have hlow : ‖gradient E z‖ / 2 ≤ ‖gradient E y‖ := by
    have hdiff := neg_le_abs (‖gradient E y‖ - ‖gradient E z‖)
    linarith
  calc
    |E y - E z| ^ (1 / 2 : ℝ) ≤ 1 :=
      Real.rpow_le_one (abs_nonneg _) he.le (by norm_num)
    _ ≤ 2 / ‖gradient E z‖ * ‖gradient E y‖ := by
      rw [div_mul_eq_mul_div]
      apply (le_div_iff₀ hg).2
      nlinarith

/-- The regular-point hypotheses hold for a nonzero linear energy,
whose gradient is a constant vector; Appendix D.1 of arXiv:2510.22026v2. -/
example (v : EucSpace 1) (hv : v ≠ 0) :
    ContinuousAt (fun y => inner (𝕜 := ℝ) v y) 0 ∧
    ContinuousAt (gradient (fun y => inner (𝕜 := ℝ) v y)) 0 ∧
    gradient (fun y => inner (𝕜 := ℝ) v y) 0 ≠ 0 := by
  have hg : gradient (fun y => inner (𝕜 := ℝ) v y) = fun _ => v := by
    apply gradient_eq
    intro z
    rw [hasGradientAt_iff_hasFDerivAt]
    exact (innerSL ℝ v).hasFDerivAt
  rw [hg]
  exact ⟨(innerSL ℝ v).continuous.continuousAt, continuousAt_const, hv⟩

/-- The local gradient inequality in Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2 extends to an open cover of a compact trajectory's
ambient set, conditional on closeness to the limiting energy. This
auxiliary lemma keeps the local analytic input explicit. -/
theorem local_gradient_inequality_cover {N : ℕ} (E : EucSpace N → ℝ)
    (K : Set (EucSpace N)) (L : ℝ) (z : EucSpace N)
    (hE : ContinuousWithinAt E K z)
    (hlocal : E z = L → ∃ alpha k : ℝ, ∃ V : Set (EucSpace N),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
        ∀ y ∈ V, |E y - E z| ^ alpha ≤ k * ‖gradient E y‖) :
    ∃ alpha k delta : ℝ, ∃ V : Set (EucSpace N),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ 0 < delta ∧ IsOpen V ∧ z ∈ V ∧
        ∀ y ∈ K, y ∈ V → |E y - L| < delta →
          |E y - L| ^ alpha ≤ k * ‖gradient E y‖ := by
  by_cases hlevel : E z = L
  · obtain ⟨alpha, k, V, ha0, ha1, hk, hV, hzV, hineq⟩ := hlocal hlevel
    exact ⟨alpha, k, 1, V, ha0, ha1, hk, zero_lt_one, hV, hzV,
      fun y _ hy _ => by simpa only [hlevel] using hineq y hy⟩
  · let delta := |E z - L| / 2
    have hd : 0 < delta := half_pos (abs_pos.mpr (sub_ne_zero.mpr hlevel))
    obtain ⟨r, hr, hball⟩ := Metric.continuousWithinAt_iff.1 hE delta hd
    refine ⟨1 / 2, 1, delta, Metric.ball z r, by norm_num, by norm_num,
      zero_lt_one, hd, Metric.isOpen_ball, Metric.mem_ball_self hr, ?_⟩
    intro y hy hyV hyE
    have hnear := hball hy hyV
    rw [Real.dist_eq, abs_sub_comm] at hnear
    have htri := abs_sub_le (E z) (E y) L
    dsimp [delta] at hnear hyE
    exfalso
    linarith

/-- The open-cover hypotheses hold for constant zero energy and the
singleton set, with exponent `1/2` and the whole-space neighborhood;
Appendix D.1 of arXiv:2510.22026v2. -/
example : (0 : EucSpace 1) ∈ ({0} : Set (EucSpace 1)) ∧
    ContinuousWithinAt (fun _ : EucSpace 1 => (0 : ℝ)) {0} 0 ∧
    ((0 : ℝ) = 0 → ∃ alpha k : ℝ, ∃ V : Set (EucSpace 1),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ (0 : EucSpace 1) ∈ V ∧
        ∀ y ∈ V, |(0 : ℝ) - 0| ^ alpha ≤ k * ‖gradient (fun _ : EucSpace 1 => (0 : ℝ)) y‖) := by
  refine ⟨Set.mem_singleton 0, continuousWithinAt_const, fun _ => ?_⟩
  refine ⟨1 / 2, 1, Set.univ, by norm_num, by norm_num, zero_lt_one,
    isOpen_univ, Set.mem_univ _, fun y _ => ?_⟩
  norm_num

end Transformer.Normalization
