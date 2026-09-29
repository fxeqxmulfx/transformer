/-
# Convex metric training with a certified global optimum

Extension of arXiv:2211.11052v1, §3.2: convex loss of a linear feature lift,
with a quadratic regularizer instead of the paper's group regularizer.
Each calibration example contrasts an equal code with a one-bit mismatch.
No key-value dictionary is stored in the learned metric. Calibration is
additional matching supervision; it is not the training protocol of the
MQAR experiments in arXiv:2312.04927v1, Appendix `app:synthetic`.
-/

import Transformer.ConvexRecall.Basic

open scoped BigOperators

noncomputable section

namespace Transformer.ConvexRecall

/-- A hinge loss for a linear query-key cost margin. -/
def hinge (margin : ℝ) : ℝ := max 0 (1 - margin)

/-- Unit coordinate margins, with quadratic metric regularization. -/
def calibrationObjective {b : ℕ} (w : MetricWeights b) : ℝ :=
  ∑ r, (hinge (w r) + (w r) ^ 2 / 4)

/-- The unit metric, which will be derived as a trained optimum. -/
def unitMetric (b : ℕ) : MetricWeights b := fun _ => 1

/-- Each coordinate loss is minimized uniquely at one. This algebraic
certificate proves a global optimum of the new training objective; it is
not an optimization claim from arXiv:2211.11052v1, §3.2. -/
theorem calibration_coordinate_certificate (t : ℝ) :
    hinge t + t ^ 2 / 4 =
      1 / 4 + (t - 1) ^ 2 / 4 + |t - 1| / 2 := by
  by_cases h : t ≤ 1
  · rw [hinge, max_eq_right (by linarith), abs_of_nonpos (by linarith)]
    ring
  · rw [hinge, max_eq_left (by linarith), abs_of_nonneg (by linarith)]
    ring

/-- The calibration loss bounds the squared distance to the unit metric.
Extension of arXiv:2211.11052v1, §3.2, using fixed mismatch features. -/
theorem calibration_error_bound {b : ℕ} (w : MetricWeights b) :
    calibrationObjective (unitMetric b) + (∑ r, (w r - 1) ^ 2) / 4 ≤
      calibrationObjective w := by
  have h : ∀ r, (1 / 4 : ℝ) + (w r - 1) ^ 2 / 4 ≤
      hinge (w r) + (w r) ^ 2 / 4 := by
    intro r
    rw [calibration_coordinate_certificate]
    have habs := abs_nonneg (w r - 1)
    linarith
  have hs := Finset.sum_le_sum (fun r (_ : r ∈ Finset.univ) => h r)
  simpa [calibrationObjective, unitMetric, hinge,
    Finset.sum_add_distrib, Finset.sum_div] using hs

/-- Training attains its global optimum at the unit metric. Extension of
arXiv:2211.11052v1, §3.2, not the original attention weight-decay problem. -/
theorem unitMetric_minimizes {b : ℕ} (w : MetricWeights b) :
    calibrationObjective (unitMetric b) ≤ calibrationObjective w := by
  have hs : 0 ≤ ∑ r, (w r - 1) ^ 2 :=
    Finset.sum_nonneg fun _ _ => sq_nonneg _
  have h := calibration_error_bound w
  linarith

/-- Every global minimizer is the unit metric. The conclusion derives the
metric margin required for recall, instead of carrying that margin as an
unproved premise. Extension of arXiv:2211.11052v1, §3.2. -/
theorem calibration_minimizer_unique {b : ℕ} (w : MetricWeights b)
    (hw : ∀ z : MetricWeights b,
      calibrationObjective w ≤ calibrationObjective z) :
    w = unitMetric b := by
  have h := calibration_error_bound w
  have hmin := hw (unitMetric b)
  have hn : ∀ r : Fin b, 0 ≤ (w r - 1) ^ 2 := fun _ => sq_nonneg _
  have hz : (∑ r, (w r - 1) ^ 2) = 0 := by
    have hsum := Finset.sum_nonneg (fun r (_ : r ∈ Finset.univ) => hn r)
    linarith
  have hall := (Finset.sum_eq_zero_iff_of_nonneg
    (fun r (_ : r ∈ Finset.univ) => hn r)).mp hz
  funext r
  have hr := hall r (Finset.mem_univ r)
  change w r = 1
  nlinarith

/-- The optimum hypothesis is satisfiable for every bit width, including
the 13-bit vocabulary encoding. Source context: arXiv:2312.04927v1,
Appendix `app:synthetic`, vocabulary size 8192. -/
example : ∀ z : MetricWeights 13,
    calibrationObjective (unitMetric 13) ≤ calibrationObjective z :=
  unitMetric_minimizes

/-- The scalar hinge is convex. This is the convex-loss requirement used
by the extension of arXiv:2211.11052v1, §3.2. -/
theorem hinge_convex : ConvexOn ℝ Set.univ hinge := by
  refine ⟨convex_univ, ?_⟩
  intro x _ y _ a t ha ht hat
  simp only [hinge, smul_eq_mul]
  apply max_le
  · exact add_nonneg (mul_nonneg ha (le_max_left _ _))
      (mul_nonneg ht (le_max_left _ _))
  · have hx := mul_le_mul_of_nonneg_left (le_max_right (0 : ℝ) (1 - x)) ha
    have hy := mul_le_mul_of_nonneg_left (le_max_right (0 : ℝ) (1 - y)) ht
    nlinarith

/-- The full training objective is convex jointly in every metric
coordinate. The binary code map is fixed; the query-key metric is trained.
Extension of arXiv:2211.11052v1, §3.2. -/
theorem calibrationObjective_convex {b : ℕ} :
    ConvexOn ℝ Set.univ (@calibrationObjective b) := by
  have hsq : ConvexOn ℝ Set.univ (fun x : ℝ => x ^ 2) :=
    (by decide : Even (2 : ℕ)).convexOn_pow
  refine ⟨convex_univ, ?_⟩
  intro w _ z _ a t ha ht hat
  simp only [calibrationObjective, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro r _
  have hh := hinge_convex.2 (Set.mem_univ (w r)) (Set.mem_univ (z r)) ha ht hat
  have hs := hsq.2 (Set.mem_univ (w r)) (Set.mem_univ (z r)) ha ht hat
  simp only [smul_eq_mul] at hh hs
  nlinarith

/-- A zero code and a code differing in just one coordinate are actual
calibration examples. The margin in their learned cost is exactly `w r`.
Extension of arXiv:2211.11052v1, §3.2; binary encoding from
arXiv:2312.04927v1, §4. -/
theorem calibration_margin {b : ℕ} (w : MetricWeights b) (r : Fin b) :
    metricCost w (fun _ => 0) (fun s => if s = r then 1 else 0) = w r := by
  classical
  simp [metricCost]

end Transformer.ConvexRecall
