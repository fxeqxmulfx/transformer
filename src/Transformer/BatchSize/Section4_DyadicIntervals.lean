/-
# Identifying the frozen state on genuine dyadic intervals

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The floor selector identifies the actual chain state at the left endpoint.
It is measurable, including at stopped and negative observation times.
-/

import Transformer.BatchSize.Section4_DyadicObservationTimes
import Mathlib.MeasureTheory.Function.Floor

open MeasureTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- On every nonempty stopped dyadic interval, the true floor
selector is exactly its left index, Section 4.3 (2)--(3).
The strict upper endpoint avoids the next interval's coefficient. -/
theorem dyadicLeftIndex_on_interval (T : NNReal) (m j : ℕ) (u : ℝ)
    (hu : (dyadicGrid T m j : ℝ) ≤ u ∧ u < dyadicGrid T m (j + 1)) :
    dyadicLeftIndex u.toNNReal m = j := by
  have hp : 0 < (2 : ℝ) ^ m := by positivity
  have hnext : (dyadicGrid T m (j + 1) : ℝ) = min (((j : ℝ) + 1) / 2 ^ m) T := by
    simp [dyadicGrid, NNReal.coe_min, NNReal.coe_div, NNReal.coe_natCast,
      NNReal.coe_pow, NNReal.coe_ofNat]
  have hleft : (dyadicGrid T m j : ℝ) = min ((j : ℝ) / 2 ^ m) T := by
    simp only [dyadicGrid, NNReal.coe_min, NNReal.coe_div, NNReal.coe_natCast,
      NNReal.coe_pow, NNReal.coe_ofNat]
  have huT : u < T := hu.2.trans_le (hnext ▸ min_le_right _ _)
  have hju : (j : ℝ) / 2 ^ m ≤ u := by
    by_contra hh
    have hmin := lt_min (lt_of_not_ge hh) huT
    rw [← hleft] at hmin
    exact (not_lt_of_ge hu.1) hmin
  have huj : u < ((j : ℝ) + 1) / 2 ^ m :=
    hu.2.trans_le (hnext ▸ min_le_left _ _)
  have hu0 : 0 ≤ u := le_trans (by positivity : (0 : ℝ) ≤ j / 2 ^ m) hju
  unfold dyadicLeftIndex
  rw [Real.coe_toNNReal u hu0]
  apply (Nat.floor_eq_iff (by positivity : (0 : ℝ) ≤ u * 2 ^ m)).mpr
  exact ⟨(div_le_iff₀ hp).mp hju, (lt_div_iff₀ hp).mp huj⟩

/-- The genuine dyadic observation selector is measurable in real
time, Section 4.3 (2)--(3), including the truncation before zero. -/
theorem dyadicLeftIndex_measurable (m : ℕ) :
    Measurable (fun u : ℝ => dyadicLeftIndex u.toNNReal m) :=
  Nat.measurable_floor.comp ((NNReal.continuous_coe.measurable.comp
    continuous_real_toNNReal.measurable).mul_const (2 ^ m))

/-- Joint nonvacuity of the dyadic interval hypotheses,
Section 4.3: an interior point of a positive stopped interval. -/
example : (dyadicGrid 2 1 1 : ℝ) ≤ 3 / 4 ∧ (3 / 4 : ℝ) < dyadicGrid 2 1 2 := by
  norm_num [dyadicGrid, NNReal.coe_min, NNReal.coe_div, NNReal.coe_natCast,
    NNReal.coe_pow, NNReal.coe_ofNat]

end Transformer.BatchSize
