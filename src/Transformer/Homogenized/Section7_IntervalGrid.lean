/-
# Finite grids approximate the original continuous time interval

ArXiv:2604.01978v1, §7, the continuous martingale maximal
inequality following eq:Gronwall_E_selfcontained.
The n-th grid has n+1 equal steps and is clipped at its final
endpoint when indexed by all natural numbers. Every clock is
nondecreasing, starts at zero and stays in the original interval.

Rounding down a given time chooses an actual point on each grid.
Its error is at most T/(n+1), so it converges from within [0,T].
Continuity then detects every strict threshold crossing on all
sufficiently fine grids, including crossings at the endpoints.
No path regularity or probability estimate is put into a definition.
-/

import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Algebra.Order.Floor.Semiring

open Set Filter Topology

namespace Transformer.Homogenized

/-- The actual n+1-step clock, clipped at T.
Source: arXiv:2604.01978v1, §7, the continuous maximal-inequality step. -/
noncomputable def intervalGridTime (T : ℝ) (n k : ℕ) : ℝ :=
  T * (min k (n + 1) : ℕ) / ((n : ℝ) + 1)

/-- The actual grid index obtained by rounding a time down.
Source: arXiv:2604.01978v1, §7, the continuous maximal-inequality step. -/
noncomputable def intervalGridIndex (T t : ℝ) (n : ℕ) : ℕ :=
  ⌊((n : ℝ) + 1) * t / T⌋₊

/-- Every clipped clock time belongs to the original time interval.
Source: arXiv:2604.01978v1, §7, the continuous maximal-inequality step. -/
theorem intervalGridTime_mem (T : ℝ) (hT : 0 ≤ T) (n k : ℕ) :
    intervalGridTime T n k ∈ Icc (0 : ℝ) T := by
  have hn : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  constructor
  · dsimp [intervalGridTime]; positivity
  · rw [intervalGridTime]
    apply (div_le_iff₀ hn).2
    have hk : ((min k (n + 1) : ℕ) : ℝ) ≤ (n : ℝ) + 1 := by exact_mod_cast Nat.min_le_right k (n + 1)
    exact mul_le_mul_of_nonneg_left hk hT

/-- The clock is nondecreasing, as required to sample a filtration.
Source: arXiv:2604.01978v1, §7, the continuous maximal-inequality step. -/
theorem intervalGridTime_monotone (T : ℝ) (hT : 0 ≤ T) (n : ℕ) : Monotone (intervalGridTime T n) := by
  intro i j hij
  apply div_le_div_of_nonneg_right _ (by positivity)
  apply mul_le_mul_of_nonneg_left _ hT
  exact_mod_cast min_le_min_right (n + 1) hij

/-- Rounding a time in [0,T] gives an index on the actual finite grid.
Source: arXiv:2604.01978v1, §7, the continuous maximal-inequality step. -/
theorem intervalGridIndex_le (T : ℝ) (hT : 0 < T) (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) (n : ℕ) :
    intervalGridIndex T t n ≤ n + 1 := by
  have ht0 := ht.1
  have harg : 0 ≤ ((n : ℝ) + 1) * t / T := by positivity
  have hf := Nat.floor_le harg
  have hu : ((n : ℝ) + 1) * t / T ≤ (n : ℝ) + 1 :=
    (div_le_iff₀ hT).2 (mul_le_mul_of_nonneg_left ht.2 (by positivity))
  have hbound := hf.trans hu
  dsimp [intervalGridIndex]
  exact_mod_cast hbound

/-- The selected clock time is below t with error at most T/(n+1).
Source: arXiv:2604.01978v1, §7, the continuous maximal-inequality step. -/
theorem intervalGridTime_bounds (T : ℝ) (hT : 0 < T) (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) (n : ℕ) :
    t - T / ((n : ℝ) + 1) ≤ intervalGridTime T n (intervalGridIndex T t n) ∧
      intervalGridTime T n (intervalGridIndex T t n) ≤ t := by
  have hn : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  have ht0 := ht.1
  have harg : 0 ≤ ((n : ℝ) + 1) * t / T := by positivity
  have hf : (intervalGridIndex T t n : ℝ) ≤ ((n : ℝ) + 1) * t / T := Nat.floor_le harg
  have hfl : ((n : ℝ) + 1) * t / T < (intervalGridIndex T t n : ℝ) + 1 := Nat.lt_floor_add_one _
  rw [intervalGridTime, min_eq_left (intervalGridIndex_le T hT t ht n)]
  constructor
  · apply (le_div_iff₀ hn).2
    have hlow := (div_lt_iff₀ hT).mp hfl
    calc
      _ = t * ((n : ℝ) + 1) - T := by field_simp
      _ ≤ _ := by nlinarith
  · apply (div_le_iff₀ hn).2
    have hhigh := (le_div_iff₀ hT).mp hf
    nlinarith

/-- The actual rounding error has the uniform absolute bound T/(n+1).
Source: arXiv:2604.01978v1, §7, the continuous maximal-inequality step. -/
theorem abs_intervalGridTime_sub_le (T : ℝ) (hT : 0 < T) (t : ℝ)
    (ht : t ∈ Icc (0 : ℝ) T) (n : ℕ) :
    |intervalGridTime T n (intervalGridIndex T t n) - t| ≤ T / ((n : ℝ) + 1) := by
  have h := intervalGridTime_bounds T hT t ht n
  rw [abs_of_nonpos (sub_nonpos.mpr h.2)]
  linarith [h.1]

/-- The selected actual clock times converge to the original time t.
Source: arXiv:2604.01978v1, §7, the continuous maximal-inequality step. -/
theorem intervalGridTime_tendsto (T : ℝ) (hT : 0 < T) (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) :
    Tendsto (fun n => intervalGridTime T n (intervalGridIndex T t n)) atTop (𝓝 t) := by
  have he : Tendsto (fun n : ℕ => T / ((n : ℝ) + 1)) atTop (𝓝 0) := by
    simpa only [mul_one_div, mul_zero] using
      (tendsto_one_div_add_atTop_nhds_zero_nat (𝕜 := ℝ)).const_mul T
  exact tendsto_of_tendsto_of_tendsto_of_le_of_le
    (by simpa only [sub_zero] using tendsto_const_nhds.sub he) tendsto_const_nhds
    (fun n => (intervalGridTime_bounds T hT t ht n).1) (fun n => (intervalGridTime_bounds T hT t ht n).2)

/-- Every strict crossing of a continuous path is detected on all sufficiently fine grids.
Source: arXiv:2604.01978v1, §7, the continuous maximal-inequality step. -/
theorem intervalGridTime_eventual_crossing (T : ℝ) (hT : 0 < T) (f : ℝ → ℝ)
    (hf : ContinuousOn f (Icc (0 : ℝ) T)) (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T)
    (a : ℝ) (ha : a < f t) :
    ∀ᶠ n : ℕ in atTop, a < f (intervalGridTime T n (intervalGridIndex T t n)) := by
  have hwithin : Tendsto (fun n => intervalGridTime T n (intervalGridIndex T t n)) atTop
      (𝓝[Icc (0 : ℝ) T] t) := tendsto_nhdsWithin_iff.mpr
    ⟨intervalGridTime_tendsto T hT t ht, .of_forall fun n => intervalGridTime_mem T hT.le n _⟩
  exact (Filter.Tendsto.comp (hf t ht) hwithin).eventually (eventually_gt_nhds ha)

/-- The actual clock starts at the original initial time.
Source: arXiv:2604.01978v1, §7, the martingale maximal inequality. -/
theorem intervalGridTime_zero (T : ℝ) (n : ℕ) : intervalGridTime T n 0 = 0 := by
  simp [intervalGridTime]

/-- The actual index chosen for the initial time is zero.
Source: arXiv:2604.01978v1, §7, the continuous maximal-inequality step. -/
theorem intervalGridIndex_zero (T : ℝ) (n : ℕ) : intervalGridIndex T 0 n = 0 := by
  simp [intervalGridIndex]

/-- The actual final clock time is exactly the original horizon.
Source: arXiv:2604.01978v1, §7, the martingale maximal inequality. -/
theorem intervalGridTime_last (T : ℝ) (n : ℕ) : intervalGridTime T n (n + 1) = T := by
  rw [intervalGridTime, min_self, Nat.cast_add, Nat.cast_one]
  field_simp

/-- A genuine interior time and nonconstant continuous path satisfy all
clock, rounding, continuity and strict-crossing premises. The first two
rounded clock times differ, so the approximation is nonconstant. -/
example : (0 : ℝ) < 1 ∧ (1 / 2 : ℝ) ∈ Icc (0 : ℝ) 1 ∧
    ContinuousOn (fun t : ℝ => t + 1) (Icc (0 : ℝ) 1) ∧ (1 : ℝ) < 1 / 2 + 1 ∧
    intervalGridTime 1 0 (intervalGridIndex 1 (1 / 2) 0) = 0 ∧
    intervalGridTime 1 1 (intervalGridIndex 1 (1 / 2) 1) = 1 / 2 ∧
    Tendsto (fun n => intervalGridTime 1 n (intervalGridIndex 1 (1 / 2) n)) atTop (𝓝 (1 / 2)) ∧
    ∀ᶠ n : ℕ in atTop, 1 < intervalGridTime 1 n (intervalGridIndex 1 (1 / 2) n) + 1 := by
  have ht : (1 / 2 : ℝ) ∈ Icc (0 : ℝ) 1 := by norm_num
  have hf : ContinuousOn (fun t : ℝ => t + 1) (Icc (0 : ℝ) 1) :=
    (continuous_id.add_const 1).continuousOn
  refine ⟨by norm_num, ht, hf, by norm_num, ?_, ?_,
    intervalGridTime_tendsto 1 one_pos (1 / 2) ht, ?_⟩
  · have hindex : intervalGridIndex 1 (1 / 2) 0 = 0 := by
      apply Nat.floor_eq_zero.mpr
      norm_num
    rw [hindex, intervalGridTime_zero]
  · norm_num [intervalGridTime, intervalGridIndex]
  · exact intervalGridTime_eventual_crossing 1 one_pos (fun t => t + 1) hf (1 / 2) ht 1 (by norm_num)

end Transformer.Homogenized
