/-
# Genuine dyadic grid neighbors of an arbitrary observation time

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Floor and ceiling indices give actual grid states, with time errors
bounded by the dyadic mesh independently of the finite horizon.
-/

import Transformer.BatchSize.Section4_DyadicEuler

open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The actual dyadic index preceding an observation, Section 4.3,
Theorem 1's Euler martingale-limit argument. -/
def dyadicLeftIndex (u : NNReal) (m : ℕ) : ℕ := ⌊(u : ℝ) * 2 ^ m⌋₊

/-- The actual dyadic index following an observation, Section 4.3,
Theorem 1. Past weights become adapted to every subsequent interval. -/
def dyadicRightIndex (u : NNReal) (m : ℕ) : ℕ := ⌈(u : ℝ) * 2 ^ m⌉₊

/-- Preceding and following indices both lie in the actual stopped
Euler chain for any larger horizon, Section 4.3 (2)--(3). -/
theorem dyadicObservation_indices_le_endpoint (u T : NNReal) (huT : u ≤ T) (m : ℕ) :
    dyadicLeftIndex u m ≤ dyadicEndpointIndex T m ∧
      dyadicRightIndex u m ≤ dyadicEndpointIndex T m := by
  have hbound : (u : ℝ) * 2 ^ m ≤ (dyadicEndpointIndex T m : ℝ) := by
    have hT : (T : ℝ) ≤ ((⌈T⌉₊ : ℕ) : ℝ) := by
      simpa only [NNReal.coe_natCast] using NNReal.coe_le_coe.mpr (Nat.le_ceil T)
    have hu := NNReal.coe_le_coe.mpr huT
    simp only [dyadicEndpointIndex, Nat.cast_mul, Nat.cast_pow, Nat.cast_ofNat]
    simpa only [mul_comm] using
      mul_le_mul_of_nonneg_right (hu.trans hT) (by positivity : (0 : ℝ) ≤ 2 ^ m)
  constructor
  · change ⌊(u : ℝ) * 2 ^ m⌋₊ ≤ dyadicEndpointIndex T m
    exact_mod_cast (Nat.floor_le (by positivity : (0 : ℝ) ≤ (u : ℝ) * 2 ^ m)).trans hbound
  · exact Nat.ceil_le.mpr hbound

/-- The actual left neighboring grid time is at or before the
observation and within one mesh, Section 4.3 (2)--(3). -/
theorem dyadicLeftTime_bounds (u T : NNReal) (huT : u ≤ T) (m : ℕ) :
    dyadicGrid T m (dyadicLeftIndex u m) ≤ u ∧
      (u : ℝ) - dyadicGrid T m (dyadicLeftIndex u m) ≤ (1 / 2 : ℝ) ^ m := by
  have hp : 0 < (2 : ℝ) ^ m := by positivity
  have hl := Nat.floor_le (by positivity : (0 : ℝ) ≤ (u : ℝ) * 2 ^ m)
  have hr := Nat.lt_floor_add_one ((u : ℝ) * 2 ^ m)
  have htime : ((dyadicLeftIndex u m : ℕ) : ℝ) / 2 ^ m ≤ u := (div_le_iff₀ hp).mpr hl
  have hgrid : (dyadicGrid T m (dyadicLeftIndex u m) : ℝ) =
      (dyadicLeftIndex u m : ℝ) / 2 ^ m := by
    simp only [dyadicGrid, NNReal.coe_min, NNReal.coe_div, NNReal.coe_natCast,
      NNReal.coe_pow, NNReal.coe_ofNat]
    exact min_eq_left (htime.trans (NNReal.coe_le_coe.mpr huT))
  constructor
  · exact NNReal.coe_le_coe.mp (hgrid ▸ htime)
  · rw [hgrid, div_pow, one_pow]
    apply (le_div_iff₀ hp).mpr
    have hmul : ((u : ℝ) - (dyadicLeftIndex u m : ℝ) / 2 ^ m) * 2 ^ m =
        (u : ℝ) * 2 ^ m - dyadicLeftIndex u m := by field_simp
    rw [hmul]
    dsimp only [dyadicLeftIndex]
    linarith

/-- The actual right neighboring grid time is at or after the
observation and within one mesh, Section 4.3 (2)--(3). -/
theorem dyadicRightTime_bounds (u T : NNReal) (huT : u ≤ T) (m : ℕ) :
    u ≤ dyadicGrid T m (dyadicRightIndex u m) ∧
      (dyadicGrid T m (dyadicRightIndex u m) : ℝ) - u ≤ (1 / 2 : ℝ) ^ m := by
  have hp : 0 < (2 : ℝ) ^ m := by positivity
  have hl := Nat.le_ceil ((u : ℝ) * 2 ^ m)
  have hr := Nat.ceil_lt_add_one (by positivity : (0 : ℝ) ≤ (u : ℝ) * 2 ^ m)
  have htime : (u : ℝ) ≤ (dyadicRightIndex u m : ℝ) / 2 ^ m := (le_div_iff₀ hp).mpr hl
  constructor
  · apply NNReal.coe_le_coe.mp
    simp only [dyadicGrid, NNReal.coe_min, NNReal.coe_div, NNReal.coe_natCast,
      NNReal.coe_pow, NNReal.coe_ofNat]
    exact le_min htime (NNReal.coe_le_coe.mpr huT)
  · have hgrid : (dyadicGrid T m (dyadicRightIndex u m) : ℝ) ≤
        (dyadicRightIndex u m : ℝ) / 2 ^ m := by
      simp only [dyadicGrid, NNReal.coe_min, NNReal.coe_div, NNReal.coe_natCast,
        NNReal.coe_pow, NNReal.coe_ofNat]
      exact min_le_left _ _
    refine (sub_le_sub_right hgrid _).trans ?_
    rw [div_pow, one_pow]
    apply (le_div_iff₀ hp).mpr
    have hmul : ((dyadicRightIndex u m : ℝ) / 2 ^ m - (u : ℝ)) * 2 ^ m =
        (dyadicRightIndex u m : ℝ) - (u : ℝ) * 2 ^ m := by field_simp
    rw [hmul]
    dsimp only [dyadicRightIndex]
    linarith

/-- Joint nonvacuity of dyadic observation-time hypotheses,
Section 4.3: a positive observation on a larger finite horizon. -/
example : (1 : NNReal) ≤ 2 := by norm_num

end Transformer.BatchSize
