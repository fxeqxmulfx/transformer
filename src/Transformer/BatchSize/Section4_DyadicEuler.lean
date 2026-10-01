/-
# Dyadic Brownian Euler approximations at an arbitrary finite time

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The clipped grids retain the same origin and every second fine time is
a coarse time. Their endpoint is exactly T, including non-dyadic T.
-/

import Transformer.BatchSize.Section4_EulerRefinement

open MeasureTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- A dyadic grid stopped at T, Section 4.3 (2)--(3). -/
def dyadicGrid (T : ℝ≥0) (m n : ℕ) : ℝ≥0 := min ((n : ℝ≥0) / 2 ^ m) T

/-- Enough dyadic steps to reach the stopped endpoint T,
Section 4.3 (2)--(3). Extra steps after T have zero length. -/
def dyadicEndpointIndex (T : ℝ≥0) (m : ℕ) : ℕ := 2 ^ m * ⌈T⌉₊

/-- The dyadic grids are increasing, Section 4.3 (2)--(3). -/
theorem dyadicGrid_monotone (T : ℝ≥0) (m : ℕ) : Monotone (dyadicGrid T m) := by
  intro i j hij
  exact min_le_min (div_le_div_of_nonneg_right (by exact_mod_cast hij) (by positivity)) le_rfl

/-- The clipped grid begins at zero, Section 4.3 (2)--(3). -/
theorem dyadicGrid_zero (T : ℝ≥0) (m : ℕ) : dyadicGrid T m 0 = 0 := by
  simp [dyadicGrid]

/-- The endpoint index reaches T exactly, Section 4.3 (2)--(3). -/
theorem dyadicGrid_endpoint (T : ℝ≥0) (m : ℕ) :
    dyadicGrid T m (dyadicEndpointIndex T m) = T := by
  simp only [dyadicGrid, dyadicEndpointIndex, Nat.cast_mul, Nat.cast_pow, Nat.cast_ofNat]
  rw [mul_div_cancel_left₀ _ (by positivity)]
  exact min_eq_right (Nat.le_ceil T)

/-- Retaining alternate fine points gives exactly the preceding
dyadic grid, Section 4.3 (2)--(3). -/
theorem dyadicGrid_paired (T : ℝ≥0) (m : ℕ) :
    pairedGrid (dyadicGrid T (m + 1)) = dyadicGrid T m := by
  funext n
  dsimp [pairedGrid, dyadicGrid]
  congr 1
  simp only [Nat.cast_mul, Nat.cast_ofNat, pow_succ]
  field_simp

/-- Every stopped-grid step is at most the dyadic mesh,
Section 4.3 (2)--(3). -/
theorem dyadicGrid_mesh (T : ℝ≥0) (m n : ℕ) :
    (dyadicGrid T m (n + 1) : ℝ) - dyadicGrid T m n ≤ (1 : ℝ≥0) / 2 ^ m := by
  simp only [dyadicGrid, NNReal.coe_min, NNReal.coe_div, NNReal.coe_pow, NNReal.coe_ofNat,
    NNReal.coe_one, NNReal.coe_natCast, Nat.cast_add, Nat.cast_one, NNReal.coe_add]
  have hp : 0 < (2 : ℝ) ^ m := by positivity
  by_cases hn : (n : ℝ) / 2 ^ m ≤ T
  · rw [min_eq_left hn]
    exact (sub_le_sub_right (min_le_left _ _) _).trans (by
      rw [add_div]
      linarith)
  · have hn' : (T : ℝ) ≤ (n : ℝ) / 2 ^ m := le_of_not_ge hn
    have hn'' : (T : ℝ) ≤ ((n : ℝ) + 1) / 2 ^ m := by
      exact hn'.trans (div_le_div_of_nonneg_right (by linarith) hp.le)
    rw [min_eq_right hn', min_eq_right hn'']
    rw [sub_self]
    positivity

/-- The dyadic mesh never exceeds one, Section 4.3 (2)--(3). -/
theorem dyadicMesh_le_one (m : ℕ) : (1 : ℝ≥0) / 2 ^ m ≤ 1 := by
  exact (div_le_one (by positivity)).mpr (one_le_pow₀ (by norm_num))

/-- The actual Brownian Euler endpoint on a stopped dyadic grid,
Section 4.3 (2)--(3). -/
def dyadicEuler {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (x₀ : EucSpace d) (T : ℝ≥0) (m : ℕ) :
    BrownianSample d → EucSpace d :=
  eulerChain b a (dyadicGrid T m) x₀ (dyadicEndpointIndex T m)

/-- Consecutive stopped dyadic Euler approximations have geometrically
decaying mean-square differences, Section 4.3 (2)--(3). -/
theorem dyadicEuler_refinement_bound {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (Kb Ka M A : ℝ≥0)
    (hb : LipschitzWith Kb b) (ha : ∀ k, LipschitzWith Ka (fun x => a x k))
    (hbM : ∀ x, ‖b x‖ ≤ M) (haA : ∀ x k, |a x k| ≤ A)
    (x₀ : EucSpace d) (T : ℝ≥0) :
    ∃ K : ℝ, 0 ≤ K ∧ ∀ m : ℕ,
      (∫ ω, ‖dyadicEuler b a x₀ T (m + 1) ω - dyadicEuler b a x₀ T m ω‖ ^ 2 ∂brownianNoiseLaw d) ≤
        K * (1 / 2 : ℝ) ^ m := by
  let C : ℝ := 4 * Kb ^ 2 + 2 * (d : ℝ) * Ka ^ 2
  let K : ℝ := C * (M ^ 2 + (d : ℝ) * A ^ 2) * T * Real.exp ((1 + C) * T)
  have hC : 0 ≤ C := by dsimp [C]; positivity
  have hK : 0 ≤ K := by dsimp [K]; positivity
  refine ⟨K, hK, ?_⟩
  intro m
  let ε : ℝ≥0 := 1 / 2 ^ (m + 1)
  have hε : ε ≤ 1 := dyadicMesh_le_one (m + 1)
  have h := eulerChain_paired_refinement_bound b a Kb Ka M A ε hb ha hbM haA
    (dyadicGrid T (m + 1)) (dyadicGrid_monotone T (m + 1)) hε
    (dyadicGrid_mesh T (m + 1)) x₀ (dyadicEndpointIndex T (m + 1))
  have hind : dyadicEndpointIndex T (m + 1) = 2 * dyadicEndpointIndex T m := by
    dsimp [dyadicEndpointIndex]
    rw [pow_succ]
    ring
  have hpair : pairedEulerChain b a (dyadicGrid T (m + 1)) x₀ (dyadicEndpointIndex T (m + 1)) =
      dyadicEuler b a x₀ T m := by
    rw [hind, pairedEulerChain_even, dyadicGrid_paired]
    rfl
  rw [hpair, dyadicGrid_endpoint, dyadicGrid_zero] at h
  simp only [NNReal.coe_zero, sub_zero] at h
  have hcoef : M ^ 2 * (ε : ℝ) ^ 2 + (d : ℝ) * A ^ 2 * ε ≤
      (M ^ 2 + (d : ℝ) * A ^ 2) * (ε : ℝ) := by
    have hε' : (ε : ℝ) ≤ 1 := NNReal.coe_le_coe.mpr hε
    have hsq : (ε : ℝ) ^ 2 ≤ ε := by nlinarith [ε.coe_nonneg]
    nlinarith [mul_le_mul_of_nonneg_left hsq (sq_nonneg (M : ℝ))]
  have heq : (ε : ℝ) = (1 / 2 : ℝ) ^ (m + 1) := by
    simp only [ε, NNReal.coe_div, NNReal.coe_pow, NNReal.coe_one, NNReal.coe_ofNat, div_pow, one_pow]
  calc
    _ ≤ C * (M ^ 2 * (ε : ℝ) ^ 2 + (d : ℝ) * A ^ 2 * ε) * T * Real.exp ((1 + C) * T) := h
    _ ≤ C * ((M ^ 2 + (d : ℝ) * A ^ 2) * (ε : ℝ)) * T * Real.exp ((1 + C) * T) := by
      gcongr
    _ = K * (1 / 2 : ℝ) ^ (m + 1) := by rw [heq]; dsimp [K]; ring
    _ ≤ K * (1 / 2 : ℝ) ^ m := by rw [pow_succ]; nlinarith [pow_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 2) m]

/-- Joint nonvacuity of dyadic convergence assumptions, Section 4.3:
bounded constant coefficients on a positive finite horizon. -/
example : LipschitzWith 0 (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    (∀ _ : Fin 1, LipschitzWith 0 (fun _ : EucSpace 1 => (1 : ℝ))) ∧
    (∀ x : EucSpace 1, ‖(fun _ : EucSpace 1 => (0 : EucSpace 1)) x‖ ≤ (1 : ℝ≥0)) ∧
    (∀ (x : EucSpace 1) (k : Fin 1), |(fun _ : EucSpace 1 => fun _ : Fin 1 => (1 : ℝ)) x k| ≤
      (1 : ℝ≥0)) :=
  ⟨LipschitzWith.const _, fun _ => LipschitzWith.const _, by simp, by simp⟩

end Transformer.BatchSize
