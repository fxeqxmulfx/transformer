/-
# Weighted norm estimates for Euler refinement

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The weighted estimate keeps perturbations proportional to the time step,
which is essential when accumulating errors over increasingly fine grids.
-/

import Transformer.BatchSize.Section4_EulerStability

noncomputable section

namespace Transformer.BatchSize

/-- A weighted square-norm bound for an Euler drift increment,
Section 4.3 (2)--(3). -/
theorem norm_add_time_smul_sq_le {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (z v : E) (δ : ℝ) (hδ : 0 ≤ δ) :
    ‖z + δ • v‖ ^ 2 ≤ (1 + δ) * ‖z‖ ^ 2 + δ * (1 + δ) * ‖v‖ ^ 2 := by
  have hnorm : ‖z + δ • v‖ ≤ ‖z‖ + δ * ‖v‖ := by
    simpa only [norm_smul, Real.norm_eq_abs, abs_of_nonneg hδ] using norm_add_le z (δ • v)
  have hsq := (sq_le_sq₀ (norm_nonneg _) (by positivity)).mpr hnorm
  nlinarith [mul_nonneg hδ (sq_nonneg (‖z‖ - ‖v‖))]

/-- Nonvacuity of the weighted drift hypothesis, Section 4.3. -/
example : (0 : ℝ) ≤ 1 / 2 := by norm_num

/-- A split-state square-norm bound, Section 4.3 (2)--(3), for comparing
current and lagged Euler coefficients. -/
theorem norm_sub_split_sq_le {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (x y u : E) : ‖x - u‖ ^ 2 ≤ 2 * ‖x - y‖ ^ 2 + 2 * ‖y - u‖ ^ 2 := by
  have h := norm_add_time_smul_sq_le (x - y) (y - u) 1 (by norm_num)
  simpa only [one_smul, sub_add_sub_cancel, one_add_one_eq_two, one_mul] using h

end Transformer.BatchSize
