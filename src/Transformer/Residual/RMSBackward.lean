module

public import Transformer.Residual.GradientPaths

/-!
# Separating the radial and tangent RMS backward

Local research for `experiments/rms_backward/README.md`, Eqs. (1)–(3).
The usual RMS VJP subtracts two terms which nearly agree on a radial
upstream gradient. The equivalent expression below isolates the small,
positive finite-epsilon term. In the coordinate statements, `m` denotes
mean-square input energy and `c` denotes the mean input/upstream product.
The identities hold for every coordinate, so they apply to the vector VJP.

These are real-arithmetic identities, not a floating-point accuracy proof.
The numerical benchmark separately checks native and candidate derivatives.
Computing a more accurate local derivative does not remove a small true
derivative, roundoff in later upstream reductions, or optimizer conditioning.
-/

@[expose] public section

namespace Transformer.Residual

/-- Benchmark Eq. (2): isolate the radial coefficient without the subtraction
of two almost equal inverse-root terms. Nonnegative energy is explicit. -/
theorem rms_inverse_gap (m eps : ℝ) (hm : 0 ≤ m) (heps : 0 < eps) :
    1 / Real.sqrt (m + eps) - m / Real.sqrt (m + eps) ^ 3 =
      eps / Real.sqrt (m + eps) ^ 3 := by
  have hp : 0 < m + eps := by linarith
  have hs : Real.sqrt (m + eps) ≠ 0 := (Real.sqrt_pos.mpr hp).ne'
  have hsq := Real.sq_sqrt hp.le
  field_simp
  nlinarith

/-- Positive epsilon and nonnegative energy include a unit-RMS vector. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 / 8388608 := by norm_num

/-- Benchmark Eqs. (1)–(2): one coordinate of the RMS VJP equals its tangent
part plus a separate radial part. Here `x*c/m` is that coordinate of the
projection of the upstream gradient onto the input. Positive energy avoids
dividing by zero; the zero-input derivative is handled below. -/
theorem rms_vjp_decomposition (m eps x g c : ℝ) (hm : 0 < m) (heps : 0 < eps) :
    g / Real.sqrt (m + eps) - x * c / Real.sqrt (m + eps) ^ 3 =
      (g - x * c / m) / Real.sqrt (m + eps) +
        eps * (x * c / m) / Real.sqrt (m + eps) ^ 3 := by
  have hp : 0 < m + eps := by linarith
  have hs : Real.sqrt (m + eps) ≠ 0 := (Real.sqrt_pos.mpr hp).ne'
  calc
    _ = (g - x * c / m) / Real.sqrt (m + eps) +
        (x * c / m) *
          (1 / Real.sqrt (m + eps) - m / Real.sqrt (m + eps) ^ 3) := by
      field_simp
      ring
    _ = _ := by
      rw [rms_inverse_gap m eps hm.le heps]
      ring

/-- Both hypotheses hold for a nonzero unit direction with finite epsilon. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 8388608 := by norm_num

/-- Benchmark Eq. (2): for an upstream gradient `a*x`, the mean product is
`a*m`; only the epsilon-sized radial part remains in every coordinate. -/
theorem rms_vjp_parallel (m eps x a : ℝ) (hm : 0 < m) (heps : 0 < eps) :
    (a * x) / Real.sqrt (m + eps) - x * (a * m) / Real.sqrt (m + eps) ^ 3 =
      a * x * eps / Real.sqrt (m + eps) ^ 3 := by
  have h := rms_vjp_decomposition m eps x (a * x) (a * m) hm heps
  have hprojection : x * (a * m) / m = a * x := by
    field_simp
  rw [hprojection, sub_self, zero_div, zero_add] at h
  calc
    _ = eps * (a * x) / Real.sqrt (m + eps) ^ 3 := h
    _ = _ := by ring

example : (0 : ℝ) < 1000000 ∧ (0 : ℝ) < 1 / 8388608 := by norm_num

/-- Benchmark Eq. (2): tangent and radial coefficients still have the
conditioning ratio epsilon/(energy+epsilon). Rearrangement does not enlarge
the true radial derivative or establish optimizer recovery. -/
theorem rms_backward_coefficient_ratio (m eps : ℝ) (hm : 0 ≤ m) (heps : 0 < eps) :
    (eps / Real.sqrt (m + eps) ^ 3) / (1 / Real.sqrt (m + eps)) = eps / (m + eps) := by
  have hp : 0 < m + eps := by linarith
  have hs : Real.sqrt (m + eps) ≠ 0 := (Real.sqrt_pos.mpr hp).ne'
  have hsq := Real.sq_sqrt hp.le
  field_simp
  nlinarith

example : (0 : ℝ) ≤ 1000000 ∧ (0 : ℝ) < 1 / 8388608 := by norm_num

/-- Benchmark Eq. (3), zero input: the scalar RMS derivative is finite
and equals the ordinary inverse-root coefficient; no input projection is needed. -/
theorem radialRMS_zero_hasDerivAt (eps : ℝ) (heps : 0 < eps) :
    HasDerivAt (radialRMS eps) (1 / Real.sqrt eps) 0 := by
  have hs : Real.sqrt eps ≠ 0 := (Real.sqrt_pos.mpr heps).ne'
  have hsq := Real.sq_sqrt heps.le
  have h := radialRMS_hasDerivAt eps 0 heps
  simp only [zero_pow (by decide : 2 ≠ 0), zero_add] at h
  apply h.congr_deriv
  field_simp
  nlinarith

example : (0 : ℝ) < 1 / 8388608 := by norm_num

/-- Benchmark scalar oracle: the derivative of squared amplitude error follows
from the finite-epsilon RMS derivative, independently of a numerical backward. -/
theorem radialRMS_loss_hasDerivAt (eps a target : ℝ) (heps : 0 < eps) :
    HasDerivAt (fun z => (radialRMS eps z - target) ^ 2)
      (2 * (radialRMS eps a - target) * eps / Real.sqrt (a ^ 2 + eps) ^ 3) a := by
  have h := ((radialRMS_hasDerivAt eps a heps).sub_const target).pow 2
  apply h.congr_deriv
  ring

example : (0 : ℝ) < 1 / 8388608 := by norm_num

end Transformer.Residual
