module

public import Transformer.Residual.Basic
public import Transformer.GPTMini.RMSNorm
public import Mathlib.Analysis.SpecialFunctions.Sqrt
public import Mathlib.Analysis.Calculus.Deriv.Pow
public import Mathlib.Analysis.Calculus.Deriv.Mul
public import Mathlib.Analysis.Calculus.Deriv.Add
public import Mathlib.Analysis.Calculus.Deriv.Comp
public import Mathlib.Analysis.Calculus.Deriv.Inv

/-!
# Normalization amplitude and residual gradient transport

Local research accompanying `experiments/residual_gradient_paths/README.md`,
Eqs. (1)–(4). These are exact real-arithmetic statements about the actual
RMSNorm and residual recurrence. They do not assert optimizer convergence,
floating-point accuracy, or generalization of a trained transformer.
The radial result concerns branch strength; learning feature directions is
still possible. The depth bound assumes the displayed fixed linear branches;
it is not a lower bound for arbitrary transformer Jacobians.
-/

@[expose] public section

namespace Transformer.Residual

/-- Normalized amplitude on a unit-RMS direction; benchmark Eq. (3). -/
noncomputable def radialRMS (eps a : ℝ) : ℝ :=
  a / Real.sqrt (a ^ 2 + eps)

/-- Benchmark Eq. (1): the scalar amplitude is the existing vector RMSNorm,
not an unrelated scalar surrogate. Unit RMS means squared norm equals width. -/
theorem rmsNormEps_unit_direction {d : ℕ} (hd : 0 < d)
    (v : EucSpace d) (hv : ‖v‖ ^ 2 = (d : ℝ)) (eps a : ℝ) :
    GPTMini.rmsNormEps eps (a • v) = radialRMS eps a • v := by
  have hdim : (0 : ℝ) < d := by exact_mod_cast hd
  have hs : Real.sqrt (d : ℝ) ≠ 0 := (Real.sqrt_pos.mpr hdim).ne'
  have hnorm : ‖a • v‖ ^ 2 + (d : ℝ) * eps = (d : ℝ) * (a ^ 2 + eps) := by
    rw [norm_smul, Real.norm_eq_abs, mul_pow, sq_abs, hv]
    ring
  rw [GPTMini.rmsNormEps, hnorm, Real.sqrt_mul hdim.le, smul_smul, radialRMS]
  congr 1
  field_simp

/-- Nonempty unit-RMS directions exist, already in dimension one. -/
example : (0 < (1 : ℕ)) ∧
    ‖(PiLp.single 2 (0 : Fin 1) (1 : ℝ) : EucSpace 1)‖ ^ 2 = (1 : ℝ) := by
  norm_num [PiLp.norm_single]

/-- Benchmark Eq. (3), including finite positive epsilon: the radial derivative
is positive, though it may be extremely small. Ideal homogeneity alone would
incorrectly replace this derivative by zero. -/
theorem radialRMS_hasDerivAt (eps a : ℝ) (heps : 0 < eps) :
    HasDerivAt (radialRMS eps) (eps / Real.sqrt (a ^ 2 + eps) ^ 3) a := by
  have hp : 0 < a ^ 2 + eps := by positivity
  have hs : Real.sqrt (a ^ 2 + eps) ≠ 0 := (Real.sqrt_pos.mpr hp).ne'
  have hsq := Real.sq_sqrt hp.le
  have hbase := ((hasDerivAt_id' a).pow 2).add_const eps
  have hroot := hbase.sqrt hp.ne'
  have hdiv := (hasDerivAt_id' a).div hroot hs
  apply hdiv.congr_deriv
  dsimp
  field_simp
  nlinarith

/-- The epsilon hypothesis holds at the exact FP32 epsilon used in the runs. -/
example : (0 : ℝ) < 1 / 8388608 := by norm_num

/-- Benchmark Eq. (3): derivative of a gain before the vector normalization,
at gain one. The gate parameter's own derivative multiplies this by the chain rule. -/
theorem before_gain_hasDerivAt (eps a : ℝ) (heps : 0 < eps) :
    HasDerivAt (fun g => radialRMS eps (a * g))
      (a * eps / Real.sqrt (a ^ 2 + eps) ^ 3) 1 := by
  have h := (radialRMS_hasDerivAt eps a heps).comp_of_eq (1 : ℝ)
    ((hasDerivAt_id' (1 : ℝ)).const_mul a) (by ring)
  unfold Function.comp at h
  apply h.congr_deriv
  ring

example : (0 : ℝ) < 1 / 8388608 := by norm_num

/-- Benchmark Eq. (3): moving the gain after normalization preserves its
amplitude derivative. The two placements agree at gain one. -/
theorem after_gain_hasDerivAt (eps a : ℝ) :
    HasDerivAt (fun g => radialRMS eps a * g) (radialRMS eps a) 1 := by
  have h := (hasDerivAt_id' (1 : ℝ)).const_mul (radialRMS eps a)
  simpa only [mul_one] using h

/-- Benchmark Eq. (3): the precise before/after derivative ratio. No bound
on model width repairs this radial attenuation for a fixed branch amplitude. -/
theorem gain_derivative_ratio (eps a : ℝ) (heps : 0 < eps) (ha : a ≠ 0) :
    (a * eps / Real.sqrt (a ^ 2 + eps) ^ 3) / radialRMS eps a =
      eps / (a ^ 2 + eps) := by
  have hp : 0 < a ^ 2 + eps := by positivity
  have hs : Real.sqrt (a ^ 2 + eps) ≠ 0 := (Real.sqrt_pos.mpr hp).ne'
  have hsq := Real.sq_sqrt hp.le
  unfold radialRMS
  field_simp
  nlinarith

/-- A nonzero branch at finite epsilon satisfies both ratio hypotheses. -/
example : (0 : ℝ) < 1 / 8388608 ∧ (1000 : ℝ) ≠ 0 := by norm_num

/-- Benchmark Eq. (4): linear branches in the existing residual recurrence
give a product, despite each layer having an identity skip. -/
theorem residualRun_linear (c x : ℝ) (depth : ℕ) :
    residualRun (fun _ y => c * y) x depth = (1 + c) ^ depth * x := by
  induction depth with
  | zero => simp [residualRun]
  | succ depth ih =>
    rw [residualRun, ih, pow_succ]
    ring

/-- Benchmark Eq. (4): transport's exact input derivative, including zero
and exponentially small derivatives, follows from the full recurrence. -/
theorem residualRun_linear_hasDerivAt (c x : ℝ) (depth : ℕ) :
    HasDerivAt (fun y => residualRun (fun _ z => c * z) y depth)
      ((1 + c) ^ depth) x := by
  simp_rw [residualRun_linear]
  have h := (hasDerivAt_id' x).const_mul ((1 + c) ^ depth)
  simpa only [mul_one] using h

/-- Counterexample to an unconditional positive derivative bound from an
identity skip: branch -x cancels it. Benchmark Eq. (4), depth one. -/
theorem identity_skip_zero_derivative (x : ℝ) :
    HasDerivAt (fun y => residualRun (fun _ z => (-1 : ℝ) * z) y 1) 0 x := by
  have h := residualRun_linear_hasDerivAt (-1) x 1
  simpa using h

/-- Benchmark Eq. (4): depth-scaled contraction retains at least half the
derivative. This is a bound for these branches, not arbitrary learned branches. -/
theorem depth_scaled_transport_lower_bound (depth : ℕ) (hd : 0 < depth) :
    (1 / 2 : ℝ) ≤ (1 - 1 / (2 * (depth : ℝ))) ^ depth := by
  have hpos : (0 : ℝ) < depth := by exact_mod_cast hd
  have hone : (1 : ℝ) ≤ depth := by exact_mod_cast hd
  have ha : (-2 : ℝ) ≤ -(1 / (2 * (depth : ℝ))) := by
    apply neg_le_neg
    apply (div_le_iff₀ (by positivity : (0 : ℝ) < 2 * depth)).mpr
    linarith
  have h := one_add_mul_le_pow ha depth
  have heq : (1 : ℝ) + depth * -(1 / (2 * (depth : ℝ))) = 1 / 2 := by
    field_simp
    ring
  rw [heq] at h
  simpa only [sub_eq_add_neg] using h

/-- The transport hypothesis includes the deepest measured case. -/
example : 0 < (32 : ℕ) := by norm_num

end Transformer.Residual
