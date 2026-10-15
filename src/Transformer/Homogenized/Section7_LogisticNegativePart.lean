import Transformer.Homogenized.Section7_LogisticBoundaryCoordinate

/-!
# Smooth global tests for the logistic state interval

ArXiv:2604.01978v1, §7, `sec: sde.logistic`, interval invariance.
Apply sqrt(w²+ε²)-w to the bounded coordinate from the preceding module.
The tests are globally smooth, nonnegative, and uniformly bounded.
Their actual original generators satisfy |LY|≤28Y everywhere.
For a state initially in [-1,1], the test is at most ε.
Thus the smooth-test martingale problem controls negative boundary
coordinates before any range property of the process is available.
-/

open Real MeasureTheory Set Filter
open scoped Topology
namespace Transformer.Homogenized

/-- The actual smooth approximation of twice the negative part.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, interval invariance. -/
noncomputable def smoothNegativePart (ε x : ℝ) : ℝ := Real.sqrt (x ^ 2 + ε ^ 2) - x

/-- A positive regularization removes the square-root singularity globally.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, interval invariance. -/
theorem contDiff_smoothNegativePart (ε : ℝ) (hε : 0 < ε) :
    ContDiff ℝ (⊤ : ℕ∞) (smoothNegativePart ε) := by
  exact ((contDiff_id.pow 2).add contDiff_const).sqrt
    (fun x => by positivity) |>.sub contDiff_id

/-- The test is nonnegative and is at most ε on nonnegative inputs.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, interval invariance. -/
theorem smoothNegativePart_bounds (ε x : ℝ) (hε : 0 < ε) :
    0 ≤ smoothNegativePart ε x ∧
      smoothNegativePart ε x ≤ 2 * |x| + ε ∧
      (0 ≤ x → smoothNegativePart ε x ≤ ε) := by
  have hab : |x| ≤ Real.sqrt (x ^ 2 + ε ^ 2) := Real.abs_le_sqrt (by nlinarith [sq_nonneg ε])
  have hsum : Real.sqrt (x ^ 2 + ε ^ 2) ≤ |x| + ε :=
    Real.sqrt_le_iff.mpr ⟨by positivity, by nlinarith [sq_abs x, abs_nonneg x]⟩
  unfold smoothNegativePart
  refine ⟨by linarith [le_abs_self x], ?_, ?_⟩
  · linarith [neg_le_abs x]
  · intro hx
    rw [abs_of_nonneg hx] at hsum
    linarith

/-- The actual first derivative of the regularized negative part.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, interval invariance. -/
theorem hasDerivAt_smoothNegativePart (ε x : ℝ) (hε : 0 < ε) :
    HasDerivAt (smoothNegativePart ε) (x / Real.sqrt (x ^ 2 + ε ^ 2) - 1) x := by
  have hH : 0 < x ^ 2 + ε ^ 2 := by positivity
  have hs : Real.sqrt (x ^ 2 + ε ^ 2) ≠ 0 := (Real.sqrt_pos.mpr hH).ne'
  have hd := ((((hasDerivAt_id x).pow 2).add_const (ε ^ 2)).sqrt hH.ne').sub (hasDerivAt_id x)
  apply hd.congr_deriv
  dsimp
  field_simp

/-- The actual second derivative, retaining the positive regularization.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, interval invariance. -/
theorem hasDerivAt_smoothNegativePart_derivative (ε x : ℝ) (hε : 0 < ε) :
    HasDerivAt (fun y => y / Real.sqrt (y ^ 2 + ε ^ 2) - 1)
      (ε ^ 2 / Real.sqrt (x ^ 2 + ε ^ 2) ^ 3) x := by
  have hH : 0 < x ^ 2 + ε ^ 2 := by positivity
  have hs : Real.sqrt (x ^ 2 + ε ^ 2) ≠ 0 := (Real.sqrt_pos.mpr hH).ne'
  have hd := ((hasDerivAt_id x).div
    ((((hasDerivAt_id x).pow 2).add_const (ε ^ 2)).sqrt hH.ne') hs).sub_const 1
  apply hd.congr_deriv
  dsimp
  field_simp
  nlinarith [Real.sq_sqrt hH.le]

/-- The multiplicative drift and variance bounds control the actual test drift.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, interval invariance. -/
theorem smoothNegativePart_drift_bound (ε x A B : ℝ) (hε : 0 < ε)
    (hA : |A| ≤ 20) (hB : B ∈ Icc (0 : ℝ) 8) :
    |-(A * x * smoothNegativePart ε x) / Real.sqrt (x ^ 2 + ε ^ 2) +
      B * x ^ 2 * ε ^ 2 / (2 * Real.sqrt (x ^ 2 + ε ^ 2) ^ 3)| ≤
        28 * smoothNegativePart ε x := by
  let s := Real.sqrt (x ^ 2 + ε ^ 2)
  let Z := smoothNegativePart ε x
  have hs : 0 < s := Real.sqrt_pos.mpr (by positivity)
  have hs2 : s ^ 2 = x ^ 2 + ε ^ 2 := Real.sq_sqrt (by positivity)
  have hxs : |x| ≤ s := Real.abs_le_sqrt (by nlinarith [sq_nonneg ε])
  have hZ : 0 ≤ Z := (smoothNegativePart_bounds ε x hε).1
  have hB0 : 0 ≤ B := hB.1
  have hf : |-(A * x * Z) / s| ≤ 20 * Z := by
    rw [abs_div, abs_neg, abs_mul, abs_mul, abs_of_nonneg hZ, abs_of_pos hs]
    apply (div_le_iff₀ hs).mpr
    calc
      _ ≤ 20 * s * Z := mul_le_mul_of_nonneg_right
        (mul_le_mul hA hxs (abs_nonneg _) (by norm_num)) hZ
      _ = _ := by ring
  have heps : ε ^ 2 ≤ 2 * Z * s := by
    have hx : x ≤ s := (le_abs_self x).trans hxs
    dsimp [Z, smoothNegativePart]
    change ε ^ 2 ≤ 2 * (s - x) * s
    nlinarith [sq_nonneg (s - x)]
  have hprod : x ^ 2 * ε ^ 2 ≤ 2 * Z * s ^ 3 := by
    calc
      _ ≤ s ^ 2 * (2 * Z * s) := mul_le_mul (by nlinarith [sq_nonneg ε]) heps
        (sq_nonneg ε) (sq_nonneg s)
      _ = _ := by ring
  have hq0 : 0 ≤ B * x ^ 2 * ε ^ 2 / (2 * s ^ 3) := by positivity
  have hq : B * x ^ 2 * ε ^ 2 / (2 * s ^ 3) ≤ 8 * Z := by
    apply (div_le_iff₀ (by positivity : 0 < 2 * s ^ 3)).mpr
    calc
      _ = B * (x ^ 2 * ε ^ 2) := by ring
      _ ≤ 8 * (2 * Z * s ^ 3) := mul_le_mul hB.2 hprod (by positivity) (by norm_num)
      _ = _ := by ring
  calc
    _ ≤ |-(A * x * Z) / s| + |B * x ^ 2 * ε ^ 2 / (2 * s ^ 3)| := abs_add_le _ _
    _ ≤ 20 * Z + 8 * Z := add_le_add hf (by rwa [abs_of_nonneg hq0])
    _ = _ := by ring

/-- The actual globally smooth test of the original logistic state.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, interval invariance. -/
noncomputable def logisticBoundaryTest (ε u : ℝ) : ℝ :=
  smoothNegativePart ε (logisticBoundaryCoordinate u)

/-- The exact original generator of the composite test, with both terms retained.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, interval invariance. -/
theorem logisticBoundaryTest_generator (ε u : ℝ) (hε : 0 < ε) :
    logisticGenerator (logisticBoundaryTest ε) u =
      -(logisticBoundaryRate u * logisticBoundaryCoordinate u * logisticBoundaryTest ε u) /
          Real.sqrt (logisticBoundaryCoordinate u ^ 2 + ε ^ 2) +
        logisticBoundaryVariance u * logisticBoundaryCoordinate u ^ 2 * ε ^ 2 /
          (2 * Real.sqrt (logisticBoundaryCoordinate u ^ 2 + ε ^ 2) ^ 3) := by
  have hs : Real.sqrt (logisticBoundaryCoordinate u ^ 2 + ε ^ 2) ≠ 0 := by positivity
  have hd : deriv (smoothNegativePart ε) = fun x => x / Real.sqrt (x ^ 2 + ε ^ 2) - 1 :=
    funext (fun x => (hasDerivAt_smoothNegativePart ε x hε).deriv)
  change logisticGenerator (fun v => smoothNegativePart ε (logisticBoundaryCoordinate v)) u = _
  rw [logisticGenerator_comp _ _ (contDiff_smoothNegativePart ε hε)
    contDiff_logisticBoundaryCoordinate, hd,
    (hasDerivAt_smoothNegativePart_derivative ε _ hε).deriv,
    (logisticBoundaryCoordinate_generator u).1]
  have hq : (1 - u ^ 2) ^ 2 * deriv logisticBoundaryCoordinate u ^ 2 =
      logisticBoundaryVariance u * logisticBoundaryCoordinate u ^ 2 / 2 := by
    linarith [(logisticBoundaryCoordinate_generator u).2]
  rw [show (1 - u ^ 2) ^ 2 *
      (ε ^ 2 / Real.sqrt (logisticBoundaryCoordinate u ^ 2 + ε ^ 2) ^ 3) *
      deriv logisticBoundaryCoordinate u ^ 2 =
      ((1 - u ^ 2) ^ 2 * deriv logisticBoundaryCoordinate u ^ 2) *
      (ε ^ 2 / Real.sqrt (logisticBoundaryCoordinate u ^ 2 + ε ^ 2) ^ 3) by ring, hq]
  unfold logisticBoundaryTest smoothNegativePart
  field_simp
  ring

/-- The composite is admissible in the original global smooth-test problem.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, interval invariance. -/
theorem contDiff_logisticBoundaryTest (ε : ℝ) (hε : 0 < ε) :
    ContDiff ℝ (⊤ : ℕ∞) (logisticBoundaryTest ε) :=
  (contDiff_smoothNegativePart ε hε).comp contDiff_logisticBoundaryCoordinate

/-- Global bounds and the small initial value needed for interval invariance.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, interval invariance. -/
theorem logisticBoundaryTest_bounds (ε u : ℝ) (hε : ε ∈ Ioc (0 : ℝ) 1) :
    0 ≤ logisticBoundaryTest ε u ∧ logisticBoundaryTest ε u ≤ 3 ∧
      |logisticGenerator (logisticBoundaryTest ε) u| ≤ 28 * logisticBoundaryTest ε u ∧
        (u ∈ Icc (-1 : ℝ) 1 → logisticBoundaryTest ε u ≤ ε) := by
  have hb := smoothNegativePart_bounds ε (logisticBoundaryCoordinate u) hε.1
  have hc := logisticBoundaryCoordinate_bounds u
  have hcoef := logisticBoundary_coefficients_bounds u
  refine ⟨hb.1, by dsimp [logisticBoundaryTest]; linarith [hb.2.1, hc.1, hε.2], ?_, ?_⟩
  · rw [logisticBoundaryTest_generator ε u hε.1]
    exact smoothNegativePart_drift_bound ε (logisticBoundaryCoordinate u)
      (logisticBoundaryRate u) (logisticBoundaryVariance u) hε.1 hcoef.1 hcoef.2
  · intro hu
    exact hb.2.2 (hc.2.mpr hu)

/-- Positive ε, genuine bounded coefficients, a positive coordinate, and an
absorbing endpoint realize all numerical and initial-state premises. -/
example : (1 : ℝ) ∈ Ioc (0 : ℝ) 1 ∧ |(-4 : ℝ)| ≤ 20 ∧
    (4 : ℝ) ∈ Icc (0 : ℝ) 8 ∧
    (0 : ℝ) ≤ logisticBoundaryCoordinate (1 / 2) ∧
    logisticBoundaryTest 1 1 = 1 := by
  norm_num [logisticBoundaryTest, smoothNegativePart, logisticBoundaryCoordinate]

end Transformer.Homogenized
