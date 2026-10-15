import Transformer.Homogenized.Logistic
import Mathlib.Analysis.Calculus.BumpFunction.InnerProduct
import Mathlib.Analysis.SpecialFunctions.Trigonometric.InverseDeriv
import Mathlib.Analysis.Calculus.ContDiff.Deriv

/-!
# Globally smooth approximations of the logistic angle test

ArXiv:2604.01978v1, §7, `sec: sde.logistic`.
The source's angle is arcsin(u), correcting the printed sin(u)
as recorded in Logistic. Its derivative is singular at ±1,
so the full smooth-test martingale problem cannot test it directly.

For 0<a<1 a genuine smooth cutoff gives a globally smooth test
equal to arcsin(a u) on a neighborhood of the whole closed
state interval. The cutoff keeps its arcsin argument strictly
between -1 and 1 everywhere. It includes both absorbing endpoints.

The actual first and second derivatives give the original
logistic generator exactly. Its residual drift is nonzero
before taking a→1; it is not replaced by the desired zero drift.
The square identity retains the original diffusion contribution
2(1-u²)²(φ'(u))² needed for the angle's quadratic variation.
The concrete smooth test below has nonzero first derivative
and strictly negative residual drift at an interior state.
-/

open Real MeasureTheory Set Filter
open scoped Topology
namespace Transformer.Homogenized

/-- A genuine global smooth test agrees locally with the regularized angle
at every state, including ±1. Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`. -/
theorem exists_contDiff_arcsin_test (a : ℝ) (ha : a ∈ Ioo (0 : ℝ) 1) :
    ∃ φ : ℝ → ℝ, ContDiff ℝ (⊤ : ℕ∞) φ ∧
      ∀ u ∈ Icc (-1 : ℝ) 1, φ =ᶠ[𝓝 u] (fun v => Real.arcsin (a * v)) := by
  have hInv : 1 < a⁻¹ := (one_lt_inv₀ ha.1).mpr ha.2
  let χ : ContDiffBump (0 : ℝ) := ⟨(1 + a⁻¹) / 2, a⁻¹, by positivity, by linarith⟩
  let φ : ℝ → ℝ := fun u => Real.arcsin (a * χ u * u)
  have harg (u : ℝ) : |a * χ u * u| < 1 := by
    by_cases hu : |u| < a⁻¹
    · have hχ : |χ u| ≤ 1 := by rw [abs_of_nonneg χ.nonneg]; exact χ.le_one
      calc
        _ = a * |χ u| * |u| := by rw [abs_mul, abs_mul, abs_of_pos ha.1]
        _ ≤ a * 1 * |u| := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hχ ha.1.le) (abs_nonneg _)
        _ = a * |u| := by ring
        _ < a * a⁻¹ := mul_lt_mul_of_pos_left hu ha.1
        _ = 1 := mul_inv_cancel₀ ha.1.ne'
    · have hz : χ u = 0 := χ.zero_of_le_dist (by
        change a⁻¹ ≤ dist u 0
        simpa only [dist_zero_right, Real.norm_eq_abs] using le_of_not_gt hu)
      simp only [hz, mul_zero, zero_mul, abs_zero]
      exact zero_lt_one
  refine ⟨φ, ?_, ?_⟩
  · apply contDiff_iff_contDiffAt.mpr
    intro u
    have h := abs_lt.mp (harg u)
    exact (Real.contDiffAt_arcsin (by linarith) (by linarith)).comp u
      ((contDiff_const.mul χ.contDiff).mul contDiff_id).contDiffAt
  · intro u hu
    have hball : u ∈ Metric.ball (0 : ℝ) χ.rIn := by
      rw [Metric.mem_ball, dist_zero_right, Real.norm_eq_abs]
      change |u| < (1 + a⁻¹) / 2
      have habs : |u| ≤ 1 := abs_le.mpr hu
      linarith
    filter_upwards [χ.eventuallyEq_one_of_mem_ball hball] with v hv
    change Real.arcsin (a * χ v * v) = Real.arcsin (a * v)
    simp only [hv, Pi.one_apply, mul_one]

/-- The regularized angle's actual first derivative.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, the angle substitution. -/
theorem hasDerivAt_arcsin_mul (a u : ℝ) (hu : |a * u| < 1) :
    HasDerivAt (fun v => Real.arcsin (a * v)) (a / Real.sqrt (1 - (a * u) ^ 2)) u := by
  have h := abs_lt.mp hu
  have hinner : HasDerivAt (fun v : ℝ => a * v) a u := by
    simpa only [id_eq, mul_one] using (hasDerivAt_id u).const_mul a
  have hd := (Real.hasDerivAt_arcsin (x := a * u) (by linarith [h.1])
    (by linarith [h.2])).comp u hinner
  change HasDerivAt (fun v => Real.arcsin (a * v))
    ((1 / Real.sqrt (1 - (a * u) ^ 2)) * a) u at hd
  apply hd.congr_deriv
  ring

/-- Differentiating the actual first derivative gives the regularized curvature.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, the Itô correction. -/
theorem hasDerivAt_arcsin_mul_derivative (a u : ℝ) (hu : |a * u| < 1) :
    HasDerivAt (fun v => a / Real.sqrt (1 - (a * v) ^ 2))
      (a ^ 3 * u / Real.sqrt (1 - (a * u) ^ 2) ^ 3) u := by
  have hH : 0 < 1 - (a * u) ^ 2 := by nlinarith [abs_lt.mp hu, sq_abs (a * u)]
  have hs : Real.sqrt (1 - (a * u) ^ 2) ≠ 0 := (Real.sqrt_pos.mpr hH).ne'
  have hinner : HasDerivAt (fun v : ℝ => 1 - (a * v) ^ 2) (-(2 * a ^ 2 * u)) u := by
    convert (((hasDerivAt_id u).const_mul a).pow 2).const_sub 1 using 1
    all_goals dsimp
    ring
  have h := (hasDerivAt_const u a).div (hinner.sqrt hH.ne') hs
  apply h.congr_deriv
  simp only [zero_mul, zero_sub]
  field_simp [hs]

/-- The original generator of the actual local regularized angle test.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`; the residual drift is retained. -/
theorem logisticGenerator_arcsin_local (a u : ℝ) (hu : |a * u| < 1) (φ : ℝ → ℝ)
    (hφ : φ =ᶠ[𝓝 u] (fun v => Real.arcsin (a * v))) :
    logisticGenerator φ u =
      -(a * (1 - a ^ 2) * u * (1 - u ^ 2)) / Real.sqrt (1 - (a * u) ^ 2) ^ 3 := by
  have hH : 0 < 1 - (a * u) ^ 2 := by nlinarith [abs_lt.mp hu, sq_abs (a * u)]
  have hs : Real.sqrt (1 - (a * u) ^ 2) ≠ 0 := (Real.sqrt_pos.mpr hH).ne'
  have hnear : ∀ᶠ v in 𝓝 u, |a * v| < 1 :=
    ((continuous_const.mul continuous_id).abs.continuousAt).eventually (gt_mem_nhds hu)
  have hD : deriv (fun v => Real.arcsin (a * v)) =ᶠ[𝓝 u]
      (fun v => a / Real.sqrt (1 - (a * v) ^ 2)) := by
    filter_upwards [hnear] with v hv
    exact (hasDerivAt_arcsin_mul a v hv).deriv
  have hDD := ((hasDerivAt_arcsin_mul_derivative a u hu).congr_of_eventuallyEq hD).deriv
  rw [logisticGenerator, hφ.deriv_eq, hφ.deriv.deriv_eq,
    (hasDerivAt_arcsin_mul a u hu).deriv, hDD]
  have hfrac : a / Real.sqrt (1 - (a * u) ^ 2) =
      a * (1 - (a * u) ^ 2) / Real.sqrt (1 - (a * u) ^ 2) ^ 3 := by
    have h3 : Real.sqrt (1 - (a * u) ^ 2) ^ 3 =
        (1 - (a * u) ^ 2) * Real.sqrt (1 - (a * u) ^ 2) := by
      rw [pow_succ _ 2, Real.sq_sqrt hH.le]
    apply (div_eq_div_iff hs (pow_ne_zero 3 hs)).mpr
    rw [h3]
    ring
  rw [hfrac]
  ring

/-- Squaring an actual smooth test gives its exact diffusion contribution.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, the angle's quadratic variation. -/
theorem logisticGenerator_square (φ : ℝ → ℝ) (hφ : ContDiff ℝ (⊤ : ℕ∞) φ) (u : ℝ) :
    logisticGenerator (fun v => φ v ^ 2) u = 2 * φ u * logisticGenerator φ u +
      2 * (1 - u ^ 2) ^ 2 * deriv φ u ^ 2 := by
  have hd (v : ℝ) : HasDerivAt φ (deriv φ v) v :=
    (hφ.differentiable (by decide) v).hasDerivAt
  have hdd (v : ℝ) : HasDerivAt (deriv φ) (deriv (deriv φ) v) v :=
    ((contDiff_infty_iff_deriv.mp hφ).2.differentiable (by decide) v).hasDerivAt
  have hsq : deriv (fun v => φ v ^ 2) = fun v => 2 * (φ v * deriv φ v) := by
    funext v
    rw [((hd v).fun_pow 2).deriv]
    simp only [Nat.cast_ofNat, Nat.reduceSub, pow_one]
    ring
  have h2 := ((hd u).mul (hdd u)).const_mul 2
  change HasDerivAt (fun v => 2 * (φ v * deriv φ v))
    (2 * (deriv φ u * deriv φ u + φ u * deriv (deriv φ) u)) u at h2
  simp only [logisticGenerator, hsq, h2.deriv]
  ring

/-- Smooth source tests have continuous actual generator densities.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, the drift compensators. -/
theorem continuous_logisticGenerator (φ : ℝ → ℝ) (hφ : ContDiff ℝ (⊤ : ℕ∞) φ) :
    Continuous (logisticGenerator φ) := by
  have hd : Continuous (deriv φ) := hφ.continuous_deriv (by simp)
  have hdd : Continuous (deriv (deriv φ)) :=
    (contDiff_infty_iff_deriv.mp hφ).2.continuous_deriv (by simp)
  unfold logisticGenerator
  fun_prop

/-- The constructed a=1/2 test realizes all smoothness and local-agreement
premises. Its derivative and residual drift are genuinely nonzero at u=1/2. -/
example : ∃ φ : ℝ → ℝ, ContDiff ℝ (⊤ : ℕ∞) φ ∧
    (∀ u ∈ Icc (-1 : ℝ) 1, φ =ᶠ[𝓝 u] (fun v => Real.arcsin ((1 / 2 : ℝ) * v))) ∧
    |(1 / 2 : ℝ) * (1 / 2)| < 1 ∧
    logisticGenerator φ (1 / 2) < 0 ∧ 0 < deriv φ (1 / 2) := by
  obtain ⟨φ, hφ, heq⟩ := exists_contDiff_arcsin_test (1 / 2) (by norm_num)
  have hu : (1 / 2 : ℝ) ∈ Icc (-1 : ℝ) 1 := by norm_num
  have hau : |(1 / 2 : ℝ) * (1 / 2)| < 1 := by norm_num
  refine ⟨φ, hφ, heq, hau, ?_, ?_⟩
  · rw [logisticGenerator_arcsin_local _ _ hau φ (heq _ hu)]
    exact div_neg_of_neg_of_pos (by norm_num) (by positivity)
  · rw [(heq _ hu).deriv_eq, (hasDerivAt_arcsin_mul _ _ hau).deriv]
    positivity

end Transformer.Homogenized
