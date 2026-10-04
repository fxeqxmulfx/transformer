/-
# The sigmoid approximation of the error function

arXiv:2405.14578, §2.1, eq. (10), and Appendix C.  To simplify Theorems 3 to 5, the paper
replaces `𝓔(B) = erf(√(B/2)μ/σ)` by `(μ/σ)/√(π/(2B) + (μ/σ)²)` (`signApprox`), a function of the
same `x = √(B/2)μ/σ`: it is `x/√(π/4 + x²)` (`signApprox_eq`, `erfApprox`).  The "≈" of eq. (10)
is not an equality (`signApprox_lt_signMean`), but the ratio of the two sides tends to `1` as
`B → 0`, the regime of Theorems 3 and 5 (`tendsto_signApprox_div_signMean_zero`), and as `B → ∞`,
the regime of Theorem 4 (`tendsto_signApprox_div_signMean_atTop`): both functions of `x` are odd,
have the slope `2/√π` at `x = 0` and tend to `1` as `x → ∞`.
-/

import Transformer.Surge.Section2_Theorem2

open Filter Topology Real

namespace Transformer.Surge

open Transformer.BatchSize

/-- The approximation `(μ/σ)/√(π/(2B) + (μ/σ)²)` of `𝓔(B)` in eq. (10). -/
noncomputable def signApprox (μ σ B : ℝ) : ℝ := μ / σ / √(π / (2 * B) + (μ / σ) ^ 2)

/-- The approximation `x/√(π/4 + x²)` of `erf(x)`. -/
noncomputable def erfApprox (x : ℝ) : ℝ := x / √(π / 4 + x ^ 2)

/-- Eq. (10) is a function of `x = √(B/2)μ/σ`, as `𝓔(B) = erf(x)` is: it is `x/√(π/4 + x²)`. -/
theorem signApprox_eq (μ σ : ℝ) {B : ℝ} (hB : 0 < B) :
    signApprox μ σ B = erfApprox (√(B / 2) * μ / σ) := by
  have hs : 0 < √(B / 2) := Real.sqrt_pos.2 (by positivity)
  have hπ : B / 2 * (π / (2 * B)) = π / 4 := by
    field_simp
    ring
  have h : π / 4 + (√(B / 2) * μ / σ) ^ 2 = B / 2 * (π / (2 * B) + (μ / σ) ^ 2) := by
    rw [mul_div_assoc, mul_pow, Real.sq_sqrt (by positivity), mul_add, hπ]
  rw [erfApprox, h, Real.sqrt_mul (by positivity), signApprox, mul_div_assoc,
    mul_div_mul_left _ _ hs.ne']

/-- The hypothesis of `signApprox_eq` is satisfiable: `B = 1`. -/
example (μ σ : ℝ) := signApprox_eq μ σ one_pos

/-- `(x/√(π/4 + x²))/erf(x) → 1` as `x → 0`: both have the slope `2/√π` at `0`. -/
theorem tendsto_erfApprox_div_zero :
    Tendsto (fun x => erfApprox x / errorFunction x) (𝓝[≠] 0) (𝓝 1) := by
  have hs : Tendsto (fun x : ℝ => √(π / 4 + x ^ 2)) (𝓝[≠] 0) (𝓝 (√(π / 4 + 0 ^ 2))) :=
    tendsto_nhdsWithin_of_tendsto_nhds
      ((by fun_prop : Continuous fun x : ℝ => √(π / 4 + x ^ 2)).tendsto 0)
  have hv : 2 / √π * √(π / 4 + 0 ^ 2) = 1 := by
    rw [zero_pow two_ne_zero, add_zero, Real.sqrt_div' _ (by norm_num),
      show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
    field_simp
  have h := errorFunction_linearization.mul hs
  rw [hv] at h
  have h' := h.inv₀ one_ne_zero
  rw [inv_one] at h'
  refine Tendsto.congr (fun x => ?_) h'
  simp only [erfApprox, div_eq_mul_inv, mul_inv, inv_inv]
  ring

/-- `x/√(π/4 + x²) → 1` as `x → ∞`, as `erf(x)` does. -/
theorem tendsto_erfApprox_atTop : Tendsto erfApprox atTop (𝓝 1) := by
  have h : Tendsto (fun x : ℝ => 1 / √(1 + π / 4 / x ^ 2)) atTop (𝓝 (1 / √(1 + 0))) :=
    tendsto_const_nhds.div ((tendsto_const_nhds.add (tendsto_const_nhds.div_atTop
      (tendsto_pow_atTop two_ne_zero))).sqrt) (by simp)
  rw [add_zero, Real.sqrt_one, div_one] at h
  refine Tendsto.congr' ?_ h
  filter_upwards [eventually_gt_atTop 0] with x hx
  have hx0 := hx.ne'
  have hq : π / 4 + x ^ 2 = x ^ 2 * (1 + π / 4 / x ^ 2) := by
    field_simp
    ring
  rw [erfApprox, hq, Real.sqrt_mul (sq_nonneg x), Real.sqrt_sq hx.le, div_mul_eq_div_div,
    div_self hx0]

/-- `(x/√(π/4 + x²))/erf(x) → 1` as `x → ∞`. -/
theorem tendsto_erfApprox_div_atTop :
    Tendsto (fun x => erfApprox x / errorFunction x) atTop (𝓝 1) := by
  have h := tendsto_erfApprox_atTop.div errorFunction_tendsto_atTop one_ne_zero
  rw [div_one] at h
  exact h

/-- `(x/√(π/4 + x²))/erf(x)` is even. -/
theorem erfApprox_div_abs (x : ℝ) :
    erfApprox |x| / errorFunction |x| = erfApprox x / errorFunction x := by
  rcases abs_cases x with ⟨h, _⟩ | ⟨h, _⟩
  · rw [h]
  · rw [h, errorFunction_neg, erfApprox, erfApprox, neg_sq, neg_div, neg_div_neg_eq]

/-- **Eq. (10)**, as `B → 0`: `(μ/σ)/√(π/(2B) + (μ/σ)²)` over `𝓔(B)` tends to `1`. -/
theorem tendsto_signApprox_div_signMean_zero {μ σ : ℝ} (hμ : μ ≠ 0) (hσ : σ ≠ 0) :
    Tendsto (fun B => signApprox μ σ B / signMean μ σ B) (𝓝[>] 0) (𝓝 1) := by
  have hx : Tendsto (fun B : ℝ => √(B / 2) * μ / σ) (𝓝[>] 0) (𝓝[≠] 0) := by
    refine tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _ ?_ ?_
    · have h : Tendsto (fun B : ℝ => √(B / 2) * μ / σ) (𝓝 0) (𝓝 (√(0 / 2) * μ / σ)) :=
        (by fun_prop : Continuous fun B : ℝ => √(B / 2) * μ / σ).tendsto 0
      rw [zero_div, Real.sqrt_zero, zero_mul, zero_div] at h
      exact tendsto_nhdsWithin_of_tendsto_nhds h
    · filter_upwards [self_mem_nhdsWithin] with B hB
      exact div_ne_zero (mul_ne_zero (Real.sqrt_pos.2 (half_pos hB)).ne' hμ) hσ
  refine Tendsto.congr' ?_ (tendsto_erfApprox_div_zero.comp hx)
  filter_upwards [self_mem_nhdsWithin] with B hB
  rw [Function.comp_apply, signApprox_eq μ σ hB, signMean]

/-- The hypotheses of `tendsto_signApprox_div_signMean_zero` are satisfiable: `μ = σ = 1`. -/
example := tendsto_signApprox_div_signMean_zero one_ne_zero one_ne_zero

/-- **Eq. (10)**, as `B → ∞`: `(μ/σ)/√(π/(2B) + (μ/σ)²)` over `𝓔(B)` tends to `1`. -/
theorem tendsto_signApprox_div_signMean_atTop {μ σ : ℝ} (hμ : μ ≠ 0) (hσ : σ ≠ 0) :
    Tendsto (fun B => signApprox μ σ B / signMean μ σ B) atTop (𝓝 1) := by
  have hx : Tendsto (fun B : ℝ => |√(B / 2) * μ / σ|) atTop atTop := by
    refine Tendsto.congr (fun B => ?_) ((Real.tendsto_sqrt_atTop.comp
      (tendsto_id.atTop_div_const two_pos)).atTop_mul_const (abs_pos.2 (div_ne_zero hμ hσ)))
    rw [Function.comp_apply, id, mul_div_assoc, abs_mul, abs_of_nonneg (Real.sqrt_nonneg _)]
  refine Tendsto.congr' ?_ (tendsto_erfApprox_div_atTop.comp hx)
  filter_upwards [eventually_gt_atTop 0] with B hB
  rw [Function.comp_apply, erfApprox_div_abs, signApprox_eq μ σ hB, signMean]

/-- The hypotheses of `tendsto_signApprox_div_signMean_atTop` are satisfiable: `μ = σ = 1`. -/
example := tendsto_signApprox_div_signMean_atTop one_ne_zero one_ne_zero

/-- **Eq. (10)** is not an equality: at `B = 2`, `μ = σ = 1`, where `x = 1`,
`(μ/σ)/√(π/(2B) + (μ/σ)²) = 1/√(π/4 + 1) < 4/(3√π) ≤ erf(1) = 𝓔(B)`. -/
theorem signApprox_lt_signMean : signApprox 1 1 2 < signMean 1 1 2 := by
  have hint : (2 : ℝ) / 3 ≤ ∫ t in (0 : ℝ)..1, Real.exp (-(t ^ 2)) := by
    have h := intervalIntegral.integral_mono_on (μ := MeasureTheory.volume) zero_le_one
      ((by fun_prop : Continuous fun t : ℝ => 1 - t ^ 2).intervalIntegrable _ _)
      (continuous_error_integrand.intervalIntegrable _ _)
      fun t _ => by linarith [Real.add_one_le_exp (-(t ^ 2))]
    have h1 : ∫ t in (0 : ℝ)..1, (1 - t ^ 2) = 2 / 3 := by
      rw [intervalIntegral.integral_sub intervalIntegrable_const
        ((continuous_pow 2).intervalIntegrable _ _), integral_pow]
      norm_num
    linarith
  have hp := Real.sqrt_pos.2 Real.pi_pos
  have hq := Real.sqrt_pos.2 (show 0 < π / 4 + 1 by positivity)
  have h34 : 3 * √π < 4 * √(π / 4 + 1) := by
    nlinarith [Real.pi_lt_d2, Real.sq_sqrt Real.pi_pos.le,
      Real.sq_sqrt (show 0 ≤ π / 4 + 1 by positivity)]
  have ha : signApprox 1 1 2 = 1 / √(π / 4 + 1) := by norm_num [signApprox]
  have hm : signMean 1 1 2 = 2 / √π * ∫ t in (0 : ℝ)..1, Real.exp (-(t ^ 2)) := by
    simp [signMean, errorFunction]
  rw [ha, hm]
  calc 1 / √(π / 4 + 1) < 4 / (3 * √π) := by
        rw [div_lt_div_iff₀ hq (by positivity)]
        linarith
    _ = 2 / √π * (2 / 3) := by
        field_simp
        ring
    _ ≤ 2 / √π * ∫ t in (0 : ℝ)..1, Real.exp (-(t ^ 2)) := by gcongr

end Transformer.Surge
