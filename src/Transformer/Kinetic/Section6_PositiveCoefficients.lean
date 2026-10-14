/-
# Kinetic theory for Transformers — the eigenvalues `a_n` are positive

The proof of `thm:U-shape` of arXiv:2605.09213v1 takes every `c_n = a_n t` positive: its
convexity step assumes `0 < c ≤ 3 - √3`, and its slope step asks `h_c'(0) < 0` of every mode.
For `β > 0` and `n ≥ 1` it is so: `ŵ_β(n) > 0` (`wHat_pos`), whence `a_n = n² ŵ_β(n) > 0`
(`aCoeff_pos`).  Expanding `e^{β cos θ} = Σ_k β^k cos^k θ / k!`, the coefficient is a series of
the integrals `∫₀^{2π} cos(mθ) cos^k θ dθ`; these are nonnegative, by the product formula
`2 cos(mθ) cos θ = cos((m+1)θ) + cos((m-1)θ)` and induction on `k`, and the one at `m = k` is
positive.
-/

import Transformer.Kinetic.Defs
import Mathlib.MeasureTheory.Integral.DominatedConvergence
import Mathlib.Analysis.SpecialFunctions.Exponential

open scoped BigOperators
open Real

namespace Transformer
namespace Kinetic

/-- `2 cos(mθ) cos θ = cos((m+1)θ) + cos((m-1)θ)`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem two_mul_cos_mul_cos (m θ : ℝ) :
    2 * (Real.cos (m * θ) * Real.cos θ) = Real.cos ((m + 1) * θ) + Real.cos ((m - 1) * θ) := by
  rw [add_mul, sub_mul, one_mul, Real.cos_add, Real.cos_sub]
  ring

/-- `∫₀^{2π} cos(mθ) dθ ≥ 0` for an integer `m`: it is `2π` at `m = 0` and `0` otherwise.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem integral_cos_int_mul_nonneg (m : ℤ) :
    0 ≤ ∫ θ in (0 : ℝ)..2 * π, Real.cos (m * θ) := by
  rcases eq_or_ne m 0 with rfl | hm
  · simp only [Int.cast_zero, zero_mul, Real.cos_zero, intervalIntegral.integral_const,
      sub_zero, smul_eq_mul, mul_one]
    positivity
  · have hm' : (m : ℝ) ≠ 0 := Int.cast_ne_zero.2 hm
    have hs : Real.sin (m * (2 * π)) = 0 := by
      rw [show (m : ℝ) * (2 * π) = ((2 * m : ℤ) : ℝ) * π by push_cast; ring]
      exact Real.sin_int_mul_pi _
    rw [intervalIntegral.integral_comp_mul_left (fun x => Real.cos x) hm', integral_cos, hs]
    simp

/-- The recursion behind the sign: `∫ cos(mθ) cos^{k+1} θ` is the mean of
`∫ cos((m+1)θ) cos^k θ` and `∫ cos((m-1)θ) cos^k θ`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem integral_cos_mul_cos_pow_succ (m : ℝ) (k : ℕ) :
    ∫ θ in (0 : ℝ)..2 * π, Real.cos (m * θ) * Real.cos θ ^ (k + 1) =
      ((∫ θ in (0 : ℝ)..2 * π, Real.cos ((m + 1) * θ) * Real.cos θ ^ k) +
        ∫ θ in (0 : ℝ)..2 * π, Real.cos ((m - 1) * θ) * Real.cos θ ^ k) / 2 := by
  have h : ∀ θ, Real.cos (m * θ) * Real.cos θ ^ (k + 1) =
      (Real.cos ((m + 1) * θ) * Real.cos θ ^ k + Real.cos ((m - 1) * θ) * Real.cos θ ^ k) / 2 :=
    fun θ => by
      rw [pow_succ]
      linear_combination (Real.cos θ ^ k / 2) * two_mul_cos_mul_cos m θ
  simp_rw [h]
  rw [intervalIntegral.integral_div, intervalIntegral.integral_add
    ((by fun_prop : Continuous fun θ => Real.cos ((m + 1) * θ) * Real.cos θ ^ k).intervalIntegrable
      _ _)
    ((by fun_prop : Continuous fun θ => Real.cos ((m - 1) * θ) * Real.cos θ ^ k).intervalIntegrable
      _ _)]

/-- `∫₀^{2π} cos(mθ) cos^k θ dθ ≥ 0` for every integer `m`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem integral_cos_mul_cos_pow_nonneg (k : ℕ) :
    ∀ m : ℤ, 0 ≤ ∫ θ in (0 : ℝ)..2 * π, Real.cos (m * θ) * Real.cos θ ^ k := by
  induction k with
  | zero => intro m; simpa using integral_cos_int_mul_nonneg m
  | succ k ih =>
    intro m
    rw [integral_cos_mul_cos_pow_succ]
    have h₁ := ih (m + 1)
    have h₂ := ih (m - 1)
    push_cast at h₁ h₂
    positivity

/-- `∫₀^{2π} cos(nθ) cos^n θ dθ > 0`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem integral_cos_mul_cos_pow_self_pos (n : ℕ) :
    0 < ∫ θ in (0 : ℝ)..2 * π, Real.cos (n * θ) * Real.cos θ ^ n := by
  induction n with
  | zero =>
    simp only [CharP.cast_eq_zero, zero_mul, Real.cos_zero, pow_zero, mul_one,
      intervalIntegral.integral_const, sub_zero, smul_eq_mul]
    positivity
  | succ n ih =>
    rw [integral_cos_mul_cos_pow_succ]
    have h := integral_cos_mul_cos_pow_nonneg n (n + 2)
    push_cast at h ⊢
    rw [add_sub_cancel_right, show (n : ℝ) + 1 + 1 = n + 2 by ring]
    positivity

/-- `2π ŵ_β(n) = Σ_k β^k/k! ∫₀^{2π} cos(nθ) cos^k θ dθ`, termwise from
`e^{β cos θ} = Σ_k (β cos θ)^k / k!`, by dominated convergence with the bound `|β|^k/k!`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem hasSum_integral_cos_mul_wBeta (β : ℝ) (n : ℕ) :
    HasSum (fun k : ℕ => β ^ k / k.factorial *
        ∫ θ in (0 : ℝ)..2 * π, Real.cos (n * θ) * Real.cos θ ^ k)
      (∫ θ in (0 : ℝ)..2 * π, Real.cos (n * θ) * wBeta β θ) := by
  have h := intervalIntegral.hasSum_integral_of_dominated_convergence
    (μ := MeasureTheory.volume) (a := 0) (b := 2 * π)
    (F := fun (k : ℕ) θ => Real.cos (n * θ) * ((β * Real.cos θ) ^ k / k.factorial))
    (f := fun θ => Real.cos (n * θ) * wBeta β θ) (fun k _ => |β| ^ k / k.factorial)
    (fun k => (by fun_prop : Continuous fun θ =>
      Real.cos (n * θ) * ((β * Real.cos θ) ^ k / k.factorial)).aestronglyMeasurable)
    (fun k => Filter.Eventually.of_forall fun θ _ => by
      rw [Real.norm_eq_abs, abs_mul, abs_div, abs_pow, abs_mul, Nat.abs_cast]
      have h₁ := Real.abs_cos_le_one (n * θ)
      have h₂ := Real.abs_cos_le_one θ
      calc |Real.cos (n * θ)| * ((|β| * |Real.cos θ|) ^ k / k.factorial)
          ≤ 1 * ((|β| * 1) ^ k / k.factorial) := by gcongr
        _ = |β| ^ k / k.factorial := by ring)
    (Filter.Eventually.of_forall fun _ _ => Real.summable_pow_div_factorial |β|)
    intervalIntegrable_const
    (Filter.Eventually.of_forall fun θ _ => by
      have := NormedSpace.expSeries_div_hasSum_exp (β * Real.cos θ)
      rw [← Real.exp_eq_exp_ℝ] at this
      exact this.mul_left (Real.cos (n * θ)))
  convert h using 1
  ext k
  rw [← intervalIntegral.integral_const_mul]
  congr 1
  ext θ
  rw [mul_pow]
  ring

/-- `ŵ_β(n) > 0` for `β > 0`: every term of `hasSum_integral_cos_mul_wBeta` is nonnegative, and
the one at `k = n` is positive.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem wHat_pos {β : ℝ} (hβ : 0 < β) (n : ℕ) : 0 < wHat β n := by
  have hle := le_hasSum (hasSum_integral_cos_mul_wBeta β n) n fun k _ =>
    mul_nonneg (by positivity) (by simpa using integral_cos_mul_cos_pow_nonneg k n)
  have hpos := mul_pos (by positivity : 0 < β ^ n / n.factorial)
    (integral_cos_mul_cos_pow_self_pos n)
  exact mul_pos (by positivity) (hpos.trans_le hle)

/-- The hypothesis of `wHat_pos` is satisfiable: `β = 1`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
example : 0 < wHat 1 1 := wHat_pos one_pos 1

/-- `a_n = n² ŵ_β(n) > 0` for `β > 0` and `n ≥ 1`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem aCoeff_pos {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 1 ≤ n) : 0 < aCoeff β n := by
  have : (0 : ℝ) < n := by exact_mod_cast hn
  exact mul_pos (by positivity) (wHat_pos hβ n)

/-- The hypotheses of `aCoeff_pos` are satisfiable: `β = 1`, `n = 1`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
example : 0 < aCoeff 1 1 := aCoeff_pos one_pos le_rfl

end Kinetic
end Transformer
