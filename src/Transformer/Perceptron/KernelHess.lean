/-
# Perceptrons and attention's mean-field landscape — the two bounds on `K_β''`

The two Hessian bounds of `lem: concavity` of arXiv:2601.21366v2: inside a
fixed fraction `λ` of the window, `K_β''` is at most
`-e^{-λ²/2}((1-λ²)/2)βe^β`, and outside the window it is at most `2βe^{β-3/2}`.

**How they are proved here.**  The source goes through the monotonicity of
`K_β''` on `[0, θ_c]` and the location `cos θ_* = t_*` of its maximum on
`[θ_c, π]`.  Neither is needed:

* inside the window, `cos θ ≥ 1 - θ²/2`, `sin²θ ≤ θ²` and
  `β²θ_c² ≤ β + 3` (`sq_mul_sq_thetaC_mem`) bound the exponential below and the
  bracket above at every `|θ| ≤ λθ_c`, with `β₀(λ) = 12/(1-λ²) + 3`;
* with `x = β(1 - cos θ) ≥ 0`, `K_β''(θ) = e^{β-x}(β(2x-1) + x - x²)`, and
  `e^y ≥ 1 + y` gives `2x - 1 ≤ 2e^{x-3/2}` and `x ≤ e^{x-1}`; this bounds
  `K_β''` by `2βe^{β-3/2}` at **every** `θ` once `β ≥ 4`, in particular on
  `[θ_c, π]`, which is the statement.

Source: arXiv:2601.21366v2, `lem: concavity`, `eq:kernel-hess-bound-lambda`,
`eq:kernel-hess-bound-max`.
-/

import Transformer.Perceptron.KernelAsymp
import Mathlib.Analysis.Complex.ExponentialBounds

open Real

namespace Transformer
namespace Perceptron

/-- **`eq:kernel-hess-bound-lambda`.**  For every `λ ∈ (0,1)` there is `β₀(λ)`
with `K_β''(θ) ≤ -e^{-λ²/2}((1-λ²)/2) β e^β` for every `β ≥ β₀(λ)` and every
`|θ| ≤ λ θ_c(β)`.  Here `β₀(λ) = 12/(1-λ²) + 3`.

Source: arXiv:2601.21366v2, `eq:kernel-hess-bound-lambda`. -/
theorem kernelK2_le_of_abs_le_lambda_thetaC (lam : ℝ) (hlam : lam ∈ Set.Ioo (0 : ℝ) 1) :
    ∃ β₀ : ℝ, 0 < β₀ ∧ ∀ β : ℝ, β₀ ≤ β → ∀ θ : ℝ, |θ| ≤ lam * thetaC β →
      kernelK2 β θ ≤ -Real.exp (-lam ^ 2 / 2) * ((1 - lam ^ 2) / 2) * β * Real.exp β := by
  obtain ⟨hl0, hl1⟩ := hlam
  have ha : 0 < 1 - lam ^ 2 := by nlinarith
  refine ⟨12 / (1 - lam ^ 2) + 3, by positivity, fun β hβ θ hθ => ?_⟩
  have h12 : 0 ≤ 12 / (1 - lam ^ 2) := by positivity
  have hβ3 : 3 ≤ β := by linarith
  have hβ0 : 0 < β := by linarith
  have hβa : 12 ≤ β * (1 - lam ^ 2) := by
    have := (div_le_iff₀ ha).mp (show 12 / (1 - lam ^ 2) ≤ β by linarith)
    linarith
  obtain ⟨-, hup⟩ := sq_mul_sq_thetaC_mem β hβ0
  have hT0 := thetaC_pos β hβ0
  set T := thetaC β
  have hl2 : lam ^ 2 < 1 := by nlinarith
  have ht2 : θ ^ 2 ≤ lam ^ 2 * T ^ 2 := by
    rw [← mul_pow]; exact sq_le_sq' (abs_le.mp hθ).1 (abs_le.mp hθ).2
  have hcos : 1 - θ ^ 2 / 2 ≤ Real.cos θ := Real.one_sub_sq_div_two_le_cos
  have hsin : Real.sin θ ^ 2 ≤ θ ^ 2 := Real.sin_sq_le_sq
  have hlT : lam ^ 2 * (β ^ 2 * T ^ 2) ≤ lam ^ 2 * (β + 3) :=
    mul_le_mul_of_nonneg_left hup (sq_nonneg lam)
  have hβθ : β ^ 2 * θ ^ 2 ≤ lam ^ 2 * (β + 3) := by
    nlinarith [mul_le_mul_of_nonneg_left ht2 (sq_nonneg β)]
  -- the exponential, from below
  set ε := 3 * lam ^ 2 / (2 * β) with hεdef
  have hε : ε * β = 3 * lam ^ 2 / 2 := by rw [hεdef]; field_simp
  have harg : β - lam ^ 2 / 2 - ε ≤ β * Real.cos θ := by
    refine le_of_mul_le_mul_left ?_ hβ0
    nlinarith [mul_le_mul_of_nonneg_left hcos (sq_nonneg β)]
  have hX := Real.exp_pos (β - lam ^ 2 / 2)
  have hE : Real.exp (β - lam ^ 2 / 2) * (1 - ε) ≤ Real.exp (β * Real.cos θ) := by
    calc Real.exp (β - lam ^ 2 / 2) * (1 - ε)
        ≤ Real.exp (β - lam ^ 2 / 2) * Real.exp (-ε) := by
          gcongr; linarith [Real.add_one_le_exp (-ε)]
      _ = Real.exp (β - lam ^ 2 / 2 - ε) := by rw [← Real.exp_add]; ring_nf
      _ ≤ Real.exp (β * Real.cos θ) := Real.exp_le_exp.mpr harg
  -- the bracket, from above
  have hB : β ^ 2 * Real.sin θ ^ 2 - β * Real.cos θ ≤ -(β * (1 - lam ^ 2)) + 4 := by
    have h1 : β * (β * θ ^ 2) ≤ β * 2 := by nlinarith
    have h2 : β * θ ^ 2 ≤ 2 := le_of_mul_le_mul_left h1 hβ0
    nlinarith [mul_le_mul_of_nonneg_left hsin (sq_nonneg β),
      mul_le_mul_of_nonneg_left hcos hβ0.le]
  -- the product
  have hB0 : -(β * (1 - lam ^ 2)) + 4 ≤ 0 := by linarith
  have hε2 : ε ≤ 1 / 2 := by nlinarith
  have hrhs : -Real.exp (-lam ^ 2 / 2) * ((1 - lam ^ 2) / 2) * β * Real.exp β
      = -(Real.exp (β - lam ^ 2 / 2) * (β * (1 - lam ^ 2) / 2)) := by
    rw [show β - lam ^ 2 / 2 = -lam ^ 2 / 2 + β by ring, Real.exp_add]; ring
  rw [kernelK2, hrhs]
  calc Real.exp (β * Real.cos θ) * (β ^ 2 * Real.sin θ ^ 2 - β * Real.cos θ)
      ≤ Real.exp (β * Real.cos θ) * (-(β * (1 - lam ^ 2)) + 4) :=
        mul_le_mul_of_nonneg_left hB (Real.exp_pos _).le
    _ ≤ Real.exp (β - lam ^ 2 / 2) * (1 - ε) * (-(β * (1 - lam ^ 2)) + 4) :=
        mul_le_mul_of_nonpos_right hE hB0
    _ ≤ -(Real.exp (β - lam ^ 2 / 2) * (β * (1 - lam ^ 2) / 2)) := by
        have hk : β * (1 - lam ^ 2) / 2 ≤ (1 - ε) * (β * (1 - lam ^ 2) - 4) := by
          nlinarith [mul_le_mul_of_nonneg_left hl2.le ha.le]
        nlinarith [mul_le_mul_of_nonneg_left hk hX.le]

/-- The hypothesis of `kernelK2_le_of_abs_le_lambda_thetaC` is satisfiable:
`λ = 1/2`. -/
example : (1 : ℝ) / 2 ∈ Set.Ioo (0 : ℝ) 1 := ⟨by norm_num, by norm_num⟩

/-- **`eq:kernel-hess-bound-max`.**  There is `β₁` with
`K_β''(θ) ≤ 2β e^{β - 3/2}` for every `β ≥ β₁` and every `θ ∈ [θ_c(β), π]`.
Here `β₁ = 4`, and the bound holds at every `θ`: inside the window
`K_β'' < 0` anyway (`kernelK2_neg`).

Source: arXiv:2601.21366v2, `eq:kernel-hess-bound-max`. -/
theorem kernelK2_le_of_thetaC_le :
    ∃ β₁ : ℝ, 0 < β₁ ∧ ∀ β : ℝ, β₁ ≤ β → ∀ θ ∈ Set.Icc (thetaC β) π,
      kernelK2 β θ ≤ 2 * β * Real.exp (β - 3 / 2) := by
  refine ⟨4, by norm_num, fun β hβ θ _ => ?_⟩
  set x := β * (1 - Real.cos θ) with hxdef
  have hx0 : 0 ≤ x := mul_nonneg (by linarith) (by linarith [Real.cos_le_one θ])
  have hc : β * Real.cos θ = β - x := by rw [hxdef]; ring
  have hP : β ^ 2 * Real.sin θ ^ 2 - β * Real.cos θ = β * (2 * x - 1) + x - x ^ 2 := by
    rw [Real.sin_sq, hxdef]; ring
  have hE := Real.exp_pos (β - x)
  have hy := Real.exp_pos (x - 3 / 2)
  have hsplit : Real.exp (β - 3 / 2) = Real.exp (β - x) * Real.exp (x - 3 / 2) := by
    rw [← Real.exp_add]; ring_nf
  rw [kernelK2, hP, hc, hsplit]
  suffices h : β * (2 * x - 1) + x - x ^ 2 ≤ 2 * β * Real.exp (x - 3 / 2) by
    nlinarith [mul_le_mul_of_nonneg_left h hE.le]
  have hy1 := Real.add_one_le_exp (x - 3 / 2)
  rcases le_or_gt 1 x with h1 | h1
  · nlinarith
  -- `x < 1`: `2x - 1 ≤ x ≤ e^{x-1} = e^{x-3/2}e^{1/2}`, and `β ≥ 4` absorbs `x - x² ≤ 1/4`
  have hh := Real.exp_pos (1 / 2)
  have hx1 : x ≤ Real.exp (x - 3 / 2) * Real.exp (1 / 2) := by
    rw [← Real.exp_add]; linarith [Real.add_one_le_exp (x - 3 / 2 + 1 / 2)]
  have he : Real.exp (1 / 2) ^ 2 < 2.7225 := by
    rw [← Real.exp_nat_mul]; norm_num; linarith [Real.exp_one_lt_d9]
  have hh1 : Real.exp (1 / 2) < 1.65 := by nlinarith
  have hye : 1 ≤ Real.exp (x - 3 / 2) * (Real.exp (1 / 2) ^ 2 * Real.exp (1 / 2)) := by
    rw [← Real.exp_nat_mul, ← Real.exp_add, ← Real.exp_add]
    exact Real.one_le_exp (by linarith)
  have hy4 : 1 ≤ Real.exp (x - 3 / 2) * 4.5 := by
    have : Real.exp (1 / 2) ^ 2 * Real.exp (1 / 2) ≤ 4.5 := by nlinarith
    nlinarith [mul_le_mul_of_nonneg_left this hy.le]
  nlinarith [mul_le_mul_of_nonneg_left hx1 (by linarith : (0 : ℝ) ≤ β),
    mul_le_mul_of_nonneg_left hh1.le hy.le, mul_nonneg (by linarith : (0 : ℝ) ≤ β - 4) hy.le]

end Perceptron
end Transformer
