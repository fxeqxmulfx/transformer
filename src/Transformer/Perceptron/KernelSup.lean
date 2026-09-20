/-
# Perceptrons and attention's mean-field landscape — the sup-norm of `K_β''`

Formalization of the computation `‖K_β''‖_∞ = β e^β` of
`rem:strictSOPD-perceptron` of arXiv:2601.21366v2: the constant the sufficient
condition for strict second-order positive-definiteness is written in.

**What the source says and what is carried here.**

* The source states the value of the sup-norm without proof.  `‖·‖_∞` over `ℝ`
  is carried as `IsGreatest (Set.range fun θ => |K_β''(θ)|) (β e^β)`: the value
  is an upper bound *and* it is attained — at `θ = 0`, where
  `K_β''(0) = -β e^β`.  A supremum that is attained leaves no room for a junk
  value, which the bare `sSup` of an unbounded family would.

* The bound itself is `|e^{β cos θ}(β²sin²θ - β cos θ)| ≤ β e^β`, which after
  dividing by `β > 0` and writing `c = cos θ` is
  `|e^{βc}(β(1 - c²) - c)| ≤ e^β`.  Below `-1 ≤ β(1-c²) - c` gives the lower
  bound, and the upper bound splits at `c = 0`: for `c ≤ 0` the exponential is
  `≤ 1` and the bracket is `≤ β + 1 ≤ e^β`; for `c > 0` the bracket is
  `≤ 2β(1-c)` and `e^{βc} = e^β / e^{β(1-c)}`, so the product is at most
  `e^β` by `2u ≤ e^u`.

Source: arXiv:2601.21366v2, `rem:strictSOPD-perceptron`.
-/

import Transformer.Perceptron.Kernel
import Mathlib.Analysis.Complex.ExponentialBounds

open Real

namespace Transformer
namespace Perceptron

/-- `2u ≤ e^u` for `u ≥ 0`: `e^u = e · e^{u-1} ≥ 2 · u`, using `1 + s ≤ e^s` at
`s = u - 1` and `2 < e`. -/
theorem two_mul_le_exp {u : ℝ} (hu : 0 ≤ u) : 2 * u ≤ Real.exp u := by
  have h1 : u ≤ Real.exp (u - 1) := by
    have := Real.add_one_le_exp (u - 1)
    linarith
  have he : (2 : ℝ) ≤ Real.exp 1 := Real.exp_one_gt_two.le
  calc 2 * u ≤ 2 * Real.exp (u - 1) := by linarith
    _ ≤ Real.exp 1 * Real.exp (u - 1) := by nlinarith [Real.exp_pos (u - 1)]
    _ = Real.exp u := by rw [← Real.exp_add]; congr 1; ring

/-- `|K_β''(θ)| ≤ β e^β` for every angle: the upper bound half of
`‖K_β''‖_∞ = β e^β`.

Source: arXiv:2601.21366v2, `rem:strictSOPD-perceptron`. -/
theorem abs_kernelK2_le (β : ℝ) (hβ : 0 < β) (θ : ℝ) : |kernelK2 β θ| ≤ β * Real.exp β := by
  have hc1 : Real.cos θ ≤ 1 := Real.cos_le_one θ
  have hc2 : -1 ≤ Real.cos θ := Real.neg_one_le_cos θ
  have hsin : Real.sin θ ^ 2 = 1 - Real.cos θ ^ 2 := Real.sin_sq θ
  have hE : (0 : ℝ) < Real.exp (β * Real.cos θ) := Real.exp_pos _
  have hkey : |Real.exp (β * Real.cos θ) * (β * (1 - Real.cos θ ^ 2) - Real.cos θ)|
      ≤ Real.exp β := by
    rw [abs_le]
    constructor
    · have hX : -1 ≤ β * (1 - Real.cos θ ^ 2) - Real.cos θ := by nlinarith
      nlinarith [mul_le_mul_of_nonneg_left hX hE.le,
        Real.exp_le_exp.mpr (show β * Real.cos θ ≤ β by nlinarith)]
    · rcases le_or_gt (Real.cos θ) 0 with hc | hc
      · have hE1 : Real.exp (β * Real.cos θ) ≤ 1 := Real.exp_le_one_iff.mpr (by nlinarith)
        have hX0 : 0 ≤ β * (1 - Real.cos θ ^ 2) - Real.cos θ := by nlinarith
        have hXle : β * (1 - Real.cos θ ^ 2) - Real.cos θ ≤ β + 1 := by nlinarith
        have := Real.add_one_le_exp β
        nlinarith [mul_le_mul hE1 hXle hX0 zero_le_one]
      · have hu : 0 ≤ β * (1 - Real.cos θ) := by nlinarith
        have hXle : β * (1 - Real.cos θ ^ 2) - Real.cos θ ≤ 2 * (β * (1 - Real.cos θ)) := by
          nlinarith
        have hEeq : Real.exp (β * Real.cos θ)
            = Real.exp β / Real.exp (β * (1 - Real.cos θ)) := by
          rw [← Real.exp_sub]; congr 1; ring
        have h2u := two_mul_le_exp hu
        calc Real.exp (β * Real.cos θ) * (β * (1 - Real.cos θ ^ 2) - Real.cos θ)
            ≤ Real.exp (β * Real.cos θ) * (2 * (β * (1 - Real.cos θ))) :=
              mul_le_mul_of_nonneg_left hXle hE.le
          _ = (2 * (β * (1 - Real.cos θ))) * (Real.exp β / Real.exp (β * (1 - Real.cos θ))) := by
              rw [hEeq]; ring
          _ ≤ Real.exp (β * (1 - Real.cos θ)) * (Real.exp β / Real.exp (β * (1 - Real.cos θ))) := by
              refine mul_le_mul_of_nonneg_right h2u ?_
              positivity
          _ = Real.exp β := by
              field_simp
  have hfac : kernelK2 β θ
      = β * (Real.exp (β * Real.cos θ) * (β * (1 - Real.cos θ ^ 2) - Real.cos θ)) := by
    rw [kernelK2, hsin]; ring
  rw [hfac, abs_mul, abs_of_pos hβ]
  exact mul_le_mul_of_nonneg_left hkey hβ.le

/-- The bound `β e^β` is attained: `K_β''(0) = -β e^β`. -/
theorem abs_kernelK2_zero (β : ℝ) (hβ : 0 < β) : |kernelK2 β 0| = β * Real.exp β := by
  have h : kernelK2 β 0 = -(β * Real.exp β) := by
    rw [kernelK2, Real.cos_zero, Real.sin_zero, mul_one]; ring
  rw [h, abs_neg, abs_of_pos (by positivity)]

/-- **`‖K_β''‖_∞ = β e^β`.**  For `β > 0` the greatest value of `|K_β''|` is
`β e^β`, attained at `θ = 0`.

Source: arXiv:2601.21366v2, `rem:strictSOPD-perceptron`. -/
theorem isGreatest_abs_kernelK2 (β : ℝ) (hβ : 0 < β) :
    IsGreatest (Set.range fun θ : ℝ => |kernelK2 β θ|) (β * Real.exp β) :=
  ⟨⟨0, abs_kernelK2_zero β hβ⟩, by
    rintro _ ⟨θ, rfl⟩
    exact abs_kernelK2_le β hβ θ⟩

/-- The hypothesis of `isGreatest_abs_kernelK2` is satisfiable: `β = 1`. -/
example : (0 : ℝ) < 1 := one_pos

end Perceptron
end Transformer
