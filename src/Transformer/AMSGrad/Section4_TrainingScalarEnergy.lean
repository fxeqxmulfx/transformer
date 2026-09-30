/-
# AMSGrad — the coordinate Lyapunov inequality

Training extension of arXiv:1904.03590v4, Algorithm 1 and §4.
The key inequality uses the actual increasing denominator and the
momentum balance. It does not assume alignment of momentum with the
current gradient, and it needs no gradient-replacement safeguard.
-/

import Transformer.AMSGrad.Section4_TrainingModels

namespace Transformer.AMSGrad

/-- The exact momentum balance and monotone positive metric imply a
coordinate Lyapunov decrease before applying the loss's smooth upper
model. Source: training extension of arXiv:1904.03590v4, Algorithm 1, §4. -/
theorem scalar_momentum_energy (η β a b u v g : ℝ)
    (hη : 0 < η) (hβ : 0 ≤ β) (hβ' : β < 1) (ha : 0 ≤ a) (hab : a ≤ b)
    (hbalance : b * v = β * a * u - η * (1 - β) * g) :
    g * v + β / (2 * η * (1 - β)) * b * v ^ 2 -
        β / (2 * η * (1 - β)) * a * u ^ 2 ≤ -b * v ^ 2 / η := by
  have hden : 0 < 2 * η * (1 - β) := by positivity
  have hbne : 1 - β ≠ 0 := (sub_pos.mpr hβ').ne'
  have hYoung : 2 * a * u * v ≤ a * u ^ 2 + a * v ^ 2 := by
    nlinarith [mul_nonneg ha (sq_nonneg (u - v))]
  have hY := mul_le_mul_of_nonneg_left hYoung hβ
  have hV := mul_le_mul_of_nonneg_left
    (mul_le_mul_of_nonneg_right hab (sq_nonneg v)) hβ
  have hbal := congrArg (fun z : ℝ => z * v) hbalance
  have hpoly : 2 * η * (1 - β) * (g * v) + β * b * v ^ 2 - β * a * u ^ 2 ≤
      -2 * (1 - β) * b * v ^ 2 := by nlinarith
  calc
    _ = (2 * η * (1 - β) * (g * v) + β * b * v ^ 2 - β * a * u ^ 2) /
        (2 * η * (1 - β)) := by field_simp [hbne]
    _ ≤ (-2 * (1 - β) * b * v ^ 2) / (2 * η * (1 - β)) :=
      div_le_div_of_nonneg_right hpoly hden.le
    _ = _ := by field_simp [hbne]

/-- The balance is satisfiable with a nonzero true gradient and nonzero
new displacement, arXiv:1904.03590v4, Algorithm 1, training extension. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧
    (1 : ℝ) * (-1 / 2) = (1 / 2) * 1 * 0 - 1 * (1 - 1 / 2) * 1 := by norm_num

end Transformer.AMSGrad
