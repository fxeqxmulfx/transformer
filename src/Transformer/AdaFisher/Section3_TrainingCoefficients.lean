/-
# AdaFisher — constants in the momentum-error Lyapunov estimate

arXiv:2405.16397v3, §3.4 and Appendix A.2, deterministic training extension.
The sufficient step condition is `2*L*eta <= damping*(1-beta)`.
It permits every beta in [0,1), including nonzero constant momentum.
-/

import Transformer.AdaFisher.Section3_TrainingBias

namespace Transformer.AdaFisher

/-- Weighted Young inequality used to absorb momentum's cross term into
loss descent. Source: arXiv:2405.16397v3, §3.4 and Appendix A.2,
deterministic training extension. -/
theorem training_young (η δ β U V : ℝ) (hη : 0 < η) (hδ : 0 < δ) :
    β * U * V ≤ δ / (4 * η) * V ^ 2 + η * β ^ 2 / δ * U ^ 2 := by
  have h := div_nonneg (sq_nonneg (δ * V - 2 * η * β * U))
    (show 0 ≤ 4 * η * δ by positivity)
  have heq : (δ * V - 2 * η * β * U) ^ 2 / (4 * η * δ) =
      δ / (4 * η) * V ^ 2 + η * β ^ 2 / δ * U ^ 2 - β * U * V := by
    field_simp
    ring
  rw [heq] at h
  linarith

/-- A concrete witness satisfies the preceding theorem's assumptions.
Source: arXiv:2405.16397v3, Algorithm 1 and §3.4, deterministic extension. -/
example : (0 : ℝ) < 1 / 40 ∧ (0 : ℝ) < 1 := by norm_num

/-- Squared lag error contracts with coefficient beta, paying only a
multiple of the squared actual displacement. Source: arXiv:2405.16397v3,
§3.4, Lipschitz-gradient assumption (i), deterministic extension. -/
theorem training_error_young (β L U V : ℝ) (hβ : 0 ≤ β) (hβ' : β < 1) :
    (β * U + L * V) ^ 2 ≤ β * U ^ 2 + L ^ 2 / (1 - β) * V ^ 2 := by
  have hden : 0 < 1 - β := sub_pos.mpr hβ'
  have h := div_nonneg (mul_nonneg hβ (sq_nonneg ((1 - β) * U - L * V))) hden.le
  have heq : β * ((1 - β) * U - L * V) ^ 2 / (1 - β) =
      β * U ^ 2 + L ^ 2 / (1 - β) * V ^ 2 - (β * U + L * V) ^ 2 := by
    field_simp [hden.ne']
    ring
  rw [heq] at h
  linarith

/-- A concrete witness satisfies the preceding theorem's assumptions.
Source: arXiv:2405.16397v3, Algorithm 1 and §3.4, deterministic extension. -/
example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 := by norm_num

/-- The sufficient step condition simultaneously absorbs smoothness and
the new momentum error into actual parameter descent. Source:
arXiv:2405.16397v3, §3.4 and Appendix A.2, explicit constant-step correction. -/
theorem training_coefficient_bounds (η β δ L : ℝ)
    (hη : 0 < η) (hβ : 0 ≤ β) (hβ' : β < 1) (hδ : 0 < δ) (hL : 0 < L)
    (hstep : 2 * L * η ≤ δ * (1 - β)) :
    L / 2 ≤ δ / (4 * η) ∧
      η / (δ * (1 - β)) * (L ^ 2 / (1 - β)) ≤ δ / (4 * η) := by
  have hden : 0 < 1 - β := sub_pos.mpr hβ'
  have hsq := pow_le_pow_left₀ (by positivity : 0 ≤ 2 * L * η) hstep 2
  constructor
  · apply (le_div_iff₀ (by positivity : 0 < 4 * η)).mpr
    have h := mul_nonneg hδ.le hβ
    nlinarith
  · have heq : η / (δ * (1 - β)) * (L ^ 2 / (1 - β)) =
        η * L ^ 2 / (δ * (1 - β) ^ 2) := by
      field_simp [hden.ne']
    rw [heq]
    apply (div_le_div_iff₀ (by positivity : 0 < δ * (1 - β) ^ 2)
      (by positivity : 0 < 4 * η)).mpr
    nlinarith only [hsq]

/-- The sufficient step condition has a nonzero-momentum domain,
arXiv:2405.16397v3, §3.4, deterministic training extension. -/
example : (0 : ℝ) < 1 / 40 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ 2 * (1 : ℝ) * (1 / 40) ≤ 1 * (1 - 9 / 10) := by
  norm_num

end Transformer.AdaFisher
