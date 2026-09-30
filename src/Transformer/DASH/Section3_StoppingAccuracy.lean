/-
# DASH — accuracy supplied by the CN identity-residual stopping rule

arXiv:2602.02016v2, §3.2, after `equation:CN-X-M`. The auxiliary
product gives a relative inverse-root error certificate. An absolute
error bound additionally contains the size of the true inverse root.
-/

import Transformer.DASH.Section3_CNConvergence

noncomputable section

namespace Transformer.DASH

/-- The actual CN inverse-root iterate is the true inverse root times
the positive `p`-th root of its product iterate. This identity, together
with the proved positivity domain, justifies interpreting `M_k≈1` as an
inverse-root accuracy certificate.
Source: arXiv:2602.02016v2, §3.2, the auxiliary-product stopping rule. -/
theorem cnScalarX_factor (p : ℕ) (a c : ℝ) (hp : 0 < p) (ha : 0 < a)
    (hc : 0 < c) (hupper : a < ((p : ℝ) + 1) * c ^ p) (k : ℕ) :
    cnScalarX p a c k =
      a ^ (-(1 / (p : ℝ))) * cnScalarM p a c k ^ (1 / (p : ℝ)) := by
  have hX := cnScalarX_pos p a c hp ha hc hupper k
  have hp' : (p : ℝ) ≠ 0 := by exact_mod_cast hp.ne'
  have hexp : (p : ℝ) * (1 / (p : ℝ)) = 1 := by field_simp
  rw [cnScalar_invariant, Real.mul_rpow ha.le (pow_nonneg hX.le p),
    ← Real.rpow_natCast, ← Real.rpow_mul hX.le, hexp, Real.rpow_one,
    ← mul_assoc, ← Real.rpow_add ha]
  simp

/-- The factorization domain is nonempty, arXiv:2602.02016v2, §3.2. -/
example : 0 < (2 : ℕ) ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (1 : ℝ) < ((2 : ℝ) + 1) * (1 : ℝ) ^ (2 : ℕ) := by norm_num

/-- After the first step, an identity residual at most `δ` guarantees
inverse-root error at most `δ` times the true inverse root. The product
bound is proved for every positive integer root order; no convergence
limit is substituted for the finite iterate.
Source: arXiv:2602.02016v2, §3.2, early stopping using `M_k` near `I`. -/
theorem cnScalarX_residual_accuracy (p : ℕ) (a c δ : ℝ) (hp : 0 < p)
    (ha : 0 < a) (hc : 0 < c) (hupper : a < ((p : ℝ) + 1) * c ^ p)
    (k : ℕ) (hstop : 1 - cnScalarM p a c (k + 1) ≤ δ) :
    |cnScalarX p a c (k + 1) - a ^ (-(1 / (p : ℝ)))| ≤
      a ^ (-(1 / (p : ℝ))) * δ := by
  have hM := cnScalarM_bounds p a c hp ha hc hupper k
  have hp' : (0 : ℝ) < p := by exact_mod_cast hp
  have horder : (1 : ℝ) ≤ p := by exact_mod_cast hp
  have hr : 0 ≤ 1 / (p : ℝ) := (div_pos (by norm_num) hp').le
  have hr' : 1 / (p : ℝ) ≤ 1 := (div_le_one hp').mpr horder
  have hroot := Real.rpow_le_one hM.1.le hM.2 hr
  have hreverse := Real.self_le_rpow_of_le_one hM.1.le hM.2 hr'
  have htarget := Real.rpow_pos_of_pos ha (-(1 / (p : ℝ)))
  rw [cnScalarX_factor p a c hp ha hc hupper (k + 1), abs_of_nonpos
    (sub_nonpos.mpr (mul_le_of_le_one_right htarget.le hroot))]
  calc
    _ = a ^ (-(1 / (p : ℝ))) * (1 - cnScalarM p a c (k + 1) ^ (1 / (p : ℝ))) := by ring
    _ ≤ a ^ (-(1 / (p : ℝ))) * (1 - cnScalarM p a c (k + 1)) :=
      mul_le_mul_of_nonneg_left (sub_le_sub_left hreverse 1) htarget.le
    _ ≤ a ^ (-(1 / (p : ℝ))) * δ := mul_le_mul_of_nonneg_left hstop htarget.le

/-- A finite CN iterate satisfies the stopping hypotheses,
arXiv:2602.02016v2, §3.2. -/
example : 0 < (2 : ℕ) ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (1 : ℝ) < ((2 : ℝ) + 1) * (1 : ℝ) ^ (2 : ℕ) ∧
    1 - cnScalarM 2 1 1 1 ≤ (1 / 10 : ℝ) := by
  norm_num [cnScalarM, cnMap, cnFactor]

end Transformer.DASH
