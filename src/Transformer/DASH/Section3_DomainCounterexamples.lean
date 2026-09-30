/-
# DASH — why symmetry and closed endpoints do not suffice

arXiv:2602.02016v2, §3.1–3.2. Real inverse roots require a positive
spectrum. The source's symmetric-input wording by itself does not
exclude singular matrices or negative eigenvalues.
-/

import Transformer.DASH.Section3_InverseRootCorrectness

open scoped Matrix

namespace Transformer.DASH

/-- A symmetric negative scalar matrix has no real inverse square root.
This refutes an unrestricted real interpretation of the source's “symmetric
`A`, `p∈ℝ`” power construction. The corrected EVD/root theorems use positive
definiteness for inverse roots. Source: arXiv:2602.02016v2, §3.1. -/
theorem symmetric_negative_inverse_root_counterexample :
    (-1 : Matrix (Fin 1) (Fin 1) ℝ).IsHermitian ∧
      ¬ ∃ X : Matrix (Fin 1) (Fin 1) ℝ, (-1 : Matrix (Fin 1) (Fin 1) ℝ) * X ^ 2 = 1 := by
  constructor
  · simp
  · rintro ⟨X, h⟩
    have heq := congrFun (congrFun h 0) 0
    norm_num [pow_two, Matrix.mul_apply, Fin.sum_univ_one] at heq
    nlinarith [sq_nonneg (X 0 0)]

/-- A symmetric zero matrix has no inverse root of any order, including a
positive integer order. Thus the zero endpoint in the printed CN domain
cannot be a valid inverse-root input. Source: arXiv:2602.02016v2, §3.1–3.2. -/
theorem symmetric_zero_inverse_root_counterexample (p : ℕ) :
    (0 : Matrix (Fin 1) (Fin 1) ℝ).IsHermitian ∧
      ¬ ∃ X : Matrix (Fin 1) (Fin 1) ℝ, (0 : Matrix (Fin 1) (Fin 1) ℝ) * X ^ p = 1 := by
  constructor
  · simp
  · rintro ⟨X, h⟩
    have heq := congrFun (congrFun h 0) 0
    norm_num at heq

end Transformer.DASH
