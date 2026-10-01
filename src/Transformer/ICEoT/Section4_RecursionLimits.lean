/-
# IC-EoT: why controls and predicted states have different extensions

arXiv:2603.22095v2, §4.1, discussion following Eqs. (33), (34), and
§4.4.3's distinction between learned-model and true-plant optimality.
These are explicit counterexamples to the extensions the source excludes.
-/

import Transformer.ICEoT.Section4_Global

noncomputable section

namespace Transformer.ICEoT

/-- Independent non-negative convex monotone gate and value branches
need not have a jointly convex product. This explains the shared-scalar
requirement of §3.3, Lemma 2, also discussed in §2.2.3. -/
theorem independent_gate_value_not_convex :
    ActivationConditions relu ∧
      ¬ ConvexOn ℝ Set.univ (fun v : ℝ × ℝ => relu v.1 * relu v.2) := by
  refine ⟨relu_conditions, ?_⟩
  intro h
  have hj := h.2 (Set.mem_univ ((0 : ℝ), (2 : ℝ)))
    (Set.mem_univ ((2 : ℝ), (0 : ℝ)))
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)
  norm_num [relu] at hj

/-- Duplicating a convex predicted state with its negative can destroy
multi-step convexity. A convex first prediction `x²` followed by the
convex monotone outer activation `ReLU(1 + negativeState)` gives
`ReLU(1-x²)`, which fails Jensen at `-1,1`. This is exactly the hazard
described in §4.1 after Eq. (34); controls, being affine, avoid it. -/
theorem negative_predicted_state_breaks_convexity :
    ConvexOn ℝ Set.univ (fun x : ℝ => x ^ 2) ∧ ActivationConditions relu ∧
      ¬ ConvexOn ℝ Set.univ (fun x : ℝ => relu (1 - x ^ 2)) := by
  refine ⟨(by decide : Even (2 : ℕ)).convexOn_pow, relu_conditions, ?_⟩
  intro h
  have hj := h.2 (Set.mem_univ (-1 : ℝ)) (Set.mem_univ (1 : ℝ))
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)
  norm_num [relu] at hj

/-- Model-optimality need not imply true-plant optimality, as emphasized
in §4.4.3: the convex surrogate `x²` is minimized at zero on `[0,1]`,
but the true cost `(x-1)²` is lower at the feasible point one. -/
theorem learned_optimum_need_not_minimize_plant :
    ConvexOn ℝ Set.univ (fun x : ℝ => x ^ 2) ∧
    IsMinOn (fun x : ℝ => x ^ 2) (Set.Icc 0 1) 0 ∧
    ¬ IsMinOn (fun x : ℝ => (x - 1) ^ 2) (Set.Icc 0 1) 0 := by
  refine ⟨(by decide : Even (2 : ℕ)).convexOn_pow, ?_, ?_⟩
  · intro x hx
    simpa using sq_nonneg x
  · intro h
    have hj := h (by norm_num : (1 : ℝ) ∈ Set.Icc 0 1)
    norm_num at hj

end Transformer.ICEoT
