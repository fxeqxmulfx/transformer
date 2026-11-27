module

public import Transformer.Optimization.PAGE.Convergence

/-! A nonconstant finite-sum witness for PAGE. The two components have
different curvature, so the two correction branches genuinely differ. -/

@[expose] public section

namespace Transformer.Optimization.PAGE

open Filter
open scoped Topology

noncomputable def twoLosses (i : Fin 2) (x : ℝ) : ℝ := if i = 0 then 0 else x^2

private theorem two_gradient (i : Fin 2) (x : ℝ) :
    gradient (twoLosses i) x = if i = 0 then 0 else 2*x := by
  by_cases h : i = 0
  · rw [show twoLosses i = fun _ => (0 : ℝ) by funext y; simp [twoLosses, h]]
    simp [h]
  · rw [show twoLosses i = fun y => y^2 by funext y; simp [twoLosses, h], ite_eq_right h]
    have hd : HasDerivAt (fun y : ℝ => y^2) (2*x) x := by
      convert (hasDerivAt_id x).pow 2 using 1 <;> first | rfl | simp
    exact hd.hasGradientAt'.gradient

private theorem two_diff (i : Fin 2) : Differentiable ℝ (twoLosses i) := by
  change Differentiable ℝ (fun x : ℝ => if i = 0 then 0 else x^2)
  by_cases h : i = 0 <;> simp only [h, reduceIte] <;> fun_prop

private theorem two_objective : objective twoLosses = quadratic := by
  funext x
  simp [objective, mean, twoLosses, Fin.sum_univ_two, quadratic]
  ring

private theorem two_average : AverageSmooth twoLosses 2 := by
  intro x y
  simp [mean, Fin.sum_univ_two, two_gradient, Real.norm_eq_abs, sq_abs]
  nlinarith [sq_nonneg (y-x)]

/-- A genuinely stochastic instance with refresh probability 1/2 and step 1/8.
Both the nonconvex budget and the PL estimate have discharged assumptions. -/
theorem twoLosses_bounds (x : ℝ) (N : ℕ) :
    (1/8 : ℝ) / 2 * ∑ t ∈ Finset.range N, expected twoLosses (1/8) (1/2) t
      (fun s => ‖gradient (objective twoLosses) s.weights‖^2) (start twoLosses x) ≤ quadratic x ∧
    expected twoLosses (1/8) (1/2) N (fun s => quadratic s.weights) (start twoLosses x) ≤
      (7/8 : ℝ)^N * quadratic x := by
  have hl : SmoothObjective (objective twoLosses) 1 := two_objective ▸ quadratic_smooth
  have hs : StableStep (1/8) (1/2) 1 2 := by norm_num [StableStep]
  have hb (y : ℝ) : 0 ≤ objective twoLosses y := by rw [two_objective]; dsimp [quadratic]; positivity
  have hpl : Dominated (objective twoLosses) 1 0 := by
    intro y
    rw [two_objective, quadratic_gradient]
    simp [quadratic, sq_abs, Real.norm_eq_abs]
    linarith
  constructor
  · simpa only [sub_zero, two_objective] using gradient_budget twoLosses two_diff
      (by norm_num : 0 < (1/8 : ℝ)) (by norm_num : 0 < (1/2 : ℝ)) (by norm_num) hl two_average hs hb x N
  · have h := pl_rate twoLosses two_diff (by norm_num : 0 < (1/8 : ℝ))
      (by norm_num : 0 < (1/2 : ℝ)) (by norm_num) hl two_average hs hb (by norm_num : (0:ℝ) < 1) hpl x N
    norm_num [two_objective] at h
    exact h

end Transformer.Optimization.PAGE
