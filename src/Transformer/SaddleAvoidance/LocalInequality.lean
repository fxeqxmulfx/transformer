/-
# Saddle avoidance — the local gradient inequality of `Normalization`

`Normalization` states the Łojasiewicz gradient inequality at a point `z` in
the form of the hypothesis `hlocal` of
`Normalization.lojasiewicz_modulated_of_local_gradient_inequality`: for some
`0 < α < 1`, `k > 0` and an open `V ∋ z`, `|E y - E z| ^ α ≤ k ‖∇E(y)‖` for
every `y ∈ V`.  That form gives `IsLojasiewiczAt E z α` with `m = 1/k`
(`isLojasiewiczAt_of_forall_mem`), so it supplies the hypothesis `hloj` of
the convergence theorems of this directory at `z`
(`exists_isLojasiewiczAt_of_local_gradient_inequality`).  Every case of the
local inequality that `Normalization` proves therefore plugs in, and so will
the general analytic case once it is proved there: the hard core of
`Normalization.lojasiewicz_modulated`, Łojasiewicz's theorem at degenerate
critical points.  In dimension at most one that case is proved, so an
analytic function on `EucSpace N`, `N ≤ 1`, satisfies the inequality at
every critical point (`exists_isLojasiewiczAt_of_analyticAt_low_dimension`).
-/

import Transformer.SaddleAvoidance.Lojasiewicz
import Transformer.Normalization.OneDimensionalGradientInequality

open Filter
open scoped Topology

namespace Transformer.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {z : E}

/-- **The local gradient inequality on an open set gives the Łojasiewicz
inequality**: if `|f y - f z| ^ a ≤ k ‖∇f(y)‖` for every `y` in an open set
`V ∋ z`, with `k > 0`, then `IsLojasiewiczAt f z a`, with `m = 1/k` and
`ε = 1`.  The hypothesis asks the inequality on both sides of the level
`f z`; `IsLojasiewiczAt` asks it only above. -/
theorem isLojasiewiczAt_of_forall_mem {a k : ℝ} {V : Set E} (hk : 0 < k) (hV : IsOpen V)
    (hzV : z ∈ V) (h : ∀ y ∈ V, |f y - f z| ^ a ≤ k * ‖gradient f y‖) :
    IsLojasiewiczAt f z a := by
  refine ⟨1 / k, by positivity, 1, one_pos, ?_⟩
  filter_upwards [hV.mem_nhds hzV] with y hy _ _
  calc 1 / k * |f y - f z| ^ a ≤ 1 / k * (k * ‖gradient f y‖) :=
        mul_le_mul_of_nonneg_left (h y hy) (by positivity)
    _ = ‖gradient f y‖ := by rw [← mul_assoc, one_div_mul_cancel hk.ne', one_mul]

/-- **The form `hlocal` of `Normalization` gives the form `hloj`.**  The
local gradient inequality at `z` as the hypothesis `hlocal` of
`Normalization.lojasiewicz_modulated_of_local_gradient_inequality` states it
(Appendix D.1, `lem: loj`, of arXiv:2510.22026v2) gives an exponent
`0 ≤ a < 1` with `IsLojasiewiczAt f z a`, which is what the hypothesis
`hloj` of `ae_exists_tendsto_isLocalMin` asks at each critical point. -/
theorem exists_isLojasiewiczAt_of_local_gradient_inequality
    (h : ∃ alpha k : ℝ, ∃ V : Set E, 0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
      ∀ y ∈ V, |f y - f z| ^ alpha ≤ k * ‖gradient f y‖) :
    ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a := by
  obtain ⟨alpha, k, V, ha0, ha1, hk, hV, hzV, hineq⟩ := h
  exact ⟨alpha, ha0.le, ha1, isLojasiewiczAt_of_forall_mem hk hV hzV hineq⟩

/-- **Analytic functions in dimension at most one satisfy the Łojasiewicz
inequality at every critical point**, with an exponent `0 ≤ a < 1`,
degenerate critical points included.  The local inequality is
`Normalization.analytic_gradient_inequality_low_dimension`. -/
theorem exists_isLojasiewiczAt_of_analyticAt_low_dimension {N : ℕ} (hN : N ≤ 1)
    {g : EucSpace N → ℝ} {w : EucSpace N} (hg : AnalyticAt ℝ g w) (hw : gradient g w = 0) :
    ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt g w a :=
  exists_isLojasiewiczAt_of_local_gradient_inequality
    (Normalization.analytic_gradient_inequality_low_dimension hN g w hg hw)

/-- The hypotheses of the first two theorems hold for `f(x) = ‖x‖² / 2` on
the line at `z = 0`, with `a = 1/2`, `k = 1` and `V = ℝ`. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    (0 : ℝ) < 1 ∧ IsOpen (Set.univ : Set ℝ) ∧ (0 : ℝ) ∈ Set.univ ∧
      (∀ y ∈ Set.univ, |f y - f 0| ^ (1 / 2 : ℝ) ≤ 1 * ‖gradient f y‖) ∧
      (∃ alpha k : ℝ, ∃ V : Set ℝ, 0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ 0 ∈ V ∧
        ∀ y ∈ V, |f y - f 0| ^ alpha ≤ k * ‖gradient f y‖) := by
  intro f
  have hineq : ∀ y ∈ Set.univ, |f y - f 0| ^ (1 / 2 : ℝ) ≤ 1 * ‖gradient f y‖ := by
    intro y _
    have h0 : f y - f 0 = 1 / 2 * ‖y‖ ^ 2 := by
      simp only [f, norm_zero]
      ring
    rw [h0, abs_of_nonneg (by positivity), one_mul, gradient_half_mul_norm_sq, id]
    exact half_mul_norm_sq_rpow_le y
  exact ⟨one_pos, isOpen_univ, Set.mem_univ 0, hineq,
    1 / 2, 1, Set.univ, by norm_num, by norm_num, one_pos, isOpen_univ, Set.mem_univ 0, hineq⟩

/-- The hypotheses of the third theorem hold for `g(y) = y₀⁴` on `EucSpace 1`
at its degenerate critical point `0`. -/
example : let p : EucSpace 1 →L[ℝ] ℝ := EuclideanSpace.proj (0 : Fin 1)
    1 ≤ 1 ∧ AnalyticAt ℝ (fun y => p y ^ 4) 0 ∧ gradient (fun y => p y ^ 4) 0 = 0 := by
  intro p
  refine ⟨le_rfl, (p.analyticAt 0).fun_pow 4, ?_⟩
  have h := (p.hasFDerivAt (x := 0)).pow 4
  rw [map_zero, zero_pow (by norm_num), smul_zero, zero_smul] at h
  refine HasGradientAt.gradient ?_
  rw [hasGradientAt_iff_hasFDerivAt, map_zero]
  exact h

end Transformer.SaddleAvoidance
