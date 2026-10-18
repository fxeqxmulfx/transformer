/-
# Saddle avoidance — analytic energies converge without `hloj`

An analytic function on a finite-dimensional real inner product space
satisfies the Łojasiewicz inequality at every point, with an exponent
`0 ≤ a < 1` (`exists_isLojasiewiczAt_of_analyticAt`, Łojasiewicz's theorem).
So the hypothesis `hloj` of the convergence theorems holds for analytic
energies, and convergence becomes a conclusion.

Let `f` be analytic with an `L`-Lipschitz gradient, `0 < η` and `η L < 1`.
An orbit of gradient descent with a limit point up to translations leaving
`f` invariant converges, to a critical point
(`exists_tendsto_of_analyticOnNhd`).  So does every orbit when the
translations span `E` (`exists_tendsto_of_span_eq_top_of_analyticOnNhd`),
and every bounded orbit (`exists_tendsto_of_isBounded_of_analyticOnNhd`).
The same holds for every trajectory of the gradient flow, with no bound on
`∇f` (`exists_tendsto_gradientFlow_of_analyticOnNhd`,
`exists_tendsto_gradientFlow_of_span_eq_top_of_analyticOnNhd`,
`exists_tendsto_gradientFlow_of_isBounded_of_analyticOnNhd`).

For bounded orbits of descent methods on analytic functions this is
Theorem 3.2 of Absil, Mahony and Andrews (SIAM J. Optim. 16, 2005), which
arXiv:1602.04915 cites for `prop:rates`.
-/

import Transformer.SaddleAvoidance.FlowLandscape
import Transformer.SaddleAvoidance.LocalInequality

open Filter Set
open scoped Topology NNReal

namespace Transformer.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  {f : E → ℝ} {L : ℝ≥0} {η : ℝ}

/-- **Gradient descent on an analytic function converges, to a critical point,
from every orbit with a limit point up to translations leaving `f` invariant**,
for `0 < η` and `η L < 1`: Łojasiewicz's theorem gives the inequality at the
limit point (`exists_isLojasiewiczAt_of_analyticAt`), so
`exists_tendsto_of_isLojasiewiczAt` applies, and the limit is critical
(`gradient_eq_zero_of_tendsto`).  For `η = 0` the orbit stands still at a point
that need not be critical. -/
theorem exists_tendsto_of_analyticOnNhd (hf : AnalyticOnNhd ℝ f univ)
    (hL : LipschitzWith L (gradient f)) (hη : 0 < η) (hηL : η * L < 1) {Λ : Set E}
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {x z : E}
    (hcl : ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ, ‖(gradientStep f η)^[k] x + v - z‖ < ε) :
    ∃ w, gradient f w = 0 ∧ Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) := by
  obtain ⟨a, ha0, ha1, hz⟩ := exists_isLojasiewiczAt_of_analyticAt (hf z (mem_univ z))
  obtain ⟨w, hw⟩ := exists_tendsto_of_isLojasiewiczAt
    (differentiableOn_univ.1 hf.differentiableOn) hL hη hηL hΛ ha0 ha1 hz hcl
  exact ⟨w, gradient_eq_zero_of_tendsto hL.continuous hη.ne' hw, hw⟩

/-- **Gradient descent on an analytic function invariant under translations
spanning `E` converges from every start, to a critical point**: every orbit has
a limit point up to the integer combinations of the translations
(`exists_frequently_norm_add_sub_lt`), which leave `f` invariant too
(`periodic_of_mem_span`). -/
theorem exists_tendsto_of_span_eq_top_of_analyticOnNhd (hf : AnalyticOnNhd ℝ f univ)
    (hL : LipschitzWith L (gradient f)) (hη : 0 < η) (hηL : η * L < 1) {Λ : Set E}
    (hspan : Submodule.span ℝ Λ = ⊤) (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) (x : E) :
    ∃ w, gradient f w = 0 ∧ Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) := by
  obtain ⟨R, hR⟩ := exists_add_mem_closedBall_of_span_eq_top hspan
  obtain ⟨z, -, hz⟩ := exists_frequently_norm_add_sub_lt (isCompact_closedBall (0 : E) R)
    (Λ := (Submodule.span ℤ Λ : Set E)) (u := fun k => (gradientStep f η)^[k] x) fun k => hR _
  exact exists_tendsto_of_analyticOnNhd hf hL hη hηL (fun v hv => periodic_of_mem_span hΛ hv) hz

/-- **A bounded orbit of gradient descent on an analytic function converges, to a
critical point**: it has a limit point. -/
theorem exists_tendsto_of_isBounded_of_analyticOnNhd (hf : AnalyticOnNhd ℝ f univ)
    (hL : LipschitzWith L (gradient f)) (hη : 0 < η) (hηL : η * L < 1) {x : E}
    (hbdd : Bornology.IsBounded (range fun k => (gradientStep f η)^[k] x)) :
    ∃ w, gradient f w = 0 ∧ Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) := by
  obtain ⟨z, -, hz⟩ := exists_frequently_norm_add_sub_lt hbdd.isCompact_closure (Λ := {0})
    (u := fun k => (gradientStep f η)^[k] x) fun k => ⟨0, mem_singleton 0, by
      rw [add_zero]
      exact subset_closure (mem_range_self k)⟩
  exact exists_tendsto_of_analyticOnNhd hf hL hη hηL (by simp) hz

/-- **A gradient flow of an analytic function converges, to a critical point, if
it has a limit point up to translations leaving `f` invariant**: `x` solves
`ẋ = -∇f(x)` from time `0` on, with no bound on `∇f`.  Łojasiewicz's theorem
gives the inequality at the limit point, so
`exists_tendsto_gradientFlow_of_isLojasiewiczAt` applies, and the limit is
critical (`gradient_eq_zero_of_tendsto_gradientFlow`). -/
theorem exists_tendsto_gradientFlow_of_analyticOnNhd (hf : AnalyticOnNhd ℝ f univ)
    {x : ℝ → E} (hx : ∀ t, 0 ≤ t → HasDerivAt x (-gradient f (x t)) t) {Λ : Set E}
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {z : E}
    (hcl : ∀ ε > 0, ∃ᶠ t in atTop, ∃ v ∈ Λ, ‖x t + v - z‖ < ε) :
    ∃ w, gradient f w = 0 ∧ Tendsto x atTop (𝓝 w) := by
  obtain ⟨a, -, ha1, hz⟩ := exists_isLojasiewiczAt_of_analyticAt (hf z (mem_univ z))
  obtain ⟨w, hw⟩ := exists_tendsto_gradientFlow_of_isLojasiewiczAt
    (differentiableOn_univ.1 hf.differentiableOn) hx hΛ ha1 hz hcl
  exact ⟨w, gradient_eq_zero_of_tendsto_gradientFlow hx
    (contDiff_gradient hf.contDiff).continuous.continuousAt hw, hw⟩

/-- **Every gradient flow of an analytic function invariant under translations
spanning `E` converges, to a critical point**: every curve has a limit point up
to their integer combinations (`exists_frequently_norm_add_sub_lt_of_span_eq_top`). -/
theorem exists_tendsto_gradientFlow_of_span_eq_top_of_analyticOnNhd
    (hf : AnalyticOnNhd ℝ f univ) {x : ℝ → E} (hx : ∀ t, 0 ≤ t → HasDerivAt x (-gradient f (x t)) t)
    {Λ : Set E} (hspan : Submodule.span ℝ Λ = ⊤) (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) :
    ∃ w, gradient f w = 0 ∧ Tendsto x atTop (𝓝 w) := by
  obtain ⟨z, hz⟩ := exists_frequently_norm_add_sub_lt_of_span_eq_top hspan x
  exact exists_tendsto_gradientFlow_of_analyticOnNhd hf hx
    (fun v hv => periodic_of_mem_span hΛ hv) hz

/-- **A gradient flow of an analytic function bounded from time `0` on converges,
to a critical point**: its values at integer times have a limit point. -/
theorem exists_tendsto_gradientFlow_of_isBounded_of_analyticOnNhd
    (hf : AnalyticOnNhd ℝ f univ) {x : ℝ → E} (hx : ∀ t, 0 ≤ t → HasDerivAt x (-gradient f (x t)) t)
    (hbdd : Bornology.IsBounded (x '' Ici 0)) :
    ∃ w, gradient f w = 0 ∧ Tendsto x atTop (𝓝 w) := by
  obtain ⟨z, -, hz⟩ := exists_frequently_norm_add_sub_lt hbdd.isCompact_closure (Λ := {0})
    (u := fun k : ℕ => x k) fun k => ⟨0, mem_singleton 0, by
      rw [add_zero]
      exact subset_closure (mem_image_of_mem x (mem_Ici.2 (Nat.cast_nonneg k)))⟩
  exact exists_tendsto_gradientFlow_of_analyticOnNhd hf hx (Λ := {0}) (by simp)
    fun ε hε => tendsto_natCast_atTop_atTop.frequently (hz ε hε)

/-- The hypotheses of the three theorems on gradient descent hold for
`f(x) = 1 - cos x` on the line with `L = 1`, `η = 1/2`, the period `2π`, which
spans `ℝ`, and the orbit of the critical point `0`, which stays there: it is
bounded, and it is near `2π` up to the period. -/
example : let f := fun x : ℝ => 1 - Real.cos x
    AnalyticOnNhd ℝ f univ ∧ LipschitzWith 1 (gradient f) ∧ (0 : ℝ) < 1 / 2 ∧
      (1 / 2 : ℝ) * ((1 : ℝ≥0) : ℝ) < 1 ∧ Submodule.span ℝ {2 * Real.pi} = ⊤ ∧
      (∀ v ∈ ({2 * Real.pi} : Set ℝ), ∀ y, f (y + v) = f y) ∧
      (∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ ({2 * Real.pi} : Set ℝ),
        ‖(gradientStep f (1 / 2))^[k] 0 + v - 2 * Real.pi‖ < ε) ∧
      Bornology.IsBounded (range fun k => (gradientStep f (1 / 2))^[k] (0 : ℝ)) := by
  intro f
  have hg : gradient f = Real.sin := funext fun x => by
    rw [gradient_eq_deriv', ((Real.hasDerivAt_cos x).const_sub 1).deriv, neg_neg]
  have horb : ∀ k, (gradientStep f (1 / 2))^[k] (0 : ℝ) = 0 :=
    Function.iterate_fixed (by simp [gradientStep, hg])
  refine ⟨analyticOnNhd_const.sub Real.analyticOnNhd_cos,
    by rw [hg]; exact Real.lipschitzWith_sin, by norm_num, by norm_num,
    Ideal.span_singleton_eq_top.2 (isUnit_iff_ne_zero.2 Real.two_pi_pos.ne'),
    fun v hv y => by simp [mem_singleton_iff.1 hv, f],
    fun ε hε => Frequently.of_forall fun k => ⟨2 * Real.pi, rfl, by
      rwa [horb, zero_add, sub_self, norm_zero]⟩, ?_⟩
  simp only [horb, range_const]
  exact Bornology.isBounded_singleton

/-- The hypotheses of the three theorems on gradient flows hold for
`f(x) = 1 - cos x` on the line and its period `2π`, which spans `ℝ`: every
trajectory of the gradient flow solves `ẋ = -sin x` and has a limit point up to
the integer multiples of the period, and the trajectory at the critical point
`0` stays there. -/
example (x₀ : ℝ) : let f := fun x : ℝ => 1 - Real.cos x
    AnalyticOnNhd ℝ f univ ∧ (∀ t, 0 ≤ t → HasDerivAt (fun t => gradientFlow f t x₀)
      (-gradient f (gradientFlow f t x₀)) t) ∧ Submodule.span ℝ {2 * Real.pi} = ⊤ ∧
      (∀ v ∈ ({2 * Real.pi} : Set ℝ), ∀ y, f (y + v) = f y) ∧
      (∀ v ∈ (Submodule.span ℤ {2 * Real.pi} : Set ℝ), ∀ y, f (y + v) = f y) ∧
      (∃ z, ∀ ε > 0, ∃ᶠ t in atTop, ∃ v ∈ (Submodule.span ℤ {2 * Real.pi} : Set ℝ),
        ‖gradientFlow f t x₀ + v - z‖ < ε) ∧
      (∀ t, 0 ≤ t → HasDerivAt (fun _ : ℝ => (0 : ℝ)) (-gradient f 0) t) ∧
      Bornology.IsBounded ((fun _ : ℝ => (0 : ℝ)) '' Ici 0) := by
  intro f
  have hg : gradient f = Real.sin := funext fun x => by
    rw [gradient_eq_deriv', ((Real.hasDerivAt_cos x).const_sub 1).deriv, neg_neg]
  have hspan : Submodule.span ℝ {2 * Real.pi} = ⊤ :=
    Ideal.span_singleton_eq_top.2 (isUnit_iff_ne_zero.2 Real.two_pi_pos.ne')
  have hΛ : ∀ v ∈ ({2 * Real.pi} : Set ℝ), ∀ y, f (y + v) = f y := fun v hv y => by
    simp [mem_singleton_iff.1 hv, f]
  refine ⟨analyticOnNhd_const.sub Real.analyticOnNhd_cos, fun t _ => hasDerivAt_gradientFlow
    (by rw [hg]; exact Real.lipschitzWith_sin)
    (fun x => by rw [hg, Real.norm_eq_abs]; exact Real.abs_sin_le_one x) t x₀, hspan, hΛ,
    fun v hv => periodic_of_mem_span hΛ hv,
    exists_frequently_norm_add_sub_lt_of_span_eq_top hspan _, fun t _ => ?_,
    Bornology.isBounded_singleton.subset (image_subset_iff.2 fun _ _ => rfl)⟩
  rw [hg, Real.sin_zero, neg_zero]
  exact hasDerivAt_const t 0

end Transformer.SaddleAvoidance
