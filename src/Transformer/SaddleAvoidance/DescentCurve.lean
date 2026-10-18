/-
# Saddle avoidance — curves that descend a function

A curve `x` *descends* `f` from time `t₀` on, at the rates `c₁` and `c₂`, when
for `t ≥ t₀` it is differentiable with `c₁ ‖∇f(x)‖² ≤ -⟪∇f(x), ẋ⟫` and
`‖ẋ‖ ≤ c₂ ‖∇f(x)‖` (`IsDescentCurve`): `f` decreases at least at the rate
`c₁ ‖∇f‖²` (`IsDescentCurve.antitoneOn`), and the curve moves at most at the
speed `c₂ ‖∇f‖`.  The gradient flow `ẋ = -∇f(x)` descends `f` at the rates `1`
and `1` (`isDescentCurve_of_gradientFlow`), and a flow `ẋ = -G(x) ∇f(x)` in a
metric `G` between `c₁` and `c₂` at the rates `c₁` and `c₂`
(`isDescentCurve_of_metric`).  Translations leaving `f` invariant carry
descent curves to descent curves (`IsDescentCurve.add_const`), and a descent
curve along which `f` stays constant does not move
(`IsDescentCurve.eq_of_comp_eq`).

The trap and the convergence of `SaddleAvoidance.FlowTrap` and
`SaddleAvoidance.FlowConvergence` hold for descent curves
(`SaddleAvoidance.DescentTrap`, `SaddleAvoidance.DescentConvergence`), so for
flows in a metric too.  Source: arXiv:2312.10794v5, §4, after
`thm: beta.interval`: "since all the intervening functions and metrics are
real-analytic", Łojasiewicz's theorem (lojasiewicz1963propriete) makes the
gradient flows of `𝖤_β` converge, for the metric of `SA` (§3.4) as for that of
`USA`.
-/

import Transformer.SaddleAvoidance.FlowTrap

open Filter Set
open scoped Topology

namespace Transformer.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {x : ℝ → E} {c₁ c₂ t₀ : ℝ}

local notation "⟪" x ", " y "⟫" => inner ℝ x y

/-- **`x` descends `f` from time `t₀` on, at the rates `c₁` and `c₂`**: for
`t ≥ t₀` it is differentiable, `f` decreases along it at least at the rate
`c₁ ‖∇f(x)‖²`, and it moves at most at the speed `c₂ ‖∇f(x)‖`. -/
def IsDescentCurve (f : E → ℝ) (c₁ c₂ t₀ : ℝ) (x : ℝ → E) : Prop :=
  ∀ t, t₀ ≤ t → DifferentiableAt ℝ x t ∧
    c₁ * ‖gradient f (x t)‖ ^ 2 ≤ -⟪gradient f (x t), deriv x t⟫ ∧
      ‖deriv x t‖ ≤ c₂ * ‖gradient f (x t)‖

/-- A curve with velocity `w` descends `f` if `w` does. -/
theorem isDescentCurve_of_hasDerivAt {w : ℝ → E} (hx : ∀ t, t₀ ≤ t → HasDerivAt x (w t) t)
    (h₁ : ∀ t, t₀ ≤ t → c₁ * ‖gradient f (x t)‖ ^ 2 ≤ -⟪gradient f (x t), w t⟫)
    (h₂ : ∀ t, t₀ ≤ t → ‖w t‖ ≤ c₂ * ‖gradient f (x t)‖) : IsDescentCurve f c₁ c₂ t₀ x :=
  fun t ht => by
    rw [(hx t ht).deriv]
    exact ⟨(hx t ht).differentiableAt, h₁ t ht, h₂ t ht⟩

/-- **A flow in a metric descends `f`**: if `c₁ ‖v‖² ≤ ⟪G(y) v, v⟫` and
`‖G(y) v‖ ≤ c₂ ‖v‖` for all `y` and `v`, a solution of `ẋ = -G(x) ∇f(x)`
descends `f` at the rates `c₁` and `c₂`. -/
theorem isDescentCurve_of_metric {G : E → E →L[ℝ] E}
    (hG₁ : ∀ y v, c₁ * ‖v‖ ^ 2 ≤ ⟪G y v, v⟫) (hG₂ : ∀ y v, ‖G y v‖ ≤ c₂ * ‖v‖)
    (hx : ∀ t, t₀ ≤ t → HasDerivAt x (-G (x t) (gradient f (x t))) t) :
    IsDescentCurve f c₁ c₂ t₀ x :=
  isDescentCurve_of_hasDerivAt hx (fun t _ => by
      rw [inner_neg_right, neg_neg, real_inner_comm]
      exact hG₁ _ _)
    fun t _ => by
      rw [norm_neg]
      exact hG₂ _ _

/-- **The gradient flow descends `f` at the rates `1` and `1`.** -/
theorem isDescentCurve_of_gradientFlow
    (hx : ∀ t, t₀ ≤ t → HasDerivAt x (-gradient f (x t)) t) : IsDescentCurve f 1 1 t₀ x :=
  isDescentCurve_of_hasDerivAt hx
    (fun t _ => by rw [inner_neg_right, neg_neg, real_inner_self_eq_norm_sq, one_mul])
    fun t _ => by rw [norm_neg, one_mul]

theorem IsDescentCurve.mono (hx : IsDescentCurve f c₁ c₂ t₀ x) {t₁ : ℝ} (h : t₀ ≤ t₁) :
    IsDescentCurve f c₁ c₂ t₁ x :=
  fun t ht => hx t (h.trans ht)

theorem IsDescentCurve.hasDerivAt (hx : IsDescentCurve f c₁ c₂ t₀ x) {t : ℝ} (ht : t₀ ≤ t) :
    HasDerivAt x (deriv x t) t :=
  (hx t ht).1.hasDerivAt

/-- **The translate of a descent curve by a vector leaving `f` invariant
descends `f`**: the velocity is the same, and so is the gradient. -/
theorem IsDescentCurve.add_const {v : E} (hv : ∀ y, f (y + v) = f y)
    (hx : IsDescentCurve f c₁ c₂ t₀ x) : IsDescentCurve f c₁ c₂ t₀ fun t => x t + v :=
  fun t ht => by
    rw [deriv_add_const, gradient_add_eq hv]
    exact ⟨(hx t ht).1.add_const v, (hx t ht).2⟩

/-- **`f` changes along a descent curve at the rate `⟪∇f, ẋ⟫`.** -/
theorem IsDescentCurve.hasDerivAt_comp (hx : IsDescentCurve f c₁ c₂ t₀ x) {t : ℝ}
    (ht : t₀ ≤ t) (hf : DifferentiableAt ℝ f (x t)) :
    HasDerivAt (fun s => f (x s)) ⟪gradient f (x t), deriv x t⟫ t := by
  have h := hf.hasFDerivAt.comp_hasDerivAt t (hx.hasDerivAt ht)
  rwa [← inner_gradient_left] at h

theorem IsDescentCurve.antitoneOn (hf : Differentiable ℝ f) (hc₁ : 0 ≤ c₁)
    (hx : IsDescentCurve f c₁ c₂ t₀ x) : AntitoneOn (fun t => f (x t)) (Ici t₀) := by
  refine antitoneOn_of_hasDerivWithinAt_nonpos (f' := fun t => ⟪gradient f (x t), deriv x t⟫)
    (convex_Ici t₀) (fun t ht => (hx.hasDerivAt_comp ht (hf _)).continuousAt.continuousWithinAt)
    (fun t ht => (hx.hasDerivAt_comp (interior_subset ht) (hf _)).hasDerivWithinAt)
    fun t ht => ?_
  nlinarith [(hx t (interior_subset ht)).2.1, mul_nonneg hc₁ (sq_nonneg ‖gradient f (x t)‖)]

/-- **A descent curve along which `f` stays constant does not move**: `f`
decreases at least at the rate `c₁ ‖∇f‖²`, so `∇f = 0`, and the speed is at
most `c₂ ‖∇f‖`. -/
theorem IsDescentCurve.eq_of_comp_eq (hf : Differentiable ℝ f)
    (hx : IsDescentCurve f c₁ c₂ t₀ x) (hc₁ : 0 < c₁)
    (hc : ∀ t, t₀ ≤ t → f (x t) = f (x t₀)) {t : ℝ} (ht : t₀ ≤ t) : x t = x t₀ := by
  have hvel : ∀ u ∈ Ici t₀, HasDerivWithinAt x 0 (Ici t₀) u := fun u hu => by
    have hd := (hx.hasDerivAt_comp hu (hf _)).hasDerivWithinAt (s := Ici t₀)
    have hconst : HasDerivWithinAt (fun s => f (x s)) 0 (Ici t₀) u :=
      (hasDerivWithinAt_const u (Ici t₀) (f (x t₀))).congr_of_mem hc hu
    have hrate := (uniqueDiffOn_Ici t₀ u hu).eq_deriv _ hd hconst
    obtain ⟨-, h₁, h₂⟩ := hx u hu
    rw [hrate, neg_zero] at h₁
    have hg : ‖gradient f (x u)‖ = 0 := by
      by_contra hne
      linarith [mul_pos hc₁ (pow_pos (lt_of_le_of_ne (norm_nonneg _) (Ne.symm hne)) 2)]
    rw [hg, mul_zero, norm_le_zero_iff] at h₂
    simpa [h₂] using (hx.hasDerivAt hu).hasDerivWithinAt (s := Ici t₀)
  have h := (convex_Ici t₀).norm_image_sub_le_of_norm_hasDerivWithin_le (C := 0) hvel
    (fun _ _ => by simp) self_mem_Ici ht
  rwa [zero_mul, norm_le_zero_iff, sub_eq_zero] at h

/-- The hypotheses of the theorems above hold for `f(x) = ‖x‖² / 2` on the
line: its gradient flow `x t = e^{-t}` descends `f` at the rates `1` and `1`,
so does the flow in the metric `G = 1`, `f` is invariant under the translation
by `0`, and `f` stays constant along the constant curve `0`, which descends
`f` too. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    let x := fun t : ℝ => Real.exp (-t)
    Differentiable ℝ f ∧ (∀ t, HasDerivAt x (-gradient f (x t)) t) ∧
      (∀ y v : ℝ, 1 * ‖v‖ ^ 2 ≤ ⟪(fun _ : ℝ => ContinuousLinearMap.id ℝ ℝ) y v, v⟫) ∧
      (∀ y v : ℝ, ‖(fun _ : ℝ => ContinuousLinearMap.id ℝ ℝ) y v‖ ≤ 1 * ‖v‖) ∧
      IsDescentCurve f 1 1 0 x ∧
      (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧ (∀ y, f (y + 0) = f y) ∧
      IsDescentCurve f 1 1 0 (fun _ => 0) ∧ ∀ t : ℝ, 0 ≤ t → f 0 = f 0 := by
  intro f x
  have hg : gradient f = id := gradient_half_mul_norm_sq
  have hd : ∀ t, HasDerivAt x (-gradient f (x t)) t := fun t => by
    rw [hg]
    simpa [x] using (hasDerivAt_neg t).exp
  refine ⟨(contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero, hd,
    fun _ v => by rw [one_mul, ContinuousLinearMap.id_apply, real_inner_self_eq_norm_sq],
    fun _ v => by rw [one_mul, ContinuousLinearMap.id_apply],
    isDescentCurve_of_gradientFlow fun t _ => hd t, zero_le_one, one_pos,
    fun y => by rw [add_zero], isDescentCurve_of_gradientFlow fun t _ => ?_, fun _ _ => rfl⟩
  rw [hg, id, neg_zero]
  exact hasDerivAt_const t 0

end Transformer.SaddleAvoidance
