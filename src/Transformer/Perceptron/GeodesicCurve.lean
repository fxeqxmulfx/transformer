/-
# Perceptrons and attention's mean-field landscape — curves issued from a minimum

The calculus behind `isSOPD_of_isMin` (`StrictSOPD`), the remark that the
minimizer of the coupled energy is a SOPD critical point.  Two curves carry it.

* Through a point `x` of the sphere, the great circle `t ↦ exp_x(t u)` has
  velocity `u` at `t = 0` (`hasDerivAt_sphereExp_smul`).  A function on `ℝ^d`
  that is minimal at `x` among the points of the sphere therefore has a
  tangential gradient `Proj_x ∇F(x) = 0` there (`proj_eq_zero_of_isMin_sphere`):
  the first-order half, which gives stationarity.

* Through a measure `μ`, a `W₂`-geodesic `ν` starts at `ν 0 = μ` and is
  weakly continuous at `0` (`geodesic_zero`, `tendsto_geodesic`).  A function
  of `t` minimal at `0`, continuous there, has a non-negative second derivative
  at `0` whenever it has one (`nonneg_of_hasDerivAt_deriv_of_isMin`): the
  second-order half.  The continuity is not decoration — `J(0) = 0`,
  `J(t) = 1 - t²` elsewhere has a global minimum at `0` and `J''(0) = -2`.

Source: arXiv:2601.21366v2, §2.2, `rem:strictSOPD-perceptron`.
-/

import Transformer.Perceptron.Geodesic
import Mathlib.Analysis.Calculus.DerivativeTest
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.MeasureTheory.Integral.DominatedConvergence

open scoped BigOperators
open Real MeasureTheory Filter Topology

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### Great circles -/

/-- `exp_x(t v) = cos(t‖v‖) x + (sin(t‖v‖)/‖v‖) v`: the great circle through `x`
with velocity `v`, the norm of `t • v` resolved. -/
theorem sphereExp_smul (x v : EucSpace d) (t : ℝ) :
    sphereExp x (t • v) = Real.cos (t * ‖v‖) • x + (Real.sin (t * ‖v‖) / ‖v‖) • v := by
  rcases eq_or_ne v 0 with rfl | hv
  · simp
  have hvn : ‖v‖ ≠ 0 := norm_ne_zero_iff.mpr hv
  rw [sphereExp, norm_smul, Real.norm_eq_abs, smul_smul]
  rcases lt_trichotomy t 0 with ht | rfl | ht
  · rw [abs_of_neg ht, neg_mul, Real.cos_neg, Real.sin_neg]
    congr 2
    field_simp [ht.ne]
  · simp
  · rw [abs_of_pos ht]
    congr 2
    field_simp [ht.ne']

/-- **The great circle `t ↦ exp_x(t v)` has velocity `v` at `t = 0`.** -/
theorem hasDerivAt_sphereExp_smul (x v : EucSpace d) :
    HasDerivAt (fun t : ℝ => sphereExp x (t • v)) v 0 := by
  rcases eq_or_ne v 0 with rfl | hv
  · simpa using hasDerivAt_const (0 : ℝ) x
  have hvn : ‖v‖ ≠ 0 := norm_ne_zero_iff.mpr hv
  have hm : HasDerivAt (fun t : ℝ => t * ‖v‖) ‖v‖ 0 := hasDerivAt_mul_const ‖v‖
  have hc := hm.cos.smul_const x
  have hs := (hm.sin.div_const ‖v‖).smul_const v
  convert hc.add hs using 1
  · exact funext fun t => sphereExp_smul x v t
  · simp [hvn]

/-- **A minimum on the sphere is a critical point of the tangential
gradient**: if `F` is minimal at `x` among the unit vectors, then
`Proj_x ∇F(x) = 0`.  Along the great circle with velocity `u = Proj_x ∇F(x)`
the derivative `⟪∇F(x), u⟫` vanishes at the minimum, and it equals `‖u‖²`. -/
theorem proj_eq_zero_of_isMin_sphere {F : EucSpace d → ℝ} {x g : EucSpace d}
    (hx : ‖x‖ = 1) (hF : HasGradientAt F g x)
    (hmin : ∀ y : EucSpace d, ‖y‖ = 1 → F x ≤ F y) : proj d x g = 0 := by
  set u := proj d x g with hu
  have hxu : inner (𝕜 := ℝ) x u = 0 := inner_proj_eq_zero hx g
  have hF' : HasFDerivAt F (InnerProductSpace.toDual ℝ (EucSpace d) g)
      (sphereExp x ((0 : ℝ) • u)) := by
    rw [zero_smul, sphereExp_zero]
    exact hasGradientAt_iff_hasFDerivAt.mp hF
  have hcomp : HasDerivAt (fun t : ℝ => F (sphereExp x (t • u))) (inner (𝕜 := ℝ) g u) 0 := by
    have h := hF'.comp_hasDerivAt (0 : ℝ) (hasDerivAt_sphereExp_smul x u)
    rw [InnerProductSpace.toDual_apply_apply] at h
    exact h
  have hloc : IsLocalMin (fun t : ℝ => F (sphereExp x (t • u))) 0 :=
    Eventually.of_forall fun t => by
      simp only [zero_smul, sphereExp_zero]
      exact hmin _ (norm_sphereExp hx (by rw [inner_smul_right, hxu, mul_zero]))
  have hgu : inner (𝕜 := ℝ) g u = 0 := hloc.hasDerivAt_eq_zero hcomp
  have huu : inner (𝕜 := ℝ) u u = inner (𝕜 := ℝ) g u := by
    nth_rewrite 1 [hu]
    rw [proj, inner_sub_left, real_inner_smul_left, hxu, mul_zero, sub_zero]
  exact inner_self_eq_zero.mp (huu.trans hgu)

/-- The hypotheses of `proj_eq_zero_of_isMin_sphere` are satisfiable: `F = 0`
at the first basis vector of `ℝ^1`, with gradient `0`. -/
example : ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))‖ = 1 ∧
    HasGradientAt (fun _ : EucSpace 1 => (0 : ℝ)) 0 (EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) ∧
    ∀ y : EucSpace 1, ‖y‖ = 1 → (0 : ℝ) ≤ 0 :=
  ⟨by simp, hasGradientAt_const _ _, fun _ _ => le_rfl⟩

/-! ### Geodesics of measures -/

/-- A geodesic starts where it is issued from: `ν 0 = μ`, since `exp_x(0) = x`. -/
theorem geodesic_zero {ξ : SSphere d → EucSpace d} {μ : Perspective.ProbSphere d}
    {ν : ℝ → Perspective.ProbSphere d} (hν : IsGeodesicFrom ξ μ ν) : ν 0 = μ := by
  obtain ⟨T, -, hT, hνT⟩ := hν 0
  have hid : T = id := funext fun x => Subtype.ext (by rw [hT, zero_smul, sphereExp_zero]; rfl)
  exact ProbabilityMeasure.toMeasure_injective (by rw [hνT, hid, Measure.map_id])

/-- **A geodesic is weakly continuous at its start**: every `x ↦ exp_x(t ξ(x))`
tends to the identity as `t → 0`, and bounded continuous functions pass to the
limit under the integral by dominated convergence. -/
theorem tendsto_geodesic {ξ : SSphere d → EucSpace d} {μ : Perspective.ProbSphere d}
    {ν : ℝ → Perspective.ProbSphere d} (hν : IsGeodesicFrom ξ μ ν) :
    Tendsto ν (𝓝 0) (𝓝 μ) := by
  choose T hTm hT hνT using hν
  rw [ProbabilityMeasure.tendsto_iff_forall_integral_tendsto]
  intro f
  have hint : ∀ t : ℝ, ∫ x, f x ∂(ν t : Measure (SSphere d)) =
      ∫ x, f (T t x) ∂(μ : Measure (SSphere d)) := fun t => by
    rw [hνT t, integral_map (hTm t).aemeasurable f.continuous.aestronglyMeasurable]
  simp_rw [hint]
  refine tendsto_integral_filter_of_dominated_convergence (fun _ => ‖f‖)
    (Eventually.of_forall fun t => (f.continuous.measurable.comp (hTm t)).aestronglyMeasurable)
    (Eventually.of_forall fun t => ae_of_all _ fun x => f.norm_coe_le_norm (T t x))
    (integrable_const ‖f‖) (ae_of_all _ fun x => ?_)
  have hx : Tendsto (fun t : ℝ => T t x) (𝓝 0) (𝓝 x) := by
    rw [tendsto_subtype_rng]
    simp_rw [hT]
    simpa using (hasDerivAt_sphereExp_smul (x : EucSpace d) (ξ x)).continuousAt.tendsto
  exact (f.continuous.tendsto x).comp hx

/-- The hypothesis of `geodesic_zero` and `tendsto_geodesic` is satisfiable: the
zero field and the constant curve, with `T = id`. -/
example (μ : Perspective.ProbSphere d) :
    IsGeodesicFrom (fun _ : SSphere d => (0 : EucSpace d)) μ (fun _ => μ) := fun _ =>
  ⟨id, measurable_id, fun x => by rw [smul_zero, sphereExp_zero]; rfl, Measure.map_id.symm⟩

/-! ### The second-derivative test -/

/-- **The second-derivative test at a global minimum**: if `J` is minimal at
`0` and continuous there, and `J''(0) = H` exists, then `H ≥ 0`.

Were `H < 0`, `J'` would be negative on some `(0, δ)`, since `J'(0) = 0`, and
`J` — continuous on `[0, δ/2]` — strictly decreasing there, below `J(0)`. -/
theorem nonneg_of_hasDerivAt_deriv_of_isMin {J : ℝ → ℝ} {H : ℝ} (hmin : ∀ t, J 0 ≤ J t)
    (hc : ContinuousAt J 0) (hH : HasDerivAt (deriv J) H 0) : 0 ≤ H := by
  refine le_of_not_gt fun hneg => ?_
  have h0 : deriv J 0 = 0 := (show IsLocalMin J 0 from Eventually.of_forall hmin).deriv_eq_zero
  have hsign := eventually_nhdsWithin_sign_eq_of_deriv_neg (hH.deriv.symm ▸ hneg) h0
  obtain ⟨δ, hδ, hδJ⟩ := mem_nhdsGT_iff_exists_Ioo_subset.mp
    (deriv_neg_right_of_sign_deriv (nhdsWithin_le_nhds hsign))
  have hδ0 : (0 : ℝ) < δ := hδ
  have hcont : ContinuousOn J (Set.Icc 0 (δ / 2)) := by
    intro y hy
    rcases eq_or_lt_of_le hy.1 with rfl | hy0
    · exact hc.continuousWithinAt
    · exact (differentiableAt_of_deriv_ne_zero
        (hδJ ⟨hy0, by linarith [hy.2]⟩ : deriv J y < 0).ne).continuousAt.continuousWithinAt
  have hanti := strictAntiOn_of_deriv_neg (convex_Icc 0 (δ / 2)) hcont fun y hy => by
    rw [interior_Icc] at hy
    exact hδJ ⟨hy.1, by linarith [hy.2]⟩
  have hlt : J (δ / 2) < J 0 :=
    hanti ⟨le_rfl, by positivity⟩ ⟨by positivity, le_rfl⟩ (by positivity)
  linarith [hmin (δ / 2)]

/-- The hypotheses of `nonneg_of_hasDerivAt_deriv_of_isMin` are satisfiable:
`J(t) = t²`, with `J' = 2t` and `J''(0) = 2`. -/
example : (∀ t : ℝ, (fun s : ℝ => s ^ 2) 0 ≤ (fun s : ℝ => s ^ 2) t) ∧
    ContinuousAt (fun s : ℝ => s ^ 2) 0 ∧
    HasDerivAt (deriv fun s : ℝ => s ^ 2) 2 0 := by
  refine ⟨fun t => by simp [sq_nonneg], (continuous_pow 2).continuousAt, ?_⟩
  have h : deriv (fun s : ℝ => s ^ 2) = fun s => 2 * s := funext fun s => by simp
  rw [h]
  simpa using (hasDerivAt_id (0 : ℝ)).const_mul 2

end Perceptron
end Transformer
