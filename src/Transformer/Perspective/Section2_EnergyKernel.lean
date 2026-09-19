/-
# §3.2 — The kernel `exp(β⟨x,y⟩)` and the partition function on the sphere

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, §3.1–§3.2.

The analytic groundwork under `eq: interaction.energy` and
`prop: existence.uniqueness.energy`: the kernel of `eq: partition.function`
is continuous in each argument and bounded by `e^β` on the sphere, because
`⟨x, x'⟩ ≤ 1` there — with equality exactly at `x = x'`, which is the fact
the maximiser half of `prop: existence.uniqueness.energy` turns on.  Hence
`Z_{β,μ}` of `eq: partition.function` is itself continuous and bounded on the
sphere, so integrable against any probability measure on it.

Used by `Perspective.Section2_EnergyMax`.
-/

import Transformer.Perspective.Section2_FlowMap
import Mathlib.MeasureTheory.Integral.Bochner.Basic

open Real MeasureTheory

namespace Transformer
namespace Perspective

variable (d : ℕ)

/-! ### The kernel on the sphere -/

/-- On the unit sphere the inner product never exceeds `1`. -/
theorem inner_sphere_le_one (x y : SSphere d) :
    inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d) ≤ 1 := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hy : ‖(y : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp y.2
  calc inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)
      ≤ ‖(x : EucSpace d)‖ * ‖(y : EucSpace d)‖ := real_inner_le_norm _ _
    _ = 1 := by rw [hx, hy, one_mul]

/-- Equality `⟨x, y⟩ = 1` on the sphere forces `x = y`: the Cauchy–Schwarz
case of equality, in the form `‖x - y‖² = 2 - 2⟨x, y⟩`. -/
theorem eq_of_inner_sphere_eq_one {x y : SSphere d}
    (h : inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d) = 1) : x = y := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hy : ‖(y : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp y.2
  have hsq : ‖(x : EucSpace d) - (y : EucSpace d)‖ ^ 2 = 0 := by
    rw [norm_sub_sq_real, hx, hy, h]; ring
  have hzero : (x : EucSpace d) - (y : EucSpace d) = 0 :=
    norm_eq_zero.mp (pow_eq_zero_iff two_ne_zero |>.mp hsq)
  exact Subtype.ext (sub_eq_zero.mp hzero)

/-- The kernel is bounded by `e^β` for `β ≥ 0`. -/
theorem exp_inner_le (β : ℝ) (hβ : 0 ≤ β) (x y : SSphere d) :
    Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) ≤ Real.exp β := by
  refine Real.exp_le_exp.mpr ?_
  nlinarith [inner_sphere_le_one d x y]

/-- The kernel is continuous in its second argument. -/
theorem continuous_expInner_right (β : ℝ) (x : SSphere d) :
    Continuous fun y : SSphere d =>
      Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) :=
  (continuous_const.mul (continuous_const.inner continuous_subtype_val)).rexp

/-- The kernel is continuous in its first argument. -/
theorem continuous_expInner_left (β : ℝ) (y : SSphere d) :
    Continuous fun x : SSphere d =>
      Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) :=
  (continuous_const.mul (continuous_subtype_val.inner continuous_const)).rexp

/-! ### The partition function as a function on the sphere -/

/-- The kernel is integrable against any probability measure on the sphere:
it is continuous and bounded by `e^β`. -/
theorem integrable_expInner (β : ℝ) (hβ : 0 ≤ β) (μ : ProbSphere d) (x : SSphere d) :
    Integrable (fun y : SSphere d =>
        Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)))
      (μ : Measure (SSphere d)) := by
  refine (integrable_const (Real.exp β)).mono'
    (continuous_expInner_right d β x).aestronglyMeasurable
    (Filter.Eventually.of_forall fun y => ?_)
  rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
  exact exp_inner_le d β hβ x y

/-- `Z_{β,μ} ≥ 0`: it is the integral of a positive function. -/
theorem partitionMu_nonneg (β : ℝ) (μ : ProbSphere d) (x : EucSpace d) :
    0 ≤ partitionMu d β μ x :=
  integral_nonneg fun _ => (Real.exp_pos _).le

/-- `Z_{β,μ}(x) ≤ e^β` for `x` on the sphere and `β ≥ 0`. -/
theorem partitionMu_le (β : ℝ) (hβ : 0 ≤ β) (μ : ProbSphere d) (x : SSphere d) :
    partitionMu d β μ (x : EucSpace d) ≤ Real.exp β := by
  rw [partitionMu]
  calc ∫ y, Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))
          ∂(μ : Measure (SSphere d))
      ≤ ∫ _ : SSphere d, Real.exp β ∂(μ : Measure (SSphere d)) :=
        integral_mono (integrable_expInner d β hβ μ x) (integrable_const _)
          (fun y => exp_inner_le d β hβ x y)
    _ = Real.exp β := by simp

/-- `Z_{β,μ}` is continuous on the sphere: dominated convergence with the
constant bound `e^β`. -/
theorem continuous_partitionMu (β : ℝ) (hβ : 0 ≤ β) (μ : ProbSphere d) :
    Continuous fun x : SSphere d => partitionMu d β μ (x : EucSpace d) := by
  refine continuous_of_dominated (bound := fun _ : SSphere d => Real.exp β)
    (fun x => (continuous_expInner_right d β x).aestronglyMeasurable)
    (fun x => Filter.Eventually.of_forall fun y => ?_) (integrable_const _)
    (Filter.Eventually.of_forall fun y => continuous_expInner_left d β y)
  rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
  exact exp_inner_le d β hβ x y

/-- `Z_{β,μ}` is integrable on the sphere: continuous and bounded. -/
theorem integrable_partitionMu (β : ℝ) (hβ : 0 ≤ β) (μ : ProbSphere d) :
    Integrable (fun x : SSphere d => partitionMu d β μ (x : EucSpace d))
      (μ : Measure (SSphere d)) := by
  refine (integrable_const (Real.exp β)).mono'
    (continuous_partitionMu d β hβ μ).aestronglyMeasurable
    (Filter.Eventually.of_forall fun x => ?_)
  rw [Real.norm_eq_abs, abs_of_nonneg (partitionMu_nonneg d β μ _)]
  exact partitionMu_le d β hβ μ x
end Perspective
end Transformer
