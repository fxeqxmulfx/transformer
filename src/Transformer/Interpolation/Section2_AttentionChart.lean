module

public import Transformer.Interpolation.Section2_AttentionHemisphere

/-!
# The original attention field in the hemisphere chart

ArXiv:2411.04551v3, §2, `prop: targets.atoms`, and §1, `eq: vf`.
Differentiating `x / ⟨x,w⟩` cancels the radial part of the projected
velocity. The induced consensus kernel is the original normalized
attention weight multiplied by the ratio of key and query heights.
On a hemisphere with margin `r`, this kernel is bounded below by
`exp(-2‖B‖) r`.

The chart is measurable everywhere, including its defined value on the
equator. A positive almost-everywhere height floor bounds it and gives
integrability under the actual law. Linear moments of the original
attention average are exactly the normalized scalar integrals.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The hemisphere chart cancels the radial projection term exactly.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem hasDerivAt_hemiChart_of_proj (Y : ℝ → EucSpace d) (w v : EucSpace d) (t : ℝ)
    (hY : HasDerivAt Y (proj d (Y t) v) t) (hs : inner ℝ (Y t) w ≠ 0) :
    HasDerivAt (fun s => Perspective.hemiChart w (Y s))
      ((inner ℝ (Y t) w)⁻¹ • v - (inner ℝ v w / (inner ℝ (Y t) w) ^ 2) • Y t) t := by
  have hh : HasDerivAt (fun s => inner ℝ (Y s) w)
      (inner ℝ v w - inner ℝ (Y t) v * inner ℝ (Y t) w) t := by
    convert hY.inner ℝ (hasDerivAt_const t w) using 1
    simp only [proj, inner_sub_left, real_inner_smul_left, inner_zero_right, zero_add]
  refine ((hh.fun_inv hs).smul hY).congr_deriv ?_
  have he : (inner ℝ (Y t) w)⁻¹ • proj d (Y t) v +
      (-(inner ℝ v w - inner ℝ (Y t) v * inner ℝ (Y t) w) / inner ℝ (Y t) w ^ 2) • Y t =
      (inner ℝ (Y t) w)⁻¹ • v +
        (-(inner ℝ v w - inner ℝ (Y t) v * inner ℝ (Y t) w) / inner ℝ (Y t) w ^ 2 -
          (inner ℝ (Y t) w)⁻¹ * inner ℝ (Y t) v) • Y t := by
    simp only [proj, smul_sub, smul_smul, sub_smul]
    abel
  rw [he]
  have hc : -(inner ℝ v w - inner ℝ (Y t) v * inner ℝ (Y t) w) / inner ℝ (Y t) w ^ 2 -
      (inner ℝ (Y t) w)⁻¹ * inner ℝ (Y t) v = -(inner ℝ v w / inner ℝ (Y t) w ^ 2) := by
    field_simp
    ring
  rw [hc, neg_smul, sub_eq_add_neg]

/-- The actual original attention weight times the chart's height ratio.
Source: arXiv:2411.04551v3, §2, characteristics of `prop: targets.atoms`. -/
noncomputable def attentionChartKernel (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (w : EucSpace d) (x y : SSphere d) : ℝ :=
  (exp (inner ℝ (B (x : EucSpace d)) (y : EucSpace d)) / Perspective.partitionMu d 1 ν (B x)) *
    (inner ℝ (y : EucSpace d) w / inner ℝ (x : EucSpace d) w)

/-- The original chart kernel has a uniform positive floor on the
supported hemisphere. Auxiliary for arXiv:2411.04551v3, §2. -/
theorem attentionChartKernel_lower (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (w : SSphere d) (x y : SSphere d) (r : ℝ) (hr : 0 < r)
    (hx : 0 < inner ℝ (x : EucSpace d) (w : EucSpace d))
    (hy : r ≤ inner ℝ (y : EucSpace d) (w : EucSpace d)) :
    exp (-(2 * ‖B‖)) * r ≤ attentionChartKernel B ν w x y := by
  have hx1 : inner ℝ (x : EucSpace d) (w : EucSpace d) ≤ 1 := by
    have hh := real_inner_le_norm (x : EucSpace d) (w : EucSpace d)
    simpa only [mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp w.2, mul_one] using hh
  have hratio : r ≤ inner ℝ (y : EucSpace d) (w : EucSpace d) /
      inner ℝ (x : EucSpace d) (w : EucSpace d) := by
    rw [le_div_iff₀ hx]
    exact (mul_le_of_le_one_right hr.le hx1).trans hy
  have ha := (attention_normalized_weight_bounds B ν x y).1
  exact mul_le_mul ha hratio hr.le ((exp_pos _).le.trans ha)

/-- Fixed-query chart kernels are continuous even at equatorial keys.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem continuous_attentionChartKernel (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (w : EucSpace d) (x : SSphere d) : Continuous (attentionChartKernel B ν w x) := by
  exact (((continuous_const.inner continuous_subtype_val).rexp.div_const _).mul
    ((continuous_subtype_val.inner continuous_const).div_const _))

/-- Compactness makes the actual chart kernel integrable.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem integrable_attentionChartKernel (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (w : EucSpace d) (x : SSphere d) : Integrable (attentionChartKernel B ν w x) (ν : Measure (SSphere d)) :=
  integrable_spherical_continuous ν _ (continuous_attentionChartKernel B ν w x)

/-- Every positive almost-everywhere height floor makes the chart
Bochner integrable. Auxiliary for arXiv:2411.04551v3, §2. -/
theorem integrable_hemiChart_of_ae_height (ν : Perspective.ProbSphere d) (w : EucSpace d)
    (r : ℝ) (hr : 0 < r)
    (hν : ∀ᵐ y : SSphere d ∂(ν : Measure (SSphere d)), r ≤ inner ℝ (y : EucSpace d) w) :
    Integrable (fun y : SSphere d => Perspective.hemiChart w (y : EucSpace d)) (ν : Measure (SSphere d)) := by
  have hm : Measurable (fun y : SSphere d => Perspective.hemiChart w (y : EucSpace d)) :=
    ((continuous_subtype_val.inner (𝕜 := ℝ) continuous_const).measurable.inv).smul continuous_subtype_val.measurable
  apply (integrable_const r⁻¹).mono' hm.aestronglyMeasurable
  filter_upwards [hν] with y hy
  rw [(Perspective.hemiChart_normalize (mem_sphere_zero_iff_norm.mp y.2) (hr.trans_le hy)).1]
  exact (inv_le_inv₀ (hr.trans_le hy) hr).mpr hy

/-- All linear chart coordinates are integrable on a supported positive
hemisphere. Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem integrable_inner_hemiChart_of_ae_height (ν : Perspective.ProbSphere d) (w u : EucSpace d)
    (r : ℝ) (hr : 0 < r)
    (hν : ∀ᵐ y : SSphere d ∂(ν : Measure (SSphere d)), r ≤ inner ℝ (y : EucSpace d) w) :
    Integrable (fun y : SSphere d => inner ℝ (Perspective.hemiChart w (y : EucSpace d)) u)
      (ν : Measure (SSphere d)) :=
  (integrable_hemiChart_of_ae_height ν w r hr hν).inner_const u

/-- A linear moment of the actual attention average equals its original
normalized scalar integral. Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem inner_attentionAverage_eq_integral (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (x u : EucSpace d) : inner ℝ (attentionAverage B ν x) u =
      ∫ y : SSphere d, (exp (inner ℝ (B x) (y : EucSpace d)) /
        Perspective.partitionMu d 1 ν (B x)) * inner ℝ (y : EucSpace d) u ∂(ν : Measure (SSphere d)) := by
  have hi : Integrable (fun y : SSphere d => exp (inner ℝ (B x) (y : EucSpace d)) •
      (y : EucSpace d)) (ν : Measure (SSphere d)) := by
    simpa only [one_mul] using Perspective.integrable_expInner_smul d 1 ν (B x)
  have he : inner ℝ (∫ y : SSphere d, exp (inner ℝ (B x) (y : EucSpace d)) •
      (y : EucSpace d) ∂(ν : Measure (SSphere d))) u =
      ∫ y : SSphere d, exp (inner ℝ (B x) (y : EucSpace d)) * inner ℝ (y : EucSpace d) u
        ∂(ν : Measure (SSphere d)) := by
    rw [real_inner_comm, ← integral_inner (𝕜 := ℝ) hi u]
    apply integral_congr_ae
    exact ae_of_all _ fun y => by
      change inner ℝ u (exp (inner ℝ (B x) (y : EucSpace d)) • (y : EucSpace d)) =
        exp (inner ℝ (B x) (y : EucSpace d)) * inner ℝ (y : EucSpace d) u
      rw [real_inner_smul_right, real_inner_comm (y : EucSpace d) u]
  rw [attentionAverage, real_inner_smul_left, he, ← integral_const_mul]
  apply integral_congr_ae
  exact ae_of_all _ fun y => by dsimp; ring

/-- A genuine Dirac law at a unit point, a stationary projected
trajectory, and a positive hemisphere margin jointly realize every
chart derivative, kernel bound and integrability hypothesis. -/
example : ∃ (B : ParamMatrix 2) (ν : Perspective.ProbSphere 2) (p : SSphere 2)
    (Y : ℝ → EucSpace 2),
    (∀ t, HasDerivAt Y (proj 2 (Y t) (p : EucSpace 2)) t) ∧
    inner ℝ (Y 0) (p : EucSpace 2) ≠ 0 ∧
    (0 : ℝ) < inner ℝ (p : EucSpace 2) (p : EucSpace 2) ∧
    (1 : ℝ) ≤ inner ℝ (p : EucSpace 2) (p : EucSpace 2) ∧
    (∀ᵐ y : SSphere 2 ∂(ν : Measure (SSphere 2)), (1 : ℝ) ≤ inner ℝ (y : EucSpace 2) (p : EucSpace 2)) ∧
    (0 : ℝ) < 1 ∧ B (p : EucSpace 2) = p := by
  let p : SSphere 2 := basePoint 1
  have hp : ‖(p : EucSpace 2)‖ = 1 := mem_sphere_zero_iff_norm.mp p.2
  have hh : inner ℝ (p : EucSpace 2) (p : EucSpace 2) = 1 := by
    rw [real_inner_self_eq_norm_sq, hp]
    norm_num
  have hproj : proj 2 (p : EucSpace 2) (p : EucSpace 2) = 0 := by
    rw [proj, hh, one_smul, sub_self]
  refine ⟨ContinuousLinearMap.id ℝ (EucSpace 2), Perspective.diracProb 2 p, p,
    (fun _ => p), ?_, by simpa only [hh] using one_ne_zero, by rw [hh]; exact zero_lt_one,
    hh.ge, ?_, zero_lt_one, rfl⟩
  · intro t
    rw [hproj]
    exact hasDerivAt_const t (p : EucSpace 2)
  · filter_upwards [Measure.support_mem_ae (μ := (Perspective.diracProb 2 p : Measure (SSphere 2)))] with y hy
    rw [eq_of_mem_support_dirac hy, hh]

end Transformer.Interpolation
