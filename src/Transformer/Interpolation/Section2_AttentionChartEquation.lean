/-
# The exact consensus equation in the hemisphere chart

ArXiv:2411.04551v3, §2, `prop: targets.atoms`, and §1, `eq: vf`.
At nonzero query and key heights the chart kernel cancels the key's
height denominator. The resulting gap integrand is almost everywhere
equal to a difference of continuous normalized linear moments. This
proves its integrability and computes its integral exactly.

Combining that identity with the derivative of the actual projected
particle gives the integral consensus equation for each chart
coordinate. On supported characteristics the proved positive hemisphere
margin supplies every nonzero-height premise.
-/

import Transformer.Interpolation.Section2_AttentionChart

open Real MeasureTheory Set
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Exact cancellation of the key height in the chart gap integrand.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem attentionChartKernel_gap_eq_ae (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (w u : EucSpace d) (x : SSphere d) (hx : inner ℝ (x : EucSpace d) w ≠ 0)
    (hν : ∀ᵐ y : SSphere d ∂(ν : Measure (SSphere d)), inner ℝ (y : EucSpace d) w ≠ 0) :
    (fun y : SSphere d => attentionChartKernel B ν w x y *
      (inner ℝ (Perspective.hemiChart w (y : EucSpace d)) u -
        inner ℝ (Perspective.hemiChart w (x : EucSpace d)) u)) =ᵐ[(ν : Measure (SSphere d))]
    (fun y : SSphere d => (inner ℝ (x : EucSpace d) w)⁻¹ *
      ((exp (inner ℝ (B (x : EucSpace d)) (y : EucSpace d)) / Perspective.partitionMu d 1 ν (B x)) *
        inner ℝ (y : EucSpace d) u) - (inner ℝ (x : EucSpace d) u / inner ℝ (x : EucSpace d) w ^ 2) *
      ((exp (inner ℝ (B (x : EucSpace d)) (y : EucSpace d)) / Perspective.partitionMu d 1 ν (B x)) *
        inner ℝ (y : EucSpace d) w)) := by
  filter_upwards [hν] with y hy
  dsimp only [attentionChartKernel, Perspective.hemiChart]
  simp only [real_inner_smul_left]
  field_simp [hx, hy, (Perspective.partitionMu_pos d 1 ν (B x)).ne']

/-- The actual consensus gap is integrable whenever the relevant
heights are nonzero. Auxiliary for arXiv:2411.04551v3, §2. -/
theorem integrable_attentionChartKernel_gap (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (w u : EucSpace d) (x : SSphere d) (hx : inner ℝ (x : EucSpace d) w ≠ 0)
    (hν : ∀ᵐ y : SSphere d ∂(ν : Measure (SSphere d)), inner ℝ (y : EucSpace d) w ≠ 0) :
    Integrable (fun y : SSphere d => attentionChartKernel B ν w x y *
      (inner ℝ (Perspective.hemiChart w (y : EucSpace d)) u -
        inner ℝ (Perspective.hemiChart w (x : EucSpace d)) u)) (ν : Measure (SSphere d)) := by
  have hi (v : EucSpace d) : Integrable (fun y : SSphere d =>
      (exp (inner ℝ (B (x : EucSpace d)) (y : EucSpace d)) / Perspective.partitionMu d 1 ν (B x)) *
        inner ℝ (y : EucSpace d) v) (ν : Measure (SSphere d)) :=
    integrable_spherical_continuous ν _
      (((continuous_const.inner continuous_subtype_val).rexp.div_const _).mul
        (continuous_subtype_val.inner continuous_const))
  exact (((hi u).const_mul (inner ℝ (x : EucSpace d) w)⁻¹).sub
    ((hi w).const_mul (inner ℝ (x : EucSpace d) u / inner ℝ (x : EucSpace d) w ^ 2))).congr
    (attentionChartKernel_gap_eq_ae B ν w u x hx hν).symm

/-- Computing the chart gap integral gives the actual chart velocity.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem integral_attentionChartKernel_gap (B : ParamMatrix d) (ν : Perspective.ProbSphere d)
    (w u : EucSpace d) (x : SSphere d) (hx : inner ℝ (x : EucSpace d) w ≠ 0)
    (hν : ∀ᵐ y : SSphere d ∂(ν : Measure (SSphere d)), inner ℝ (y : EucSpace d) w ≠ 0) :
    (∫ y : SSphere d, attentionChartKernel B ν w x y *
      (inner ℝ (Perspective.hemiChart w (y : EucSpace d)) u -
        inner ℝ (Perspective.hemiChart w (x : EucSpace d)) u) ∂(ν : Measure (SSphere d))) =
      (inner ℝ (x : EucSpace d) w)⁻¹ * inner ℝ (attentionAverage B ν x) u -
        (inner ℝ (x : EucSpace d) u / inner ℝ (x : EucSpace d) w ^ 2) *
          inner ℝ (attentionAverage B ν x) w := by
  have hi (v : EucSpace d) : Integrable (fun y : SSphere d =>
      (exp (inner ℝ (B (x : EucSpace d)) (y : EucSpace d)) / Perspective.partitionMu d 1 ν (B x)) *
        inner ℝ (y : EucSpace d) v) (ν : Measure (SSphere d)) :=
    integrable_spherical_continuous ν _
      (((continuous_const.inner continuous_subtype_val).rexp.div_const _).mul
        (continuous_subtype_val.inner continuous_const))
  rw [integral_congr_ae (attentionChartKernel_gap_eq_ae B ν w u x hx hν),
    integral_sub ((hi u).const_mul _) ((hi w).const_mul _), integral_const_mul, integral_const_mul,
    ← inner_attentionAverage_eq_integral, ← inner_attentionAverage_eq_integral]

/-- Every actual attention particle satisfies the original integral
consensus equation in chart coordinates. Auxiliary for
arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem hasDerivAt_inner_hemiChart_attentionFlow (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (w u : EucSpace d) (x : SSphere d) (t : ℝ)
    (hx : inner ℝ (attentionFlow B μ hμ x t : EucSpace d) w ≠ 0)
    (hν : ∀ᵐ y : SSphere d ∂(μ t : Measure (SSphere d)), inner ℝ (y : EucSpace d) w ≠ 0) :
    HasDerivAt (fun s => inner ℝ (Perspective.hemiChart w (attentionFlow B μ hμ x s : EucSpace d)) u)
      (∫ y : SSphere d, attentionChartKernel B (μ t) w (attentionFlow B μ hμ x t) y *
        (inner ℝ (Perspective.hemiChart w (y : EucSpace d)) u -
          inner ℝ (Perspective.hemiChart w (attentionFlow B μ hμ x t : EucSpace d)) u)
        ∂(μ t : Measure (SSphere d))) t := by
  have hd := hasDerivAt_attentionFlow B μ hμ x t
  rw [fullVF_eq_proj_attentionAverage] at hd
  have hh := (hasDerivAt_hemiChart_of_proj _ w _ t hd hx).inner ℝ (hasDerivAt_const t u)
  apply hh.congr_deriv
  rw [integral_attentionChartKernel_gap B (μ t) w u _ hx hν]
  simp only [inner_sub_left, real_inner_smul_left, inner_zero_right]
  ring

/-- The positive initial support margin supplies all chart-consensus
premises along every supported characteristic. Auxiliary for
arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem hasDerivAt_inner_hemiChart_attentionFlow_of_support
    (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (w u : EucSpace d) (r : ℝ) (hr : 0 < r)
    (h0 : ∀ y ∈ (μ 0 : Measure (SSphere d)).support, r ≤ inner ℝ w (y : EucSpace d))
    (x : SSphere d) (hx : x ∈ (μ 0 : Measure (SSphere d)).support) (t : ℝ) (ht : 0 ≤ t) :
    HasDerivAt (fun s => inner ℝ (Perspective.hemiChart w (attentionFlow B μ hμ x s : EucSpace d)) u)
      (∫ y : SSphere d, attentionChartKernel B (μ t) w (attentionFlow B μ hμ x t) y *
        (inner ℝ (Perspective.hemiChart w (y : EucSpace d)) u -
          inner ℝ (Perspective.hemiChart w (attentionFlow B μ hμ x t : EucSpace d)) u)
        ∂(μ t : Measure (SSphere d))) t := by
  apply hasDerivAt_inner_hemiChart_attentionFlow B μ hμ w u x t
  · rw [real_inner_comm]
    exact (hr.trans_le (le_height_attentionFlow_of_support B μ hμ w r hr h0 x hx t ht)).ne'
  · filter_upwards [Measure.support_mem_ae (μ := (μ t : Measure (SSphere d)))] with y hy
    rw [real_inner_comm]
    exact (hr.trans_le (le_height_attention_support B μ hμ w r hr h0 t ht y hy)).ne'

/-- A genuine nonzero attention weak law in a positive supporting
hemisphere jointly realizes the original weak, nonzero-height and
supported-chart hypotheses, at a nonnegative time. -/
example : ∃ (B : ParamMatrix 2) (μ : ℝ → Perspective.ProbSphere 2)
    (hμ : cauchyPB 2 (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace 2), B, 0, 0, 0⟩) μ)
    (w x : SSphere 2),
    (∀ y ∈ (μ 0 : Measure (SSphere 2)).support, (1 : ℝ) ≤ inner ℝ (w : EucSpace 2) (y : EucSpace 2)) ∧
    x ∈ (μ 0 : Measure (SSphere 2)).support ∧ inner ℝ (x : EucSpace 2) (w : EucSpace 2) ≠ 0 ∧
    (∀ᵐ y : SSphere 2 ∂(μ 1 : Measure (SSphere 2)), inner ℝ (y : EucSpace 2) (w : EucSpace 2) ≠ 0) ∧
    inner ℝ (attentionFlow B μ hμ x 1 : EucSpace 2) (w : EucSpace 2) ≠ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 := by
  let B := ContinuousLinearMap.id ℝ (EucSpace 2)
  let p : SSphere 2 := basePoint 1
  let μ : ℝ → Perspective.ProbSphere 2 := fun _ => Perspective.diracProb 2 p
  have hθ : pureAttentionParams 2 1 = (fun _ => ⟨B, B, 0, 0, 0⟩) := by
    funext t
    simp only [pureAttentionParams, B, one_smul]
  have hμ : cauchyPB 2 (fun _ => ⟨B, B, 0, 0, 0⟩) μ := by
    rw [← hθ]
    exact cauchyPB_pureAttention_dirac 1 p
  have hh : inner ℝ (p : EucSpace 2) (p : EucSpace 2) = 1 := by
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp p.2]
    norm_num
  have h0 (y : SSphere 2) (hy : y ∈ (μ 0 : Measure (SSphere 2)).support) :
      (1 : ℝ) ≤ inner ℝ (p : EucSpace 2) (y : EucSpace 2) := by
    rw [eq_of_mem_support_dirac hy, hh]
  obtain ⟨x, hx⟩ := Measure.nonempty_support (IsProbabilityMeasure.ne_zero (μ 0 : Measure (SSphere 2)))
  refine ⟨B, μ, hμ, p, x, h0, hx, ?_, ?_, ?_, zero_lt_one, zero_le_one⟩
  · rw [eq_of_mem_support_dirac hx, hh]
    exact one_ne_zero
  · filter_upwards [Measure.support_mem_ae (μ := (μ 1 : Measure (SSphere 2)))] with y hy
    rw [eq_of_mem_support_dirac hy, hh]
    exact one_ne_zero
  · rw [real_inner_comm]
    exact (zero_lt_one.trans_le (le_height_attentionFlow_of_support B μ hμ p 1 zero_lt_one h0 x hx 1 zero_le_one)).ne'

end Transformer.Interpolation
