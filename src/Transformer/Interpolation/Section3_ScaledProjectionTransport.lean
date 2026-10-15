/-
# Characteristic transport at the source's rescaled phase speeds

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3.1,
the final time-rescaling step of `lem: first.quadrant`.
Every original weak solution on a phase with field `κ Proj_x w`
is the actual pushforward at time `κ(b-a)`. The inverse map, support
transport and endpoint uniqueness use the same speed. These conclusions
retain both endpoints of the phase, including control switches.
The speed may vanish or be negative: the zero field has the identity
map, and a negative speed runs the same sphere group backward.
The test argument therefore needs no positivity assumption on the speed.
-/

import Transformer.Interpolation.Section3_ScaledProjectionTests
import Transformer.Interpolation.Section3_ProjectionTransport

open Real MeasureTheory

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Every original weak solution obeys the exact characteristic formula
after the source's constant time rescaling.
Source: arXiv:2411.04551v3, §1 and §3.1, `lem: first.quadrant`. -/
theorem cauchyPB_scaled_projection_map (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (w : SSphere d) (κ a b : ℝ) (hab : a < b)
    (hfield : ∀ s ∈ Set.Ioo a b, ∀ x : SSphere d,
      fullVF d θ (μ s) s (x : EucSpace d) = κ • proj d (x : EucSpace d) (w : EucSpace d)) :
    (μ b : Measure (SSphere d)) =
      Measure.map (sphereProjectionFlow w (κ * (b - a))) (μ a : Measure (SSphere d)) := by
  have hw : ‖(w : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp w.2
  have hf : Continuous (sphereProjectionFlow w (κ * (b - a))) :=
    (continuous_sphereProjectionFlow w).comp (continuous_const.prodMk continuous_id)
  apply Perspective.measure_eq_of_integral_mono_eq
  intro k I
  let ψ (y : EucSpace d) := Perspective.mono I (projectionExtension (w : EucSpace d) (κ * b) y)
  have hψ : ContDiff ℝ (⊤ : ℕ∞) ψ :=
    ContDiff.comp (f := projectionExtension (w : EucSpace d) (κ * b)) (g := Perspective.mono I)
      (Perspective.contDiff_mono I ⊤) (contDiff_projectionExtension_slice _ hw (κ * b))
  have hcon := scaled_projection_pullback_integral_eq θ μ hμ (w : EucSpace d) hw κ a b
    hab hfield ψ hψ
  have hleft (x : SSphere d) :
      ψ (projectionExtension (w : EucSpace d) (-(κ * b)) (x : EucSpace d)) =
        Perspective.mono I (x : EucSpace d) := by
    dsimp only [ψ]
    rw [projectionExtension_comp _ _ hw (mem_sphere_zero_iff_norm.mp x.2),
      add_neg_cancel, projectionFlow_zero]
  have hright (x : SSphere d) :
      ψ (projectionExtension (w : EucSpace d) (-(κ * a)) (x : EucSpace d)) =
        Perspective.mono I (sphereProjectionFlow w (κ * (b - a)) x : EucSpace d) := by
    dsimp only [ψ]
    rw [projectionExtension_comp _ _ hw (mem_sphere_zero_iff_norm.mp x.2),
      show κ * b + -(κ * a) = κ * (b - a) by ring]
    rfl
  calc
    _ = ∫ x, ψ (projectionExtension (w : EucSpace d) (-(κ * b)) (x : EucSpace d))
        ∂(μ b : Measure (SSphere d)) := integral_congr_ae (ae_of_all _ fun x => (hleft x).symm)
    _ = ∫ x, ψ (projectionExtension (w : EucSpace d) (-(κ * a)) (x : EucSpace d))
        ∂(μ a : Measure (SSphere d)) := hcon
    _ = ∫ x, Perspective.mono I (sphereProjectionFlow w (κ * (b - a)) x : EucSpace d)
        ∂(μ a : Measure (SSphere d)) := integral_congr_ae (ae_of_all _ hright)
    _ = _ := (integral_map hf.measurable.aemeasurable
      ((Perspective.continuous_mono I).comp continuous_subtype_val).measurable.aestronglyMeasurable).symm

/-- An initial phase retains its exact original initial law.
Source: arXiv:2411.04551v3, §1 and §3.1, `lem: first.quadrant`. -/
theorem cauchyPB_scaled_projection_map_initial (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (w : SSphere d) (κ T : ℝ) (hT : 0 < T)
    (hfield : ∀ s ∈ Set.Ioo 0 T, ∀ x : SSphere d,
      fullVF d θ (μ s) s (x : EucSpace d) = κ • proj d (x : EucSpace d) (w : EucSpace d)) :
    (μ T : Measure (SSphere d)) =
      Measure.map (sphereProjectionFlow w (κ * T)) (μ 0 : Measure (SSphere d)) := by
  simpa only [sub_zero] using cauchyPB_scaled_projection_map θ μ hμ w κ 0 T hT hfield

/-- The actual inverse flow recovers the original law after rescaling.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem cauchyPB_scaled_projection_inverse_map (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (w : SSphere d) (κ T : ℝ) (hT : 0 < T)
    (hfield : ∀ s ∈ Set.Ioo 0 T, ∀ x : SSphere d,
      fullVF d θ (μ s) s (x : EucSpace d) = κ • proj d (x : EucSpace d) (w : EucSpace d)) :
    Measure.map (sphereProjectionFlow w (-(κ * T))) (μ T : Measure (SSphere d)) =
      (μ 0 : Measure (SSphere d)) := by
  have hm (s : ℝ) : Measurable (sphereProjectionFlow w s) :=
    ((continuous_sphereProjectionFlow w).comp (continuous_const.prodMk continuous_id)).measurable
  rw [cauchyPB_scaled_projection_map_initial θ μ hμ w κ T hT hfield,
    Measure.map_map (hm (-(κ * T))) (hm (κ * T))]
  have heq : sphereProjectionFlow w (-(κ * T)) ∘ sphereProjectionFlow w (κ * T) = id :=
    funext fun x => sphereProjectionFlow_neg w x (κ * T)
  rw [heq, Measure.map_id]

/-- The original support moves by the exact rescaled characteristic map.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem cauchyPB_scaled_projection_support (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (w : SSphere d) (κ a b : ℝ) (hab : a < b)
    (hfield : ∀ s ∈ Set.Ioo a b, ∀ x : SSphere d,
      fullVF d θ (μ s) s (x : EucSpace d) = κ • proj d (x : EucSpace d) (w : EucSpace d)) :
    (μ b : Measure (SSphere d)).support ⊆
      sphereProjectionFlow w (κ * (b - a)) '' (μ a : Measure (SSphere d)).support := by
  rw [cauchyPB_scaled_projection_map θ μ hμ w κ a b hab hfield]
  exact support_map_subset_image _ _
    ((continuous_sphereProjectionFlow w).comp (continuous_const.prodMk continuous_id))

/-- The first rescaled phase carries the original initial support by
the exact flow map used in the source's construction.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem cauchyPB_scaled_projection_support_initial (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (w : SSphere d) (κ T : ℝ) (hT : 0 < T)
    (hfield : ∀ s ∈ Set.Ioo 0 T, ∀ x : SSphere d,
      fullVF d θ (μ s) s (x : EucSpace d) = κ • proj d (x : EucSpace d) (w : EucSpace d)) :
    (μ T : Measure (SSphere d)).support ⊆
      sphereProjectionFlow w (κ * T) '' (μ 0 : Measure (SSphere d)).support := by
  simpa only [sub_zero] using cauchyPB_scaled_projection_support θ μ hμ w κ 0 T hT hfield

/-- Actual weak solutions with equal initial laws have equal endpoint
laws at the same rescaled speed. Source: arXiv:2411.04551v3,
§1 and §3.1, `lem: first.quadrant`. -/
theorem cauchyPB_scaled_projection_endpoint_unique (θ : TimeParams d)
    (μ ν : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) (hν : cauchyPB d θ ν)
    (w : SSphere d) (κ a b : ℝ) (hab : a < b) (hinit : μ a = ν a)
    (hμfield : ∀ s ∈ Set.Ioo a b, ∀ x : SSphere d,
      fullVF d θ (μ s) s (x : EucSpace d) = κ • proj d (x : EucSpace d) (w : EucSpace d))
    (hνfield : ∀ s ∈ Set.Ioo a b, ∀ x : SSphere d,
      fullVF d θ (ν s) s (x : EucSpace d) = κ • proj d (x : EucSpace d) (w : EucSpace d)) :
    (μ b : Measure (SSphere d)) = (ν b : Measure (SSphere d)) := by
  rw [cauchyPB_scaled_projection_map θ μ hμ w κ a b hab hμfield,
    cauchyPB_scaled_projection_map θ ν hν w κ a b hab hνfield, hinit]

/-- All hypotheses hold simultaneously at a nonzero actual weak solution
and a nonzero speed. The repeated law also witnesses endpoint uniqueness.
Source: arXiv:2411.04551v3, §1 and §3.1, `lem: first.quadrant`. -/
example : cauchyPB 2 (pureAttentionParams 2 0)
      (fun _ => Perspective.diracProb 2 (basePoint 1)) ∧ (0 : ℝ) < 1 ∧
    ((fun _ : ℝ => Perspective.diracProb 2 (basePoint 1)) 0 =
      (fun _ : ℝ => Perspective.diracProb 2 (basePoint 1)) 0) ∧
    ∀ s ∈ Set.Ioo (0 : ℝ) 1, ∀ x : SSphere 2,
      fullVF 2 (pureAttentionParams 2 0) (Perspective.diracProb 2 (basePoint 1)) s
        (x : EucSpace 2) = (1 : ℝ) • proj 2 (x : EucSpace 2) (basePoint 1 : EucSpace 2) := by
  refine ⟨cauchyPB_pureAttention_dirac 0 (basePoint 1), one_pos, rfl, ?_⟩
  intro s _ x
  rw [one_smul, fullVF_pureAttention_zero]
  simp [Perspective.diracProb]

end Transformer.Interpolation
