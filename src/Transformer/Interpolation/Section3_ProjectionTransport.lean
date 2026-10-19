module

public import Transformer.Interpolation.Section3_ProjectionWeakTests
public import Transformer.Perspective.SphereMoments

/-!
# The characteristic formula for every original weak solution

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3.1,
`lem: first.quadrant`. Conservation of inverse-flow tests determines
the original measure by its polynomial moments. Consequently every
original weak solution on a constant-projection phase is exactly the
pushforward by the constructed sphere flow. Its support is contained
in the image of its original support, and equal initial laws give
equal endpoint laws. The measure curve has no extra representation
or regularity assumption.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Every original weak solution has the actual characteristic formula
on a constant-projection phase, including both endpoints.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3.1,
`lem: first.quadrant`. -/
theorem cauchyPB_projection_map (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (w : SSphere d) (a b : ℝ) (hab : a < b)
    (hfield : ∀ s ∈ Set.Ioo a b, ∀ x : SSphere d,
      fullVF d θ (μ s) s (x : EucSpace d) = proj d (x : EucSpace d) (w : EucSpace d)) :
    (μ b : Measure (SSphere d)) =
      Measure.map (sphereProjectionFlow w (b - a)) (μ a : Measure (SSphere d)) := by
  have hw : ‖(w : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp w.2
  have hf : Continuous (sphereProjectionFlow w (b - a)) :=
    (continuous_sphereProjectionFlow w).comp (continuous_const.prodMk continuous_id)
  apply Perspective.measure_eq_of_integral_mono_eq
  intro k I
  let ψ (y : EucSpace d) := Perspective.mono I (projectionExtension (w : EucSpace d) b y)
  have hψ : ContDiff ℝ (⊤ : ℕ∞) ψ :=
    ContDiff.comp (f := projectionExtension (w : EucSpace d) b) (g := Perspective.mono I)
      (Perspective.contDiff_mono I ⊤) (contDiff_projectionExtension_slice _ hw b)
  have hcon := projection_pullback_integral_eq θ μ hμ (w : EucSpace d) hw a b hab hfield ψ hψ
  have hleft (x : SSphere d) :
      ψ (projectionExtension (w : EucSpace d) (-b) (x : EucSpace d)) =
        Perspective.mono I (x : EucSpace d) := by
    dsimp only [ψ]
    rw [projectionExtension_comp _ _ hw (mem_sphere_zero_iff_norm.mp x.2),
      add_neg_cancel, projectionFlow_zero]
  have hright (x : SSphere d) :
      ψ (projectionExtension (w : EucSpace d) (-a) (x : EucSpace d)) =
        Perspective.mono I (sphereProjectionFlow w (b - a) x : EucSpace d) := by
    dsimp only [ψ]
    rw [projectionExtension_comp _ _ hw (mem_sphere_zero_iff_norm.mp x.2)]
    rfl
  calc
    _ = ∫ x, ψ (projectionExtension (w : EucSpace d) (-b) (x : EucSpace d))
        ∂(μ b : Measure (SSphere d)) := integral_congr_ae (ae_of_all _ fun x => (hleft x).symm)
    _ = ∫ x, ψ (projectionExtension (w : EucSpace d) (-a) (x : EucSpace d))
        ∂(μ a : Measure (SSphere d)) := hcon
    _ = ∫ x, Perspective.mono I (sphereProjectionFlow w (b - a) x : EucSpace d)
        ∂(μ a : Measure (SSphere d)) := integral_congr_ae (ae_of_all _ hright)
    _ = _ := (integral_map hf.measurable.aemeasurable
      ((Perspective.continuous_mono I).comp continuous_subtype_val).measurable.aestronglyMeasurable).symm

/-- A first phase transports the original initial law by the actual
sphere flow. Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem cauchyPB_projection_map_initial (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (w : SSphere d) (T : ℝ) (hT : 0 < T)
    (hfield : ∀ s ∈ Set.Ioo 0 T, ∀ x : SSphere d,
      fullVF d θ (μ s) s (x : EucSpace d) = proj d (x : EucSpace d) (w : EucSpace d)) :
    (μ T : Measure (SSphere d)) =
      Measure.map (sphereProjectionFlow w T) (μ 0 : Measure (SSphere d)) := by
  simpa only [sub_zero] using cauchyPB_projection_map θ μ hμ w 0 T hT hfield

/-- The inverse characteristic recovers the original initial law exactly.
Source: arXiv:2411.04551v3, §1 and §3.1, `lem: first.quadrant`. -/
theorem cauchyPB_projection_inverse_map (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (w : SSphere d) (T : ℝ) (hT : 0 < T)
    (hfield : ∀ s ∈ Set.Ioo 0 T, ∀ x : SSphere d,
      fullVF d θ (μ s) s (x : EucSpace d) = proj d (x : EucSpace d) (w : EucSpace d)) :
    Measure.map (sphereProjectionFlow w (-T)) (μ T : Measure (SSphere d)) =
      (μ 0 : Measure (SSphere d)) := by
  have hm (s : ℝ) : Measurable (sphereProjectionFlow w s) :=
    ((continuous_sphereProjectionFlow w).comp
      (continuous_const.prodMk continuous_id)).measurable
  rw [cauchyPB_projection_map_initial θ μ hμ w T hT hfield, Measure.map_map (hm (-T)) (hm T)]
  have heq : sphereProjectionFlow w (-T) ∘ sphereProjectionFlow w T = id :=
    funext fun x => sphereProjectionFlow_neg w x T
  rw [heq, Measure.map_id]

/-- Continuous maps of the compact original sphere carry the support
into the image of the original support. Source: arXiv:2411.04551v3,
§3.1, the support transport in `lem: first.quadrant`. -/
theorem support_map_subset_image (ν : Measure (SSphere d)) (f : SSphere d → SSphere d)
    (hf : Continuous f) : (Measure.map f ν).support ⊆ f '' ν.support := by
  have hclosed : IsClosed (f '' ν.support) :=
    ((Measure.isClosed_support (μ := ν)).isCompact.image hf).isClosed
  apply Measure.support_subset_of_isClosed hclosed
  change Measure.map f ν (f '' ν.support)ᶜ = 0
  rw [Measure.map_apply hf.measurable hclosed.measurableSet.compl]
  apply measure_mono_null (t := ν.supportᶜ) _ Measure.measure_compl_support
  intro x hx hxS
  exact hx ⟨x, hxS, rfl⟩

/-- The original support is transported by the actual characteristic map
on each constant-field phase. Source: arXiv:2411.04551v3, §3.1,
`lem: first.quadrant`. -/
theorem cauchyPB_projection_support (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (w : SSphere d) (a b : ℝ) (hab : a < b)
    (hfield : ∀ s ∈ Set.Ioo a b, ∀ x : SSphere d,
      fullVF d θ (μ s) s (x : EucSpace d) = proj d (x : EucSpace d) (w : EucSpace d)) :
    (μ b : Measure (SSphere d)).support ⊆
      sphereProjectionFlow w (b - a) '' (μ a : Measure (SSphere d)).support := by
  rw [cauchyPB_projection_map θ μ hμ w a b hab hfield]
  exact support_map_subset_image _ _
    ((continuous_sphereProjectionFlow w).comp (continuous_const.prodMk continuous_id))

/-- Equal initial laws give equal endpoint laws for every original weak
solution on the same constant-field phase. Source: arXiv:2411.04551v3,
§1, the uniqueness used by §3.1, `lem: first.quadrant`. -/
theorem cauchyPB_projection_endpoint_unique (θ : TimeParams d)
    (μ ν : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) (hν : cauchyPB d θ ν)
    (w : SSphere d) (a b : ℝ) (hab : a < b) (hinit : μ a = ν a)
    (hμfield : ∀ s ∈ Set.Ioo a b, ∀ x : SSphere d,
      fullVF d θ (μ s) s (x : EucSpace d) = proj d (x : EucSpace d) (w : EucSpace d))
    (hνfield : ∀ s ∈ Set.Ioo a b, ∀ x : SSphere d,
      fullVF d θ (ν s) s (x : EucSpace d) = proj d (x : EucSpace d) (w : EucSpace d)) :
    (μ b : Measure (SSphere d)) = (ν b : Measure (SSphere d)) := by
  rw [cauchyPB_projection_map θ μ hμ w a b hab hμfield,
    cauchyPB_projection_map θ ν hν w a b hab hνfield, hinit]

/-- All transport hypotheses hold with a nonzero unit axis and a nonzero
actual weak solution. The second copy realizes endpoint uniqueness too.
Source: arXiv:2411.04551v3, §1 and §3.1, `lem: first.quadrant`. -/
example : cauchyPB 2 (pureAttentionParams 2 0)
      (fun _ => Perspective.diracProb 2 (basePoint 1)) ∧ (0 : ℝ) < 1 ∧
    ((fun _ : ℝ => Perspective.diracProb 2 (basePoint 1)) 0 =
      (fun _ : ℝ => Perspective.diracProb 2 (basePoint 1)) 0) ∧
    ∀ s ∈ Set.Ioo (0 : ℝ) 1, ∀ x : SSphere 2,
      fullVF 2 (pureAttentionParams 2 0) (Perspective.diracProb 2 (basePoint 1)) s
        (x : EucSpace 2) = proj 2 (x : EucSpace 2) (basePoint 1 : EucSpace 2) := by
  refine ⟨cauchyPB_pureAttention_dirac 0 (basePoint 1), one_pos, rfl, ?_⟩
  intro s _ x
  rw [fullVF_pureAttention_zero]
  simp [Perspective.diracProb]

/-- The support-map continuity input holds for the actual identity map
of the original circle. -/
example : Continuous (fun x : SSphere 2 => x) := continuous_id

end Transformer.Interpolation
