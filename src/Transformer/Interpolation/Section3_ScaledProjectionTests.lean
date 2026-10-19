module

public import Transformer.Interpolation.Section3_ProjectionWeakTests

/-!
# Inverse-characteristic tests after the source's time rescaling

ArXiv:2411.04551v3, §3.1, the final time-rescaling step of
`lem: first.quadrant`. A phase of speed `κ` uses the actual inverse
flow at time `-κt`. Its backward derivative cancels the actual flux
of `κ Proj_x w`. Conservation still holds at both phase endpoints.
The formulas hold for every real speed; the construction uses positive ones.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Scaling the constant direction scales the original tangent field.
Source: arXiv:2411.04551v3, §1, the projection in `eq: vf`. -/
theorem projection_field_smul (w x : EucSpace d) (κ : ℝ) :
    proj d x (κ • w) = κ • proj d x w := by
  rw [proj, proj, real_inner_smul_right]
  module

/-- The original weak flux has the same speed factor.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem projectionTestFlux_smul_axis (φ : EucSpace d → ℝ) (w x : EucSpace d) (κ : ℝ) :
    projectionTestFlux φ (κ • w) x = κ * projectionTestFlux φ w x := by
  rw [projectionTestFlux, projection_field_smul, map_smul, smul_eq_mul, projectionTestFlux]

/-- Inverse tests remain smooth after the actual constant time rescaling.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem contDiff_projectionPullback_scale (w : EucSpace d) (hw : ‖w‖ = 1)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ (⊤ : ℕ∞) φ) (κ : ℝ) :
    ContDiff ℝ (⊤ : ℕ∞) (fun q : ℝ × EucSpace d =>
      φ (projectionExtension w (-(κ * q.1)) q.2)) :=
  ContDiff.comp (f := fun q : ℝ × EucSpace d => (κ * q.1, q.2))
    (g := fun q : ℝ × EucSpace d => φ (projectionExtension w (-q.1) q.2))
    (contDiff_projectionPullback w hw φ hφ)
    ((contDiff_const.mul contDiff_fst).prodMk contDiff_snd)

/-- The actual backward transport identity at speed `κ`.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem hasDerivAt_projection_pullback_scale (w x : EucSpace d)
    (hw : ‖w‖ = 1) (hx : ‖x‖ = 1) (φ : EucSpace d → ℝ)
    (hφ : ContDiff ℝ 1 φ) (κ t : ℝ) :
    HasDerivAt (fun s => φ (projectionExtension w (-(κ * s)) x))
      (-(projectionTestFlux (fun y => φ (projectionExtension w (-(κ * t)) y)) (κ • w) x)) t := by
  have hd := (hasDerivAt_projection_pullback w x hw hx φ hφ (κ * t)).scomp t
    ((hasDerivAt_id t).const_mul κ)
  simp only [Function.comp_def, mul_one, smul_eq_mul] at hd
  apply hd.congr_deriv
  rw [projectionTestFlux_smul_axis, projectionTestFlux]
  ring

/-- Every actual weak solution conserves the inverse-flow test on an
original phase of speed `κ`. Source: arXiv:2411.04551v3, §1,
`eq: cauchy.pb`, and §3.1, `lem: first.quadrant`. -/
theorem hasDerivAt_scaled_projection_pullback_integral (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (w : EucSpace d) (hw : ‖w‖ = 1) (κ a b : ℝ)
    (hfield : ∀ s ∈ Set.Ioo a b, ∀ x : SSphere d,
      fullVF d θ (μ s) s (x : EucSpace d) = κ • proj d (x : EucSpace d) w)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ (⊤ : ℕ∞) φ)
    (t : ℝ) (ht : t ∈ Set.Ioo a b) :
    HasDerivAt (fun s => ∫ x, φ (projectionExtension w (-(κ * s)) (x : EucSpace d))
      ∂(μ s : Measure (SSphere d))) 0 t := by
  let F (q : ℝ × EucSpace d) := φ (projectionExtension w (-(κ * q.1)) q.2)
  have hF : ContDiff ℝ 2 F := (contDiff_projectionPullback_scale w hw φ hφ κ).of_le (by simp)
  have hfield' : ∀ s ∈ Set.Ioo a b, ∀ x : SSphere d,
      fullVF d θ (μ s) s (x : EucSpace d) = proj d (x : EucSpace d) (κ • w) := by
    intro s hs x
    rw [hfield s hs x, projection_field_smul]
  have hG (x : SSphere d) : timeTestDerivative F (t, (x : EucSpace d)) =
      -(projectionTestFlux (fun y => F (t, y)) (κ • w) (x : EucSpace d)) :=
    (hasDerivAt_spacetime_slice F hF t (x : EucSpace d)).unique
      (hasDerivAt_projection_pullback_scale w (x : EucSpace d) hw
        (mem_sphere_zero_iff_norm.mp x.2) φ (hφ.of_le (by simp)) κ t)
  have hd := hasDerivAt_cauchyPB_projection_spacetime θ μ hμ (κ • w) a b hfield' F hF t ht
  apply hd.congr_deriv
  have heq := integral_congr_ae (ae_of_all (μ t : Measure (SSphere d)) hG)
  rw [heq, integral_neg, add_neg_cancel]

/-- The actual inverse-test integral is continuous through switches at
every rescaled speed. Source: arXiv:2411.04551v3, §1 and §3.1,
`lem: first.quadrant`. -/
theorem continuous_scaled_projection_pullback_integral (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (w : EucSpace d) (hw : ‖w‖ = 1) (κ : ℝ)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ (⊤ : ℕ∞) φ) :
    Continuous (fun s => ∫ x, φ (projectionExtension w (-(κ * s)) (x : EucSpace d))
      ∂(μ s : Measure (SSphere d))) :=
  continuous_cauchyPB_spacetime_test θ μ hμ _
    ((contDiff_projectionPullback_scale w hw φ hφ κ).of_le (by simp))

/-- Conservation includes both original phase endpoints after rescaling.
Source: arXiv:2411.04551v3, §3.1, the time-rescaled phases in
`lem: first.quadrant`. -/
theorem scaled_projection_pullback_integral_eq (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (w : EucSpace d) (hw : ‖w‖ = 1) (κ a b : ℝ) (hab : a < b)
    (hfield : ∀ s ∈ Set.Ioo a b, ∀ x : SSphere d,
      fullVF d θ (μ s) s (x : EucSpace d) = κ • proj d (x : EucSpace d) w)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ (⊤ : ℕ∞) φ) :
    (∫ x, φ (projectionExtension w (-(κ * b)) (x : EucSpace d)) ∂(μ b : Measure (SSphere d))) =
      ∫ x, φ (projectionExtension w (-(κ * a)) (x : EucSpace d))
        ∂(μ a : Measure (SSphere d)) := by
  let I (s : ℝ) := ∫ x, φ (projectionExtension w (-(κ * s)) (x : EucSpace d))
    ∂(μ s : Measure (SSphere d))
  have hc : Continuous I := continuous_scaled_projection_pullback_integral θ μ hμ w hw κ φ hφ
  have hd (s : ℝ) (hs : s ∈ Set.Ioo a b) : HasDerivAt I 0 s :=
    hasDerivAt_scaled_projection_pullback_integral θ μ hμ w hw κ a b hfield φ hφ s hs
  obtain ⟨c, hc, hderiv⟩ := exists_deriv_eq_slope I hab hc.continuousOn
    (fun s hs => (hd s hs).differentiableAt.differentiableWithinAt)
  rw [(hd c hc).deriv] at hderiv
  have hne : b - a ≠ 0 := (sub_pos.mpr hab).ne'
  exact sub_eq_zero.mp ((div_eq_zero_iff.mp hderiv.symm).resolve_right hne)

/-- At an initial phase the inverse test retains its original initial
expectation after the actual time rescaling.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem scaled_projection_pullback_integral_eq_initial (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (w : EucSpace d) (hw : ‖w‖ = 1) (κ T : ℝ) (hT : 0 < T)
    (hfield : ∀ s ∈ Set.Ioo 0 T, ∀ x : SSphere d,
      fullVF d θ (μ s) s (x : EucSpace d) = κ • proj d (x : EucSpace d) w)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ (⊤ : ℕ∞) φ) :
    (∫ x, φ (projectionExtension w (-(κ * T)) (x : EucSpace d)) ∂(μ T : Measure (SSphere d))) =
      ∫ x, φ (x : EucSpace d) ∂(μ 0 : Measure (SSphere d)) := by
  rw [scaled_projection_pullback_integral_eq θ μ hμ w hw κ 0 T hT hfield φ hφ]
  apply integral_congr_ae
  exact ae_of_all _ fun x => by
    dsimp only
    rw [mul_zero, neg_zero, projectionExtension_eq w _ hw (mem_sphere_zero_iff_norm.mp x.2),
      projectionFlow_zero]

/-- All weak-equation, phase, unit-axis and test hypotheses have a
nonzero simultaneous witness. The speed is one; the original field
is nonzero at points away from its two poles. -/
example : cauchyPB 2 (pureAttentionParams 2 0)
      (fun _ => Perspective.diracProb 2 (basePoint 1)) ∧
    ‖(basePoint 1 : EucSpace 2)‖ = 1 ∧ (0 : ℝ) < 1 ∧
    (1 / 2 : ℝ) ∈ Set.Ioo (0 : ℝ) 1 ∧
    ContDiff ℝ (⊤ : ℕ∞) (fun x : EucSpace 2 => inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) x) ∧
    ∀ s ∈ Set.Ioo (0 : ℝ) 1, ∀ x : SSphere 2,
      fullVF 2 (pureAttentionParams 2 0) (Perspective.diracProb 2 (basePoint 1)) s
        (x : EucSpace 2) = (1 : ℝ) • proj 2 (x : EucSpace 2) (basePoint 1 : EucSpace 2) := by
  refine ⟨cauchyPB_pureAttention_dirac 0 (basePoint 1),
    mem_sphere_zero_iff_norm.mp (basePoint 1).2, one_pos, by norm_num,
    contDiff_const.inner ℝ contDiff_id, ?_⟩
  intro s _ x
  rw [one_smul, fullVF_pureAttention_zero]
  simp [Perspective.diracProb]

end Transformer.Interpolation
