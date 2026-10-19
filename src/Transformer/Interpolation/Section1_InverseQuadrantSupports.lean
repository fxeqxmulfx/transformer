module

public import Transformer.Interpolation.Section3_QuadrantPhaseSequence
public import Transformer.Interpolation.Section4_TargetSupportError

/-!
# Original inverse quadrant phases preserve arbitrarily small target errors

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, §3.1,
`lem: first.quadrant`, and §5, inversion after matching.
The actual two-projection phase sequence pushes every probability
forward by its original point map. Reversing its phase list gives
the exact inverse point map on every original probability law.

For finitely many target atoms, continuity of that inverse map gives
one positive concentration radius working for all targets. Thus laws
concentrated near their transported targets return near the original
targets after reversal. All assertions concern actual complete law
maps; no approximation by Dirac inputs is assumed for the reversal.

At every positive physical horizon the reversed phases have actual
zero-score controls, global weak existence and the corresponding
genuine Wasserstein accuracy for every original weak endpoint.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- The actual forward phases push every original probability by
the exact original two-projection point map. Source: arXiv:2411.04551v3,
§3.1, the composition in `lem: first.quadrant`. -/
theorem coe_twoProjection_phaseLaw (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ : ℝ) (ν : Perspective.ProbSphere d) :
    (phaseSequenceLaw (twoProjectionPhaseList i w₁ w₂ τ₁ τ₂) ν : Measure (SSphere d)) =
      Measure.map (fun x => sphereProjectionFlow w₂ τ₂ (sphereProjectionFlow w₁ τ₁ x))
        (ν : Measure (SSphere d)) := by
  have h₁ : Measurable (sphereProjectionFlow w₁ τ₁) :=
    ((continuous_sphereProjectionFlow w₁).comp (continuous_const.prodMk continuous_id)).measurable
  have h₂ : Measurable (sphereProjectionFlow w₂ τ₂) :=
    ((continuous_sphereProjectionFlow w₂).comp (continuous_const.prodMk continuous_id)).measurable
  rw [phaseSequenceLaw_twoProjection, coe_projectionLaw, coe_projectionLaw, Measure.map_map h₂ h₁]
  rfl

/-- Reversing the actual original phases reverses both projection
clocks and their order. Source: arXiv:2411.04551v3, §1 and §3.1,
inversion of the quadrant reduction. -/
theorem inverse_twoProjection_phaseLaw (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ : ℝ) (ν : Perspective.ProbSphere d) :
    phaseSequenceLaw (inversePhaseList (twoProjectionPhaseList i w₁ w₂ τ₁ τ₂)) ν =
      projectionLaw w₁ (projectionLaw w₂ ν (-τ₂)) (-τ₁) := by
  change ridgeLaw (τ₁ • (w₁ : EucSpace d)) 0 1
    (ridgeLaw (τ₂ • (w₂ : EucSpace d)) 0 1 ν ((-1 : ℝ) * 1)) ((-1 : ℝ) * 1) = _
  simp only [mul_one]
  rw [ridgeLaw_constant_projection i w₂, ridgeLaw_constant_projection i w₁,
    mul_neg_one, mul_neg_one]

/-- The actual inverse phase law pushes every original input by the
inverse point map. Source: arXiv:2411.04551v3, §1 and §3.1. -/
theorem coe_inverse_twoProjection_phaseLaw (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ : ℝ) (ν : Perspective.ProbSphere d) :
    (phaseSequenceLaw (inversePhaseList (twoProjectionPhaseList i w₁ w₂ τ₁ τ₂)) ν :
      Measure (SSphere d)) =
      Measure.map (fun x => sphereProjectionFlow w₁ (-τ₁) (sphereProjectionFlow w₂ (-τ₂) x))
        (ν : Measure (SSphere d)) := by
  have h₁ : Measurable (sphereProjectionFlow w₁ (-τ₁)) :=
    ((continuous_sphereProjectionFlow w₁).comp (continuous_const.prodMk continuous_id)).measurable
  have h₂ : Measurable (sphereProjectionFlow w₂ (-τ₂)) :=
    ((continuous_sphereProjectionFlow w₂).comp (continuous_const.prodMk continuous_id)).measurable
  rw [inverse_twoProjection_phaseLaw, coe_projectionLaw, coe_projectionLaw, Measure.map_map h₁ h₂]
  rfl

/-- The actual inverse point map returns every transported unit point
to its original location. Source: arXiv:2411.04551v3, §3.1. -/
theorem inverse_twoProjection_point (w₁ w₂ x : SSphere d) (τ₁ τ₂ : ℝ) :
    sphereProjectionFlow w₁ (-τ₁) (sphereProjectionFlow w₂ (-τ₂)
      (sphereProjectionFlow w₂ τ₂ (sphereProjectionFlow w₁ τ₁ x))) = x := by
  rw [sphereProjectionFlow_neg, sphereProjectionFlow_neg]

/-- A positive concentration radius can be chosen simultaneously for
all original target atoms before reversing the quadrant phases.
Source auxiliary: arXiv:2411.04551v3, §1 and §5, error under inversion. -/
theorem exists_inverse_twoProjection_radius (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ : ℝ) (Y : Idx N → SSphere d) (ε : ℝ) (hε : 0 < ε) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ k, ∀ ν : Perspective.ProbSphere d,
      (ν : Measure (SSphere d)).support ⊆
        Metric.closedBall (sphereProjectionFlow w₂ τ₂ (sphereProjectionFlow w₁ τ₁ (Y k))) δ →
      (phaseSequenceLaw (inversePhaseList (twoProjectionPhaseList i w₁ w₂ τ₁ τ₂)) ν :
        Measure (SSphere d)).support ⊆ Metric.closedBall (Y k) ε := by
  let F (x : SSphere d) := sphereProjectionFlow w₂ τ₂ (sphereProjectionFlow w₁ τ₁ x)
  let G (x : SSphere d) := sphereProjectionFlow w₁ (-τ₁) (sphereProjectionFlow w₂ (-τ₂) x)
  have hG : Continuous G :=
    ((continuous_sphereProjectionFlow w₁).comp (continuous_const.prodMk continuous_id)).comp
      ((continuous_sphereProjectionFlow w₂).comp (continuous_const.prodMk continuous_id))
  have heps (k : Idx N) : ∃ δ : ℝ, 0 < δ ∧ ∀ x, dist x (F (Y k)) < δ → dist (G x) (Y k) < ε := by
    obtain ⟨δ, hδ, hbound⟩ :=
      Metric.continuousAt_iff.mp (hG.continuousAt : ContinuousAt G (F (Y k))) ε hε
    refine ⟨δ, hδ, ?_⟩
    intro x hx
    have hh := hbound hx
    change dist (G x) (sphereProjectionFlow w₁ (-τ₁) (sphereProjectionFlow w₂ (-τ₂)
      (sphereProjectionFlow w₂ τ₂ (sphereProjectionFlow w₁ τ₁ (Y k))))) < ε at hh
    rwa [inverse_twoProjection_point] at hh
  choose δ hδ hbound using heps
  obtain ⟨r, hr, hsmall⟩ := exists_common_positive_lt δ hδ
  refine ⟨r, hr, ?_⟩
  intro k ν hν y hy
  rw [coe_inverse_twoProjection_phaseLaw] at hy
  obtain ⟨x, hx, rfl⟩ := support_map_subset_image _ _ hG hy
  exact (hbound k x ((Metric.mem_closedBall.mp (hν hx)).trans_lt (hsmall k))).le

/-- The reversed original phases admit common controls taking every
sufficiently concentrated law near its original target, for every
original weak solution and any positive horizon. Source auxiliary:
arXiv:2411.04551v3, §1 and §5, the final inversion step. -/
theorem exists_inverse_twoProjection_accuracy_controls (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ : ℝ) (Y : Idx N → SSphere d) (ε H : ℝ) (hε : 0 < ε) (hH : 0 < H) :
    ∃ (δ : ℝ) (θ : TimeParams d), 0 < δ ∧ PiecewiseConstant d θ H 3 ∧ (∀ s, (θ s).B = 0) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d θ μ) ∧
      ∀ k, ∀ ν : Perspective.ProbSphere d,
        (ν : Measure (SSphere d)).support ⊆
          Metric.closedBall (sphereProjectionFlow w₂ τ₂ (sphereProjectionFlow w₁ τ₁ (Y k))) δ →
        ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν → cauchyPB d θ μ →
          W2 d (μ H : Measure (SSphere d)) (Measure.dirac (Y k)) ≤ ε := by
  obtain ⟨δ, hδ, hbound⟩ := exists_inverse_twoProjection_radius i w₁ w₂ τ₁ τ₂ Y ε hε
  obtain ⟨θ, hpc, hB, hex, hend⟩ := exists_phase_sequence_realization i
    (inversePhaseList (twoProjectionPhaseList i w₁ w₂ τ₁ τ₂))
    (inversePhaseList_valid _ (twoProjectionPhaseList_valid i w₁ w₂ τ₁ τ₂)) H hH
  refine ⟨δ, θ, hδ, ?_, hB, hex, ?_⟩
  · simpa only [inversePhaseList_length, twoProjectionPhaseList_length] using hpc
  · intro k ν hν μ h0 hμ
    rw [hend μ hμ, h0]
    exact W2_le_radius_of_support _ (Y k) ε hε.le (hbound k ν hν)

/-- Actual coordinate axes, a nontrivial finite target pair and positive
clocks, tolerance and horizon jointly realize the inverse-construction
hypotheses. The targets may be antipodal and outside the quadrant. -/
example : ∃ (w₁ w₂ : SSphere 3) (Y : Idx 2 → SSphere 3),
    w₁ ≠ w₂ ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≠ 0 ∧ Function.Injective Y ∧
      Y 1 = antipode 3 (Y 0) := by
  let p := basePoint 2
  let Y : Idx 2 → SSphere 3 := ![p, antipode 3 p]
  refine ⟨p, ⟨PiLp.single 2 (1 : Fin 3) (1 : ℝ), by simp⟩, Y,
    ?_, one_pos, one_pos, one_ne_zero, ?_, rfl⟩
  · intro he
    have hh := congrArg (fun z : SSphere 3 => (z : EucSpace 3) (0 : Fin 3)) he
    norm_num [p, basePoint] at hh
  · intro k l he
    fin_cases k <;> fin_cases l
    · rfl
    · exact False.elim (antipode_ne 3 p he.symm)
    · exact False.elim (antipode_ne 3 p he)
    · rfl

end Transformer.Interpolation
