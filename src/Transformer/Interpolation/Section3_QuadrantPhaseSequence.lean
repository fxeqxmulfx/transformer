/-
# The original two-flow quadrant reduction as autonomous phases

ArXiv:2411.04551v3, §3.1, `lem: first.quadrant`.
The constant projection vector field is also an actual single-neuron
field with zero input weight and bias one. Original weak uniqueness
therefore identifies its law with the already constructed projection
law at the exact scaled clock, for every real time and every input.

The two compact-support projection flows can consequently be used
inside arbitrary original autonomous phase sequences. Their complete
laws, Dirac endpoints, injectivity and positive-quadrant support
conclusion are retained. This is a representation of the existing
reduction, without an additional controllability hypothesis.
-/

import Transformer.Interpolation.Section1_PhaseSequenceInverse
import Transformer.Interpolation.Section3_TwoContractions

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- The actual constant-gate ridge law is exactly the original
projection law at its scaled clock. Source: arXiv:2411.04551v3,
§3.1, the constant projection phases of `lem: first.quadrant`. -/
theorem ridgeLaw_constant_projection (i : Fin d) (w : SSphere d)
    (κ : ℝ) (ν : Perspective.ProbSphere d) (t : ℝ) :
    ridgeLaw (κ • (w : EucSpace d)) 0 1 ν t = projectionLaw w ν (κ * t) := by
  let θ : TimeParams d := fun _ => reluRidgeParams i (κ • (w : EucSpace d)) 0 1
  have hμ : cauchyPB d θ (fun s => projectionLaw w ν (κ * s)) := by
    apply cauchyPB_projectionLaw_scale
    intro s x
    rw [fullVF_reluRidgeParams]
    simp only [inner_zero_right, sub_zero, max_eq_left zero_le_one, one_smul]
    exact projection_field_smul (w : EucSpace d) (x : EucSpace d) κ
  have he := cauchyPB_ridge_eq i (κ • (w : EucSpace d)) 0 1 ν
    (fun s => projectionLaw w ν (κ * s)) hμ (by simp)
  exact (congrFun he t).symm

/-- Every original unit atom follows the explicit projection flow.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem projectionLaw_diracProb (w x : SSphere d) (t : ℝ) :
    projectionLaw w (Perspective.diracProb d x) t =
      Perspective.diracProb d (sphereProjectionFlow w t x) := by
  apply Subtype.ext
  change Measure.map (sphereProjectionFlow w t) (Measure.dirac x) = _
  exact Measure.map_dirac x

/-- The two actual single-neuron phases with their intrinsic times.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
noncomputable def twoProjectionPhaseList (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ : ℝ) : List (AutonomousPhase d) :=
  [ridgePhase i (τ₁ • (w₁ : EucSpace d)) 0 1,
    ridgePhase i (τ₂ • (w₂ : EucSpace d)) 0 1]

/-- The original quadrant reduction contains two actual phases.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem twoProjectionPhaseList_length (i : Fin d) (w₁ w₂ : SSphere d) (τ₁ τ₂ : ℝ) :
    (twoProjectionPhaseList i w₁ w₂ τ₁ τ₂).length = 2 := rfl

/-- Both actual phases have proved original weak laws and zero scores.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem twoProjectionPhaseList_valid (i : Fin d) (w₁ w₂ : SSphere d) (τ₁ τ₂ : ℝ) :
    ∀ P ∈ twoProjectionPhaseList i w₁ w₂ τ₁ τ₂, IsAutonomousPhase P ∧ P.1.B = 0 := by
  intro P hP
  rcases List.mem_cons.mp hP with rfl | hP
  · exact ⟨isAutonomousPhase_ridge _ _ _ _, ridgePhase_B_zero _ _ _ _⟩
  · have he := List.mem_singleton.mp hP
    subst P
    exact ⟨isAutonomousPhase_ridge _ _ _ _, ridgePhase_B_zero _ _ _ _⟩

/-- The original phase composition retains both exact projection
clocks on every probability law. Source: arXiv:2411.04551v3,
§3.1, composition in `lem: first.quadrant`. -/
theorem phaseSequenceLaw_twoProjection (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ : ℝ) (ν : Perspective.ProbSphere d) :
    phaseSequenceLaw (twoProjectionPhaseList i w₁ w₂ τ₁ τ₂) ν =
      projectionLaw w₂ (projectionLaw w₁ ν τ₁) τ₂ := by
  change ridgeLaw (τ₂ • (w₂ : EucSpace d)) 0 1
    (ridgeLaw (τ₁ • (w₁ : EucSpace d)) 0 1 ν 1) 1 = _
  rw [ridgeLaw_constant_projection i w₁, ridgeLaw_constant_projection i w₂, mul_one, mul_one]

/-- Original atomic endpoints are the same actual two-flow points.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem phaseSequenceLaw_twoProjection_dirac (i : Fin d) (w₁ w₂ : SSphere d)
    (τ₁ τ₂ : ℝ) (x : SSphere d) :
    phaseSequenceLaw (twoProjectionPhaseList i w₁ w₂ τ₁ τ₂) (Perspective.diracProb d x) =
      Perspective.diracProb d (sphereProjectionFlow w₂ τ₂ (sphereProjectionFlow w₁ τ₁ x)) := by
  rw [phaseSequenceLaw_twoProjection, projectionLaw_diracProb, projectionLaw_diracProb]

/-- The original two-flow point map preserves distinct atoms.
Source: arXiv:2411.04551v3, §3.1, invertibility in `lem: first.quadrant`. -/
theorem injective_twoProjection_map (w₁ w₂ : SSphere d) (τ₁ τ₂ : ℝ) :
    Function.Injective (fun x => sphereProjectionFlow w₂ τ₂ (sphereProjectionFlow w₁ τ₁ x)) := by
  have hinv : Function.LeftInverse
      (fun x => sphereProjectionFlow w₁ (-τ₁) (sphereProjectionFlow w₂ (-τ₂) x))
      (fun x => sphereProjectionFlow w₂ τ₂ (sphereProjectionFlow w₁ τ₁ x)) := by
    intro x
    change sphereProjectionFlow w₁ (-τ₁) (sphereProjectionFlow w₂ (-τ₂)
      (sphereProjectionFlow w₂ τ₂ (sphereProjectionFlow w₁ τ₁ x))) = x
    rw [sphereProjectionFlow_neg, sphereProjectionFlow_neg]
  exact hinv.injective

/-- Actual original support data missing a point admit a two-phase
quadrant preprocessing sequence. Source: arXiv:2411.04551v3,
§3.1, `lem: first.quadrant`; the support hole is unchanged. -/
theorem exists_quadrant_preprocessing_phases (hd : 2 ≤ d) (i : Fin d)
    (ν : Idx N → Perspective.ProbSphere d)
    (hne : (⋃ k, (ν k : Measure (SSphere d)).support) ≠ Set.univ) :
    ∃ L : List (AutonomousPhase d), L.length = 2 ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      ∀ k, (phaseSequenceLaw L (ν k) : Measure (SSphere d)).support ⊆ positiveQuadrant d := by
  obtain ⟨w₁, w₂, τ₁, τ₂, _, _, hquad⟩ := exists_two_projection_quadrant_supports hd ν hne
  refine ⟨twoProjectionPhaseList i w₁ w₂ τ₁ τ₂, rfl,
    twoProjectionPhaseList_valid i w₁ w₂ τ₁ τ₂, ?_⟩
  intro k x hx
  rw [phaseSequenceLaw_twoProjection, coe_projectionLaw] at hx
  obtain ⟨y, hy, rfl⟩ := support_map_subset_image _ _
    ((continuous_sphereProjectionFlow w₂).comp (continuous_const.prodMk continuous_id)) hx
  rw [coe_projectionLaw] at hy
  obtain ⟨z, hz, rfl⟩ := support_map_subset_image _ _
    ((continuous_sphereProjectionFlow w₁).comp (continuous_const.prodMk continuous_id)) hy
  exact hquad k z hz

/-- The two actual preprocessing phases can be realized and followed
by further original phase sequences at any positive physical time.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`;
the general phase realization adds one constant terminal piece. -/
theorem exists_quadrant_preprocessing_controls (hd : 2 ≤ d) (i : Fin d)
    (ν : Idx N → Perspective.ProbSphere d)
    (hne : (⋃ k, (ν k : Measure (SSphere d)).support) ≠ Set.univ) (T : ℝ) (hT : 0 < T) :
    ∃ θ : TimeParams d, PiecewiseConstant d θ T 3 ∧ (∀ s, (θ s).B = 0) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d θ μ) ∧
      ∀ k, ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν k → cauchyPB d θ μ →
        (μ T : Measure (SSphere d)).support ⊆ positiveQuadrant d := by
  obtain ⟨L, hlen, hL, hquad⟩ := exists_quadrant_preprocessing_phases hd i ν hne
  obtain ⟨θ, hpc, hB, hex, hend⟩ := exists_phase_sequence_realization i L hL T hT
  refine ⟨θ, by simpa only [hlen] using hpc, hB, hex, ?_⟩
  intro k μ h0 hμ
  rw [hend μ hμ, h0]
  exact hquad k

/-- A genuine positive circle atom supplies the original support-hole
and dimension hypotheses. Nonzero real clocks and a nonzero original
projection axis also inhabit every algebraic phase statement above. -/
example : ∃ ν : Idx 1 → Perspective.ProbSphere 2, 2 ≤ 2 ∧
    (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧
      (⋃ k, (ν k : Measure (SSphere 2)).support) ≠ Set.univ := by
  let p := basePoint 1
  let ν : Idx 1 → Perspective.ProbSphere 2 := fun _ => Perspective.diracProb 2 p
  refine ⟨ν, le_rfl, one_ne_zero, one_pos, ?_⟩
  intro he
  have hm : antipode 2 p ∈ ⋃ k, (ν k : Measure (SSphere 2)).support := he.symm ▸ Set.mem_univ _
  obtain ⟨k, hk⟩ := Set.mem_iUnion.mp hm
  exact antipode_ne 2 p (eq_of_mem_support_dirac hk)

end Transformer.Interpolation
